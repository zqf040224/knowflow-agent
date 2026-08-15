"""Production /api/chat request runtime.

This module owns the outer chat orchestration boundary: request normalization,
attachment hydration, task planning, and tool execution. The default production
path is a LangGraph StateGraph; the planner and legacy modes remain explicit
rollback paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Iterable, Optional, TypedDict

from agents.document_graph_runner import DocumentGraphRunner
from agents.document_graph_state import DocumentGraphState
from agents.document_graph_steps import DocumentGraphSteps

from chat_architecture import (
    INTENT_CLARIFY,
    INTENT_DOC_DRAFTING,
    INTENT_DOC_FORMATTING,
    INTENT_FORM_TEMPLATE_EXPORT,
    INTENT_IDENTITY_HELP,
    INTENT_KNOWLEDGE_QA,
    INTENT_SPREADSHEET_TRANSFORM,
    IntentRouter,
    RouteResult,
)
from chat_events import error_event, parse_sse_events, sse
from graph_state_validation import validate_graph_update
from task_planner import (
    TOOL_CLARIFY,
    TOOL_DRAFT_DOCUMENT,
    TOOL_FORMAT_DOCUMENT,
    TOOL_IDENTITY_HELP,
    TOOL_KNOWLEDGE_QA,
    TOOL_PREPARE_FORM_EXPORT,
    TOOL_PREPARE_SPREADSHEET_TRANSFORM,
    TaskPlan,
)

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.graph import END, StateGraph
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised only when dependency is missing
    END = "__end__"
    StateGraph = None
    InMemorySaver = None
    JsonPlusSerializer = None
    Command = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(exc)


logger = logging.getLogger(__name__)


class ChatRunNotFoundError(LookupError):
    """The authenticated user cannot access the requested graph run."""


class ChatRunConflictError(RuntimeError):
    """The run exists but is not currently resumable."""


class ChatRunCancelledError(RuntimeError):
    """The owning session was deleted while the graph was running."""


@dataclass
class ChatRuntimeDependencies:
    memory: Any
    upload_manager: Any
    reimbursement_detector: Callable[[str, str], str]
    lightweight_stream: Callable
    document_format_stream: Callable
    document_draft_stream: Callable
    rag_qa_stream: Callable
    intent_classifier: Optional[Callable[[dict[str, Any]], Any]] = None
    task_planner: Any = None
    tool_orchestrator: Any = None
    checkpointer: Any = None
    persistence: Any = None
    artifact_store: Any = None
    run_id_factory: Optional[Callable[[], str]] = None
    workflow_version: str = "chat-v3"


@dataclass
class PreparedChatRequest:
    run_id: str
    request_id: str
    message: str
    display_message: str
    mode: str
    session_id: str
    user_id: str
    user_info: Any
    user_metadata: Optional[dict]
    attachments: list[dict]


@dataclass(frozen=True)
class ChatRunContext:
    """Per-invocation values that must never be persisted in graph state."""

    user_info: Any
    execution_token: str = ""
    lease: Any = None
    recovery: bool = False
    source_workflow_version: str = ""
    document_runtime_factory: Optional[Callable[[dict[str, Any], bool], Any]] = None
    document_node_observer: Optional[Callable[[dict[str, Any], str], None]] = None
    document_failure_callback: Optional[
        Callable[[dict[str, Any], str, Exception], None]
    ] = None
    document_update_sanitizer: Optional[
        Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    ] = None
    document_runtime_cache: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    raise_document_node_errors: bool = True

    def get_document_execution(
        self,
        state: dict[str, Any],
        *,
        prepare: bool = False,
    ) -> Any:
        run_id = str(state.get("run_id") or "")
        if run_id not in self.document_runtime_cache:
            if not callable(self.document_runtime_factory):
                raise RuntimeError("Native document runtime factory is unavailable")
            self.document_runtime_cache[run_id] = self.document_runtime_factory(
                state,
                prepare,
            )
        return self.document_runtime_cache[run_id]

    def get_document_orchestrator(self, state: dict[str, Any]) -> Any:
        return self.get_document_execution(state).orchestrator

    def resolve_document_request(self, state: dict[str, Any]) -> str:
        """Hydrate the document request from runtime-only attachment storage.

        The parent and child checkpoints deliberately retain only the original
        request plus controlled attachment references.  The potentially large
        attachment body is reconstructed in the per-invocation runtime and must
        never be copied back into graph State.
        """

        execution = self.get_document_execution(state)
        prepared_run = getattr(execution, "prepared_run", None)
        if prepared_run is not None:
            return str(getattr(prepared_run, "request_with_context", "") or "")
        return str(
            state.get("request_with_context")
            or state.get("input_user_request")
            or state.get("request_message")
            or ""
        )

    def resolve_document_previous_context(self, state: dict[str, Any]) -> str:
        execution = self.get_document_execution(state)
        prepared_run = getattr(execution, "prepared_run", None)
        if prepared_run is not None:
            return str(getattr(prepared_run, "previous_context", "") or "")
        return str(state.get("previous_context") or "")

    def resolve_document_runtime_snapshot(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        execution = self.get_document_execution(state)
        snapshot = getattr(execution, "runtime_snapshot", None)
        return dict(snapshot) if isinstance(snapshot, dict) else {}

    def sanitize_document_update(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        if callable(self.document_update_sanitizer):
            # A mounted child node can receive a narrowed State view without
            # the parent's attachment references.  Supply the runtime-only
            # redaction inputs on a throwaway mapping; they are never returned
            # by the sanitizer and therefore never enter a checkpoint.
            sanitizer_state = dict(state)
            execution = self.get_document_execution(state)
            sanitizer_state["_runtime_attachment_bodies"] = list(
                getattr(execution, "attachment_bodies", []) or []
            )
            sanitizer_state["_runtime_document_snapshot"] = dict(
                getattr(execution, "runtime_snapshot", {}) or {}
            )
            return self.document_update_sanitizer(sanitizer_state, update)
        return update

    @staticmethod
    def encode_document_event(event: dict[str, Any]) -> str:
        return sse(event)

    def observe_document_node(self, state: dict[str, Any], node: str) -> None:
        if callable(self.document_node_observer):
            self.document_node_observer(state, node)

    def handle_document_failure(
        self,
        state: dict[str, Any],
        step: str,
        exc: Exception,
    ) -> None:
        if callable(self.document_failure_callback):
            self.document_failure_callback(state, step, exc)


class ChatExecutionLease:
    """Keep a long-running node owned while its LLM/tool call is blocked."""

    def __init__(
        self,
        persistence: Any,
        *,
        run_id: str,
        user_id: str,
        execution_token: str,
        interval_seconds: float = 10.0,
        max_consecutive_failures: int = 1,
    ) -> None:
        self.persistence = persistence
        self.run_id = run_id
        self.user_id = user_id
        self.execution_token = execution_token
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self._node = "prepare"
        self._node_lock = Lock()
        self._stop = Event()
        self._lost = Event()
        self._loss_reason = "graph execution lease was lost"
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        heartbeat = getattr(self.persistence, "heartbeat_run", None)
        if not callable(heartbeat) or not self.execution_token:
            return
        if self._thread is not None:
            return
        self._thread = Thread(
            target=self._run,
            name=f"graph-lease-{self.run_id[:12]}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop.wait(self.interval_seconds):
            with self._node_lock:
                node = self._node
            try:
                record = self.persistence.heartbeat_run(
                    self.run_id,
                    user_id=self.user_id,
                    execution_token=self.execution_token,
                    current_node=node,
                )
            except Exception:
                consecutive_failures += 1
                logger.exception(
                    "Graph lease heartbeat failed run_id=%s", self.run_id
                )
                if consecutive_failures >= self.max_consecutive_failures:
                    self._loss_reason = (
                        "graph execution lease heartbeat failed; "
                        "ownership can no longer be verified"
                    )
                    self._lost.set()
                    return
                continue
            consecutive_failures = 0
            if record is None:
                self._loss_reason = "graph execution lease was lost"
                self._lost.set()
                return

    def set_current_node(self, node: str) -> None:
        with self._node_lock:
            self._node = str(node)

    def assert_owned(self) -> None:
        if self._lost.is_set():
            raise ChatRunConflictError(self._loss_reason)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)


class ReservedHttpChatStream:
    """Close-aware iterator for a run reserved before HTTP response creation.

    Closing a never-started Python generator does not execute its ``finally``
    block.  This wrapper supplies the missing lifecycle edge without changing
    any yielded SSE bytes. Once the first ``next()`` begins, close is delegated
    to the graph generator so its existing disconnect handling remains the
    single authority for an active run.
    """

    def __init__(self, stream: Iterable[str], *, on_unstarted_close: Callable[[], None]):
        self._stream = iter(stream)
        self._on_unstarted_close = on_unstarted_close
        self._lock = Lock()
        self._started = False
        self._closed = False

    def __iter__(self) -> "ReservedHttpChatStream":
        return self

    def __next__(self) -> str:
        with self._lock:
            if self._closed:
                raise StopIteration
            self._started = True
        try:
            return next(self._stream)
        except StopIteration:
            with self._lock:
                self._closed = True
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
        try:
            if not started:
                self._on_unstarted_close()
        finally:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()


class ChatGraphState(TypedDict, total=False):
    run_id: str
    request_id: str
    idempotency_key: str
    user_id: str
    session_id: str
    request_message: str
    message: str
    display_message: str
    mode: str
    file_ids: list[str]
    attachments: list[dict[str, Any]]
    task_plan: dict[str, Any]
    step_index: int
    working_message: str
    tool_results: list[dict[str, Any]]
    current_step_done: dict[str, Any]
    current_step_failed: bool
    plan_events_emitted: bool
    compat_orchestrator: bool
    status: str
    explicit_memory_recorded: bool
    final_event: dict[str, Any]
    error: dict[str, str]


DocumentToolGraphState = TypedDict(
    "DocumentToolGraphState",
    {
        **ChatGraphState.__annotations__,
        **DocumentGraphState.__annotations__,
        "native_document_replay": bool,
    },
    total=False,
)


class ChatGraphRuntime:
    """Outer production chat graph.

    Dispatch contract:
    - knowledge_qa -> RagQaStreamService
    - doc_drafting -> DocumentDraftStreamService -> AgentOrchestrator
    - doc_formatting -> DocumentFormatStreamService
    - lightweight intents -> LightweightChatStreamService
    """
    GRAPH_INTENTS = {
        INTENT_IDENTITY_HELP,
        INTENT_CLARIFY,
        INTENT_FORM_TEMPLATE_EXPORT,
        INTENT_SPREADSHEET_TRANSFORM,
        INTENT_DOC_FORMATTING,
        INTENT_DOC_DRAFTING,
        INTENT_KNOWLEDGE_QA,
    }

    GRAPH_MODE_ALIASES = {"graph", "langgraph", "on", "true", "1"}
    PLANNER_MODE_ALIASES = {"planner", "task_planner", "off", "false", "0"}
    LEGACY_MODE_ALIASES = {"legacy", "pipeline"}
    RECOVERY_WINDOW_DAYS = 7
    RECOVERY_LEASE_STALE_SECONDS = 30
    PREVIOUS_WORKFLOW_VERSION = "chat-v2"
    UNSTARTED_HTTP_ERROR = "http response closed before graph execution started"

    def __init__(
        self,
        deps: ChatRuntimeDependencies,
        *,
        runtime_mode: Optional[str] = None,
        use_langgraph: Optional[bool] = None,
    ):
        self.deps = deps
        self._runtime_mode = self.resolve_runtime_mode(runtime_mode, use_langgraph=use_langgraph)
        self._use_langgraph = self._runtime_mode == "graph"
        if self._runtime_mode in {"graph", "planner"}:
            self._require_planner_dependencies()
        if self._use_langgraph and not LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "CHAT_RUNTIME=graph requires LangGraph, but it could not be imported: "
                f"{LANGGRAPH_IMPORT_ERROR or 'unknown import error'}"
            )
        self._checkpointer = None
        self._telemetry_lock = Lock()
        self._telemetry = {
            "run_id": "",
            "current_node": "",
            "checkpoint_id": "",
        }
        if self._use_langgraph:
            app_env = os.getenv("APP_ENV", "development").strip().lower()
            if app_env in {"prod", "production"} and (
                self.deps.checkpointer is None or self.deps.persistence is None
            ):
                raise RuntimeError(
                    "Production LangGraph requires an explicitly configured "
                    "persistent checkpointer and run ledger"
                )
            self._checkpointer = self.deps.checkpointer or self._new_test_checkpointer()
        self._graph = self._build_graph() if self._use_langgraph else None

    @staticmethod
    def _new_test_checkpointer() -> Any:
        """Give every non-wired runtime an isolated, strict in-memory saver."""
        if InMemorySaver is None or JsonPlusSerializer is None:
            raise RuntimeError("LangGraph checkpointer dependencies are unavailable")
        serde = JsonPlusSerializer(
            pickle_fallback=False,
            allowed_json_modules=None,
            allowed_msgpack_modules=None,
        )
        return InMemorySaver(serde=serde)

    @classmethod
    def resolve_runtime_mode(
        cls,
        explicit: Optional[str] = None,
        *,
        use_langgraph: Optional[bool] = None,
    ) -> str:
        """Normalize the supported runtime matrix to graph/planner/legacy."""
        if explicit is not None and use_langgraph is not None:
            raise ValueError("Specify runtime_mode or use_langgraph, not both")
        if use_langgraph is not None:
            return "graph" if use_langgraph else "legacy"

        value = explicit if explicit is not None else os.getenv("CHAT_RUNTIME", "graph")
        normalized = str(value or "graph").strip().lower()
        if normalized in cls.GRAPH_MODE_ALIASES:
            return "graph"
        if normalized in cls.PLANNER_MODE_ALIASES:
            return "planner"
        if normalized in cls.LEGACY_MODE_ALIASES:
            return "legacy"
        supported = sorted(
            cls.GRAPH_MODE_ALIASES | cls.PLANNER_MODE_ALIASES | cls.LEGACY_MODE_ALIASES
        )
        raise ValueError(
            f"Unsupported CHAT_RUNTIME value {value!r}; expected one of: {', '.join(supported)}"
        )

    def _require_planner_dependencies(self) -> None:
        missing = []
        if self.deps.task_planner is None:
            missing.append("task_planner")
        if self.deps.tool_orchestrator is None:
            missing.append("tool_orchestrator")
        if missing:
            raise RuntimeError(
                f"CHAT_RUNTIME={self._runtime_mode} requires: {', '.join(missing)}"
            )

    @property
    def uses_langgraph(self) -> bool:
        return bool(self._use_langgraph)

    @property
    def runtime_mode(self) -> str:
        return self._runtime_mode

    def stream(self, raw_data: dict, *, user_id: str, user_info: Any) -> Iterable[str]:
        return self._open_stream(
            raw_data,
            user_id=user_id,
            user_info=user_info,
            reserve_before_return=False,
        )

    def stream_http(
        self,
        raw_data: dict,
        *,
        user_id: str,
        user_info: Any,
    ) -> Iterable[str]:
        """Reserve graph identity before Flask commits the SSE response.

        ``stream()`` intentionally remains lazy for non-HTTP callers that may
        construct and close an iterator without consuming it.  The HTTP entry
        point uses this variant so the atomic ``start_run`` identity check can
        still become a real 409 response rather than an exception raised after
        WSGI has already sent ``200 text/event-stream`` headers.
        """

        return self._open_stream(
            raw_data,
            user_id=user_id,
            user_info=user_info,
            reserve_before_return=True,
        )

    def _open_stream(
        self,
        raw_data: dict,
        *,
        user_id: str,
        user_info: Any,
        reserve_before_return: bool,
    ) -> Iterable[str]:
        initial_state = self._initial_state(raw_data, user_id=user_id)
        session_id_explicit = self._session_id_is_explicit(raw_data)
        if self._runtime_mode == "graph":
            # Detect an idempotency-key payload mismatch before Flask commits a
            # streaming 200 response.  The durable start_run call repeats the
            # check to close races between this read and the insert.
            self._preflight_request_identity(
                initial_state,
                allow_implicit_session_rebind=not session_id_explicit,
            )

        reserved_dispatch: tuple[Any, ChatRunContext] | None = None
        if self._runtime_mode == "graph" and reserve_before_return:
            execution_token = uuid.uuid4().hex
            record = self._start_persisted_run(
                initial_state,
                execution_token=execution_token,
                allow_implicit_session_rebind=not session_id_explicit,
            )
            record = self._reclaim_unstarted_http_run(
                record,
                initial_state,
                execution_token=execution_token,
            )
            lease = (
                ChatExecutionLease(
                    self.deps.persistence,
                    run_id=initial_state["run_id"],
                    user_id=initial_state["user_id"],
                    execution_token=execution_token,
                )
                if self.deps.persistence is not None
                else None
            )
            context = self._new_run_context(
                user_info=user_info,
                execution_token=execution_token,
                lease=lease,
                recovery=False,
                source_workflow_version=self.deps.workflow_version,
            )
            reserved_dispatch = (record, context)

        def dispatch() -> Iterable[str]:
            # Non-HTTP callers retain lazy start semantics. The HTTP path has
            # already reserved identity above so Flask can return a real 409.
            if reserved_dispatch is not None:
                record, context = reserved_dispatch
            else:
                execution_token = uuid.uuid4().hex
                lease = (
                    ChatExecutionLease(
                        self.deps.persistence,
                        run_id=initial_state["run_id"],
                        user_id=initial_state["user_id"],
                        execution_token=execution_token,
                    )
                    if self.deps.persistence is not None
                    else None
                )
                context = self._new_run_context(
                    user_info=user_info,
                    execution_token=execution_token,
                    lease=lease,
                    recovery=False,
                    source_workflow_version=self.deps.workflow_version,
                )
                record = None
            if self._runtime_mode == "graph":
                if reserved_dispatch is None:
                    record = self._start_persisted_run(
                        initial_state,
                        execution_token=context.execution_token,
                        allow_implicit_session_rebind=not session_id_explicit,
                    )
                if record is not None and (
                    record.run_id != initial_state["run_id"]
                    or getattr(record, "execution_token", context.execution_token)
                    != context.execution_token
                ):
                    initial_state.update({
                        "run_id": record.run_id,
                        "request_id": record.request_id,
                        "session_id": record.session_id,
                    })
                    stream = self._replay_idempotent_run(record)
                else:
                    stream = self._stream_graph(initial_state, context)
            elif self._runtime_mode == "planner":
                stream = self._stream_planner(initial_state, context)
            else:
                stream = self._stream_legacy(initial_state, context)
            yield from self._enrich_public_stream(stream, initial_state)

        stream = dispatch()
        if reserved_dispatch is None:
            return stream
        record, context = reserved_dispatch
        if not self._owns_reserved_run(record, initial_state, context):
            return stream
        return ReservedHttpChatStream(
            stream,
            on_unstarted_close=lambda: self._abort_unstarted_http_run(
                initial_state,
                context,
            ),
        )

    @staticmethod
    def _session_id_is_explicit(raw_data: Any) -> bool:
        if not isinstance(raw_data, dict):
            return False
        return bool(str(raw_data.get("session_id") or "").strip())

    def run_status(self, run_id: str, *, user_id: str) -> dict[str, Any]:
        """Return authenticated run/checkpoint metadata without graph state bodies."""
        record = self._owned_run(run_id, user_id=user_id)
        checkpoint_id = ""
        interrupt_ids: list[str] = []
        if self._graph is not None:
            try:
                snapshot = self._graph.get_state(
                    {"configurable": {"thread_id": record.thread_id}}
                )
            except Exception:
                logger.exception("Could not read run checkpoint run_id=%s", run_id)
            else:
                configurable = (getattr(snapshot, "config", None) or {}).get(
                    "configurable", {}
                )
                checkpoint_id = str(configurable.get("checkpoint_id") or "")
                interrupt_ids = self._snapshot_interrupt_ids(snapshot)
        return {
            "run_id": record.run_id,
            "session_id": record.session_id,
            "request_id": record.request_id,
            "status": record.status,
            "current_node": record.current_node,
            "workflow_version": record.workflow_version,
            "checkpoint_id": checkpoint_id,
            "interrupt_ids": interrupt_ids,
            "updated_at": record.updated_at,
        }

    def resume(
        self,
        run_id: str,
        resume_value: Any,
        *,
        user_id: str,
        user_info: Any,
        interrupt_id: str = "",
    ) -> Iterable[str]:
        """Resume an authenticated HITL interrupt on the existing thread."""
        if not self._use_langgraph or Command is None:
            raise ChatRunConflictError("当前运行模式不支持恢复")
        record = self._owned_run(run_id, user_id=user_id)
        self._assert_workflow_compatible(record)
        if record.status != "interrupted":
            raise ChatRunConflictError("该运行当前没有待恢复的审批")
        self._assert_recovery_window(record)
        config = {"configurable": {"thread_id": record.thread_id}}
        snapshot = self._graph.get_state(config)
        checkpoint_id, interrupt_ids = self._snapshot_tokens(snapshot)
        if not interrupt_ids:
            raise ChatRunConflictError("该运行没有可恢复的 interrupt")
        if interrupt_id and interrupt_id not in interrupt_ids:
            raise ChatRunConflictError("interrupt_id 不匹配或已被处理")
        selected_interrupt_id = interrupt_id or interrupt_ids[0]
        state = dict(getattr(snapshot, "values", None) or {})
        state.update({
            "run_id": record.run_id,
            "request_id": record.request_id,
            "user_id": record.user_id,
            "session_id": record.session_id,
        })

        def claimed_stream() -> Iterable[str]:
            execution_token = uuid.uuid4().hex
            if not self.deps.persistence.claim_resume(
                record.run_id,
                user_id=user_id,
                expected_checkpoint_id=checkpoint_id or None,
                expected_interrupt_id=selected_interrupt_id,
                new_execution_token=execution_token,
            ):
                yield sse({
                    "type": "error",
                    "message": "该 interrupt 已被处理或运行状态已变化",
                    "run_id": record.run_id,
                })
                return
            stream = self._stream_resume(
                state,
                self._new_run_context(
                    user_info=user_info,
                    execution_token=execution_token,
                    lease=ChatExecutionLease(
                        self.deps.persistence,
                        run_id=record.run_id,
                        user_id=user_id,
                        execution_token=execution_token,
                    ),
                    recovery=False,
                    source_workflow_version=record.workflow_version,
                ),
                Command(resume=resume_value),
                config,
            )
            yield from self._enrich_public_stream(stream, state)

        return claimed_stream()

    def recover(
        self,
        run_id: str,
        *,
        user_id: str,
        user_info: Any,
    ) -> Iterable[str]:
        """Continue a failed or stale run from its latest durable checkpoint."""
        if not self._use_langgraph:
            raise ChatRunConflictError("当前运行模式不支持恢复")
        record = self._owned_run(run_id, user_id=user_id)
        self._assert_workflow_compatible(record)
        if record.status not in {"failed", "running"}:
            raise ChatRunConflictError("该运行当前不可从 checkpoint 恢复")
        self._assert_recovery_window(record)
        if record.status == "running" and not self._run_lease_is_stale(record):
            raise ChatRunConflictError("该运行仍在活跃执行，不能抢占")

        config = {"configurable": {"thread_id": record.thread_id}}
        try:
            snapshot = self._graph.get_state(config)
        except Exception as exc:
            raise ChatRunConflictError("未找到可恢复的 checkpoint") from exc
        state = dict(getattr(snapshot, "values", None) or {})
        if not state:
            raise ChatRunConflictError("未找到可恢复的 checkpoint")
        state.update({
            "run_id": record.run_id,
            "request_id": record.request_id,
            "user_id": record.user_id,
            "session_id": record.session_id,
        })
        expected_updated_at = record.updated_at
        checkpoint_id, interrupt_ids = self._snapshot_tokens(snapshot)
        final_event = state.get("final_event")
        terminal_done = (
            not tuple(getattr(snapshot, "next", ()) or ())
            and not interrupt_ids
            and isinstance(final_event, dict)
            and final_event.get("type") == "done"
        )

        def claimed_stream() -> Iterable[str]:
            execution_token = uuid.uuid4().hex
            if not self.deps.persistence.claim_recovery(
                record.run_id,
                user_id=user_id,
                expected_updated_at=expected_updated_at,
                new_execution_token=execution_token,
            ):
                yield sse({
                    "type": "error",
                    "message": "该运行已被其他 worker 接管或状态已变化",
                    "run_id": record.run_id,
                })
                return
            context = self._new_run_context(
                user_info=user_info,
                execution_token=execution_token,
                lease=ChatExecutionLease(
                    self.deps.persistence,
                    run_id=record.run_id,
                    user_id=user_id,
                    execution_token=execution_token,
                ),
                recovery=True,
                source_workflow_version=record.workflow_version,
            )
            if terminal_done:
                stream = self._stream_terminal_recovery(
                    state,
                    context,
                    checkpoint_id=checkpoint_id,
                )
            else:
                stream = self._stream_recover(state, context, config)
            yield from self._enrich_public_stream(stream, state)

        return claimed_stream()

    def _new_run_context(
        self,
        *,
        user_info: Any,
        execution_token: str,
        lease: Any,
        recovery: bool,
        source_workflow_version: str = "",
    ) -> ChatRunContext:
        return ChatRunContext(
            user_info=user_info,
            execution_token=execution_token,
            lease=lease,
            recovery=recovery,
            source_workflow_version=(
                str(source_workflow_version or self.deps.workflow_version)
            ),
            document_runtime_factory=lambda state, prepare: (
                self._create_native_document_execution(
                    state,
                    user_info=user_info,
                    execution_token=execution_token,
                    recovery=recovery,
                    prepare=prepare,
                )
            ),
            document_node_observer=lambda state, node: self._mark_current_node(
                state,
                f"document.{node}",
                execution_token=execution_token,
                lease=lease,
            ),
            document_failure_callback=lambda state, step, exc: (
                self._record_native_document_failure(
                    state,
                    user_info=user_info,
                    step=step,
                    error=exc,
                    execution_token=execution_token,
                )
            ),
            document_update_sanitizer=(
                lambda state, update: self._sanitize_native_document_update(
                    state,
                    update,
                )
            ),
        )

    def _initial_state(self, raw_data: dict, *, user_id: str) -> ChatGraphState:
        data = raw_data if isinstance(raw_data, dict) else {}
        raw_message = str(data.get("message") or "")
        display_message = str(data.get("display_message") or raw_message)
        request_id = str(data.get("request_id") or uuid.uuid4().hex)
        idempotency_key = str(data.get("idempotency_key") or "")
        raw_file_ids = data.get("file_ids") or []
        if not isinstance(raw_file_ids, (list, tuple)):
            raw_file_ids = [raw_file_ids]
        file_ids = [str(file_id) for file_id in raw_file_ids if file_id is not None]
        existing_run = None
        if self._runtime_mode == "graph" and not self._session_id_is_explicit(data):
            existing_run = self._lookup_idempotent_run(
                str(user_id),
                idempotency_key or request_id,
            )
        if existing_run is not None:
            # The ledger lookup is already scoped by user_id. Reuse its durable
            # session before hashing the retry so a different Gunicorn worker's
            # empty process cache cannot manufacture a conflicting session.
            session_id = str(existing_run.session_id)
        else:
            session_id = self.deps.memory.get_or_create_session(
                user_id,
                data.get("session_id"),
            )
        run_id_factory = self.deps.run_id_factory or (lambda: str(uuid.uuid4()))
        initial_state = {
            # A run is a concurrency and recovery boundary.  It is always
            # generated by the server and never reused from client payload.
            "run_id": str(run_id_factory()),
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "user_id": str(user_id),
            "session_id": str(session_id),
            "request_message": raw_message,
            "message": raw_message,
            "display_message": display_message,
            "mode": str(data.get("mode") or "chat"),
            "file_ids": file_ids,
            "attachments": [],
            "step_index": 0,
            "working_message": raw_message,
            "tool_results": [],
            "current_step_done": {},
            "current_step_failed": False,
            "plan_events_emitted": False,
            "compat_orchestrator": False,
            "status": "received",
            "explicit_memory_recorded": False,
        }
        return validate_graph_update(initial_state, node="chat.initial_state")

    @staticmethod
    def _request_digest(state: ChatGraphState) -> str:
        """Hash the stable request meaning, excluding generated run identity."""

        attachment_refs = [
            str(file_id)
            for file_id in (state.get("file_ids") or [])
            if str(file_id)
        ]
        if not attachment_refs:
            attachment_refs = [
                str(item.get("file_id") or "")
                for item in (state.get("attachments") or [])
                if isinstance(item, dict) and str(item.get("file_id") or "")
            ]
        payload = {
            "version": 1,
            "user_id": str(state.get("user_id") or ""),
            "session_id": str(state.get("session_id") or ""),
            "message": str(
                state.get("request_message")
                if state.get("request_message") is not None
                else state.get("message") or ""
            ),
            "display_message": str(state.get("display_message") or ""),
            "mode": str(state.get("mode") or "chat"),
            "attachment_refs": attachment_refs,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _assert_request_identity(
        self,
        record: Any,
        state: ChatGraphState,
        *,
        allow_implicit_session_rebind: bool = False,
    ) -> None:
        if allow_implicit_session_rebind:
            record_session_id = str(getattr(record, "session_id", "") or "")
            if record_session_id:
                state["session_id"] = record_session_id
        metadata = getattr(record, "metadata", None)
        stored_digest = (
            str(metadata.get("request_digest") or "").strip()
            if isinstance(metadata, dict)
            else ""
        )
        expected_digest = self._request_digest(state)
        if stored_digest != expected_digest:
            raise ChatRunConflictError(
                "相同幂等键对应的请求内容不一致，请使用新的 request_id 或 idempotency_key"
            )

    def _lookup_idempotent_run(self, user_id: str, key: str) -> Any:
        persistence = self.deps.persistence
        lookup = (
            getattr(persistence, "get_run_by_idempotency_key", None)
            if persistence is not None
            else None
        )
        if not callable(lookup):
            return None
        normalized_key = str(key or "")
        if not normalized_key:
            return None
        return lookup(str(user_id), "chat", normalized_key)

    def _preflight_request_identity(
        self,
        state: ChatGraphState,
        *,
        allow_implicit_session_rebind: bool = False,
    ) -> None:
        key = str(state.get("idempotency_key") or state.get("request_id") or "")
        if not key:
            return
        record = self._lookup_idempotent_run(state["user_id"], key)
        if record is not None:
            self._assert_request_identity(
                record,
                state,
                allow_implicit_session_rebind=allow_implicit_session_rebind,
            )

    def _reclaim_unstarted_http_run(
        self,
        record: Any,
        state: ChatGraphState,
        *,
        execution_token: str,
    ) -> Any:
        """Atomically reuse a reservation abandoned before graph iteration.

        Other failed runs may already have checkpoints or durable effects and
        must continue through the explicit recovery API. This narrowly scoped
        sentinel is the only failed state known to have never entered the
        graph, so replaying the original initial State is safe.
        """

        if (
            record is None
            or getattr(record, "status", "") != "failed"
            or getattr(record, "error", "") != self.UNSTARTED_HTTP_ERROR
        ):
            return record
        persistence = self.deps.persistence
        claim_recovery = getattr(persistence, "claim_recovery", None)
        if not callable(claim_recovery):
            raise ChatRunConflictError("运行存储不支持原子重新认领")
        expected_updated_at = getattr(record, "updated_at", None)
        if expected_updated_at is None:
            raise ChatRunConflictError("运行记录缺少可验证的更新时间")
        claimed = claim_recovery(
            record.run_id,
            user_id=state["user_id"],
            expected_updated_at=expected_updated_at,
            new_execution_token=execution_token,
        )
        refreshed = persistence.get_owned_run(
            record.run_id,
            user_id=state["user_id"],
        )
        if refreshed is None:
            raise ChatRunConflictError("运行记录已删除或不属于当前用户")
        self._assert_request_identity(refreshed, state)
        if not claimed:
            # A concurrent retry may have won the CAS. Return its current
            # record so the normal idempotent replay path reports that status;
            # this request must never execute the graph with the loser's token.
            return refreshed
        if (
            getattr(refreshed, "status", "") != "running"
            or getattr(refreshed, "execution_token", "") != execution_token
        ):
            raise ChatRunConflictError("运行重新认领后的执行权无法验证")
        state.update({
            "run_id": refreshed.run_id,
            "request_id": refreshed.request_id,
            "session_id": refreshed.session_id,
        })
        return refreshed

    @staticmethod
    def _owns_reserved_run(
        record: Any,
        state: ChatGraphState,
        context: ChatRunContext,
    ) -> bool:
        return bool(
            record is not None
            and getattr(record, "status", "") == "running"
            and getattr(record, "run_id", "") == state.get("run_id")
            and getattr(record, "execution_token", "")
            == context.execution_token
        )

    def _abort_unstarted_http_run(
        self,
        state: ChatGraphState,
        context: ChatRunContext,
    ) -> None:
        # finish_run is an owner/token CAS. If iteration started or another
        # worker took over, this token cannot terminate that active execution.
        self._fail_persisted_run_best_effort(
            state["run_id"],
            user_id=state["user_id"],
            execution_token=context.execution_token,
            error=self.UNSTARTED_HTTP_ERROR,
        )

    def _stream_graph(
        self,
        initial_state: ChatGraphState,
        context: ChatRunContext,
    ) -> Iterable[str]:
        emitted_error = False
        emitted_done = False
        pending_done_chunks: list[str] = []
        persisted_status = "running"
        run_id = initial_state["run_id"]
        config = {
            "configurable": {"thread_id": run_id},
            "metadata": {
                "run_id": run_id,
                "request_id": initial_state["request_id"],
                "user_id": initial_state["user_id"],
                "session_id": initial_state["session_id"],
                "workflow_version": self.deps.workflow_version,
            },
        }
        lease = context.lease
        if lease is not None:
            lease.start()
        try:
            for envelope in self._graph.stream(
                initial_state,
                context=context,
                config=config,
                stream_mode="custom",
                version="v2",
                subgraphs=True,
                durability=self._durability(),
            ):
                if not isinstance(envelope, dict) or envelope.get("type") != "custom":
                    continue
                chunk = envelope.get("data")
                if not isinstance(chunk, str):
                    raise TypeError("Chat graph custom stream chunks must be strings")
                events = self._parse_chunk_events(chunk)
                emitted_error = emitted_error or any(
                    event.get("type") == "error" for event in events
                )
                emitted_done = emitted_done or any(
                    event.get("type") == "done" for event in events
                )
                if any(event.get("type") == "done" for event in events):
                    pending_done_chunks.append(chunk)
                else:
                    yield chunk
            self._assert_run_active(initial_state)
            if lease is not None:
                lease.assert_owned()
            checkpoint_id, interrupt_ids = self._checkpoint_tokens(config)
            interrupted = bool(interrupt_ids)
            status = (
                "interrupted"
                if interrupted
                else ("failed" if emitted_error or not emitted_done else "succeeded")
            )
            self._finish_persisted_run(
                run_id,
                user_id=initial_state["user_id"],
                execution_token=context.execution_token,
                status=status,
                checkpoint_id=checkpoint_id,
                interrupt_id=interrupt_ids[0] if interrupt_ids else "",
            )
            persisted_status = status
            if status == "succeeded":
                yield from pending_done_chunks
            elif status == "failed" and not emitted_error:
                yield sse(error_event("请求未能完成，请稍后重试"))
        except ChatRunCancelledError:
            self._cleanup_cancelled_run(initial_state)
            yield sse(error_event("会话已删除，当前运行已取消"))
        except GeneratorExit:
            if persisted_status == "running":
                self._fail_active_tool_effect_best_effort(
                    initial_state,
                    config=config,
                    execution_token=context.execution_token,
                    error="client disconnected before graph completion",
                )
                self._fail_persisted_run_best_effort(
                    run_id,
                    user_id=initial_state["user_id"],
                    execution_token=context.execution_token,
                    error="client disconnected before graph completion",
                )
            raise
        except Exception as exc:
            cancellation_event = self._cancellation_event_after_failure(
                initial_state
            )
            if cancellation_event is not None:
                yield cancellation_event
                return
            logger.exception(
                "Chat graph execution failed request_id=%s session_id=%s",
                initial_state.get("request_id", ""),
                initial_state.get("session_id", ""),
            )
            self._fail_active_tool_effect_best_effort(
                initial_state,
                config=config,
                execution_token=context.execution_token,
                error=str(exc) or "chat graph execution failed",
            )
            self._fail_persisted_run_best_effort(
                run_id,
                user_id=initial_state["user_id"],
                execution_token=context.execution_token,
                error="chat graph execution failed",
            )
            if not emitted_error:
                yield sse(error_event("请求处理失败，请稍后重试"))
        finally:
            if lease is not None:
                lease.stop()

    def _stream_resume(
        self,
        state: ChatGraphState,
        context: ChatRunContext,
        command: Any,
        config: dict[str, Any],
    ) -> Iterable[str]:
        return self._stream_checkpoint_continuation(
            state,
            context,
            command,
            config,
            current_node="resume",
            durability="sync",
            incomplete_message="恢复运行未能完成，请稍后重试",
            failure_message="恢复运行失败，请稍后重试",
        )

    def _stream_recover(
        self,
        state: ChatGraphState,
        context: ChatRunContext,
        config: dict[str, Any],
    ) -> Iterable[str]:
        return self._stream_checkpoint_continuation(
            state,
            context,
            None,
            config,
            current_node="recover",
            durability=self._durability(),
            incomplete_message="checkpoint 续跑未能完成，请稍后重试",
            failure_message="checkpoint 续跑失败，请稍后重试",
        )

    def _stream_terminal_recovery(
        self,
        state: ChatGraphState,
        context: ChatRunContext,
        *,
        checkpoint_id: str,
    ) -> Iterable[str]:
        """Commit and replay a final event already durable in the checkpoint."""
        lease = context.lease
        if lease is not None:
            lease.start()
        try:
            self._mark_current_node(
                state,
                "recover_terminal",
                execution_token=context.execution_token,
                lease=lease,
            )
            final_event = dict(state.get("final_event") or {})
            if final_event.get("type") != "done":
                raise ChatRunConflictError("checkpoint 不包含可重放的终态")
            self._finish_persisted_run(
                state["run_id"],
                user_id=state["user_id"],
                execution_token=context.execution_token,
                status="succeeded",
                checkpoint_id=checkpoint_id,
            )
            yield sse(final_event)
        except ChatRunCancelledError:
            self._cleanup_cancelled_run(state)
            yield sse(error_event("会话已删除，当前运行已取消"))
        except Exception:
            cancellation_event = self._cancellation_event_after_failure(state)
            if cancellation_event is not None:
                yield cancellation_event
                return
            logger.exception(
                "Terminal checkpoint recovery failed run_id=%s",
                state.get("run_id", ""),
            )
            self._fail_persisted_run_best_effort(
                state["run_id"],
                user_id=state["user_id"],
                execution_token=context.execution_token,
                error="terminal checkpoint recovery failed",
            )
            yield sse(error_event("checkpoint 终态恢复失败，请稍后重试"))
        finally:
            if lease is not None:
                lease.stop()

    def _stream_checkpoint_continuation(
        self,
        state: ChatGraphState,
        context: ChatRunContext,
        graph_input: Any,
        config: dict[str, Any],
        *,
        current_node: str,
        durability: str,
        incomplete_message: str,
        failure_message: str,
    ) -> Iterable[str]:
        run_id = state["run_id"]
        emitted_error = False
        emitted_done = False
        pending_done_chunks: list[str] = []
        persisted_status = "running"
        lease = context.lease
        if lease is not None:
            lease.start()
        try:
            if current_node == "recover" and graph_input is None:
                if self._fast_forward_completed_tool_effect(config):
                    latest = self._graph.get_state(config)
                    state.update(dict(getattr(latest, "values", None) or {}))
            self._mark_current_node(
                state,
                current_node,
                execution_token=context.execution_token,
                lease=lease,
            )
            for envelope in self._graph.stream(
                graph_input,
                context=context,
                config=config,
                stream_mode="custom",
                version="v2",
                subgraphs=True,
                durability=durability,
            ):
                if not isinstance(envelope, dict) or envelope.get("type") != "custom":
                    continue
                chunk = envelope.get("data")
                if not isinstance(chunk, str):
                    raise TypeError("Chat graph custom stream chunks must be strings")
                events = self._parse_chunk_events(chunk)
                emitted_error = emitted_error or any(
                    event.get("type") == "error" for event in events
                )
                emitted_done = emitted_done or any(
                    event.get("type") == "done" for event in events
                )
                if any(event.get("type") == "done" for event in events):
                    pending_done_chunks.append(chunk)
                else:
                    yield chunk
            self._assert_run_active(state)
            if lease is not None:
                lease.assert_owned()
            if not emitted_done:
                latest = self._graph.get_state(config)
                latest_values = dict(getattr(latest, "values", None) or {})
                checkpoint_done = latest_values.get("final_event")
                if (
                    not tuple(getattr(latest, "next", ()) or ())
                    and isinstance(checkpoint_done, dict)
                    and checkpoint_done.get("type") == "done"
                ):
                    pending_done_chunks.append(sse(checkpoint_done))
                    emitted_done = True
            checkpoint_id, interrupt_ids = self._checkpoint_tokens(config)
            interrupted = bool(interrupt_ids)
            status = (
                "interrupted"
                if interrupted
                else ("failed" if emitted_error or not emitted_done else "succeeded")
            )
            self._finish_persisted_run(
                run_id,
                user_id=state["user_id"],
                execution_token=context.execution_token,
                status=status,
                checkpoint_id=checkpoint_id,
                interrupt_id=interrupt_ids[0] if interrupt_ids else "",
            )
            persisted_status = status
            if status == "succeeded":
                yield from pending_done_chunks
            elif status == "failed" and not emitted_error:
                yield sse(error_event(incomplete_message))
        except ChatRunCancelledError:
            self._cleanup_cancelled_run(state)
            yield sse(error_event("会话已删除，当前运行已取消"))
        except GeneratorExit:
            if persisted_status == "running":
                self._fail_active_tool_effect_best_effort(
                    state,
                    config=config,
                    execution_token=context.execution_token,
                    error=f"client disconnected before graph {current_node} completed",
                )
                self._fail_persisted_run_best_effort(
                    run_id,
                    user_id=state["user_id"],
                    execution_token=context.execution_token,
                    error=f"client disconnected before graph {current_node} completed",
                )
            raise
        except Exception as exc:
            cancellation_event = self._cancellation_event_after_failure(state)
            if cancellation_event is not None:
                yield cancellation_event
                return
            logger.exception("Chat graph %s failed run_id=%s", current_node, run_id)
            self._fail_active_tool_effect_best_effort(
                state,
                config=config,
                execution_token=context.execution_token,
                error=str(exc) or f"chat graph {current_node} failed",
            )
            self._fail_persisted_run_best_effort(
                run_id,
                user_id=state["user_id"],
                execution_token=context.execution_token,
                error=f"chat graph {current_node} failed",
            )
            if not emitted_error:
                yield sse(error_event(failure_message))
        finally:
            if lease is not None:
                lease.stop()

    def _stream_planner(
        self,
        initial_state: ChatGraphState,
        context: ChatRunContext,
    ) -> Iterable[str]:
        state = dict(initial_state)
        state.update(self._prepare_node(state))
        prepared = self._prepared_from_state(state, context.user_info)
        task_plan = self._plan_request(prepared)
        return self.deps.tool_orchestrator.stream(prepared, task_plan)

    def _stream_legacy(
        self,
        initial_state: ChatGraphState,
        context: ChatRunContext,
    ) -> Iterable[str]:
        state = dict(initial_state)
        state.update(self._prepare_node(state))
        prepared = self._prepared_from_state(state, context.user_info)
        route = self._route_request(prepared)
        return self._dispatch_request(prepared, route)

    @staticmethod
    def _validated_state_node(
        node: Callable[[ChatGraphState], dict[str, Any]],
        node_name: str,
    ) -> Callable[[ChatGraphState], dict[str, Any]]:
        def invoke(state) -> dict[str, Any]:
            return validate_graph_update(node(state), node=f"chat.{node_name}")

        invoke.__name__ = f"validated_{node_name}"
        return invoke

    @staticmethod
    def _validated_runtime_node(
        node: Callable[..., dict[str, Any]],
        node_name: str,
    ) -> Callable[..., dict[str, Any]]:
        def invoke(
            state,
            runtime: Runtime[ChatRunContext],
        ) -> dict[str, Any]:
            return validate_graph_update(
                node(state, runtime),
                node=f"chat.{node_name}",
            )

        invoke.__name__ = f"validated_{node_name}"
        return invoke

    def _build_graph(self):
        if StateGraph is None:
            raise RuntimeError(f"LangGraph is not available: {LANGGRAPH_IMPORT_ERROR}")

        workflow = StateGraph(ChatGraphState, context_schema=ChatRunContext)
        workflow.add_node(
            "prepare", self._validated_state_node(self._prepare_node, "prepare")
        )
        workflow.add_node(
            "plan_tools",
            self._validated_runtime_node(self._plan_tools_node, "plan_tools"),
        )
        workflow.add_node(
            "select_step",
            self._validated_runtime_node(self._select_step_node, "select_step"),
        )
        tool_nodes = {
            TOOL_KNOWLEDGE_QA: "tool_knowledge_qa",
            TOOL_DRAFT_DOCUMENT: "tool_draft_document",
            TOOL_FORMAT_DOCUMENT: "tool_format_document",
            TOOL_PREPARE_FORM_EXPORT: "tool_prepare_form_export",
            TOOL_PREPARE_SPREADSHEET_TRANSFORM: "tool_prepare_spreadsheet_transform",
            TOOL_CLARIFY: "tool_clarify",
            TOOL_IDENTITY_HELP: "tool_identity_help",
        }
        for tool_name, node_name in tool_nodes.items():
            if tool_name == TOOL_DRAFT_DOCUMENT:
                # Compile without a saver: as a native child node it inherits
                # the parent's thread/checkpointer and receives its own nested
                # checkpoint namespace from LangGraph.
                workflow.add_node(
                    node_name,
                    self._build_tool_subgraph(tool_name),
                )
            else:
                workflow.add_node(
                    node_name,
                    self._validated_runtime_node(
                        self._tool_node(tool_name), node_name
                    ),
                )
        workflow.add_node(
            "unsupported_tool",
            self._validated_runtime_node(
                self._unsupported_tool_node, "unsupported_tool"
            ),
        )
        workflow.add_node(
            "execute_tools_compat",
            self._validated_runtime_node(
                self._execute_tools_node, "execute_tools_compat"
            ),
        )
        workflow.add_node(
            "collect_result",
            self._validated_runtime_node(
                self._collect_result_node, "collect_result"
            ),
        )
        workflow.add_node(
            "finalize",
            self._validated_runtime_node(self._finalize_node, "finalize"),
        )
        workflow.add_node(
            "error_terminal",
            self._validated_state_node(
                self._error_terminal_node, "error_terminal"
            ),
        )
        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "plan_tools")
        workflow.add_edge("plan_tools", "select_step")
        workflow.add_conditional_edges(
            "select_step",
            self._route_selected_step,
            {
                **{tool_name: node_name for tool_name, node_name in tool_nodes.items()},
                "unsupported": "unsupported_tool",
                "compat": "execute_tools_compat",
                "finalize": "finalize",
            },
        )
        for node_name in tool_nodes.values():
            workflow.add_edge(node_name, "collect_result")
        workflow.add_edge("unsupported_tool", "error_terminal")
        workflow.add_edge("execute_tools_compat", END)
        workflow.add_conditional_edges(
            "collect_result",
            self._route_after_collect,
            {
                "select_step": "select_step",
                "finalize": "finalize",
                "error": "error_terminal",
            },
        )
        workflow.add_edge("finalize", END)
        workflow.add_edge("error_terminal", END)
        return workflow.compile(checkpointer=self._checkpointer)

    def _build_tool_subgraph(self, tool_name: str):
        if tool_name == TOOL_DRAFT_DOCUMENT and self._native_document_service() is not None:
            child = StateGraph(
                DocumentToolGraphState,
                context_schema=ChatRunContext,
            )
            child.add_node(
                "document_dispatch",
                self._validated_state_node(
                    lambda _state: {}, "document_dispatch"
                ),
            )
            child.add_node(
                "document_enter",
                self._validated_runtime_node(
                    self._native_document_enter, "document_enter"
                ),
            )
            child.add_node(
                "document_workflow",
                DocumentGraphRunner.build_native_subgraph(),
            )
            child.add_node(
                "document_commit",
                self._validated_runtime_node(
                    self._native_document_commit, "document_commit"
                ),
            )

            # ``execute`` is intentionally retained for one release. A chat-v2
            # checkpoint may already have this child node scheduled; keeping the
            # name lets the new topology resume it without replaying prior nodes.
            child.add_node(
                "execute",
                self._validated_runtime_node(
                    self._tool_node(tool_name), "document_execute_legacy"
                ),
            )
            child.set_entry_point("document_dispatch")
            child.add_conditional_edges(
                "document_dispatch",
                self._route_native_document_dispatch,
                {
                    "native": "document_enter",
                    "legacy": "execute",
                },
            )
            child.add_conditional_edges(
                "document_enter",
                self._route_native_document_entry,
                {
                    "document": "document_workflow",
                    "replay": "document_commit",
                },
            )
            child.add_edge("document_workflow", "document_commit")
            child.add_edge("document_commit", END)
            child.add_edge("execute", END)
            return child.compile(name=f"{tool_name}_subgraph")

        child = StateGraph(ChatGraphState, context_schema=ChatRunContext)
        child.add_node(
            "execute",
            self._validated_runtime_node(
                self._tool_node(tool_name), f"{tool_name}_execute"
            ),
        )
        child.set_entry_point("execute")
        child.add_edge("execute", END)
        return child.compile(name=f"{tool_name}_subgraph")

    def _native_document_service(self) -> Any:
        """Return the production draft service when it supports native mounting."""

        candidates = []
        orchestrator = self.deps.tool_orchestrator
        registry = getattr(orchestrator, "registry", None)
        if registry is not None:
            try:
                tool = registry.get(TOOL_DRAFT_DOCUMENT)
            except (KeyError, AttributeError):
                pass
            else:
                candidates.append(getattr(tool.stream, "__self__", None))
        candidates.append(getattr(self.deps.document_draft_stream, "__self__", None))
        for candidate in candidates:
            if candidate is None:
                continue
            if all(
                callable(getattr(candidate, method, None))
                for method in (
                    "create_native_execution",
                    "build_public_done",
                    "_record_failure",
                    "_update_common_doc_types",
                )
            ):
                return candidate
        return None

    @staticmethod
    def _route_native_document_entry(state: DocumentToolGraphState) -> str:
        return "replay" if state.get("native_document_replay") else "document"

    def _route_native_document_dispatch(
        self,
        _state: DocumentToolGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> str:
        # If LangGraph cannot attach a rolling-deploy child checkpoint to the
        # expanded schema, still force the previous release through its retained
        # execute node instead of treating missing document fields as a result.
        if runtime.context.source_workflow_version == self.PREVIOUS_WORKFLOW_VERSION:
            return "legacy"
        return "native"

    @staticmethod
    def _draft_step(state: dict[str, Any]):
        task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
        index = int(state.get("step_index", 0) or 0)
        if index >= len(task_plan.steps):
            raise RuntimeError("Selected document step is out of range")
        step = task_plan.steps[index]
        if step.tool != TOOL_DRAFT_DOCUMENT:
            raise RuntimeError(
                f"Selected tool mismatch: expected {TOOL_DRAFT_DOCUMENT}, got {step.tool}"
            )
        return task_plan, index, step

    def _raise_native_document_boundary_failure(
        self,
        state: dict[str, Any],
        runtime: Runtime[ChatRunContext],
        *,
        step: str,
        error: Exception,
    ) -> None:
        self._record_native_document_failure(
            state,
            user_info=runtime.context.user_info,
            step=step,
            error=error,
            execution_token=runtime.context.execution_token,
        )
        runtime.stream_writer(sse({
            "type": "run_failed",
            "message": str(error)[:500],
            "step": step,
        }))
        runtime.stream_writer(sse({
            "type": "error",
            "message": "生成失败，请稍后重试",
        }))
        raise error

    def _native_document_enter(
        self,
        state: DocumentToolGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> DocumentToolGraphState:
        node_name = f"tool_{TOOL_DRAFT_DOCUMENT}"
        self._mark_current_node(
            state,
            "document.enter",
            execution_token=runtime.context.execution_token,
            lease=runtime.context.lease,
        )
        _task_plan, index, _step = self._draft_step(state)
        effect_name = f"step:{index + 1}:{TOOL_DRAFT_DOCUMENT}"
        claimed, replay = self._claim_effect(
            state,
            node=node_name,
            effect=effect_name,
            execution_token=runtime.context.execution_token,
            allow_reclaim=runtime.context.recovery,
        )
        if not claimed:
            if not isinstance(replay, dict):
                raise RuntimeError("Persisted document effect has no reusable result")
            return {
                "native_document_replay": True,
                "current_step_done": dict(replay.get("step_done") or {}),
                "current_step_failed": bool(replay.get("failed")),
                "status": "tool_replayed",
            }

        try:
            runtime.stream_writer(sse({
                "type": "thinking_start",
                "message": "开始拆解写作任务",
            }))
            execution = runtime.context.get_document_execution(state, prepare=True)
            prepared_run = getattr(execution, "prepared_run", None)
            if prepared_run is None:
                raise RuntimeError("Native document runtime did not prepare a run")
            initial = DocumentGraphRunner._initial_state(prepared_run, state["run_id"])
            safe_request = str(
                state.get("working_message")
                or state.get("request_message")
                or state.get("message")
                or ""
            )
            initial.update({
                "run_id": state["run_id"],
                "session_id": state["session_id"],
                "user_id": state["user_id"],
                "display_message": state.get("display_message", ""),
                "parent_step_index": index,
                "native_document_replay": False,
                # Keep attachment bodies and prior-turn hydrated prompts in
                # Runtime.context. These fields are checkpointed, so they hold
                # only the original text and controlled artifact references.
                "input_user_request": safe_request,
                "request_with_context": safe_request,
                "previous_context": "",
                "user_request": safe_request,
            })
            return initial
        except Exception as exc:
            self._raise_native_document_boundary_failure(
                state,
                runtime,
                step="enter",
                error=exc,
            )
            raise  # pragma: no cover - helper always raises

    def _native_document_commit(
        self,
        state: DocumentToolGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> DocumentToolGraphState:
        try:
            return self._native_document_commit_impl(state, runtime)
        except Exception as exc:
            self._raise_native_document_boundary_failure(
                state,
                runtime,
                step="commit",
                error=exc,
            )
            raise  # pragma: no cover - helper always raises

    def _native_document_commit_impl(
        self,
        state: DocumentToolGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> DocumentToolGraphState:
        self._mark_current_node(
            state,
            "document.commit",
            execution_token=runtime.context.execution_token,
            lease=runtime.context.lease,
        )
        if state.get("native_document_replay"):
            return {
                "current_step_done": dict(state.get("current_step_done") or {}),
                "current_step_failed": bool(state.get("current_step_failed")),
                "status": "tool_replayed",
            }

        service = self._native_document_service()
        if service is None:
            raise RuntimeError("Native document service is unavailable")
        task_plan, index, step = self._draft_step(state)
        execution = runtime.context.get_document_execution(state)
        orchestrator = execution.orchestrator
        document_content = str(state.get("document_content") or "")
        if not document_content or state.get("run_status") == "failed":
            raise RuntimeError(
                str(state.get("error_message") or "Document workflow produced no content")
            )

        ctx = DocumentGraphSteps._context_from_state(
            orchestrator,
            state,
            runtime,
        )
        ctx.run_records.append({
            "step": "orchestrator_runtime",
            "runtime": "langgraph",
            "stream": True,
            "quality_status": state.get("quality_status", "passed"),
        })
        response = orchestrator._build_document_run_result(
            ctx,
            document_content,
            str(
                getattr(execution.prepared_run, "user_request", "")
                or state.get("request_message", "")
            ),
            runtime="langgraph",
            quality_status=state.get("quality_status", "passed"),
            revisions_applied=int(state.get("revision_round", 0) or 0),
            run_id=state["run_id"],
            effect_scope=str(
                getattr(execution, "effect_scope", "")
                or f"step:{index + 1}"
            ),
        )

        # Node-level model think messages were sanitized before checkpointing.
        # Rebuild the public log from that safe State instead of the live
        # orchestrator, which still holds runtime-only prompt context.
        safe_think_update = runtime.context.sanitize_document_update(
            state,
            {"think_log": list(orchestrator.think_log or [])},
        )
        validate_graph_update(
            safe_think_update,
            node="chat.document_commit_think_log",
        )
        orchestrator.think_log = [
            dict(item)
            for item in (safe_think_update.get("think_log") or [])
            if isinstance(item, dict)
        ]
        final_message = (
            "当前最优版本已确认，正在输出正文"
            if state.get("quality_status") == "max_revisions"
            else "最终版本已确认，正在输出正文"
        )
        think_handler = orchestrator._think_handler()
        think_handler("Orchestrator", "📄", final_message)
        completion = f"文档生成完成，共{len(orchestrator.think_log)}个思考步骤"
        think_handler("Orchestrator", "✅", completion)

        # Finish every fallible business write and persist the idempotent
        # receipt before exposing answer_start/answer_done. If anything below
        # this boundary fails, the client has not been told that the document
        # completed and recovery can safely retry the commit.
        lease = runtime.context.lease
        if lease is not None:
            lease.assert_owned()
        route = self.deps.tool_orchestrator._route_for_step(step, task_plan.route)
        effect_scope = str(
            getattr(execution, "effect_scope", "")
            or f"step:{index + 1}"
        )
        public_done = service.build_public_done(
            {"type": "done", **response},
            think_log=orchestrator.think_log,
            user_id=state["user_id"],
            user_info=runtime.context.user_info,
            session_id=state["session_id"],
            stored_user_message=execution.stored_user_message,
            route=route,
            parent_run_id=state["run_id"],
            effect_scope=effect_scope,
        )
        service._update_common_doc_types(
            execution.profile,
            state["user_id"],
            execution.stored_user_message,
        )
        if lease is not None:
            lease.assert_owned()
        effect_name = f"step:{index + 1}:{TOOL_DRAFT_DOCUMENT}"
        self._complete_effect(
            state,
            node=f"tool_{TOOL_DRAFT_DOCUMENT}",
            effect=effect_name,
            result={"step_done": public_done, "failed": False},
            execution_token=runtime.context.execution_token,
        )

        runtime.stream_writer(sse({
            "type": "think",
            "agent": "Orchestrator",
            "emoji": "📄",
            "message": final_message,
        }))
        runtime.stream_writer(sse({
            "type": "thinking_done",
            "summary": "写作、审核和反思完成，开始输出最终正文",
        }))
        is_final_step = index == len(task_plan.steps) - 1
        if is_final_step:
            runtime.stream_writer(sse({
                "type": "answer_start",
                "message": "开始输出正文",
            }))
            for chunk in DocumentGraphRunner._iter_final_content_chunks(document_content):
                runtime.stream_writer(sse({"type": "answer_delta", "data": chunk}))
                runtime.stream_writer(sse({"type": "content", "data": chunk}))
            runtime.stream_writer(sse({
                "type": "answer_done",
                "answer": document_content,
            }))
        runtime.stream_writer(sse({
            "type": "think",
            "agent": "Orchestrator",
            "emoji": "✅",
            "message": completion,
        }))
        return {
            "current_step_done": public_done,
            "current_step_failed": False,
            "status": "tool_completed",
        }

    def _create_native_document_execution(
        self,
        state: dict[str, Any],
        *,
        user_info: Any,
        execution_token: str,
        recovery: bool,
        prepare: bool,
    ) -> Any:
        service = self._native_document_service()
        if service is None:
            raise RuntimeError("Native document service is unavailable")
        index = int(
            state.get("parent_step_index", state.get("step_index", 0)) or 0
        )
        effect_scope = f"step:{index + 1}"

        # A resumed nested node does not revisit document_enter. Transfer the
        # effect fence to this worker before rebuilding runtime-only services.
        if recovery and not prepare:
            claimed, replay = self._claim_effect(
                state,
                node=f"tool_{TOOL_DRAFT_DOCUMENT}",
                effect=f"step:{index + 1}:{TOOL_DRAFT_DOCUMENT}",
                execution_token=execution_token,
                allow_reclaim=True,
            )
            if not claimed:
                raise ChatRunConflictError(
                    "Document effect completed before its parent checkpoint; "
                    "recover the parent result instead"
                )

        working_state = dict(state)
        working_state["message"] = (
            state.get("working_message")
            or state.get("request_message")
            or state.get("message")
            or state.get("input_user_request")
            or ""
        )
        message = self._prepared_from_state(
            working_state,
            user_info,
        ).message
        artifact_store = self.deps.artifact_store
        frozen_snapshot: dict[str, Any] = {}
        load_snapshot = getattr(
            artifact_store,
            "load_document_runtime_context",
            None,
        )
        if callable(load_snapshot):
            loaded = load_snapshot(
                run_id=str(state.get("run_id") or ""),
                user_id=str(state.get("user_id") or ""),
            )
            if isinstance(loaded, dict):
                frozen_snapshot = dict(loaded)

        save_snapshot = getattr(
            artifact_store,
            "snapshot_document_runtime_context",
            None,
        )
        if not frozen_snapshot and callable(save_snapshot):
            frozen_snapshot = save_snapshot(
                run_id=str(state.get("run_id") or ""),
                user_id=str(state.get("user_id") or ""),
                payload=self._capture_document_runtime_snapshot(
                    state,
                    message=message,
                ),
            )

        display_message = state.get("display_message")
        if display_message is None:
            display_message = (
                state.get("request_message")
                if state.get("request_message") is not None
                else state.get("input_user_request") or ""
            )
        execution = service.create_native_execution(
            message=message,
            # Preserve an explicit empty display value for file-only requests.
            # Falling back to the hydrated ``message`` would write attachment
            # bodies into conversation memory.
            display_message=str(display_message),
            session_id=str(state.get("session_id") or ""),
            user_id=str(state.get("user_id") or ""),
            user_info=user_info,
            run_id=str(state.get("run_id") or ""),
            effect_scope=effect_scope,
            # Preparation is retry-safe for a run_id and reconstructs the
            # runtime-only hydrated request after a worker restart.  It remains
            # outside checkpoint State even when this factory is called while
            # resuming an inner document node.
            prepare=True,
            runtime_snapshot=frozen_snapshot or None,
        )
        execution.attachment_bodies = self._attachment_contents_from_state(state)
        snapshot = getattr(execution, "runtime_snapshot", None)
        if not isinstance(snapshot, dict):
            snapshot = {}
        if frozen_snapshot:
            snapshot = frozen_snapshot
        execution.runtime_snapshot = dict(snapshot)
        execution.orchestrator._document_runtime_snapshot = dict(snapshot)
        execution.orchestrator._current_agent_memory_context = str(
            snapshot.get("memory_context") or ""
        )
        frozen_profile = snapshot.get("user_profile")
        if isinstance(frozen_profile, dict) and frozen_profile:
            setter = getattr(execution.orchestrator, "set_user_profile", None)
            if callable(setter):
                setter(dict(frozen_profile))
            else:
                execution.orchestrator.user_profile = dict(frozen_profile)
        if state.get("think_log"):
            execution.orchestrator.think_log = list(state.get("think_log") or [])
        return execution

    def _capture_document_runtime_snapshot(
        self,
        state: dict[str, Any],
        *,
        message: str,
    ) -> dict[str, Any]:
        """Freeze memory/profile inputs before document preparation mutates them."""

        memory = self.deps.memory
        session_id = str(state.get("session_id") or "")
        user_id = str(state.get("user_id") or "")
        run_id = str(state.get("run_id") or "")

        profile_reader = getattr(memory, "get_user_profile", None)
        profile = profile_reader(user_id) if callable(profile_reader) else None
        if isinstance(profile, dict):
            user_profile = dict(profile)
        else:
            user_profile = {
                name: getattr(profile, name)
                for name in (
                    "preferred_font",
                    "preferred_size",
                    "writing_style",
                    "common_doc_types",
                    "name",
                    "department",
                )
                if profile is not None and hasattr(profile, name)
            }

        history_reader = getattr(memory, "get_session_history", None)
        history = (
            history_reader(session_id, limit=10)
            if callable(history_reader)
            else []
        )
        conversation_history: list[dict[str, str]] = []
        for item in history or []:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
            else:
                role = getattr(item, "role", "")
                content = getattr(item, "content", "")
            conversation_history.append({
                "role": str(role or ""),
                "content": str(content or ""),
            })
        memory_context = ""
        recall = getattr(memory, "get_agent_context_for_prompt", None)
        try:
            if callable(recall):
                try:
                    memory_context = recall(
                        session_id,
                        memory_query=message,
                        max_messages=6,
                        max_chars=3600,
                    )
                except TypeError:
                    memory_context = recall(session_id, memory_query=message)
            else:
                fallback = getattr(memory, "get_context_for_prompt", None)
                if callable(fallback):
                    try:
                        memory_context = fallback(
                            session_id,
                            max_messages=6,
                            memory_query=message,
                        )
                    except TypeError:
                        memory_context = fallback(session_id, max_messages=6)
        except Exception:
            logger.exception(
                "Could not freeze document memory context run_id=%s",
                run_id,
            )
            memory_context = ""
        memory_context = str(memory_context or "").strip()
        if len(memory_context) > 3600:
            memory_context = memory_context[:3597] + "..."

        context_reader = getattr(memory, "get_context", None)
        last_document = ""
        last_plan: dict[str, Any] = {}
        previous_context = ""
        if callable(context_reader):
            last_document = str(
                context_reader(session_id, "last_document", "") or ""
            )
            raw_last_plan = context_reader(session_id, "last_plan", {}) or {}
            if isinstance(raw_last_plan, dict):
                last_plan = dict(raw_last_plan)
            prepared_snapshot = context_reader(
                session_id,
                f"document_prepare:{run_id}",
                None,
            )
            if isinstance(prepared_snapshot, dict):
                previous_context = str(
                    prepared_snapshot.get("previous_context") or ""
                )
            else:
                previous_context = str(
                    context_reader(session_id, "last_request", "") or ""
                )

        return {
            "memory_context": memory_context,
            "user_profile": user_profile,
            "last_document": last_document,
            "last_plan": last_plan,
            "previous_context": previous_context,
            "conversation_history": conversation_history,
        }

    def _record_native_document_failure(
        self,
        state: dict[str, Any],
        *,
        user_info: Any,
        step: str,
        error: Exception,
        execution_token: str = "",
    ) -> None:
        index = int(
            state.get("parent_step_index", state.get("step_index", 0)) or 0
        )
        try:
            self._fail_effect(
                state,
                node=f"tool_{TOOL_DRAFT_DOCUMENT}",
                effect=f"step:{index + 1}:{TOOL_DRAFT_DOCUMENT}",
                error=f"{step}: {error}",
                execution_token=execution_token,
            )
        except Exception:
            # Failure accounting must not replace the original Writer,
            # Reviewer, retrieval, or commit exception.
            logger.exception(
                "Could not fail native document effect run_id=%s step=%s",
                state.get("run_id", ""),
                step,
            )
        service = self._native_document_service()
        if service is None:
            return
        message = str(
            state.get("input_user_request")
            or state.get("working_message")
            or state.get("request_message")
            or ""
        )
        try:
            service._record_failure(
                str(state.get("user_id") or ""),
                user_info,
                str(state.get("session_id") or ""),
                message,
                f"{step}: {error}",
                run_id=str(state.get("run_id") or ""),
                effect_scope=f"step:{index + 1}",
            )
        except Exception:
            logger.exception(
                "Could not record native document failure run_id=%s step=%s",
                state.get("run_id", ""),
                step,
            )

    def _prepare_node(
        self,
        state: ChatGraphState,
        runtime: Optional[Runtime[ChatRunContext]] = None,
    ) -> ChatGraphState:
        self._mark_current_node(
            state,
            "prepare",
            execution_token=(runtime.context.execution_token if runtime else ""),
            lease=(runtime.context.lease if runtime else None),
        )
        user_id = state["user_id"]
        raw_message = state.get("request_message", "")
        display_message = state.get("display_message") or raw_message
        session_id = state["session_id"]
        memory_recorded = bool(state.get("explicit_memory_recorded"))
        remember_explicit = getattr(self.deps.memory, "remember_explicit_memory", None)
        if not memory_recorded and callable(remember_explicit):
            # Long-term memory is opt-in at the user-message boundary.  Raw
            # uploads and model output are deliberately never persisted here.
            claimed, replay = self._claim_effect(
                state,
                node="prepare",
                effect="explicit_memory",
                execution_token=(
                    runtime.context.execution_token if runtime else ""
                ),
                allow_reclaim=bool(runtime and runtime.context.recovery),
            )
            if claimed:
                try:
                    remember_explicit(user_id, session_id, display_message)
                    self._complete_effect(
                        state,
                        node="prepare",
                        effect="explicit_memory",
                        result={"recorded": True},
                        execution_token=(
                            runtime.context.execution_token if runtime else ""
                        ),
                    )
                except Exception as exc:
                    self._fail_effect(
                        state,
                        node="prepare",
                        effect="explicit_memory",
                        error=str(exc),
                        execution_token=(
                            runtime.context.execution_token if runtime else ""
                        ),
                    )
                    raise
            elif not isinstance(replay, dict) or not replay.get("recorded"):
                raise RuntimeError("Persisted memory effect has no reusable result")
            memory_recorded = True
        attachments: list[dict[str, Any]] = []
        if self.deps.artifact_store is not None:
            attachments = self.deps.artifact_store.snapshot_attachments(
                run_id=state["run_id"],
                user_id=user_id,
                file_ids=list(state.get("file_ids") or []),
            )
        else:
            # Unit-test and explicit rollback wiring can omit the durable store;
            # even then checkpoint state contains only references, never bodies.
            for file_id in state.get("file_ids", []) or []:
                content = self.deps.upload_manager.get_temp_content(file_id, user_id)
                if not content:
                    continue
                info = self.deps.upload_manager.get_temp_file_info(file_id, user_id) or {}
                filename = str(info.get("filename") or file_id)
                attachments.append({
                    "file_id": str(file_id),
                    "filename": filename,
                    "char_count": len(info.get("content") or content or ""),
                    "is_spreadsheet": Path(filename).suffix.lower() in {".xlsx", ".xls", ".csv"},
                })

        return {
            "message": raw_message,
            "file_ids": [],
            "attachments": attachments,
            "status": "prepared",
            "explicit_memory_recorded": memory_recorded,
        }

    def _prepared_from_state(self, state: ChatGraphState, user_info: Any) -> PreparedChatRequest:
        attachments = list(state.get("attachments") or [])
        message = self._hydrate_attachment_message(state, attachments)
        step_index = int(
            state.get("parent_step_index", state.get("step_index", 0)) or 0
        )
        metadata = {
            "run_id": state.get("run_id", ""),
            "request_id": state.get("request_id", ""),
            "step_index": step_index,
            "effect_scope": f"step:{step_index + 1}",
        }
        if attachments:
            metadata["attached_files"] = attachments
        return PreparedChatRequest(
            run_id=state.get("run_id", ""),
            request_id=state.get("request_id", ""),
            message=message,
            display_message=state.get("display_message", ""),
            mode=state.get("mode", "chat"),
            session_id=state["session_id"],
            user_id=state["user_id"],
            user_info=user_info,
            user_metadata=metadata,
            attachments=attachments,
        )

    def _hydrate_attachment_message(
        self,
        state: ChatGraphState,
        attachments: list[dict[str, Any]],
    ) -> str:
        message = state.get("message", "")
        if not attachments:
            return message
        if self.deps.artifact_store is not None:
            return self.deps.artifact_store.hydrate_message(
                run_id=state["run_id"],
                user_id=state["user_id"],
                message=message,
                attachments=attachments,
            )

        blocks = []
        for attachment in attachments:
            file_id = str(attachment.get("file_id") or "")
            content = self.deps.upload_manager.get_temp_content(
                file_id,
                state["user_id"],
            )
            if content:
                blocks.append(f"[文件内容]\n{content}\n[/文件内容]")
        if not blocks:
            return message
        return "\n\n".join(blocks) + "\n\n[用户提问]\n" + message

    def _sanitize_native_document_update(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Remove verbatim upload bodies from model-derived checkpoint fields.

        Any model-derived metadata can echo attachment, profile, memory, or
        history text.  Sanitize the complete node delta; only the active draft
        is preserved verbatim because ``document_content`` is the intended
        first-class output of the workflow.
        """

        redaction_rules = self._document_redaction_rules(state)
        if not redaction_rules:
            return update

        def redact(value: Any) -> Any:
            if isinstance(value, str):
                result = value
                for fragment, replacement in redaction_rules:
                    result = result.replace(fragment, replacement)
                return result
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {
                    (redact(key) if isinstance(key, str) else key): redact(item)
                    for key, item in value.items()
                }
            return value

        sanitized = {}
        for field_name, value in update.items():
            sanitized[field_name] = (
                value if field_name == "document_content" else redact(value)
            )
        return sanitized

    def _attachment_contents_from_state(
        self,
        state: dict[str, Any],
    ) -> list[str]:
        contents: list[str] = []
        for raw_content in state.get("_runtime_attachment_bodies", []) or []:
            normalized = str(raw_content or "").strip()
            if normalized:
                contents.append(normalized)
        for attachment in state.get("attachments", []) or []:
            content = ""
            if self.deps.artifact_store is not None:
                content = self.deps.artifact_store.load_content(
                    run_id=str(state.get("run_id") or ""),
                    artifact_id=str(attachment.get("artifact_id") or ""),
                    user_id=str(state.get("user_id") or ""),
                )
            else:
                file_id = str(attachment.get("file_id") or "")
                if file_id:
                    content = self.deps.upload_manager.get_temp_content(
                        file_id,
                        str(state.get("user_id") or ""),
                    ) or ""
            normalized = str(content or "").strip()
            if normalized:
                contents.append(normalized)

        return list(dict.fromkeys(contents))

    def _document_redaction_rules(
        self,
        state: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """Build longest-first rules for runtime data that State must not keep."""

        contents = self._attachment_contents_from_state(state)
        replacements: dict[str, str] = {}
        for content in contents:
            replacements[content] = "[附件正文见受控引用]"
            for part in re.split(r"(?:\r?\n)+|(?<=[。！？!?])", content):
                part = part.strip()
                if len(part) >= 12:
                    replacements[part] = "[附件正文见受控引用]"

        # Memory, profile and history are restored from an immutable artifact
        # at runtime.  LLM nodes may echo those inputs into their plan or think
        # log, so remove exact non-trivial leaves before checkpointing them.
        snapshot = state.get("_runtime_document_snapshot") or {}

        def collect(value: Any) -> None:
            if isinstance(value, str):
                normalized = value.strip()
                if len(normalized) >= 4:
                    replacements.setdefault(
                        normalized,
                        "[运行时上下文见受控快照]",
                    )
                    for part in re.split(
                        r"(?:\r?\n)+|(?<=[。！？!?])",
                        normalized,
                    ):
                        fragment = part.strip()
                        if len(fragment) >= 12:
                            replacements.setdefault(
                                fragment,
                                "[运行时上下文见受控快照]",
                            )
                return
            if isinstance(value, list):
                for item in value:
                    collect(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    collect(item)

        if isinstance(snapshot, dict):
            collect(snapshot)

        return sorted(
            replacements.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    def _durability(self) -> str:
        persistence = self.deps.persistence
        if persistence is None:
            return "async"
        return str(persistence.config.default_durability)

    def _claim_effect(
        self,
        state: ChatGraphState,
        *,
        node: str,
        effect: str,
        execution_token: str = "",
        allow_reclaim: bool = False,
    ) -> tuple[bool, Any]:
        persistence = self.deps.persistence
        claim = getattr(persistence, "claim_effect", None) if persistence else None
        if not callable(claim):
            return True, None
        if claim(
            state["run_id"],
            node,
            effect,
            user_id=state["user_id"],
            execution_token=execution_token or None,
            allow_reclaim=allow_reclaim,
            stale_seconds=(
                self.RECOVERY_LEASE_STALE_SECONDS if allow_reclaim else None
            ),
        ):
            return True, None
        # A failed claim can mean either replay/conflict or that a session
        # deletion barrier won the database race after the node-level check.
        # Recheck the durable barrier before inspecting an old receipt.
        self._assert_run_active(state)
        record = persistence.get_effect(
            state["run_id"],
            node,
            effect,
            user_id=state["user_id"],
        )
        if record is not None and record.status == "completed":
            return False, record.result
        status = getattr(record, "status", "missing")
        raise ChatRunConflictError(
            f"Effect {node}:{effect} cannot be replayed safely (status={status})"
        )

    def _complete_effect(
        self,
        state: ChatGraphState,
        *,
        node: str,
        effect: str,
        result: Any,
        execution_token: str = "",
    ) -> None:
        persistence = self.deps.persistence
        complete = getattr(persistence, "complete_effect", None) if persistence else None
        if callable(complete):
            record = complete(
                state["run_id"],
                node,
                effect,
                user_id=state["user_id"],
                execution_token=execution_token or None,
                result=result,
            )
            if record is None:
                self._assert_run_active(state)
                raise ChatRunConflictError(
                    f"Effect {node}:{effect} completion lease was lost"
                )

    def _fail_effect(
        self,
        state: ChatGraphState,
        *,
        node: str,
        effect: str,
        error: str,
        execution_token: str = "",
    ) -> None:
        persistence = self.deps.persistence
        fail = getattr(persistence, "fail_effect", None) if persistence else None
        if callable(fail):
            record = fail(
                state["run_id"],
                node,
                effect,
                user_id=state["user_id"],
                execution_token=execution_token or None,
                error=str(error or "")[:2000],
            )
            if record is None:
                self._assert_run_active(state)
                raise ChatRunConflictError(
                    f"Effect {node}:{effect} failure lease was lost"
                )

    @staticmethod
    def _active_tool_effect(state: dict[str, Any]) -> tuple[str, str] | None:
        try:
            task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
            index = int(state.get("step_index", 0) or 0)
            step = task_plan.steps[index]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None
        if not step.tool:
            return None
        return (
            f"tool_{step.tool}",
            f"step:{index + 1}:{step.tool}",
        )

    def _fail_active_tool_effect_best_effort(
        self,
        state: dict[str, Any],
        *,
        config: dict[str, Any],
        execution_token: str,
        error: str,
    ) -> None:
        persistence = self.deps.persistence
        if persistence is None:
            return
        get_effect = getattr(persistence, "get_effect", None)
        if not callable(get_effect):
            return
        latest_state = dict(state)
        try:
            snapshot = self._graph.get_state(config)
            latest_state.update(dict(getattr(snapshot, "values", None) or {}))
            target = self._active_tool_effect(latest_state)
            if target is None:
                return
            node, effect = target
            record = get_effect(
                latest_state["run_id"],
                node,
                effect,
                user_id=latest_state["user_id"],
            )
            if record is None or getattr(record, "status", "") != "claimed":
                return
            self._fail_effect(
                latest_state,
                node=node,
                effect=effect,
                error=error,
                execution_token=execution_token,
            )
        except Exception:
            logger.exception(
                "Could not fail active tool effect run_id=%s",
                state.get("run_id", ""),
            )

    def _fast_forward_completed_tool_effect(
        self,
        config: dict[str, Any],
    ) -> bool:
        """Advance a parent checkpoint when the durable effect is already done.

        With async durability, a worker can complete business writes and the
        effect receipt before the parent graph records the child-node update.
        Re-executing document_commit would duplicate those writes, so recovery
        applies the receipt as the child node's durable output instead.
        """

        persistence = self.deps.persistence
        if persistence is None:
            return False
        get_effect = getattr(persistence, "get_effect", None)
        if not callable(get_effect):
            return False
        snapshot = self._graph.get_state(config)
        state = dict(getattr(snapshot, "values", None) or {})
        target = self._active_tool_effect(state)
        if target is None:
            return False
        node, effect = target
        if node not in tuple(getattr(snapshot, "next", ()) or ()):
            return False
        record = get_effect(
            state["run_id"],
            node,
            effect,
            user_id=state["user_id"],
        )
        if record is None or getattr(record, "status", "") != "completed":
            return False
        result = getattr(record, "result", None)
        if not isinstance(result, dict):
            raise RuntimeError(f"Completed effect {node}:{effect} has no result")
        replay_update = validate_graph_update(
            {
                "current_step_done": dict(result.get("step_done") or {}),
                "current_step_failed": bool(result.get("failed")),
                "status": "tool_replayed",
            },
            node=f"chat.{node}.replay",
        )
        self._graph.update_state(
            config,
            replay_update,
            as_node=node,
        )
        logger.info(
            "Fast-forwarded completed tool effect run_id=%s node=%s effect=%s",
            state.get("run_id", ""),
            node,
            effect,
        )
        return True

    def _start_persisted_run(
        self,
        state: ChatGraphState,
        *,
        execution_token: str,
        allow_implicit_session_rebind: bool = False,
    ) -> Any:
        persistence = self.deps.persistence
        if persistence is None:
            return None
        self._set_telemetry(run_id=state["run_id"], current_node="prepare")
        try:
            record = persistence.start_run(
                run_id=state["run_id"],
                thread_id=state["run_id"],
                graph_name="chat",
                user_id=state["user_id"],
                session_id=state["session_id"],
                request_id=state["request_id"],
                idempotency_key=state.get("idempotency_key") or None,
                workflow_version=self.deps.workflow_version,
                current_node="prepare",
                execution_token=execution_token,
                durability=self._durability(),
                metadata={
                    "configured_runtime": self._runtime_mode,
                    "request_digest": self._request_digest(state),
                },
            )
        except Exception as exc:
            conflict_type = str(getattr(exc, "conflict_type", "") or "")
            if (
                allow_implicit_session_rebind
                and conflict_type == "request_digest_mismatch"
            ):
                key = str(
                    state.get("idempotency_key")
                    or state.get("request_id")
                    or ""
                )
                replay = self._lookup_idempotent_run(state["user_id"], key)
                if replay is not None:
                    # Another worker can win between preflight and INSERT after
                    # both workers resolved different implicit sessions. Bind
                    # this retry to the winner and validate every other payload
                    # field against its stored digest before replaying it.
                    self._assert_request_identity(
                        replay,
                        state,
                        allow_implicit_session_rebind=True,
                    )
                    return replay
            if (
                type(exc).__name__ == "GraphRunIdentityConflictError"
                or conflict_type
            ):
                raise ChatRunConflictError(str(exc)) from exc
            raise
        self._assert_request_identity(
            record,
            state,
            allow_implicit_session_rebind=allow_implicit_session_rebind,
        )
        logger.info(
            "LangGraph run dispatch run_id=%s workflow_version=%s "
            "current_node=%s checkpointer=%s",
            record.run_id,
            record.workflow_version,
            record.current_node,
            getattr(persistence.config, "backend", "unknown"),
        )
        return record

    def _replay_idempotent_run(self, record: Any) -> Iterable[str]:
        """Return the prior result/status without executing side effects again."""
        config = {"configurable": {"thread_id": record.thread_id}}
        if record.status == "succeeded":
            snapshot = self._graph.get_state(config)
            values = dict(getattr(snapshot, "values", None) or {})
            final_event = values.get("final_event")
            if isinstance(final_event, dict) and final_event.get("type") == "done":
                yield sse(final_event)
                return
        messages = {
            "running": "相同请求正在处理中，请使用 run_id 查询进度",
            "interrupted": "相同请求正在等待确认，请使用原 run_id 恢复",
            "failed": "相同请求此前失败，请使用原 run_id 重试",
            "cancelled": "相同请求已取消",
        }
        yield sse({
            "type": "error",
            "message": messages.get(record.status, "相同请求已经存在"),
            "run_id": record.run_id,
            "status": record.status,
        })

    def _mark_current_node(
        self,
        state: ChatGraphState,
        node: str,
        *,
        execution_token: str = "",
        lease: Any = None,
    ) -> None:
        persistence = self.deps.persistence
        if persistence is None:
            return
        if lease is not None:
            lease.set_current_node(node)
        self._assert_run_active(state)
        if lease is not None:
            try:
                lease.assert_owned()
            except ChatRunConflictError:
                # A background heartbeat observes the same database fence as
                # the foreground node. If it lost ownership because deletion
                # won the race, preserve cancellation semantics and cleanup.
                self._assert_run_active(state)
                raise
        heartbeat = getattr(persistence, "heartbeat_run", None)
        if callable(heartbeat) and execution_token:
            record = heartbeat(
                state["run_id"],
                user_id=state["user_id"],
                execution_token=execution_token,
                current_node=node,
            )
        else:
            # Compatibility for rollback/test adapters. Persistent graph
            # wiring always supplies heartbeat_run plus an execution token.
            record = persistence.update_run(
                state["run_id"],
                status="running",
                current_node=node,
            )
        if record is None:
            # The barrier may have committed after the pre-heartbeat check;
            # distinguish intentional cancellation from an ownership conflict.
            self._assert_run_active(state)
            owned = persistence.get_owned_run(
                state["run_id"],
                user_id=state["user_id"],
            )
            if owned is None:
                raise ChatRunCancelledError("run ledger row was deleted")
            raise ChatRunConflictError("graph execution lease was lost")
        self._set_telemetry(run_id=state["run_id"], current_node=node)

    def _assert_run_active(self, state: ChatGraphState) -> None:
        persistence = self.deps.persistence
        if persistence is None:
            return
        session_checker = getattr(
            persistence, "is_session_deletion_requested", None
        )
        if callable(session_checker) and session_checker(
            state["session_id"],
            user_id=state["user_id"],
        ):
            raise ChatRunCancelledError("session deletion was requested")
        checker = getattr(persistence, "is_thread_deletion_requested", None)
        if callable(checker) and checker(
            state["run_id"],
            user_id=state["user_id"],
        ):
            raise ChatRunCancelledError("session deletion was requested")

    def _cancellation_event_after_failure(
        self,
        state: ChatGraphState,
    ) -> Optional[str]:
        """Let a committed deletion fence win a concurrent node failure."""

        try:
            self._assert_run_active(state)
        except ChatRunCancelledError:
            self._cleanup_cancelled_run(state)
            return sse(error_event("会话已删除，当前运行已取消"))
        except Exception:
            # Preserve the original node failure when the best-effort barrier
            # recheck itself is temporarily unavailable.
            logger.exception(
                "Could not recheck deletion barrier after graph failure run_id=%s",
                state.get("run_id", ""),
            )
        return None

    def _cleanup_cancelled_run(self, state: ChatGraphState) -> None:
        persistence = self.deps.persistence
        if persistence is not None:
            try:
                persistence.request_session_deletion(
                    state["session_id"],
                    user_id=state["user_id"],
                    reason="active_run_cancelled",
                )
            except Exception:
                logger.exception(
                    "Could not reapply graph deletion barrier run_id=%s",
                    state.get("run_id", ""),
                )
            try:
                # The first deletion may already have removed the ledger row,
                # so session lookup can no longer find a checkpoint recreated
                # by an in-flight node. Delete the known thread id directly
                # after this worker has stopped progressing.
                persistence.delete_thread(state["run_id"])
            except Exception:
                logger.exception(
                    "Could not delete cancelled graph thread run_id=%s",
                    state.get("run_id", ""),
                )
        try:
            self.deps.memory.delete_session(state["session_id"])
        except Exception:
            logger.exception(
                "Could not remove messages recreated after session deletion session=%s",
                state.get("session_id", ""),
            )
        if self.deps.artifact_store is not None:
            try:
                self.deps.artifact_store.delete_run(state["run_id"])
            except OSError:
                logger.exception(
                    "Could not remove cancelled run artifacts run_id=%s",
                    state.get("run_id", ""),
                )

    def _finish_persisted_run(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        status: str,
        error: str = "",
        checkpoint_id: str = "",
        interrupt_id: str = "",
    ) -> None:
        persistence = self.deps.persistence
        if persistence is None:
            return
        current_node = (
            "__end__"
            if status == "succeeded"
            else ("interrupt" if status == "interrupted" else "error")
        )
        finish = getattr(persistence, "finish_run", None)
        if callable(finish) and execution_token:
            record = finish(
                run_id,
                user_id=user_id,
                execution_token=execution_token,
                status=status,
                error=error,
                current_node=current_node,
                checkpoint_id=checkpoint_id,
                interrupt_id=interrupt_id,
            )
        else:
            record = persistence.update_run(
                run_id,
                status=status,
                error=error,
                current_node=current_node,
                checkpoint_id=checkpoint_id,
                interrupt_id=interrupt_id,
            )
        if record is None:
            owned = persistence.get_owned_run(run_id, user_id=user_id)
            if owned is None:
                raise ChatRunCancelledError("run ledger row was deleted")
            session_checker = getattr(
                persistence, "is_session_deletion_requested", None
            )
            if callable(session_checker) and session_checker(
                owned.session_id,
                user_id=user_id,
            ):
                raise ChatRunCancelledError("session deletion was requested")
            thread_checker = getattr(
                persistence, "is_thread_deletion_requested", None
            )
            if callable(thread_checker) and thread_checker(
                run_id,
                user_id=user_id,
            ):
                raise ChatRunCancelledError("session deletion was requested")
            raise ChatRunConflictError("graph execution lease was lost")
        self._set_telemetry(
            run_id=run_id,
            current_node=current_node,
            checkpoint_id=checkpoint_id,
        )
        logger.info(
            "LangGraph run finish run_id=%s status=%s current_node=%s "
            "checkpoint_id=%s",
            run_id,
            status,
            current_node,
            checkpoint_id,
        )

    def _fail_persisted_run_best_effort(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        error: str,
    ) -> None:
        try:
            self._finish_persisted_run(
                run_id,
                user_id=user_id,
                execution_token=execution_token,
                status="failed",
                error=error,
            )
        except Exception:
            logger.exception("Could not persist failed graph run run_id=%s", run_id)

    def _owned_run(self, run_id: str, *, user_id: str) -> Any:
        persistence = self.deps.persistence
        if persistence is None:
            raise ChatRunNotFoundError("运行不存在或无权限")
        record = persistence.get_owned_run(str(run_id), user_id=str(user_id))
        if record is None or record.graph_name != "chat":
            raise ChatRunNotFoundError("运行不存在或无权限")
        return record

    def _assert_workflow_compatible(self, record: Any) -> None:
        recorded = str(getattr(record, "workflow_version", "") or "")
        current = str(self.deps.workflow_version or "")
        if recorded == current:
            return
        if current == "chat-v3" and recorded == self.PREVIOUS_WORKFLOW_VERSION:
            return
        raise ChatRunConflictError(
            f"运行使用工作流 {recorded or 'unknown'}，当前版本 {current or 'unknown'} "
            "不支持安全恢复"
        )

    def _assert_recovery_window(self, record: Any) -> None:
        raw_updated_at = str(getattr(record, "updated_at", "") or "")
        try:
            updated_at = datetime.fromisoformat(raw_updated_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ChatRunConflictError("运行时间信息无效，不能安全恢复") from exc
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.RECOVERY_WINDOW_DAYS
        )
        if updated_at.astimezone(timezone.utc) < cutoff:
            raise ChatRunConflictError("该运行已超过 7 天恢复窗口")

    def _run_lease_is_stale(self, record: Any) -> bool:
        raw_updated_at = str(getattr(record, "updated_at", "") or "")
        try:
            updated_at = datetime.fromisoformat(raw_updated_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        stale_before = datetime.now(timezone.utc) - timedelta(
            seconds=self.RECOVERY_LEASE_STALE_SECONDS
        )
        return updated_at.astimezone(timezone.utc) <= stale_before

    @staticmethod
    def _snapshot_interrupt_ids(snapshot: Any) -> list[str]:
        interrupt_ids: list[str] = []
        for task in getattr(snapshot, "tasks", ()) or ():
            for item in getattr(task, "interrupts", ()) or ():
                interrupt_id = str(getattr(item, "id", "") or "")
                if interrupt_id and interrupt_id not in interrupt_ids:
                    interrupt_ids.append(interrupt_id)
        return interrupt_ids

    @classmethod
    def _snapshot_tokens(cls, snapshot: Any) -> tuple[str, list[str]]:
        configurable = (getattr(snapshot, "config", None) or {}).get(
            "configurable", {}
        )
        return (
            str(configurable.get("checkpoint_id") or ""),
            cls._snapshot_interrupt_ids(snapshot),
        )

    def _checkpoint_tokens(
        self,
        config: dict[str, Any],
    ) -> tuple[str, list[str]]:
        snapshot = self._graph.get_state(config)
        return self._snapshot_tokens(snapshot)

    def _graph_has_interrupt(self, config: dict[str, Any]) -> bool:
        try:
            _, interrupt_ids = self._checkpoint_tokens(config)
        except Exception:
            logger.exception("Could not inspect graph interrupts")
            return False
        return bool(interrupt_ids)

    def _enrich_public_stream(
        self,
        stream: Iterable[str],
        state: ChatGraphState,
    ) -> Iterable[str]:
        """Add compatible run metadata without changing event ordering."""
        iterator = iter(stream)
        try:
            for chunk in iterator:
                if not isinstance(chunk, str):
                    raise TypeError("Chat stream chunks must be strings")
                events = self._parse_chunk_events(chunk)
                if not events:
                    yield chunk
                    continue
                changed = False
                for event in events:
                    event_type = event.get("type")
                    if event_type in {"session", "run_done", "tool_confirm_required", "done"}:
                        if not event.get("run_id"):
                            event["run_id"] = state["run_id"]
                            changed = True
                    if event_type == "tool_confirm_required":
                        data = event.get("data")
                        if isinstance(data, dict):
                            if not data.get("run_id"):
                                data["run_id"] = state["run_id"]
                                changed = True
                            if "interrupt_id" not in data:
                                data["interrupt_id"] = ""
                                changed = True
                    elif event_type == "done":
                        defaults = {
                            # Public SSE remains byte-equivalent across graph and
                            # rollback runners; the effective engine is exposed by
                            # health/log metadata instead of changing the payload.
                            "runtime": self.deps.workflow_version,
                            "quality_status": "not_applicable",
                            "revisions_applied": 0,
                        }
                        if "revision_rounds" in event and "revisions_applied" not in event:
                            defaults["revisions_applied"] = int(event.get("revision_rounds") or 0)
                        for key, value in defaults.items():
                            if key not in event:
                                event[key] = value
                                changed = True
                yield "".join(sse(event) for event in events) if changed else chunk
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def _set_telemetry(
        self,
        *,
        run_id: str,
        current_node: str,
        checkpoint_id: str = "",
    ) -> None:
        with self._telemetry_lock:
            self._telemetry.update({
                "run_id": str(run_id or ""),
                "current_node": str(current_node or ""),
                "checkpoint_id": str(checkpoint_id or ""),
            })

    def health(self) -> dict[str, Any]:
        persistence = self.deps.persistence
        backend = (
            persistence.config.backend
            if persistence is not None
            else ("memory" if self._use_langgraph else "none")
        )
        with self._telemetry_lock:
            telemetry = dict(self._telemetry)
        return {
            "ok": True,
            "configured_runtime": os.getenv("CHAT_RUNTIME", "graph"),
            "effective_runtime": self._runtime_mode,
            "checkpointer": backend,
            "workflow_version": self.deps.workflow_version,
            "durability": self._durability() if self._use_langgraph else "none",
            **telemetry,
        }

    def _route_request(self, prepared: PreparedChatRequest) -> RouteResult:
        conversation_context = self._conversation_context_for_request(prepared)
        has_last_document = bool(self.deps.memory.get_context(prepared.session_id, "last_document", "") or "")
        return IntentRouter(
            self.deps.reimbursement_detector,
            intent_classifier=self.deps.intent_classifier,
        ).route(
            message=prepared.message,
            display_message=prepared.display_message,
            mode=prepared.mode,
            attachments=prepared.attachments,
            conversation_context=conversation_context,
            has_last_document=has_last_document,
        )

    def _plan_request(self, prepared: PreparedChatRequest) -> TaskPlan:
        conversation_context = self._conversation_context_for_request(prepared)
        has_last_document = bool(self.deps.memory.get_context(prepared.session_id, "last_document", "") or "")
        plan = self.deps.task_planner.plan(
            message=prepared.message,
            display_message=prepared.display_message,
            mode=prepared.mode,
            attachments=prepared.attachments,
            conversation_context=conversation_context,
            has_last_document=has_last_document,
            user_info=prepared.user_info,
        )
        if isinstance(plan, TaskPlan):
            payload = plan.to_dict()
        elif isinstance(plan, dict):
            payload = plan
        elif hasattr(plan, "to_dict"):
            payload = plan.to_dict()
        else:
            raise TypeError("TaskPlanner.plan() must return TaskPlan or a serializable mapping")
        try:
            serialized = json.loads(json.dumps(payload, ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise TypeError("TaskPlanner returned a non-serializable plan") from exc
        serialized = self._sanitize_native_document_update(
            {
                "run_id": prepared.run_id,
                "user_id": prepared.user_id,
                "attachments": prepared.attachments,
            },
            {"plan": serialized},
        )["plan"]
        return TaskPlan.from_dict(serialized)

    def _plan_tools_node(
        self,
        state: ChatGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> ChatGraphState:
        self._mark_current_node(
            state,
            "plan_tools",
            execution_token=runtime.context.execution_token,
            lease=runtime.context.lease,
        )
        claimed, replay = self._claim_effect(
            state,
            node="plan_tools",
            effect="task_plan",
            execution_token=runtime.context.execution_token,
            allow_reclaim=runtime.context.recovery,
        )
        if not claimed:
            task_plan = (replay or {}).get("task_plan")
            if not isinstance(task_plan, dict):
                raise RuntimeError("Persisted task plan effect has no reusable result")
            return {"task_plan": task_plan, "status": "planned"}
        prepared = self._prepared_from_state(state, runtime.context.user_info)
        try:
            plan = self._plan_request(prepared)
            payload = plan.to_dict()
            self._complete_effect(
                state,
                node="plan_tools",
                effect="task_plan",
                result={"task_plan": payload},
                execution_token=runtime.context.execution_token,
            )
        except Exception as exc:
            self._fail_effect(
                state,
                node="plan_tools",
                effect="task_plan",
                error=str(exc),
                execution_token=runtime.context.execution_token,
            )
            raise
        return {"task_plan": payload, "status": "planned"}

    def _select_step_node(
        self,
        state: ChatGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> ChatGraphState:
        self._mark_current_node(
            state,
            "select_step",
            execution_token=runtime.context.execution_token,
            lease=runtime.context.lease,
        )
        orchestrator = self.deps.tool_orchestrator
        if not hasattr(orchestrator, "registry"):
            return {"compat_orchestrator": True, "status": "executing"}

        task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
        index = int(state.get("step_index", 0) or 0)
        update: ChatGraphState = {"status": "selecting_step"}
        if not state.get("plan_events_emitted"):
            for event in (
                {"type": "start"},
                {"type": "session", "session_id": state["session_id"]},
                {"type": "thinking_start", "message": "正在规划工具调用"},
                {"type": "tool_plan", "data": task_plan.to_dict()},
                {
                    "type": "think",
                    "agent": "TaskPlanner",
                    "emoji": "",
                    "message": orchestrator._plan_summary(task_plan),
                },
            ):
                runtime.stream_writer(sse(event))
            update["plan_events_emitted"] = True

        if index < len(task_plan.steps):
            step = task_plan.steps[index]
            tool = orchestrator.registry.get(step.tool)
            runtime.stream_writer(sse({
                "type": "tool_call",
                "data": {
                    "index": index + 1,
                    "tool": tool.name,
                    "reason": step.reason,
                    "risk_level": step.risk_level or tool.risk_level,
                    "requires_confirmation": step.requires_confirmation,
                },
            }))
            runtime.stream_writer(sse({
                "type": "think",
                "agent": "ToolOrchestrator",
                "emoji": "",
                "message": f"调用工具: {tool.name}",
            }))
        return update

    def _route_selected_step(self, state: ChatGraphState) -> str:
        if state.get("compat_orchestrator"):
            return "compat"
        task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
        index = int(state.get("step_index", 0) or 0)
        if index >= len(task_plan.steps):
            return "finalize"
        tool_name = task_plan.steps[index].tool
        supported = {
            TOOL_KNOWLEDGE_QA,
            TOOL_DRAFT_DOCUMENT,
            TOOL_FORMAT_DOCUMENT,
            TOOL_PREPARE_FORM_EXPORT,
            TOOL_PREPARE_SPREADSHEET_TRANSFORM,
            TOOL_CLARIFY,
            TOOL_IDENTITY_HELP,
        }
        return tool_name if tool_name in supported else "unsupported"

    def _tool_node(self, expected_tool: str) -> Callable[..., ChatGraphState]:
        def execute(
            state: ChatGraphState,
            runtime: Runtime[ChatRunContext],
        ) -> ChatGraphState:
            return self._execute_single_tool_node(
                state,
                runtime,
                expected_tool=expected_tool,
            )

        execute.__name__ = f"execute_{expected_tool}"
        return execute

    def _execute_single_tool_node(
        self,
        state: ChatGraphState,
        runtime: Runtime[ChatRunContext],
        *,
        expected_tool: str,
    ) -> ChatGraphState:
        node_name = f"tool_{expected_tool}"
        self._mark_current_node(
            state,
            node_name,
            execution_token=runtime.context.execution_token,
            lease=runtime.context.lease,
        )
        task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
        index = int(state.get("step_index", 0) or 0)
        if index >= len(task_plan.steps):
            raise RuntimeError("Selected tool step is out of range")
        step = task_plan.steps[index]
        if step.tool != expected_tool:
            raise RuntimeError(
                f"Selected tool mismatch: expected {expected_tool}, got {step.tool}"
            )

        effect_name = f"step:{index + 1}:{expected_tool}"
        claimed, replay = self._claim_effect(
            state,
            node=node_name,
            effect=effect_name,
            execution_token=runtime.context.execution_token,
            allow_reclaim=runtime.context.recovery,
        )
        if not claimed:
            if not isinstance(replay, dict):
                raise RuntimeError("Persisted tool effect has no reusable result")
            return {
                "current_step_done": dict(replay.get("step_done") or {}),
                "current_step_failed": bool(replay.get("failed")),
                "status": "tool_replayed",
            }

        orchestrator = self.deps.tool_orchestrator
        tool = orchestrator.registry.get(expected_tool)
        working_state = dict(state)
        working_state["message"] = state.get("working_message") or state.get("message", "")
        prepared = self._prepared_from_state(
            working_state,
            runtime.context.user_info,
        )
        route = orchestrator._route_for_step(step, task_plan.route)
        is_final_step = index == len(task_plan.steps) - 1
        step_done: dict[str, Any] | None = None
        failed = False
        pending_success_chunks: list[str] = []
        try:
            for event in orchestrator._stream_tool_events(tool, prepared, route):
                event_type = event.get("type")
                if event_type in {"start", "session", "route", "run_done"}:
                    continue
                if event_type == "done":
                    step_done = event
                    continue
                if event_type == "error":
                    failed = True
                    step_done = event
                    break
                if not is_final_step and event_type in {
                    "answer_start",
                    "answer_delta",
                    "answer_done",
                    "content",
                }:
                    continue
                pending_success_chunks.append(sse(event))

            if step_done is None:
                step_done = orchestrator._missing_tool_done(prepared, step, route)
            if failed:
                # Keep the parent checkpoint parked before this tool and leave
                # a reclaimable failed effect.  Completing a failed tool effect
                # would make recovery replay the error forever and prevent the
                # document graph from retrying its failed Writer/Reviewer node.
                runtime.stream_writer(sse(step_done))
                raise RuntimeError(
                    str(step_done.get("message") or "工具执行失败")
                )
            lease = runtime.context.lease
            if lease is not None:
                lease.assert_owned()
            self._complete_effect(
                state,
                node=node_name,
                effect=effect_name,
                result={"step_done": step_done, "failed": failed},
                execution_token=runtime.context.execution_token,
            )
            for chunk in pending_success_chunks:
                runtime.stream_writer(chunk)
        except Exception as exc:
            self._fail_effect(
                state,
                node=node_name,
                effect=effect_name,
                error=str(exc),
                execution_token=runtime.context.execution_token,
            )
            raise
        return {
            "current_step_done": step_done,
            "current_step_failed": failed,
            "status": "tool_failed" if failed else "tool_completed",
        }

    def _collect_result_node(
        self,
        state: ChatGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> ChatGraphState:
        self._mark_current_node(
            state,
            "collect_result",
            execution_token=runtime.context.execution_token,
            lease=runtime.context.lease,
        )
        if state.get("current_step_failed"):
            step_done = dict(state.get("current_step_done") or {})
            return {
                "status": "failed",
                "error": {
                    "type": "tool_error",
                    "message": str(step_done.get("message") or "工具执行失败"),
                },
            }

        orchestrator = self.deps.tool_orchestrator
        task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
        index = int(state.get("step_index", 0) or 0)
        step = task_plan.steps[index]
        step_done = dict(state.get("current_step_done") or {})
        prepared_state = dict(state)
        prepared_state["message"] = state.get("working_message") or state.get("message", "")
        prepared = self._prepared_from_state(
            prepared_state,
            runtime.context.user_info,
        )
        route = orchestrator._route_for_step(step, task_plan.route)
        if not step_done:
            step_done = orchestrator._missing_tool_done(prepared, step, route)

        is_final_step = index == len(task_plan.steps) - 1
        working_message = state.get("working_message") or state.get("message", "")
        if not is_final_step:
            working_message = orchestrator._message_with_tool_result(
                working_message,
                step,
                step_done,
            )
        tool_result_event = orchestrator._tool_result_event(step, step_done, index + 1)
        runtime.stream_writer(sse(tool_result_event))
        if step.requires_confirmation or step_done.get("actions"):
            runtime.stream_writer(sse({
                "type": "tool_confirm_required",
                "data": {
                    "tool": step.tool,
                    "actions": step_done.get("actions", []),
                    "message": step_done.get("answer", ""),
                },
            }))

        tool_results = list(state.get("tool_results") or [])
        tool_results.append(tool_result_event["data"])
        return {
            "step_index": index + 1,
            "working_message": working_message,
            "tool_results": tool_results,
            "current_step_done": {},
            "current_step_failed": False,
            "final_event": step_done if is_final_step else {},
            "status": "step_collected",
        }

    @staticmethod
    def _route_after_collect(state: ChatGraphState) -> str:
        if state.get("status") == "failed" or state.get("error"):
            return "error"
        task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
        if int(state.get("step_index", 0) or 0) >= len(task_plan.steps):
            return "finalize"
        return "select_step"

    def _finalize_node(
        self,
        state: ChatGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> ChatGraphState:
        self._mark_current_node(
            state,
            "finalize",
            execution_token=runtime.context.execution_token,
            lease=runtime.context.lease,
        )
        task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
        final_done = dict(state.get("final_event") or {})
        if not final_done:
            prepared = self._prepared_from_state(state, runtime.context.user_info)
            final_done = self.deps.tool_orchestrator._empty_done(prepared, task_plan)
        final_done.setdefault("plan", {})
        if isinstance(final_done["plan"], dict):
            final_done["plan"] = {
                **final_done["plan"],
                "task_planner": task_plan.to_dict(),
                "tool_results": list(state.get("tool_results") or []),
            }
        runtime.stream_writer(sse({
            "type": "run_done",
            "session_id": state["session_id"],
            "intent": final_done.get("intent", ""),
        }))
        runtime.stream_writer(sse(final_done))
        return {
            "status": "completed",
            "final_event": final_done,
        }

    def _unsupported_tool_node(
        self,
        state: ChatGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> ChatGraphState:
        self._mark_current_node(
            state,
            "unsupported_tool",
            execution_token=runtime.context.execution_token,
            lease=runtime.context.lease,
        )
        task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
        index = int(state.get("step_index", 0) or 0)
        tool_name = (
            task_plan.steps[index].tool
            if index < len(task_plan.steps)
            else "unknown"
        )
        message = f"不支持的工具步骤: {tool_name}"
        runtime.stream_writer(sse(error_event(message)))
        return {
            "status": "failed",
            "error": {"type": "unsupported_tool", "message": message},
        }

    @staticmethod
    def _error_terminal_node(state: ChatGraphState) -> ChatGraphState:
        return {"status": "failed"}

    def _conversation_context_for_request(self, prepared: PreparedChatRequest) -> str:
        """Use the current request, not the preceding turn, to retrieve memories."""
        try:
            return self.deps.memory.get_context_for_prompt(
                prepared.session_id,
                max_messages=5,
                memory_query=prepared.display_message,
            )
        except TypeError:
            # Third-party/test memory adapters can keep the previous method
            # signature; they simply do not participate in long-term recall.
            return self.deps.memory.get_context_for_prompt(prepared.session_id, max_messages=5)

    def _execute_tools_node(
        self,
        state: ChatGraphState,
        runtime: Runtime[ChatRunContext],
    ) -> ChatGraphState:
        self._mark_current_node(
            state,
            "execute_tools",
            execution_token=runtime.context.execution_token,
            lease=runtime.context.lease,
        )
        prepared = self._prepared_from_state(state, runtime.context.user_info)
        task_plan = TaskPlan.from_dict(state.get("task_plan") or {})
        final_event: dict[str, Any] = {}
        pending_done_chunk: Optional[str] = None
        failed = False

        for chunk in self.deps.tool_orchestrator.stream(prepared, task_plan):
            if not isinstance(chunk, str):
                raise TypeError("ToolOrchestrator.stream() must yield string SSE chunks")
            events = self._parse_chunk_events(chunk)
            contains_done = False
            for event in events:
                event_type = event.get("type")
                if event_type == "done":
                    final_event = event
                    contains_done = True
                elif event_type == "error":
                    final_event = event
                    failed = True
            if contains_done:
                pending_done_chunk = chunk
            else:
                runtime.stream_writer(chunk)

        # Keep the public success contract intact: done is emitted only after
        # every tool chunk completed successfully, and remains the final event.
        if pending_done_chunk is not None:
            runtime.stream_writer(pending_done_chunk)

        update: ChatGraphState = {
            "status": "failed" if failed else "completed",
            "final_event": final_event,
        }
        if failed:
            update["error"] = {
                "type": "tool_error",
                "message": str(final_event.get("message") or "请求处理失败，请稍后重试"),
            }
        return update

    @staticmethod
    def _parse_chunk_events(chunk: str) -> list[dict[str, Any]]:
        try:
            return parse_sse_events([chunk])
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    def _dispatch_request(
        self,
        prepared: PreparedChatRequest,
        route: RouteResult,
    ) -> Iterable[str]:
        intent = route.intent
        if intent not in self.GRAPH_INTENTS:
            intent = INTENT_KNOWLEDGE_QA

        if intent == INTENT_KNOWLEDGE_QA:
            stream = self.deps.rag_qa_stream(
                prepared.message,
                prepared.session_id,
                prepared.user_id,
                prepared.user_info,
                prepared.display_message,
                prepared.user_metadata,
                route,
            )
        elif intent == INTENT_DOC_FORMATTING:
            stream = self.deps.document_format_stream(
                prepared.message,
                prepared.session_id,
                prepared.user_id,
                prepared.user_info,
                prepared.display_message,
                prepared.user_metadata,
                route,
            )
        elif intent == INTENT_DOC_DRAFTING:
            stream = self.deps.document_draft_stream(
                prepared.message,
                prepared.session_id,
                prepared.user_id,
                prepared.user_info,
                prepared.display_message,
                prepared.user_metadata,
                route,
            )
        else:
            stream = self.deps.lightweight_stream(
                prepared.message,
                prepared.session_id,
                prepared.user_id,
                prepared.user_info,
                prepared.display_message,
                prepared.user_metadata,
                route,
            )
        return stream
