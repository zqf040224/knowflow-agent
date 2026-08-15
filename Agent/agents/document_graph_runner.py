"""Reusable LangGraph runner for document generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar
from uuid import uuid4

from agents.document_graph_state import DocumentGraphRuntime, DocumentGraphState
from agents.document_graph_steps import DocumentGraphSteps
from graph_state_validation import validate_graph_update

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.graph import END, StateGraph
    from namespaced_checkpointer import NamespacedCheckpointSaver

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = ""
except Exception as exc:
    END = "__end__"
    StateGraph = None
    InMemorySaver = None
    JsonPlusSerializer = None
    NamespacedCheckpointSaver = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(exc)


@dataclass
class DocumentGraphRunResult:
    ctx: Any
    document_content: str
    quality_status: str
    revisions_applied: int
    run_id: str


class DocumentGraphExecutionError(RuntimeError):
    """Raised when a fail-closed node terminates, retaining retryable state."""

    def __init__(self, message: str, *, state: DocumentGraphState | None = None):
        super().__init__(message)
        self.state = dict(state or {})
        self.document_content = self.state.get("document_content", "")
        self.run_id = self.state.get("run_id", "")


class DocumentGraphRunner:
    _compiled_graph: ClassVar[Any] = None
    _compiled_graphs: ClassVar[dict[int, Any]] = {}
    _default_checkpointer: ClassVar[Any] = None
    _compile_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        orchestrator: Any,
        *,
        checkpointer: Any = None,
        durability: str = "async",
    ):
        self.orchestrator = orchestrator
        # Empty saver implementations can be false-y. Treat only ``None`` as
        # "use the shared default" so tests and callers retain true isolation.
        self.checkpointer = (
            checkpointer
            if checkpointer is not None
            else self._shared_default_checkpointer()
        )
        self.durability = durability
        self.graph = self._shared_graph(self.checkpointer)

    @classmethod
    def _shared_default_checkpointer(cls):
        if cls._default_checkpointer is not None:
            return cls._default_checkpointer
        with cls._compile_lock:
            if cls._default_checkpointer is None:
                if InMemorySaver is None or JsonPlusSerializer is None:
                    raise RuntimeError("LangGraph checkpointer dependencies are unavailable")
                cls._default_checkpointer = InMemorySaver(
                    serde=JsonPlusSerializer(
                        pickle_fallback=False,
                        allowed_json_modules=None,
                        allowed_msgpack_modules=None,
                    )
                )
        return cls._default_checkpointer

    @classmethod
    def _shared_graph(cls, checkpointer: Any):
        key = id(checkpointer)
        if key in cls._compiled_graphs:
            return cls._compiled_graphs[key]
        with cls._compile_lock:
            if key not in cls._compiled_graphs:
                cls._compiled_graphs[key] = cls._build_graph(checkpointer)
                if cls._compiled_graph is None:
                    cls._compiled_graph = cls._compiled_graphs[key]
        return cls._compiled_graphs[key]

    @classmethod
    def _workflow(cls):
        if StateGraph is None:
            raise RuntimeError(f"LangGraph is not available: {LANGGRAPH_IMPORT_ERROR}")

        workflow = StateGraph(DocumentGraphState, context_schema=DocumentGraphRuntime)
        steps = DocumentGraphSteps()

        workflow.add_node("context_plan", steps.context_plan)
        workflow.add_node("retrieval", steps.retrieval)
        workflow.add_node("write", steps.write)
        workflow.add_node("review", steps.review)
        workflow.add_node("reflection", steps.reflection)
        workflow.add_node("decide", steps.decide)
        workflow.add_node("finalize", steps.finalize)
        workflow.add_node("error_terminal", steps.error_terminal)

        workflow.set_entry_point("context_plan")
        workflow.add_conditional_edges(
            "context_plan",
            steps.route_after_context_plan,
            {"retrieval": "retrieval", "error": "error_terminal"},
        )
        workflow.add_conditional_edges(
            "retrieval",
            steps.route_after_retrieval,
            {"write": "write", "error": "error_terminal"},
        )
        workflow.add_conditional_edges("write", steps.route_after_write, {
            "review": "review",
            "error": "error_terminal",
        })
        workflow.add_conditional_edges("review", steps.route_after_review, {
            "reflection": "reflection",
            "decide": "decide",
            "error": "error_terminal",
        })
        workflow.add_edge("reflection", "decide")
        workflow.add_conditional_edges("decide", steps.route_after_decide, {
            "write": "write",
            "finalize": "finalize",
        })
        workflow.add_edge("finalize", END)
        workflow.add_edge("error_terminal", END)
        return workflow

    @classmethod
    def _build_graph(cls, checkpointer: Any):
        workflow = cls._workflow()
        child_checkpointer = NamespacedCheckpointSaver(checkpointer, "document")
        return workflow.compile(
            checkpointer=child_checkpointer,
            name="document_workflow",
        )

    @classmethod
    def build_native_subgraph(cls):
        """Compile the document topology for parent-checkpointer inheritance."""
        return cls._workflow().compile(name="document_workflow")

    @staticmethod
    def _initial_state(prepared_run: Any, run_id: str) -> DocumentGraphState:
        explicit_display_message = getattr(
            prepared_run,
            "display_message",
            None,
        )
        persisted_user_message = getattr(
            prepared_run,
            "persisted_user_message",
            None,
        )
        safe_display_message = (
            explicit_display_message
            if explicit_display_message is not None
            else persisted_user_message
            if persisted_user_message is not None
            else getattr(prepared_run, "user_request", "")
        )
        safe_request = str(safe_display_message or "")
        return validate_graph_update({
            # The hydrated request (including attachment bodies) belongs only in
            # DocumentGraphRuntime.  These checkpointed identity fields retain
            # the user-visible request, including an explicitly empty file-only
            # message.
            "input_user_request": safe_request,
            "run_id": run_id,
            "session_id": str(getattr(prepared_run, "session_id", "") or ""),
            "user_id": str(getattr(prepared_run, "user_id", "") or ""),
            "display_message": str(safe_display_message or ""),
            "parent_step_index": int(
                getattr(prepared_run, "parent_step_index", 0) or 0
            ),
            "request_with_context": safe_request,
            "previous_context": "",
            "user_request": safe_request,
            "context_analysis": {},
            "plan": {},
            "search_context": "",
            "knowledge_context": "",
            "knowledge_sources": [],
            "search_sources": [],
            "evidence_items": [],
            "compact_evidence": [],
            "revision_history": [],
            "run_records": [],
            "last_document": "",
            "last_plan": {},
            "user_constraints": [],
            "unresolved_questions": [],
            "user_profile": None,
            "memory_context": "",
            "audit_summary": {},
            "think_log": [],
            "document_content": "",
            "revision_round": 0,
            "continue_revision": False,
            "review_meta": {},
            "reflection_meta": {},
            "reflection_done": False,
            "should_reflect": False,
            "quality_status": "pending",
            "run_status": "running",
            "error_step": "",
            "error_message": "",
            "errors": [],
        }, node="document.initial_state")

    @staticmethod
    def _config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    @staticmethod
    def _resolve_run_id(prepared_run: Any, thread_id: str) -> str:
        return str(thread_id or getattr(prepared_run, "run_id", "") or uuid4())

    @staticmethod
    def _attachment_redaction_fragments(request: str) -> tuple[str, ...]:
        fragments: set[str] = set()
        for match in re.finditer(
            r"\[文件内容\]\s*(.*?)\s*\[/文件内容\]",
            str(request or ""),
            flags=re.DOTALL,
        ):
            content = match.group(1).strip()
            if not content:
                continue
            fragments.add(content)
            for part in re.split(r"(?:\r?\n)+|(?<=[。！？!?])", content):
                normalized = part.strip()
                if len(normalized) >= 12:
                    fragments.add(normalized)
        return tuple(sorted(fragments, key=len, reverse=True))

    @classmethod
    def _runtime_redaction_fragments(
        cls,
        *,
        hydrated_request: str,
        hydrated_previous_context: str,
        runtime_snapshot: dict[str, Any],
    ) -> tuple[str, ...]:
        """Collect non-output runtime leaves that nodes must not checkpoint.

        Attachment bodies are always private, even when short. For recalled
        context/profile/history leaves, ignore strings shorter than four
        characters to avoid replacing common punctuation, roles, and labels.
        Long leaves also contribute line/sentence fragments so partial model
        echoes are removed from plan, audit, and think-log updates.
        """

        fragments = set(cls._attachment_redaction_fragments(hydrated_request))

        def add_text(value: str) -> None:
            normalized = str(value or "").strip()
            if len(normalized) < 4:
                return
            fragments.add(normalized)
            for part in re.split(r"(?:\r?\n)+|(?<=[。！？!?])", normalized):
                candidate = part.strip()
                if len(candidate) >= 12:
                    fragments.add(candidate)

        def collect(value: Any) -> None:
            if isinstance(value, str):
                add_text(value)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        add_text(hydrated_previous_context)
        collect(runtime_snapshot)
        return tuple(sorted(fragments, key=len, reverse=True))

    def _runtime_snapshot(self, prepared_run: Any) -> dict[str, Any]:
        existing = getattr(self.orchestrator, "_document_runtime_snapshot", {})
        snapshot = dict(existing) if isinstance(existing, dict) else {}
        snapshot.setdefault(
            "memory_context",
            str(
                getattr(self.orchestrator, "_current_agent_memory_context", "")
                or ""
            ),
        )
        profile = getattr(self.orchestrator, "user_profile", None)
        if isinstance(profile, dict):
            snapshot.setdefault("user_profile", dict(profile))

        memory = getattr(self.orchestrator, "memory", None)
        session_id = str(getattr(prepared_run, "session_id", "") or "")
        context_reader = getattr(memory, "get_context", None)
        if session_id and callable(context_reader):
            snapshot.setdefault(
                "last_document",
                str(context_reader(session_id, "last_document", "") or ""),
            )
            raw_last_plan = context_reader(session_id, "last_plan", {}) or {}
            snapshot.setdefault(
                "last_plan",
                dict(raw_last_plan) if isinstance(raw_last_plan, dict) else {},
            )
        snapshot.setdefault(
            "previous_context",
            str(getattr(prepared_run, "previous_context", "") or ""),
        )
        return snapshot

    def _runtime(
        self,
        think_handler,
        *,
        streaming: bool,
        prepared_run: Any = None,
    ) -> DocumentGraphRuntime:
        hydrated_request = (
            str(getattr(prepared_run, "request_with_context", "") or "")
            if prepared_run is not None
            else None
        )
        hydrated_previous_context = (
            str(getattr(prepared_run, "previous_context", "") or "")
            if prepared_run is not None
            else None
        )
        runtime_snapshot = (
            self._runtime_snapshot(prepared_run)
            if prepared_run is not None
            else None
        )
        return DocumentGraphRuntime(
            orchestrator=self.orchestrator,
            think_handler=think_handler,
            streaming=streaming,
            hydrated_request=hydrated_request,
            hydrated_previous_context=hydrated_previous_context,
            runtime_snapshot=runtime_snapshot,
            redaction_fragments=self._runtime_redaction_fragments(
                hydrated_request=hydrated_request or "",
                hydrated_previous_context=hydrated_previous_context or "",
                runtime_snapshot=runtime_snapshot or {},
            ),
        )

    def _checkpoint_input(
        self,
        prepared_run: Any,
        run_id: str,
    ) -> tuple[DocumentGraphState | None, DocumentGraphState | None]:
        """Choose a fresh input, checkpoint resume, or terminal replay.

        A document run shares the parent ``thread_id`` but is stored in the
        controlled ``document`` namespace.  Re-submitting the initial state for
        an existing thread would restart context/retrieval/writer nodes and can
        produce a different document after a process crash.  LangGraph resumes
        an unfinished checkpoint only when the input is ``None``; completed
        checkpoints are replayed directly without invoking any node.
        """
        snapshot = self.graph.get_state(self._config(run_id))
        values = dict(snapshot.values or {})
        # When invoked from a retained legacy child node, LangGraph can seed the
        # physical namespace with the parent's shared identity fields before the
        # standalone document graph has ever run. That is not a document
        # checkpoint: require the workflow's immutable request fields before
        # considering it resumable or terminal.
        required_identity_fields = {
            "run_id",
            "input_user_request",
            "request_with_context",
        }
        if not values or not required_identity_fields.issubset(values):
            return self._initial_state(prepared_run, run_id), None
        if snapshot.next:
            return None, None
        if values.get("run_status") == "failed":
            error_step = str(values.get("error_step") or "")
            retry_after_node = {
                # Marking the preceding node as completed schedules only the
                # failed node while retaining every successful prior result.
                "write": "retrieval",
                "review": "write",
            }.get(error_step)
            if retry_after_node:
                retry_update = validate_graph_update(
                    {
                        "run_status": "running",
                        "quality_status": "pending",
                        "continue_revision": False,
                        "error_step": "",
                        "error_message": "",
                    },
                    node="document.retry_state",
                )
                self.graph.update_state(
                    self._config(run_id),
                    retry_update,
                    as_node=retry_after_node,
                )
                return None, None
        return None, values

    def _result_from_state(self, state: DocumentGraphState) -> DocumentGraphRunResult:
        if state.get("run_status") == "failed":
            step = state.get("error_step", "document_workflow")
            message = state.get("error_message", "document generation failed")
            raise DocumentGraphExecutionError(f"{step}: {message}", state=state)

        ctx = DocumentGraphSteps.state_to_context(self.orchestrator, state)
        ctx.run_records.append({
            "step": "orchestrator_runtime",
            "runtime": "langgraph",
            "stream": False,
            "quality_status": state.get("quality_status", "passed"),
        })
        return DocumentGraphRunResult(
            ctx=ctx,
            document_content=state.get("document_content", ""),
            quality_status=state.get("quality_status", "passed"),
            revisions_applied=int(state.get("revision_round", 0)),
            run_id=state.get("run_id", ""),
        )

    def run(self, prepared_run: Any, *, think_handler, thread_id: str = "") -> DocumentGraphRunResult:
        run_id = self._resolve_run_id(prepared_run, thread_id)
        graph_input, terminal_state = self._checkpoint_input(prepared_run, run_id)
        state = terminal_state
        if state is None:
            state = self.graph.invoke(
                graph_input,
                config=self._config(run_id),
                context=self._runtime(
                    think_handler,
                    streaming=False,
                    prepared_run=prepared_run,
                ),
                durability=self.durability,
            )
        return self._result_from_state(state)

    def stream(
        self,
        prepared_run: Any,
        *,
        think_handler,
        thread_id: str = "",
        user_request: str,
    ):
        """Expose node custom events while retaining the reviewed final-answer contract."""
        run_id = self._resolve_run_id(prepared_run, thread_id)
        graph_input, terminal_state = self._checkpoint_input(prepared_run, run_id)
        final_state = terminal_state
        if final_state is None:
            chunks = self.graph.stream(
                graph_input,
                config=self._config(run_id),
                context=self._runtime(
                    think_handler,
                    streaming=True,
                    prepared_run=prepared_run,
                ),
                stream_mode=["custom", "values"],
                durability=self.durability,
            )
            for item in chunks:
                if isinstance(item, tuple) and len(item) == 2:
                    mode, payload = item
                else:
                    mode, payload = "custom", item
                if mode == "custom":
                    yield payload
                elif mode == "values":
                    final_state = payload

        if final_state is None:
            raise DocumentGraphExecutionError("document_workflow: graph returned no final state")

        result = self._result_from_state(final_state)
        result.ctx.run_records[-1]["stream"] = True
        response = self.orchestrator._build_document_run_result(
            result.ctx,
            result.document_content,
            str(
                user_request
                if getattr(prepared_run, "persisted_user_message", None) is None
                else getattr(prepared_run, "persisted_user_message")
            ),
            runtime="langgraph",
            quality_status=result.quality_status,
            revisions_applied=result.revisions_applied,
            run_id=result.run_id,
            effect_scope=str(
                getattr(prepared_run, "effect_scope", "") or ""
            ),
        )

        final_message = (
            "当前最优版本已确认，正在输出正文"
            if result.quality_status == "max_revisions"
            else "最终版本已确认，正在输出正文"
        )
        think_handler("Orchestrator", "📄", final_message)
        yield {"type": "think", "agent": "Orchestrator", "emoji": "📄", "message": final_message}
        yield {"type": "thinking_done", "summary": "写作、审核和反思完成，开始输出最终正文"}
        yield {"type": "answer_start", "message": "开始输出正文"}
        for chunk in self._iter_final_content_chunks(result.document_content):
            yield {"type": "answer_delta", "data": chunk}
            yield {"type": "content", "data": chunk}
        yield {"type": "answer_done", "answer": result.document_content}

        completion = f"文档生成完成，共{len(self.orchestrator.think_log)}个思考步骤"
        think_handler("Orchestrator", "✅", completion)
        yield {"type": "think", "agent": "Orchestrator", "emoji": "✅", "message": completion}
        yield {
            "type": "done",
            **response,
        }

    @staticmethod
    def _iter_final_content_chunks(text: str, chunk_size: int = 45):
        for start in range(0, len(text or ""), chunk_size):
            yield text[start:start + chunk_size]
