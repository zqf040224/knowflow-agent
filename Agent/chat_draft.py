"""Streaming official-document drafting pipeline for chat doc generation."""

from __future__ import annotations

import json
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from chat_architecture import INTENT_DOC_DRAFTING
from chat_events import route_actions, route_event, route_intent, route_payload, source_details_from_results, sse

logger = logging.getLogger(__name__)


def _persisted_user_message(message: Any, display_message: Any) -> str:
    """Keep an explicitly empty display message from falling back to hydration."""

    return str(message if display_message is None else display_message)


def _effect_scope(user_metadata: Any) -> str:
    if not isinstance(user_metadata, dict):
        return ""
    return str(user_metadata.get("effect_scope") or "").strip()


def _effect_key(
    run_id: str,
    effect: str,
    *,
    effect_scope: str = "",
) -> str | None:
    normalized_run_id = str(run_id or "").strip()
    if not normalized_run_id:
        return None
    prefix = f"{normalized_run_id}:tool_draft_document"
    normalized_scope = str(effect_scope or "").strip()
    if normalized_scope:
        prefix = f"{prefix}:{normalized_scope}"
    return f"{prefix}:{effect}"


def _supported_kwargs(callback: Callable[..., Any], **values: Any) -> dict[str, Any]:
    """Pass new runtime-only keywords without breaking retained test adapters."""

    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return values
    names = {parameter.name for parameter in parameters}
    return {name: value for name, value in values.items() if name in names}


@dataclass
class DocumentDraftDependencies:
    memory: Any
    orchestrator_factory: Callable[..., Any]
    resolve_export_template: Callable[[str, dict, str], str]
    record_agent_run_token_usage: Callable[..., None]
    record_token_usage: Callable[..., None]
    record_token_usage_best_effort: Callable[..., None] | None = None


@dataclass
class NativeDocumentExecution:
    """Runtime-only services for a parent-mounted document subgraph."""

    orchestrator: Any
    profile: Any
    stored_user_message: str
    prepared_run: Any = None
    runtime_snapshot: dict[str, Any] = field(default_factory=dict)
    attachment_bodies: list[str] = field(default_factory=list)
    effect_scope: str = ""


class DocumentDraftStreamService:
    def __init__(self, deps: DocumentDraftDependencies):
        self.deps = deps

    def stream(
        self,
        message,
        session_id,
        user_id,
        user_info=None,
        display_message=None,
        user_metadata=None,
        route=None,
    ):
        """Agent 流式生成 - 带思考过程显示."""
        logger.info("用户 %s 使用会话: %s", user_id, session_id)
        stored_user_message = _persisted_user_message(message, display_message)
        doc_content = ""
        think_log = []
        parent_run_id = ""
        effect_scope = _effect_scope(user_metadata)
        if isinstance(user_metadata, dict):
            parent_run_id = str(user_metadata.get("run_id") or "")

        try:
            logger.info("开始 Agent 流式生成，用户消息: %s...", message[:50])
            profile = self.deps.memory.get_user_profile(user_id)
            if profile:
                logger.info("用户偏好: %s, %s", profile.preferred_font, profile.preferred_size)

            context = self.deps.memory.get_context_for_prompt(session_id, max_messages=5)
            if context:
                logger.info("会话上下文:\n%s...", context[:200])

            yield sse({"type": "start"})
            yield sse({"type": "session", "session_id": session_id})
            if route:
                yield route_event(route)
            yield sse({"type": "thinking_start", "message": "开始拆解写作任务"})

            format_info = ""
            if profile:
                format_info = f"用户偏好：{profile.preferred_font} {profile.preferred_size}，风格：{profile.writing_style}"
            runner = self.deps.orchestrator_factory(session_id, profile=profile, user_info=user_info)

            pending_answer_events: list[dict[str, Any]] = []
            runner_done: dict[str, Any] | None = None
            runner_error: dict[str, Any] | None = None
            answer_phase = False
            runner_stream = runner.run_stream(
                message + (f"\n\n[{format_info}]" if format_info else ""),
                on_think=lambda agent, emoji, msg: think_log.append({"agent": agent, "emoji": emoji, "message": msg}),
                session_id=session_id,
                run_id=parent_run_id,
                **_supported_kwargs(
                    runner.run_stream,
                    persisted_user_message=stored_user_message,
                    effect_scope=effect_scope,
                ),
            )
            for event in runner_stream:
                event_type = event.get("type")
                if event_type == "done":
                    runner_done = dict(event)
                    continue
                if event_type == "error":
                    runner_error = dict(event)
                    continue
                if event_type in {
                    "content",
                    "answer_start",
                    "answer_delta",
                    "answer_done",
                    "thinking_done",
                }:
                    answer_phase = True
                    pending_answer_events.append(dict(event))
                    if event_type == "content":
                        doc_content += event.get("data", "")
                    continue
                if answer_phase:
                    # Completion think events belong to the same commit gate as
                    # the answer that precedes them.
                    pending_answer_events.append(dict(event))
                    continue
                if event_type in {"plan", "think"}:
                    yield sse(event)
                elif event_type == "reasoning_chunk":
                    continue
                elif event_type == "reflection":
                    reflection_data = dict(event.get("data") or {})
                    reflection_data.pop("reasoning_content", None)
                    yield sse({"type": "reflection", "data": reflection_data})
                else:
                    yield sse(event)

            if runner_error is not None:
                self._record_failure(
                    user_id,
                    user_info,
                    session_id,
                    message,
                    str(runner_error.get("message") or "document runner failed"),
                    run_id=parent_run_id,
                    effect_scope=effect_scope,
                )
                yield sse(runner_error)
                return
            if runner_done is None:
                raise RuntimeError("Document runner returned no terminal done event")

            # The graph and linear rollback paths share this commit gate. All
            # fallible usage, export and profile writes finish before any answer
            # or successful terminal event becomes visible to the client.
            done = self.build_public_done(
                runner_done,
                think_log=think_log,
                user_id=user_id,
                user_info=user_info,
                session_id=session_id,
                stored_user_message=stored_user_message,
                route=route,
                parent_run_id=parent_run_id,
                effect_scope=effect_scope,
            )
            self._update_common_doc_types(profile, user_id, stored_user_message)

            for event in pending_answer_events:
                yield sse(event)
            yield sse({
                "type": "run_done",
                "session_id": session_id,
                "intent": route_intent(route, INTENT_DOC_DRAFTING),
            })
            yield sse(done)
            logger.info("Agent 生成完成，内容长度: %s", len(doc_content))
        except json.JSONDecodeError as exc:
            logger.error("JSON 解析错误: %s", exc)
            self._record_failure(
                user_id,
                user_info,
                session_id,
                message,
                str(exc),
                run_id=parent_run_id,
                effect_scope=effect_scope,
            )
            yield sse({"type": "error", "message": "数据解析错误，请重试"})
        except Exception as exc:
            logger.exception("Agent 生成过程中发生未知错误: %s", exc)
            self._record_failure(
                user_id,
                user_info,
                session_id,
                message,
                str(exc),
                run_id=parent_run_id,
                effect_scope=effect_scope,
            )
            yield sse({"type": "error", "message": "生成失败，请稍后重试"})

    def create_native_execution(
        self,
        *,
        message: str,
        display_message: str,
        session_id: str,
        user_id: str,
        user_info: Any,
        run_id: str,
        prepare: bool,
        runtime_snapshot: dict[str, Any] | None = None,
        effect_scope: str = "",
    ) -> NativeDocumentExecution:
        """Create runtime services without putting them into graph State."""

        profile = self.deps.memory.get_user_profile(user_id)
        stored_user_message = _persisted_user_message(message, display_message)
        orchestrator = self.deps.orchestrator_factory(
            session_id,
            profile=profile,
            user_info=user_info,
        )
        orchestrator._current_effect_run_id = str(run_id or "")
        orchestrator._current_effect_scope = str(effect_scope or "").strip()
        orchestrator._current_persisted_user_message = stored_user_message
        frozen_snapshot = dict(runtime_snapshot or {})
        frozen_profile = frozen_snapshot.get("user_profile")
        if isinstance(frozen_profile, dict) and frozen_profile:
            setter = getattr(orchestrator, "set_user_profile", None)
            if callable(setter):
                setter(dict(frozen_profile))
            else:
                orchestrator.user_profile = dict(frozen_profile)
        prepared_run = None
        effective_snapshot: dict[str, Any] = {}
        if prepare:
            format_info = ""
            if isinstance(frozen_profile, dict) and frozen_profile:
                format_info = (
                    f"用户偏好：{frozen_profile.get('preferred_font', '')} "
                    f"{frozen_profile.get('preferred_size', '')}，"
                    f"风格：{frozen_profile.get('writing_style', '')}"
                )
            elif profile:
                format_info = (
                    f"用户偏好：{profile.preferred_font} {profile.preferred_size}，"
                    f"风格：{profile.writing_style}"
                )
            request = message + (f"\n\n[{format_info}]" if format_info else "")
            prepare_document_run = orchestrator._prepare_document_run
            prepared_run = prepare_document_run(
                request,
                session_id=session_id,
                run_id=run_id,
                **_supported_kwargs(
                    prepare_document_run,
                    persisted_user_message=stored_user_message,
                    effect_scope=effect_scope,
                ),
            )
            if frozen_snapshot:
                frozen_previous = str(
                    frozen_snapshot.get("previous_context") or ""
                )
                prepared_run.previous_context = frozen_previous
                prepared_run.request_with_context = (
                    f"之前的需求：{frozen_previous}\n\n当前需求：{prepared_run.user_request}"
                    if frozen_previous
                    else prepared_run.user_request
                )
                effective_snapshot = frozen_snapshot
            else:
                profile_snapshot = getattr(orchestrator, "user_profile", None)
                if not isinstance(profile_snapshot, dict):
                    profile_snapshot = {
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
                history_reader = getattr(
                    self.deps.memory,
                    "get_session_history",
                    None,
                )
                history = (
                    history_reader(session_id, limit=10)
                    if callable(history_reader)
                    else []
                )
                conversation_history = []
                for message_item in history or []:
                    if isinstance(message_item, dict):
                        role = message_item.get("role", "")
                        content = message_item.get("content", "")
                    else:
                        role = getattr(message_item, "role", "")
                        content = getattr(message_item, "content", "")
                    conversation_history.append({
                        "role": str(role or ""),
                        "content": str(content or ""),
                    })
                context_reader = getattr(self.deps.memory, "get_context", None)
                last_document = ""
                last_plan: dict[str, Any] = {}
                if callable(context_reader):
                    last_document = str(
                        context_reader(session_id, "last_document", "") or ""
                    )
                    raw_last_plan = (
                        context_reader(session_id, "last_plan", {}) or {}
                    )
                    if isinstance(raw_last_plan, dict):
                        last_plan = dict(raw_last_plan)
                effective_snapshot = {
                    "memory_context": str(
                        getattr(
                            orchestrator,
                            "_current_agent_memory_context",
                            "",
                        )
                        or ""
                    ),
                    "user_profile": dict(profile_snapshot or {}),
                    "last_document": last_document,
                    "last_plan": last_plan,
                    "previous_context": str(
                        getattr(prepared_run, "previous_context", "") or ""
                    ),
                    "conversation_history": conversation_history,
                }
            orchestrator._document_runtime_snapshot = dict(effective_snapshot)
            orchestrator._current_agent_memory_context = str(
                effective_snapshot.get("memory_context") or ""
            )
        return NativeDocumentExecution(
            orchestrator=orchestrator,
            profile=profile,
            stored_user_message=stored_user_message,
            prepared_run=prepared_run,
            runtime_snapshot=effective_snapshot,
            effect_scope=str(effect_scope or "").strip(),
        )

    def build_public_done(
        self,
        event: dict[str, Any],
        *,
        think_log: list[dict[str, Any]],
        user_id: str,
        user_info: Any,
        session_id: str,
        stored_user_message: str,
        route: Any,
        parent_run_id: str,
        effect_scope: str = "",
    ) -> dict[str, Any]:
        """Apply the existing public done contract to either document runner."""

        usage_recorder = self.deps.record_agent_run_token_usage
        usage_recorder(
            event.get("run_records", []),
            user_id=user_id,
            user_info=user_info,
            session_id=session_id,
            mode="agent",
            run_id=parent_run_id,
            **_supported_kwargs(
                usage_recorder,
                effect_scope=effect_scope,
            ),
        )
        document = event.get("document", "")
        return {
            "type": "done",
            "intent": route_intent(route, INTENT_DOC_DRAFTING),
            "answer": document,
            "document": document,
            "think_log": think_log,
            "session_id": session_id,
            "plan": event.get("plan"),
            "route": route_payload(route),
            "actions": route_actions(route),
            "export_template": self.deps.resolve_export_template(
                document,
                event.get("plan") or {},
                stored_user_message,
            ),
            "export_spreadsheet_template": "",
            "run_records": event.get("run_records", []),
            "source_filenames": list(dict.fromkeys(event.get("source_filenames", [])))[:8],
            "source_details": source_details_from_results(event.get("source_details", [])),
            "audit_summary": event.get("audit_summary", {}),
            "runtime": event.get("runtime", "langgraph"),
            "quality_status": event.get("quality_status", "pending"),
            "revision_rounds": int(event.get("revision_rounds", 0) or 0),
            "revisions_applied": int(event.get("revision_rounds", 0) or 0),
            "run_id": event.get("run_id") or parent_run_id,
        }

    def _record_failure(
        self,
        user_id,
        user_info,
        session_id,
        message,
        error_message: str,
        *,
        run_id: str = "",
        effect_scope: str = "",
    ) -> None:
        recorder = (
            self.deps.record_token_usage_best_effort
            or self.deps.record_token_usage
        )
        try:
            recorder(
                user_id=user_id,
                user_info=user_info,
                session_id=session_id,
                mode="agent",
                agent="AgentPipeline",
                model="mixed",
                prompt_chars=len(message or ""),
                status="failed",
                error_message=error_message,
                effect_key=_effect_key(
                    run_id,
                    "token_usage_pipeline_failure",
                    effect_scope=effect_scope,
                ),
            )
        except Exception:
            logger.warning("Failed to record document pipeline failure", exc_info=True)

    def _update_common_doc_types(self, profile, user_id: str, stored_user_message: str) -> None:
        if not profile:
            return
        doc_types = list(profile.common_doc_types or [])
        for doc_type in ["通知", "请示", "报告", "对策建议"]:
            if doc_type in stored_user_message and doc_type not in doc_types:
                doc_types.append(doc_type)
                self.deps.memory.update_user_profile(user_id, {"common_doc_types": doc_types})
