"""Streaming official-document drafting pipeline for chat doc generation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from chat_architecture import INTENT_DOC_DRAFTING
from chat_events import route_actions, route_event, route_intent, route_payload, source_details_from_results, sse

logger = logging.getLogger(__name__)


@dataclass
class DocumentDraftDependencies:
    memory: Any
    orchestrator_factory: Callable[..., Any]
    resolve_export_template: Callable[[str, dict, str], str]
    record_agent_run_token_usage: Callable[..., None]
    record_token_usage: Callable[..., None]


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
        stored_user_message = display_message or message
        doc_content = ""
        think_log = []

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

            for event in runner.run_stream(
                message + (f"\n\n[{format_info}]" if format_info else ""),
                on_think=lambda agent, emoji, msg: think_log.append({"agent": agent, "emoji": emoji, "message": msg}),
                session_id=session_id,
            ):
                event_type = event.get("type")
                if event_type == "content":
                    doc_content += event.get("data", "")
                    yield sse(event)
                elif event_type in {"plan", "think"}:
                    yield sse(event)
                elif event_type == "reasoning_chunk":
                    continue
                elif event_type == "reflection":
                    reflection_data = dict(event.get("data") or {})
                    reflection_data.pop("reasoning_content", None)
                    yield sse({"type": "reflection", "data": reflection_data})
                elif event_type == "done":
                    self.deps.record_agent_run_token_usage(
                        event.get("run_records", []),
                        user_id=user_id,
                        user_info=user_info,
                        session_id=session_id,
                        mode="agent",
                    )
                    document = event.get("document", "")
                    yield sse({"type": "run_done", "session_id": session_id, "intent": route_intent(route, INTENT_DOC_DRAFTING)})
                    yield sse({
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
                    })
                else:
                    yield sse(event)

            logger.info("Agent 生成完成，内容长度: %s", len(doc_content))
            self._update_common_doc_types(profile, user_id, stored_user_message)
        except json.JSONDecodeError as exc:
            logger.error("JSON 解析错误: %s", exc)
            self._record_failure(user_id, user_info, session_id, message, str(exc))
            yield sse({"type": "error", "message": "数据解析错误，请重试"})
        except Exception as exc:
            logger.exception("Agent 生成过程中发生未知错误: %s", exc)
            self._record_failure(user_id, user_info, session_id, message, str(exc))
            yield sse({"type": "error", "message": "生成失败，请稍后重试"})

    def _record_failure(self, user_id, user_info, session_id, message, error_message: str) -> None:
        self.deps.record_token_usage(
            user_id=user_id,
            user_info=user_info,
            session_id=session_id,
            mode="agent",
            agent="AgentPipeline",
            model="mixed",
            prompt_chars=len(message or ""),
            status="failed",
            error_message=error_message,
        )

    def _update_common_doc_types(self, profile, user_id: str, stored_user_message: str) -> None:
        if not profile:
            return
        doc_types = profile.common_doc_types or []
        for doc_type in ["通知", "请示", "报告", "对策建议"]:
            if doc_type in stored_user_message and doc_type not in doc_types:
                doc_types.append(doc_type)
                self.deps.memory.update_user_profile(user_id, {"common_doc_types": doc_types})
