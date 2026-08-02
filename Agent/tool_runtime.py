"""Tool registry and orchestrator for planned chat execution."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional

from chat_architecture import RouteResult
from chat_events import parse_sse_events, sse
from task_planner import (
    TOOL_CLARIFY,
    TOOL_DRAFT_DOCUMENT,
    TOOL_FORMAT_DOCUMENT,
    TOOL_IDENTITY_HELP,
    TOOL_KNOWLEDGE_QA,
    TOOL_PREPARE_FORM_EXPORT,
    TOOL_PREPARE_SPREADSHEET_TRANSFORM,
    TOOL_TO_INTENT,
    TaskPlan,
    TaskStep,
)


@dataclass(frozen=True)
class ChatTool:
    name: str
    description: str
    risk_level: str
    input_schema: dict[str, Any]
    stream: Callable[..., Iterable[str]]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ChatTool] = {}

    def register(self, tool: ChatTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ChatTool:
        if name not in self._tools:
            raise KeyError(f"Unknown chat tool: {name}")
        return self._tools[name]

    def list_public(self) -> list[dict[str, Any]]:
        return [tool.to_public_dict() for tool in self._tools.values()]


class ToolOrchestrator:
    """Execute planned tools and expose a single compatible SSE stream."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def stream(self, prepared, task_plan: TaskPlan) -> Iterable[str]:
        yield sse({"type": "start"})
        yield sse({"type": "session", "session_id": prepared.session_id})
        yield sse({"type": "thinking_start", "message": "正在规划工具调用"})
        yield sse({"type": "tool_plan", "data": task_plan.to_dict()})
        yield sse({
            "type": "think",
            "agent": "TaskPlanner",
            "emoji": "",
            "message": self._plan_summary(task_plan),
        })

        final_done: Optional[dict[str, Any]] = None
        tool_results = []
        current_message = prepared.message
        for index, step in enumerate(task_plan.steps):
            is_final_step = index == len(task_plan.steps) - 1
            tool = self.registry.get(step.tool)
            tool_call = {
                "index": index + 1,
                "tool": tool.name,
                "reason": step.reason,
                "risk_level": step.risk_level or tool.risk_level,
                "requires_confirmation": step.requires_confirmation,
            }
            yield sse({"type": "tool_call", "data": tool_call})
            yield sse({
                "type": "think",
                "agent": "ToolOrchestrator",
                "emoji": "",
                "message": f"调用工具: {tool.name}",
            })

            route = self._route_for_step(step, task_plan.route)
            step_done = None
            step_prepared = self._prepared_for_step(prepared, current_message)
            for event in self._stream_tool_events(tool, step_prepared, route):
                event_type = event.get("type")
                if event_type in {"start", "session", "route"}:
                    continue
                if event_type == "done":
                    step_done = event
                    continue
                if event_type == "run_done":
                    continue
                if not is_final_step and event_type in {"answer_start", "answer_delta", "answer_done", "content"}:
                    continue
                yield sse(event)

            if step_done is None:
                step_done = self._missing_tool_done(prepared, step, route)
            if not is_final_step:
                current_message = self._message_with_tool_result(current_message, step, step_done)
            tool_result = self._tool_result_event(step, step_done, index + 1)
            tool_results.append(tool_result["data"])
            yield sse(tool_result)
            if step.requires_confirmation or step_done.get("actions"):
                yield sse({
                    "type": "tool_confirm_required",
                    "data": {
                        "tool": step.tool,
                        "actions": step_done.get("actions", []),
                        "message": step_done.get("answer", ""),
                    },
                })
            final_done = step_done

        if final_done is None:
            final_done = self._empty_done(prepared, task_plan)
        final_done.setdefault("plan", {})
        if isinstance(final_done["plan"], dict):
            final_done["plan"] = {
                **final_done["plan"],
                "task_planner": task_plan.to_dict(),
                "tool_results": tool_results,
            }
        yield sse({"type": "run_done", "session_id": prepared.session_id, "intent": final_done.get("intent", "")})
        yield sse(final_done)

    def _stream_tool_events(self, tool: ChatTool, prepared, route) -> list[dict[str, Any]]:
        raw_chunks = tool.stream(
            prepared.message,
            prepared.session_id,
            prepared.user_id,
            prepared.user_info,
            prepared.display_message,
            prepared.user_metadata,
            route,
        )
        for chunk in raw_chunks:
            events = parse_sse_events([chunk])
            for event in events:
                yield event

    @staticmethod
    def _prepared_for_step(prepared, message: str):
        return SimpleNamespace(
            message=message,
            display_message=prepared.display_message,
            session_id=prepared.session_id,
            user_id=prepared.user_id,
            user_info=prepared.user_info,
            user_metadata=prepared.user_metadata,
            attachments=getattr(prepared, "attachments", []),
        )

    @staticmethod
    def _message_with_tool_result(message: str, step: TaskStep, done: dict[str, Any]) -> str:
        result_text = done.get("document") or done.get("answer") or ""
        if not result_text:
            return message
        return (
            f"{message}\n\n[工具结果: {step.tool}]\n"
            f"{result_text[:3000]}\n"
            "[/工具结果]\n"
        )

    @staticmethod
    def _route_for_step(step: TaskStep, fallback_route) -> RouteResult:
        fallback_payload = fallback_route.to_dict() if hasattr(fallback_route, "to_dict") else {}
        payload = step.input.get("route") if isinstance(step.input, dict) else None
        if not isinstance(payload, dict):
            payload = fallback_payload
        intent = TOOL_TO_INTENT.get(step.tool, payload.get("intent", "knowledge_qa"))
        return RouteResult(
            intent=intent,
            confidence=float(payload.get("confidence", 0.8) or 0.8),
            reason=step.reason or payload.get("reason", "TaskPlanner selected tool"),
            document_type=payload.get("document_type", ""),
            template_key=payload.get("template_key", ""),
            requires_retrieval=bool(payload.get("requires_retrieval", intent in {"knowledge_qa", "doc_drafting", "doc_formatting"})),
            actions=payload.get("actions", []) if isinstance(payload.get("actions", []), list) else [],
        )

    @staticmethod
    def _plan_summary(task_plan: TaskPlan) -> str:
        tools = " -> ".join(step.tool for step in task_plan.steps)
        source = "LLM" if task_plan.source == "llm" else "规则"
        return f"{source}规划: {tools or '无工具'}"

    @staticmethod
    def _tool_result_event(step: TaskStep, done: dict[str, Any], index: int) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "data": {
                "index": index,
                "tool": step.tool,
                "intent": done.get("intent", ""),
                "answer_chars": len(done.get("answer") or done.get("document") or ""),
                "action_count": len(done.get("actions", []) or []),
                "source_count": len(done.get("source_details", []) or done.get("source_filenames", []) or []),
            },
        }

    @staticmethod
    def _missing_tool_done(prepared, step: TaskStep, route) -> dict[str, Any]:
        return {
            "type": "done",
            "intent": getattr(route, "intent", TOOL_TO_INTENT.get(step.tool, "")),
            "answer": "工具没有返回可用结果，请稍后重试。",
            "document": "",
            "session_id": prepared.session_id,
            "plan": {"task_type": "工具执行"},
            "route": route.to_dict() if hasattr(route, "to_dict") else {},
            "actions": [],
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": [],
            "source_details": [],
            "audit_summary": {},
        }

    @staticmethod
    def _empty_done(prepared, task_plan: TaskPlan) -> dict[str, Any]:
        return {
            "type": "done",
            "intent": "",
            "answer": "我还没有找到可执行的工具步骤，请补充你的具体需求。",
            "document": "",
            "session_id": prepared.session_id,
            "plan": {"task_type": task_plan.task_type, "task_planner": task_plan.to_dict()},
            "route": task_plan.route.to_dict() if task_plan.route else {},
            "actions": [],
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": [],
            "source_details": [],
            "audit_summary": {},
        }
