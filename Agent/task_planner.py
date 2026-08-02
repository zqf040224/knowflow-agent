"""Task planning for tool-selected chat execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
    VALID_INTENTS,
)


TOOL_KNOWLEDGE_QA = "knowledge_qa"
TOOL_DRAFT_DOCUMENT = "draft_document"
TOOL_FORMAT_DOCUMENT = "format_document"
TOOL_PREPARE_FORM_EXPORT = "prepare_form_export"
TOOL_PREPARE_SPREADSHEET_TRANSFORM = "prepare_spreadsheet_transform"
TOOL_CLARIFY = "clarify"
TOOL_IDENTITY_HELP = "identity_help"

ACTION_TOOLS = {TOOL_PREPARE_FORM_EXPORT, TOOL_PREPARE_SPREADSHEET_TRANSFORM}
VALID_TOOLS = {
    TOOL_KNOWLEDGE_QA,
    TOOL_DRAFT_DOCUMENT,
    TOOL_FORMAT_DOCUMENT,
    TOOL_PREPARE_FORM_EXPORT,
    TOOL_PREPARE_SPREADSHEET_TRANSFORM,
    TOOL_CLARIFY,
    TOOL_IDENTITY_HELP,
}

INTENT_TO_TOOL = {
    INTENT_KNOWLEDGE_QA: TOOL_KNOWLEDGE_QA,
    INTENT_DOC_DRAFTING: TOOL_DRAFT_DOCUMENT,
    INTENT_DOC_FORMATTING: TOOL_FORMAT_DOCUMENT,
    INTENT_FORM_TEMPLATE_EXPORT: TOOL_PREPARE_FORM_EXPORT,
    INTENT_SPREADSHEET_TRANSFORM: TOOL_PREPARE_SPREADSHEET_TRANSFORM,
    INTENT_CLARIFY: TOOL_CLARIFY,
    INTENT_IDENTITY_HELP: TOOL_IDENTITY_HELP,
}

TOOL_TO_INTENT = {
    TOOL_KNOWLEDGE_QA: INTENT_KNOWLEDGE_QA,
    TOOL_DRAFT_DOCUMENT: INTENT_DOC_DRAFTING,
    TOOL_FORMAT_DOCUMENT: INTENT_DOC_FORMATTING,
    TOOL_PREPARE_FORM_EXPORT: INTENT_FORM_TEMPLATE_EXPORT,
    TOOL_PREPARE_SPREADSHEET_TRANSFORM: INTENT_SPREADSHEET_TRANSFORM,
    TOOL_CLARIFY: INTENT_CLARIFY,
    TOOL_IDENTITY_HELP: INTENT_IDENTITY_HELP,
}


@dataclass
class TaskStep:
    tool: str
    reason: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "reason": self.reason,
            "input": self.input,
            "risk_level": self.risk_level,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class TaskPlan:
    task_type: str
    steps: list[TaskStep]
    requires_confirmation: bool = False
    final_response_mode: str = "tool_output"
    source: str = "rules"
    route: Optional[RouteResult] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "steps": [step.to_dict() for step in self.steps],
            "requires_confirmation": self.requires_confirmation,
            "final_response_mode": self.final_response_mode,
            "source": self.source,
            "route": self.route.to_dict() if self.route else {},
        }


class TaskPlanner:
    """Plan chat requests as tool calls, with IntentRouter as fallback."""

    def __init__(
        self,
        reimbursement_detector: Optional[Callable[[str, str], str]] = None,
        planner_classifier: Optional[Callable[[dict[str, Any]], Any]] = None,
        intent_classifier: Optional[Callable[[dict[str, Any]], Any]] = None,
    ):
        self.reimbursement_detector = reimbursement_detector
        self.planner_classifier = planner_classifier
        self.intent_classifier = intent_classifier

    def plan(
        self,
        *,
        message: str,
        display_message: str = "",
        mode: str = "chat",
        attachments: Optional[list[dict]] = None,
        conversation_context: str = "",
        has_last_document: bool = False,
        user_info: Any = None,
    ) -> TaskPlan:
        attachments = attachments or []
        fallback = self._fallback_plan(
            message=message,
            display_message=display_message,
            mode=mode,
            attachments=attachments,
            conversation_context=conversation_context,
            has_last_document=has_last_document,
        )
        raw = self._call_planner_classifier(
            fallback=fallback,
            message=display_message or message,
            attachments=attachments,
            conversation_context=conversation_context,
            has_last_document=has_last_document,
            user_info=user_info,
        )
        planned = self._normalize_classifier_plan(raw, fallback)
        return planned or fallback

    def _fallback_plan(
        self,
        *,
        message: str,
        display_message: str,
        mode: str,
        attachments: list[dict],
        conversation_context: str,
        has_last_document: bool,
    ) -> TaskPlan:
        route = IntentRouter(
            self.reimbursement_detector,
            intent_classifier=self.intent_classifier,
        ).route(
            message=message,
            display_message=display_message,
            mode=mode,
            attachments=attachments,
            conversation_context=conversation_context,
            has_last_document=has_last_document,
        )
        text = display_message or message or ""
        primary_tool = INTENT_TO_TOOL.get(route.intent, TOOL_KNOWLEDGE_QA)
        tools = [primary_tool]
        if self._looks_like_research_then_write(text) and primary_tool not in {
            TOOL_FORMAT_DOCUMENT,
            TOOL_PREPARE_FORM_EXPORT,
            TOOL_PREPARE_SPREADSHEET_TRANSFORM,
            TOOL_CLARIFY,
            TOOL_IDENTITY_HELP,
        }:
            tools = [TOOL_KNOWLEDGE_QA, TOOL_DRAFT_DOCUMENT]
        steps = [self._step_for_tool(tool, route, text) for tool in tools]
        return TaskPlan(
            task_type=route.document_type or route.intent,
            steps=steps,
            requires_confirmation=any(step.requires_confirmation for step in steps),
            final_response_mode="confirm_actions" if any(step.requires_confirmation for step in steps) else "tool_output",
            source="rules",
            route=route,
        )

    def _call_planner_classifier(self, **payload):
        classifier = self.planner_classifier
        if not classifier:
            return None
        try:
            safe_payload = dict(payload)
            fallback = safe_payload.get("fallback")
            safe_payload["fallback"] = fallback.to_dict() if hasattr(fallback, "to_dict") else {}
            route = getattr(fallback, "route", None)
            safe_payload["rule_result"] = route.to_dict() if hasattr(route, "to_dict") else {}
            safe_payload["allowed_intents"] = sorted(VALID_INTENTS)
            return classifier(safe_payload)
        except Exception:
            return None

    def _normalize_classifier_plan(self, raw: Any, fallback: TaskPlan) -> Optional[TaskPlan]:
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if not isinstance(raw, dict):
            return None

        if raw.get("intent") and not raw.get("steps"):
            intent = str(raw.get("intent") or "").strip()
            tool = INTENT_TO_TOOL.get(intent)
            if not tool:
                return None
            fallback_route = fallback.route
            route = RouteResult(
                intent=intent,
                confidence=float(raw.get("confidence", 0.8) or 0.8),
                reason=str(raw.get("reason") or "LLM planner selected intent").strip(),
                document_type=str(raw.get("document_type") or fallback.task_type).strip(),
                template_key=str(raw.get("template_key") or getattr(fallback_route, "template_key", "") or "").strip(),
                requires_retrieval=intent in {INTENT_KNOWLEDGE_QA, INTENT_DOC_DRAFTING, INTENT_DOC_FORMATTING},
                actions=getattr(fallback_route, "actions", []) if getattr(fallback_route, "intent", "") == intent else [],
            )
            step = self._step_for_tool(tool, route, "")
            return TaskPlan(
                task_type=route.document_type or intent,
                steps=[step],
                requires_confirmation=step.requires_confirmation,
                final_response_mode="confirm_actions" if step.requires_confirmation else "tool_output",
                source="llm",
                route=route,
            )

        steps_raw = raw.get("steps") or []
        steps: list[TaskStep] = []
        for item in steps_raw:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or item.get("name") or "").strip()
            if tool not in VALID_TOOLS:
                continue
            steps.append(TaskStep(
                tool=tool,
                reason=str(item.get("reason") or "").strip(),
                input=item.get("input") if isinstance(item.get("input"), dict) else {},
                risk_level="confirm" if tool in ACTION_TOOLS else str(item.get("risk_level") or "low"),
                requires_confirmation=bool(item.get("requires_confirmation", tool in ACTION_TOOLS)),
            ))
        if not steps:
            return None
        return TaskPlan(
            task_type=str(raw.get("task_type") or fallback.task_type),
            steps=steps[:4],
            requires_confirmation=any(step.requires_confirmation for step in steps),
            final_response_mode=str(raw.get("final_response_mode") or fallback.final_response_mode),
            source="llm",
            route=fallback.route,
        )

    def _step_for_tool(self, tool: str, route: RouteResult, text: str) -> TaskStep:
        reason = route.reason or "根据用户需求选择工具"
        risk = "confirm" if tool in ACTION_TOOLS else "low"
        return TaskStep(
            tool=tool,
            reason=reason,
            input={"message": text, "route": route.to_dict()},
            risk_level=risk,
            requires_confirmation=tool in ACTION_TOOLS,
        )

    @staticmethod
    def _looks_like_research_then_write(text: str) -> bool:
        compact = (text or "").replace(" ", "")
        lookup_markers = ("先查", "查询", "检索", "根据", "依据", "结合")
        write_markers = ("写", "起草", "撰写", "整理一份", "形成", "生成")
        return any(m in compact for m in lookup_markers) and any(m in compact for m in write_markers)
