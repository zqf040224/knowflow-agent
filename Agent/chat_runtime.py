"""Production /api/chat request runtime.

This module owns the outer chat orchestration boundary: request normalization,
attachment hydration, IntentRouter classification, and dispatch to the intent
stream services. Add new chat intents here only after adding the intent contract
in chat_architecture.py and a stream service behind ChatServiceContainer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypedDict

from chat_architecture import (
    INTENT_CLARIFY,
    INTENT_DOC_DRAFTING,
    INTENT_DOC_FORMATTING,
    INTENT_FORM_TEMPLATE_EXPORT,
    INTENT_IDENTITY_HELP,
    INTENT_KNOWLEDGE_QA,
    INTENT_SPREADSHEET_TRANSFORM,
    IntentRouter,
)

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised only when dependency is missing
    END = "__end__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(exc)


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


@dataclass
class PreparedChatRequest:
    message: str
    display_message: str
    mode: str
    session_id: str
    user_id: str
    user_info: Any
    user_metadata: Optional[dict]
    attachments: list[dict]


class ChatGraphState(TypedDict, total=False):
    raw_data: dict
    user_id: str
    user_info: Any
    prepared: PreparedChatRequest
    route: Any
    task_plan: Any
    stream: Iterable[str]


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

    def __init__(self, deps: ChatRuntimeDependencies, *, use_langgraph: Optional[bool] = None):
        self.deps = deps
        self._use_langgraph = self._resolve_use_langgraph(use_langgraph)
        self._graph = self._build_graph() if self._use_langgraph else None

    @staticmethod
    def _resolve_use_langgraph(explicit: Optional[bool]) -> bool:
        if explicit is not None:
            return bool(explicit and LANGGRAPH_AVAILABLE)
        value = os.getenv("CHAT_RUNTIME", "langgraph").strip().lower()
        if value in {"legacy", "pipeline", "off", "false", "0"}:
            return False
        return LANGGRAPH_AVAILABLE

    @property
    def uses_langgraph(self) -> bool:
        return bool(self._use_langgraph)

    def stream(self, raw_data: dict, *, user_id: str, user_info: Any) -> Iterable[str]:
        initial_state: ChatGraphState = {
            "raw_data": raw_data or {},
            "user_id": user_id,
            "user_info": user_info,
        }
        if self._should_use_task_planner():
            state = dict(initial_state)
            state.update(self._prepare_node(state))
            state.update(self._plan_tools_node(state))
            state.update(self._execute_tools_node(state))
            return state["stream"]
        if self._graph is None:
            state = dict(initial_state)
            state.update(self._prepare_node(state))
            state.update(self._route_node(state))
            state.update(self._dispatch_node(state))
        else:
            state = self._graph.invoke(initial_state)
        return state["stream"]

    def _should_use_task_planner(self) -> bool:
        value = os.getenv("CHAT_RUNTIME", "task_planner").strip().lower()
        if value in {"legacy", "pipeline"}:
            return False
        return bool(self.deps.task_planner and self.deps.tool_orchestrator)

    def _build_graph(self):
        if StateGraph is None:
            raise RuntimeError(f"LangGraph is not available: {LANGGRAPH_IMPORT_ERROR}")

        workflow = StateGraph(ChatGraphState)
        workflow.add_node("prepare", self._prepare_node)
        workflow.add_node("route", self._route_node)
        for intent in self.GRAPH_INTENTS:
            workflow.add_node(intent, self._dispatch_node)
        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "route")
        workflow.add_conditional_edges(
            "route",
            self._route_to_intent_node,
            {intent: intent for intent in self.GRAPH_INTENTS},
        )
        for intent in self.GRAPH_INTENTS:
            workflow.add_edge(intent, END)
        return workflow.compile()

    def _prepare_node(self, state: ChatGraphState) -> ChatGraphState:
        data = state.get("raw_data") or {}
        user_id = state["user_id"]
        raw_message = data.get("message", "") or ""
        display_message = data.get("display_message") or raw_message
        mode = data.get("mode", "chat") or "chat"
        session_id = self.deps.memory.get_or_create_session(user_id, data.get("session_id"))
        remember_explicit = getattr(self.deps.memory, "remember_explicit_memory", None)
        if callable(remember_explicit):
            # Long-term memory is opt-in at the user-message boundary.  Raw
            # uploads and model output are deliberately never persisted here.
            remember_explicit(user_id, session_id, display_message)
        message = raw_message
        attachments = []

        attached_contents = []
        for file_id in data.get("file_ids", []) or []:
            content = self.deps.upload_manager.get_temp_content(file_id, user_id)
            if not content:
                continue
            attached_contents.append(f"[文件内容]\n{content}\n[/文件内容]")
            info = self.deps.upload_manager.get_temp_file_info(file_id, user_id) or {}
            filename = info.get("filename") or file_id
            attachments.append({
                "file_id": file_id,
                "filename": filename,
                "char_count": len(info.get("content") or content or ""),
                "is_spreadsheet": Path(filename).suffix.lower() in {".xlsx", ".xls", ".csv"},
            })

        if attached_contents:
            message = "\n\n".join(attached_contents) + "\n\n[用户提问]\n" + message

        return {
            "prepared": PreparedChatRequest(
                message=message,
                display_message=display_message,
                mode=mode,
                session_id=session_id,
                user_id=user_id,
                user_info=state.get("user_info"),
                user_metadata={"attached_files": attachments} if attachments else None,
                attachments=attachments,
            )
        }

    def _route_node(self, state: ChatGraphState) -> ChatGraphState:
        prepared = state["prepared"]
        conversation_context = self._conversation_context_for_request(prepared)
        has_last_document = bool(self.deps.memory.get_context(prepared.session_id, "last_document", "") or "")
        route = IntentRouter(
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
        return {"route": route}

    def _plan_tools_node(self, state: ChatGraphState) -> ChatGraphState:
        prepared = state["prepared"]
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
        return {"task_plan": plan, "route": plan.route}

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

    def _execute_tools_node(self, state: ChatGraphState) -> ChatGraphState:
        return {"stream": self.deps.tool_orchestrator.stream(state["prepared"], state["task_plan"])}

    def _route_to_intent_node(self, state: ChatGraphState) -> str:
        intent = getattr(state.get("route"), "intent", INTENT_KNOWLEDGE_QA)
        return intent if intent in self.GRAPH_INTENTS else INTENT_KNOWLEDGE_QA

    def _dispatch_node(self, state: ChatGraphState) -> ChatGraphState:
        prepared = state["prepared"]
        route = state.get("route")
        intent = getattr(route, "intent", INTENT_KNOWLEDGE_QA)
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
        return {"stream": stream}
