"""Pipeline dispatching for the unified chat endpoint.

Lightweight, deterministic chat flows live here so app.py can stay focused on
request parsing and Response wrapping. The large RAG and document flows are
still adapted through FunctionPipeline while their dependencies are gradually
split out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from chat_architecture import (
    INTENT_CLARIFY,
    INTENT_DOC_DRAFTING,
    INTENT_DOC_FORMATTING,
    INTENT_FORM_TEMPLATE_EXPORT,
    INTENT_IDENTITY_HELP,
    INTENT_KNOWLEDGE_QA,
    INTENT_SPREADSHEET_TRANSFORM,
)
from chat_events import (
    route_actions as _route_actions,
    route_event as _route_event,
    route_intent as _route_intent,
    route_payload as _route_payload,
    route_template_key as _route_template_key,
    sse as _sse,
)


@dataclass
class ChatPipelineRequest:
    message: str
    display_message: str
    session_id: str
    user_id: str
    user_info: Any = None
    user_metadata: Optional[dict] = None
    route: Any = None


@dataclass
class ChatPipelineRuntime:
    memory: Any
    reimbursement_template_files: Dict[str, str]
    assistant_identity_response: Callable[[], str]


class FunctionPipeline:
    def __init__(self, name: str, handler: Callable[[ChatPipelineRequest], Any]):
        self.name = name
        self.handler = handler

    def stream(self, request: ChatPipelineRequest):
        return self.handler(request)


class IdentityHelpPipeline:
    name = "IdentityHelpPipeline"

    def __init__(self, runtime: ChatPipelineRuntime):
        self.runtime = runtime

    def stream(self, request: ChatPipelineRequest):
        message = request.display_message
        session_id = request.session_id
        route = request.route
        response = self.runtime.assistant_identity_response()
        plan = {
            "document_type": "身份说明",
            "task_type": "问答检索",
            "need_web_search": False,
        }

        self.runtime.memory.add_message(session_id, "user", message)
        yield _sse({"type": "start"})
        yield _sse({"type": "session", "session_id": session_id})
        if route:
            yield _route_event(route)
        yield _sse({"type": "content", "data": response, "session_id": session_id})

        self.runtime.memory.add_message(session_id, "assistant", response, metadata={"type": "identity", "plan": plan})
        self.runtime.memory.set_context(session_id, "last_request", message)
        self.runtime.memory.set_context(session_id, "last_answer", response)
        self.runtime.memory.set_context(session_id, "last_answer_plan", plan)
        self.runtime.memory.update_rolling_summary(session_id, message, response, plan, [])

        yield _sse({
            "type": "done",
            "intent": _route_intent(route, INTENT_IDENTITY_HELP),
            "answer": response,
            "document": "",
            "session_id": session_id,
            "plan": plan,
            "route": _route_payload(route),
            "actions": _route_actions(route),
            "export_template": "",
            "export_spreadsheet_template": "",
            "audit_summary": {},
            "source_filenames": [],
            "source_details": [],
        })


class ClarifyPipeline:
    name = "ClarifyPipeline"

    def __init__(self, runtime: ChatPipelineRuntime):
        self.runtime = runtime

    def stream(self, request: ChatPipelineRequest):
        message = request.display_message
        session_id = request.session_id
        route = request.route
        payload = _route_payload(route)
        document_type = payload.get("document_type", "")
        reason = payload.get("reason", "")
        if document_type == "报销表单":
            response = "请说明要导出哪一种报销表：差旅费、会议费、劳务费&专家咨询费，还是其他费用报销。"
        elif document_type == "格式转换":
            response = "请先上传或粘贴需要转换的材料，并明确说明要改为公文格式。"
        else:
            response = "我需要再确认一下你的具体需求，请补充要查询、起草、转换或导出的对象。"
        plan = {
            "document_type": document_type or "澄清需求",
            "task_type": "澄清需求",
            "need_web_search": False,
        }

        self.runtime.memory.add_message(session_id, "user", message, metadata=request.user_metadata or {})
        yield _sse({"type": "start"})
        yield _sse({"type": "session", "session_id": session_id})
        if route:
            yield _route_event(route)
        yield _sse({"type": "think", "agent": "IntentRouter", "emoji": "🧭", "message": reason or "需要补充信息"})
        yield _sse({"type": "content", "data": response, "session_id": session_id})

        self.runtime.memory.add_message(session_id, "assistant", response, metadata={
            "type": INTENT_CLARIFY,
            "plan": plan,
            "route": payload,
        })
        self.runtime.memory.set_context(session_id, "last_request", message)
        self.runtime.memory.set_context(session_id, "last_answer", response)
        self.runtime.memory.set_context(session_id, "last_answer_plan", plan)
        self.runtime.memory.update_rolling_summary(session_id, message, response, plan, [])

        yield _sse({
            "type": "done",
            "intent": INTENT_CLARIFY,
            "answer": response,
            "document": "",
            "session_id": session_id,
            "plan": plan,
            "route": payload,
            "actions": _route_actions(route),
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": [],
            "source_details": [],
            "audit_summary": {},
        })


class FormExportPipeline:
    name = "FormExportPipeline"

    def __init__(self, runtime: ChatPipelineRuntime):
        self.runtime = runtime

    def stream(self, request: ChatPipelineRequest):
        message = request.display_message
        session_id = request.session_id
        route = request.route
        template_key = _route_template_key(route)
        filename = self.runtime.reimbursement_template_files.get(template_key, "报销表.xlsx")
        action_label = (_route_actions(route) or [{"label": "导出报销表"}])[0].get("label", "导出报销表")
        response = f"已识别为报销表单导出需求，请点击下方“{action_label}”下载公共资料模板《{filename}》。"
        plan = {
            "document_type": "报销表单",
            "task_type": "表单导出",
            "need_web_search": False,
        }

        self.runtime.memory.add_message(session_id, "user", message, metadata=request.user_metadata or {})
        yield _sse({"type": "start"})
        yield _sse({"type": "session", "session_id": session_id})
        yield _route_event(route)
        yield _sse({"type": "think", "agent": "IntentRouter", "emoji": "🧭", "message": "已识别报销表单导出需求"})
        yield _sse({"type": "content", "data": response, "session_id": session_id})

        source_filenames = [filename]
        source_details = [{"filename": filename}]
        self.runtime.memory.add_message(session_id, "assistant", response, metadata={
            "type": INTENT_FORM_TEMPLATE_EXPORT,
            "plan": plan,
            "route": _route_payload(route),
            "actions": _route_actions(route),
            "source_filenames": source_filenames,
        })
        self.runtime.memory.set_context(session_id, "last_request", message)
        self.runtime.memory.set_context(session_id, "last_answer", response)
        self.runtime.memory.set_context(session_id, "last_answer_plan", plan)
        self.runtime.memory.update_rolling_summary(session_id, message, response, plan, source_filenames)

        yield _sse({
            "type": "done",
            "intent": INTENT_FORM_TEMPLATE_EXPORT,
            "answer": response,
            "document": "",
            "session_id": session_id,
            "plan": plan,
            "route": _route_payload(route),
            "actions": _route_actions(route),
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": source_filenames,
            "source_details": source_details,
            "audit_summary": {},
        })


class SpreadsheetTransformPipeline:
    name = "SpreadsheetTransformPipeline"

    def __init__(self, runtime: ChatPipelineRuntime):
        self.runtime = runtime

    def stream(self, request: ChatPipelineRequest):
        message = request.display_message
        session_id = request.session_id
        route = request.route
        actions = _route_actions(route)
        filename = actions[0].get("filename", "表格.xlsx") if actions else "表格.xlsx"
        response = f"已识别为表格处理需求：将按你的规则处理《{filename}》。请点击下方“处理并导出表格”执行并下载结果。"
        plan = {
            "document_type": "表格处理",
            "task_type": "表格处理",
            "need_web_search": False,
        }

        self.runtime.memory.add_message(session_id, "user", message, metadata=request.user_metadata or {})
        yield _sse({"type": "start"})
        yield _sse({"type": "session", "session_id": session_id})
        yield _route_event(route)
        yield _sse({"type": "think", "agent": "IntentRouter", "emoji": "🧭", "message": "已识别上传表格处理需求"})
        yield _sse({"type": "content", "data": response, "session_id": session_id})

        self.runtime.memory.add_message(session_id, "assistant", response, metadata={
            "type": INTENT_SPREADSHEET_TRANSFORM,
            "plan": plan,
            "route": _route_payload(route),
            "actions": actions,
        })
        self.runtime.memory.set_context(session_id, "last_request", message)
        self.runtime.memory.set_context(session_id, "last_answer", response)
        self.runtime.memory.set_context(session_id, "last_answer_plan", plan)
        self.runtime.memory.update_rolling_summary(session_id, message, response, plan, [])

        yield _sse({
            "type": "done",
            "intent": INTENT_SPREADSHEET_TRANSFORM,
            "answer": response,
            "document": "",
            "session_id": session_id,
            "plan": plan,
            "route": _route_payload(route),
            "actions": actions,
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": [],
            "source_details": [],
            "audit_summary": {},
        })


class ChatPipelineDispatcher:
    def __init__(
        self,
        *,
        runtime: ChatPipelineRuntime,
        document_format_stream: Callable,
        document_draft_stream: Callable,
        rag_qa_stream: Callable,
    ):
        self.default_intent = INTENT_KNOWLEDGE_QA
        self.pipelines: Dict[str, Any] = {
            INTENT_IDENTITY_HELP: IdentityHelpPipeline(runtime),
            INTENT_CLARIFY: ClarifyPipeline(runtime),
            INTENT_FORM_TEMPLATE_EXPORT: FormExportPipeline(runtime),
            INTENT_SPREADSHEET_TRANSFORM: SpreadsheetTransformPipeline(runtime),
            INTENT_DOC_FORMATTING: FunctionPipeline(
                "DocumentFormatPipeline",
                lambda req: document_format_stream(
                    req.message,
                    req.session_id,
                    req.user_id,
                    req.user_info,
                    req.display_message,
                    req.user_metadata,
                    req.route,
                ),
            ),
            INTENT_DOC_DRAFTING: FunctionPipeline(
                "DocumentDraftPipeline",
                lambda req: document_draft_stream(
                    req.message,
                    req.session_id,
                    req.user_id,
                    req.user_info,
                    req.display_message,
                    req.user_metadata,
                    req.route,
                ),
            ),
            INTENT_KNOWLEDGE_QA: FunctionPipeline(
                "RagQaPipeline",
                lambda req: rag_qa_stream(
                    req.message,
                    req.session_id,
                    req.user_id,
                    req.user_info,
                    req.display_message,
                    req.user_metadata,
                    req.route,
                ),
            ),
        }

    def pipeline_for(self, intent: str) -> FunctionPipeline:
        return self.pipelines.get(intent) or self.pipelines[self.default_intent]

    def stream(self, request: ChatPipelineRequest):
        intent = getattr(request.route, "intent", self.default_intent)
        return self.pipeline_for(intent).stream(request)
