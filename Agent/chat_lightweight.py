"""Deterministic lightweight chat intent streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from chat_architecture import (
    INTENT_CLARIFY,
    INTENT_DOC_FORMATTING,
    INTENT_FORM_TEMPLATE_EXPORT,
    INTENT_IDENTITY_HELP,
    INTENT_SPREADSHEET_TRANSFORM,
)
from chat_events import route_actions, route_event, route_intent, route_payload, route_template_key, sse, text_stream_sse


def assistant_identity_response() -> str:
    return """我是智能知识库助手，面向智能知识库平台的智能知识库平台提供服务。

我可以帮助你：
1. 检索本地知识库资料，围绕制度文件、历史材料、业务文档进行问答和归纳。
2. 根据知识库内容整理依据、提炼要点，并尽量说明参考来源。
3. 起草和修改公文，包括通知、请示、报告、函、建议等常见文种。
4. 处理上传材料，支持摘要、润色、改写、续写、格式转换和结构整理。
5. 在多轮对话中延续上下文，围绕上一版内容继续补充、调整或重写。
6. 将生成内容导入编辑器，并配合导出 Word 文档。

你可以直接告诉我你的目标，例如“帮我检索某份制度文件的要点”“根据这些材料写一份通知”“把上一版内容再压缩一些”。"""


@dataclass
class LightweightChatDependencies:
    memory: Any
    reimbursement_template_files: dict[str, str]
    assistant_identity_response: Callable[[], str] = assistant_identity_response


class LightweightChatStreamService:
    def __init__(self, deps: LightweightChatDependencies):
        self.deps = deps

    def stream(
        self,
        message,
        session_id,
        user_id=None,
        user_info=None,
        display_message=None,
        user_metadata: Optional[dict] = None,
        route=None,
    ):
        intent = route_intent(route, "")
        display = display_message or message
        if intent == INTENT_IDENTITY_HELP:
            yield from self._identity(display, session_id, route)
        elif intent == INTENT_CLARIFY:
            yield from self._clarify(display, session_id, route, user_metadata)
        elif intent == INTENT_FORM_TEMPLATE_EXPORT:
            yield from self._form_export(display, session_id, route, user_metadata)
        elif intent == INTENT_SPREADSHEET_TRANSFORM:
            yield from self._spreadsheet_transform(display, session_id, route, user_metadata)
        else:
            yield sse({"type": "error", "message": "暂不支持的轻量意图"})

    def _identity(self, message, session_id, route=None):
        response = self.deps.assistant_identity_response()
        plan = {
            "document_type": "身份说明",
            "task_type": "问答检索",
            "need_web_search": False,
        }

        self.deps.memory.add_message(session_id, "user", message)
        yield sse({"type": "start"})
        yield sse({"type": "session", "session_id": session_id})
        if route:
            yield route_event(route)
        yield sse({"type": "thinking_start", "message": "正在准备身份说明"})
        yield sse({"type": "thinking_done", "summary": "已完成身份说明整理"})
        yield sse({"type": "answer_start", "message": "开始输出正文", "session_id": session_id})
        yield from text_stream_sse(response, session_id=session_id)
        yield sse({"type": "answer_done", "answer": response, "session_id": session_id})

        self.deps.memory.add_message(session_id, "assistant", response, metadata={"type": "identity", "plan": plan})
        self.deps.memory.set_context(session_id, "last_request", message)
        self.deps.memory.set_context(session_id, "last_answer", response)
        self.deps.memory.set_context(session_id, "last_answer_plan", plan)
        self.deps.memory.update_rolling_summary(session_id, message, response, plan, [])

        yield sse({"type": "run_done", "session_id": session_id, "intent": route_intent(route, INTENT_IDENTITY_HELP)})
        yield sse({
            "type": "done",
            "intent": route_intent(route, INTENT_IDENTITY_HELP),
            "answer": response,
            "document": "",
            "session_id": session_id,
            "plan": plan,
            "route": route_payload(route),
            "actions": route_actions(route),
            "export_template": "",
            "export_spreadsheet_template": "",
            "audit_summary": {},
            "source_filenames": [],
            "source_details": [],
        })

    def _clarify(self, message, session_id, route=None, user_metadata=None):
        payload = route_payload(route)
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

        self.deps.memory.add_message(session_id, "user", message, metadata=user_metadata or {})
        yield sse({"type": "start"})
        yield sse({"type": "session", "session_id": session_id})
        if route:
            yield route_event(route)
        yield sse({"type": "thinking_start", "message": "正在判断需要补充的信息"})
        yield sse({"type": "think", "agent": "IntentRouter", "emoji": "🧭", "message": reason or "需要补充信息"})
        yield sse({"type": "thinking_done", "summary": "已确定需要澄清的问题"})
        yield sse({"type": "answer_start", "message": "开始输出正文", "session_id": session_id})
        yield from text_stream_sse(response, session_id=session_id)
        yield sse({"type": "answer_done", "answer": response, "session_id": session_id})

        self.deps.memory.add_message(session_id, "assistant", response, metadata={
            "type": INTENT_CLARIFY,
            "plan": plan,
            "route": payload,
        })
        self.deps.memory.set_context(session_id, "last_request", message)
        self.deps.memory.set_context(session_id, "last_answer", response)
        self.deps.memory.set_context(session_id, "last_answer_plan", plan)
        self.deps.memory.update_rolling_summary(session_id, message, response, plan, [])

        yield sse({"type": "run_done", "session_id": session_id, "intent": INTENT_CLARIFY})
        yield sse({
            "type": "done",
            "intent": INTENT_CLARIFY,
            "answer": response,
            "document": "",
            "session_id": session_id,
            "plan": plan,
            "route": payload,
            "actions": route_actions(route),
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": [],
            "source_details": [],
            "audit_summary": {},
        })

    def _form_export(self, message, session_id, route=None, user_metadata=None):
        template_key = route_template_key(route)
        filename = self.deps.reimbursement_template_files.get(template_key, "报销表.xlsx")
        action_label = (route_actions(route) or [{"label": "导出报销表"}])[0].get("label", "导出报销表")
        response = f"已识别为报销表单导出需求，请点击下方“{action_label}”下载公共资料模板《{filename}》。"
        plan = {
            "document_type": "报销表单",
            "task_type": "表单导出",
            "need_web_search": False,
        }

        self.deps.memory.add_message(session_id, "user", message, metadata=user_metadata or {})
        yield sse({"type": "start"})
        yield sse({"type": "session", "session_id": session_id})
        yield route_event(route)
        yield sse({"type": "thinking_start", "message": "正在识别可导出的模板"})
        yield sse({"type": "think", "agent": "IntentRouter", "emoji": "🧭", "message": "已识别报销表单导出需求"})
        yield sse({"type": "thinking_done", "summary": "已匹配报销表单模板"})
        yield sse({"type": "answer_start", "message": "开始输出正文", "session_id": session_id})
        yield from text_stream_sse(response, session_id=session_id)
        yield sse({"type": "answer_done", "answer": response, "session_id": session_id})

        source_filenames = [filename]
        source_details = [{"filename": filename}]
        self.deps.memory.add_message(session_id, "assistant", response, metadata={
            "type": INTENT_FORM_TEMPLATE_EXPORT,
            "plan": plan,
            "route": route_payload(route),
            "actions": route_actions(route),
            "source_filenames": source_filenames,
        })
        self.deps.memory.set_context(session_id, "last_request", message)
        self.deps.memory.set_context(session_id, "last_answer", response)
        self.deps.memory.set_context(session_id, "last_answer_plan", plan)
        self.deps.memory.update_rolling_summary(session_id, message, response, plan, source_filenames)

        yield sse({"type": "run_done", "session_id": session_id, "intent": INTENT_FORM_TEMPLATE_EXPORT})
        yield sse({
            "type": "done",
            "intent": INTENT_FORM_TEMPLATE_EXPORT,
            "answer": response,
            "document": "",
            "session_id": session_id,
            "plan": plan,
            "route": route_payload(route),
            "actions": route_actions(route),
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": source_filenames,
            "source_details": source_details,
            "audit_summary": {},
        })

    def _spreadsheet_transform(self, message, session_id, route=None, user_metadata=None):
        actions = route_actions(route)
        filename = actions[0].get("filename", "表格.xlsx") if actions else "表格.xlsx"
        response = f"已识别为表格处理需求：将按你的规则处理《{filename}》。请点击下方“处理并导出表格”执行并下载结果。"
        plan = {
            "document_type": "表格处理",
            "task_type": "表格处理",
            "need_web_search": False,
        }

        self.deps.memory.add_message(session_id, "user", message, metadata=user_metadata or {})
        yield sse({"type": "start"})
        yield sse({"type": "session", "session_id": session_id})
        yield route_event(route)
        yield sse({"type": "thinking_start", "message": "正在识别表格处理动作"})
        yield sse({"type": "think", "agent": "IntentRouter", "emoji": "🧭", "message": "已识别上传表格处理需求"})
        yield sse({"type": "thinking_done", "summary": "已准备表格处理动作"})
        yield sse({"type": "answer_start", "message": "开始输出正文", "session_id": session_id})
        yield from text_stream_sse(response, session_id=session_id)
        yield sse({"type": "answer_done", "answer": response, "session_id": session_id})

        self.deps.memory.add_message(session_id, "assistant", response, metadata={
            "type": INTENT_SPREADSHEET_TRANSFORM,
            "plan": plan,
            "route": route_payload(route),
            "actions": actions,
        })
        self.deps.memory.set_context(session_id, "last_request", message)
        self.deps.memory.set_context(session_id, "last_answer", response)
        self.deps.memory.set_context(session_id, "last_answer_plan", plan)
        self.deps.memory.update_rolling_summary(session_id, message, response, plan, [])

        yield sse({"type": "run_done", "session_id": session_id, "intent": INTENT_SPREADSHEET_TRANSFORM})
        yield sse({
            "type": "done",
            "intent": INTENT_SPREADSHEET_TRANSFORM,
            "answer": response,
            "document": "",
            "session_id": session_id,
            "plan": plan,
            "route": route_payload(route),
            "actions": actions,
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": [],
            "source_details": [],
            "audit_summary": {},
        })
