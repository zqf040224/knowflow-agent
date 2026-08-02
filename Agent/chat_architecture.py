"""
Intent routing and unified chat action contracts for the RAG workbench.

This module is deliberately free of Flask, vector-store and LLM dependencies so
the product boundary can be tested without booting the full app.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


INTENT_KNOWLEDGE_QA = "knowledge_qa"
INTENT_DOC_DRAFTING = "doc_drafting"
INTENT_DOC_FORMATTING = "doc_formatting"
INTENT_FORM_TEMPLATE_EXPORT = "form_template_export"
INTENT_SPREADSHEET_TRANSFORM = "spreadsheet_transform"
INTENT_IDENTITY_HELP = "identity_help"
INTENT_CLARIFY = "clarify"
VALID_INTENTS = {
    INTENT_KNOWLEDGE_QA,
    INTENT_DOC_DRAFTING,
    INTENT_DOC_FORMATTING,
    INTENT_FORM_TEMPLATE_EXPORT,
    INTENT_SPREADSHEET_TRANSFORM,
    INTENT_IDENTITY_HELP,
    INTENT_CLARIFY,
}

DOC_FORMAT_MARKERS = (
    "改为公文格式",
    "改成公文格式",
    "改成公文",
    "转成公文",
    "套用公文",
    "规范公文",
    "公文格式",
    "按公文",
    "正式公文",
)
DOC_FORMAT_COMMAND_MARKERS = (
    "改为公文格式",
    "改成公文格式",
    "改成公文",
    "转成公文",
    "套用公文",
    "规范公文",
    "按公文",
    "正式公文",
)

DOC_GENERATION_VERBS = (
    "写一份", "写一篇", "写篇", "写一个", "写个", "写一版",
    "起草", "拟一份", "拟写", "生成", "撰写", "出一份", "帮我写",
)
DOC_SOFT_GENERATION_VERBS = ("做一份", "做一个", "弄一版", "出个", "出一版", "整理一份", "拟定", "草拟", "准备一份", "形成一份")
DOC_NOUNS = (
    "通知",
    "请示",
    "报告",
    "函",
    "议案",
    "审议",
    "会议纪要",
    "方案",
    "制度",
    "办法",
    "公文",
)
DOC_CONTEXT_NOUNS = DOC_NOUNS + ("材料", "文档", "正文")
DOC_OUTPUT_NOUNS = DOC_CONTEXT_NOUNS + (
    "文稿",
    "初稿",
    "稿子",
    "提纲",
    "大纲",
    "发言稿",
    "讲话稿",
    "汇报稿",
    "调研材料",
    "研究材料",
)
DOC_GUIDANCE_MARKERS = (
    "怎么写",
    "如何写",
    "怎样写",
    "写法",
    "格式要求",
    "模板",
    "范文",
    "主要讲什么",
    "主要内容",
    "需要哪些材料",
    "准备什么材料",
    "要准备什么",
    "需要准备什么",
    "材料有哪些",
)
REVISION_MARKERS = (
    "上一条",
    "上一版",
    "上一个",
    "刚才",
    "基于上",
    "继续",
    "再改",
    "修改",
    "改写",
    "润色",
    "压缩",
    "改短",
    "精简",
    "更正式",
    "补充",
    "重写",
)

EXPORT_MARKERS = ("导出", "下载", "生成", "给我", "我要", "发我", "提供", "出个")
FORM_EXPORT_MARKERS = ("报销表", "报销单", "审批单", "模板", "表格", "xlsx", "excel")
SPREADSHEET_OPERATION_PATTERN = re.compile(
    r"(排序|排列|筛选|过滤|导出|降序|升序|从高到低|从低到高|大于|小于|等于|包含|分组|前\s*\d+\s*(条|行|名|个)?)"
)
WEAK_DOCUMENT_SIGNAL_MARKERS = DOC_CONTEXT_NOUNS + (
    "word",
    "docx",
    "正式",
    "规范",
    "办公会",
)
WEAK_FORM_SIGNAL_MARKERS = FORM_EXPORT_MARKERS + ("报销", "费用", "审批")
WEAK_SPREADSHEET_SIGNAL_MARKERS = ("表格", "excel", "xlsx", "xls", "csv", "sheet")


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()


def is_identity_query(message: str) -> bool:
    text = compact_text(message)
    if not text or len(text) > 80:
        return False
    patterns = (
        "你是谁", "你是什么", "你叫啥", "你叫什么", "介绍一下你", "介绍下你",
        "你能做什么", "你可以做什么", "你会做什么", "你的功能", "有什么功能",
        "你有什么用", "你能帮我做什么", "你可以帮我做什么", "你是干什么的",
        "whoareyou", "whatareyou", "whatcanyoudo",
    )
    return any(pattern in text for pattern in patterns)


def explicit_document_request(message: str) -> bool:
    text = compact_text(message)
    if not text:
        return False
    if any(marker in text for marker in DOC_GUIDANCE_MARKERS):
        return False
    if any(verb in text for verb in DOC_GENERATION_VERBS) and any(noun in text for noun in DOC_OUTPUT_NOUNS):
        return True
    if any(verb in text for verb in DOC_SOFT_GENERATION_VERBS) and any(noun in text for noun in DOC_CONTEXT_NOUNS):
        return True
    if text.startswith(("请写", "写")) and any(noun in text for noun in DOC_OUTPUT_NOUNS):
        return True
    return any(marker in text for marker in ("导出word", "导出docx", "生成word", "生成docx"))


def explicit_document_formatting(message: str) -> bool:
    text = compact_text(message)
    return any(marker in text for marker in DOC_FORMAT_MARKERS)


def document_formatting_needs_material(message: str) -> bool:
    text = compact_text(message)
    return any(marker in text for marker in DOC_FORMAT_COMMAND_MARKERS)


def document_revision_request(message: str, has_last_document: bool) -> bool:
    if not has_last_document:
        return False
    text = compact_text(message)
    return any(marker in text for marker in REVISION_MARKERS)


def is_spreadsheet_attachment(item: Dict) -> bool:
    filename = str(item.get("filename") or "")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return bool(item.get("is_spreadsheet")) or ext in {"xlsx", "xls", "csv"}


def wants_form_template_export(message: str, template_key: str) -> bool:
    text = compact_text(message)
    if not template_key:
        return False
    has_export_marker = any(marker in text for marker in EXPORT_MARKERS) or ("需要" in text and "模板" in text)
    return has_export_marker and any(marker in text for marker in FORM_EXPORT_MARKERS)


def wants_ambiguous_form_template_export(message: str, template_key: str) -> bool:
    text = compact_text(message)
    if template_key:
        return False
    has_export_marker = any(marker in text for marker in EXPORT_MARKERS) or ("需要" in text and "模板" in text)
    return has_export_marker and any(marker in text for marker in FORM_EXPORT_MARKERS)


def document_type_for(message: str) -> str:
    text = compact_text(message)
    if any(marker in text for marker in ("议案", "审议", "办公会")):
        return "会议审议案"
    if "提纲" in text or "大纲" in text:
        return "材料提纲"
    if "发言稿" in text or "讲话稿" in text:
        return "发言稿"
    for noun in ("通知", "请示", "报告", "函", "会议纪要", "方案", "制度", "办法"):
        if noun in text:
            return noun
    if "材料" in text:
        return "材料"
    return "通用公文"


def has_weak_document_signal(message: str) -> bool:
    text = compact_text(message)
    return any(marker in text for marker in WEAK_DOCUMENT_SIGNAL_MARKERS)


def has_weak_form_signal(message: str) -> bool:
    text = compact_text(message)
    return any(marker in text for marker in WEAK_FORM_SIGNAL_MARKERS)


def has_weak_spreadsheet_signal(message: str, attachments: List[Dict]) -> bool:
    text = compact_text(message)
    if any(is_spreadsheet_attachment(item) for item in attachments):
        return True
    return any(marker in text for marker in WEAK_SPREADSHEET_SIGNAL_MARKERS)


@dataclass
class RouteResult:
    intent: str
    confidence: float
    reason: str
    document_type: str = ""
    template_key: str = ""
    requires_retrieval: bool = True
    actions: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reason": self.reason,
            "document_type": self.document_type,
            "template_key": self.template_key,
            "requires_retrieval": self.requires_retrieval,
            "actions": self.actions,
        }


class IntentRouter:
    def __init__(
        self,
        reimbursement_detector: Optional[Callable[[str, str], str]] = None,
        intent_classifier: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self.reimbursement_detector = reimbursement_detector or (lambda text, requested="auto": "")
        self.intent_classifier = intent_classifier

    def route(
        self,
        *,
        message: str,
        display_message: str = "",
        mode: str = "quick",
        attachments: Optional[List[Dict]] = None,
        conversation_context: str = "",
        has_last_document: bool = False,
    ) -> RouteResult:
        user_message = display_message or message or ""
        attachments = attachments or []
        result = self._route_by_rules(
            user_message=user_message,
            attachments=attachments,
            has_last_document=has_last_document,
        )
        return self._with_classifier_fallback(
            result=result,
            user_message=user_message,
            attachments=attachments,
            conversation_context=conversation_context,
            has_last_document=has_last_document,
        )

    def _route_by_rules(
        self,
        *,
        user_message: str,
        attachments: List[Dict],
        has_last_document: bool,
    ) -> RouteResult:
        has_attachment = bool(attachments)
        spreadsheet_files = [item for item in attachments if is_spreadsheet_attachment(item)]
        template_key = self.reimbursement_detector(user_message, "auto") or ""

        if is_identity_query(user_message) and not has_attachment:
            return RouteResult(
                intent=INTENT_IDENTITY_HELP,
                confidence=0.99,
                reason="用户询问助手身份或能力",
                document_type="身份说明",
                requires_retrieval=False,
            )

        if spreadsheet_files and SPREADSHEET_OPERATION_PATTERN.search(user_message):
            file_info = spreadsheet_files[0]
            return RouteResult(
                intent=INTENT_SPREADSHEET_TRANSFORM,
                confidence=0.94,
                reason="用户对上传表格提出筛选、排序或导出处理需求",
                document_type="表格处理",
                requires_retrieval=False,
                actions=[{
                    "type": "spreadsheet_transform",
                    "label": "处理并导出表格",
                    "file_id": file_info.get("file_id", ""),
                    "filename": file_info.get("filename", "表格.xlsx"),
                    "instruction": user_message,
                }],
            )

        if wants_form_template_export(user_message, template_key):
            labels = {
                "travel": "导出差旅费报销表",
                "meeting": "导出会议费报销表",
                "labor_expert": "导出劳务费&专家咨询费报销表",
                "other": "导出其他费用报销表",
            }
            return RouteResult(
                intent=INTENT_FORM_TEMPLATE_EXPORT,
                confidence=0.96,
                reason="用户明确要求导出报销类表单模板",
                document_type="报销表单",
                template_key=template_key,
                requires_retrieval=False,
                actions=[{
                    "type": "export_xlsx_template",
                    "label": labels.get(template_key, "导出报销表"),
                    "template_key": template_key,
                }],
            )

        if wants_ambiguous_form_template_export(user_message, template_key):
            return RouteResult(
                intent=INTENT_CLARIFY,
                confidence=0.74,
                reason="用户想导出报销表单，但未说明具体模板类型",
                document_type="报销表单",
                requires_retrieval=False,
            )

        if explicit_document_formatting(user_message) and has_attachment:
            return RouteResult(
                intent=INTENT_DOC_FORMATTING,
                confidence=0.92,
                reason="用户上传附件并明确要求套用公文格式",
                document_type="格式转换",
                requires_retrieval=True,
            )

        if document_formatting_needs_material(user_message) and not has_attachment:
            return RouteResult(
                intent=INTENT_CLARIFY,
                confidence=0.76,
                reason="用户要求套用公文格式，但未提供待转换材料",
                document_type="格式转换",
                requires_retrieval=False,
            )

        if document_revision_request(user_message, has_last_document):
            return RouteResult(
                intent=INTENT_DOC_DRAFTING,
                confidence=0.88,
                reason="用户基于上一版公文继续修改",
                document_type="续写修改",
                requires_retrieval=True,
            )

        if explicit_document_request(user_message):
            return RouteResult(
                intent=INTENT_DOC_DRAFTING,
                confidence=0.9,
                reason="用户明确要求起草或生成公文类文档",
                document_type=document_type_for(user_message),
                requires_retrieval=True,
            )

        return RouteResult(
            intent=INTENT_KNOWLEDGE_QA,
            confidence=0.82,
            reason="未命中文档生成或工具动作，按知识库客服问答处理",
            document_type="知识库问答",
            template_key=template_key,
            requires_retrieval=True,
        )

    def _with_classifier_fallback(
        self,
        *,
        result: RouteResult,
        user_message: str,
        attachments: List[Dict],
        conversation_context: str,
        has_last_document: bool,
    ) -> RouteResult:
        if not self.intent_classifier or not self._should_call_classifier(result, user_message, attachments):
            return result
        payload = {
            "message": user_message,
            "attachments": attachments,
            "conversation_context": conversation_context,
            "has_last_document": has_last_document,
            "rule_result": result.to_dict(),
            "allowed_intents": sorted(VALID_INTENTS),
        }
        try:
            raw = self.intent_classifier(payload)
            classified = self._normalize_classifier_result(raw, result, user_message, attachments)
            return classified or result
        except Exception:
            return result

    def _should_call_classifier(self, result: RouteResult, user_message: str, attachments: List[Dict]) -> bool:
        if result.intent == INTENT_CLARIFY:
            return True
        if result.intent != INTENT_KNOWLEDGE_QA:
            return False
        return (
            result.confidence < 0.85
            and (
                has_weak_document_signal(user_message)
                or has_weak_form_signal(user_message)
                or has_weak_spreadsheet_signal(user_message, attachments)
            )
        )

    def _normalize_classifier_result(
        self,
        raw: Any,
        fallback: RouteResult,
        user_message: str,
        attachments: List[Dict],
    ) -> Optional[RouteResult]:
        if raw is None:
            return None
        if isinstance(raw, RouteResult):
            candidate = raw
        elif isinstance(raw, dict):
            intent = str(raw.get("intent") or "").strip()
            if intent not in VALID_INTENTS:
                return None
            try:
                confidence = float(raw.get("confidence", fallback.confidence))
            except (TypeError, ValueError):
                confidence = fallback.confidence
            document_type = str(raw.get("document_type") or "").strip()
            template_key = str(raw.get("template_key") or "").strip()
            candidate = RouteResult(
                intent=intent,
                confidence=max(0.0, min(confidence, 0.95)),
                reason=str(raw.get("reason") or "LLM intent classifier fallback").strip(),
                document_type=document_type,
                template_key=template_key,
                requires_retrieval=intent in {
                    INTENT_KNOWLEDGE_QA,
                    INTENT_DOC_DRAFTING,
                    INTENT_DOC_FORMATTING,
                },
            )
        else:
            return None

        if candidate.intent not in VALID_INTENTS:
            return None
        if candidate.intent == INTENT_FORM_TEMPLATE_EXPORT and not candidate.template_key:
            return None
        if candidate.intent == INTENT_DOC_FORMATTING and not attachments:
            return None
        if candidate.intent == INTENT_SPREADSHEET_TRANSFORM and not any(is_spreadsheet_attachment(item) for item in attachments):
            return None

        candidate.requires_retrieval = candidate.intent in {
            INTENT_KNOWLEDGE_QA,
            INTENT_DOC_DRAFTING,
            INTENT_DOC_FORMATTING,
        }
        if not candidate.document_type:
            candidate.document_type = self._default_document_type(candidate.intent, user_message, fallback)
        candidate.actions = self._classifier_actions(candidate.intent, candidate.template_key, user_message, attachments)
        return candidate

    def _default_document_type(self, intent: str, user_message: str, fallback: RouteResult) -> str:
        if intent == INTENT_DOC_DRAFTING:
            return document_type_for(user_message)
        if intent == INTENT_DOC_FORMATTING:
            return "格式转换"
        if intent == INTENT_FORM_TEMPLATE_EXPORT:
            return "报销表单"
        if intent == INTENT_SPREADSHEET_TRANSFORM:
            return "表格处理"
        if intent == INTENT_IDENTITY_HELP:
            return "身份说明"
        if intent == INTENT_CLARIFY:
            return fallback.document_type or "澄清需求"
        return "知识库问答"

    def _classifier_actions(
        self,
        intent: str,
        template_key: str,
        user_message: str,
        attachments: List[Dict],
    ) -> List[Dict]:
        if intent == INTENT_FORM_TEMPLATE_EXPORT and template_key:
            labels = {
                "travel": "导出差旅费报销表",
                "meeting": "导出会议费报销表",
                "labor_expert": "导出劳务费&专家咨询费报销表",
                "other": "导出其他费用报销表",
            }
            return [{
                "type": "export_xlsx_template",
                "label": labels.get(template_key, "导出报销表"),
                "template_key": template_key,
            }]
        if intent == INTENT_SPREADSHEET_TRANSFORM:
            spreadsheet_files = [item for item in attachments if is_spreadsheet_attachment(item)]
            if spreadsheet_files:
                file_info = spreadsheet_files[0]
                return [{
                    "type": "spreadsheet_transform",
                    "label": "处理并导出表格",
                    "file_id": file_info.get("file_id", ""),
                    "filename": file_info.get("filename", "表格.xlsx"),
                    "instruction": user_message,
                }]
        return []
