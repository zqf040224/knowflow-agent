from chat_architecture import (
    INTENT_CLARIFY,
    INTENT_DOC_DRAFTING,
    INTENT_DOC_FORMATTING,
    INTENT_FORM_TEMPLATE_EXPORT,
    INTENT_KNOWLEDGE_QA,
    INTENT_SPREADSHEET_TRANSFORM,
    IntentRouter,
)


def reimbursement_detector(text, requested="auto"):
    compact = "".join(str(text or "").split())
    if "差旅费" in compact:
        return "travel"
    if "会议费" in compact:
        return "meeting"
    if "劳务费" in compact or "专家咨询费" in compact:
        return "labor_expert"
    if "其他费用" in compact:
        return "other"
    return ""


def route(message, attachments=None, has_last_document=False):
    return IntentRouter(reimbursement_detector).route(
        message=message,
        display_message=message,
        mode="quick",
        attachments=attachments or [],
        has_last_document=has_last_document,
    )


def test_knowledge_qa_is_default_for_nas_usage():
    result = route("NAS服务器如何使用？")
    assert result.intent == INTENT_KNOWLEDGE_QA
    assert result.requires_retrieval is True


def test_knowledge_qa_is_default_for_center_file_questions():
    questions = [
        "场地使用收费标准在哪里看？",
        "会议费报销需要准备什么材料？",
        "劳务费和专家咨询费有什么要求？",
        "办公场地出租出借怎么申请？",
        "存储服务器运营方案主要讲什么？",
    ]
    for question in questions:
        result = route(question)
        assert result.intent == INTENT_KNOWLEDGE_QA, question
        assert result.actions == []


def test_reimbursement_question_stays_knowledge_qa():
    result = route("差旅费报销需要填哪些信息？")
    assert result.intent == INTENT_KNOWLEDGE_QA
    assert result.template_key == "travel"
    assert result.actions == []
    result = route("差旅费报销表需要填哪些信息？")
    assert result.intent == INTENT_KNOWLEDGE_QA
    assert result.template_key == "travel"
    assert result.actions == []


def test_form_template_export_action():
    result = route("我要导出差旅费报销表")
    assert result.intent == INTENT_FORM_TEMPLATE_EXPORT
    assert result.template_key == "travel"
    assert result.actions[0]["type"] == "export_xlsx_template"


def test_all_reimbursement_template_export_actions():
    cases = {
        "我要导出会议费报销表": "meeting",
        "下载劳务费审批单模板": "labor_expert",
        "给我专家咨询费报销表": "labor_expert",
        "需要其他费用报销表模板": "other",
    }
    for question, expected_template in cases.items():
        result = route(question)
        assert result.intent == INTENT_FORM_TEMPLATE_EXPORT, question
        assert result.template_key == expected_template
        assert result.actions[0]["type"] == "export_xlsx_template"


def test_document_drafting_is_explicit_only():
    result = route("帮我写一份关于NAS培训的通知")
    assert result.intent == INTENT_DOC_DRAFTING
    assert result.document_type == "通知"


def test_new_report_generation_routes_to_drafting():
    cases = [
        "写一篇新的报告",
        "请写一个关于低空经济的研究报告",
        "写个关于某市科技创新的报告",
        "帮我写一篇关于示例合作区的调研材料",
        "起草一份办公会汇报稿",
    ]
    for message in cases:
        result = route(message)
        assert result.intent == INTENT_DOC_DRAFTING, message


def test_outline_and_speech_generation_routes_to_drafting():
    cases = {
        "帮我写一个调研提纲": "材料提纲",
        "请写一版会议发言稿": "发言稿",
        "拟写一个交流讲话稿": "发言稿",
    }
    for message, document_type in cases.items():
        result = route(message)
        assert result.intent == INTENT_DOC_DRAFTING, message
        assert result.document_type == document_type


def test_report_writing_guidance_stays_knowledge_qa():
    cases = [
        "报告怎么写？",
        "有没有报告模板？",
        "给我看一篇报告范文",
        "这个报告主要讲什么？",
        "写报告需要哪些材料？",
        "写报告要准备什么材料？",
    ]
    for message in cases:
        result = route(message)
        assert result.intent == INTENT_KNOWLEDGE_QA, message


def test_soft_document_drafting_phrases_are_supported():
    result = route("帮我弄一版办公会审议材料")
    assert result.intent == INTENT_DOC_DRAFTING
    assert result.document_type == "会议审议案"


def test_document_formatting_requires_attachment_and_marker():
    result = route(
        "把这段改为公文格式",
        attachments=[{"file_id": "tmp1", "filename": "材料.docx", "is_spreadsheet": False}],
    )
    assert result.intent == INTENT_DOC_FORMATTING


def test_document_formatting_without_material_clarifies():
    result = route("把这段改为公文格式")
    assert result.intent == INTENT_CLARIFY
    assert result.document_type == "格式转换"


def test_document_format_knowledge_question_stays_qa():
    result = route("公文格式要求有哪些？")
    assert result.intent == INTENT_KNOWLEDGE_QA


def test_revision_request_uses_doc_pipeline_only_when_last_document_exists():
    assert route("请基于上一条回复改写为更精简的版本").intent == INTENT_KNOWLEDGE_QA
    result = route("请基于上一条回复改写为更精简的版本", has_last_document=True)
    assert result.intent == INTENT_DOC_DRAFTING
    assert result.document_type == "续写修改"


def test_ambiguous_form_export_clarifies():
    result = route("给我导出报销表模板")
    assert result.intent == INTENT_CLARIFY
    assert result.document_type == "报销表单"


def test_spreadsheet_transform_action():
    result = route(
        "筛选这个表格里金额大于5000的记录并导出",
        attachments=[{"file_id": "sheet1", "filename": "预算.xlsx", "is_spreadsheet": True}],
    )
    assert result.intent == INTENT_SPREADSHEET_TRANSFORM
    assert result.actions[0]["type"] == "spreadsheet_transform"
    assert result.actions[0]["file_id"] == "sheet1"


def test_classifier_not_called_for_clear_knowledge_qa():
    calls = []

    def classifier(payload):
        calls.append(payload)
        return {"intent": INTENT_DOC_DRAFTING}

    result = IntentRouter(reimbursement_detector, intent_classifier=classifier).route(
        message="NAS服务器如何使用？",
        display_message="NAS服务器如何使用？",
        mode="quick",
    )

    assert result.intent == INTENT_KNOWLEDGE_QA
    assert calls == []


def test_classifier_can_refine_weak_document_signal():
    def classifier(payload):
        assert payload["rule_result"]["intent"] == INTENT_KNOWLEDGE_QA
        return {
            "intent": INTENT_DOC_DRAFTING,
            "confidence": 0.89,
            "reason": "用户想形成正式材料",
            "document_type": "会议审议案",
        }

    result = IntentRouter(reimbursement_detector, intent_classifier=classifier).route(
        message="这个办公会材料麻烦处理一下",
        display_message="这个办公会材料麻烦处理一下",
        mode="quick",
    )

    assert result.intent == INTENT_DOC_DRAFTING
    assert result.document_type == "会议审议案"
    assert result.actions == []


def test_classifier_can_refine_ambiguous_form_export():
    def classifier(payload):
        assert payload["rule_result"]["intent"] == INTENT_CLARIFY
        return {
            "intent": INTENT_FORM_TEMPLATE_EXPORT,
            "confidence": 0.9,
            "reason": "补充分辨为会议费模板",
            "document_type": "报销表单",
            "template_key": "meeting",
        }

    result = IntentRouter(reimbursement_detector, intent_classifier=classifier).route(
        message="给我导出报销表模板",
        display_message="给我导出报销表模板",
        mode="quick",
    )

    assert result.intent == INTENT_FORM_TEMPLATE_EXPORT
    assert result.template_key == "meeting"
    assert result.actions == [{"type": "export_xlsx_template", "label": "导出会议费报销表", "template_key": "meeting"}]


def test_classifier_invalid_intent_falls_back_to_rules():
    def classifier(payload):
        return {"intent": "make_anything", "confidence": 0.99}

    result = IntentRouter(reimbursement_detector, intent_classifier=classifier).route(
        message="给我导出报销表模板",
        display_message="给我导出报销表模板",
        mode="quick",
    )

    assert result.intent == INTENT_CLARIFY
    assert result.document_type == "报销表单"
