from task_planner import (
    TOOL_DRAFT_DOCUMENT,
    TOOL_FORMAT_DOCUMENT,
    TOOL_KNOWLEDGE_QA,
    TOOL_PREPARE_FORM_EXPORT,
    TOOL_PREPARE_SPREADSHEET_TRANSFORM,
    TaskPlanner,
)


def reimbursement_detector(text, requested="auto"):
    if "会议费" in text:
        return "meeting"
    return ""


def tool_names(plan):
    return [step.tool for step in plan.steps]


def build_planner(classifier=None):
    return TaskPlanner(reimbursement_detector, planner_classifier=classifier)


def test_task_planner_selects_knowledge_qa_for_fact_question():
    plan = build_planner().plan(message="会议费标准是多少", display_message="会议费标准是多少")

    assert tool_names(plan) == [TOOL_KNOWLEDGE_QA]
    assert plan.requires_confirmation is False


def test_task_planner_selects_draft_document_for_writing_request():
    plan = build_planner().plan(message="帮我写一份通知", display_message="帮我写一份通知")

    assert tool_names(plan) == [TOOL_DRAFT_DOCUMENT]


def test_task_planner_selects_format_document_for_uploaded_formatting():
    plan = build_planner().plan(
        message="[文件内容]\n材料\n[/文件内容]\n\n[用户提问]\n把上传材料改成公文格式",
        display_message="把上传材料改成公文格式",
        attachments=[{"file_id": "doc1", "filename": "材料.docx", "is_spreadsheet": False}],
    )

    assert tool_names(plan) == [TOOL_FORMAT_DOCUMENT]


def test_task_planner_prepares_form_export_action():
    plan = build_planner().plan(message="导出会议费报销表", display_message="导出会议费报销表")

    assert tool_names(plan) == [TOOL_PREPARE_FORM_EXPORT]
    assert plan.requires_confirmation is True
    assert plan.final_response_mode == "confirm_actions"


def test_task_planner_prepares_spreadsheet_transform_action():
    plan = build_planner().plan(
        message="把表格按金额降序排序",
        display_message="把表格按金额降序排序",
        attachments=[{"file_id": "sheet1", "filename": "预算.xlsx", "is_spreadsheet": True}],
    )

    assert tool_names(plan) == [TOOL_PREPARE_SPREADSHEET_TRANSFORM]
    assert plan.requires_confirmation is True


def test_task_planner_selects_multi_step_research_then_write():
    plan = build_planner().plan(message="先查制度再写流程说明", display_message="先查制度再写流程说明")

    assert tool_names(plan) == [TOOL_KNOWLEDGE_QA, TOOL_DRAFT_DOCUMENT]


def test_task_planner_falls_back_when_classifier_fails():
    def classifier(_payload):
        raise RuntimeError("planner unavailable")

    plan = build_planner(classifier).plan(message="帮我写一份通知", display_message="帮我写一份通知")

    assert plan.source == "rules"
    assert tool_names(plan) == [TOOL_DRAFT_DOCUMENT]


def test_task_planner_accepts_classifier_plan():
    def classifier(_payload):
        return {
            "task_type": "复合任务",
            "steps": [
                {"tool": TOOL_KNOWLEDGE_QA, "reason": "先查依据"},
                {"tool": TOOL_DRAFT_DOCUMENT, "reason": "再起草"},
            ],
        }

    plan = build_planner(classifier).plan(message="处理一下", display_message="处理一下")

    assert plan.source == "llm"
    assert tool_names(plan) == [TOOL_KNOWLEDGE_QA, TOOL_DRAFT_DOCUMENT]
