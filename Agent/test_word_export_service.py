from io import BytesIO

from word_export_service import (
    create_docx,
    detect_export_template,
    detect_reimbursement_template,
    explicit_document_request,
    parse_document_format,
    reimbursement_template_path,
    resolve_export_template,
    resolve_reimbursement_template,
)


def test_parse_document_format_classifies_common_official_document_parts():
    parts = parse_document_format("""关于召开会议的通知
各部门：
一、会议时间
请准时参加。
智能知识库平台
2026年6月7日""")

    assert [part["type"] for part in parts] == [
        "title",
        "recipient",
        "heading1",
        "body",
        "signature_unit",
        "signature_date",
    ]


def test_template_detection_and_resolution():
    assert detect_export_template("关于审议测试事项的议案") == "review_proposal"
    assert detect_export_template("普通通知", "unknown") == "default"
    assert resolve_export_template("正文", {"document_type": "会议审议案"}, "用户请求") == "review_proposal"
    assert explicit_document_request("帮我写一份通知") is True
    assert explicit_document_request("公文格式有哪些要求") is True


def test_reimbursement_template_detection_and_path(tmp_path):
    files = {"meeting": "会议费.xlsx"}

    assert detect_reimbursement_template("请导出会议费报销表") == "meeting"
    assert detect_reimbursement_template("", "labor") == "labor_expert"
    assert resolve_reimbursement_template("", {"document_type": "差旅费报销"}, "") == "travel"
    assert reimbursement_template_path("meeting", template_dir=tmp_path, template_files=files) == tmp_path / "会议费.xlsx"
    assert str(reimbursement_template_path("missing", template_dir=tmp_path, template_files=files)) == "."


def test_create_docx_outputs_serializable_document():
    doc = create_docx("""关于召开会议的通知
各部门：
请准时参加。""")
    buffer = BytesIO()

    doc.save(buffer)

    assert buffer.getvalue().startswith(b"PK")


def test_create_review_proposal_docx_outputs_serializable_document():
    doc = create_docx("关于审议测试事项的议案\n各位领导：\n以上，请审议。", "review_proposal")
    buffer = BytesIO()

    doc.save(buffer)

    assert buffer.getvalue().startswith(b"PK")
