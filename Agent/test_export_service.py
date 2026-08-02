from pathlib import Path

from export_service import DOCX_MIMETYPE, XLSX_MIMETYPE, ExportService, ExportServiceDependencies


class FakeDoc:
    def save(self, buffer):
        buffer.write(b"docx-bytes")


def build_service(
    *,
    template_path=None,
    workbook_from_text=lambda text: {"text": text},
    workbook_to_bytes=lambda workbook: b"xlsx-bytes",
    detect_reimbursement_template=lambda text, requested: "",
    resolve_reimbursement_template=lambda text, plan, user_request: "",
):
    return ExportService(ExportServiceDependencies(
        create_docx=lambda text, template: FakeDoc(),
        detect_export_template=lambda text, template: "review_proposal" if "议案" in text else "default",
        workbook_from_text=workbook_from_text,
        workbook_to_bytes=workbook_to_bytes,
        safe_filename_stem=lambda text, default: text.strip() or default,
        detect_reimbursement_template=detect_reimbursement_template,
        resolve_reimbursement_template=resolve_reimbursement_template,
        reimbursement_template_path=lambda key: Path(template_path) if template_path else Path("/missing/template.xlsx"),
        reimbursement_template_files={"meeting": "会议费.xlsx"},
    ))


def test_export_docx_returns_payload_and_filename():
    result = build_service().export_docx({"content": "关于审议测试议案", "template_type": "auto"})

    assert result.success
    assert result.payload == b"docx-bytes"
    assert result.mimetype == DOCX_MIMETYPE
    assert result.filename == "议案_关于审议测试议案.docx"


def test_export_docx_rejects_empty_content():
    result = build_service().export_docx({"content": ""})

    assert not result.success
    assert result.status == 400
    assert result.error == "没有内容可以导出"


def test_export_xlsx_returns_payload_and_filename():
    result = build_service().export_xlsx({"content": "|姓名|金额|\n|张三|100|"})

    assert result.success
    assert result.payload == b"xlsx-bytes"
    assert result.mimetype == XLSX_MIMETYPE
    assert result.filename.startswith("表格_")
    assert result.filename.endswith(".xlsx")


def test_export_xlsx_reports_generation_error():
    def fail_workbook_from_text(text):
        raise RuntimeError("parse failed")

    result = build_service(workbook_from_text=fail_workbook_from_text).export_xlsx({"content": "表格"})

    assert not result.success
    assert result.status == 500
    assert "parse failed" in result.error


def test_export_reimbursement_returns_template_path(tmp_path):
    template = tmp_path / "meeting.xlsx"
    template.write_bytes(b"template")
    service = build_service(
        template_path=template,
        detect_reimbursement_template=lambda text, requested: "meeting",
    )

    result = service.export_reimbursement_xlsx({"content": "导出会议费报销表"})

    assert result.success
    assert result.path == template
    assert result.filename == "会议费.xlsx"
    assert result.mimetype == XLSX_MIMETYPE


def test_export_reimbursement_reports_unknown_template():
    result = build_service().export_reimbursement_xlsx({"content": "导出报销表"})

    assert not result.success
    assert result.status == 400
    assert "未识别到报销表单类型" in result.error


def test_export_reimbursement_reports_missing_template():
    service = build_service(
        detect_reimbursement_template=lambda text, requested: "meeting",
    )

    result = service.export_reimbursement_xlsx({"content": "导出会议费报销表"})

    assert not result.success
    assert result.status == 404
    assert "报销模板文件不存在" in result.error
