"""File export services for generated documents and spreadsheet templates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


DOCX_MIMETYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIMETYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@dataclass
class ExportResult:
    filename: str = ""
    mimetype: str = ""
    payload: Optional[bytes] = None
    path: Optional[Path] = None
    error: str = ""
    status: int = 200

    @property
    def success(self) -> bool:
        return not self.error


@dataclass
class ExportServiceDependencies:
    create_docx: Callable[[str, str], Any]
    detect_export_template: Callable[[str, str], str]
    workbook_from_text: Callable[[str], Any]
    workbook_to_bytes: Callable[[Any], bytes]
    safe_filename_stem: Callable[[str, str], str]
    detect_reimbursement_template: Callable[[str, str], str]
    resolve_reimbursement_template: Callable[[str, dict, str], str]
    reimbursement_template_path: Callable[[str], Path]
    reimbursement_template_files: dict[str, str]


class ExportService:
    def __init__(self, deps: ExportServiceDependencies):
        self.deps = deps

    def export_docx(self, data: dict) -> ExportResult:
        request_data = data or {}
        text = request_data.get("content", "")
        template_type = request_data.get("template_type", "auto")
        if not text:
            return ExportResult(error="没有内容可以导出", status=400)

        resolved_template = self.deps.detect_export_template(text, template_type)
        doc = self.deps.create_docx(text, resolved_template)
        buffer = BytesIO()
        doc.save(buffer)

        clean_name = text[:20].replace("\n", "").replace("/", "").replace("\\", "").strip()
        if not clean_name:
            clean_name = "公文文档"
        prefix = "议案" if resolved_template == "review_proposal" else "公文"
        return ExportResult(
            filename=f"{prefix}_{clean_name}.docx",
            mimetype=DOCX_MIMETYPE,
            payload=buffer.getvalue(),
        )

    def export_xlsx(self, data: dict) -> ExportResult:
        request_data = data or {}
        text = request_data.get("content", "")
        if not text or not text.strip():
            return ExportResult(error="没有内容可以导出", status=400)

        try:
            workbook = self.deps.workbook_from_text(text)
            payload = self.deps.workbook_to_bytes(workbook)
        except Exception as exc:
            logger.exception("Excel 导出失败: %s", exc)
            return ExportResult(error=f"导出失败: {str(exc)[:120]}", status=500)

        clean_name = self.deps.safe_filename_stem(text[:20], "公文表格")
        return ExportResult(
            filename=f"表格_{clean_name}.xlsx",
            mimetype=XLSX_MIMETYPE,
            payload=payload,
        )

    def export_reimbursement_xlsx(self, data: dict) -> ExportResult:
        request_data = data or {}
        template_key = self.deps.detect_reimbursement_template(
            request_data.get("content", ""),
            request_data.get("template_type", "auto"),
        )
        if not template_key:
            template_key = self.deps.resolve_reimbursement_template(
                request_data.get("content", ""),
                request_data.get("plan") if isinstance(request_data.get("plan"), dict) else {},
                request_data.get("user_request", ""),
            )
        if not template_key:
            return ExportResult(
                error="未识别到报销表单类型，请说明差旅费、会议费、劳务费/专家咨询费或其他费用报销。",
                status=400,
            )

        template_path = self.deps.reimbursement_template_path(template_key)
        if not template_path.exists():
            logger.error("报销模板不存在: %s", template_path)
            return ExportResult(error="报销模板文件不存在，请检查公共资料是否已入库。", status=404)

        return ExportResult(
            filename=self.deps.reimbursement_template_files[template_key],
            mimetype=XLSX_MIMETYPE,
            path=template_path,
        )
