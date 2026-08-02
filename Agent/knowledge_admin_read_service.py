"""Read-only admin services for knowledge files and structured spreadsheets."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

XLSX_MIMETYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@dataclass
class AdminFileExportResult:
    filename: str = ""
    payload: Optional[bytes] = None
    mimetype: str = XLSX_MIMETYPE
    error: str = ""
    status: int = 200

    @property
    def success(self) -> bool:
        return not self.error


@dataclass
class KnowledgeAdminReadDependencies:
    knowledge_base: Any
    knowledge_manifest: Any
    spreadsheet_store: Any
    workbook_from_structured_rows: Callable[[list[dict]], Any]
    workbook_to_bytes: Callable[[Any], bytes]
    safe_filename_stem: Callable[[str, str], str]


class KnowledgeAdminReadService:
    def __init__(self, deps: KnowledgeAdminReadDependencies):
        self.deps = deps

    def knowledge_files(self, *, limit: int = 200, filters: dict = None) -> dict:
        rows = self._knowledge_file_rows(limit=limit, filters=filters or {})
        return {"success": True, "files": rows, "total": len(rows)}

    def knowledge_audit(self, *, limit: int = 80, content_hash: str = None) -> dict:
        rows = self.deps.knowledge_manifest.recent_audit(
            limit=limit,
            content_hash=content_hash or None,
        )
        return {"success": True, "audit": rows, "total": len(rows)}

    def spreadsheets(self, *, limit: int = 100) -> dict:
        return {
            "success": True,
            "files": self.deps.spreadsheet_store.list_files(limit=limit),
        }

    def spreadsheet_rows(
        self,
        content_hash: str,
        *,
        sheet_name: str = None,
        row_start: int = None,
        row_end: int = None,
    ) -> dict:
        rows = self.deps.spreadsheet_store.get_rows_by_source(
            content_hash,
            sheet_name=sheet_name,
            row_start=row_start,
            row_end=row_end,
        )
        return {"success": True, "rows": rows[:500], "total": len(rows)}

    def export_spreadsheet(self, content_hash: str) -> AdminFileExportResult:
        rows = self.deps.spreadsheet_store.get_rows_by_source(content_hash)
        if not rows:
            return AdminFileExportResult(error="未找到可导出的表格数据", status=404)

        try:
            workbook = self.deps.workbook_from_structured_rows(rows)
            payload = self.deps.workbook_to_bytes(workbook)
        except Exception as exc:
            logger.exception("结构化表格导出失败: %s", exc)
            return AdminFileExportResult(error=f"导出失败: {str(exc)[:120]}", status=500)

        filename = f"{self.deps.safe_filename_stem(rows[0].get('filename'), '结构化表格')}_结构化导出.xlsx"
        return AdminFileExportResult(filename=filename, payload=payload)

    def _knowledge_file_rows(self, *, limit: int = 500, filters: dict = None) -> list:
        filters = filters or {}
        vector_counts = {}
        vector_examples = {}
        for meta in getattr(self.deps.knowledge_base, "metadatas", []) or []:
            source_key = (
                meta.get("content_hash")
                or meta.get("source_path")
                or meta.get("source")
                or meta.get("filename")
            )
            if not source_key:
                continue
            content_hash = meta.get("content_hash") or f"legacy:{hashlib.sha1(str(source_key).encode('utf-8')).hexdigest()}"
            vector_counts[content_hash] = vector_counts.get(content_hash, 0) + 1
            vector_examples.setdefault(content_hash, meta)

        spreadsheet_counts = {
            item["content_hash"]: item
            for item in self.deps.spreadsheet_store.list_files(limit=10000)
        }

        rows = []
        manifest_hashes = set()
        for record in self.deps.knowledge_manifest.all_records(limit=limit):
            content_hash = record["content_hash"]
            manifest_hashes.add(content_hash)
            sheet_info = spreadsheet_counts.get(content_hash, {})
            source_path = record.get("source_path") or ""
            archived_path = record.get("archived_path") or ""
            row = dict(record)
            row.update({
                "file_exists": bool(source_path) and Path(source_path).exists(),
                "archived_exists": bool(archived_path) and Path(archived_path).exists(),
                "vector_chunk_count": vector_counts.get(content_hash, 0),
                "spreadsheet_rows_found": sheet_info.get("row_count", 0),
                "spreadsheet_sheet_count": sheet_info.get("sheet_count", 0),
                "spreadsheet_validation": sheet_info.get("validation", {}),
                "managed": True,
            })
            rows.append(row)

        for content_hash, meta in vector_examples.items():
            if content_hash in manifest_hashes:
                continue
            source_path = meta.get("source_path") or meta.get("source") or ""
            rows.append({
                "content_hash": content_hash,
                "filename": meta.get("filename") or Path(source_path).name or "历史文件",
                "source_path": source_path,
                "archived_path": "",
                "category": meta.get("category", ""),
                "access_level": meta.get("access_level", "public"),
                "department": meta.get("department", ""),
                "uploaded_by": meta.get("uploaded_by", ""),
                "uploaded_at": meta.get("uploaded_at", ""),
                "source_type": meta.get("source_type", "document"),
                "parser_type": meta.get("parser_type", meta.get("file_type", "")),
                "chunk_count": vector_counts.get(content_hash, 0),
                "spreadsheet_row_count": 0,
                "status": "legacy",
                "error_message": "历史索引缺少 manifest/content_hash，仅支持查看；重新上传后可完整管理",
                "updated_at": meta.get("uploaded_at", ""),
                "file_exists": bool(source_path) and Path(source_path).exists(),
                "archived_exists": False,
                "vector_chunk_count": vector_counts.get(content_hash, 0),
                "spreadsheet_rows_found": 0,
                "spreadsheet_sheet_count": 0,
                "managed": False,
            })

        return [row for row in rows if self._matches_filters(row, filters)]

    @staticmethod
    def _matches_filters(row: dict, filters: dict) -> bool:
        search = (filters.get("q") or "").strip().lower()
        status = filters.get("status") or ""
        source_type = filters.get("source_type") or ""
        access_level = filters.get("access_level") or ""

        if status and row.get("status") != status:
            return False
        if source_type and row.get("source_type") != source_type:
            return False
        if access_level and row.get("access_level") != access_level:
            return False
        if search:
            haystack = " ".join([
                row.get("filename", ""),
                row.get("source_path", ""),
                row.get("category", ""),
                row.get("department", ""),
                row.get("content_hash", ""),
            ]).lower()
            if search not in haystack:
                return False
        return True
