from pathlib import Path

from knowledge_admin_read_service import (
    XLSX_MIMETYPE,
    KnowledgeAdminReadDependencies,
    KnowledgeAdminReadService,
)


class FakeKnowledgeBase:
    metadatas = [
        {"content_hash": "hash_1", "filename": "制度.docx", "source_path": "/missing/制度.docx"},
        {"content_hash": "hash_1", "filename": "制度.docx", "source_path": "/missing/制度.docx"},
        {"filename": "历史.docx", "source_path": "/legacy/history.docx", "category": "公共资料"},
    ]


class FakeManifest:
    def all_records(self, limit=500):
        return [{
            "content_hash": "hash_1",
            "filename": "制度.docx",
            "source_path": "/missing/制度.docx",
            "archived_path": "",
            "category": "公共资料",
            "access_level": "public",
            "department": "",
            "uploaded_by": "admin",
            "uploaded_at": "2026-06-07",
            "source_type": "document",
            "parser_type": "docx",
            "chunk_count": 2,
            "spreadsheet_row_count": 0,
            "status": "completed",
            "updated_at": "2026-06-07",
        }][:limit]

    def recent_audit(self, limit=80, content_hash=None):
        rows = [
            {"content_hash": "hash_1", "action": "upload"},
            {"content_hash": "hash_2", "action": "delete"},
        ]
        if content_hash:
            rows = [row for row in rows if row["content_hash"] == content_hash]
        return rows[:limit]


class FakeSpreadsheetStore:
    def __init__(self):
        self.files = [{
            "content_hash": "hash_1",
            "filename": "制度.xlsx",
            "row_count": 3,
            "sheet_count": 1,
            "validation": {"ok": True},
        }]
        self.rows = {
            "hash_1": [
                {"filename": "制度.xlsx", "sheet_name": "Sheet1", "row_number": 1, "value": "A"},
                {"filename": "制度.xlsx", "sheet_name": "Sheet1", "row_number": 2, "value": "B"},
            ]
        }

    def list_files(self, limit=100):
        return self.files[:limit]

    def get_rows_by_source(self, content_hash, sheet_name=None, row_start=None, row_end=None):
        rows = self.rows.get(content_hash, [])
        if sheet_name:
            rows = [row for row in rows if row.get("sheet_name") == sheet_name]
        if row_start is not None:
            rows = [row for row in rows if row.get("row_number", 0) >= row_start]
        if row_end is not None:
            rows = [row for row in rows if row.get("row_number", 0) <= row_end]
        return rows


def build_service(store=None, workbook_from_structured_rows=None):
    return KnowledgeAdminReadService(KnowledgeAdminReadDependencies(
        knowledge_base=FakeKnowledgeBase(),
        knowledge_manifest=FakeManifest(),
        spreadsheet_store=store or FakeSpreadsheetStore(),
        workbook_from_structured_rows=workbook_from_structured_rows or (lambda rows: {"rows": rows}),
        workbook_to_bytes=lambda workbook: b"xlsx-bytes",
        safe_filename_stem=lambda value, default: Path(value or default).stem,
    ))


def test_knowledge_admin_read_service_merges_manifest_vector_and_spreadsheet_state():
    service = build_service()

    payload = service.knowledge_files(limit=20)

    assert payload["success"] is True
    assert payload["total"] == 2
    managed = next(row for row in payload["files"] if row["content_hash"] == "hash_1")
    legacy = next(row for row in payload["files"] if row["status"] == "legacy")
    assert managed["managed"] is True
    assert managed["vector_chunk_count"] == 2
    assert managed["spreadsheet_rows_found"] == 3
    assert managed["spreadsheet_sheet_count"] == 1
    assert legacy["managed"] is False
    assert legacy["content_hash"].startswith("legacy:")


def test_knowledge_admin_read_service_filters_files_and_reads_audit():
    service = build_service()

    filtered = service.knowledge_files(filters={"status": "completed", "q": "制度"})
    audit = service.knowledge_audit(limit=10, content_hash="hash_2")

    assert filtered["total"] == 1
    assert filtered["files"][0]["content_hash"] == "hash_1"
    assert audit == {"success": True, "audit": [{"content_hash": "hash_2", "action": "delete"}], "total": 1}


def test_knowledge_admin_read_service_lists_rows_and_exports_spreadsheet():
    service = build_service()

    files = service.spreadsheets(limit=1)
    rows = service.spreadsheet_rows("hash_1", row_start=2)
    export = service.export_spreadsheet("hash_1")

    assert files == {"success": True, "files": [FakeSpreadsheetStore().files[0]]}
    assert rows == {"success": True, "rows": [FakeSpreadsheetStore().rows["hash_1"][1]], "total": 1}
    assert export.success
    assert export.payload == b"xlsx-bytes"
    assert export.filename == "制度_结构化导出.xlsx"
    assert export.mimetype == XLSX_MIMETYPE


def test_knowledge_admin_read_service_export_errors():
    empty_store = FakeSpreadsheetStore()
    empty_store.rows = {}
    missing = build_service(store=empty_store).export_spreadsheet("missing")

    def fail_workbook(rows):
        raise RuntimeError("boom")

    failed = build_service(workbook_from_structured_rows=fail_workbook).export_spreadsheet("hash_1")

    assert missing.status == 404
    assert missing.error == "未找到可导出的表格数据"
    assert failed.status == 500
    assert "boom" in failed.error
