import threading
from datetime import datetime

from knowledge_admin_write_service import (
    KnowledgeAdminWriteDependencies,
    KnowledgeAdminWriteService,
)


class FakeKnowledgeBase:
    def __init__(self, *, fail_delete=False, fail_metadata=False, fail_replace=False):
        self.fail_delete = fail_delete
        self.fail_metadata = fail_metadata
        self.fail_replace = fail_replace
        self.deleted = []
        self.metadata_updates = []
        self.replacements = []
        self.restored = []

    def snapshot_state(self):
        return {"kb": "snapshot"}

    def delete_by_content_hash(self, content_hash):
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.deleted.append(content_hash)
        return 3

    def restore_state(self, snapshot):
        self.restored.append(snapshot)

    def update_metadata_by_content_hash(self, content_hash, updates):
        if self.fail_metadata:
            raise RuntimeError("metadata failed")
        self.metadata_updates.append((content_hash, updates))
        return 4

    def replace_by_content_hash(self, content_hash, documents):
        if self.fail_replace:
            raise RuntimeError("replace failed")
        self.replacements.append((content_hash, documents))
        return 5


class FakeSpreadsheetStore:
    def __init__(self):
        self.deleted = []
        self.metadata_updates = []
        self.upserts = []
        self.restored = []

    def snapshot_file(self, content_hash):
        return {"sheet": content_hash}

    def delete_file(self, content_hash):
        self.deleted.append(content_hash)

    def restore_file_snapshot(self, snapshot):
        self.restored.append(snapshot)

    def update_file_metadata(self, content_hash, *, category, access_level, department):
        self.metadata_updates.append((content_hash, category, access_level, department))
        return 2

    def upsert_file_rows(self, **kwargs):
        self.upserts.append(kwargs)
        return len(kwargs["rows"])


class FakeManifest:
    def __init__(self, record=None):
        self.record = record
        self.archived = []
        self.reindexed = []
        self.restored = []

    def get_record(self, content_hash):
        return dict(self.record) if self.record else None

    def mark_archived(self, content_hash, archived_path):
        self.archived.append((content_hash, archived_path))
        if self.record:
            self.record["archived_path"] = archived_path
            self.record["status"] = "archived"

    def update_metadata(self, content_hash, *, category, access_level, department):
        if self.record:
            self.record["category"] = category
            self.record["access_level"] = access_level
            self.record["department"] = department

    def mark_reindexed(self, content_hash, **kwargs):
        self.reindexed.append((content_hash, kwargs))
        if self.record:
            self.record.update(kwargs)
            self.record["status"] = "active"

    def restore_record(self, snapshot):
        self.restored.append(snapshot)
        self.record = dict(snapshot)


class FakeUploadManager:
    def __init__(self, archived_path=None, *, hash_value="hash_1", prepared=None):
        self.archived_path = archived_path
        self.hash_value = hash_value
        self.prepared = prepared or {
            "success": True,
            "documents": [{"page_content": "doc"}],
            "is_spreadsheet": False,
            "source_type": "document",
            "parser_type": "plain_text",
        }
        self.archived_sources = []
        self.hash_checks = []
        self.build_calls = []

    def archive_knowledge_file(self, source_path):
        self.archived_sources.append(source_path)
        if self.archived_path:
            return self.archived_path
        return ""

    def _file_sha256(self, file_path):
        self.hash_checks.append(file_path)
        return self.hash_value

    def build_knowledge_documents(self, **kwargs):
        self.build_calls.append(kwargs)
        return self.prepared


class FakeAuditService:
    def __init__(self, *, audits, clears):
        self.audits = audits
        self.clears = clears

    def backup_storage(self, action, content_hash):
        return f"/backup/{action}/{content_hash}"

    def record_audit(self, *args, **kwargs):
        self.audits.append((args, kwargs))

    def clear_runtime_cache(self):
        self.clears.append(True)


def build_service(
    *,
    record=None,
    kb=None,
    spreadsheet_store=None,
    manifest=None,
    upload_manager=None,
    audits=None,
    clears=None,
    parse_spreadsheet=None,
):
    audit_calls = audits if audits is not None else []
    clear_calls = clears if clears is not None else []
    return KnowledgeAdminWriteService(KnowledgeAdminWriteDependencies(
        knowledge_base=kb or FakeKnowledgeBase(),
        spreadsheet_store=spreadsheet_store or FakeSpreadsheetStore(),
        knowledge_manifest=manifest or FakeManifest(record),
        upload_manager=upload_manager or FakeUploadManager(),
        admin_lock=threading.RLock(),
        access_levels={"public", "restricted"},
        department_dirs={"示例部门甲", "示例部门乙", "财务部"},
        parse_spreadsheet=parse_spreadsheet or (lambda *args, **kwargs: []),
        now_factory=lambda: datetime(2026, 6, 7, 9, 0, 0),
        audit_service=FakeAuditService(audits=audit_calls, clears=clear_calls),
    ))


def test_delete_knowledge_file_validates_request():
    service = build_service()

    no_confirm, no_confirm_status = service.delete_knowledge_file("hash_1", {"reason": "删除"})
    no_reason, no_reason_status = service.delete_knowledge_file("hash_1", {"confirm": True, "reason": ""})
    missing, missing_status = service.delete_knowledge_file("hash_1", {"confirm": True, "reason": "删除"})

    assert no_confirm_status == 400
    assert no_confirm == {"success": False, "message": "请确认删除操作"}
    assert no_reason_status == 400
    assert no_reason == {"success": False, "message": "请填写删除原因"}
    assert missing_status == 404
    assert missing == {"success": False, "message": "未找到该知识库文件"}


def test_delete_knowledge_file_success(tmp_path):
    source = tmp_path / "source.docx"
    source.write_text("doc", encoding="utf-8")
    record = {
        "content_hash": "hash_1",
        "filename": "source.docx",
        "source_path": str(source),
        "archived_path": "",
    }
    kb = FakeKnowledgeBase()
    store = FakeSpreadsheetStore()
    manifest = FakeManifest(record)
    audits = []
    clears = []
    before_record = dict(record)
    service = build_service(
        kb=kb,
        spreadsheet_store=store,
        manifest=manifest,
        upload_manager=FakeUploadManager(archived_path="/archive/source.docx"),
        audits=audits,
        clears=clears,
    )

    payload, status = service.delete_knowledge_file("hash_1", {"confirm": True, "reason": "测试删除"})

    assert status == 200
    assert payload == {
        "success": True,
        "message": "已软删除并归档",
        "vector_removed": 3,
        "archived_path": "/archive/source.docx",
        "backup_path": "/backup/delete/hash_1",
    }
    assert kb.deleted == ["hash_1"]
    assert store.deleted == ["hash_1"]
    assert manifest.archived == [("hash_1", "/archive/source.docx")]
    assert clears == [True]
    assert audits[0][0][:2] == ("delete", before_record)
    assert audits[0][1]["status"] == "success"
    assert "测试删除" in audits[0][1]["message"]


def test_delete_knowledge_file_rolls_back_snapshots_and_archived_file(tmp_path):
    source = tmp_path / "source.docx"
    archived = tmp_path / "archived.docx"
    source.write_text("doc", encoding="utf-8")

    class MovingUploadManager:
        def archive_knowledge_file(self, source_path):
            source.rename(archived)
            return str(archived)

    record = {
        "content_hash": "hash_1",
        "filename": "source.docx",
        "source_path": str(source),
        "archived_path": "",
    }
    kb = FakeKnowledgeBase(fail_delete=True)
    store = FakeSpreadsheetStore()
    manifest = FakeManifest(record)
    audits = []
    clears = []
    service = build_service(
        kb=kb,
        spreadsheet_store=store,
        manifest=manifest,
        upload_manager=MovingUploadManager(),
        audits=audits,
        clears=clears,
    )

    payload, status = service.delete_knowledge_file("hash_1", {"confirm": True, "reason": "测试删除"})

    assert status == 500
    assert payload["success"] is False
    assert payload["rolled_back"] is True
    assert payload["rollback_errors"] == []
    assert source.exists()
    assert not archived.exists()
    assert kb.restored == [{"kb": "snapshot"}]
    assert store.restored == [{"sheet": "hash_1"}]
    assert manifest.restored == [record]
    assert clears == []
    assert audits[0][1]["status"] == "failed"
    assert "delete failed" in audits[0][1]["message"]


def test_update_knowledge_file_metadata_validates_request():
    empty = build_service()
    missing, missing_status = empty.update_knowledge_file_metadata("hash_1", {})

    record = {
        "content_hash": "hash_1",
        "category": "",
        "access_level": "public",
        "department": "",
    }
    service = build_service(record=record)
    no_category, no_category_status = service.update_knowledge_file_metadata("hash_1", {})
    bad_access, bad_access_status = service.update_knowledge_file_metadata("hash_1", {
        "category": "公共资料",
        "access_level": "secret",
    })
    bad_department, bad_department_status = service.update_knowledge_file_metadata("hash_1", {
        "category": "财务部",
        "access_level": "restricted",
        "department": "不存在",
    })

    assert missing_status == 404
    assert missing == {"success": False, "message": "未找到该知识库文件"}
    assert no_category_status == 400
    assert no_category == {"success": False, "message": "分类不能为空"}
    assert bad_access_status == 400
    assert bad_access == {"success": False, "message": "访问级别无效"}
    assert bad_department_status == 400
    assert bad_department == {"success": False, "message": "restricted 权限必须选择有效部门"}


def test_update_knowledge_file_metadata_success():
    record = {
        "content_hash": "hash_1",
        "category": "公共资料",
        "access_level": "public",
        "department": "",
    }
    kb = FakeKnowledgeBase()
    store = FakeSpreadsheetStore()
    manifest = FakeManifest(record)
    audits = []
    clears = []
    before_record = dict(record)
    service = build_service(kb=kb, spreadsheet_store=store, manifest=manifest, audits=audits, clears=clears)

    payload, status = service.update_knowledge_file_metadata("hash_1", {
        "category": "财务部",
        "access_level": "restricted",
        "department": "财务部",
    })

    expected = {"category": "财务部", "access_level": "restricted", "department": "财务部"}
    assert status == 200
    assert payload == {
        "success": True,
        "message": "元数据已更新",
        "vector_updated": 4,
        "spreadsheet_updated": 2,
        "backup_path": "/backup/metadata/hash_1",
    }
    assert kb.metadata_updates == [("hash_1", expected)]
    assert store.metadata_updates == [("hash_1", "财务部", "restricted", "财务部")]
    assert manifest.record["category"] == "财务部"
    assert clears == [True]
    assert audits[0][0][:2] == ("metadata", before_record)
    assert audits[0][1]["status"] == "success"


def test_update_knowledge_file_metadata_clears_department_for_public_and_rolls_back():
    record = {
        "content_hash": "hash_1",
        "category": "公共资料",
        "access_level": "restricted",
        "department": "财务部",
    }
    kb = FakeKnowledgeBase(fail_metadata=True)
    store = FakeSpreadsheetStore()
    manifest = FakeManifest(record)
    audits = []
    clears = []
    service = build_service(kb=kb, spreadsheet_store=store, manifest=manifest, audits=audits, clears=clears)

    payload, status = service.update_knowledge_file_metadata("hash_1", {
        "category": "公共资料",
        "access_level": "public",
        "department": "财务部",
    })

    assert status == 500
    assert payload["success"] is False
    assert payload["rolled_back"] is True
    assert payload["rollback_errors"] == []
    assert kb.restored == [{"kb": "snapshot"}]
    assert store.restored == [{"sheet": "hash_1"}]
    assert manifest.restored == [record]
    assert clears == []
    assert audits[0][1]["status"] == "failed"
    assert "'department': ''" in audits[0][1]["message"]
    assert "metadata failed" in audits[0][1]["message"]


def test_reindex_knowledge_file_validates_record_path_hash_and_parse_result(tmp_path):
    missing_payload, missing_status = build_service().reindex_knowledge_file("hash_1", user_id="u1")

    no_file_record = {
        "content_hash": "hash_1",
        "filename": "missing.docx",
        "source_path": str(tmp_path / "missing.docx"),
        "archived_path": "",
    }
    no_file_payload, no_file_status = build_service(record=no_file_record).reindex_knowledge_file(
        "hash_1",
        user_id="u1",
    )

    source = tmp_path / "source.docx"
    source.write_text("doc", encoding="utf-8")
    record = {
        "content_hash": "hash_1",
        "filename": "source.docx",
        "source_path": str(source),
        "archived_path": "",
    }
    bad_hash_payload, bad_hash_status = build_service(
        record=record,
        upload_manager=FakeUploadManager(hash_value="other_hash"),
    ).reindex_knowledge_file("hash_1", user_id="u1")

    parse_fail_payload, parse_fail_status = build_service(
        record=record,
        upload_manager=FakeUploadManager(prepared={"success": False, "message": "解析失败"}),
    ).reindex_knowledge_file("hash_1", user_id="u1")

    assert missing_status == 404
    assert missing_payload == {"success": False, "message": "未找到该知识库文件"}
    assert no_file_status == 400
    assert no_file_payload == {"success": False, "message": "原文件和归档文件都不存在，无法重建"}
    assert bad_hash_status == 409
    assert bad_hash_payload == {
        "success": False,
        "message": "文件内容与入库哈希不一致，已阻止重建以避免数据错配",
    }
    assert parse_fail_status == 400
    assert parse_fail_payload == {"success": False, "message": "解析失败"}


def test_reindex_knowledge_file_success_for_document(tmp_path):
    source = tmp_path / "source.docx"
    source.write_text("doc", encoding="utf-8")
    record = {
        "content_hash": "hash_1",
        "filename": "source.docx",
        "source_path": str(source),
        "archived_path": "",
        "category": "公共资料",
        "access_level": "public",
        "department": "",
    }
    kb = FakeKnowledgeBase()
    store = FakeSpreadsheetStore()
    manifest = FakeManifest(record)
    upload_manager = FakeUploadManager(prepared={
        "success": True,
        "documents": [{"page_content": "doc1"}, {"page_content": "doc2"}],
        "is_spreadsheet": False,
        "source_type": "document",
        "parser_type": "plain_text",
    })
    audits = []
    clears = []
    before_record = dict(record)
    service = build_service(
        kb=kb,
        spreadsheet_store=store,
        manifest=manifest,
        upload_manager=upload_manager,
        audits=audits,
        clears=clears,
    )

    payload, status = service.reindex_knowledge_file("hash_1", user_id="fallback_user")

    assert status == 200
    assert payload == {
        "success": True,
        "message": "重建索引完成",
        "removed_previous_vectors": 5,
        "chunks": 2,
        "spreadsheet_rows": 0,
        "backup_path": "/backup/reindex/hash_1",
    }
    assert upload_manager.hash_checks == [str(source)]
    assert upload_manager.build_calls[0]["user_id"] == "fallback_user"
    assert upload_manager.build_calls[0]["uploaded_at"] == "2026-06-07T09:00:00"
    assert kb.replacements == [("hash_1", [{"page_content": "doc1"}, {"page_content": "doc2"}])]
    assert store.deleted == ["hash_1"]
    assert store.upserts == []
    assert manifest.reindexed == [("hash_1", {
        "source_path": str(source),
        "source_type": "document",
        "parser_type": "plain_text",
        "chunk_count": 2,
        "spreadsheet_row_count": 0,
    })]
    assert clears == [True]
    assert audits[0][0][:2] == ("reindex", before_record)
    assert audits[0][1]["status"] == "success"


def test_reindex_knowledge_file_success_for_spreadsheet(tmp_path):
    source = tmp_path / "source.xlsx"
    source.write_text("sheet", encoding="utf-8")
    record = {
        "content_hash": "hash_1",
        "filename": "source.xlsx",
        "source_path": str(source),
        "archived_path": "",
        "category": "财务",
        "access_level": "restricted",
        "department": "财务部",
        "uploaded_by": "owner",
        "uploaded_at": "2026-06-01T10:00:00",
    }
    store = FakeSpreadsheetStore()
    manifest = FakeManifest(record)
    parse_calls = []
    prepared = {
        "success": True,
        "documents": [{"page_content": "sheet"}],
        "is_spreadsheet": True,
        "source_type": "spreadsheet",
        "parser_type": "spreadsheet",
        "access_level": "restricted",
        "department": "财务部",
    }
    service = build_service(
        spreadsheet_store=store,
        manifest=manifest,
        upload_manager=FakeUploadManager(prepared=prepared),
        parse_spreadsheet=lambda *args, **kwargs: parse_calls.append((args, kwargs)) or [{"row": 1}, {"row": 2}],
    )

    payload, status = service.reindex_knowledge_file("hash_1", user_id="fallback_user")

    assert status == 200
    assert payload["spreadsheet_rows"] == 2
    assert parse_calls == [((source,), {"display_filename": "source.xlsx"})]
    assert store.deleted == []
    assert store.upserts == [{
        "content_hash": "hash_1",
        "filename": "source.xlsx",
        "source_path": str(source),
        "category": "财务",
        "access_level": "restricted",
        "department": "财务部",
        "uploaded_by": "owner",
        "uploaded_at": "2026-06-01T10:00:00",
        "rows": [{"row": 1}, {"row": 2}],
    }]
    assert manifest.reindexed[0][1]["source_type"] == "spreadsheet"
    assert manifest.reindexed[0][1]["spreadsheet_row_count"] == 2


def test_reindex_knowledge_file_rolls_back_after_mutation_failure(tmp_path):
    source = tmp_path / "source.docx"
    source.write_text("doc", encoding="utf-8")
    record = {
        "content_hash": "hash_1",
        "filename": "source.docx",
        "source_path": str(source),
        "archived_path": "",
    }
    kb = FakeKnowledgeBase(fail_replace=True)
    store = FakeSpreadsheetStore()
    manifest = FakeManifest(record)
    audits = []
    clears = []
    service = build_service(kb=kb, spreadsheet_store=store, manifest=manifest, audits=audits, clears=clears)

    payload, status = service.reindex_knowledge_file("hash_1", user_id="u1")

    assert status == 500
    assert payload["success"] is False
    assert payload["rolled_back"] is True
    assert payload["rollback_errors"] == []
    assert kb.restored == [{"kb": "snapshot"}]
    assert store.restored == [{"sheet": "hash_1"}]
    assert manifest.restored == [record]
    assert clears == []
    assert audits[0][1]["status"] == "failed"
    assert "replace failed" in audits[0][1]["message"]
