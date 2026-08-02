from datetime import datetime
from pathlib import Path

from knowledge_admin_audit_service import (
    KnowledgeAdminAuditDependencies,
    KnowledgeAdminAuditService,
)


class FakeCacheManager:
    def __init__(self):
        self.cleared = 0

    def clear_knowledge_cache(self):
        self.cleared += 1


class FakeKnowledgeAgent:
    def __init__(self):
        self.refreshed = 0
        self.cache_manager = FakeCacheManager()

    def refresh(self):
        self.refreshed += 1


class FakeManifest:
    def __init__(self, db_path=None):
        self.db_path = db_path
        self.audit_calls = []

    def record_audit(self, **kwargs):
        self.audit_calls.append(kwargs)


class FakePathHolder:
    def __init__(self, **paths):
        for key, value in paths.items():
            setattr(self, key, value)


def build_service(tmp_path, *, manifest=None, knowledge_base=None, spreadsheet_store=None, agent=None):
    actor = {"actor_id": "u1", "actor_name": "alice"}
    service = KnowledgeAdminAuditService(KnowledgeAdminAuditDependencies(
        knowledge_agent=agent or FakeKnowledgeAgent(),
        knowledge_base=knowledge_base or FakePathHolder(),
        knowledge_manifest=manifest or FakeManifest(),
        spreadsheet_store=spreadsheet_store or FakePathHolder(),
        admin_backup_dir=tmp_path / "backups",
        actor_provider=lambda: actor,
        now_factory=lambda: datetime(2026, 6, 7, 9, 0, 0),
    ))
    return service


def test_clear_runtime_cache_refreshes_agent_and_cache(tmp_path):
    agent = FakeKnowledgeAgent()
    service = build_service(tmp_path, agent=agent)

    service.clear_runtime_cache()

    assert agent.refreshed == 1
    assert agent.cache_manager.cleared == 1


def test_backup_storage_copies_existing_files_and_ignores_missing(tmp_path):
    index_path = tmp_path / "index.faiss"
    manifest_db = tmp_path / "manifest.sqlite"
    index_path.write_text("index", encoding="utf-8")
    manifest_db.write_text("manifest", encoding="utf-8")
    kb = FakePathHolder(index_path=index_path, pkl_path=tmp_path / "missing.pkl", config_path=None)
    manifest = FakeManifest(db_path=manifest_db)
    service = build_service(tmp_path, knowledge_base=kb, manifest=manifest)

    backup_path = service.backup_storage("delete!", "abcdef1234567890")

    backup = Path(backup_path)
    assert backup.name == "20260607090000_delete__abcdef123456"
    assert (backup / "index.faiss").read_text(encoding="utf-8") == "index"
    assert (backup / "manifest.sqlite").read_text(encoding="utf-8") == "manifest"


def test_backup_storage_removes_empty_backup_directory(tmp_path):
    service = build_service(tmp_path)

    backup_path = service.backup_storage("delete", "hash")

    assert backup_path == ""
    assert not (tmp_path / "backups").exists() or list((tmp_path / "backups").iterdir()) == []


def test_record_audit_includes_actor_and_before_after(tmp_path):
    manifest = FakeManifest()
    service = build_service(tmp_path, manifest=manifest)

    service.record_audit(
        "metadata",
        {"content_hash": "h1", "filename": "a.docx"},
        status="success",
        message="done",
        before={"old": 1},
        after={"new": 2},
    )

    assert manifest.audit_calls == [{
        "content_hash": "h1",
        "filename": "a.docx",
        "action": "metadata",
        "status": "success",
        "message": "done",
        "backup_path": "",
        "before": {"old": 1},
        "after": {"new": 2},
        "actor_id": "u1",
        "actor_name": "alice",
    }]


def test_record_upload_audit_maps_success_duplicate_and_failed(tmp_path):
    manifest = FakeManifest()
    service = build_service(tmp_path, manifest=manifest)

    service.record_upload_audit(
        {"success": True, "content_hash": "h1", "filename": "ok.docx", "message": "ok"},
        filename="ok.docx",
        category="公共资料",
        department="财务部",
    )
    service.record_upload_audit(
        {"success": False, "duplicate": True, "content_hash": "h2", "message": "dup"},
        filename="dup.docx",
        category="公共资料",
    )
    service.record_upload_audit(
        {"success": False, "content_hash": "h3", "message": "bad"},
        filename="bad.docx",
        category="公共资料",
    )

    assert [call["status"] for call in manifest.audit_calls] == ["success", "duplicate", "failed"]
    assert manifest.audit_calls[0]["after"]["filename"] == "ok.docx"
    assert manifest.audit_calls[1]["after"] is None
    assert manifest.audit_calls[2]["filename"] == "bad.docx"
