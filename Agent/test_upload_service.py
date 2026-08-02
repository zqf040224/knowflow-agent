from pathlib import Path
from types import SimpleNamespace

from upload_service import UploadService, UploadServiceDependencies


class FakeUploadManager:
    def __init__(self):
        self.valid = (True, "OK")
        self.saved = []
        self.temp_contents = {}
        self.knowledge_results = []
        self.stats = {"temp_files": 1}

    def validate_file(self, file):
        return self.valid

    def save_temp_file(self, file, user_id):
        self.saved.append((file, user_id))
        return "/tmp/upload.txt", "上传.txt"

    def process_temp_upload(self, file_path, filename, user_id):
        return {
            "success": True,
            "file_id": "file_1",
            "filename": filename,
            "user_id": user_id,
        }

    def process_knowledge_upload(self, **kwargs):
        self.knowledge_results.append(kwargs)
        return {"success": True, "content_hash": "hash_1"}

    def get_temp_content(self, file_id, user_id):
        return self.temp_contents.get((file_id, user_id))

    def get_upload_stats(self):
        return self.stats


class FakeMemory:
    def __init__(self):
        self.messages = []

    def get_or_create_session(self, user_id):
        return "session_1"

    def add_message(self, **kwargs):
        self.messages.append(kwargs)


class FakeKnowledgeAgent:
    def __init__(self):
        self.refreshed = False
        self.cache_manager = SimpleNamespace(cleared=False, clear_knowledge_cache=self.clear_cache)

    def refresh(self):
        self.refreshed = True

    def clear_cache(self):
        self.cache_manager.cleared = True


class FakeManifest:
    def consistency_report(self, knowledge_base, spreadsheet_db_path):
        return {
            "ok": False,
            "checked_count": 2,
            "issue_count": 11,
            "issues": list(range(12)),
        }


def build_service(tmp_path, upload_manager=None, memory=None, knowledge_agent=None, audits=None):
    root = tmp_path / "kb"
    (root / "公共资料").mkdir(parents=True)
    (root / "财务部").mkdir()
    (root / "示例部门甲").mkdir()
    audit_calls = audits if audits is not None else []
    return UploadService(UploadServiceDependencies(
        upload_manager=upload_manager or FakeUploadManager(),
        memory=memory or FakeMemory(),
        knowledge_base=object(),
        knowledge_agent=knowledge_agent or FakeKnowledgeAgent(),
        knowledge_manifest=FakeManifest(),
        knowledge_source_dir=root,
        spreadsheet_db_path=Path(":memory:"),
        department_dirs={"财务部", "示例部门甲", "不存在部门"},
        record_knowledge_upload_audit=lambda *args, **kwargs: audit_calls.append((args, kwargs)),
    ))


def test_upload_service_filters_categories_by_role_and_department(tmp_path):
    service = build_service(tmp_path)

    user_categories = service.allowed_categories(user_role="user", user_department="财务部")
    admin_categories = service.allowed_categories(user_role="admin", user_department="")

    assert [item["id"] for item in user_categories] == ["公共资料", "财务部"]
    assert [item["id"] for item in admin_categories] == ["公共资料", "示例部门甲", "财务部"]


def test_upload_service_temp_upload_and_preview(tmp_path):
    upload_manager = FakeUploadManager()
    upload_manager.temp_contents[("file_1", "user_1")] = "x" * 501
    service = build_service(tmp_path, upload_manager=upload_manager)

    result, status = service.upload_file(object(), user_id="user_1", user_role="user", user_department="财务部")
    preview, preview_status = service.get_temp_file_content("file_1", user_id="user_1")

    assert status == 200
    assert result["success"] is True
    assert result["filename"] == "上传.txt"
    assert preview_status == 200
    assert preview["char_count"] == 501
    assert preview["content"].endswith("...")


def test_upload_service_rejects_invalid_and_missing_temp_content(tmp_path):
    upload_manager = FakeUploadManager()
    upload_manager.valid = (False, "文件为空")
    service = build_service(tmp_path, upload_manager=upload_manager)

    invalid, invalid_status = service.upload_file(object(), user_id="user_1", user_role="user", user_department="")
    missing, missing_status = service.get_temp_file_content("missing", user_id="user_1")

    assert invalid_status == 400
    assert invalid == {"success": False, "message": "文件为空"}
    assert missing_status == 404
    assert missing == {"success": False, "message": "文件不存在或已过期"}


def test_upload_service_knowledge_upload_rejects_missing_or_forbidden_category(tmp_path):
    service = build_service(tmp_path)

    missing, missing_status = service.upload_file(
        object(),
        user_id="user_1",
        user_role="user",
        user_department="财务部",
        mode="knowledge",
        category="",
    )
    forbidden, forbidden_status = service.upload_file(
        object(),
        user_id="user_1",
        user_role="user",
        user_department="财务部",
        mode="knowledge",
        category="示例部门甲",
    )

    assert missing_status == 400
    assert missing == {"success": False, "message": "请选择知识库分类"}
    assert forbidden_status == 403
    assert forbidden == {"success": False, "message": "无权限上传到该知识库分类"}


def test_upload_service_knowledge_upload_success_side_effects(tmp_path):
    upload_manager = FakeUploadManager()
    memory = FakeMemory()
    knowledge_agent = FakeKnowledgeAgent()
    audits = []
    service = build_service(
        tmp_path,
        upload_manager=upload_manager,
        memory=memory,
        knowledge_agent=knowledge_agent,
        audits=audits,
    )

    result, status = service.upload_file(
        object(),
        user_id="user_1",
        user_role="user",
        user_department="财务部",
        mode="knowledge",
        category="财务部",
    )

    assert status == 200
    assert result == {"success": True, "content_hash": "hash_1"}
    assert upload_manager.knowledge_results[0]["category"] == "财务部"
    assert upload_manager.knowledge_results[0]["department"] == "财务部"
    assert knowledge_agent.refreshed is True
    assert knowledge_agent.cache_manager.cleared is True
    assert memory.messages[0]["metadata"] == {"type": "upload", "category": "财务部", "filename": "上传.txt"}
    assert audits[0][1] == {"filename": "上传.txt", "category": "财务部", "department": "财务部"}


def test_upload_service_stats_truncates_consistency_issues(tmp_path):
    service = build_service(tmp_path)

    payload = service.upload_stats()

    assert payload["success"] is True
    assert payload["stats"] == {"temp_files": 1}
    assert payload["ingestion_consistency"]["ok"] is False
    assert payload["ingestion_consistency"]["issues"] == list(range(10))
