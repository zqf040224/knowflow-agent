from io import BytesIO
from types import SimpleNamespace

from flask import g


def test_auth_login_route_sets_cookie_from_service(monkeypatch):
    import app as app_module

    class FakeAuthService:
        def login(self, data, *, ip_address, user_agent):
            assert data == {"username": "alice", "password": "pw"}
            assert ip_address == "127.0.0.1"
            assert user_agent == "pytest"
            return SimpleNamespace(
                payload={"success": True, "token": "jwt"},
                status=200,
                set_cookie_token="jwt",
            )

    monkeypatch.setattr(app_module.context, "auth_route_service", FakeAuthService())

    with app_module.app.test_request_context(
        "/api/auth/login",
        method="POST",
        json={"username": "alice", "password": "pw"},
        headers={"User-Agent": "pytest"},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    ):
        response, status = app_module.api_login.__wrapped__()

    assert status == 200
    assert response.get_json() == {"success": True, "token": "jwt"}
    assert "token=jwt" in response.headers["Set-Cookie"]


def test_auth_profile_get_does_not_require_json_content_type(monkeypatch):
    import app as app_module

    calls = []

    class FakeAuthService:
        def profile(self, *, method, user_id, data):
            calls.append((method, user_id, data))
            return SimpleNamespace(
                payload={"success": True, "user": {"user_id": user_id}},
                status=200,
            )

    monkeypatch.setattr(app_module.context, "auth_route_service", FakeAuthService())

    with app_module.app.test_request_context("/api/auth/profile", method="GET"):
        g.user_id = "u1"
        response, status = app_module.app.view_functions["api_profile"].__wrapped__()

    assert status == 200
    assert response.get_json() == {"success": True, "user": {"user_id": "u1"}}
    assert calls == [("GET", "u1", {})]


def test_search_and_spreadsheet_routes_delegate_to_runtime_query_service(monkeypatch):
    import app as app_module

    calls = []

    class FakeRuntimeService:
        def search(self, data, *, user_info):
            calls.append(("search", data, user_info))
            return [{"filename": "a.docx"}]

        def spreadsheet_query(self, data, *, user_info):
            calls.append(("spreadsheet", data, user_info))
            return {"success": True, "rows": [], "count": 0}

    user_info = object()
    monkeypatch.setattr(app_module.context, "runtime_query_service", lambda: FakeRuntimeService())
    monkeypatch.setattr(app_module.context, "get_user_info", lambda: user_info)

    with app_module.app.test_request_context("/api/search", method="POST", json={"query": "制度"}):
        search_response = app_module.search.__wrapped__()
    with app_module.app.test_request_context("/api/spreadsheets/query", method="POST", json={"keyword": "预算"}):
        spreadsheet_response = app_module.query_spreadsheets.__wrapped__()

    assert search_response.get_json() == [{"filename": "a.docx"}]
    assert spreadsheet_response.get_json() == {"success": True, "rows": [], "count": 0}
    assert calls == [
        ("search", {"query": "制度"}, user_info),
        ("spreadsheet", {"keyword": "预算"}, user_info),
    ]


def test_admin_users_get_does_not_require_json_content_type(monkeypatch):
    import app as app_module

    calls = []

    class FakeAccountAdminService:
        def users(self, user_id, data, *, method):
            calls.append((user_id, data, method))
            return {"success": True, "users": []}

    monkeypatch.setattr(app_module.context, "account_admin_service", FakeAccountAdminService())

    with app_module.app.test_request_context("/api/admin/users", method="GET"):
        g.user_id = "admin_1"
        response = app_module.app.view_functions["admin_users"].__wrapped__.__wrapped__()

    assert response.get_json() == {"success": True, "users": []}
    assert calls == [("admin_1", {}, "GET")]


def test_chat_route_delegates_only_to_chat_runtime(monkeypatch):
    import app as app_module

    calls = []
    user_info = object()

    class FakeRuntime:
        def stream(self, data, *, user_id, user_info):
            calls.append((data, user_id, user_info))
            yield "data: {}\n\n"

    monkeypatch.setattr(app_module.context, "chat_runtime", lambda: FakeRuntime())
    monkeypatch.setattr(app_module.context, "get_user_info", lambda: user_info)

    with app_module.app.test_request_context("/api/chat", method="POST", json={"message": "你好"}):
        g.user_id = "user_1"
        response = app_module.app.view_functions["chat"].__wrapped__.__wrapped__()

    assert response.mimetype == "text/event-stream"
    assert "data: {}" in "".join(response.response)
    assert calls == [({"message": "你好"}, "user_1", user_info)]


def test_reindex_route_returns_404_before_job_when_record_missing(monkeypatch):
    import app as app_module

    class FakeManifest:
        def get_record(self, content_hash):
            assert content_hash == "missing"
            return None

    monkeypatch.setattr(app_module.context, "knowledge_manifest", FakeManifest())

    with app_module.app.test_request_context("/api/admin/knowledge-files/missing/reindex", method="POST"):
        response, status = app_module.app.view_functions["admin_reindex_knowledge_file"].__wrapped__.__wrapped__("missing")

    assert status == 404
    assert response.get_json() == {"success": False, "message": "未找到该知识库文件"}


def test_reindex_route_submits_background_job(monkeypatch):
    import app as app_module

    calls = []

    class FakeManifest:
        def get_record(self, content_hash):
            return {"content_hash": content_hash}

    class FakeWriteService:
        def reindex_knowledge_file(self, content_hash, *, user_id):
            calls.append(("reindex", content_hash, user_id))
            return {"success": True, "message": "ok"}, 200

    class FakeJobService:
        def submit(self, job_type, user_id, payload, runner, *, message):
            calls.append(("submit", job_type, user_id, payload, message))
            assert runner() == ({"success": True, "message": "ok"}, 200)
            return {"success": True, "job_id": "job_1", "status": "queued", "message": message}

    monkeypatch.setattr(app_module.context, "knowledge_manifest", FakeManifest())
    monkeypatch.setattr(app_module.context, "knowledge_admin_write_service", lambda: FakeWriteService())
    monkeypatch.setattr(app_module.context, "job_service", lambda: FakeJobService())

    with app_module.app.test_request_context("/api/admin/knowledge-files/hash_1/reindex", method="POST"):
        g.user_id = "admin_1"
        response, status = app_module.app.view_functions["admin_reindex_knowledge_file"].__wrapped__.__wrapped__("hash_1")

    assert status == 202
    assert response.get_json()["job_id"] == "job_1"
    assert calls == [
        ("submit", "knowledge_reindex", "admin_1", {"content_hash": "hash_1"}, "重建任务已提交"),
        ("reindex", "hash_1", "admin_1"),
    ]


def test_job_route_delegates_to_job_service(monkeypatch):
    import app as app_module

    class FakeJobService:
        def get_job(self, job_id, *, user_id, role):
            return {
                "success": True,
                "job": {
                    "job_id": job_id,
                    "type": "knowledge_upload",
                    "status": "succeeded",
                    "message": "done",
                    "result": {},
                    "error": "",
                    "created_at": "",
                    "started_at": "",
                    "finished_at": "",
                },
            }, 200

    monkeypatch.setattr(app_module.context, "job_service", lambda: FakeJobService())

    with app_module.app.test_request_context("/api/jobs/job_1", method="GET"):
        g.user_id = "u1"
        g.role = "user"
        response, status = app_module.app.view_functions["get_job"].__wrapped__("job_1")

    assert status == 200
    assert response.get_json()["job"]["job_id"] == "job_1"


def test_knowledge_upload_route_submits_background_job(monkeypatch):
    import app as app_module

    calls = []
    prepared = {
        "success": True,
        "file_path": "/tmp/upload.docx",
        "filename": "upload.docx",
        "user_id": "u1",
        "category": "公共资料",
        "department": "",
    }

    class FakeUploadService:
        def prepare_knowledge_upload(self, file, **kwargs):
            calls.append(("prepare", file.filename, kwargs))
            return prepared, 200

        def process_prepared_knowledge_upload(self, payload):
            calls.append(("process", payload))
            return {"success": True, "message": "已入库"}, 200

    class FakeJobService:
        def submit(self, job_type, user_id, payload, runner, *, message):
            calls.append(("submit", job_type, user_id, payload, message))
            assert runner() == ({"success": True, "message": "已入库"}, 200)
            return {"success": True, "job_id": "job_upload", "status": "queued", "message": message}

    monkeypatch.setattr(app_module.context, "upload_service", lambda: FakeUploadService())
    monkeypatch.setattr(app_module.context, "job_service", lambda: FakeJobService())

    with app_module.app.test_request_context(
        "/api/upload",
        method="POST",
        data={
            "mode": "knowledge",
            "category": "公共资料",
            "file": (BytesIO(b"content"), "upload.docx"),
        },
        content_type="multipart/form-data",
    ):
        g.user_id = "u1"
        g.role = "admin"
        g.department = ""
        response, status = app_module.app.view_functions["upload_file"].__wrapped__()

    assert status == 202
    assert response.get_json()["job_id"] == "job_upload"
    assert calls[0][0] == "prepare"
    assert calls[1] == (
        "submit",
        "knowledge_upload",
        "u1",
        {"filename": "upload.docx", "category": "公共资料", "department": ""},
        "上传入库任务已提交",
    )
    assert calls[2] == ("process", prepared)


def test_knowledge_upload_route_cleans_prepared_file_when_job_submit_fails(monkeypatch):
    import app as app_module

    calls = []
    prepared = {
        "success": True,
        "file_path": "/tmp/upload.docx",
        "filename": "upload.docx",
        "user_id": "u1",
        "category": "公共资料",
        "department": "",
    }

    class FakeUploadService:
        def prepare_knowledge_upload(self, file, **kwargs):
            calls.append(("prepare", file.filename, kwargs))
            return prepared, 200

        def cleanup_prepared_upload(self, payload):
            calls.append(("cleanup", payload))

    class FailingJobService:
        def submit(self, job_type, user_id, payload, runner, *, message):
            calls.append(("submit", job_type, user_id, payload, message))
            raise RuntimeError("sqlite locked")

    monkeypatch.setattr(app_module.context, "upload_service", lambda: FakeUploadService())
    monkeypatch.setattr(app_module.context, "job_service", lambda: FailingJobService())

    with app_module.app.test_request_context(
        "/api/upload",
        method="POST",
        data={
            "mode": "knowledge",
            "category": "公共资料",
            "file": (BytesIO(b"content"), "upload.docx"),
        },
        content_type="multipart/form-data",
    ):
        g.user_id = "u1"
        g.role = "admin"
        g.department = ""
        response, status = app_module.app.view_functions["upload_file"].__wrapped__()

    assert status == 500
    assert response.get_json() == {"success": False, "message": "上传入库任务提交失败，请重试"}
    assert calls[0][0] == "prepare"
    assert calls[1][0] == "submit"
    assert calls[2] == ("cleanup", prepared)
