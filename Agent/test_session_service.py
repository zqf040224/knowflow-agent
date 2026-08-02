from types import SimpleNamespace

from session_service import SessionService, SessionServiceDependencies


def message(role, content, timestamp="2026-06-07T00:00:00", metadata=None):
    return SimpleNamespace(
        role=role,
        content=content,
        timestamp=timestamp,
        metadata=metadata or {},
    )


class FakeMemory:
    def __init__(self):
        self.sessions = [
            {"session_id": "s1", "title": "会话一", "doc_type": "通知", "message_count": 2},
            {"session_id": "s2", "title": "会话二", "doc_type": "", "message_count": 1},
        ]
        self.histories = {
            "s1": [
                message("user", "查询制度", metadata={"a": 1}),
                message("assistant", "制度内容很长" * 20),
            ],
            "s2": [message("user", "其他内容")],
        }
        self.deleted = []
        self.profile = SimpleNamespace(
            user_id="user_1",
            name="测试用户",
            department="测试部门",
            preferred_font="仿宋",
            preferred_size="三号",
            common_doc_types=["通知"],
            writing_style="简洁正式",
        )
        self.profile_updates = []
        self.forgotten_memory_ids = []

    def list_user_sessions(self, user_id, limit=10):
        return self.sessions[:limit]

    def get_stats(self):
        return {"session_count": len(self.sessions)}

    def create_session(self, user_id, title="", doc_type=""):
        return "new_session"

    def get_owned_session(self, user_id, session_id):
        return session_id if session_id in self.histories and user_id == "user_1" else None

    def get_session_history(self, session_id):
        return self.histories[session_id]

    def delete_session(self, session_id):
        self.deleted.append(session_id)
        return True

    def get_user_profile(self, user_id):
        return self.profile if user_id == "user_1" else None

    def update_user_profile(self, user_id, data):
        self.profile_updates.append((user_id, data))
        return {"success": True, "profile": data}

    def list_long_term_memories(self, user_id):
        return [{"id": 7, "memory_type": "explicit_user_memory", "content": "每周五提交周报"}]

    def forget_explicit_memory(self, user_id, memory_id):
        self.forgotten_memory_ids.append((user_id, memory_id))
        return user_id == "user_1" and memory_id == 7


def build_service(memory=None):
    return SessionService(SessionServiceDependencies(memory=memory or FakeMemory()))


def test_session_service_lists_and_creates_sessions():
    service = build_service()

    listed = service.list_sessions("user_1", limit=1)
    created = service.create_session("user_1", {"title": "新会话", "doc_type": "报告"})

    assert listed["success"] is True
    assert listed["sessions"] == [{"session_id": "s1", "title": "会话一", "doc_type": "通知", "message_count": 2}]
    assert listed["stats"] == {"session_count": 2}
    assert created == {"success": True, "session_id": "new_session", "message": "会话创建成功"}


def test_session_service_serializes_session_and_denies_foreign_access():
    service = build_service()

    payload, status = service.get_session("user_1", "s1")
    denied, denied_status = service.get_session("user_2", "s1")

    assert status == 200
    assert payload["messages"][0] == {
        "role": "user",
        "content": "查询制度",
        "timestamp": "2026-06-07T00:00:00",
        "metadata": {"a": 1},
    }
    assert denied_status == 403
    assert denied == {"success": False, "error": "会话不存在或无权限"}


def test_session_service_deletes_only_owned_sessions():
    memory = FakeMemory()
    service = build_service(memory)

    payload, status = service.delete_session("user_1", "s1")
    denied, denied_status = service.delete_session("user_2", "s1")

    assert status == 200
    assert payload == {"success": True, "message": "会话及其会话级记忆已删除"}
    assert memory.deleted == ["s1"]
    assert denied_status == 403
    assert denied == {"success": False, "error": "无权限删除此会话"}


def test_session_service_limits_messages_and_profile_operations():
    memory = FakeMemory()
    service = build_service(memory)

    messages_payload, status = service.get_session_messages("user_1", "s1", limit=1)
    profile = service.get_user_profile("user_1")
    missing_profile = service.get_user_profile("other")
    updated = service.update_user_profile("user_1", {"preferred_font": "黑体"})

    assert status == 200
    assert messages_payload["count"] == 2
    assert len(messages_payload["messages"]) == 1
    assert profile["profile"]["name"] == "测试用户"
    assert profile["profile"]["common_doc_types"] == ["通知"]
    assert missing_profile == {"success": False, "message": "用户不存在"}
    assert updated == {"success": True, "profile": {"preferred_font": "黑体"}}
    assert memory.profile_updates == [("user_1", {"preferred_font": "黑体"})]


def test_session_service_lists_and_forgets_only_explicit_memories():
    memory = FakeMemory()
    service = build_service(memory)

    listed = service.list_long_term_memories("user_1")
    forgotten, status = service.forget_long_term_memory("user_1", 7)
    missing, missing_status = service.forget_long_term_memory("user_1", 8)

    assert listed["memories"][0]["content"] == "每周五提交周报"
    assert status == 200
    assert forgotten == {"success": True, "message": "已忘记该条用户记忆"}
    assert missing_status == 404
    assert memory.forgotten_memory_ids == [("user_1", 7), ("user_1", 8)]
    assert "资料设置" in missing["error"]


def test_session_service_user_stats_and_search_history():
    service = build_service()

    stats = service.user_stats("user_1")
    search = service.search_history("user_1", {"keyword": "很长"})

    assert stats == {
        "success": True,
        "stats": {
            "total_sessions": 2,
            "total_messages": 3,
            "doc_types": ["通知"],
        },
    }
    assert search["success"] is True
    assert search["keyword"] == "很长"
    assert search["total"] == 1
    assert search["results"][0]["session_id"] == "s1"
    assert search["results"][0]["content"].endswith("...")
