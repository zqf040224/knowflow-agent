from account_admin_service import AccountAdminDependencies, AccountAdminService


class FakeAuthManager:
    def __init__(self):
        self.register_calls = []
        self.toggle_calls = []
        self.reset_calls = []

    def register_user(self, **kwargs):
        self.register_calls.append(kwargs)
        return {"success": True, "user_id": "u2"}

    def list_users(self, admin_user_id):
        return {"success": True, "users": [{"user_id": "u1"}, {"user_id": "u2"}]}

    def toggle_user_status(self, admin_user_id, user_id, is_active):
        self.toggle_calls.append((admin_user_id, user_id, is_active))
        return {"success": True}

    def reset_password(self, admin_user_id, user_id, new_password):
        self.reset_calls.append((admin_user_id, user_id, new_password))
        return {"success": True}

    def get_login_logs(self, admin_user_id):
        return {"success": True, "logs": [{"user_id": "u1"}]}


class FakeMemory:
    def get_stats(self):
        return {"total_sessions": 3}


def build_service(tmp_path):
    (tmp_path / "示例部门甲").mkdir()
    auth = FakeAuthManager()
    service = AccountAdminService(AccountAdminDependencies(
        auth_manager=auth,
        memory=FakeMemory(),
        knowledge_source_dir=tmp_path,
        department_dirs={"示例部门甲", "示例部门乙"},
    ))
    return service, auth


def test_departments_and_registration_validation(tmp_path):
    service, auth = build_service(tmp_path)

    assert service.existing_departments() == ["示例部门甲"]
    assert service.departments_payload() == {
        "success": True,
        "departments": [{"id": "示例部门甲", "name": "示例部门甲"}],
    }
    assert service.register_from_admin_payload({
        "username": "alice",
        "password": "pw",
        "name": "Alice",
        "role": "user",
        "department": "",
    }) == {"success": False, "message": "普通用户必须选择所属部门"}

    result = service.register_from_admin_payload({
        "username": "alice",
        "password": "pw",
        "name": "Alice",
        "role": "user",
        "department": "示例部门甲",
    })

    assert result == {"success": True, "user_id": "u2"}
    assert auth.register_calls[0]["department"] == "示例部门甲"


def test_admin_actions_delegate_to_auth_manager(tmp_path):
    service, auth = build_service(tmp_path)

    assert service.users("admin")["success"] is True
    assert service.toggle_user_status("admin", "u1", {"is_active": False}) == {"success": True}
    assert service.reset_password("admin", "u1", {"new_password": "pw2"}) == {"success": True}
    assert service.login_logs("admin") == {"success": True, "logs": [{"user_id": "u1"}]}
    assert service.stats("admin") == {
        "success": True,
        "stats": {
            "users": 2,
            "active_sessions": 3,
            "today_logins": 1,
            "total_sessions": 3,
        },
    }
    assert auth.toggle_calls == [("admin", "u1", False)]
    assert auth.reset_calls == [("admin", "u1", "pw2")]
