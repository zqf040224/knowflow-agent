from datetime import datetime, timedelta

from auth_route_service import AuthRouteDependencies, AuthRouteService, LoginRateLimiter


class FakeAuthManager:
    def __init__(self):
        self.login_results = []
        self.login_calls = []
        self.register_calls = []
        self.password_calls = []
        self.users = {
            "u1": {
                "user_id": "u1",
                "username": "alice",
                "name": "Alice",
                "department": "财务部",
                "role": "user",
            }
        }
        self.update_calls = []

    def login(self, **kwargs):
        self.login_calls.append(kwargs)
        if self.login_results:
            return self.login_results.pop(0)
        return {"success": False, "message": "bad"}

    def register_user(self, **kwargs):
        self.register_calls.append(kwargs)
        return {"success": True, "user_id": "new"}

    def change_password(self, **kwargs):
        self.password_calls.append(kwargs)
        return {"success": True}

    def get_user_by_id(self, user_id):
        return self.users.get(user_id)

    def update_user(self, user_id, data):
        self.update_calls.append((user_id, data))
        return {"success": True, "user_id": user_id}


def build_service(auth_manager=None, now=None):
    clock = {"now": now or datetime(2026, 6, 7, 9, 0, 0)}
    service = AuthRouteService(AuthRouteDependencies(
        auth_manager=auth_manager or FakeAuthManager(),
        now_factory=lambda: clock["now"],
    ))
    return service, clock


def test_login_success_sets_cookie_token_and_records_auth_call():
    auth = FakeAuthManager()
    auth.login_results.append({"success": True, "token": "jwt", "message": "ok"})
    service, _clock = build_service(auth)

    result = service.login(
        {"username": " alice ", "password": "pw"},
        ip_address="1.2.3.4",
        user_agent="pytest",
    )

    assert result.status == 200
    assert result.payload["success"] is True
    assert result.set_cookie_token == "jwt"
    assert auth.login_calls == [{
        "username": "alice",
        "password": "pw",
        "ip_address": "1.2.3.4",
        "user_agent": "pytest",
    }]


def test_login_validates_credentials_before_auth_manager():
    auth = FakeAuthManager()
    service, _clock = build_service(auth)

    result = service.login({"username": "", "password": ""}, ip_address="", user_agent="")

    assert result.payload == {"success": False, "message": "请输入用户名和密码"}
    assert result.status == 200
    assert auth.login_calls == []


def test_login_rate_limiter_blocks_after_failed_attempts_and_resets_on_success():
    auth = FakeAuthManager()
    auth.login_results.extend([{"success": False} for _ in range(5)])
    service, clock = build_service(auth)

    for _ in range(5):
        service.login({"username": "alice", "password": "bad"}, ip_address="ip", user_agent="")
    blocked = service.login({"username": "alice", "password": "bad"}, ip_address="ip", user_agent="")

    assert blocked.status == 429
    assert blocked.payload["message"] == "尝试次数过多，请稍后再试"

    clock["now"] += timedelta(seconds=61)
    auth.login_results.append({"success": True, "token": "jwt"})
    success = service.login({"username": "alice", "password": "pw"}, ip_address="ip", user_agent="")

    assert success.status == 200
    assert success.set_cookie_token == "jwt"


def test_public_register_forces_plain_user_and_empty_department():
    auth = FakeAuthManager()
    service, _clock = build_service(auth)

    result = service.public_register({
        "username": "bob",
        "password": "pw",
        "role": "admin",
        "department": "财务部",
    })

    assert result.payload["success"] is True
    assert auth.register_calls == [{
        "username": "bob",
        "password": "pw",
        "name": "bob",
        "department": "",
        "role": "user",
    }]


def test_profile_get_put_and_change_password():
    auth = FakeAuthManager()
    service, _clock = build_service(auth)

    profile = service.profile(method="GET", user_id="u1")
    missing = service.profile(method="GET", user_id="missing")
    updated = service.profile(method="PUT", user_id="u1", data={"name": "A"})
    password = service.change_password("u1", {"old_password": "old", "new_password": "new"})

    assert profile.payload["user"]["department"] == "财务部"
    assert missing.payload == {"success": False, "message": "用户不存在"}
    assert updated.payload == {"success": True, "user_id": "u1"}
    assert auth.update_calls == [("u1", {"name": "A"})]
    assert password.payload == {"success": True}
    assert auth.password_calls == [{
        "user_id": "u1",
        "old_password": "old",
        "new_password": "new",
    }]


def test_login_rate_limiter_cleanup_removes_expired_failures():
    now = {"value": datetime(2026, 6, 7, 9, 0, 0)}
    limiter = LoginRateLimiter(max_attempts=1, window_seconds=60, now_factory=lambda: now["value"])

    limiter.record("ip", success=False)
    assert limiter.is_blocked("ip") is True

    now["value"] += timedelta(seconds=61)
    assert limiter.is_blocked("ip") is False
