"""Authentication route service helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass
class AuthRouteDependencies:
    auth_manager: Any
    now_factory: Callable[[], datetime]


@dataclass
class AuthRouteResult:
    payload: dict
    status: int = 200
    set_cookie_token: str = ""
    clear_cookie: bool = False


class LoginRateLimiter:
    def __init__(self, *, max_attempts: int = 5, window_seconds: int = 60, now_factory: Callable[[], datetime]):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.now_factory = now_factory
        self._attempts = {}

    def is_blocked(self, ip: str) -> bool:
        self._cleanup()
        now = self.now_factory().timestamp()
        failures = [
            timestamp for timestamp, ok in self._attempts.get(ip, [])
            if not ok and now - timestamp < self.window_seconds
        ]
        return len(failures) >= self.max_attempts

    def record(self, ip: str, success: bool) -> None:
        now = self.now_factory().timestamp()
        if success:
            self._attempts.pop(ip, None)
            return
        self._attempts.setdefault(ip, []).append((now, success))

    def _cleanup(self) -> None:
        now = self.now_factory().timestamp()
        for ip in list(self._attempts.keys()):
            self._attempts[ip] = [
                item for item in self._attempts[ip]
                if now - item[0] < self.window_seconds
            ]
            if not self._attempts[ip]:
                del self._attempts[ip]


class AuthRouteService:
    def __init__(self, deps: AuthRouteDependencies, *, login_rate_limiter: LoginRateLimiter | None = None):
        self.deps = deps
        self.login_rate_limiter = login_rate_limiter or LoginRateLimiter(now_factory=deps.now_factory)

    def login(self, data: dict, *, ip_address: str, user_agent: str) -> AuthRouteResult:
        ip = ip_address or "unknown"
        if self.login_rate_limiter.is_blocked(ip):
            return AuthRouteResult({"success": False, "message": "尝试次数过多，请稍后再试"}, status=429)

        request_data = data or {}
        username = (request_data.get("username") or "").strip()
        password = request_data.get("password", "")
        if not username or not password:
            return AuthRouteResult({"success": False, "message": "请输入用户名和密码"})

        result = self.deps.auth_manager.login(
            username=username,
            password=password,
            ip_address=ip,
            user_agent=user_agent or "",
        )
        self.login_rate_limiter.record(ip, result.get("success", False))
        return AuthRouteResult(
            result,
            set_cookie_token=result.get("token", "") if result.get("success") else "",
        )

    def logout(self) -> AuthRouteResult:
        return AuthRouteResult({"success": True, "message": "已登出"}, clear_cookie=True)

    def public_register(self, data: dict) -> AuthRouteResult:
        request_data = data or {}
        result = self.deps.auth_manager.register_user(
            username=request_data.get("username"),
            password=request_data.get("password"),
            name=request_data.get("username", ""),
            department="",
            role="user",
        )
        return AuthRouteResult(result)

    def change_password(self, user_id: str, data: dict) -> AuthRouteResult:
        request_data = data or {}
        result = self.deps.auth_manager.change_password(
            user_id=user_id,
            old_password=request_data.get("old_password"),
            new_password=request_data.get("new_password"),
        )
        return AuthRouteResult(result)

    def profile(self, *, method: str, user_id: str, data: dict | None = None) -> AuthRouteResult:
        if method == "PUT":
            return AuthRouteResult(self.deps.auth_manager.update_user(user_id, data or {}))

        user = self.deps.auth_manager.get_user_by_id(user_id)
        if not user:
            return AuthRouteResult({"success": False, "message": "用户不存在"})
        return AuthRouteResult({
            "success": True,
            "user": {
                "user_id": user["user_id"],
                "username": user["username"],
                "name": user["name"],
                "department": user["department"],
                "role": user["role"],
            },
        })
