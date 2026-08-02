"""Admin account-management service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AccountAdminDependencies:
    auth_manager: Any
    memory: Any
    knowledge_source_dir: Path
    department_dirs: set[str]


class AccountAdminService:
    def __init__(self, deps: AccountAdminDependencies):
        self.deps = deps

    def existing_departments(self) -> list:
        return sorted(
            dept for dept in self.deps.department_dirs
            if (self.deps.knowledge_source_dir / dept).is_dir()
        )

    def departments_payload(self) -> dict:
        return {
            "success": True,
            "departments": [{"id": dept, "name": dept} for dept in self.existing_departments()],
        }

    def validate_department(self, role: str, department: str) -> tuple[bool, str, str]:
        role = role if role in {"user", "admin"} else "user"
        department = (department or "").strip()
        existing_departments = set(self.existing_departments())

        if role == "user" and not department:
            return False, department, "普通用户必须选择所属部门"
        if department and department not in existing_departments:
            return False, department, "部门不存在，请从已有部门中选择"
        return True, department, ""

    def register_from_admin_payload(self, data: dict) -> dict:
        request_data = data or {}
        role = request_data.get("role", "user")
        ok, department, message = self.validate_department(role, request_data.get("department", ""))
        if not ok:
            return {"success": False, "message": message}

        return self.deps.auth_manager.register_user(
            username=request_data.get("username"),
            password=request_data.get("password"),
            name=request_data.get("name", ""),
            department=department,
            role=role if role in {"user", "admin"} else "user",
        )

    def users(self, admin_user_id: str, data: dict | None = None, *, method: str = "GET") -> dict:
        if method == "POST":
            return self.register_from_admin_payload(data or {})
        return self.deps.auth_manager.list_users(admin_user_id)

    def toggle_user_status(self, admin_user_id: str, user_id: str, data: dict) -> dict:
        return self.deps.auth_manager.toggle_user_status(
            admin_user_id,
            user_id,
            (data or {}).get("is_active", True),
        )

    def reset_password(self, admin_user_id: str, user_id: str, data: dict) -> dict:
        return self.deps.auth_manager.reset_password(
            admin_user_id,
            user_id,
            (data or {}).get("new_password"),
        )

    def login_logs(self, admin_user_id: str) -> dict:
        return self.deps.auth_manager.get_login_logs(admin_user_id)

    def stats(self, admin_user_id: str) -> dict:
        users_result = self.deps.auth_manager.list_users(admin_user_id)
        today_logins = self.deps.auth_manager.get_login_logs(admin_user_id)
        total_sessions = self.deps.memory.get_stats().get("total_sessions", 0)
        return {
            "success": True,
            "stats": {
                "users": len(users_result.get("users", [])),
                "active_sessions": total_sessions,
                "today_logins": len(today_logins.get("logs", [])),
                "total_sessions": total_sessions,
            },
        }
