"""Session, profile, and chat history services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SessionServiceDependencies:
    memory: Any


class SessionService:
    def __init__(self, deps: SessionServiceDependencies):
        self.deps = deps

    def list_sessions(self, user_id: str, *, limit: int = 20) -> dict:
        memory = self.deps.memory
        return {
            "success": True,
            "sessions": memory.list_user_sessions(user_id, limit=limit),
            "stats": memory.get_stats(),
        }

    def create_session(self, user_id: str, data: dict) -> dict:
        request_data = data or {}
        session_id = self.deps.memory.create_session(
            user_id=user_id,
            title=request_data.get("title", ""),
            doc_type=request_data.get("doc_type", ""),
        )
        return {
            "success": True,
            "session_id": session_id,
            "message": "会话创建成功",
        }

    def get_session(self, user_id: str, session_id: str) -> tuple[dict, int]:
        if not self._owns_session(user_id, session_id):
            return {"success": False, "error": "会话不存在或无权限"}, 403

        history = self.deps.memory.get_session_history(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp,
                    "metadata": message.metadata or {},
                }
                for message in history
            ],
        }, 200

    def delete_session(self, user_id: str, session_id: str) -> tuple[dict, int]:
        if not self._owns_session(user_id, session_id):
            return {"success": False, "error": "无权限删除此会话"}, 403

        self.deps.memory.delete_session(session_id)
        return {"success": True, "message": "会话及其会话级记忆已删除"}, 200

    def get_session_messages(self, user_id: str, session_id: str, *, limit: int = 20) -> tuple[dict, int]:
        if not self._owns_session(user_id, session_id):
            return {"success": False, "error": "无权限访问此会话"}, 403

        messages = self.deps.memory.get_session_history(session_id)
        return {
            "success": True,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "metadata": message.metadata or {},
                }
                for message in messages[-limit:]
            ],
            "count": len(messages),
        }, 200

    def get_user_profile(self, user_id: str) -> dict:
        profile = self.deps.memory.get_user_profile(user_id)
        if not profile:
            return {"success": False, "message": "用户不存在"}
        return {
            "success": True,
            "profile": {
                "user_id": profile.user_id,
                "name": profile.name,
                "department": profile.department,
                "preferred_font": profile.preferred_font,
                "preferred_size": profile.preferred_size,
                "common_doc_types": profile.common_doc_types,
                "writing_style": profile.writing_style,
            },
        }

    def update_user_profile(self, user_id: str, data: dict) -> dict:
        return self.deps.memory.update_user_profile(user_id, data or {})

    def list_long_term_memories(self, user_id: str) -> dict:
        return {
            "success": True,
            "memories": self.deps.memory.list_long_term_memories(user_id),
        }

    def forget_long_term_memory(self, user_id: str, memory_id: int) -> tuple[dict, int]:
        if not self.deps.memory.forget_explicit_memory(user_id, memory_id):
            return {"success": False, "error": "记忆不存在、无权限，或应通过资料设置修改"}, 404
        return {"success": True, "message": "已忘记该条用户记忆"}, 200

    def user_stats(self, user_id: str) -> dict:
        user_sessions = self.deps.memory.list_user_sessions(user_id)
        return {
            "success": True,
            "stats": {
                "total_sessions": len(user_sessions),
                "total_messages": sum(session.get("message_count", 0) for session in user_sessions),
                "doc_types": list({
                    session.get("doc_type", "")
                    for session in user_sessions
                    if session.get("doc_type")
                }),
            },
        }

    def search_history(self, user_id: str, data: dict) -> dict:
        keyword = (data or {}).get("keyword", "")
        sessions = self.deps.memory.list_user_sessions(user_id, limit=100)

        results = []
        for session in sessions:
            history = self.deps.memory.get_session_history(session["session_id"])
            for message in history:
                if keyword in message.content:
                    results.append({
                        "session_id": session["session_id"],
                        "session_title": session.get("title", "未命名"),
                        "doc_type": session.get("doc_type", "通用"),
                        "role": message.role,
                        "content": (
                            message.content[:100] + "..."
                            if len(message.content) > 100
                            else message.content
                        ),
                        "timestamp": message.timestamp,
                    })
                    break

        return {
            "success": True,
            "keyword": keyword,
            "results": results,
            "total": len(results),
        }

    def _owns_session(self, user_id: str, session_id: str) -> bool:
        return self.deps.memory.get_owned_session(user_id, session_id) == session_id
