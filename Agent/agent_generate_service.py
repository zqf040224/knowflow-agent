"""Non-streaming document generation service for the legacy agent endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AgentGenerateDependencies:
    memory: Any
    orchestrator_factory: Callable[..., Any]


class AgentGenerateService:
    def __init__(self, deps: AgentGenerateDependencies):
        self.deps = deps

    def generate(self, data: dict, *, user_id: str, user_info: Any = None) -> dict:
        request_data = data or {}
        message = request_data.get("message", "")
        session_id = self.deps.memory.get_or_create_session(user_id, request_data.get("session_id"))
        remember_explicit = getattr(self.deps.memory, "remember_explicit_memory", None)
        if callable(remember_explicit):
            remember_explicit(user_id, session_id, message)
        think_log = []

        def on_think(agent_name, emoji, msg):
            think_log.append({"agent": agent_name, "emoji": emoji, "message": msg})

        profile = self.deps.memory.get_user_profile(user_id)
        format_info = ""
        if profile:
            format_info = f"用户偏好：{profile.preferred_font} {profile.preferred_size}"

        runner = self.deps.orchestrator_factory(session_id, profile=profile, user_info=user_info)
        result = runner.run(
            message + (f"\n\n[{format_info}]" if format_info else ""),
            on_think=on_think,
            session_id=session_id,
        )

        return {
            "document": result["document"],
            "plan": result["plan"],
            "think_log": think_log,
            "confidence": result["confidence"],
            "revision_rounds": result["revision_rounds"],
        }
