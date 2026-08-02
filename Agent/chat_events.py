"""Shared SSE and route helpers for chat streaming services.

All chat streaming responses pass through this module so the backend keeps a
single event contract even while individual pipelines are refactored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CHAT_EVENT_TYPES = {
    "start",
    "session",
    "route",
    "thinking_start",
    "thinking_done",
    "context_start",
    "context_end",
    "plan_start",
    "plan",
    "tool_plan",
    "tool_call",
    "tool_result",
    "tool_confirm_required",
    "write_start",
    "answer_start",
    "answer_delta",
    "answer_done",
    "think",
    "content",
    "reasoning_chunk",
    "reflection",
    "done",
    "run_done",
    "error",
}

DONE_DEFAULTS = {
    "intent": "",
    "answer": "",
    "document": "",
    "session_id": "",
    "plan": {},
    "route": {},
    "actions": [],
    "export_template": "",
    "export_spreadsheet_template": "",
    "source_filenames": [],
    "source_details": [],
    "audit_summary": {},
}


@dataclass(frozen=True)
class ChatEvent:
    type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return normalize_event({"type": self.type, **self.payload})


def normalize_event(payload: dict[str, Any]) -> dict[str, Any]:
    event = dict(payload or {})
    event_type = event.get("type") or "error"
    event["type"] = event_type
    if event_type == "done":
        normalized = dict(DONE_DEFAULTS)
        normalized.update(event)
        return normalized
    if event_type == "error":
        event.setdefault("message", "请求处理失败，请稍后重试")
    if event_type == "content":
        event.setdefault("data", "")
    if event_type == "answer_delta":
        event.setdefault("data", "")
    if event_type == "answer_start":
        event.setdefault("message", "开始输出正文")
    if event_type == "answer_done":
        event.setdefault("answer", "")
    if event_type == "thinking_done":
        event.setdefault("summary", "思考完成")
    if event_type == "think":
        event.setdefault("agent", "")
        event.setdefault("emoji", "")
        event.setdefault("message", "")
    return event


def event(event_type: str, **payload) -> dict[str, Any]:
    return ChatEvent(event_type, payload).to_dict()


def start_event(**payload) -> dict[str, Any]:
    return event("start", **payload)


def session_event(session_id: str) -> dict[str, Any]:
    return event("session", session_id=session_id)


def think_event(agent: str, emoji: str, message: str) -> dict[str, Any]:
    return event("think", agent=agent, emoji=emoji, message=message)


def thinking_start_event(message: str = "开始思考") -> dict[str, Any]:
    return event("thinking_start", message=message)


def thinking_done_event(*, summary: str = "思考完成", step_count: int = 0, elapsed_ms: int = 0) -> dict[str, Any]:
    return event("thinking_done", summary=summary, step_count=step_count, elapsed_ms=elapsed_ms)


def plan_start_event(message: str) -> dict[str, Any]:
    return event("plan_start", message=message)


def plan_event(plan: dict[str, Any]) -> dict[str, Any]:
    return event("plan", data=plan or {})


def write_start_event(message: str) -> dict[str, Any]:
    return event("write_start", message=message)


def content_event(data: str, *, session_id: str = "") -> dict[str, Any]:
    payload = {"data": data}
    if session_id:
        payload["session_id"] = session_id
    return event("content", **payload)


def answer_start_event(message: str = "开始输出正文", *, session_id: str = "") -> dict[str, Any]:
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    return event("answer_start", **payload)


def answer_delta_event(data: str, *, session_id: str = "") -> dict[str, Any]:
    payload = {"data": data}
    if session_id:
        payload["session_id"] = session_id
    return event("answer_delta", **payload)


def answer_done_event(answer: str, *, session_id: str = "") -> dict[str, Any]:
    payload = {"answer": answer}
    if session_id:
        payload["session_id"] = session_id
    return event("answer_done", **payload)


def iter_text_chunks(text: str, chunk_size: int = 18):
    text = text or ""
    if not text:
        return
    for start in range(0, len(text), chunk_size):
        yield text[start:start + chunk_size]


def text_stream_sse(text: str, *, session_id: str = "", chunk_size: int = 18):
    for chunk in iter_text_chunks(text, chunk_size=chunk_size):
        yield sse(answer_delta_event(chunk, session_id=session_id))
        yield sse(content_event(chunk, session_id=session_id))


def reasoning_chunk_event(data: str) -> dict[str, Any]:
    return event("reasoning_chunk", data=data)


def reflection_event(data: Any) -> dict[str, Any]:
    return event("reflection", data=data)


def error_event(message: str) -> dict[str, Any]:
    return event("error", message=message)


def done_event(**payload) -> dict[str, Any]:
    return event("done", **payload)


def run_done_event(**payload) -> dict[str, Any]:
    return event("run_done", **payload)


def sse(payload: dict) -> str:
    return f"data: {json.dumps(normalize_event(payload), ensure_ascii=False)}\n\n"


def parse_sse_events(chunks: Iterable[str]) -> list[dict[str, Any]]:
    events = []
    for chunk in chunks:
        if not isinstance(chunk, str) or not chunk.startswith("data: "):
            continue
        events.append(json.loads(chunk[6:]))
    return events


def route_payload(route) -> dict:
    return route.to_dict() if hasattr(route, "to_dict") else {}


def route_actions(route) -> list:
    payload = route_payload(route)
    return payload.get("actions", []) if isinstance(payload, dict) else []


def route_intent(route, default: str = "") -> str:
    payload = route_payload(route)
    return payload.get("intent", default) if isinstance(payload, dict) else default


def route_template_key(route) -> str:
    payload = route_payload(route)
    return payload.get("template_key", "") if isinstance(payload, dict) else ""


def route_event(route) -> str:
    return sse(event("route", data=route_payload(route)))


def source_details_from_results(results, limit=10):
    """Convert knowledge metadata to frontend-friendly source filenames only."""
    details = []
    seen = set()
    for item in (results or []):
        filename = item.get("filename") or Path(item.get("source", "")).name
        if not filename or filename in seen:
            continue
        seen.add(filename)
        details.append({"filename": filename})
        if len(details) >= limit:
            break
    return details
