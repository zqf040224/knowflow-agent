import json
from types import SimpleNamespace

from chat_draft import DocumentDraftDependencies, DocumentDraftStreamService
from agents.document_stream_runner import DocumentStreamRunner


class FakeMemory:
    def __init__(self):
        self.profile = SimpleNamespace(
            preferred_font="仿宋",
            preferred_size="三号",
            writing_style="简洁正式",
            common_doc_types=[],
        )
        self.profile_updates = []

    def get_user_profile(self, user_id):
        return self.profile

    def get_context_for_prompt(self, session_id, max_messages=5):
        return "user: 上一轮"

    def update_user_profile(self, user_id, data):
        self.profile_updates.append((user_id, data))


class FakeRunner:
    def __init__(self):
        self.requests = []

    def run_stream(self, message, on_think=None, session_id=None):
        self.requests.append((message, session_id))
        if on_think:
            on_think("Planner", "🧭", "已制定计划")
        yield {"type": "think", "agent": "Planner", "emoji": "🧭", "message": "已制定计划"}
        yield {"type": "plan", "data": {"document_type": "通知", "task_type": "公文生成"}}
        yield {"type": "content", "data": "正文"}
        yield {
            "type": "done",
            "document": "正文",
            "plan": {"document_type": "通知", "task_type": "公文生成"},
            "run_records": [{"step": "write", "llm_usage": {"agent": "Writer", "model": "fake"}}],
            "source_filenames": ["制度.docx", "制度.docx"],
            "source_details": [{"filename": "制度.docx"}],
            "audit_summary": {"passed": True},
        }


class ReasoningRunner:
    def run_stream(self, message, on_think=None, session_id=None):
        yield {"type": "start"}
        yield {"type": "reasoning_chunk", "data": "内部推理不应外显"}
        yield {"type": "reflection", "data": {"weaknesses": ["问题"], "reasoning_content": "内部完整推理"}}
        yield {
            "type": "done",
            "document": "正文",
            "plan": {"document_type": "通知", "task_type": "公文生成"},
            "run_records": [],
            "source_filenames": [],
            "source_details": [],
            "audit_summary": {"passed": True},
        }


class FailingRunner:
    def run_stream(self, message, on_think=None, session_id=None):
        raise RuntimeError("orchestrator failed")
        yield {}


def parse_sse(chunks):
    events = []
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    return events


def build_service(memory, runner, token_calls, run_usage_calls):
    return DocumentDraftStreamService(DocumentDraftDependencies(
        memory=memory,
        orchestrator_factory=lambda session_id, profile=None, user_info=None: runner,
        resolve_export_template=lambda text, plan, request: "default",
        record_agent_run_token_usage=lambda *args, **kwargs: run_usage_calls.append((args, kwargs)),
        record_token_usage=lambda **kwargs: token_calls.append(kwargs),
    ))


def test_document_draft_stream_success_contract():
    memory = FakeMemory()
    runner = FakeRunner()
    token_calls = []
    run_usage_calls = []
    service = build_service(memory, runner, token_calls, run_usage_calls)

    events = parse_sse(service.stream(
        "帮我写一份通知",
        "session_1",
        "user_1",
        user_info=SimpleNamespace(username="tester"),
        display_message="帮我写一份通知",
        route=SimpleNamespace(to_dict=lambda: {"intent": "doc_drafting", "actions": []}),
    ))

    assert [event["type"] for event in events[:3]] == ["start", "session", "route"]
    done = events[-1]
    assert done["type"] == "done"
    assert done["intent"] == "doc_drafting"
    assert done["document"] == "正文"
    assert done["think_log"] == [{"agent": "Planner", "emoji": "🧭", "message": "已制定计划"}]
    assert done["export_template"] == "default"
    assert done["source_filenames"] == ["制度.docx"]
    assert run_usage_calls[0][1]["mode"] == "agent"
    assert token_calls == []
    assert memory.profile_updates == [("user_1", {"common_doc_types": ["通知"]})]
    assert "用户偏好：仿宋 三号" in runner.requests[0][0]


def test_document_draft_stream_hides_internal_reasoning_events():
    memory = FakeMemory()
    token_calls = []
    run_usage_calls = []
    service = build_service(memory, ReasoningRunner(), token_calls, run_usage_calls)

    events = parse_sse(service.stream(
        "帮我写一份通知",
        "session_1",
        "user_1",
        display_message="帮我写一份通知",
    ))

    assert "reasoning_chunk" not in [event["type"] for event in events]
    reflection = next(event for event in events if event["type"] == "reflection")
    assert reflection["data"] == {"weaknesses": ["问题"]}
    assert events[-1]["type"] == "done"


def test_document_runner_sanitizes_unsupported_meeting_specifics():
    text = "定于2026年6月25日（星期四）下午3:00在示例单位A栋会议室召开会议。"
    sanitized = DocumentStreamRunner._sanitize_unsupported_specifics(
        text,
        "6月25日下午3点在会议室召开部门例会",
    )

    assert "A栋" not in sanitized
    assert "星期四" not in sanitized
    assert "在会议室召开会议" in sanitized


def test_document_runner_keeps_user_supplied_meeting_specifics():
    text = "定于2026年6月25日（星期四）下午3:00在示例单位A栋会议室召开会议。"
    sanitized = DocumentStreamRunner._sanitize_unsupported_specifics(
        text,
        "6月25日星期四下午3点在示例单位A栋会议室召开部门例会",
    )

    assert "A栋" in sanitized
    assert "星期四" in sanitized


def test_document_draft_stream_records_failure():
    memory = FakeMemory()
    token_calls = []
    run_usage_calls = []
    service = build_service(memory, FailingRunner(), token_calls, run_usage_calls)

    events = parse_sse(service.stream(
        "帮我写一份通知",
        "session_1",
        "user_1",
        display_message="帮我写一份通知",
    ))

    assert events[-1]["type"] == "error"
    assert token_calls[-1]["status"] == "failed"
    assert token_calls[-1]["mode"] == "agent"
    assert run_usage_calls == []
