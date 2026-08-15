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

    def run_stream(self, message, on_think=None, session_id=None, run_id=""):
        self.requests.append((message, session_id, run_id))
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
    def run_stream(self, message, on_think=None, session_id=None, run_id=""):
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
    def run_stream(self, message, on_think=None, session_id=None, run_id=""):
        raise RuntimeError("orchestrator failed")
        yield {}


class TerminalRunner:
    def __init__(self, *, fail_after_answer=False):
        self.fail_after_answer = fail_after_answer
        self.call = None

    def run_stream(
        self,
        message,
        on_think=None,
        session_id=None,
        run_id="",
        persisted_user_message=None,
        effect_scope="",
    ):
        self.call = {
            "message": message,
            "session_id": session_id,
            "run_id": run_id,
            "persisted_user_message": persisted_user_message,
            "effect_scope": effect_scope,
        }
        yield {"type": "plan", "data": {"document_type": "通知"}}
        yield {"type": "answer_start", "message": "开始输出正文"}
        yield {"type": "content", "data": "正文"}
        yield {"type": "answer_done", "answer": "正文"}
        if self.fail_after_answer:
            raise RuntimeError("memory commit failed")
        yield {
            "type": "done",
            "document": "正文",
            "plan": {"document_type": "通知", "task_type": "公文生成"},
            "run_records": [{"step": "write", "llm_usage": {"agent": "Writer"}}],
            "source_filenames": [],
            "source_details": [],
            "audit_summary": {},
        }


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
        user_metadata={"run_id": "run-draft"},
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
    assert run_usage_calls[0][1]["run_id"] == "run-draft"
    assert token_calls == []
    assert memory.profile_updates == [("user_1", {"common_doc_types": ["通知"]})]
    assert "用户偏好：仿宋 三号" in runner.requests[0][0]
    assert runner.requests[0][2] == "run-draft"


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


def test_document_draft_stream_passes_hydrated_prompt_but_only_safe_display_text_for_persistence():
    memory = FakeMemory()
    runner = TerminalRunner()
    service = build_service(memory, runner, [], [])

    events = parse_sse(service.stream(
        "请起草通知\n\n[文件内容]SECRET_ATTACHMENT_BODY",
        "session_1",
        "user_1",
        display_message="请根据附件起草通知",
        user_metadata={"run_id": "run-safe", "effect_scope": "step:2"},
    ))

    assert events[-1]["type"] == "done"
    assert "SECRET_ATTACHMENT_BODY" in runner.call["message"]
    assert runner.call["persisted_user_message"] == "请根据附件起草通知"
    assert runner.call["effect_scope"] == "step:2"


def test_document_draft_stream_does_not_fall_back_to_hydrated_prompt_for_empty_display_text():
    runner = TerminalRunner()
    service = build_service(FakeMemory(), runner, [], [])

    events = parse_sse(service.stream(
        "[文件内容]SECRET_ATTACHMENT_BODY",
        "session_1",
        "user_1",
        display_message="",
        user_metadata={"run_id": "run-file-only"},
    ))

    assert events[-1]["type"] == "done"
    assert "SECRET_ATTACHMENT_BODY" in runner.call["message"]
    assert runner.call["persisted_user_message"] == ""


def test_document_draft_commit_gate_hides_answer_when_usage_commit_fails():
    memory = FakeMemory()
    runner = TerminalRunner()
    failures = []
    service = DocumentDraftStreamService(DocumentDraftDependencies(
        memory=memory,
        orchestrator_factory=lambda *_args, **_kwargs: runner,
        resolve_export_template=lambda *_args: "default",
        record_agent_run_token_usage=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(RuntimeError("billing unavailable")),
        record_token_usage=lambda **kwargs: failures.append(kwargs),
    ))

    events = parse_sse(service.stream(
        "帮我写一份通知",
        "session_1",
        "user_1",
        display_message="帮我写一份通知",
        user_metadata={"run_id": "run-billing", "effect_scope": "step:2"},
    ))

    event_types = [event["type"] for event in events]
    assert events[-1]["type"] == "error"
    assert not {"answer_start", "content", "answer_done", "run_done", "done"}.intersection(event_types)
    assert failures[-1]["effect_key"] == (
        "run-billing:tool_draft_document:step:2:token_usage_pipeline_failure"
    )


def test_document_draft_commit_gate_hides_answer_when_profile_commit_fails():
    class ProfileFailingMemory(FakeMemory):
        def update_user_profile(self, user_id, data):
            raise RuntimeError("profile unavailable")

    runner = TerminalRunner()
    service = build_service(ProfileFailingMemory(), runner, [], [])

    events = parse_sse(service.stream(
        "帮我写一份通知",
        "session_1",
        "user_1",
        display_message="帮我写一份通知",
        user_metadata={"run_id": "run-profile"},
    ))

    event_types = [event["type"] for event in events]
    assert events[-1]["type"] == "error"
    assert not {"answer_start", "content", "answer_done", "run_done", "done"}.intersection(event_types)


def test_document_draft_commit_gate_hides_answer_when_runner_commit_fails():
    runner = TerminalRunner(fail_after_answer=True)
    service = build_service(FakeMemory(), runner, [], [])

    events = parse_sse(service.stream(
        "帮我写一份通知",
        "session_1",
        "user_1",
        display_message="帮我写一份通知",
        user_metadata={"run_id": "run-memory"},
    ))

    event_types = [event["type"] for event in events]
    assert events[-1]["type"] == "error"
    assert not {"answer_start", "content", "answer_done", "run_done", "done"}.intersection(event_types)
