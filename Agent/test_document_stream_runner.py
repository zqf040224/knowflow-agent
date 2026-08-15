from types import SimpleNamespace

import pytest

from agents.document_stream_runner import DocumentStreamRunner


class FakeContext:
    def __init__(self):
        self.user_request = "写通知"
        self.plan = {"task_type": "公文生成", "document_type": "通知", "need_web_search": False}
        self.context_analysis = {"key_points": ["开会"]}
        self.search_context = ""
        self.knowledge_context = ""
        self.knowledge_sources = [{"filename": "source.docx"}]
        self.evidence_items = []
        self.compact_evidence = []
        self.revision_history = []
        self.run_records = []
        self.audit_summary = {}
        self.last_document = ""
        self.last_plan = {}
        self.user_constraints = []
        self.unresolved_questions = []


class FakeResult:
    def __init__(self, *, metadata=None, success=True, error=""):
        self.metadata = metadata or {}
        self.success = success
        self.error_info = {"error": error} if error else {}


class FakeWriter:
    def __init__(self, chunks=None):
        self.chunks = ["正文"] if chunks is None else chunks

    def process_stream(self, payload):
        self.payload = payload
        for chunk in self.chunks:
            yield chunk


class FakeOrchestrator:
    MAX_TOTAL_ROUNDS = 3

    def __init__(self, *, always_revise=False, fail_review=False):
        self.think_log = []
        self.always_revise = always_revise
        self.fail_review = fail_review
        self.writer = FakeWriter()
        self.reflection = None
        self.memory = None
        self.session_id = None

    def _on_think(self, agent_name, emoji, message):
        self.think_log.append({"agent": agent_name, "emoji": emoji, "message": message})

    def _step_context_plan(self, request_with_context, previous_context, cb):
        return FakeContext()

    def _step_knowledge(self, ctx, cb):
        return ctx

    def _step_search(self, ctx, cb):
        return ctx

    def _step_review(self, ctx, document_content, cb):
        if self.fail_review:
            return FakeResult(success=False, error="review unavailable")
        return FakeResult(metadata={
            "needs_revision": self.always_revise,
            "spreadsheet_audit": {"ok": True},
            "confidence": 0.9,
        })

    def _record_step(self, ctx, step, start_time, **extra):
        ctx.run_records.append({"step": step, **extra})

    def _build_evidence_items(self, ctx):
        return [{"filename": "source.docx"}]

    def _compact_evidence_items(self, evidence_items):
        return evidence_items

    def _writer_search_context(self, ctx, revision_round):
        return ""

    def _writer_knowledge_context(self, ctx, revision_round):
        return ""

    def _merged_key_points(self, ctx):
        return ["开会"]

    def _should_reflect(self, ctx, review_meta, revision_round):
        return False

    def _combined_revision_focus(self, review_meta, reflection_meta=None):
        return []

    def _source_filenames(self, ctx):
        return ["source.docx"]

    def _source_details(self, ctx):
        return [{"filename": "source.docx"}]

    def _context_snapshot(self, ctx):
        return {"document_type": ctx.plan.get("document_type", "")}


class RecordingMemory:
    def __init__(self):
        self.messages = []
        self.summaries = []
        self.context = {}

    def add_message(self, session_id, role, content, metadata=None):
        self.messages.append((session_id, role, content, metadata or {}))

    def set_context(self, session_id, key, value):
        self.context[(session_id, key)] = value

    def update_rolling_summary(
        self, session_id, message, response, plan, sources, *, effect_key=None
    ):
        self.summaries.append(
            (session_id, message, response, plan, sources, effect_key)
        )


def test_document_stream_runner_emits_public_event_contract():
    orchestrator = FakeOrchestrator()

    events = list(DocumentStreamRunner(orchestrator).run(
        SimpleNamespace(request_with_context="写通知", previous_context=""),
        user_request="写通知",
    ))

    event_types = [event["type"] for event in events]
    done = events[-1]

    assert event_types[:4] == ["context_start", "context_end", "plan_start", "plan"]
    assert "write_start" in event_types
    assert "content" in event_types
    assert event_types[-1] == "done"
    assert "content_reset" not in event_types
    assert done["document"] == "正文"
    assert done["source_filenames"] == ["source.docx"]
    assert done["audit_summary"] == {"ok": True}
    assert done["quality_status"] == "passed"
    assert [record["step"] for record in done["run_records"]] == [
        "context_plan",
        "retrieval",
        "write",
        "review",
    ]


def test_document_stream_runner_marks_max_revisions_without_pass_message():
    events = list(DocumentStreamRunner(FakeOrchestrator(always_revise=True)).run(
        SimpleNamespace(request_with_context="写通知", previous_context="", run_id="run-max"),
        user_request="写通知",
    ))

    done = events[-1]
    assert done["quality_status"] == "max_revisions"
    assert done["revision_rounds"] == 2
    assert done["run_id"] == "run-max"
    assert any(event.get("type") == "think" and "已达最大修订轮次" in event["message"] for event in events)
    assert not any(
        event.get("type") == "think"
        and event.get("agent") == "Reviewer"
        and "审核通过" in event.get("message", "")
        for event in events
    )


def test_document_stream_runner_fails_closed_when_reviewer_unavailable():
    with pytest.raises(RuntimeError, match="review unavailable"):
        list(DocumentStreamRunner(FakeOrchestrator(fail_review=True)).run(
            SimpleNamespace(request_with_context="写通知", previous_context=""),
            user_request="写通知",
        ))


def test_document_stream_runner_fails_closed_before_review_when_writer_is_empty():
    orchestrator = FakeOrchestrator()
    orchestrator.writer = FakeWriter([])
    orchestrator._step_review = lambda *_args, **_kwargs: pytest.fail(
        "Reviewer must not run for an empty draft"
    )

    with pytest.raises(RuntimeError, match="Writer returned empty document content"):
        list(DocumentStreamRunner(orchestrator).run(
            SimpleNamespace(request_with_context="写通知", previous_context=""),
            user_request="写通知",
        ))


def test_document_stream_runner_forwards_parent_run_effect_keys_to_memory():
    orchestrator = FakeOrchestrator()
    orchestrator.memory = RecordingMemory()
    orchestrator.session_id = "session-1"
    orchestrator._current_effect_run_id = "parent-run"

    list(DocumentStreamRunner(orchestrator).run(
        SimpleNamespace(
            request_with_context="写通知",
            previous_context="",
            run_id="parent-run",
        ),
        user_request="写通知",
    ))

    assert orchestrator.memory.messages[0][3]["effect_key"] == (
        "parent-run:tool_draft_document:message_assistant"
    )
    assert orchestrator.memory.summaries[0][-1] == (
        "parent-run:tool_draft_document:rolling_summary"
    )


def test_document_stream_runner_scopes_effect_keys_and_uses_safe_summary_text():
    orchestrator = FakeOrchestrator()
    orchestrator.memory = RecordingMemory()
    orchestrator.session_id = "session-1"
    orchestrator._current_effect_run_id = "parent-run"

    list(DocumentStreamRunner(orchestrator).run(
        SimpleNamespace(
            request_with_context="写通知\n\n[文件内容]SECRET_ATTACHMENT_BODY",
            previous_context="",
            run_id="parent-run",
            persisted_user_message="请根据附件起草通知",
            effect_scope="step:2",
        ),
        user_request="写通知\n\n[文件内容]SECRET_ATTACHMENT_BODY",
    ))

    assert orchestrator.memory.messages[0][3]["effect_key"] == (
        "parent-run:tool_draft_document:step:2:message_assistant"
    )
    assert orchestrator.memory.summaries[0][1] == "请根据附件起草通知"
    assert "SECRET_ATTACHMENT_BODY" not in orchestrator.memory.summaries[0][1]
    assert orchestrator.memory.summaries[0][-1] == (
        "parent-run:tool_draft_document:step:2:rolling_summary"
    )


def test_document_stream_runner_does_not_key_generated_rollback_run():
    orchestrator = FakeOrchestrator()
    orchestrator.memory = RecordingMemory()
    orchestrator.session_id = "session-1"
    orchestrator._current_effect_run_id = ""

    list(DocumentStreamRunner(orchestrator).run(
        SimpleNamespace(
            request_with_context="写通知",
            previous_context="",
            run_id="internally-generated-run",
        ),
        user_request="写通知",
    ))

    assert "effect_key" not in orchestrator.memory.messages[0][3]
    assert orchestrator.memory.summaries[0][-1] is None
