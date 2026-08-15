import json
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agents.document_graph_runner import (
    DocumentGraphExecutionError,
    DocumentGraphRunner,
    LANGGRAPH_AVAILABLE,
)
from agents.orchestrator import AgentOrchestrator
from agents.reviewer_agent import ReviewerAgent


class FakeContext:
    def __init__(self):
        self.user_request = "写通知"
        self.context_analysis = {}
        self.plan = {"task_type": "公文生成", "document_type": "通知", "need_web_search": False}
        self.search_context = ""
        self.knowledge_context = ""
        self.knowledge_sources = []
        self.search_sources = []
        self.evidence_items = []
        self.compact_evidence = []
        self.revision_history = []
        self.run_records = []
        self.audit_summary = {}
        self.last_document = ""
        self.last_plan = {}
        self.user_constraints = []
        self.unresolved_questions = []
        self.user_profile = None
        self.memory_context = ""


class FakeResult:
    def __init__(self, *, content="", metadata=None, success=True, error=""):
        self.content = content
        self.metadata = metadata or {}
        self.success = success
        self.error_info = {"error": error} if error else {}


class FakeOrchestrator:
    MAX_TOTAL_ROUNDS = 3

    def __init__(self, *, reflect=False, always_revise=False, fail_write=False, fail_review=False):
        self.reflect = reflect
        self.always_revise = always_revise
        self.fail_write = fail_write
        self.fail_review = fail_review
        self.calls = []
        self.write_count = 0
        self.review_count = 0
        self.think_log = []

    def _step_context_plan(self, request_with_context, previous_context, think_handler):
        self.calls.append(("context_plan", request_with_context, previous_context))
        return FakeContext()

    def _step_search(self, ctx, think_handler):
        self.calls.append(("search",))
        return ctx

    def _step_knowledge(self, ctx, think_handler):
        self.calls.append(("knowledge",))
        ctx.knowledge_sources = [{"filename": "source.docx"}]
        return ctx

    def _step_write(self, ctx, think_handler):
        self.write_count += 1
        self.calls.append(("write", self.write_count, ctx.last_document))
        if self.fail_write:
            return FakeResult(success=False, error="writer unavailable")
        return FakeResult(content=f"doc-v{self.write_count}")

    def _step_write_stream(self, ctx, revision_round, think_handler):
        return self._step_write(ctx, think_handler).content

    def _step_review(self, ctx, document_content, think_handler):
        self.review_count += 1
        self.calls.append(("review", self.review_count, document_content))
        if self.fail_review:
            return FakeResult(success=False, error="review unavailable")
        needs_revision = self.always_revise or (self.review_count == 1 and not self.reflect)
        return FakeResult(metadata={
            "needs_revision": needs_revision,
            "revision_focus": ["结构"],
            "suggestions": ["重写标题"],
            "format_check": {"issues": []},
            "content_check": {"issues": []},
            "logic_check": {"issues": []},
            "language_check": {"issues": []},
            "fact_check": {"issues": []},
            "spreadsheet_audit": {"ok": True},
            "confidence": 0.7 if needs_revision else 0.95,
        })

    def _step_reflection(self, ctx, document_content, think_handler):
        self.calls.append(("reflection", document_content))
        return FakeResult(metadata={
            "needs_revision": False,
            "revision_suggestions": ["无"],
            "weaknesses": [],
            "counter_arguments": [],
            "logic_score": 0.9,
        })

    def _record_step(self, ctx, step, start_time, **extra):
        ctx.run_records.append({"step": step, **extra})

    def _build_evidence_items(self, ctx):
        return [{"filename": "source.docx"}]

    def _compact_evidence_items(self, evidence_items):
        return [{"filename": item["filename"]} for item in evidence_items]

    def _combined_revision_focus(self, review_meta, reflection_meta=None):
        return review_meta.get("revision_focus") or []

    def _should_reflect(self, ctx, review_meta, revision_round):
        return self.reflect and revision_round == 0

    @staticmethod
    def _sanitize_document_output(text, user_request):
        return text

    def _build_document_run_result(self, ctx, document_content, user_request, **extra):
        return {
            "document": document_content,
            "plan": ctx.plan,
            "think_log": self.think_log,
            "run_records": ctx.run_records,
            "source_filenames": ["source.docx"],
            "source_details": [{"filename": "source.docx"}],
            "audit_summary": ctx.audit_summary,
            **extra,
        }


def think_collector():
    events = []

    def on_think(agent_name, emoji, message):
        events.append((agent_name, emoji, message))

    return events, on_think


def prepared_run():
    return SimpleNamespace(
        user_request="写通知",
        request_with_context="写通知",
        previous_context="",
    )


def test_document_prepare_snapshot_keeps_previous_context_stable_on_retry():
    class Memory:
        def __init__(self):
            self.context = {("session-1", "last_request"): "上一轮需求"}
            self.messages = []

        def add_message(self, session_id, role, content, metadata=None):
            self.messages.append((session_id, role, content, metadata or {}))

        def get_context(self, session_id, key, default=None):
            return self.context.get((session_id, key), default)

        def set_context(self, session_id, key, value):
            self.context[(session_id, key)] = value

    memory = Memory()
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.memory = memory
    orchestrator.session_id = "session-1"
    orchestrator.think_log = []
    orchestrator._current_effect_run_id = ""
    orchestrator._current_agent_memory_context = ""
    orchestrator._recall_agent_context = lambda _request: ""

    first = orchestrator._prepare_document_run(
        "当前需求",
        session_id="session-1",
        run_id="stable-run-id",
    )
    replay = orchestrator._prepare_document_run(
        "当前需求",
        session_id="session-1",
        run_id="stable-run-id",
    )

    assert first.previous_context == "上一轮需求"
    assert replay.previous_context == "上一轮需求"
    assert replay.request_with_context == first.request_with_context
    assert memory.context[("session-1", "last_request")] == "当前需求"
    with pytest.raises(RuntimeError, match="reused with a different request"):
        orchestrator._prepare_document_run(
            "不同需求",
            session_id="session-1",
            run_id="stable-run-id",
        )


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_runner_revises_until_review_passes():
    orchestrator = FakeOrchestrator()
    events, on_think = think_collector()

    result = DocumentGraphRunner(orchestrator).run(
        prepared_run(),
        think_handler=on_think,
        thread_id="session-revision",
    )

    assert result.document_content == "doc-v2"
    assert [item["step"] for item in result.ctx.run_records] == [
        "context_plan",
        "retrieval",
        "write",
        "review",
        "write",
        "review",
        "orchestrator_runtime",
    ]
    assert result.ctx.last_document == "doc-v1"
    assert result.ctx.audit_summary == {"ok": True}
    assert result.ctx.run_records[-1] == {
        "step": "orchestrator_runtime",
        "runtime": "langgraph",
        "stream": False,
        "quality_status": "passed",
    }
    assert any("第1轮已汇总审核意见" in message for _agent, _emoji, message in events)
    assert any(agent == "Reviewer" for agent, _emoji, _message in events)


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_runner_can_route_through_reflection_once():
    orchestrator = FakeOrchestrator(reflect=True)
    _events, on_think = think_collector()

    result = DocumentGraphRunner(orchestrator).run(
        prepared_run(),
        think_handler=on_think,
        thread_id="session-reflection",
    )

    assert result.document_content == "doc-v1"
    assert ("reflection", "doc-v1") in orchestrator.calls
    assert [call[0] for call in orchestrator.calls].count("reflection") == 1
    assert [item["step"] for item in result.ctx.run_records] == [
        "context_plan",
        "retrieval",
        "write",
        "review",
        "reflection",
        "orchestrator_runtime",
    ]
    assert result.ctx.revision_history[-1]["source"] == "reflection"


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_runner_marks_max_revisions_without_pass_message():
    orchestrator = FakeOrchestrator(always_revise=True)
    events, on_think = think_collector()

    result = DocumentGraphRunner(orchestrator).run(
        prepared_run(),
        think_handler=on_think,
        thread_id="session-max",
    )

    assert result.document_content == "doc-v3"
    assert result.quality_status == "max_revisions"
    assert result.revisions_applied == 2
    assert any("已达最大修订轮次" in message for _a, _e, message in events)
    assert not any(a == "Reviewer" and "审核通过" in message for a, _e, message in events)


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
@pytest.mark.parametrize("failure", ["write", "review"])
def test_document_graph_runner_fails_closed(failure):
    orchestrator = FakeOrchestrator(
        fail_write=failure == "write",
        fail_review=failure == "review",
    )
    _events, on_think = think_collector()

    with pytest.raises(DocumentGraphExecutionError, match=failure) as exc_info:
        DocumentGraphRunner(orchestrator).run(
            prepared_run(),
            think_handler=on_think,
            thread_id=f"session-{failure}",
        )

    assert exc_info.value.state["run_status"] == "failed"
    if failure == "review":
        assert exc_info.value.document_content == "doc-v1"


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_runner_reuses_compiled_graph_and_streams_custom_contract():
    first = DocumentGraphRunner(FakeOrchestrator())
    second_orchestrator = FakeOrchestrator()
    second = DocumentGraphRunner(second_orchestrator)
    events, on_think = think_collector()

    output = list(second.stream(
        prepared_run(),
        think_handler=on_think,
        thread_id="session-stream",
        user_request="写通知",
    ))

    assert first.graph is second.graph
    event_types = [event["type"] for event in output]
    assert event_types[:4] == ["context_start", "context_end", "plan_start", "plan"]
    assert "write_start" in event_types
    assert "content" in event_types
    assert event_types[-1] == "done"
    assert output[-1]["document"] == "doc-v2"
    assert output[-1]["runtime"] == "langgraph"


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_uses_parent_checkpointer_with_document_namespace():
    saver = InMemorySaver()
    runner = DocumentGraphRunner(FakeOrchestrator(), checkpointer=saver)
    _events, on_think = think_collector()

    result = runner.run(
        prepared_run(),
        think_handler=on_think,
        thread_id="parent-run-1",
    )

    assert runner.graph.checkpointer.delegate is saver
    assert result.run_id == "parent-run-1"
    snapshot = runner.graph.get_state({
        "configurable": {"thread_id": "parent-run-1"}
    })
    assert snapshot.values["quality_status"] == "passed"
    checkpoint = saver.get_tuple({
        "configurable": {
            "thread_id": "parent-run-1",
            "checkpoint_ns": "document",
        }
    })
    assert checkpoint is not None


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_replays_terminal_checkpoint_without_rerunning_nodes():
    saver = InMemorySaver()
    orchestrator = FakeOrchestrator()
    runner = DocumentGraphRunner(orchestrator, checkpointer=saver)
    _events, on_think = think_collector()

    first = runner.run(
        prepared_run(),
        think_handler=on_think,
        thread_id="parent-terminal-replay",
    )
    calls_after_first_run = list(orchestrator.calls)
    second = runner.run(
        prepared_run(),
        think_handler=on_think,
        thread_id="parent-terminal-replay",
    )

    assert first.document_content == second.document_content == "doc-v2"
    assert orchestrator.calls == calls_after_first_run


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_resumes_pending_checkpoint_without_restarting_writer():
    saver = InMemorySaver()
    orchestrator = FakeOrchestrator()
    runner = DocumentGraphRunner(orchestrator, checkpointer=saver)
    _events, on_think = think_collector()
    run_id = "parent-mid-node-resume"

    interrupted_state = runner.graph.invoke(
        runner._initial_state(prepared_run(), run_id),
        config=runner._config(run_id),
        context=runner._runtime(on_think, streaming=False),
        durability=runner.durability,
        interrupt_after=["write"],
    )
    assert interrupted_state["document_content"] == "doc-v1"
    assert runner.graph.get_state(runner._config(run_id)).next == ("review",)
    calls_before_resume = [call[0] for call in orchestrator.calls]

    result = runner.run(
        prepared_run(),
        think_handler=on_think,
        thread_id=run_id,
    )

    assert result.document_content == "doc-v2"
    assert [call[0] for call in orchestrator.calls][:len(calls_before_resume)] == calls_before_resume
    assert [call[0] for call in orchestrator.calls].count("context_plan") == 1
    assert [call[0] for call in orchestrator.calls].count("knowledge") == 1
    assert [call[0] for call in orchestrator.calls].count("write") == 2


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_standalone_graph_keeps_hydrated_attachment_runtime_only_and_resumes_empty_display():
    secret = "SECRET_ATTACHMENT_BODY_THAT_MUST_NOT_ENTER_CHECKPOINTS"
    previous_secret = "SECRET_PREVIOUS_CONTEXT_MUST_STAY_RUNTIME_ONLY"
    snapshot_memory_secret = "SECRET_RECALLED_MEMORY_MUST_STAY_RUNTIME_ONLY"
    snapshot_document_secret = "SECRET_PREVIOUS_DOCUMENT_MUST_STAY_RUNTIME_ONLY"
    snapshot_profile_secret = "SECRET_PROFILE_VALUE_MUST_STAY_RUNTIME_ONLY"
    secrets = (
        secret,
        previous_secret,
        snapshot_memory_secret,
        snapshot_document_secret,
        snapshot_profile_secret,
    )
    hydrated_request = (
        f"[文件内容]\n{secret}\n[/文件内容]\n\n[用户提问]\n"
    )
    prepared = SimpleNamespace(
        user_request=hydrated_request,
        request_with_context=hydrated_request,
        previous_context=previous_secret,
        persisted_user_message="",
        display_message="",
        session_id="session-file-only",
        run_id="standalone-file-only",
    )

    class EchoingOrchestrator(FakeOrchestrator):
        def __init__(self):
            super().__init__()
            self._current_agent_memory_context = snapshot_memory_secret
            self.user_profile = {
                "department": snapshot_profile_secret,
                "short_label": "通知",
            }
            self._document_runtime_snapshot = {
                "memory_context": snapshot_memory_secret,
                "last_document": snapshot_document_secret,
                "last_plan": {"private_note": snapshot_profile_secret},
                "user_profile": dict(self.user_profile),
                "conversation_history": [{
                    "role": "user",
                    "content": previous_secret,
                }],
            }

        def _step_context_plan(self, request_with_context, previous_context, think_handler):
            self.calls.append(("context_plan", request_with_context, previous_context))
            think_handler(
                "Planner",
                "🧠",
                f"分析 {secret} {previous_context} {snapshot_memory_secret}",
            )
            ctx = FakeContext()
            ctx.user_request = request_with_context
            ctx.context_analysis = {
                "echo": request_with_context,
                "previous_echo": previous_context,
                "snapshot_echo": {
                    "memory": snapshot_memory_secret,
                    "document": snapshot_document_secret,
                    "profile": snapshot_profile_secret,
                },
            }
            ctx.plan = {
                "task_type": "公文生成",
                "document_type": "通知",
                "need_web_search": False,
                "echo": request_with_context,
                "previous_echo": previous_context,
                "snapshot_echo": snapshot_document_secret,
            }
            ctx.memory_context = snapshot_memory_secret
            ctx.last_document = snapshot_document_secret
            ctx.last_plan = {"private_note": snapshot_profile_secret}
            ctx.user_profile = dict(self.user_profile)
            return ctx

        def _step_write(self, ctx, think_handler):
            self.calls.append(("write_request", ctx.user_request))
            return super()._step_write(ctx, think_handler)

    saver = InMemorySaver()
    first_orchestrator = EchoingOrchestrator()
    first_runner = DocumentGraphRunner(first_orchestrator, checkpointer=saver)
    _events, on_think = think_collector()
    run_id = prepared.run_id

    initial = first_runner._initial_state(prepared, run_id)
    assert initial["display_message"] == ""
    assert initial["input_user_request"] == ""
    assert initial["request_with_context"] == ""
    assert all(
        item not in json.dumps(initial, ensure_ascii=False)
        for item in secrets
    )

    runtime_context = first_runner._runtime(
        on_think,
        streaming=False,
        prepared_run=prepared,
    )
    assert "通知" not in runtime_context.redaction_fragments
    sanitized_probe = runtime_context.sanitize_document_update(
        {},
        {
            "plan": {"echo": snapshot_document_secret},
            "document_content": snapshot_document_secret,
        },
    )
    assert snapshot_document_secret not in sanitized_probe["plan"]["echo"]
    assert sanitized_probe["document_content"] == snapshot_document_secret
    first_runner.graph.invoke(
        initial,
        config=first_runner._config(run_id),
        context=runtime_context,
        durability=first_runner.durability,
        interrupt_after=["context_plan"],
    )
    pending = first_runner.graph.get_state(first_runner._config(run_id))
    assert pending.next == ("retrieval",)
    assert "SECRET" not in json.dumps(pending.values, ensure_ascii=False)
    assert "SECRET" not in repr((saver.storage, saver.writes))
    assert secret in first_orchestrator.calls[0][1]
    assert previous_secret in first_orchestrator.calls[0][2]

    resumed_orchestrator = EchoingOrchestrator()
    resumed_runner = DocumentGraphRunner(resumed_orchestrator, checkpointer=saver)
    result = resumed_runner.run(
        prepared,
        think_handler=on_think,
        thread_id=run_id,
    )

    assert result.document_content == "doc-v2"
    assert not any(call[0] == "context_plan" for call in resumed_orchestrator.calls)
    write_request = next(
        call[1]
        for call in resumed_orchestrator.calls
        if call[0] == "write_request"
    )
    assert secret in write_request
    terminal = resumed_runner.graph.get_state(resumed_runner._config(run_id))
    assert "SECRET" not in json.dumps(terminal.values, ensure_ascii=False)
    assert "SECRET" not in repr((saver.storage, saver.writes))


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_retries_failed_reviewer_with_preserved_draft():
    saver = InMemorySaver()
    orchestrator = FakeOrchestrator(fail_review=True)
    runner = DocumentGraphRunner(orchestrator, checkpointer=saver)
    _events, on_think = think_collector()
    run_id = "parent-review-retry"

    with pytest.raises(DocumentGraphExecutionError, match="review") as first_error:
        runner.run(prepared_run(), think_handler=on_think, thread_id=run_id)
    assert first_error.value.document_content == "doc-v1"

    orchestrator.fail_review = False
    result = runner.run(prepared_run(), think_handler=on_think, thread_id=run_id)

    assert result.document_content == "doc-v1"
    assert [call[0] for call in orchestrator.calls].count("context_plan") == 1
    assert [call[0] for call in orchestrator.calls].count("knowledge") == 1
    assert [call[0] for call in orchestrator.calls].count("write") == 1
    assert [call[0] for call in orchestrator.calls].count("review") == 2


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_stream_and_invoke_share_writer_business_logic():
    class DivergentLegacyStreamOrchestrator(FakeOrchestrator):
        def __init__(self):
            super().__init__()
            self.legacy_stream_write_calls = 0

        def _step_write_stream(self, ctx, revision_round, think_handler):
            self.legacy_stream_write_calls += 1
            return "different-stream-only-document"

    invoke_orchestrator = DivergentLegacyStreamOrchestrator()
    stream_orchestrator = DivergentLegacyStreamOrchestrator()
    _events, on_think = think_collector()
    invoke_result = DocumentGraphRunner(
        invoke_orchestrator,
        checkpointer=InMemorySaver(),
    ).run(
        prepared_run(),
        think_handler=on_think,
        thread_id="document-invoke-parity",
    )
    stream_events = list(DocumentGraphRunner(
        stream_orchestrator,
        checkpointer=InMemorySaver(),
    ).stream(
        prepared_run(),
        think_handler=on_think,
        thread_id="document-stream-parity",
        user_request="写通知",
    ))

    assert stream_events[-1]["document"] == invoke_result.document_content
    assert invoke_orchestrator.legacy_stream_write_calls == 0
    assert stream_orchestrator.legacy_stream_write_calls == 0


def test_orchestrator_turns_swallowed_reviewer_exception_into_failure():
    class SwallowingReviewer:
        name = "Reviewer"

        def call_llm(self, *_args, **_kwargs):
            raise RuntimeError("review model offline")

        @staticmethod
        def _parse_json_response(value):
            return value

        def process(self, _payload, on_think=None):
            try:
                self.call_llm("prompt")
            except RuntimeError:
                return FakeResult(metadata={
                    "needs_revision": False,
                    "confidence": 0.7,
                })

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.reviewer = SwallowingReviewer()

    result = orchestrator._step_review(FakeContext(), "正文", lambda *_args: None)

    assert result.success is False
    assert result.metadata["review_unavailable"] is True
    assert "offline" in result.error_info["error"]


def test_orchestrator_rejects_real_reviewer_invalid_model_json():
    reviewer = ReviewerAgent.__new__(ReviewerAgent)
    reviewer.name = "Reviewer"
    reviewer.call_llm = lambda *_args, **_kwargs: "not valid json"
    reviewer._audit_spreadsheet_facts = lambda *_args: {
        "passed": True,
        "issues": [],
        "verified_claims": [],
        "unverified_claims": [],
        "spreadsheet_evidence_count": 0,
    }
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.reviewer = reviewer
    document = (
        "关于开展专题学习的通知\n\n"
        + "为进一步提升工作质量，现组织开展专题学习，请各部门结合实际认真落实。" * 8
        + "\n\n示例单位\n2026年8月9日"
    )

    result = orchestrator._step_review(FakeContext(), document, lambda *_args: None)

    assert result.success is False
    assert result.metadata["review_unavailable"] is True
    assert "Expecting value" in result.error_info["error"]


def test_orchestrator_rejects_incomplete_or_contradictory_review_metadata():
    incomplete = FakeResult(metadata={
        "needs_revision": False,
        "confidence": 0.9,
        "format_check": {},
        "content_check": {"passed": True, "issues": []},
        "logic_check": {"passed": True, "issues": []},
        "language_check": {"passed": True, "issues": []},
        "fact_check": {"passed": True, "issues": []},
        "suggestions": [],
        "revision_focus": [],
    })
    with pytest.raises(RuntimeError, match=r"format_check\.passed"):
        AgentOrchestrator._validated_review_metadata(incomplete)

    contradictory = FakeResult(metadata={
        "needs_revision": False,
        "confidence": 0.9,
        "format_check": {"passed": False, "issues": ["标题错误"]},
        "content_check": {"passed": True, "issues": []},
        "logic_check": {"passed": True, "issues": []},
        "language_check": {"passed": True, "issues": []},
        "fact_check": {"passed": True, "issues": []},
        "suggestions": ["修改标题"],
        "revision_focus": ["标题"],
    })
    with pytest.raises(RuntimeError, match="does not request revision"):
        AgentOrchestrator._validated_review_metadata(contradictory)


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph is not installed")
def test_document_graph_generates_unique_serializable_run_state_without_thread_id():
    runner = DocumentGraphRunner(FakeOrchestrator())
    _events, on_think = think_collector()

    first = runner.run(prepared_run(), think_handler=on_think)
    second = DocumentGraphRunner(FakeOrchestrator()).run(
        prepared_run(),
        think_handler=on_think,
    )

    assert first.run_id
    assert second.run_id
    assert first.run_id != second.run_id
    assert first.run_id != "default"
    json.dumps(runner._initial_state(prepared_run(), first.run_id), ensure_ascii=False)


def test_orchestrator_prefers_graph_by_default_and_keeps_linear_kill_switch(monkeypatch):
    monkeypatch.delenv("AGENT_ORCHESTRATOR", raising=False)
    assert AgentOrchestrator._langgraph_requested() is True

    monkeypatch.setenv("AGENT_ORCHESTRATOR", "linear")
    assert AgentOrchestrator._langgraph_requested() is False
