from types import SimpleNamespace

import pytest

from agents.document_graph_state import DocumentGraphRuntime
from agents.document_graph_steps import DocumentGraphSteps
from agents.orchestrator import AgentOrchestrator


class FakeContext:
    def __init__(self):
        self.user_request = "写通知"
        self.context_analysis = {}
        self.plan = {"need_web_search": False, "task_type": "公文生成"}
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

    def __init__(self):
        self.recorded = []

    def _step_review(self, ctx, document_content, think_handler):
        return FakeResult(metadata={
            "needs_revision": True,
            "revision_focus": ["结构"],
            "suggestions": ["补标题"],
            "format_check": {"issues": ["格式"]},
            "content_check": {"issues": ["内容"]},
            "logic_check": {"issues": ["逻辑"]},
            "language_check": {"issues": ["语言"]},
            "fact_check": {"issues": ["事实"]},
            "spreadsheet_audit": {"ok": True},
            "confidence": 0.66,
        })

    def _step_reflection(self, ctx, document_content, think_handler):
        return FakeResult(metadata={
            "needs_revision": True,
            "revision_suggestions": ["加依据"],
            "weaknesses": ["依据不足", "表述略空"],
            "counter_arguments": ["缺少反例"],
            "logic_score": 0.72,
        })

    def _record_step(self, ctx, step, start_time, **extra):
        ctx.run_records.append({"step": step, **extra})

    def _combined_revision_focus(self, review_meta, reflection_meta=None):
        return review_meta.get("revision_focus") or (reflection_meta or {}).get("revision_suggestions") or []

    def _should_reflect(self, ctx, review_meta, revision_round):
        return True


def build_runtime(orchestrator=None):
    events = []

    def think_handler(agent_name, emoji, message):
        events.append((agent_name, emoji, message))

    context = DocumentGraphRuntime(
        orchestrator=orchestrator or FakeOrchestrator(),
        think_handler=think_handler,
    )
    return SimpleNamespace(context=context), events


@pytest.fixture(autouse=True)
def no_custom_stream(monkeypatch):
    monkeypatch.setattr(
        "agents.document_graph_steps.get_stream_writer",
        lambda: (lambda _event: None),
    )


def state_for(orchestrator, ctx=None, **extra):
    state = DocumentGraphSteps.context_to_state(orchestrator, ctx or FakeContext())
    state.update({
        "document_content": "正文",
        "revision_round": 0,
        "review_meta": {},
        "reflection_meta": {},
        "reflection_done": False,
        "run_status": "running",
        "errors": [],
        **extra,
    })
    return state


def test_review_step_records_audit_history_and_reflection_route():
    orchestrator = FakeOrchestrator()
    runtime, _events = build_runtime(orchestrator)

    update = DocumentGraphSteps().review(state_for(orchestrator, revision_round=1), runtime)

    assert update["review_meta"]["needs_revision"] is True
    assert update["audit_summary"] == {"ok": True}
    assert update["should_reflect"] is True
    assert update["run_records"][0]["step"] == "review"
    assert update["revision_history"][0]["round"] == 2
    assert DocumentGraphSteps.route_after_review(update) == "reflection"


def test_reflection_marker_lives_in_state_and_emits_revision_hint():
    orchestrator = FakeOrchestrator()
    runtime, events = build_runtime(orchestrator)

    update = DocumentGraphSteps().reflection(state_for(orchestrator), runtime)

    assert update["reflection_done"] is True
    assert update["reflection_meta"]["needs_revision"] is True
    assert update["revision_history"][0]["source"] == "reflection"
    assert any("依据不足" in message for _agent, _emoji, message in events)


def test_decide_distinguishes_revision_pass_and_max_revisions():
    orchestrator = FakeOrchestrator()
    runtime, events = build_runtime(orchestrator)
    steps = DocumentGraphSteps()

    revise = steps.decide(state_for(
        orchestrator,
        document_content="第一版",
        review_meta={"needs_revision": True, "revision_focus": ["结构"]},
    ), runtime)
    passed = steps.decide(state_for(
        orchestrator,
        review_meta={"needs_revision": False},
    ), runtime)
    exhausted = steps.decide(state_for(
        orchestrator,
        revision_round=2,
        review_meta={"needs_revision": True},
    ), runtime)

    assert revise["revision_round"] == 1
    assert revise["continue_revision"] is True
    assert revise["last_document"] == "第一版"
    assert passed["quality_status"] == "passed"
    assert exhausted["quality_status"] == "max_revisions"
    assert not any(
        agent == "Reviewer" and "审核通过" in message
        for agent, _emoji, message in events[-1:]
    )


def test_graph_state_validator_rejects_runtime_objects_instead_of_stringifying():
    class AuthenticationObject:
        def __repr__(self):
            return "<secret-auth-object>"

    with pytest.raises(TypeError, match="unsupported runtime object"):
        AgentOrchestrator._graph_json_safe(AuthenticationObject())


def test_review_failure_routes_to_error_terminal():
    orchestrator = FakeOrchestrator()
    orchestrator._step_review = lambda *_args, **_kwargs: FakeResult(
        success=False,
        error="review unavailable",
    )
    runtime, _events = build_runtime(orchestrator)

    update = DocumentGraphSteps().review(state_for(orchestrator), runtime)

    assert update["run_status"] == "failed"
    assert update["error_step"] == "review"
    assert DocumentGraphSteps.route_after_review(update) == "error"
