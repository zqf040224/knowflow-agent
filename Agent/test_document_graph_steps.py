from agents.document_graph_steps import DocumentGraphSteps


class FakeContext:
    def __init__(self):
        self.plan = {"need_web_search": False, "task_type": "公文生成"}
        self.knowledge_sources = []
        self.evidence_items = []
        self.compact_evidence = []
        self.revision_history = []
        self.run_records = []
        self.audit_summary = {}
        self.last_document = ""


class FakeResult:
    def __init__(self, *, content="", metadata=None):
        self.content = content
        self.metadata = metadata or {}


class FakeOrchestrator:
    MAX_TOTAL_ROUNDS = 3

    def __init__(self):
        self._reflection_done = False
        self.recorded = []
        self.reflect = False

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
        return review_meta.get("revision_focus") or reflection_meta.get("revision_suggestions") or []

    def _should_reflect(self, ctx, review_meta, revision_round):
        return self.reflect


def build_steps(orchestrator=None):
    events = []

    def think_handler(agent_name, emoji, message):
        events.append((agent_name, emoji, message))

    return DocumentGraphSteps(orchestrator or FakeOrchestrator(), think_handler), events


def test_review_step_records_audit_and_history_entry():
    ctx = FakeContext()
    steps, _events = build_steps()

    state = steps.review({"ctx": ctx, "document_content": "正文", "revision_round": 1})

    assert state["review_meta"]["needs_revision"] is True
    assert ctx.audit_summary == {"ok": True}
    assert ctx.run_records == [{
        "step": "review",
        "round": 2,
        "needs_revision": True,
        "confidence": 0.66,
    }]
    assert ctx.revision_history == [{
        "round": 2,
        "needs_revision": True,
        "revision_focus": ["结构"],
        "suggestions": ["补标题"],
        "format_issues": ["格式"],
        "content_issues": ["内容"],
        "logic_issues": ["逻辑"],
        "language_issues": ["语言"],
        "fact_issues": ["事实"],
        "spreadsheet_audit": {"ok": True},
        "confidence": 0.66,
    }]


def test_reflection_step_marks_done_and_emits_revision_hint():
    orchestrator = FakeOrchestrator()
    ctx = FakeContext()
    steps, events = build_steps(orchestrator)

    state = steps.reflection({"ctx": ctx, "document_content": "正文", "revision_round": 0})

    assert orchestrator._reflection_done is True
    assert state["reflection_meta"]["needs_revision"] is True
    assert ctx.run_records[0]["step"] == "reflection"
    assert ctx.revision_history[0]["source"] == "reflection"
    assert ctx.revision_history[0]["revision_focus"] == ["加依据"]
    assert any("依据不足" in message for _agent, _emoji, message in events)


def test_decide_step_routes_revision_or_success():
    ctx = FakeContext()
    steps, events = build_steps()

    revise = steps.decide({
        "ctx": ctx,
        "document_content": "第一版",
        "revision_round": 0,
        "review_meta": {"needs_revision": True, "revision_focus": ["结构"]},
    })
    route = steps.route_after_decide(revise)

    success = steps.decide({
        "ctx": ctx,
        "revision_round": 1,
        "review_meta": {"needs_revision": False},
    })

    assert revise == {"ctx": ctx, "revision_round": 1, "continue_revision": True}
    assert ctx.last_document == "第一版"
    assert route == "write"
    assert success == {"ctx": ctx, "continue_revision": False}
    assert any(agent == "Reviewer" for agent, _emoji, _message in events)


def test_route_after_review_uses_reflection_guard():
    orchestrator = FakeOrchestrator()
    orchestrator.reflect = True
    ctx = FakeContext()
    steps, _events = build_steps(orchestrator)

    assert steps.route_after_review({"ctx": ctx, "review_meta": {}, "revision_round": 0}) == "reflection"

    orchestrator._reflection_done = True
    assert steps.route_after_review({"ctx": ctx, "review_meta": {}, "revision_round": 0}) == "decide"
