from types import SimpleNamespace

from agents.document_linear_runner import DocumentLinearRunner


class FakeContext:
    def __init__(self):
        self.plan = {"task_type": "公文生成", "document_type": "通知", "need_web_search": False}
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

    def __init__(self, *, reflect=False):
        self.reflect = reflect
        self._reflection_done = False
        self.write_count = 0
        self.review_count = 0
        self.calls = []

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
        return FakeResult(content=f"doc-v{self.write_count}")

    def _step_review(self, ctx, document_content, think_handler):
        self.review_count += 1
        self.calls.append(("review", self.review_count, document_content))
        needs_revision = self.review_count == 1 and not self.reflect
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


def test_document_linear_runner_revises_until_review_passes():
    orchestrator = FakeOrchestrator()
    events, on_think = think_collector()

    result = DocumentLinearRunner(orchestrator).run(prepared_run(), think_handler=on_think)

    assert result.document_content == "doc-v2"
    assert [item["step"] for item in result.ctx.run_records] == [
        "context_plan",
        "retrieval",
        "write",
        "review",
        "write",
        "review",
    ]
    assert result.ctx.last_document == "doc-v1"
    assert result.ctx.audit_summary == {"ok": True}
    assert any("第1轮已汇总审核意见" in message for _agent, _emoji, message in events)
    assert any(agent == "Reviewer" for agent, _emoji, _message in events)


def test_document_linear_runner_can_reflect_before_success():
    orchestrator = FakeOrchestrator(reflect=True)
    _events, on_think = think_collector()

    result = DocumentLinearRunner(orchestrator).run(prepared_run(), think_handler=on_think)

    assert result.document_content == "doc-v1"
    assert ("reflection", "doc-v1") in orchestrator.calls
    assert orchestrator._reflection_done is True
    assert [item["step"] for item in result.ctx.run_records] == [
        "context_plan",
        "retrieval",
        "write",
        "review",
        "reflection",
    ]
    assert result.ctx.revision_history[-1]["source"] == "reflection"
