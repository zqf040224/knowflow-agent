from types import SimpleNamespace

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
    def __init__(self, *, metadata=None):
        self.metadata = metadata or {}


class FakeWriter:
    def __init__(self, chunks=None):
        self.chunks = chunks or ["正文"]

    def process_stream(self, payload):
        self.payload = payload
        for chunk in self.chunks:
            yield chunk


class FakeOrchestrator:
    MAX_TOTAL_ROUNDS = 3

    def __init__(self):
        self.think_log = []
        self._reflection_done = False
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
        return FakeResult(metadata={
            "needs_revision": False,
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
    assert [record["step"] for record in done["run_records"]] == [
        "context_plan",
        "retrieval",
        "write",
        "review",
    ]
