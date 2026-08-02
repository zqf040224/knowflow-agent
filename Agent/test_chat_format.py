import json
from pathlib import Path
from types import SimpleNamespace

from chat_format import DocumentFormatDependencies, DocumentFormatStreamService


class FakeMemory:
    def __init__(self):
        self.messages = []
        self.context = {}
        self.summaries = []

    def add_message(self, session_id, role, content, metadata=None):
        self.messages.append((session_id, role, content, metadata or {}))

    def set_context(self, session_id, key, value):
        self.context[(session_id, key)] = value

    def update_rolling_summary(self, session_id, message, response, plan, sources):
        self.summaries.append((session_id, message, response, plan, sources))


class FakeKnowledgeAgent:
    def process(self, payload):
        return SimpleNamespace(
            success=True,
            content="参考内容",
            metadata={"results": [{"filename": "制度.docx"}, {"source": "/tmp/材料.pdf"}]},
        )


class FakeWriter:
    def __init__(self):
        self.last_usage = {
            "agent": "Writer",
            "model": "fake-model",
            "stream": True,
            "prompt_chars": 10,
            "completion_chars": 8,
            "duration_ms": 12,
        }

    def process_stream(self, input_data):
        yield "第一段"
        yield "第二段"


class FailingWriter:
    last_usage = {}

    def process_stream(self, input_data):
        raise RuntimeError("writer failed")
        yield ""


def parse_sse(chunks):
    events = []
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    return events


def build_service(memory, usage_calls, writer_factory=FakeWriter):
    service = DocumentFormatStreamService(DocumentFormatDependencies(
        memory=memory,
        knowledge_agent=FakeKnowledgeAgent(),
        spreadsheet_db_path=Path("/tmp/nonexistent.sqlite"),
        writer_factory=writer_factory,
        resolve_export_template=lambda text, plan, request: "default",
        record_token_usage=lambda **kwargs: usage_calls.append(kwargs),
    ))
    service._audit_spreadsheet_facts = lambda document_content, evidence_items: {
        "passed": True,
        "issues": [],
        "verified_claims": [],
        "unverified_claims": [],
        "spreadsheet_evidence_count": 0,
    }
    return service


def test_document_format_stream_success_contract():
    memory = FakeMemory()
    usage_calls = []
    service = build_service(memory, usage_calls)

    events = parse_sse(service.stream(
        "[文件内容]\n材料\n[/文件内容]\n\n[用户提问]\n请改为公文格式",
        "session_1",
        "user_1",
        user_info=SimpleNamespace(to_dict=lambda: {"user_id": "user_1"}),
        display_message="请改为公文格式",
        route=SimpleNamespace(to_dict=lambda: {"intent": "doc_formatting", "actions": []}),
    ))

    event_types = [event["type"] for event in events]
    assert event_types[:3] == ["start", "session", "route"]
    assert "thinking_start" in event_types
    assert "thinking_done" in event_types
    assert "answer_start" in event_types
    assert [event["data"] for event in events if event["type"] == "answer_delta"] == ["第一段", "第二段"]
    assert "plan" in event_types
    assert "content" in event_types
    done = events[-1]
    assert done["type"] == "done"
    assert done["intent"] == "doc_formatting"
    assert done["document"] == "第一段第二段"
    assert done["export_template"] == "default"
    assert done["source_filenames"] == ["制度.docx", "材料.pdf"]
    assert memory.context[("session_1", "last_document")] == "第一段第二段"
    assert usage_calls[0]["status"] == "success"
    assert usage_calls[0]["mode"] == "document"


def test_document_format_stream_records_failure():
    memory = FakeMemory()
    usage_calls = []
    service = build_service(memory, usage_calls, writer_factory=FailingWriter)

    events = parse_sse(service.stream(
        "请处理格式",
        "session_1",
        "user_1",
        display_message="请处理格式",
    ))

    assert events[-1]["type"] == "error"
    assert usage_calls[-1]["status"] == "failed"
    assert usage_calls[-1]["mode"] == "quick"
