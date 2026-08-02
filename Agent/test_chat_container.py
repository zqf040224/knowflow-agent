import json
from pathlib import Path

from chat_container import ChatContainerDependencies, ChatServiceContainer
from chat_runtime import LANGGRAPH_AVAILABLE


class FakeMemory:
    def __init__(self):
        self.messages = []
        self.context = {}

    def get_or_create_session(self, user_id, session_id=None):
        return session_id or "session_1"

    def get_context_for_prompt(self, session_id, max_messages=5):
        return ""

    def get_context(self, session_id, key, default=None):
        return self.context.get((session_id, key), default)

    def add_message(self, session_id, role, content, metadata=None):
        self.messages.append((session_id, role, content, metadata or {}))

    def set_context(self, session_id, key, value):
        self.context[(session_id, key)] = value

    def update_rolling_summary(self, session_id, message, response, plan, sources):
        pass


class FakeUploadManager:
    def get_temp_content(self, file_id, user_id):
        return ""

    def get_temp_file_info(self, file_id, user_id):
        return {}


def parse_sse(chunks):
    events = []
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    return events


def build_container(memory=None):
    return ChatServiceContainer(ChatContainerDependencies(
        memory=memory or FakeMemory(),
        upload_manager=FakeUploadManager(),
        knowledge_agent=object(),
        deepseek_api_key="",
        spreadsheet_db_path=Path(":memory:"),
        reimbursement_template_files={"meeting": "会议费.xlsx"},
        reimbursement_detector=lambda text, requested="auto": "",
        orchestrator_factory=lambda *args, **kwargs: object(),
        writer_factory=lambda: object(),
        resolve_export_template=lambda text, plan=None, user_request="": "",
        record_token_usage=lambda **kwargs: None,
        record_agent_run_token_usage=lambda *args, **kwargs: None,
    ))


def test_chat_container_caches_services_and_runtime(monkeypatch):
    container = build_container()

    assert container.rag_qa_service() is container.rag_qa_service()
    assert container.document_format_service() is container.document_format_service()
    assert container.document_draft_service() is container.document_draft_service()
    assert container.lightweight_chat_service() is container.lightweight_chat_service()

    monkeypatch.setenv("CHAT_RUNTIME", "legacy")
    legacy_runtime = container.chat_runtime()
    assert legacy_runtime is container.chat_runtime()
    assert legacy_runtime.uses_langgraph is False

    monkeypatch.setenv("CHAT_RUNTIME", "langgraph")
    graph_runtime = container.chat_runtime()
    assert graph_runtime is container.chat_runtime()
    assert graph_runtime is not legacy_runtime
    assert graph_runtime.uses_langgraph is LANGGRAPH_AVAILABLE


def test_chat_container_runtime_preserves_identity_contract(monkeypatch):
    memory = FakeMemory()
    container = build_container(memory)
    monkeypatch.setenv("CHAT_RUNTIME", "legacy")

    events = parse_sse(container.chat_runtime().stream(
        {"message": "你是谁？"},
        user_id="user_1",
        user_info=None,
    ))

    assert events[-1]["type"] == "done"
    assert events[-1]["intent"] == "identity_help"
    assert "智能知识库助手" in events[-1]["answer"]
    assert memory.context[("session_1", "last_answer")]
