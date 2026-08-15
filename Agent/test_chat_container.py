import json
from pathlib import Path

from chat_container import ChatContainerDependencies, ChatServiceContainer
from chat_runtime import ChatGraphRuntime


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

    def update_rolling_summary(
        self, session_id, message, response, plan, sources, *, effect_key=None
    ):
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
    assert graph_runtime.runtime_mode == "graph"
    assert graph_runtime.uses_langgraph is True


def test_chat_container_default_runtime_streams_through_graph(monkeypatch):
    memory = FakeMemory()
    container = build_container(memory)
    monkeypatch.delenv("CHAT_RUNTIME", raising=False)

    runtime = container.chat_runtime()
    events = parse_sse(runtime.stream(
        {"message": "你是谁？"},
        user_id="user_1",
        user_info=None,
    ))

    event_types = [event["type"] for event in events]
    assert runtime.runtime_mode == "graph"
    assert "tool_plan" in event_types
    assert "tool_call" in event_types
    assert event_types[-2:] == ["run_done", "done"]
    assert events[-1]["intent"] == "identity_help"
    run_id = events[-1]["run_id"]
    checkpoint_next_nodes = {
        node
        for snapshot in runtime._graph.get_state_history({
            "configurable": {"thread_id": run_id}
        })
        for node in snapshot.next
    }
    assert {
        "prepare",
        "plan_tools",
        "select_step",
        "tool_identity_help",
        "collect_result",
        "finalize",
    }.issubset(checkpoint_next_nodes)


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


def test_real_graph_and_planner_rollback_emit_byte_identical_sse(monkeypatch):
    monkeypatch.delenv("CHAT_RUNTIME", raising=False)
    container = build_container(FakeMemory())
    graph_runtime = container.chat_runtime()
    graph_runtime.deps.run_id_factory = lambda: "fixed-run"
    planner_runtime = ChatGraphRuntime(
        graph_runtime.deps,
        runtime_mode="planner",
    )

    payload = {"message": "你是谁？", "session_id": "same-session"}
    graph_chunks = list(graph_runtime.stream(payload, user_id="u1", user_info=None))
    planner_chunks = list(planner_runtime.stream(payload, user_id="u1", user_info=None))

    assert graph_chunks == planner_chunks
    assert parse_sse(graph_chunks)[-1]["type"] == "done"
