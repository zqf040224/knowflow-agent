import json
from types import SimpleNamespace

from chat_architecture import (
    INTENT_CLARIFY,
    INTENT_DOC_DRAFTING,
    INTENT_DOC_FORMATTING,
    INTENT_FORM_TEMPLATE_EXPORT,
    INTENT_IDENTITY_HELP,
    INTENT_SPREADSHEET_TRANSFORM,
)
from chat_lightweight import LightweightChatDependencies, LightweightChatStreamService
from chat_runtime import ChatGraphRuntime, ChatRuntimeDependencies


class FakeMemory:
    def __init__(self):
        self.messages = []
        self.context = {}
        self.summaries = []
        self.long_term_context = ""
        self.remembered = []

    def get_or_create_session(self, user_id, session_id=None):
        return session_id or "session_1"

    def get_context_for_prompt(self, session_id, max_messages=5, memory_query=None):
        return self.long_term_context

    def get_memory_context(self, user_id, query, limit=3):
        return self.long_term_context

    def remember_explicit_memory(self, user_id, session_id, message):
        self.remembered.append((user_id, session_id, message))

    def get_context(self, session_id, key, default=None):
        return self.context.get((session_id, key), default)

    def add_message(self, session_id, role, content, metadata=None):
        self.messages.append((session_id, role, content, metadata or {}))

    def set_context(self, session_id, key, value):
        self.context[(session_id, key)] = value

    def update_rolling_summary(self, session_id, message, response, plan, sources):
        self.summaries.append((session_id, message, response, plan, sources))


class FakeUploadManager:
    def get_temp_content(self, file_id, user_id):
        return {
            "file_1": "这是一段待转换材料。",
            "sheet_1": "姓名,金额\n张三,10\n李四,20",
        }.get(file_id, "")

    def get_temp_file_info(self, file_id, user_id):
        if file_id == "sheet_1":
            return {"filename": "预算.xlsx", "content": "姓名,金额\n张三,10\n李四,20"}
        return {"filename": "材料.docx", "content": "这是一段待转换材料。"}


def parse_sse(chunks):
    events = []
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    return events


def build_runtime(calls, memory=None, *, use_langgraph=True):
    runtime_memory = memory or FakeMemory()

    def stream_handler(name):
        def _stream(*args):
            calls.append((name, args))
            yield name
        return _stream

    return ChatGraphRuntime(
        ChatRuntimeDependencies(
            memory=runtime_memory,
            upload_manager=FakeUploadManager(),
            reimbursement_detector=lambda text, requested="auto": "travel" if "差旅费" in text else "",
            lightweight_stream=LightweightChatStreamService(LightweightChatDependencies(
                memory=runtime_memory,
                reimbursement_template_files={
                    "travel": "差旅费.xlsx",
                    "meeting": "会议费.xlsx",
                    "labor_expert": "劳务费&专家咨询费.xlsx",
                    "other": "其他费用报销.xlsx",
                },
                assistant_identity_response=lambda: "我是智能知识库助手。",
            )).stream,
            document_format_stream=stream_handler("format"),
            document_draft_stream=stream_handler("draft"),
            rag_qa_stream=stream_handler("rag"),
        ),
        use_langgraph=use_langgraph,
    )


def test_chat_graph_runtime_preserves_identity_sse_contract():
    calls = []
    memory = FakeMemory()
    runtime = build_runtime(calls, memory)

    output = list(runtime.stream(
        {"message": "你是谁？"},
        user_id="user_1",
        user_info=SimpleNamespace(username="tester"),
    ))

    events = parse_sse(output)
    done = events[-1]
    assert calls == []
    assert done["type"] == "done"
    assert done["intent"] == INTENT_IDENTITY_HELP
    assert done["document"] == ""
    assert done["export_template"] == ""
    assert memory.context[("session_1", "last_answer")].startswith("我是智能知识库助手")
    assert ("session_1", "last_document") not in memory.context


def test_chat_graph_runtime_hydrates_attachments_before_dispatch():
    calls = []
    runtime = build_runtime(calls)

    output = list(runtime.stream(
        {"message": "请改为公文格式", "file_ids": ["file_1"], "session_id": "session_x"},
        user_id="user_1",
        user_info=SimpleNamespace(username="tester"),
    ))

    assert output == ["format"]
    name, args = calls[0]
    assert name == "format"
    assert "[文件内容]\n这是一段待转换材料。\n[/文件内容]" in args[0]
    assert args[1] == "session_x"
    assert args[4] == "请改为公文格式"
    assert args[5]["attached_files"][0]["filename"] == "材料.docx"
    assert args[6].intent == INTENT_DOC_FORMATTING


def test_chat_graph_runtime_routes_knowledge_qa_directly_to_rag_handler():
    calls = []
    runtime = build_runtime(calls)

    output = list(runtime.stream(
        {"message": "制度文件在哪里查？"},
        user_id="user_1",
        user_info=SimpleNamespace(username="tester"),
    ))

    assert output == ["rag"]
    name, args = calls[0]
    assert name == "rag"
    assert args[0] == "制度文件在哪里查？"
    assert args[1] == "session_1"
    assert args[6].intent == "knowledge_qa"


def test_chat_graph_runtime_routes_document_drafting_directly_to_draft_handler():
    calls = []
    runtime = build_runtime(calls)

    output = list(runtime.stream(
        {"message": "帮我写一份正式通知"},
        user_id="user_1",
        user_info=SimpleNamespace(username="tester"),
    ))

    assert output == ["draft"]
    name, args = calls[0]
    assert name == "draft"
    assert args[0] == "帮我写一份正式通知"
    assert args[6].intent == INTENT_DOC_DRAFTING


def test_chat_graph_runtime_routes_lightweight_intents_to_lightweight_service():
    scenarios = [
        ("给我导出差旅费报销表", INTENT_FORM_TEMPLATE_EXPORT),
        ("给我导出报销表模板", INTENT_CLARIFY),
        ("把这个表格按金额从高到低排序", INTENT_SPREADSHEET_TRANSFORM),
    ]

    for message, expected_intent in scenarios:
        calls = []
        runtime = build_runtime(calls)
        payload = {"message": message}
        if expected_intent == INTENT_SPREADSHEET_TRANSFORM:
            payload["file_ids"] = ["sheet_1"]

        events = parse_sse(runtime.stream(
            payload,
            user_id="user_1",
            user_info=SimpleNamespace(username="tester"),
        ))

        assert calls == []
        assert events[-1]["type"] == "done"
        assert events[-1]["intent"] == expected_intent


def test_chat_graph_runtime_uses_task_planner_when_available(monkeypatch):
    monkeypatch.delenv("CHAT_RUNTIME", raising=False)
    planned = []
    executed = []

    class FakeTaskPlanner:
        def plan(self, **kwargs):
            planned.append(kwargs)
            return SimpleNamespace(route=SimpleNamespace(intent="knowledge_qa"), steps=[])

    class FakeToolOrchestrator:
        def stream(self, prepared, task_plan):
            executed.append((prepared, task_plan))
            yield "planned-stream"

    memory = FakeMemory()
    memory.long_term_context = "【跨会话长期记忆】\n- 用户偏好字体：黑体"
    runtime = ChatGraphRuntime(ChatRuntimeDependencies(
        memory=memory,
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda text, requested="auto": "",
        lightweight_stream=lambda *args: iter(()),
        document_format_stream=lambda *args: iter(()),
        document_draft_stream=lambda *args: iter(()),
        rag_qa_stream=lambda *args: iter(()),
        task_planner=FakeTaskPlanner(),
        tool_orchestrator=FakeToolOrchestrator(),
    ), use_langgraph=False)

    output = list(runtime.stream(
        {"message": "制度文件在哪里查？"},
        user_id="user_1",
        user_info=SimpleNamespace(username="tester"),
    ))

    assert output == ["planned-stream"]
    assert planned[0]["display_message"] == "制度文件在哪里查？"
    assert "用户偏好字体：黑体" in planned[0]["conversation_context"]
    assert memory.remembered == [("user_1", "session_1", "制度文件在哪里查？")]
    assert executed[0][0].session_id == "session_1"


def test_chat_graph_runtime_legacy_env_bypasses_task_planner(monkeypatch):
    monkeypatch.setenv("CHAT_RUNTIME", "legacy")
    calls = []

    class FailingTaskPlanner:
        def plan(self, **kwargs):
            raise AssertionError("task planner should be bypassed")

    runtime = build_runtime(calls, use_langgraph=False)
    runtime.deps.task_planner = FailingTaskPlanner()
    runtime.deps.tool_orchestrator = object()

    output = list(runtime.stream(
        {"message": "制度文件在哪里查？"},
        user_id="user_1",
        user_info=SimpleNamespace(username="tester"),
    ))

    assert output == ["rag"]
