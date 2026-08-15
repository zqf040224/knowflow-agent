import json
from types import SimpleNamespace

import pytest

from chat_architecture import (
    INTENT_CLARIFY,
    INTENT_DOC_DRAFTING,
    INTENT_DOC_FORMATTING,
    INTENT_FORM_TEMPLATE_EXPORT,
    INTENT_IDENTITY_HELP,
    INTENT_SPREADSHEET_TRANSFORM,
    RouteResult,
)
from chat_events import sse
from chat_lightweight import LightweightChatDependencies, LightweightChatStreamService
from chat_runtime import (
    ChatGraphRuntime,
    ChatRunConflictError,
    ChatRunContext,
    ChatRuntimeDependencies,
)
from task_planner import TaskPlan, TaskStep
from tool_runtime import ChatTool, ToolOrchestrator, ToolRegistry


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

    def update_rolling_summary(
        self, session_id, message, response, plan, sources, *, effect_key=None
    ):
        self.summaries.append(
            (session_id, message, response, plan, sources, effect_key)
        )


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


def build_runtime(calls, memory=None, *, runtime_mode="legacy"):
    runtime_memory = memory or FakeMemory()

    def stream_handler(name):
        def _stream(*args):
            calls.append((name, args))
            yield name
        return _stream

    kwargs = {} if runtime_mode is None else {"runtime_mode": runtime_mode}
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
        **kwargs,
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


def test_chat_graph_runtime_uses_real_state_graph_and_runtime_context(monkeypatch):
    monkeypatch.delenv("CHAT_RUNTIME", raising=False)
    planned = []
    executed = []
    raw_chunks = [
        sse({"type": "start"}),
        sse({"type": "content", "data": "planned-stream"}),
        sse({"type": "done", "intent": "knowledge_qa", "answer": "planned-stream"}),
    ]

    class FakeTaskPlanner:
        def plan(self, **kwargs):
            planned.append(kwargs)
            route = RouteResult(
                intent="knowledge_qa",
                confidence=0.9,
                reason="test",
            )
            return TaskPlan(
                task_type="knowledge_qa",
                steps=[TaskStep(tool="knowledge_qa", reason="test")],
                route=route,
            )

    class FakeToolOrchestrator:
        def stream(self, prepared, task_plan):
            executed.append((prepared, task_plan))
            yield from raw_chunks

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
    ))

    user_info = SimpleNamespace(username="tester")

    output = list(runtime.stream(
        {"message": "制度文件在哪里查？"},
        user_id="user_1",
        user_info=user_info,
    ))

    assert runtime.runtime_mode == "graph"
    assert runtime.uses_langgraph is True
    assert output[:2] == raw_chunks[:2]
    done = parse_sse(output)[-1]
    assert done["type"] == "done"
    assert done["runtime"] == "chat-v3"
    assert done["run_id"]
    assert planned[0]["display_message"] == "制度文件在哪里查？"
    assert planned[0]["user_info"] is user_info
    assert "用户偏好字体：黑体" in planned[0]["conversation_context"]
    assert memory.remembered == [("user_1", "session_1", "制度文件在哪里查？")]
    assert executed[0][0].session_id == "session_1"
    assert executed[0][0].user_info is user_info
    assert isinstance(executed[0][1], TaskPlan)


def test_graph_state_is_json_serializable_and_excludes_runtime_user_info():
    class FakeTaskPlanner:
        def plan(self, **_kwargs):
            return TaskPlan(
                task_type="knowledge_qa",
                steps=[],
                route=RouteResult("knowledge_qa", 1.0, "test"),
            )

    class FakeToolOrchestrator:
        def stream(self, prepared, task_plan):
            yield sse({"type": "done", "intent": "knowledge_qa", "answer": "ok"})

    runtime = ChatGraphRuntime(ChatRuntimeDependencies(
        memory=FakeMemory(),
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda text, requested="auto": "",
        lightweight_stream=lambda *args: iter(()),
        document_format_stream=lambda *args: iter(()),
        document_draft_stream=lambda *args: iter(()),
        rag_qa_stream=lambda *args: iter(()),
        task_planner=FakeTaskPlanner(),
        tool_orchestrator=FakeToolOrchestrator(),
    ), runtime_mode="graph")
    initial_state = runtime._initial_state(
        {"message": "hello", "file_ids": ["file_1"]},
        user_id="user_1",
    )
    final_state = runtime._graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": initial_state["run_id"]}},
        context=ChatRunContext(user_info=SimpleNamespace(username="tester")),
    )

    json.dumps(final_state, ensure_ascii=False)
    assert "user_info" not in final_state
    assert "prepared" not in final_state
    assert "stream" not in final_state
    assert "[文件内容]" not in final_state["message"]
    assert final_state["status"] == "completed"
    assert final_state["task_plan"]["route"]["intent"] == "knowledge_qa"


def test_draft_document_is_registered_as_native_checkpointer_inheriting_subgraph():
    class FakeTaskPlanner:
        def plan(self, **_kwargs):
            return TaskPlan(task_type="document_drafting", steps=[])

    class FakeToolOrchestrator:
        def stream(self, prepared, task_plan):
            yield sse({"type": "done", "document": "ok"})

    runtime = ChatGraphRuntime(ChatRuntimeDependencies(
        memory=FakeMemory(),
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda text, requested="auto": "",
        lightweight_stream=lambda *args: iter(()),
        document_format_stream=lambda *args: iter(()),
        document_draft_stream=lambda *args: iter(()),
        rag_qa_stream=lambda *args: iter(()),
        task_planner=FakeTaskPlanner(),
        tool_orchestrator=FakeToolOrchestrator(),
    ), runtime_mode="graph")

    subgraphs = dict(runtime._graph.get_subgraphs())
    assert "tool_draft_document" in subgraphs
    assert subgraphs["tool_draft_document"].name == "draft_document_subgraph"
    assert subgraphs["tool_draft_document"].checkpointer is None


def test_graph_and_planner_modes_preserve_identical_raw_sse():
    chunks = [
        sse({"type": "start"}),
        sse({"type": "answer_delta", "data": "same"}),
        sse({"type": "done", "intent": "knowledge_qa", "answer": "same"}),
    ]

    class FakeTaskPlanner:
        def plan(self, **_kwargs):
            return TaskPlan(
                task_type="knowledge_qa",
                steps=[],
                route=RouteResult("knowledge_qa", 1.0, "test"),
            )

    class FakeToolOrchestrator:
        def stream(self, prepared, task_plan):
            yield from chunks

    def make_runtime(mode):
        return ChatGraphRuntime(ChatRuntimeDependencies(
            memory=FakeMemory(),
            upload_manager=FakeUploadManager(),
            reimbursement_detector=lambda text, requested="auto": "",
            lightweight_stream=lambda *args: iter(()),
            document_format_stream=lambda *args: iter(()),
            document_draft_stream=lambda *args: iter(()),
            rag_qa_stream=lambda *args: iter(()),
                task_planner=FakeTaskPlanner(),
                tool_orchestrator=FakeToolOrchestrator(),
                run_id_factory=lambda: "fixed-run-id",
            ), runtime_mode=mode)

    payload = {"message": "same request", "session_id": "same_session"}
    graph_output = list(make_runtime("graph").stream(payload, user_id="user_1", user_info=None))
    planner_output = list(make_runtime("planner").stream(payload, user_id="user_1", user_info=None))

    assert graph_output == planner_output
    assert graph_output[:2] == chunks[:2]
    assert parse_sse(graph_output)[-1]["run_id"] == "fixed-run-id"


def test_draft_native_subgraph_preserves_planner_sse_byte_for_byte():
    route = RouteResult(
        intent=INTENT_DOC_DRAFTING,
        confidence=1.0,
        reason="test",
    )
    task_plan = TaskPlan(
        task_type="document_drafting",
        steps=[TaskStep(tool="draft_document", reason="test")],
        route=route,
    )

    class FixedPlanner:
        def plan(self, **_kwargs):
            return task_plan

    def draft_stream(
        _message,
        session_id,
        _user_id,
        _user_info,
        _display_message,
        _user_metadata,
        _route,
    ):
        yield sse({"type": "start"})
        yield sse({"type": "session", "session_id": session_id})
        yield sse({"type": "content", "data": "正文"})
        yield sse({
            "type": "done",
            "intent": INTENT_DOC_DRAFTING,
            "answer": "正文",
            "document": "正文",
            "session_id": session_id,
            "plan": {"task_type": "document_drafting"},
            "actions": [],
            "source_filenames": [],
            "source_details": [],
        })

    def make_runtime(mode):
        registry = ToolRegistry()
        registry.register(ChatTool(
            name="draft_document",
            description="test",
            risk_level="low",
            input_schema={"message": "string"},
            stream=draft_stream,
        ))
        return ChatGraphRuntime(ChatRuntimeDependencies(
            memory=FakeMemory(),
            upload_manager=FakeUploadManager(),
            reimbursement_detector=lambda text, requested="auto": "",
            lightweight_stream=lambda *args: iter(()),
            document_format_stream=lambda *args: iter(()),
            document_draft_stream=draft_stream,
            rag_qa_stream=lambda *args: iter(()),
            task_planner=FixedPlanner(),
            tool_orchestrator=ToolOrchestrator(registry),
            run_id_factory=lambda: "fixed-draft-run",
        ), runtime_mode=mode)

    payload = {
        "message": "写通知",
        "session_id": "same-session",
        "file_ids": ["file_1"],
    }
    graph_output = list(make_runtime("graph").stream(
        payload,
        user_id="user-1",
        user_info=None,
    ))
    planner_output = list(make_runtime("planner").stream(
        payload,
        user_id="user-1",
        user_info=None,
    ))

    assert graph_output == planner_output
    assert "正文" in "".join(graph_output)


def test_production_draft_service_mounts_full_native_graph_and_preserves_sse():
    from langgraph.checkpoint.memory import InMemorySaver

    from agents.document_graph_runner import DocumentGraphRunner
    from chat_draft import DocumentDraftDependencies, DocumentDraftStreamService
    from test_document_graph_runner import FakeOrchestrator

    route = RouteResult(
        intent=INTENT_DOC_DRAFTING,
        confidence=1.0,
        reason="native document test",
    )
    task_plan = TaskPlan(
        task_type="document_drafting",
        steps=[TaskStep(tool="draft_document", reason="native document test")],
        route=route,
    )

    class FixedPlanner:
        def plan(self, **_kwargs):
            return task_plan

    class DraftMemory(FakeMemory):
        def get_user_profile(self, _user_id):
            return None

    class NativeOrchestrator(FakeOrchestrator):
        def __init__(self, session_id):
            super().__init__(reflect=True)
            self.session_id = session_id
            self._current_effect_run_id = ""

        def _prepare_document_run(self, user_request, session_id=None, *, run_id=""):
            self.think_log = []
            self.session_id = session_id or self.session_id
            return SimpleNamespace(
                user_request=user_request,
                request_with_context=user_request,
                previous_context="",
                session_id=self.session_id,
                run_id=run_id,
            )

        def _think_handler(self, on_think=None):
            def handler(agent, emoji, message):
                self.think_log.append({
                    "agent": agent,
                    "emoji": emoji,
                    "message": message,
                })
                if on_think:
                    on_think(agent, emoji, message)

            return handler

        def run_stream(self, user_request, on_think=None, session_id=None, *, run_id=""):
            prepared = self._prepare_document_run(
                user_request,
                session_id=session_id,
                run_id=run_id,
            )
            runner = DocumentGraphRunner(self, checkpointer=InMemorySaver())
            yield from runner.stream(
                prepared,
                think_handler=self._think_handler(on_think),
                thread_id=run_id,
                user_request=user_request,
            )

    def make_runtime(mode):
        memory = DraftMemory()
        service = DocumentDraftStreamService(DocumentDraftDependencies(
            memory=memory,
            orchestrator_factory=lambda session_id, **_kwargs: NativeOrchestrator(session_id),
            resolve_export_template=lambda *_args: "official_document",
            record_agent_run_token_usage=lambda *_args, **_kwargs: None,
            record_token_usage=lambda **_kwargs: None,
        ))
        registry = ToolRegistry()
        registry.register(ChatTool(
            name="draft_document",
            description="native test",
            risk_level="low",
            input_schema={"message": "string"},
            stream=service.stream,
        ))
        return ChatGraphRuntime(ChatRuntimeDependencies(
            memory=memory,
            upload_manager=FakeUploadManager(),
            reimbursement_detector=lambda *_args: "",
            lightweight_stream=lambda *_args: iter(()),
            document_format_stream=lambda *_args: iter(()),
            document_draft_stream=service.stream,
            rag_qa_stream=lambda *_args: iter(()),
            task_planner=FixedPlanner(),
            tool_orchestrator=ToolOrchestrator(registry),
            run_id_factory=lambda: "native-document-run",
        ), runtime_mode=mode)

    graph_runtime = make_runtime("graph")
    parent_subgraphs = dict(graph_runtime._graph.get_subgraphs())
    draft_subgraph = parent_subgraphs["tool_draft_document"]
    assert draft_subgraph.checkpointer is None
    nested = dict(draft_subgraph.get_subgraphs())
    document_graph = nested["document_workflow"]
    assert document_graph.checkpointer is None
    assert {
        "context_plan",
        "retrieval",
        "write",
        "review",
        "reflection",
        "decide",
        "finalize",
    }.issubset(document_graph.get_graph().nodes)

    payload = {
        "message": "写通知",
        "session_id": "same-session",
        "file_ids": ["file_1"],
    }
    graph_output = list(graph_runtime.stream(payload, user_id="user-1", user_info=None))
    planner_output = list(make_runtime("planner").stream(
        payload,
        user_id="user-1",
        user_info=None,
    ))

    assert graph_output == planner_output
    events = parse_sse(graph_output)
    assert events[-1]["type"] == "done"
    assert sum(event["type"] == "done" for event in events) == 1
    assert events[-1]["document"] == "doc-v1"
    checkpoint = graph_runtime._graph.get_state({
        "configurable": {"thread_id": "native-document-run"},
    })
    checkpoint_json = json.dumps(checkpoint.values, ensure_ascii=False)
    assert "这是一段待转换材料。" not in checkpoint_json
    assert '"file_id": "file_1"' in checkpoint_json
    checkpoint_history = graph_runtime._graph.get_state_history({
        "configurable": {"thread_id": "native-document-run"},
    })
    assert all(
        "这是一段待转换材料。" not in json.dumps(
            snapshot.values,
            ensure_ascii=False,
        )
        for snapshot in checkpoint_history
    )

    first_context = graph_runtime._new_run_context(
        user_info=None,
        execution_token="run-one-token",
        lease=None,
        recovery=False,
    )
    second_context = graph_runtime._new_run_context(
        user_info=None,
        execution_token="run-two-token",
        lease=None,
        recovery=False,
    )
    common = {
        "session_id": "same-session",
        "user_id": "user-1",
        "working_message": "写通知",
        "display_message": "写通知",
        "attachments": [{"file_id": "file_1", "filename": "材料.docx"}],
    }
    first_execution = first_context.get_document_execution(
        {**common, "run_id": "concurrent-run-one"},
        prepare=True,
    )
    second_execution = second_context.get_document_execution(
        {**common, "run_id": "concurrent-run-two"},
        prepare=True,
    )
    assert first_context.document_runtime_cache is not second_context.document_runtime_cache
    assert first_execution.orchestrator is not second_execution.orchestrator
    assert "[文件内容]" in first_execution.prepared_run.user_request


def _build_native_snapshot_runtime(
    *,
    memory,
    upload_manager,
    artifact_store,
    checkpointer,
    run_id,
    node_inputs,
):
    """Build the real native document mount with deterministic fake agents."""

    from agents.document_graph_runner import DocumentGraphRunner
    from chat_draft import DocumentDraftDependencies, DocumentDraftStreamService
    from test_document_graph_runner import FakeContext, FakeOrchestrator

    route = RouteResult(
        intent=INTENT_DOC_DRAFTING,
        confidence=1.0,
        reason="native snapshot test",
    )
    task_plan = TaskPlan(
        task_type="document_drafting",
        steps=[TaskStep(tool="draft_document", reason="native snapshot test")],
        route=route,
    )

    class FixedPlanner:
        def plan(self, **_kwargs):
            return task_plan

    class SnapshotOrchestrator(FakeOrchestrator):
        def __init__(self, session_id, profile):
            super().__init__(reflect=True)
            self.session_id = session_id
            self.user_profile = {
                name: getattr(profile, name)
                for name in (
                    "preferred_font",
                    "preferred_size",
                    "writing_style",
                    "common_doc_types",
                    "name",
                    "department",
                )
                if profile is not None and hasattr(profile, name)
            }
            self._current_effect_run_id = ""
            self._current_agent_memory_context = ""
            self._document_runtime_snapshot = {}

        def _prepare_document_run(self, user_request, session_id=None, *, run_id=""):
            self.think_log = []
            self.session_id = session_id or self.session_id
            self._current_agent_memory_context = memory.long_term_context
            previous_context = memory.get_context(
                self.session_id,
                "last_request",
                "",
            )
            request_with_context = user_request
            if previous_context:
                request_with_context = (
                    f"之前的需求：{previous_context}\n\n当前需求：{user_request}"
                )
            return SimpleNamespace(
                user_request=user_request,
                request_with_context=request_with_context,
                previous_context=previous_context,
                session_id=self.session_id,
                run_id=run_id,
            )

        def _step_context_plan(self, request_with_context, previous_context, think_handler):
            snapshot = dict(self._document_runtime_snapshot)
            node_inputs.append({
                "request_with_context": request_with_context,
                "previous_context": previous_context,
                "memory_context": self._current_agent_memory_context,
                "user_profile": dict(self.user_profile),
                "last_document": snapshot.get("last_document"),
                "last_plan": dict(snapshot.get("last_plan") or {}),
                "conversation_history": list(
                    snapshot.get("conversation_history") or []
                ),
            })
            ctx = FakeContext()
            ctx.user_request = request_with_context
            # Simulate an LLM planner copying its entire attachment prompt into
            # both checkpointed fields. The native boundary must redact it.
            ctx.context_analysis = {"planner_echo": request_with_context}
            ctx.plan = {
                "task_type": "公文生成",
                "document_type": "通知",
                "need_web_search": False,
                "planner_echo": request_with_context,
            }
            ctx.user_profile = dict(self.user_profile)
            ctx.memory_context = self._current_agent_memory_context
            ctx.last_document = str(snapshot.get("last_document") or "")
            ctx.last_plan = dict(snapshot.get("last_plan") or {})
            return ctx

        def _think_handler(self, on_think=None):
            def handler(agent, emoji, message):
                self.think_log.append({
                    "agent": agent,
                    "emoji": emoji,
                    "message": message,
                })
                if on_think:
                    on_think(agent, emoji, message)

            return handler

        def run_stream(self, user_request, on_think=None, session_id=None, *, run_id=""):
            prepared = self._prepare_document_run(
                user_request,
                session_id=session_id,
                run_id=run_id,
            )
            yield from DocumentGraphRunner(self).stream(
                prepared,
                think_handler=self._think_handler(on_think),
                thread_id=run_id,
                user_request=user_request,
            )

    service = DocumentDraftStreamService(DocumentDraftDependencies(
        memory=memory,
        orchestrator_factory=(
            lambda session_id, profile=None, **_kwargs: SnapshotOrchestrator(
                session_id,
                profile,
            )
        ),
        resolve_export_template=lambda *_args: "official_document",
        record_agent_run_token_usage=lambda *_args, **_kwargs: None,
        record_token_usage=lambda **_kwargs: None,
    ))
    registry = ToolRegistry()
    registry.register(ChatTool(
        name="draft_document",
        description="native snapshot test",
        risk_level="low",
        input_schema={"message": "string"},
        stream=service.stream,
    ))
    return ChatGraphRuntime(ChatRuntimeDependencies(
        memory=memory,
        upload_manager=upload_manager,
        reimbursement_detector=lambda *_args: "",
        lightweight_stream=lambda *_args: iter(()),
        document_format_stream=lambda *_args: iter(()),
        document_draft_stream=service.stream,
        rag_qa_stream=lambda *_args: iter(()),
        task_planner=FixedPlanner(),
        tool_orchestrator=ToolOrchestrator(registry),
        checkpointer=checkpointer,
        artifact_store=artifact_store,
        run_id_factory=lambda: run_id,
    ), runtime_mode="graph")


class _NativeSnapshotMemory(FakeMemory):
    def __init__(
        self,
        *,
        memory_context,
        profile_marker,
        last_document,
        history_marker,
    ):
        super().__init__()
        self.long_term_context = memory_context
        self.profile = SimpleNamespace(
            preferred_font=profile_marker,
            preferred_size="小四",
            writing_style="正式",
            common_doc_types=[],
            name=profile_marker,
            department="研发部",
        )
        self.history = [{"role": "user", "content": history_marker}]
        self.context.update({
            ("snapshot-session", "last_request"): "首次上一轮需求",
            ("snapshot-session", "last_document"): last_document,
            ("snapshot-session", "last_plan"): {"marker": "FIRST_PLAN"},
        })

    def get_user_profile(self, _user_id):
        return self.profile

    def get_session_history(self, _session_id, limit=10):
        return list(self.history[-limit:])

    def update_user_profile(self, _user_id, updates):
        for key, value in updates.items():
            setattr(self.profile, key, value)


class _NativeSnapshotUploads:
    def __init__(self, body):
        self.body = body

    def get_temp_content(self, file_id, user_id):
        if file_id == "snapshot-file" and user_id == "snapshot-user":
            return self.body
        return ""

    def get_temp_file_info(self, file_id, user_id):
        if file_id == "snapshot-file" and user_id == "snapshot-user":
            return {"filename": "受控材料.docx", "content": self.body}
        return None


def test_native_document_all_checkpoint_namespaces_exclude_runtime_secrets(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    from graph_artifacts import GraphArtifactStore

    attachment_body = "ATTACHMENT_SECRET_这是一段应只存在受控附件中的完整原文。"
    memory_marker = "MEMORY_CONTEXT_SECRET_长期记忆"
    last_document_marker = "LAST_DOCUMENT_SECRET_历史公文正文"
    profile_marker = "PROFILE_SECRET_内部用户标识"
    history_marker = "HISTORY_SECRET_历史对话"
    run_id = "native-secret-checkpoint-run"
    saver = InMemorySaver()
    uploads = _NativeSnapshotUploads(attachment_body)
    artifact_store = GraphArtifactStore(tmp_path / "graph_runs", uploads)
    memory = _NativeSnapshotMemory(
        memory_context=memory_marker,
        profile_marker=profile_marker,
        last_document=last_document_marker,
        history_marker=history_marker,
    )
    node_inputs = []
    runtime = _build_native_snapshot_runtime(
        memory=memory,
        upload_manager=uploads,
        artifact_store=artifact_store,
        checkpointer=saver,
        run_id=run_id,
        node_inputs=node_inputs,
    )

    events = parse_sse(runtime.stream(
        {
            "message": "请根据附件起草文件",
            "session_id": "snapshot-session",
            "file_ids": ["snapshot-file"],
        },
        user_id="snapshot-user",
        user_info=None,
    ))

    assert events[-1]["type"] == "done"
    checkpoints = list(saver.list({
        "configurable": {"thread_id": run_id},
    }))
    namespaces = {
        item.config["configurable"].get("checkpoint_ns", "")
        for item in checkpoints
    }
    assert "" in namespaces
    assert any(namespace for namespace in namespaces)

    channel_values = [
        item.checkpoint.get("channel_values", {})
        for item in checkpoints
    ]
    serialized_values = [
        json.dumps(values, ensure_ascii=False)
        for values in channel_values
    ]
    for secret in (
        attachment_body,
        memory_marker,
        last_document_marker,
        profile_marker,
        history_marker,
    ):
        assert all(secret not in payload for payload in serialized_values)

    planner_states = [
        values
        for values in channel_values
        if isinstance(values.get("plan"), dict)
        and values["plan"].get("planner_echo")
    ]
    assert planner_states
    assert all(
        attachment_body not in json.dumps(values["plan"], ensure_ascii=False)
        and attachment_body not in json.dumps(
            values.get("context_analysis", {}),
            ensure_ascii=False,
        )
        for values in planner_states
    )
    assert any(
        "[附件正文见受控引用]" in json.dumps(values, ensure_ascii=False)
        for values in planner_states
    )
    assert any(
        any(
            attachment.get("file_id") == "snapshot-file"
            and attachment.get("artifact_id")
            for attachment in values.get("attachments", [])
        )
        for values in channel_values
    )


def test_rebuilt_runtime_reuses_first_document_snapshot_and_node_inputs(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    from graph_artifacts import GraphArtifactStore

    run_id = "native-runtime-snapshot-run"
    attachment_body = "ATTACHMENT_FOR_RUNTIME_SNAPSHOT"
    uploads = _NativeSnapshotUploads(attachment_body)
    artifact_store = GraphArtifactStore(tmp_path / "graph_runs", uploads)
    memory = _NativeSnapshotMemory(
        memory_context="FIRST_MEMORY_CONTEXT",
        profile_marker="FIRST_PROFILE",
        last_document="FIRST_LAST_DOCUMENT",
        history_marker="FIRST_HISTORY",
    )
    node_inputs = []
    first_runtime = _build_native_snapshot_runtime(
        memory=memory,
        upload_manager=uploads,
        artifact_store=artifact_store,
        checkpointer=InMemorySaver(),
        run_id=run_id,
        node_inputs=node_inputs,
    )
    payload = {
        "message": "请起草一份文件",
        "session_id": "snapshot-session",
        "file_ids": ["snapshot-file"],
    }

    first_events = parse_sse(first_runtime.stream(
        payload,
        user_id="snapshot-user",
        user_info=None,
    ))
    assert first_events[-1]["type"] == "done"
    first_snapshot = artifact_store.load_document_runtime_context(
        run_id=run_id,
        user_id="snapshot-user",
    )
    assert first_snapshot["memory_context"] == "FIRST_MEMORY_CONTEXT"
    assert first_snapshot["user_profile"]["name"] == "FIRST_PROFILE"
    assert first_snapshot["last_document"] == "FIRST_LAST_DOCUMENT"
    # The immutable runtime packet also captures the current user turn. Keep
    # the assertion focused on the pre-existing history that could otherwise
    # drift between workers.
    assert first_snapshot["conversation_history"][0] == {
        "role": "user",
        "content": "FIRST_HISTORY",
    }

    memory.long_term_context = "SECOND_MEMORY_CONTEXT"
    memory.profile = SimpleNamespace(
        preferred_font="SECOND_PROFILE",
        preferred_size="五号",
        writing_style="第二份风格",
        common_doc_types=[],
        name="SECOND_PROFILE",
        department="第二部门",
    )
    memory.history = [{"role": "user", "content": "SECOND_HISTORY"}]
    memory.context.update({
        ("snapshot-session", "last_request"): "SECOND_PREVIOUS_REQUEST",
        ("snapshot-session", "last_document"): "SECOND_LAST_DOCUMENT",
        ("snapshot-session", "last_plan"): {"marker": "SECOND_PLAN"},
    })

    rebuilt_runtime = _build_native_snapshot_runtime(
        memory=memory,
        upload_manager=uploads,
        artifact_store=artifact_store,
        # A fresh saver simulates a process-local runtime rebuild while the
        # durable run artifact directory remains the source of runtime context.
        checkpointer=InMemorySaver(),
        run_id=run_id,
        node_inputs=node_inputs,
    )
    refs = artifact_store.snapshot_attachments(
        run_id=run_id,
        user_id="snapshot-user",
        file_ids=["snapshot-file"],
    )
    rebuilt_context = rebuilt_runtime._new_run_context(
        user_info=None,
        execution_token="rebuilt-runtime",
        lease=None,
        recovery=True,
    )
    rebuilt_execution = rebuilt_context.get_document_execution({
        "run_id": run_id,
        "session_id": "snapshot-session",
        "user_id": "snapshot-user",
        "message": payload["message"],
        "request_message": payload["message"],
        "working_message": payload["message"],
        "display_message": payload["message"],
        "attachments": refs,
        "parent_step_index": 0,
    }, prepare=True)
    assert rebuilt_execution.runtime_snapshot == first_snapshot

    rebuilt_events = parse_sse(rebuilt_runtime.stream(
        payload,
        user_id="snapshot-user",
        user_info=None,
    ))
    assert rebuilt_events[-1]["type"] == "done"
    rebuilt_node_input = node_inputs[-1]
    assert rebuilt_node_input["memory_context"] == "FIRST_MEMORY_CONTEXT"
    assert rebuilt_node_input["user_profile"]["name"] == "FIRST_PROFILE"
    assert rebuilt_node_input["last_document"] == "FIRST_LAST_DOCUMENT"
    assert rebuilt_node_input["last_plan"] == {"marker": "FIRST_PLAN"}
    assert (
        rebuilt_node_input["conversation_history"]
        == first_snapshot["conversation_history"]
    )


def test_graph_exception_emits_one_error_and_no_done():
    class FakeTaskPlanner:
        def plan(self, **_kwargs):
            return TaskPlan(task_type="knowledge_qa", steps=[])

    class FailingToolOrchestrator:
        def stream(self, prepared, task_plan):
            yield sse({"type": "start"})
            raise RuntimeError("boom")

    runtime = ChatGraphRuntime(ChatRuntimeDependencies(
        memory=FakeMemory(),
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda text, requested="auto": "",
        lightweight_stream=lambda *args: iter(()),
        document_format_stream=lambda *args: iter(()),
        document_draft_stream=lambda *args: iter(()),
        rag_qa_stream=lambda *args: iter(()),
        task_planner=FakeTaskPlanner(),
        tool_orchestrator=FailingToolOrchestrator(),
    ), runtime_mode="graph")

    events = parse_sse(runtime.stream({"message": "hello"}, user_id="user_1", user_info=None))

    assert [event["type"] for event in events] == ["start", "error"]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("graph", "graph"),
        ("langgraph", "graph"),
        ("on", "graph"),
        ("true", "graph"),
        ("1", "graph"),
        ("planner", "planner"),
        ("task_planner", "planner"),
        ("off", "planner"),
        ("false", "planner"),
        ("0", "planner"),
        ("legacy", "legacy"),
        ("pipeline", "legacy"),
    ],
)
def test_runtime_mode_matrix(configured, expected):
    assert ChatGraphRuntime.resolve_runtime_mode(configured) == expected


def test_runtime_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported CHAT_RUNTIME"):
        ChatGraphRuntime.resolve_runtime_mode("mystery")


def test_workflow_recovery_accepts_previous_release_only():
    class FakeTaskPlanner:
        def plan(self, **_kwargs):
            return TaskPlan(task_type="knowledge_qa", steps=[])

    class FakeToolOrchestrator:
        def stream(self, prepared, task_plan):
            yield sse({"type": "done", "answer": "ok"})

    runtime = ChatGraphRuntime(ChatRuntimeDependencies(
        memory=FakeMemory(),
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda *_args: "",
        lightweight_stream=lambda *_args: iter(()),
        document_format_stream=lambda *_args: iter(()),
        document_draft_stream=lambda *_args: iter(()),
        rag_qa_stream=lambda *_args: iter(()),
        task_planner=FakeTaskPlanner(),
        tool_orchestrator=FakeToolOrchestrator(),
    ), runtime_mode="graph")

    runtime._assert_workflow_compatible(SimpleNamespace(workflow_version="chat-v3"))
    runtime._assert_workflow_compatible(SimpleNamespace(workflow_version="chat-v2"))
    with pytest.raises(ChatRunConflictError, match="不支持安全恢复"):
        runtime._assert_workflow_compatible(
            SimpleNamespace(workflow_version="chat-v1")
        )


def test_chat_graph_runtime_legacy_env_bypasses_task_planner(monkeypatch):
    monkeypatch.setenv("CHAT_RUNTIME", "legacy")
    calls = []

    class FailingTaskPlanner:
        def plan(self, **kwargs):
            raise AssertionError("task planner should be bypassed")

    runtime = build_runtime(calls, runtime_mode=None)
    runtime.deps.task_planner = FailingTaskPlanner()
    runtime.deps.tool_orchestrator = object()

    output = list(runtime.stream(
        {"message": "制度文件在哪里查？"},
        user_id="user_1",
        user_info=SimpleNamespace(username="tester"),
    ))

    assert output == ["rag"]
