import json
from types import SimpleNamespace

from chat_events import sse
from task_planner import TOOL_DRAFT_DOCUMENT, TOOL_KNOWLEDGE_QA, TaskPlan, TaskStep
from tool_runtime import ChatTool, ToolOrchestrator, ToolRegistry


def parse_sse(chunks):
    events = []
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    return events


def prepared():
    return SimpleNamespace(
        message="问题",
        display_message="问题",
        session_id="session_1",
        user_id="user_1",
        user_info=None,
        user_metadata=None,
    )


def stream_for(answer, intent="knowledge_qa", actions=None):
    actions = actions or []

    def _stream(*_args):
        yield sse({"type": "start"})
        yield sse({"type": "session", "session_id": "session_1"})
        yield sse({"type": "content", "data": answer, "session_id": "session_1"})
        yield sse({
            "type": "done",
            "intent": intent,
            "answer": answer,
            "document": answer if intent == "doc_drafting" else "",
            "session_id": "session_1",
            "plan": {"task_type": "测试"},
            "actions": actions,
        })

    return _stream


def test_tool_registry_registers_and_lists_tools():
    registry = ToolRegistry()
    registry.register(ChatTool(
        name=TOOL_KNOWLEDGE_QA,
        description="知识问答",
        risk_level="low",
        input_schema={"message": "string"},
        stream=stream_for("回答"),
    ))

    assert registry.get(TOOL_KNOWLEDGE_QA).description == "知识问答"
    assert registry.list_public()[0]["name"] == TOOL_KNOWLEDGE_QA


def test_tool_orchestrator_emits_plan_call_result_and_final_done():
    registry = ToolRegistry()
    registry.register(ChatTool(
        name=TOOL_KNOWLEDGE_QA,
        description="知识问答",
        risk_level="low",
        input_schema={},
        stream=stream_for("依据回答"),
    ))
    plan = TaskPlan(task_type="问答", steps=[TaskStep(tool=TOOL_KNOWLEDGE_QA, reason="查资料")])

    events = parse_sse(ToolOrchestrator(registry).stream(prepared(), plan))
    event_types = [event["type"] for event in events]

    assert event_types[:3] == ["start", "session", "thinking_start"]
    assert "tool_plan" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "content" in event_types
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "依据回答"
    assert events[-1]["plan"]["task_planner"]["task_type"] == "问答"


def test_tool_orchestrator_confirm_required_for_action_tools():
    registry = ToolRegistry()
    registry.register(ChatTool(
        name=TOOL_KNOWLEDGE_QA,
        description="知识问答",
        risk_level="confirm",
        input_schema={},
        stream=stream_for("请确认", actions=[{"type": "export_xlsx_template"}]),
    ))
    plan = TaskPlan(
        task_type="导出",
        steps=[TaskStep(tool=TOOL_KNOWLEDGE_QA, reason="准备动作", requires_confirmation=True)],
        requires_confirmation=True,
    )

    events = parse_sse(ToolOrchestrator(registry).stream(prepared(), plan))

    confirm = [event for event in events if event["type"] == "tool_confirm_required"][0]
    assert confirm["data"]["actions"] == [{"type": "export_xlsx_template"}]
    assert events[-1]["actions"] == [{"type": "export_xlsx_template"}]


def test_tool_orchestrator_uses_last_tool_done_for_multi_step_task():
    registry = ToolRegistry()
    registry.register(ChatTool(
        name=TOOL_KNOWLEDGE_QA,
        description="知识问答",
        risk_level="low",
        input_schema={},
        stream=stream_for("查到依据", intent="knowledge_qa"),
    ))
    registry.register(ChatTool(
        name=TOOL_DRAFT_DOCUMENT,
        description="写作",
        risk_level="low",
        input_schema={},
        stream=stream_for("最终文档", intent="doc_drafting"),
    ))
    plan = TaskPlan(task_type="复合", steps=[
        TaskStep(tool=TOOL_KNOWLEDGE_QA, reason="先查"),
        TaskStep(tool=TOOL_DRAFT_DOCUMENT, reason="再写"),
    ])

    events = parse_sse(ToolOrchestrator(registry).stream(prepared(), plan))

    assert [event["data"]["tool"] for event in events if event["type"] == "tool_call"] == [
        TOOL_KNOWLEDGE_QA,
        TOOL_DRAFT_DOCUMENT,
    ]
    assert events[-1]["intent"] == "doc_drafting"
    assert events[-1]["document"] == "最终文档"
