import json
from types import SimpleNamespace

from chat_architecture import INTENT_CLARIFY, INTENT_FORM_TEMPLATE_EXPORT, INTENT_SPREADSHEET_TRANSFORM, RouteResult
from chat_lightweight import LightweightChatDependencies, LightweightChatStreamService


class FakeMemory:
    def __init__(self):
        self.messages = []
        self.context = {}
        self.summaries = []

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


def parse_sse(chunks):
    events = []
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    return events


def service(memory=None):
    return LightweightChatStreamService(LightweightChatDependencies(
        memory=memory or FakeMemory(),
        reimbursement_template_files={"meeting": "会议费.xlsx"},
        assistant_identity_response=lambda: "我是智能知识库助手。",
    ))


def test_lightweight_clarify_contract():
    route = RouteResult(
        intent=INTENT_CLARIFY,
        confidence=0.76,
        reason="缺少材料",
        document_type="格式转换",
        requires_retrieval=False,
    )

    events = parse_sse(service().stream("改为公文格式", "s1", route=route))

    assert events[-1]["type"] == "done"
    assert events[-1]["intent"] == INTENT_CLARIFY
    assert events[-1]["document"] == ""
    assert "请先上传" in events[-1]["answer"]


def test_lightweight_form_export_contract():
    memory = FakeMemory()
    route = RouteResult(
        intent=INTENT_FORM_TEMPLATE_EXPORT,
        confidence=0.96,
        reason="导出会议费报销表",
        document_type="报销表单",
        template_key="meeting",
        requires_retrieval=False,
        actions=[{"type": "export_xlsx_template", "label": "导出会议费报销表", "template_key": "meeting"}],
    )

    events = parse_sse(service(memory).stream("导出会议费报销表", "s1", route=route))

    assert events[-1]["intent"] == INTENT_FORM_TEMPLATE_EXPORT
    assert events[-1]["source_details"] == [{"filename": "会议费.xlsx"}]
    assert memory.context[("s1", "last_answer")]


def test_lightweight_spreadsheet_transform_contract():
    route = SimpleNamespace(to_dict=lambda: {
        "intent": INTENT_SPREADSHEET_TRANSFORM,
        "actions": [{"filename": "预算.xlsx", "type": "spreadsheet_transform"}],
    })

    events = parse_sse(service().stream("筛选预算表", "s1", route=route))

    assert events[-1]["intent"] == INTENT_SPREADSHEET_TRANSFORM
    assert events[-1]["actions"] == [{"filename": "预算.xlsx", "type": "spreadsheet_transform"}]
    assert "预算.xlsx" in events[-1]["answer"]


def test_lightweight_long_response_streams_multiple_answer_deltas():
    long_response = "我是智能知识库助手，可以检索资料、整理依据、起草公文、处理上传材料，并在多轮对话中延续上下文。"
    stream = LightweightChatStreamService(LightweightChatDependencies(
        memory=FakeMemory(),
        reimbursement_template_files={"meeting": "会议费.xlsx"},
        assistant_identity_response=lambda: long_response,
    ))
    route = RouteResult(
        intent="identity_help",
        confidence=0.99,
        reason="身份说明",
        document_type="身份说明",
        requires_retrieval=False,
    )

    events = parse_sse(stream.stream("你是谁？", "s1", route=route))
    deltas = [event["data"] for event in events if event["type"] == "answer_delta"]

    assert len(deltas) > 1
    assert "".join(deltas) == events[-1]["answer"]


def test_lightweight_message_effect_keys_follow_selected_tool_node():
    memory = FakeMemory()
    route = RouteResult(
        intent=INTENT_CLARIFY,
        confidence=0.8,
        reason="需要补充信息",
        requires_retrieval=False,
    )

    parse_sse(service(memory).stream(
        "请帮我处理",
        "s1",
        user_metadata={"run_id": "run-light", "request_id": "request-1"},
        route=route,
    ))

    assert memory.messages[0][3]["effect_key"] == "run-light:tool_clarify:message_user"
    assert memory.messages[1][3]["effect_key"] == "run-light:tool_clarify:message_assistant"
    assert memory.summaries[0][-1] == "run-light:tool_clarify:rolling_summary"


def test_lightweight_effect_keys_include_optional_step_scope():
    memory = FakeMemory()
    route = RouteResult(
        intent=INTENT_CLARIFY,
        confidence=0.8,
        reason="需要补充信息",
        requires_retrieval=False,
    )

    parse_sse(service(memory).stream(
        "请帮我处理",
        "s1",
        user_metadata={"run_id": "run-light", "effect_scope": "step:2"},
        route=route,
    ))

    assert memory.messages[0][3]["effect_key"] == (
        "run-light:tool_clarify:step:2:message_user"
    )
    assert memory.messages[1][3]["effect_key"] == (
        "run-light:tool_clarify:step:2:message_assistant"
    )
