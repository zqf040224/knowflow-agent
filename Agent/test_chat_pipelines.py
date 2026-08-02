from types import SimpleNamespace

from chat_architecture import (
    INTENT_DOC_DRAFTING,
    INTENT_FORM_TEMPLATE_EXPORT,
    INTENT_IDENTITY_HELP,
    INTENT_KNOWLEDGE_QA,
    RouteResult,
)
from chat_pipelines import ChatPipelineDispatcher, ChatPipelineRequest, ChatPipelineRuntime


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


def make_runtime(memory=None):
    return ChatPipelineRuntime(
        memory=memory or FakeMemory(),
        reimbursement_template_files={
            "travel": "差旅费.xlsx",
            "meeting": "会议费.xlsx",
            "labor_expert": "劳务费&专家咨询费.xlsx",
            "other": "其他费用报销.xlsx",
        },
        assistant_identity_response=lambda: "我是智能知识库助手。",
    )


def make_dispatcher(calls, memory=None):
    def handler(name):
        def _stream(*args):
            calls.append((name, args))
            yield name
        return _stream

    return ChatPipelineDispatcher(
        runtime=make_runtime(memory),
        document_format_stream=handler("format"),
        document_draft_stream=handler("draft"),
        rag_qa_stream=handler("rag"),
    )


def request(intent):
    return ChatPipelineRequest(
        message="内部消息",
        display_message="显示消息",
        session_id="s1",
        user_id="u1",
        user_info=SimpleNamespace(username="tester"),
        user_metadata={"attached_files": []},
        route=SimpleNamespace(intent=intent),
    )


def parse_sse_events(chunks):
    events = []
    for chunk in chunks:
        if not isinstance(chunk, str) or not chunk.startswith("data: "):
            continue
        import json
        events.append(json.loads(chunk[6:]))
    return events


def test_identity_help_done_contract_does_not_create_document_state():
    calls = []
    memory = FakeMemory()
    output = list(make_dispatcher(calls, memory).stream(request(INTENT_IDENTITY_HELP)))
    events = parse_sse_events(output)
    done = events[-1]
    assert calls == []
    assert done["type"] == "done"
    assert done["intent"] == INTENT_IDENTITY_HELP
    assert done["document"] == ""
    assert done["export_template"] == ""
    assert done["actions"] == []
    assert memory.context[("s1", "last_answer")] == "我是智能知识库助手。"
    assert ("s1", "last_document") not in memory.context


def test_dispatcher_routes_form_export():
    calls = []
    memory = FakeMemory()
    req = request(INTENT_FORM_TEMPLATE_EXPORT)
    req.route = RouteResult(
        intent=INTENT_FORM_TEMPLATE_EXPORT,
        confidence=0.96,
        reason="用户明确要求导出报销类表单模板",
        document_type="报销表单",
        template_key="meeting",
        requires_retrieval=False,
        actions=[{"type": "export_xlsx_template", "label": "导出会议费报销表", "template_key": "meeting"}],
    )
    output = list(make_dispatcher(calls, memory).stream(req))
    events = parse_sse_events(output)
    done = events[-1]
    assert calls == []
    assert done["intent"] == INTENT_FORM_TEMPLATE_EXPORT
    assert done["document"] == ""
    assert done["export_spreadsheet_template"] == ""
    assert done["actions"] == [{"type": "export_xlsx_template", "label": "导出会议费报销表", "template_key": "meeting"}]
    assert done["source_details"] == [{"filename": "会议费.xlsx"}]
    assert memory.context[("s1", "last_answer")]
    assert ("s1", "last_document") not in memory.context


def test_dispatcher_routes_document_drafting_with_full_context():
    calls = []
    output = list(make_dispatcher(calls).stream(request(INTENT_DOC_DRAFTING)))
    assert output == ["draft"]
    assert calls[0][0] == "draft"
    assert calls[0][1][0] == "内部消息"
    assert calls[0][1][2] == "u1"


def test_dispatcher_unknown_intent_falls_back_to_rag():
    calls = []
    output = list(make_dispatcher(calls).stream(request("unknown_intent")))
    assert output == ["rag"]
    assert calls[0][0] == "rag"


def test_dispatcher_routes_knowledge_qa():
    calls = []
    output = list(make_dispatcher(calls).stream(request(INTENT_KNOWLEDGE_QA)))
    assert output == ["rag"]
    assert calls[0][0] == "rag"
