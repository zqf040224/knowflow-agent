import json

from chat_events import (
    CHAT_EVENT_TYPES,
    answer_delta_event,
    answer_done_event,
    answer_start_event,
    content_event,
    done_event,
    error_event,
    normalize_event,
    parse_sse_events,
    run_done_event,
    sse,
    think_event,
    thinking_done_event,
    thinking_start_event,
)


def test_done_event_includes_stable_frontend_contract_fields():
    payload = done_event(
        intent="knowledge_qa",
        answer="答复",
        session_id="session_1",
        plan={"task_type": "问答检索"},
    )

    assert payload == {
        "type": "done",
        "intent": "knowledge_qa",
        "answer": "答复",
        "document": "",
        "session_id": "session_1",
        "plan": {"task_type": "问答检索"},
        "route": {},
        "actions": [],
        "export_template": "",
        "export_spreadsheet_template": "",
        "source_filenames": [],
        "source_details": [],
        "audit_summary": {},
    }


def test_sse_normalizes_existing_pipeline_payloads():
    raw = sse({"type": "done", "answer": "只填了 answer"})
    events = parse_sse_events([raw])

    assert events[0]["type"] == "done"
    assert events[0]["answer"] == "只填了 answer"
    assert events[0]["document"] == ""
    assert events[0]["actions"] == []
    assert events[0]["audit_summary"] == {}


def test_event_factory_normalizes_common_stream_events():
    assert content_event("abc") == {"type": "content", "data": "abc"}
    assert answer_start_event(session_id="s1") == {
        "type": "answer_start",
        "message": "开始输出正文",
        "session_id": "s1",
    }
    assert answer_delta_event("abc", session_id="s1") == {
        "type": "answer_delta",
        "data": "abc",
        "session_id": "s1",
    }
    assert answer_done_event("完整回答", session_id="s1") == {
        "type": "answer_done",
        "answer": "完整回答",
        "session_id": "s1",
    }
    assert thinking_start_event() == {"type": "thinking_start", "message": "开始思考"}
    assert thinking_done_event(step_count=3, elapsed_ms=1200) == {
        "type": "thinking_done",
        "summary": "思考完成",
        "step_count": 3,
        "elapsed_ms": 1200,
    }
    assert run_done_event(session_id="s1") == {"type": "run_done", "session_id": "s1"}
    assert think_event("Planner", "", "已制定计划") == {
        "type": "think",
        "agent": "Planner",
        "emoji": "",
        "message": "已制定计划",
    }
    assert error_event("") == {"type": "error", "message": ""}
    assert normalize_event({}) == {"type": "error", "message": "请求处理失败，请稍后重试"}


def test_parse_sse_events_ignores_non_data_chunks():
    chunks = [
        "event: ping\n\n",
        "data: " + json.dumps({"type": "start"}) + "\n\n",
        "plain text",
    ]

    assert parse_sse_events(chunks) == [{"type": "start"}]


def test_chat_event_type_registry_matches_public_contract():
    assert {
        "start",
        "session",
        "route",
        "thinking_start",
        "thinking_done",
        "think",
        "plan",
        "tool_plan",
        "tool_call",
        "tool_result",
        "tool_confirm_required",
        "write_start",
        "answer_start",
        "answer_delta",
        "answer_done",
        "content",
        "reasoning_chunk",
        "reflection",
        "done",
        "run_done",
        "error",
    }.issubset(CHAT_EVENT_TYPES)
