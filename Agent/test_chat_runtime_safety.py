from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from chat_architecture import RouteResult
from chat_draft import DocumentDraftDependencies, DocumentDraftStreamService
from chat_events import parse_sse_events
from chat_runtime import (
    ChatExecutionLease,
    ChatGraphRuntime,
    ChatRunCancelledError,
    ChatRunConflictError,
    ChatRuntimeDependencies,
)
from task_planner import TaskPlan, TaskStep
from test_chat_runtime import FakeMemory, FakeUploadManager


def _deps(tool_orchestrator):
    return ChatRuntimeDependencies(
        memory=FakeMemory(),
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda *_args: "",
        lightweight_stream=lambda *_args: iter(()),
        document_format_stream=lambda *_args: iter(()),
        document_draft_stream=lambda *_args: iter(()),
        rag_qa_stream=lambda *_args: iter(()),
        task_planner=object(),
        tool_orchestrator=tool_orchestrator,
    )


def _tool_state():
    plan = TaskPlan(
        task_type="knowledge_qa",
        steps=[TaskStep(tool="knowledge_qa", reason="test")],
        route=RouteResult("knowledge_qa", 1.0, "test"),
    )
    return {
        "run_id": "run-1",
        "request_id": "request-1",
        "user_id": "user-1",
        "session_id": "session-1",
        "request_message": "question",
        "message": "question",
        "working_message": "question",
        "display_message": "question",
        "mode": "chat",
        "attachments": [],
        "task_plan": plan.to_dict(),
        "step_index": 0,
    }


class _ToolOrchestrator:
    def __init__(self, events):
        self.events = events
        self.registry = SimpleNamespace(get=lambda _name: object())

    def _route_for_step(self, _step, route):
        return route

    def _stream_tool_events(self, _tool, _prepared, _route):
        yield from self.events

    def _missing_tool_done(self, _prepared, _step, _route):
        return {"type": "done", "answer": "fallback"}


def _runtime_context(writer):
    return SimpleNamespace(
        context=SimpleNamespace(
            user_info=None,
            execution_token="token",
            lease=None,
            recovery=False,
        ),
        stream_writer=writer,
    )


class _CommittedDeletionBarrier:
    def is_session_deletion_requested(self, session_id, *, user_id):
        assert (session_id, user_id) == ("session-1", "user-1")
        return True

    def is_thread_deletion_requested(self, *_args, **_kwargs):
        pytest.fail("the committed session barrier must win first")


class _OrdinaryNodeFailureGraph:
    def stream(self, *_args, **_kwargs):
        raise RuntimeError("ordinary node failure")


def _assert_single_deletion_cancellation(chunks):
    assert parse_sse_events(chunks) == [{
        "type": "error",
        "message": "会话已删除，当前运行已取消",
    }]


def test_active_run_stops_on_session_barrier_before_thread_tombstone_exists():
    calls = []

    class Persistence:
        def is_session_deletion_requested(self, session_id, *, user_id):
            calls.append(("session", session_id, user_id))
            return True

        def is_thread_deletion_requested(self, *_args, **_kwargs):
            pytest.fail("session barrier must stop the run before thread lookup")

    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime.deps.persistence = Persistence()

    with pytest.raises(ChatRunCancelledError, match="session deletion"):
        runtime._assert_run_active(_tool_state())

    assert calls == [("session", "session-1", "user-1")]


def test_stream_graph_node_failure_yields_deletion_cancellation_once():
    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime.deps.persistence = _CommittedDeletionBarrier()
    runtime._graph = _OrdinaryNodeFailureGraph()
    cleaned = []
    runtime._cleanup_cancelled_run = lambda state: cleaned.append(state["run_id"])
    runtime._fail_active_tool_effect_best_effort = (
        lambda *_args, **_kwargs: pytest.fail("cancelled run must not fail effects")
    )
    runtime._fail_persisted_run_best_effort = (
        lambda *_args, **_kwargs: pytest.fail("cancelled run must not become failed")
    )

    chunks = list(runtime._stream_graph(
        _tool_state(),
        SimpleNamespace(execution_token="token", lease=None),
    ))

    _assert_single_deletion_cancellation(chunks)
    assert cleaned == ["run-1"]


def test_checkpoint_node_failure_yields_deletion_cancellation_once():
    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime.deps.persistence = _CommittedDeletionBarrier()
    runtime._graph = _OrdinaryNodeFailureGraph()
    runtime._mark_current_node = lambda *_args, **_kwargs: None
    cleaned = []
    runtime._cleanup_cancelled_run = lambda state: cleaned.append(state["run_id"])
    runtime._fail_active_tool_effect_best_effort = (
        lambda *_args, **_kwargs: pytest.fail("cancelled run must not fail effects")
    )
    runtime._fail_persisted_run_best_effort = (
        lambda *_args, **_kwargs: pytest.fail("cancelled run must not become failed")
    )

    chunks = list(runtime._stream_checkpoint_continuation(
        _tool_state(),
        SimpleNamespace(execution_token="token", lease=None),
        object(),
        {"configurable": {"thread_id": "run-1"}},
        current_node="resume",
        durability="sync",
        incomplete_message="incomplete",
        failure_message="ordinary failure",
    ))

    _assert_single_deletion_cancellation(chunks)
    assert cleaned == ["run-1"]


def test_terminal_recovery_failure_yields_deletion_cancellation_once():
    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime.deps.persistence = _CommittedDeletionBarrier()
    runtime._mark_current_node = lambda *_args, **_kwargs: None
    runtime._finish_persisted_run = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("ordinary terminal commit failure")
        )
    )
    runtime._fail_persisted_run_best_effort = (
        lambda *_args, **_kwargs: pytest.fail("cancelled run must not become failed")
    )
    state = {**_tool_state(), "final_event": {"type": "done", "answer": "ok"}}
    cleaned = []
    runtime._cleanup_cancelled_run = lambda value: cleaned.append(value["run_id"])

    chunks = list(runtime._stream_terminal_recovery(
        state,
        SimpleNamespace(execution_token="token", lease=None),
        checkpoint_id="checkpoint-1",
    ))

    _assert_single_deletion_cancellation(chunks)
    assert cleaned == ["run-1"]


def test_heartbeat_rejection_after_barrier_race_maps_to_cancellation():
    class Persistence:
        def __init__(self):
            self.barrier_checks = 0

        def is_session_deletion_requested(self, _session_id, *, user_id):
            assert user_id == "user-1"
            self.barrier_checks += 1
            return self.barrier_checks > 1

        def is_thread_deletion_requested(self, *_args, **_kwargs):
            return False

        def heartbeat_run(self, *_args, **_kwargs):
            return None

        def get_owned_run(self, *_args, **_kwargs):
            return SimpleNamespace(session_id="session-1")

    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime.deps.persistence = Persistence()

    with pytest.raises(ChatRunCancelledError, match="session deletion"):
        runtime._mark_current_node(
            _tool_state(),
            "execute_tools",
            execution_token="worker-1",
        )


def test_background_heartbeat_barrier_loss_preserves_cancellation_semantics():
    class Persistence:
        def heartbeat_run(self, *_args, **_kwargs):
            return None

        def is_session_deletion_requested(self, session_id, *, user_id):
            assert (session_id, user_id) == ("session-1", "user-1")
            return True

        def is_thread_deletion_requested(self, *_args, **_kwargs):
            pytest.fail("session barrier must win before thread lookup")

    class ImmediateTick:
        def wait(self, _timeout):
            return False

    persistence = Persistence()
    lease = ChatExecutionLease(
        persistence,
        run_id="run-1",
        user_id="user-1",
        execution_token="worker-1",
    )
    lease._stop = ImmediateTick()
    lease._run()
    with pytest.raises(ChatRunConflictError, match="lease was lost"):
        lease.assert_owned()

    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime.deps.persistence = persistence
    with pytest.raises(ChatRunCancelledError, match="session deletion"):
        runtime._mark_current_node(
            _tool_state(),
            "execute_tools",
            execution_token="worker-1",
            lease=lease,
        )


def test_effect_and_finish_barrier_rejections_map_to_cancellation():
    class Persistence:
        def is_session_deletion_requested(self, session_id, *, user_id):
            assert (session_id, user_id) == ("session-1", "user-1")
            return True

        def is_thread_deletion_requested(self, *_args, **_kwargs):
            return False

        def claim_effect(self, *_args, **_kwargs):
            return False

        def complete_effect(self, *_args, **_kwargs):
            return None

        def finish_run(self, *_args, **_kwargs):
            return None

        def get_owned_run(self, *_args, **_kwargs):
            return SimpleNamespace(session_id="session-1")

    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime.deps.persistence = Persistence()
    state = _tool_state()

    with pytest.raises(ChatRunCancelledError, match="session deletion"):
        runtime._claim_effect(
            state,
            node="tool_knowledge_qa",
            effect="step:1",
            execution_token="worker-1",
        )
    with pytest.raises(ChatRunCancelledError, match="session deletion"):
        runtime._complete_effect(
            state,
            node="tool_knowledge_qa",
            effect="step:1",
            result={"ok": True},
            execution_token="worker-1",
        )
    with pytest.raises(ChatRunCancelledError, match="session deletion"):
        runtime._finish_persisted_run(
            state["run_id"],
            user_id=state["user_id"],
            execution_token="worker-1",
            status="succeeded",
        )


def test_tool_success_events_are_released_only_after_effect_receipt():
    order = []
    orchestrator = _ToolOrchestrator([
        {"type": "answer_start"},
        {"type": "answer_delta", "data": "ok"},
        {"type": "answer_done"},
        {"type": "done", "answer": "ok"},
    ])
    runtime = ChatGraphRuntime(_deps(orchestrator), runtime_mode="planner")
    runtime._claim_effect = lambda *_args, **_kwargs: (True, None)
    runtime._complete_effect = lambda *_args, **_kwargs: order.append("complete")
    runtime._fail_effect = lambda *_args, **_kwargs: order.append("failed")

    runtime._execute_single_tool_node(
        _tool_state(),
        _runtime_context(lambda chunk: order.append(chunk)),
        expected_tool="knowledge_qa",
    )

    assert order[0] == "complete"
    assert [
        event["type"]
        for chunk in order[1:]
        for event in parse_sse_events([chunk])
    ] == ["answer_start", "answer_delta", "answer_done"]


def test_tool_failure_discards_buffered_success_events():
    written = []
    orchestrator = _ToolOrchestrator([
        {"type": "answer_start"},
        {"type": "answer_delta", "data": "partial"},
        {"type": "error", "message": "failed"},
    ])
    runtime = ChatGraphRuntime(_deps(orchestrator), runtime_mode="planner")
    runtime._claim_effect = lambda *_args, **_kwargs: (True, None)
    runtime._complete_effect = lambda *_args, **_kwargs: pytest.fail(
        "failed effects must not complete"
    )
    runtime._fail_effect = lambda *_args, **_kwargs: None

    with pytest.raises(RuntimeError, match="failed"):
        runtime._execute_single_tool_node(
            _tool_state(),
            _runtime_context(written.append),
            expected_tool="knowledge_qa",
        )

    assert [
        event["type"]
        for chunk in written
        for event in parse_sse_events([chunk])
    ] == ["error"]


def test_heartbeat_exception_fails_closed_after_bounded_attempts():
    class BrokenPersistence:
        def __init__(self):
            self.calls = 0

        def heartbeat_run(self, *_args, **_kwargs):
            self.calls += 1
            raise ConnectionError("network unavailable")

    class ImmediateTick:
        def wait(self, _timeout):
            return False

        def set(self):
            return None

    persistence = BrokenPersistence()
    lease = ChatExecutionLease(
        persistence,
        run_id="run-1",
        user_id="user-1",
        execution_token="token",
        max_consecutive_failures=1,
    )
    lease._stop = ImmediateTick()
    lease._run()

    assert persistence.calls == 1
    with pytest.raises(ChatRunConflictError, match="ownership can no longer"):
        lease.assert_owned()


def test_native_document_execution_receives_current_step_effect_scope():
    captured = {}
    orchestrator = SimpleNamespace(
        think_log=[],
        _document_runtime_snapshot={},
        _current_agent_memory_context="",
    )
    execution = SimpleNamespace(
        orchestrator=orchestrator,
        attachment_bodies=[],
        runtime_snapshot={},
        effect_scope="",
    )

    class Service:
        def create_native_execution(self, **kwargs):
            captured.update(kwargs)
            execution.effect_scope = kwargs["effect_scope"]
            return execution

    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime._native_document_service = lambda: Service()
    state = _tool_state()
    state["step_index"] = 1
    state["display_message"] = ""
    state["message"] = "HYDRATED_ATTACHMENT_SECRET"
    state["working_message"] = "HYDRATED_ATTACHMENT_SECRET"

    result = runtime._create_native_document_execution(
        state,
        user_info=None,
        execution_token="token",
        recovery=False,
        prepare=True,
    )

    assert captured["effect_scope"] == "step:2"
    assert captured["display_message"] == ""
    assert result.effect_scope == "step:2"


def test_native_document_failures_use_distinct_step_scoped_accounting_keys():
    recorded = []
    service = DocumentDraftStreamService(DocumentDraftDependencies(
        memory=None,
        orchestrator_factory=lambda *_args, **_kwargs: None,
        resolve_export_template=lambda *_args: "default",
        record_agent_run_token_usage=lambda *_args, **_kwargs: None,
        record_token_usage=lambda **kwargs: recorded.append(kwargs),
        record_token_usage_best_effort=lambda **kwargs: recorded.append(kwargs),
    ))
    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime._native_document_service = lambda: service
    runtime._fail_effect = lambda *_args, **_kwargs: None
    state = _tool_state()
    state["run_id"] = "run-two-document-failures"

    for step_index in (0, 1):
        runtime._record_native_document_failure(
            {**state, "step_index": step_index},
            user_info=None,
            step="write",
            error=RuntimeError("writer unavailable"),
            execution_token="token",
        )

    assert [item["effect_key"] for item in recorded] == [
        "run-two-document-failures:tool_draft_document:step:1:token_usage_pipeline_failure",
        "run-two-document-failures:tool_draft_document:step:2:token_usage_pipeline_failure",
    ]


def test_document_sanitizer_covers_review_reflection_history_and_errors():
    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    runtime._document_redaction_rules = lambda _state: [
        ("RUNTIME_SECRET", "[运行时上下文见受控快照]")
    ]
    update = {
        "review_meta": {"suggestions": ["RUNTIME_SECRET"]},
        "reflection_meta": {"weaknesses": ["RUNTIME_SECRET"]},
        "revision_history": [{"comment": "RUNTIME_SECRET"}],
        "errors": [{"message": "RUNTIME_SECRET"}],
        "error_message": "Reviewer echoed RUNTIME_SECRET",
        "document_content": "intended document RUNTIME_SECRET",
    }

    sanitized = runtime._sanitize_native_document_update({}, update)

    for field_name in (
        "review_meta",
        "reflection_meta",
        "revision_history",
        "errors",
        "error_message",
    ):
        assert "RUNTIME_SECRET" not in str(sanitized[field_name])
    assert sanitized["document_content"] == update["document_content"]


def test_document_redaction_rules_cover_partial_runtime_snapshot_echoes():
    runtime = ChatGraphRuntime(_deps(object()), runtime_mode="planner")
    first = "PREVIOUS_CONTEXT_SECRET_SENTENCE_ONE。"
    second = "PREVIOUS_CONTEXT_SECRET_SENTENCE_TWO！"

    rules = dict(runtime._document_redaction_rules({
        "_runtime_document_snapshot": {
            "previous_context": first + second,
        },
    }))

    assert first in rules
    assert second in rules
    assert rules[first] == "[运行时上下文见受控快照]"


def test_planner_and_legacy_import_without_langgraph_but_graph_fails_closed():
    code = r'''
import importlib.abc
import sys

class BlockLangGraph(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "langgraph" or fullname.startswith("langgraph."):
            raise ModuleNotFoundError("blocked langgraph for rollback test")
        return None

sys.meta_path.insert(0, BlockLangGraph())
from chat_runtime import ChatGraphRuntime, ChatRuntimeDependencies

class Memory:
    def get_or_create_session(self, _user_id, session_id=None):
        return session_id or "session"

deps = ChatRuntimeDependencies(
    memory=Memory(), upload_manager=object(),
    reimbursement_detector=lambda *_args: "",
    lightweight_stream=lambda *_args: iter(()),
    document_format_stream=lambda *_args: iter(()),
    document_draft_stream=lambda *_args: iter(()),
    rag_qa_stream=lambda *_args: iter(()),
    task_planner=object(), tool_orchestrator=object(),
)
assert ChatGraphRuntime(deps, runtime_mode="planner").runtime_mode == "planner"
assert ChatGraphRuntime(deps, runtime_mode="legacy").runtime_mode == "legacy"
try:
    ChatGraphRuntime(deps, runtime_mode="graph")
except RuntimeError as exc:
    assert "requires LangGraph" in str(exc)
else:
    raise AssertionError("graph mode silently degraded")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(__file__).rsplit("/", 1)[0],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
