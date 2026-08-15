from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from chat_architecture import RouteResult
from chat_events import sse
from chat_runtime import (
    ChatGraphRuntime,
    ChatRunConflictError,
    ChatRuntimeDependencies,
)
from task_planner import TaskPlan, TaskStep
from test_chat_runtime import FakeMemory, FakeUploadManager, parse_sse
from tool_runtime import ChatTool, ToolOrchestrator, ToolRegistry


class FakePersistence:
    def __init__(self, *, fail_finish: bool = False):
        self.config = SimpleNamespace(default_durability="async", backend="memory")
        self.checkpointer = InMemorySaver()
        self.fail_finish = fail_finish
        self.records = {}
        self.by_key = {}
        self.status_events = []
        self.claim_calls = []
        self.recovery_claim_calls = []

    def start_run(self, **kwargs):
        key = (kwargs["user_id"], kwargs["graph_name"], kwargs["idempotency_key"] or kwargs["request_id"])
        if key in self.by_key:
            return self.by_key[key]
        values = dict(kwargs)
        values.update({
            "idempotency_key": kwargs["idempotency_key"] or kwargs["request_id"],
            "status": "running",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_id": "",
            "interrupt_id": "",
            "execution_token": kwargs.get("execution_token") or "generated-token",
        })
        record = SimpleNamespace(**values)
        self.records[record.run_id] = record
        self.by_key[key] = record
        return record

    def heartbeat_run(self, run_id, *, user_id, execution_token,
                      current_node, **_kwargs):
        record = self.get_owned_run(run_id, user_id=user_id)
        if (record is None or record.status != "running"
                or record.execution_token != execution_token):
            return None
        record.current_node = current_node
        record.updated_at = datetime.now(timezone.utc).isoformat()
        return record

    def finish_run(self, run_id, *, user_id, execution_token, status,
                   current_node, error="", checkpoint_id=None,
                   interrupt_id=None, **_kwargs):
        if self.fail_finish and status == "succeeded":
            self.fail_finish = False
            raise RuntimeError("ledger unavailable")
        record = self.get_owned_run(run_id, user_id=user_id)
        if (record is None or record.status != "running"
                or record.execution_token != execution_token):
            return None
        return self.update_run(
            run_id,
            status=status,
            error=error,
            current_node=current_node,
            checkpoint_id=checkpoint_id,
            interrupt_id=interrupt_id,
        )

    def update_run(self, run_id, *, status, error="", current_node=None,
                   checkpoint_id=None, interrupt_id=None, **_kwargs):
        if self.fail_finish and status == "succeeded":
            self.fail_finish = False
            raise RuntimeError("ledger unavailable")
        record = self.records.get(run_id)
        if record is None:
            return None
        record.status = status
        record.error = error
        if current_node is not None:
            record.current_node = current_node
        if checkpoint_id is not None:
            record.checkpoint_id = checkpoint_id
        if interrupt_id is not None:
            record.interrupt_id = interrupt_id
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self.status_events.append(status)
        return record

    def get_owned_run(self, run_id, *, user_id):
        record = self.records.get(run_id)
        return record if record is not None and record.user_id == user_id else None

    def get_run_by_idempotency_key(self, user_id, graph_name, idempotency_key):
        return self.by_key.get((user_id, graph_name, idempotency_key))

    def claim_resume(self, run_id, *, user_id, expected_checkpoint_id=None,
                     expected_interrupt_id=None, new_execution_token=None,
                     **_kwargs):
        self.claim_calls.append((run_id, expected_checkpoint_id, expected_interrupt_id))
        record = self.get_owned_run(run_id, user_id=user_id)
        if record is None or record.status != "interrupted":
            return False
        if expected_checkpoint_id and record.checkpoint_id != expected_checkpoint_id:
            return False
        if expected_interrupt_id and record.interrupt_id != expected_interrupt_id:
            return False
        record.status = "running"
        record.execution_token = new_execution_token or "resume-token"
        record.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def claim_recovery(self, run_id, *, user_id, expected_updated_at,
                       new_execution_token):
        self.recovery_claim_calls.append((run_id, expected_updated_at))
        record = self.get_owned_run(run_id, user_id=user_id)
        if (record is None or record.status not in {"failed", "running"}
                or record.updated_at != expected_updated_at):
            return False
        record.status = "running"
        record.execution_token = new_execution_token
        record.updated_at = datetime.now(timezone.utc).isoformat()
        return True


def make_runtime(persistence, run_ids, calls):
    class Planner:
        def plan(self, **_kwargs):
            calls["plan"] += 1
            return TaskPlan(
                task_type="knowledge_qa",
                steps=[TaskStep(tool="knowledge_qa", reason="test")],
                route=RouteResult("knowledge_qa", 1.0, "test"),
            )

    class Tools:
        def stream(self, _prepared, _plan):
            calls["tool"] += 1
            yield sse({"type": "start"})
            yield sse({"type": "content", "data": "ok"})
            yield sse({"type": "done", "intent": "knowledge_qa", "answer": "ok"})

    return ChatGraphRuntime(ChatRuntimeDependencies(
        memory=FakeMemory(),
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda *_args: "",
        lightweight_stream=lambda *_args: iter(()),
        document_format_stream=lambda *_args: iter(()),
        document_draft_stream=lambda *_args: iter(()),
        rag_qa_stream=lambda *_args: iter(()),
        task_planner=Planner(),
        tool_orchestrator=Tools(),
        checkpointer=persistence.checkpointer,
        persistence=persistence,
        run_id_factory=lambda: next(run_ids),
    ), runtime_mode="graph")


def test_idempotent_request_replays_checkpoint_without_reexecuting_tools():
    persistence = FakePersistence()
    calls = {"plan": 0, "tool": 0}
    runtime = make_runtime(persistence, iter(["run-1", "run-2"]), calls)
    payload = {"message": "same", "request_id": "request-1"}

    first = parse_sse(runtime.stream(payload, user_id="u1", user_info=None))
    second = parse_sse(runtime.stream(payload, user_id="u1", user_info=None))

    assert calls == {"plan": 1, "tool": 1}
    assert first[-1]["type"] == second[-1]["type"] == "done"
    assert first[-1]["run_id"] == second[-1]["run_id"] == "run-1"
    assert len(persistence.records["run-1"].metadata["request_digest"]) == 64


def test_idempotency_key_rejects_a_different_request_before_streaming():
    persistence = FakePersistence()
    runtime = make_runtime(
        persistence,
        iter(["run-1", "run-2"]),
        {"plan": 0, "tool": 0},
    )
    first = {
        "message": "first payload",
        "request_id": "request-1",
        "idempotency_key": "stable-key",
    }
    second = {
        "message": "different payload",
        "request_id": "request-2",
        "idempotency_key": "stable-key",
    }

    assert parse_sse(runtime.stream(first, user_id="u1", user_info=None))[-1][
        "type"
    ] == "done"
    with pytest.raises(ChatRunConflictError, match="幂等键"):
        runtime.stream(second, user_id="u1", user_info=None)


def test_implicit_session_retry_reuses_durable_session_across_workers():
    persistence = FakePersistence()
    calls = {"plan": 0, "tool": 0}

    class WorkerMemory(FakeMemory):
        def __init__(self, generated_session):
            super().__init__()
            self.generated_session = generated_session
            self.session_calls = []

        def get_or_create_session(self, user_id, session_id=None):
            self.session_calls.append((user_id, session_id))
            return session_id or self.generated_session

    first_runtime = make_runtime(persistence, iter(["run-1"]), calls)
    first_memory = WorkerMemory("session-worker-a")
    first_runtime.deps.memory = first_memory
    payload = {
        "message": "same payload",
        "request_id": "request-cross-worker",
    }

    first = parse_sse(first_runtime.stream(payload, user_id="u1", user_info=None))

    second_runtime = make_runtime(persistence, iter(["run-2"]), calls)
    second_memory = WorkerMemory("session-worker-b")
    second_runtime.deps.memory = second_memory
    second = parse_sse(second_runtime.stream(payload, user_id="u1", user_info=None))

    assert first[-1]["type"] == second[-1]["type"] == "done"
    assert first[-1]["run_id"] == second[-1]["run_id"] == "run-1"
    assert persistence.records["run-1"].session_id == "session-worker-a"
    # The retry found the user-scoped durable run before consulting this
    # worker's empty in-process session cache.
    assert second_memory.session_calls == []
    assert calls == {"plan": 1, "tool": 1}


def test_explicit_session_remains_part_of_request_identity():
    persistence = FakePersistence()
    runtime = make_runtime(
        persistence,
        iter(["run-1", "run-2"]),
        {"plan": 0, "tool": 0},
    )
    first = {
        "message": "same payload",
        "request_id": "request-explicit-session-a",
        "idempotency_key": "explicit-session-key",
        "session_id": "session-a",
    }
    second = {
        **first,
        "request_id": "request-explicit-session-b",
        "session_id": "session-b",
    }

    assert parse_sse(runtime.stream(first, user_id="u1", user_info=None))[-1][
        "type"
    ] == "done"
    with pytest.raises(ChatRunConflictError, match="幂等键"):
        runtime.stream(second, user_id="u1", user_info=None)


def test_http_stream_reserves_identity_before_return_during_insert_race():
    class RacingPersistence(FakePersistence):
        def get_run_by_idempotency_key(self, *_args):
            return None

        def start_run(self, **_kwargs):
            conflict = RuntimeError("concurrent request identity conflict")
            conflict.conflict_type = "request_id_conflict"
            raise conflict

    persistence = RacingPersistence()
    runtime = make_runtime(
        persistence,
        iter(["run-1"]),
        {"plan": 0, "tool": 0},
    )

    with pytest.raises(ChatRunConflictError, match="concurrent"):
        runtime.stream_http(
            {"message": "payload", "request_id": "request-race"},
            user_id="u1",
            user_info=None,
        )


def test_insert_race_rebinds_only_an_implicit_session_to_the_winner():
    persistence = FakePersistence()
    runtime = make_runtime(
        persistence,
        iter(["unused"]),
        {"plan": 0, "tool": 0},
    )
    losing_state = runtime._initial_state(
        {
            "message": "same payload",
            "request_id": "request-race-rebind",
            "session_id": "session-worker-b",
        },
        user_id="u1",
    )
    winning_state = {**losing_state, "session_id": "session-worker-a"}
    winner = SimpleNamespace(
        run_id="run-winner",
        thread_id="run-winner",
        graph_name="chat",
        user_id="u1",
        session_id="session-worker-a",
        request_id="request-race-rebind",
        workflow_version="chat-v3",
        current_node="prepare",
        status="running",
        execution_token="winner-token",
        metadata={"request_digest": runtime._request_digest(winning_state)},
    )
    persistence.by_key[("u1", "chat", "request-race-rebind")] = winner

    def raise_race(**_kwargs):
        conflict = RuntimeError("idempotency key already won by another worker")
        conflict.conflict_type = "request_digest_mismatch"
        raise conflict

    persistence.start_run = raise_race

    replay = runtime._start_persisted_run(
        losing_state,
        execution_token="loser-token",
        allow_implicit_session_rebind=True,
    )

    assert replay is winner
    assert losing_state["session_id"] == "session-worker-a"


def test_done_is_not_emitted_until_ledger_finish_succeeds():
    persistence = FakePersistence(fail_finish=True)
    runtime = make_runtime(persistence, iter(["run-1"]), {"plan": 0, "tool": 0})

    events = parse_sse(runtime.stream(
        {"message": "hello", "request_id": "request-1"},
        user_id="u1",
        user_info=None,
    ))

    assert events[-1]["type"] == "error"
    assert "done" not in [event["type"] for event in events]
    assert persistence.records["run-1"].status == "failed"


def test_failed_draft_tool_remains_recoverable_and_retries_same_parent_node():
    persistence = FakePersistence()
    calls = {"tool": 0}

    class Planner:
        def plan(self, **_kwargs):
            return TaskPlan(
                task_type="document_drafting",
                steps=[TaskStep(tool="draft_document", reason="test")],
                route=RouteResult("doc_drafting", 1.0, "test"),
            )

    def draft_stream(
        _message,
        session_id,
        _user_id,
        _user_info,
        _display_message,
        _metadata,
        _route,
    ):
        calls["tool"] += 1
        if calls["tool"] == 1:
            yield sse({"type": "error", "message": "review unavailable"})
            return
        yield sse({"type": "content", "data": "recovered document"})
        yield sse({
            "type": "done",
            "intent": "doc_drafting",
            "answer": "recovered document",
            "document": "recovered document",
            "session_id": session_id,
            "plan": {},
            "actions": [],
            "source_filenames": [],
            "source_details": [],
        })

    registry = ToolRegistry()
    registry.register(ChatTool(
        name="draft_document",
        description="test",
        risk_level="low",
        input_schema={"message": "string"},
        stream=draft_stream,
    ))
    runtime = ChatGraphRuntime(ChatRuntimeDependencies(
        memory=FakeMemory(),
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda *_args: "",
        lightweight_stream=lambda *_args: iter(()),
        document_format_stream=lambda *_args: iter(()),
        document_draft_stream=draft_stream,
        rag_qa_stream=lambda *_args: iter(()),
        task_planner=Planner(),
        tool_orchestrator=ToolOrchestrator(registry),
        checkpointer=persistence.checkpointer,
        persistence=persistence,
        run_id_factory=lambda: "run-draft-retry",
    ), runtime_mode="graph")

    first = parse_sse(runtime.stream(
        {"message": "写通知", "request_id": "request-draft-retry"},
        user_id="u1",
        user_info=None,
    ))
    recovered = parse_sse(runtime.recover(
        "run-draft-retry",
        user_id="u1",
        user_info=None,
    ))

    assert first[-1] == {
        "type": "error",
        "message": "review unavailable",
    }
    assert recovered[-1]["type"] == "done"
    assert recovered[-1]["document"] == "recovered document"
    assert calls["tool"] == 2
    assert persistence.records["run-draft-retry"].status == "succeeded"


def test_consumer_close_marks_unfinished_run_failed():
    persistence = FakePersistence()
    runtime = make_runtime(persistence, iter(["run-1"]), {"plan": 0, "tool": 0})
    stream = runtime.stream(
        {"message": "hello", "request_id": "request-1"},
        user_id="u1",
        user_info=None,
    )

    assert next(stream)
    stream.close()

    assert persistence.records["run-1"].status == "failed"
    assert "client disconnected" in persistence.records["run-1"].error


def test_unstarted_http_close_fails_reservation_and_same_request_retries():
    persistence = FakePersistence()
    calls = {"plan": 0, "tool": 0}
    runtime = make_runtime(persistence, iter(["run-1", "run-2"]), calls)
    payload = {
        "message": "hello",
        "request_id": "request-unstarted-http",
        "session_id": "session-1",
    }

    stream = runtime.stream_http(payload, user_id="u1", user_info=None)

    assert persistence.records["run-1"].status == "running"
    assert calls == {"plan": 0, "tool": 0}
    stream.close()
    stream.close()

    abandoned = persistence.records["run-1"]
    assert abandoned.status == "failed"
    assert abandoned.error == runtime.UNSTARTED_HTTP_ERROR
    abandoned_updated_at = abandoned.updated_at
    assert calls == {"plan": 0, "tool": 0}

    retried = parse_sse(
        runtime.stream_http(payload, user_id="u1", user_info=None)
    )

    assert retried[-1]["type"] == "done"
    assert retried[-1]["run_id"] == "run-1"
    assert persistence.records["run-1"].status == "succeeded"
    assert persistence.recovery_claim_calls == [
        ("run-1", abandoned_updated_at)
    ]
    assert calls == {"plan": 1, "tool": 1}


def test_started_http_close_uses_active_disconnect_failure_path():
    persistence = FakePersistence()
    runtime = make_runtime(
        persistence,
        iter(["run-1"]),
        {"plan": 0, "tool": 0},
    )
    stream = runtime.stream_http(
        {
            "message": "hello",
            "request_id": "request-started-http",
            "session_id": "session-1",
        },
        user_id="u1",
        user_info=None,
    )

    assert next(stream)
    stream.close()

    record = persistence.records["run-1"]
    assert record.status == "failed"
    assert "client disconnected" in record.error
    assert record.error != runtime.UNSTARTED_HTTP_ERROR


def test_http_stream_lifecycle_wrapper_preserves_sse_bytes():
    lazy_persistence = FakePersistence()
    http_persistence = FakePersistence()
    payload = {
        "message": "same",
        "request_id": "request-byte-equivalence",
        "session_id": "session-1",
    }
    lazy = make_runtime(
        lazy_persistence,
        iter(["same-run"]),
        {"plan": 0, "tool": 0},
    )
    http = make_runtime(
        http_persistence,
        iter(["same-run"]),
        {"plan": 0, "tool": 0},
    )

    assert list(lazy.stream(payload, user_id="u1", user_info=None)) == list(
        http.stream_http(payload, user_id="u1", user_info=None)
    )


def test_resume_claim_is_atomic_and_second_request_is_rejected():
    persistence = FakePersistence()
    runtime = make_runtime(persistence, iter(["unused"]), {"plan": 0, "tool": 0})
    updated_at = datetime.now(timezone.utc).isoformat()
    record = SimpleNamespace(
        run_id="run-1", thread_id="run-1", graph_name="chat", user_id="u1",
        session_id="s1", request_id="r1", workflow_version="chat-v2",
        current_node="interrupt", status="interrupted", updated_at=updated_at,
        checkpoint_id="cp-1", interrupt_id="int-1",
    )
    persistence.records[record.run_id] = record
    snapshot = SimpleNamespace(
        config={"configurable": {"checkpoint_id": "cp-1"}},
        values={},
        tasks=(SimpleNamespace(interrupts=(SimpleNamespace(id="int-1"),)),),
    )
    runtime._graph = SimpleNamespace(get_state=lambda _config: snapshot)
    runtime._stream_resume = lambda *_args, **_kwargs: iter([
        sse({"type": "done", "answer": "resumed"})
    ])

    first = runtime.resume("run-1", {"approved": True}, user_id="u1", user_info=None)
    second = runtime.resume("run-1", {"approved": True}, user_id="u1", user_info=None)
    assert parse_sse(first)[-1]["type"] == "done"
    assert parse_sse(second)[-1]["type"] == "error"

    assert persistence.claim_calls == [
        ("run-1", "cp-1", "int-1"),
        ("run-1", "cp-1", "int-1"),
    ]


def test_unstarted_stream_and_resume_do_not_create_or_claim_runs():
    persistence = FakePersistence()
    runtime = make_runtime(persistence, iter(["run-1"]), {"plan": 0, "tool": 0})

    stream = runtime.stream(
        {"message": "hello", "request_id": "request-1"},
        user_id="u1",
        user_info=None,
    )
    stream.close()
    assert persistence.records == {}

    record = SimpleNamespace(
        run_id="run-int", thread_id="run-int", graph_name="chat", user_id="u1",
        session_id="s1", request_id="r1", workflow_version="chat-v2",
        current_node="interrupt", status="interrupted",
        updated_at=datetime.now(timezone.utc).isoformat(),
        checkpoint_id="cp", interrupt_id="int",
    )
    persistence.records[record.run_id] = record
    snapshot = SimpleNamespace(
        config={"configurable": {"checkpoint_id": "cp"}},
        values={},
        tasks=(SimpleNamespace(interrupts=(SimpleNamespace(id="int"),)),),
    )
    runtime._graph = SimpleNamespace(get_state=lambda _config: snapshot)
    resume_stream = runtime.resume(
        "run-int", True, user_id="u1", user_info=None
    )
    resume_stream.close()

    assert record.status == "interrupted"
    assert persistence.claim_calls == []


def test_resume_rejects_runs_older_than_seven_days():
    persistence = FakePersistence()
    runtime = make_runtime(persistence, iter(["unused"]), {"plan": 0, "tool": 0})
    record = SimpleNamespace(
        run_id="run-old", thread_id="run-old", graph_name="chat", user_id="u1",
        session_id="s1", request_id="r1", workflow_version="chat-v2",
        current_node="interrupt", status="interrupted",
        updated_at=(datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),
        checkpoint_id="cp", interrupt_id="int",
    )
    persistence.records[record.run_id] = record

    with pytest.raises(ChatRunConflictError, match="超过 7 天"):
        runtime.resume("run-old", True, user_id="u1", user_info=None)


def test_recover_replays_terminal_checkpoint_without_reexecuting_tools():
    persistence = FakePersistence()
    calls = {"plan": 0, "tool": 0}
    runtime = make_runtime(persistence, iter(["run-1"]), calls)

    first = parse_sse(runtime.stream(
        {"message": "hello", "request_id": "request-1"},
        user_id="u1",
        user_info=None,
    ))
    assert first[-1]["type"] == "done"
    record = persistence.records["run-1"]
    persistence.update_run("run-1", status="failed", current_node="error")

    recovered = parse_sse(runtime.recover(
        "run-1", user_id="u1", user_info=None
    ))

    assert recovered[-1]["type"] == "done"
    assert recovered[-1]["run_id"] == "run-1"
    assert record.status == "succeeded"
    assert calls == {"plan": 1, "tool": 1}


def test_recover_claim_is_lazy_and_atomic():
    persistence = FakePersistence()
    runtime = make_runtime(persistence, iter(["unused"]), {"plan": 0, "tool": 0})
    updated_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    record = SimpleNamespace(
        run_id="run-failed", thread_id="run-failed", graph_name="chat",
        user_id="u1", session_id="s1", request_id="r1",
        workflow_version="chat-v2", current_node="tool", status="failed",
        updated_at=updated_at, checkpoint_id="cp", interrupt_id="",
        execution_token="old-token",
    )
    persistence.records[record.run_id] = record
    snapshot = SimpleNamespace(
        config={"configurable": {"checkpoint_id": "cp"}},
        values={
            "run_id": record.run_id,
            "request_id": record.request_id,
            "user_id": record.user_id,
            "session_id": record.session_id,
        },
        tasks=(),
        next=("tool_knowledge_qa",),
    )
    runtime._graph = SimpleNamespace(get_state=lambda _config: snapshot)
    runtime._stream_recover = lambda *_args, **_kwargs: iter([
        sse({"type": "done", "answer": "recovered"})
    ])

    first = runtime.recover("run-failed", user_id="u1", user_info=None)
    second = runtime.recover("run-failed", user_id="u1", user_info=None)
    assert persistence.recovery_claim_calls == []

    assert parse_sse(first)[-1]["type"] == "done"
    assert parse_sse(second)[-1]["type"] == "error"
    assert len(persistence.recovery_claim_calls) == 2


def test_recover_does_not_steal_fresh_running_lease():
    persistence = FakePersistence()
    runtime = make_runtime(persistence, iter(["unused"]), {"plan": 0, "tool": 0})
    record = SimpleNamespace(
        run_id="run-live", thread_id="run-live", graph_name="chat",
        user_id="u1", session_id="s1", request_id="r1",
        workflow_version="chat-v2", current_node="tool", status="running",
        updated_at=datetime.now(timezone.utc).isoformat(), checkpoint_id="cp",
        interrupt_id="", execution_token="live-token",
    )
    persistence.records[record.run_id] = record

    with pytest.raises(ChatRunConflictError, match="活跃执行"):
        runtime.recover("run-live", user_id="u1", user_info=None)


def test_production_graph_refuses_implicit_inmemory_fallback(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="Production LangGraph requires"):
        ChatGraphRuntime(ChatRuntimeDependencies(
            memory=FakeMemory(),
            upload_manager=FakeUploadManager(),
            reimbursement_detector=lambda *_args: "",
            lightweight_stream=lambda *_args: iter(()),
            document_format_stream=lambda *_args: iter(()),
            document_draft_stream=lambda *_args: iter(()),
            rag_qa_stream=lambda *_args: iter(()),
            task_planner=object(),
            tool_orchestrator=object(),
        ), runtime_mode="graph")
