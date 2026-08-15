"""Opt-in acceptance tests against a real PostgreSQL server.

Set ``LANGGRAPH_TEST_POSTGRES_DSN`` to a disposable PostgreSQL database.  The
test creates and drops an isolated schema, so the normal local test suite can
skip it safely while ``docker-compose.langgraph-test.yml`` supplies a complete
ephemeral database automatically.  One test verifies independent pools in one
process; another uses spawned Python processes and an abrupt worker exit.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import uuid
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from datetime import datetime, timezone
from typing import TypedDict

import pytest

from graph_persistence import (
    GraphPersistenceConfig,
    GraphPersistenceConfigurationError,
    GraphRunIdentityConflictError,
    GraphSessionDeletionRequestedError,
    PostgresGraphRunLedger,
    create_graph_persistence,
    setup_postgres_persistence,
)
from langgraph.graph import END, StateGraph


POSTGRES_TEST_DSN = os.getenv("LANGGRAPH_TEST_POSTGRES_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not POSTGRES_TEST_DSN,
    reason="set LANGGRAPH_TEST_POSTGRES_DSN to run real PostgreSQL acceptance",
)


class CounterState(TypedDict):
    value: int
    trace: list[str]


@pytest.fixture()
def isolated_postgres_dsn():
    psycopg = pytest.importorskip("psycopg")
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    schema = f"langgraph_accept_{uuid.uuid4().hex}"
    with psycopg.connect(POSTGRES_TEST_DSN, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )
    scoped_dsn = make_conninfo(
        POSTGRES_TEST_DSN,
        options=f"-c search_path={schema}",
    )
    try:
        yield scoped_dsn, schema
    finally:
        with psycopg.connect(POSTGRES_TEST_DSN, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


def _counter_graph(checkpointer):
    workflow = StateGraph(CounterState)

    def first(state: CounterState) -> CounterState:
        return {
            "value": int(state.get("value", 0)) + 1,
            "trace": [*state.get("trace", []), "first"],
        }

    def second(state: CounterState) -> CounterState:
        return {
            "value": int(state.get("value", 0)) + 1,
            "trace": [*state.get("trace", []), "second"],
        }

    workflow.add_node("first", first)
    workflow.add_node("second", second)
    workflow.set_entry_point("first")
    workflow.add_edge("first", "second")
    workflow.add_edge("second", END)
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["first"],
    )


def _postgres_config(dsn: str) -> GraphPersistenceConfig:
    return GraphPersistenceConfig(
        environment="production",
        backend="postgres",
        postgres_dsn=dsn,
        run_ledger_backend="postgres",
        postgres_pool_min_size=1,
        postgres_pool_max_size=2,
        postgres_pool_timeout_seconds=3.0,
        worker_processes=4,
        default_durability="sync",
    )


def _checkpoint_then_exit_worker(
    dsn: str,
    run_id: str,
    sender,
) -> None:
    """Persist one graph step, then emulate a worker killed before cleanup."""

    runtime = None
    try:
        runtime = create_graph_persistence(_postgres_config(dsn))
        execution_token = f"worker-crash-{uuid.uuid4().hex}"
        runtime.start_run(
            run_id=run_id,
            thread_id=run_id,
            graph_name="postgres_multiprocess_acceptance",
            user_id="acceptance-user",
            session_id="acceptance-session",
            request_id=f"request-{run_id}",
            workflow_version="postgres-acceptance-v1",
            current_node="first",
            execution_token=execution_token,
            durability="sync",
            # The persistence CAS deliberately has no wall-clock policy; the
            # chat runtime applies the stale-lease window before calling it.
            started_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        graph = _counter_graph(runtime.checkpointer)
        graph_config = {"configurable": {"thread_id": run_id}}
        graph.invoke(
            {"value": 0, "trace": []},
            graph_config,
            durability="sync",
        )
        snapshot = graph.get_state(graph_config)
        sender.send({
            "ok": True,
            "pid": os.getpid(),
            "next": list(snapshot.next),
            "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
        })
        sender.close()
    except BaseException as exc:
        try:
            sender.send({"ok": False, "pid": os.getpid(), "error": repr(exc)})
            sender.close()
        finally:
            os._exit(91)

    # Intentionally bypass runtime.close()/atexit to model a killed Gunicorn
    # worker.  The operating system closes its PostgreSQL sockets.
    os._exit(17)


def _recover_in_new_worker(dsn: str, run_id: str, sender) -> None:
    runtime = None
    try:
        runtime = create_graph_persistence(_postgres_config(dsn))
        record = runtime.get_owned_run(run_id, user_id="acceptance-user")
        if record is None:
            raise AssertionError("recovering worker could not read the run ledger")
        execution_token = f"worker-recover-{uuid.uuid4().hex}"
        if not runtime.claim_recovery(
            run_id,
            user_id="acceptance-user",
            expected_updated_at=record.updated_at,
            new_execution_token=execution_token,
        ):
            raise AssertionError("recovering worker lost the recovery CAS")

        graph = _counter_graph(runtime.checkpointer)
        graph_config = {"configurable": {"thread_id": run_id}}
        before = graph.get_state(graph_config)
        result = graph.invoke(None, graph_config, durability="sync")
        final_snapshot = graph.get_state(graph_config)
        finished = runtime.finish_run(
            run_id,
            user_id="acceptance-user",
            execution_token=execution_token,
            status="succeeded",
            current_node="__end__",
            checkpoint_id=final_snapshot.config["configurable"]["checkpoint_id"],
        )
        if finished is None:
            raise AssertionError("recovering worker lost the execution lease")
        sender.send({
            "ok": True,
            "pid": os.getpid(),
            "next_before": list(before.next),
            "result": result,
            "status": finished.status,
        })
        sender.close()
    except BaseException as exc:
        try:
            sender.send({"ok": False, "pid": os.getpid(), "error": repr(exc)})
            sender.close()
        finally:
            if runtime is not None:
                runtime.close()
        raise
    finally:
        if runtime is not None:
            runtime.close()


def test_real_postgres_setup_and_two_independent_pool_recovery(
    isolated_postgres_dsn,
):
    psycopg = pytest.importorskip("psycopg")
    scoped_dsn, schema = isolated_postgres_dsn
    config = _postgres_config(scoped_dsn)

    # A worker must fail before serving requests when the independent
    # migration job has not run.
    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="schema is missing or outdated",
    ):
        create_graph_persistence(config)

    # Deployment migrations are forward-only and safe to run again.
    setup_postgres_persistence(config)
    setup_postgres_persistence(config)

    with psycopg.connect(scoped_dsn, autocommit=True) as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = %s",
                (schema,),
            ).fetchall()
        }
    assert {
        "uq_graph_runs_idempotency",
        "uq_graph_runs_request",
        "uq_graph_session_deletions_owner",
        "uq_graph_effects_key",
    }.issubset(indexes)

    thread_id = f"pg-thread-{uuid.uuid4().hex}"
    run_id = thread_id
    first_token = f"worker-a-{uuid.uuid4().hex}"
    second_token = f"worker-b-{uuid.uuid4().hex}"
    graph_config = {"configurable": {"thread_id": thread_id}}

    first_runtime = create_graph_persistence(config)
    second_runtime = None
    try:
        first_runtime.start_run(
            run_id=run_id,
            thread_id=thread_id,
            graph_name="postgres_acceptance",
            user_id="acceptance-user",
            session_id="acceptance-session",
            request_id=f"request-{run_id}",
            workflow_version="postgres-acceptance-v1",
            current_node="first",
            execution_token=first_token,
            durability="sync",
        )
        first_graph = _counter_graph(first_runtime.checkpointer)
        first_graph.invoke(
            {"value": 0, "trace": []},
            graph_config,
            durability="sync",
        )
        first_snapshot = first_graph.get_state(graph_config)
        assert first_snapshot.next == ("second",)
        checkpoint_id = first_snapshot.config["configurable"]["checkpoint_id"]
        failed = first_runtime.finish_run(
            run_id,
            user_id="acceptance-user",
            execution_token=first_token,
            status="failed",
            current_node="simulated_worker_exit",
            checkpoint_id=checkpoint_id,
            error="simulated worker exit",
        )
        assert failed is not None

        # A separately constructed runtime in this same process owns a
        # different pool, sees the durable checkpoint, and wins the recovery
        # CAS once.  The next test covers a true process boundary.
        second_runtime = create_graph_persistence(config)
        assert (
            first_runtime.checkpointer_handle.resource
            is not second_runtime.checkpointer_handle.resource
        )
        second_graph = _counter_graph(second_runtime.checkpointer)
        assert second_graph.get_state(graph_config).next == ("second",)
        recovered_record = second_runtime.get_owned_run(
            run_id, user_id="acceptance-user"
        )
        assert recovered_record is not None
        assert second_runtime.claim_recovery(
            run_id,
            user_id="acceptance-user",
            expected_updated_at=recovered_record.updated_at,
            new_execution_token=second_token,
        )
        assert not first_runtime.claim_recovery(
            run_id,
            user_id="acceptance-user",
            expected_updated_at=recovered_record.updated_at,
            new_execution_token=f"loser-{uuid.uuid4().hex}",
        )

        result = second_graph.invoke(None, graph_config, durability="sync")
        assert result == {"value": 2, "trace": ["first", "second"]}
        final_snapshot = second_graph.get_state(graph_config)
        finished = second_runtime.finish_run(
            run_id,
            user_id="acceptance-user",
            execution_token=second_token,
            status="succeeded",
            current_node="__end__",
            checkpoint_id=final_snapshot.config["configurable"]["checkpoint_id"],
        )
        assert finished is not None
        assert finished.status == "succeeded"
    finally:
        if second_runtime is not None:
            second_runtime.close()
        first_runtime.close()


def test_real_postgres_session_barrier_and_identity_conflicts(
    isolated_postgres_dsn,
):
    scoped_dsn, _schema = isolated_postgres_dsn
    config = _postgres_config(scoped_dsn)
    setup_postgres_persistence(config)
    runtime = create_graph_persistence(config)
    try:
        common = {
            "graph_name": "postgres_identity_acceptance",
            "user_id": "acceptance-user",
            "session_id": "active-session",
            "workflow_version": "postgres-acceptance-v1",
            "durability": "sync",
        }
        first = runtime.start_run(
            run_id="identity-run-a",
            thread_id="identity-thread-a",
            request_id="identity-request-a",
            idempotency_key="identity-idem-a",
            metadata={"request_digest": "sha256:a"},
            **common,
        )
        runtime.start_run(
            run_id="identity-run-b",
            thread_id="identity-thread-b",
            request_id="identity-request-b",
            idempotency_key="identity-idem-b",
            metadata={"request_digest": "sha256:b"},
            **common,
        )

        with pytest.raises(GraphRunIdentityConflictError) as dual_exc:
            runtime.start_run(
                run_id="identity-run-conflict",
                thread_id="identity-thread-conflict",
                request_id="identity-request-a",
                idempotency_key="identity-idem-b",
                **common,
            )
        assert dual_exc.value.conflict_type == (
            "request_id_idempotency_key_mismatch"
        )
        assert dual_exc.value.existing_run_id == first.run_id

        with pytest.raises(GraphRunIdentityConflictError) as digest_exc:
            runtime.start_run(
                run_id="identity-run-digest-conflict",
                thread_id="identity-thread-digest-conflict",
                request_id="identity-request-replay",
                idempotency_key="identity-idem-a",
                metadata={"request_digest": "sha256:different"},
                **common,
            )
        assert digest_exc.value.conflict_type == "request_digest_mismatch"

        runtime.run_ledger.enqueue_session_deletion(
            "deleted-session",
            user_id="acceptance-user",
            reason="user_delete",
        )
        with pytest.raises(GraphSessionDeletionRequestedError):
            runtime.start_run(
                run_id="deleted-session-run",
                thread_id="deleted-session-thread",
                request_id="deleted-session-request",
                graph_name="postgres_identity_acceptance",
                user_id="acceptance-user",
                session_id="deleted-session",
                workflow_version="postgres-acceptance-v1",
                durability="sync",
            )

        barrier_locked = threading.Event()
        release_barrier = threading.Event()

        class CoordinatedPostgresLedger(PostgresGraphRunLedger):
            def _acquire_session_lock(
                self, cursor, *, user_id, session_id
            ):
                super()._acquire_session_lock(
                    cursor,
                    user_id=user_id,
                    session_id=session_id,
                )
                if not barrier_locked.is_set():
                    barrier_locked.set()
                    if not release_barrier.wait(timeout=5):
                        raise AssertionError(
                            "test did not release PostgreSQL session lock"
                        )

        coordinated = CoordinatedPostgresLedger(
            runtime.checkpointer_handle.resource
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            deletion = executor.submit(
                coordinated.enqueue_session_deletion,
                "concurrent-delete-session",
                user_id="acceptance-user",
                reason="user_delete",
            )
            assert barrier_locked.wait(timeout=5)
            start = executor.submit(
                coordinated.start_run,
                run_id="concurrent-delete-run",
                thread_id="concurrent-delete-thread",
                graph_name="postgres_identity_acceptance",
                user_id="acceptance-user",
                session_id="concurrent-delete-session",
                request_id="concurrent-delete-request",
                workflow_version="postgres-acceptance-v1",
                durability="sync",
            )
            try:
                with pytest.raises(FutureTimeoutError):
                    start.result(timeout=0.2)
            finally:
                release_barrier.set()
            deletion.result(timeout=5)
            with pytest.raises(GraphSessionDeletionRequestedError):
                start.result(timeout=5)

        assert coordinated.get_run("concurrent-delete-run") is None

        recovery_run = runtime.start_run(
            run_id="concurrent-recovery-run",
            thread_id="concurrent-recovery-run",
            graph_name="postgres_identity_acceptance",
            user_id="acceptance-user",
            session_id="concurrent-recovery-session",
            request_id="concurrent-recovery-request",
            workflow_version="postgres-acceptance-v1",
            execution_token="worker-before-delete",
            durability="sync",
        )
        runtime.finish_run(
            recovery_run.run_id,
            user_id="acceptance-user",
            execution_token=recovery_run.execution_token,
            status="failed",
            current_node="error",
        )
        recovery_record = runtime.get_owned_run(
            recovery_run.run_id,
            user_id="acceptance-user",
        )
        assert recovery_record is not None

        claim_barrier_locked = threading.Event()
        release_claim_barrier = threading.Event()

        class RecoveryCoordinatedLedger(PostgresGraphRunLedger):
            def _acquire_session_lock(
                self, cursor, *, user_id, session_id
            ):
                super()._acquire_session_lock(
                    cursor,
                    user_id=user_id,
                    session_id=session_id,
                )
                if not claim_barrier_locked.is_set():
                    claim_barrier_locked.set()
                    if not release_claim_barrier.wait(timeout=5):
                        raise AssertionError(
                            "test did not release recovery session lock"
                        )

        recovery_coordinated = RecoveryCoordinatedLedger(
            runtime.checkpointer_handle.resource
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            deletion = executor.submit(
                recovery_coordinated.enqueue_session_deletion,
                "concurrent-recovery-session",
                user_id="acceptance-user",
                reason="user_delete",
            )
            assert claim_barrier_locked.wait(timeout=5)
            claim = executor.submit(
                recovery_coordinated.claim_recovery,
                recovery_run.run_id,
                user_id="acceptance-user",
                expected_updated_at=recovery_record.updated_at,
                new_execution_token="worker-after-delete",
            )
            try:
                with pytest.raises(FutureTimeoutError):
                    claim.result(timeout=0.2)
            finally:
                release_claim_barrier.set()
            deletion.result(timeout=5)
            assert claim.result(timeout=5) is False

        active_run = runtime.start_run(
            run_id="concurrent-active-write-run",
            thread_id="concurrent-active-write-run",
            graph_name="postgres_identity_acceptance",
            user_id="acceptance-user",
            session_id="concurrent-active-write-session",
            request_id="concurrent-active-write-request",
            workflow_version="postgres-acceptance-v1",
            execution_token="worker-active-before-delete",
            durability="sync",
        )
        for effect in ("message:complete", "message:fail"):
            assert runtime.claim_effect(
                active_run.run_id,
                "send",
                effect,
                user_id="acceptance-user",
                execution_token=active_run.execution_token,
            )

        active_barrier_locked = threading.Event()
        release_active_barrier = threading.Event()

        class ActiveWriteCoordinatedLedger(PostgresGraphRunLedger):
            def _acquire_session_lock(
                self, cursor, *, user_id, session_id
            ):
                super()._acquire_session_lock(
                    cursor,
                    user_id=user_id,
                    session_id=session_id,
                )
                if not active_barrier_locked.is_set():
                    active_barrier_locked.set()
                    if not release_active_barrier.wait(timeout=5):
                        raise AssertionError(
                            "test did not release active-write session lock"
                        )

        active_coordinated = ActiveWriteCoordinatedLedger(
            runtime.checkpointer_handle.resource
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            deletion = executor.submit(
                active_coordinated.enqueue_session_deletion,
                "concurrent-active-write-session",
                user_id="acceptance-user",
                reason="user_delete",
            )
            assert active_barrier_locked.wait(timeout=5)
            heartbeat = executor.submit(
                active_coordinated.heartbeat_run,
                active_run.run_id,
                user_id="acceptance-user",
                execution_token=active_run.execution_token,
                current_node="execute_tools",
            )
            try:
                with pytest.raises(FutureTimeoutError):
                    heartbeat.result(timeout=0.2)
            finally:
                release_active_barrier.set()
            deletion.result(timeout=5)
            assert heartbeat.result(timeout=5) is None

        assert active_coordinated.update_run(
            active_run.run_id,
            status="succeeded",
            current_node="__end__",
        ) is None
        assert active_coordinated.finish_run(
            active_run.run_id,
            user_id="acceptance-user",
            execution_token=active_run.execution_token,
            status="succeeded",
            current_node="__end__",
        ) is None
        assert active_coordinated.claim_effect(
            active_run.run_id,
            "send",
            "message:new",
            user_id="acceptance-user",
            execution_token=active_run.execution_token,
        ) is False
        assert active_coordinated.complete_effect(
            active_run.run_id,
            "send",
            "message:complete",
            user_id="acceptance-user",
            execution_token=active_run.execution_token,
            result={"sent": True},
        ) is None
        assert active_coordinated.fail_effect(
            active_run.run_id,
            "send",
            "message:fail",
            user_id="acceptance-user",
            execution_token=active_run.execution_token,
            error="must remain claimed",
        ) is None
        active_record = active_coordinated.get_owned_run(
            active_run.run_id,
            user_id="acceptance-user",
        )
        assert active_record is not None
        assert active_record.status == "running"
        assert {
            item.effect: item.status
            for item in active_coordinated.list_run_effects(
                active_run.run_id,
                user_id="acceptance-user",
            )
        } == {
            "message:complete": "claimed",
            "message:fail": "claimed",
        }
    finally:
        runtime.close()


def test_real_postgres_recovers_across_process_after_abrupt_worker_exit(
    isolated_postgres_dsn,
):
    scoped_dsn, _schema = isolated_postgres_dsn
    setup_postgres_persistence(_postgres_config(scoped_dsn))
    run_id = f"pg-process-{uuid.uuid4().hex}"
    process_context = multiprocessing.get_context("spawn")

    crash_receiver, crash_sender = process_context.Pipe(duplex=False)
    crashing_worker = process_context.Process(
        target=_checkpoint_then_exit_worker,
        args=(scoped_dsn, run_id, crash_sender),
        name="langgraph-checkpoint-worker",
    )
    crashing_worker.start()
    crash_sender.close()
    assert crash_receiver.poll(30), "checkpoint worker did not report in time"
    crash_report = crash_receiver.recv()
    crashing_worker.join(timeout=30)
    assert not crashing_worker.is_alive(), "checkpoint worker did not exit"
    assert crash_report.get("ok") is True, crash_report
    assert crash_report["next"] == ["second"]
    assert crash_report["checkpoint_id"]
    assert crashing_worker.exitcode == 17

    recovery_receiver, recovery_sender = process_context.Pipe(duplex=False)
    recovering_worker = process_context.Process(
        target=_recover_in_new_worker,
        args=(scoped_dsn, run_id, recovery_sender),
        name="langgraph-recovery-worker",
    )
    recovering_worker.start()
    recovery_sender.close()
    assert recovery_receiver.poll(30), "recovery worker did not report in time"
    recovery_report = recovery_receiver.recv()
    recovering_worker.join(timeout=30)
    assert not recovering_worker.is_alive(), "recovery worker did not exit"
    assert recovering_worker.exitcode == 0
    assert recovery_report.get("ok") is True, recovery_report
    assert recovery_report["next_before"] == ["second"]
    assert recovery_report["result"] == {
        "value": 2,
        "trace": ["first", "second"],
    }
    assert recovery_report["status"] == "succeeded"
    assert crash_report["pid"] != recovery_report["pid"]
    assert crash_report["pid"] != os.getpid()
    assert recovery_report["pid"] != os.getpid()
