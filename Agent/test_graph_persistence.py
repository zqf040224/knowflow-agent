from __future__ import annotations

import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

import graph_persistence as persistence
from graph_persistence import (
    CheckpointerHandle,
    GraphPersistenceConfig,
    GraphPersistenceConfigurationError,
    GraphPersistenceOwnershipError,
    GraphPersistenceRuntime,
    GraphRunIdentityConflictError,
    GraphSessionDeletionRequestedError,
    PostgresGraphRunLedger,
    SqliteGraphRunLedger,
    create_checkpointer,
)


def make_config(tmp_path, *, backend="memory", environment="development", dsn=""):
    return GraphPersistenceConfig(
        environment=environment,
        backend=backend,
        sqlite_path=tmp_path / "checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "graph_runs.sqlite",
        postgres_dsn=dsn,
        run_ledger_backend="postgres" if backend == "postgres" else "sqlite",
    )


def run_identity(run_id, *, user_id="user-1", session_id="session-1"):
    return {
        "user_id": user_id,
        "session_id": session_id,
        "request_id": f"request-{run_id}",
        "workflow_version": "workflow-v1",
    }


def test_config_rejects_non_postgres_production_backends():
    with pytest.raises(GraphPersistenceConfigurationError, match="Production requires"):
        GraphPersistenceConfig.from_env(
            {
                "APP_ENV": "production",
                "LANGGRAPH_CHECKPOINTER_BACKEND": "memory",
            }
        )

    with pytest.raises(GraphPersistenceConfigurationError, match="POSTGRES_DSN"):
        GraphPersistenceConfig.from_env(
            {
                "APP_ENV": "production",
                "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
            }
        )


def test_config_rejects_unsafe_production_msgpack():
    with pytest.raises(GraphPersistenceConfigurationError, match="STRICT_MSGPACK"):
        GraphPersistenceConfig.from_env(
            {
                "APP_ENV": "production",
                "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
                "LANGGRAPH_POSTGRES_DSN": "postgresql://example.invalid/db",
                "LANGGRAPH_STRICT_MSGPACK": "false",
            }
        )


def test_config_rejects_local_run_ledger_in_production():
    with pytest.raises(GraphPersistenceConfigurationError, match="RUN_LEDGER"):
        GraphPersistenceConfig.from_env(
            {
                "APP_ENV": "production",
                "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
                "LANGGRAPH_POSTGRES_DSN": "postgresql://example.invalid/db",
                "LANGGRAPH_RUN_LEDGER_BACKEND": "sqlite",
            }
        )


def test_development_defaults_to_sqlite_checkpointer_and_ledger():
    config = GraphPersistenceConfig.from_env({})

    assert config.backend == "sqlite"
    assert config.run_ledger_backend == "sqlite"
    assert config.sqlite_path == (
        persistence.PROJECT_ROOT / "data/langgraph.sqlite"
    ).resolve()


def test_config_parses_pool_retention_and_project_relative_paths():
    config = GraphPersistenceConfig.from_env(
        {
            "LANGGRAPH_CHECKPOINTER_BACKEND": "sqlite",
            "LANGGRAPH_SQLITE_PATH": "data/test-checkpoints.sqlite",
            "LANGGRAPH_RUN_LEDGER_SQLITE_PATH": "data/test-runs.sqlite",
            "LANGGRAPH_POOL_MIN_SIZE": "2",
            "LANGGRAPH_POOL_MAX_SIZE": "2",
            "LANGGRAPH_POOL_TIMEOUT_SECONDS": "7.5",
            "LANGGRAPH_DURABILITY": "sync",
            "LANGGRAPH_CHECKPOINT_RETENTION_DAYS": "45",
            "LANGGRAPH_RETENTION_BATCH_SIZE": "17",
        }
    )

    assert config.sqlite_path == (
        persistence.PROJECT_ROOT / "data/test-checkpoints.sqlite"
    ).resolve()
    assert config.run_ledger_sqlite_path == (
        persistence.PROJECT_ROOT / "data/test-runs.sqlite"
    ).resolve()
    assert config.postgres_pool_min_size == 2
    assert config.postgres_pool_max_size == 2
    assert config.postgres_pool_timeout_seconds == 7.5
    assert config.default_durability == "sync"
    assert config.retention_days == 45
    assert config.retention_batch_size == 17


def test_config_rejects_inverted_pool_bounds():
    with pytest.raises(GraphPersistenceConfigurationError, match="MAX_SIZE"):
        GraphPersistenceConfig.from_env(
            {
                "LANGGRAPH_POOL_MIN_SIZE": "4",
                "LANGGRAPH_POOL_MAX_SIZE": "2",
            }
        )


def test_config_rejects_multi_process_local_persistence():
    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="Multi-process LangGraph requires",
    ):
        GraphPersistenceConfig.from_env({
            "APP_ENV": "development",
            "LANGGRAPH_CHECKPOINTER_BACKEND": "sqlite",
            "WEB_CONCURRENCY": "4",
        })

    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="Multi-process LangGraph requires",
    ):
        GraphPersistenceConfig.from_env({
            "APP_ENV": "development",
            "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
            "LANGGRAPH_RUN_LEDGER_BACKEND": "sqlite",
            "LANGGRAPH_POSTGRES_DSN": "postgresql://example.invalid/db",
            "WEB_CONCURRENCY": "2",
        })

    postgres = GraphPersistenceConfig.from_env({
        "APP_ENV": "production",
        "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
        "LANGGRAPH_POSTGRES_DSN": "postgresql://example.invalid/db",
        "WEB_CONCURRENCY": "4",
    })
    assert postgres.worker_processes == 4


def test_config_rejects_postgres_connection_budget_above_eight():
    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="must not exceed 8 total PostgreSQL connections",
    ):
        GraphPersistenceConfig.from_env({
            "APP_ENV": "production",
            "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
            "LANGGRAPH_POSTGRES_DSN": "postgresql://example.invalid/db",
            "LANGGRAPH_POOL_MAX_SIZE": "2",
            "WEB_CONCURRENCY": "5",
        })

    accepted = GraphPersistenceConfig.from_env({
        "APP_ENV": "production",
        "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
        "LANGGRAPH_POSTGRES_DSN": "postgresql://example.invalid/db",
        "LANGGRAPH_POOL_MAX_SIZE": "1",
        "WEB_CONCURRENCY": "8",
    })
    assert accepted.worker_processes * accepted.postgres_pool_max_size == 8


@pytest.mark.parametrize(
    "environment",
    ["staging", "test", "prod"],
)
def test_config_rejects_unlocked_app_environments(environment):
    with pytest.raises(GraphPersistenceConfigurationError, match="APP_ENV"):
        GraphPersistenceConfig.from_env({"APP_ENV": environment})


def test_config_rejects_env_memory_backend_and_pool_above_two():
    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="sqlite or postgres",
    ):
        GraphPersistenceConfig.from_env({
            "LANGGRAPH_CHECKPOINTER_BACKEND": "memory",
        })

    with pytest.raises(GraphPersistenceConfigurationError, match="must not exceed 2"):
        GraphPersistenceConfig.from_env({"LANGGRAPH_POOL_MAX_SIZE": "9"})


def test_memory_factory_applies_strict_msgpack_before_saver_creation(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LANGGRAPH_STRICT_MSGPACK", raising=False)
    config = make_config(tmp_path)

    with create_checkpointer(config) as handle:
        assert handle.backend == "memory"
        assert handle.saver is not None
        assert os.environ["LANGGRAPH_STRICT_MSGPACK"] == "true"
        assert handle.saver.serde._allowed_msgpack_modules is None


def test_graph_persistence_factory_bundles_memory_saver_and_sqlite_ledger(tmp_path):
    config = make_config(tmp_path)

    with persistence.create_graph_persistence(config) as runtime:
        record = runtime.start_run(
            run_id="run-bundle",
            thread_id="thread-bundle",
            graph_name="chat",
            **run_identity("bundle"),
        )

        assert runtime.checkpointer is not None
        assert isinstance(runtime.run_ledger, SqliteGraphRunLedger)
        assert record.durability == "async"
        assert config.run_ledger_sqlite_path.exists()


def test_sqlite_checkpointer_factory_when_optional_dependency_is_available(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    config = make_config(tmp_path, backend="sqlite")

    with create_checkpointer(config) as handle:
        assert handle.backend == "sqlite"
        assert handle.resource is not None
        assert config.sqlite_path.exists()


class FakePool:
    instances = []

    @staticmethod
    def check_connection(connection):
        return None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.open_calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def open(self, *, wait, timeout):
        self.open_calls.append({"wait": wait, "timeout": timeout})

    def close(self):
        self.closed = True

    def connection(self):
        raise AssertionError("test did not expect a real database query")


class FakePostgresSaver:
    instances = []

    def __init__(self, pool, *, serde=None):
        self.pool = pool
        self.serde = serde
        self.setup_calls = 0
        self.__class__.instances.append(self)

    def setup(self):
        self.setup_calls += 1


def test_postgres_factory_uses_bounded_pool_without_implicit_setup(
    tmp_path, monkeypatch
):
    FakePool.instances.clear()
    FakePostgresSaver.instances.clear()
    fake_dict_row = object()
    monkeypatch.setattr(
        persistence,
        "_load_postgres_dependencies",
        lambda: (FakePostgresSaver, FakePool, fake_dict_row),
    )
    config = make_config(
        tmp_path,
        backend="postgres",
        environment="production",
        dsn="postgresql://secret@example.invalid/graph",
    )

    handle = create_checkpointer(config)
    pool = FakePool.instances[-1]
    saver = FakePostgresSaver.instances[-1]

    assert pool.kwargs["min_size"] == 1
    assert pool.kwargs["max_size"] == 2
    assert pool.kwargs["open"] is False
    assert pool.kwargs["kwargs"] == {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": fake_dict_row,
    }
    assert pool.open_calls == [{"wait": True, "timeout": 5.0}]
    assert saver.setup_calls == 0
    assert saver.serde._allowed_msgpack_modules is None

    handle.close()
    assert pool.closed is True


def test_postgres_setup_is_an_explicit_separate_operation(tmp_path, monkeypatch):
    FakePool.instances.clear()
    FakePostgresSaver.instances.clear()
    monkeypatch.setattr(
        persistence,
        "_load_postgres_dependencies",
        lambda: (FakePostgresSaver, FakePool, object()),
    )
    config = make_config(
        tmp_path,
        backend="postgres",
        environment="production",
        dsn="postgresql://secret@example.invalid/graph",
    )

    persistence.setup_postgres_checkpointer(config)

    assert FakePostgresSaver.instances[-1].setup_calls == 1
    assert FakePool.instances[-1].closed is True


def test_full_postgres_setup_includes_shared_graph_runs_schema(tmp_path, monkeypatch):
    FakePool.instances.clear()
    FakePostgresSaver.instances.clear()
    ledger_setup_pools = []
    monkeypatch.setattr(
        persistence,
        "_load_postgres_dependencies",
        lambda: (FakePostgresSaver, FakePool, object()),
    )
    monkeypatch.setattr(
        PostgresGraphRunLedger,
        "setup",
        lambda self: ledger_setup_pools.append(self.pool),
    )
    config = make_config(
        tmp_path,
        backend="postgres",
        environment="production",
        dsn="postgresql://secret@example.invalid/graph",
    )

    persistence.setup_postgres_persistence(config)

    assert FakePostgresSaver.instances[-1].setup_calls == 1
    assert ledger_setup_pools == [FakePool.instances[-1]]
    assert FakePool.instances[-1].closed is True


class RecordingCursor:
    def __init__(self, statements):
        self.statements = statements
        self.last_statement = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, statement, params=None):
        self.last_statement = " ".join(statement.split())
        self.statements.append((self.last_statement, params))

    def fetchone(self):
        from langgraph.checkpoint.postgres.base import MIGRATIONS

        return {"latest_v": len(MIGRATIONS) - 1}

    def fetchall(self):
        if "FROM pg_indexes" not in self.last_statement:
            return []
        return [
            {
                "indexname": "uq_graph_runs_request",
                "indexdef": (
                    "CREATE UNIQUE INDEX uq_graph_runs_request ON graph_runs "
                    "USING btree (user_id, graph_name, request_id)"
                ),
            },
            {
                "indexname": "uq_graph_runs_idempotency",
                "indexdef": (
                    "CREATE UNIQUE INDEX uq_graph_runs_idempotency ON graph_runs "
                    "USING btree (user_id, graph_name, idempotency_key)"
                ),
            },
            {
                "indexname": "uq_graph_effects_key",
                "indexdef": (
                    "CREATE UNIQUE INDEX uq_graph_effects_key ON graph_effects "
                    "USING btree (run_id, node, effect)"
                ),
            },
            {
                "indexname": "uq_graph_session_deletions_owner",
                "indexdef": (
                    "CREATE UNIQUE INDEX uq_graph_session_deletions_owner "
                    "ON graph_session_deletions USING btree "
                    "(user_id, session_id)"
                ),
            },
        ]


class RecordingConnection:
    def __init__(self, statements):
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def cursor(self):
        return RecordingCursor(self.statements)


class RecordingPool:
    def __init__(self):
        self.statements = []

    def connection(self):
        return RecordingConnection(self.statements)


def test_postgres_session_lock_uses_transaction_scoped_stable_identity():
    statements = []
    cursor = RecordingCursor(statements)

    PostgresGraphRunLedger._acquire_session_lock(
        cursor,
        user_id="owner",
        session_id="session-delete",
    )

    assert len(statements) == 1
    sql, params = statements[0]
    assert "pg_advisory_xact_lock" in sql
    assert "hashtextextended" in sql
    assert params == ('["owner","session-delete"]',)


def test_postgres_session_lock_transaction_wraps_autocommit_connection():
    events = []

    class Transaction:
        def __enter__(self):
            events.append("begin")

        def __exit__(self, exc_type, exc, traceback):
            events.append("commit" if exc_type is None else "rollback")

    class Connection:
        def transaction(self):
            return Transaction()

    with PostgresGraphRunLedger._transaction(Connection()):
        events.append("body")

    assert events == ["begin", "body", "commit"]


class PostgresCasCursor:
    def __init__(self, state, statements):
        self.state = state
        self.statements = statements
        self.rowcount = 0
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, statement, params=None):
        normalized = " ".join(statement.split())
        self.statements.append((normalized, params))
        self.rowcount = 0
        self._row = None
        if normalized.startswith("SELECT session_id FROM graph_runs"):
            run_id, user_id = tuple(params or ())
            if (
                self.state["run_id"] == run_id
                and self.state["user_id"] == user_id
            ):
                self._row = {"session_id": self.state["session_id"]}
            return
        if "pg_advisory_xact_lock" in normalized:
            return
        if not normalized.startswith("UPDATE graph_runs SET status = 'running'"):
            return

        values = list(params or ())
        new_execution_token, _updated_at, run_id, user_id = values[:4]
        value_index = 4
        checkpoint_matches = True
        interrupt_matches = True
        if "AND checkpoint_id = %s" in normalized:
            checkpoint_matches = self.state["checkpoint_id"] == values[value_index]
            value_index += 1
        if "AND interrupt_id = %s" in normalized:
            interrupt_matches = self.state["interrupt_id"] == values[value_index]
        if (
            self.state["run_id"] == run_id
            and self.state["user_id"] == user_id
            and self.state["status"] == "interrupted"
            and checkpoint_matches
            and interrupt_matches
        ):
            self.state["status"] = "running"
            self.state["execution_token"] = new_execution_token
            self.rowcount = 1

    def fetchone(self):
        return self._row


class PostgresCasConnection:
    def __init__(self, state, statements):
        self.state = state
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def cursor(self):
        return PostgresCasCursor(self.state, self.statements)

    def transaction(self):
        statements = self.statements

        class Transaction:
            def __enter__(self):
                statements.append(("BEGIN", ()))

            def __exit__(self, exc_type, exc, traceback):
                statements.append(
                    ("COMMIT" if exc_type is None else "ROLLBACK", ())
                )

        return Transaction()


class PostgresCasPool:
    def __init__(self):
        self.state = {
            "run_id": "run-pg",
            "user_id": "owner",
            "session_id": "session-pg",
            "checkpoint_id": "checkpoint-pg",
            "interrupt_id": "interrupt-pg",
            "execution_token": "worker-old",
            "status": "interrupted",
        }
        self.statements = []

    def connection(self):
        return PostgresCasConnection(self.state, self.statements)


class PostgresLeaseCasCursor:
    def __init__(self, state, statements):
        self.state = state
        self.statements = statements
        self.rowcount = 0
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, statement, params=None):
        normalized = " ".join(statement.split())
        values = tuple(params or ())
        self.statements.append((normalized, values))
        self.rowcount = 0
        self._row = None

        if normalized.startswith("SELECT user_id, session_id FROM graph_runs"):
            (run_id,) = values
            if self.state["run_id"] == run_id:
                self._row = {
                    "user_id": self.state["user_id"],
                    "session_id": self.state["session_id"],
                }
            return
        if normalized.startswith("SELECT session_id FROM graph_runs"):
            run_id, user_id = values
            if (
                self.state["run_id"] == run_id
                and self.state["user_id"] == user_id
            ):
                self._row = {"session_id": self.state["session_id"]}
            return
        if "pg_advisory_xact_lock" in normalized:
            return

        if normalized.startswith("SELECT * FROM graph_runs"):
            run_id, user_id = values
            if (
                self.state["run_id"] == run_id
                and self.state["user_id"] == user_id
            ):
                self._row = dict(self.state)
            return

        if "SET status = %s, updated_at = %s, finished_at = %s" in normalized:
            (
                status,
                updated_at,
                finished_at,
                error,
                current_node,
                checkpoint_id,
                interrupt_id,
                run_id,
            ) = values
            if self.state["run_id"] == run_id and not self.state["deleted"]:
                self.state.update(
                    status=status,
                    updated_at=updated_at,
                    finished_at=finished_at,
                    error=error,
                )
                if current_node is not None:
                    self.state["current_node"] = current_node
                if checkpoint_id is not None:
                    self.state["checkpoint_id"] = checkpoint_id
                if interrupt_id is not None:
                    self.state["interrupt_id"] = interrupt_id
                self.rowcount = 1
            return

        if "SET current_node = %s, updated_at = %s" in normalized:
            current_node, updated_at, run_id, user_id, token = values
            if self._owns_running(run_id, user_id, token):
                self.state.update(
                    current_node=current_node, updated_at=updated_at
                )
                self.rowcount = 1
            return

        if "SET status = %s, current_node = %s" in normalized:
            (
                status,
                current_node,
                updated_at,
                finished_at,
                error,
                checkpoint_id,
                interrupt_id,
                run_id,
                user_id,
                token,
            ) = values
            if self._owns_running(run_id, user_id, token):
                self.state.update(
                    status=status,
                    current_node=current_node,
                    updated_at=updated_at,
                    finished_at=finished_at,
                    error=error,
                )
                if checkpoint_id is not None:
                    self.state["checkpoint_id"] = checkpoint_id
                if interrupt_id is not None:
                    self.state["interrupt_id"] = interrupt_id
                self.rowcount = 1
            return

        if "status IN ('failed', 'running')" in normalized:
            new_token, updated_at, run_id, user_id, expected_updated_at = values
            if (
                self.state["run_id"] == run_id
                and self.state["user_id"] == user_id
                and self.state["status"] in {"failed", "running"}
                and self.state["updated_at"] == expected_updated_at
            ):
                self.state.update(
                    status="running",
                    execution_token=new_token,
                    updated_at=updated_at,
                    finished_at=None,
                    error="",
                )
                self.rowcount = 1

    def _owns_running(self, run_id, user_id, token):
        return (
            self.state["run_id"] == run_id
            and self.state["user_id"] == user_id
            and self.state["status"] == "running"
            and self.state["execution_token"] == token
            and not self.state.get("deleted", False)
        )

    def fetchone(self):
        return self._row


class PostgresLeaseCasConnection:
    def __init__(self, state, statements):
        self.state = state
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def cursor(self):
        return PostgresLeaseCasCursor(self.state, self.statements)

    def transaction(self):
        statements = self.statements

        class Transaction:
            def __enter__(self):
                statements.append(("BEGIN", ()))

            def __exit__(self, exc_type, exc, traceback):
                statements.append(
                    ("COMMIT" if exc_type is None else "ROLLBACK", ())
                )

        return Transaction()


class PostgresLeaseCasPool:
    def __init__(self):
        self.state = {
            "run_id": "run-lease-pg",
            "thread_id": "thread-lease-pg",
            "graph_name": "chat",
            "user_id": "owner",
            "session_id": "session-1",
            "request_id": "request-lease-pg",
            "idempotency_key": "request-lease-pg",
            "workflow_version": "workflow-v1",
            "current_node": "prepare",
            "checkpoint_id": "",
            "interrupt_id": "",
            "execution_token": "worker-old",
            "status": "running",
            "durability": "async",
            "started_at": "2026-08-09T10:00:00.000000+00:00",
            "updated_at": "2026-08-09T10:00:00.000000+00:00",
            "finished_at": None,
            "error": "",
            "metadata_json": "{}",
            "deleted": False,
        }
        self.statements = []

    def connection(self):
        return PostgresLeaseCasConnection(self.state, self.statements)


class PostgresEffectCasCursor:
    def __init__(self, state, statements):
        self.state = state
        self.statements = statements
        self.rowcount = 0
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, statement, params=None):
        normalized = " ".join(statement.split())
        values = tuple(params or ())
        self.statements.append((normalized, values))
        self.rowcount = 0
        self._row = None
        if normalized.startswith("SELECT session_id FROM graph_runs"):
            run_id, user_id = values
            if (
                self.state["run_id"] == run_id
                and self.state["user_id"] == user_id
            ):
                self._row = {"session_id": self.state["session_id"]}
            return
        if "pg_advisory_xact_lock" in normalized:
            return
        if normalized.startswith("SELECT * FROM graph_effects"):
            run_id, node, effect, user_id = values
            row = self.state["effects"].get((run_id, node, effect))
            if row is not None and row["user_id"] == user_id:
                self._row = dict(row)
            return

        if normalized.startswith("UPDATE graph_effects"):
            if "status = 'completed'" in normalized:
                result_json = values[0]
                error = ""
                new_status = "completed"
            elif "status = 'failed'" in normalized:
                result_json = None
                error = values[0]
                new_status = "failed"
            else:
                return
            (
                _payload,
                updated_at,
                finished_at,
                run_id,
                node,
                effect,
                user_id,
                run_user_id,
                fence_token,
                effect_token,
                run_token,
            ) = values
            row = self.state["effects"].get((run_id, node, effect))
            run_active = (
                self.state["run_id"] == run_id
                and self.state["user_id"] == run_user_id
                and self.state["status"] == "running"
                and not self.state["deleted"]
            )
            fence_matches = run_active and (
                fence_token == ""
                or (
                    row is not None
                    and row["execution_token"] == effect_token
                    and self.state["execution_token"] == run_token
                )
            )
            if (
                row is not None
                and row["user_id"] == user_id
                and row["status"] == "claimed"
                and fence_matches
            ):
                row.update(
                    status=new_status,
                    result_json=result_json,
                    error=error,
                    updated_at=updated_at,
                    finished_at=finished_at,
                )
                self.rowcount = 1
            return

        if not normalized.startswith("INSERT INTO graph_effects"):
            return
        (
            node,
            effect,
            token,
            claimed_at,
            updated_at,
            run_id,
            user_id,
            fence_token,
            run_token,
            *reclaim_values,
        ) = values
        key = (run_id, node, effect)
        run_matches = (
            run_id == self.state["run_id"]
            and user_id == self.state["user_id"]
            and not self.state["deleted"]
            and (
                fence_token == ""
                or (
                    self.state["status"] == "running"
                    and self.state["execution_token"] == run_token
                )
            )
        )
        if not run_matches:
            return
        existing = self.state["effects"].get(key)
        may_reclaim = False
        if existing is not None and "DO UPDATE SET" in normalized:
            stale_cutoff = reclaim_values[0] if reclaim_values else None
            may_reclaim = existing["status"] == "failed" or (
                existing["status"] == "claimed"
                and stale_cutoff is not None
                and existing["updated_at"] < stale_cutoff
            )
        if existing is None or may_reclaim:
            self.state["effects"][key] = {
                "run_id": run_id,
                "node": node,
                "effect": effect,
                "user_id": user_id,
                "execution_token": token,
                "status": "claimed",
                "result_json": None,
                "error": "",
                "claimed_at": claimed_at,
                "updated_at": updated_at,
                "finished_at": None,
            }
            self.rowcount = 1

    def fetchone(self):
        return self._row


class PostgresEffectCasConnection:
    def __init__(self, state, statements):
        self.state = state
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def cursor(self):
        return PostgresEffectCasCursor(self.state, self.statements)

    def transaction(self):
        statements = self.statements

        class Transaction:
            def __enter__(self):
                statements.append(("BEGIN", ()))

            def __exit__(self, exc_type, exc, traceback):
                statements.append(
                    ("COMMIT" if exc_type is None else "ROLLBACK", ())
                )

        return Transaction()


class PostgresEffectCasPool:
    def __init__(self):
        self.state = {
            "run_id": "run-effect-pg",
            "user_id": "owner",
            "session_id": "session-effect-pg",
            "status": "running",
            "execution_token": "worker-old",
            "deleted": False,
            "effects": {},
        }
        self.statements = []

    def connection(self):
        return PostgresEffectCasConnection(self.state, self.statements)


def test_postgres_ledger_setup_creates_owner_idempotency_and_tombstone_schema():
    pool = RecordingPool()

    PostgresGraphRunLedger(pool).setup()

    sql = "\n".join(statement for statement, _params in pool.statements)
    assert "CREATE TABLE IF NOT EXISTS graph_runs" in sql
    assert "user_id TEXT NOT NULL" in sql
    assert "idempotency_key TEXT NOT NULL" in sql
    assert "workflow_version TEXT NOT NULL" in sql
    assert "current_node TEXT NOT NULL" in sql
    assert "checkpoint_id TEXT NOT NULL" in sql
    assert "interrupt_id TEXT NOT NULL" in sql
    assert "execution_token TEXT NOT NULL" in sql
    assert "uq_graph_runs_idempotency" in sql
    assert "CREATE TABLE IF NOT EXISTS graph_thread_deletions" in sql
    assert "artifact_run_ids_json TEXT NOT NULL" in sql
    assert "CREATE TABLE IF NOT EXISTS graph_session_deletions" in sql
    assert "UNIQUE (user_id, session_id)" in sql
    assert "uq_graph_session_deletions_owner" in sql
    assert "CREATE TABLE IF NOT EXISTS graph_effects" in sql
    assert "PRIMARY KEY (run_id, node, effect)" in sql
    assert "uq_graph_effects_key" in sql


def test_postgres_schema_verification_includes_effect_ledger():
    pool = RecordingPool()

    PostgresGraphRunLedger(pool).verify_schema()

    sql = "\n".join(statement for statement, _params in pool.statements)
    assert "FROM checkpoint_migrations" in sql
    assert "MAX(v) AS latest_v" in sql
    assert "FROM checkpoints" in sql
    assert "FROM checkpoint_blobs" in sql
    assert "FROM checkpoint_writes" in sql
    assert "task_path" in sql
    assert "FROM graph_effects" in sql
    assert "result_json" in sql
    assert "finished_at" in sql
    assert "execution_token" in sql
    assert "artifact_run_ids_json" in sql
    assert "FROM graph_session_deletions" in sql
    assert "metadata_json" in sql
    assert "last_error" in sql
    assert "FROM pg_indexes" in sql


@pytest.mark.parametrize("failure_mode", ["missing", "nonunique"])
def test_postgres_schema_verification_rejects_missing_or_nonunique_indexes(
    failure_mode,
):
    class InvalidIndexCursor(RecordingCursor):
        def fetchall(self):
            rows = super().fetchall()
            if failure_mode == "missing":
                return [
                    row
                    for row in rows
                    if row["indexname"] != "uq_graph_runs_request"
                ]
            return [
                {
                    **row,
                    "indexdef": row["indexdef"].replace(
                        "CREATE UNIQUE INDEX", "CREATE INDEX"
                    ),
                }
                if row["indexname"] == "uq_graph_runs_idempotency"
                else row
                for row in rows
            ]

    class InvalidIndexConnection(RecordingConnection):
        def cursor(self):
            return InvalidIndexCursor(self.statements)

    class InvalidIndexPool(RecordingPool):
        def connection(self):
            return InvalidIndexConnection(self.statements)

    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="schema is missing or outdated",
    ):
        PostgresGraphRunLedger(InvalidIndexPool()).verify_schema()


def test_postgres_schema_verification_rejects_missing_critical_column():
    class MissingColumnCursor(RecordingCursor):
        def execute(self, statement, params=None):
            normalized = " ".join(statement.split())
            if "metadata_json" in normalized and "FROM graph_runs" in normalized:
                raise RuntimeError("column metadata_json does not exist")
            return super().execute(statement, params)

    class MissingColumnConnection(RecordingConnection):
        def cursor(self):
            return MissingColumnCursor(self.statements)

    class MissingColumnPool(RecordingPool):
        def connection(self):
            return MissingColumnConnection(self.statements)

    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="schema is missing or outdated",
    ):
        PostgresGraphRunLedger(MissingColumnPool()).verify_schema()


def test_postgres_schema_verification_rejects_outdated_official_migrations():
    class OutdatedCursor(RecordingCursor):
        def fetchone(self):
            from langgraph.checkpoint.postgres.base import MIGRATIONS

            return {"latest_v": len(MIGRATIONS) - 2}

    class OutdatedConnection(RecordingConnection):
        def cursor(self):
            return OutdatedCursor(self.statements)

    class OutdatedPool(RecordingPool):
        def connection(self):
            return OutdatedConnection(self.statements)

    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="schema is missing or outdated",
    ):
        PostgresGraphRunLedger(OutdatedPool()).verify_schema()


def test_postgres_claim_resume_is_an_atomic_compare_and_swap():
    pool = PostgresCasPool()
    ledger = PostgresGraphRunLedger(pool)

    assert (
        ledger.claim_resume(
            "run-pg",
            user_id="other",
            expected_checkpoint_id="checkpoint-pg",
            new_execution_token="worker-other",
        )
        is False
    )
    assert (
        ledger.claim_resume(
            "run-pg",
            user_id="owner",
            expected_checkpoint_id="checkpoint-pg",
            expected_interrupt_id="interrupt-pg",
            new_execution_token="worker-resume",
        )
        is True
    )
    assert (
        ledger.claim_resume(
            "run-pg",
            user_id="owner",
            expected_checkpoint_id="checkpoint-pg",
            expected_interrupt_id="interrupt-pg",
            new_execution_token="worker-duplicate",
        )
        is False
    )

    sql = "\n".join(statement for statement, _params in pool.statements)
    assert "SELECT session_id FROM graph_runs" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "BEGIN" in sql and "COMMIT" in sql
    assert "status = 'interrupted'" in sql
    assert "checkpoint_id = %s" in sql
    assert "interrupt_id = %s" in sql
    assert pool.state["status"] == "running"
    assert pool.state["execution_token"] == "worker-resume"


def test_postgres_execution_lease_sql_cas_rejects_stale_worker():
    pool = PostgresLeaseCasPool()
    ledger = PostgresGraphRunLedger(pool)
    heartbeat_at = datetime(2026, 8, 9, 10, 0, 1, tzinfo=timezone.utc)

    heartbeat = ledger.heartbeat_run(
        "run-lease-pg",
        user_id="owner",
        execution_token="worker-old",
        current_node="plan",
        updated_at=heartbeat_at,
    )
    assert heartbeat is not None
    expected_updated_at = heartbeat.updated_at
    assert (
        ledger.claim_recovery(
            "run-lease-pg",
            user_id="owner",
            expected_updated_at=expected_updated_at,
            new_execution_token="worker-new",
        )
        is True
    )
    assert (
        ledger.claim_recovery(
            "run-lease-pg",
            user_id="owner",
            expected_updated_at=expected_updated_at,
            new_execution_token="worker-duplicate",
        )
        is False
    )
    assert (
        ledger.finish_run(
            "run-lease-pg",
            user_id="owner",
            execution_token="worker-old",
            status="succeeded",
            current_node="__end__",
        )
        is None
    )
    new_heartbeat = ledger.heartbeat_run(
        "run-lease-pg",
        user_id="owner",
        execution_token="worker-new",
        current_node="execute_tools",
    )
    assert new_heartbeat is not None
    finished = ledger.finish_run(
        "run-lease-pg",
        user_id="owner",
        execution_token="worker-new",
        status="interrupted",
        current_node="approval",
        checkpoint_id="checkpoint-new",
        interrupt_id="interrupt-new",
    )
    assert finished is not None
    assert finished.status == "interrupted"
    assert finished.execution_token == "worker-new"
    assert finished.checkpoint_id == "checkpoint-new"

    sql = "\n".join(statement for statement, _params in pool.statements)
    assert "SELECT session_id FROM graph_runs" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "BEGIN" in sql and "COMMIT" in sql
    assert "AND status = 'running' AND execution_token = %s" in sql
    assert "status IN ('failed', 'running') AND updated_at = %s" in sql
    assert "SET status = %s, current_node = %s" in sql


def test_postgres_session_barrier_fences_run_and_effect_writes():
    lease_pool = PostgresLeaseCasPool()
    lease_ledger = PostgresGraphRunLedger(lease_pool)
    lease_pool.state["deleted"] = True

    assert lease_ledger.update_run(
        "run-lease-pg",
        status="succeeded",
        current_node="__end__",
    ) is None
    assert lease_ledger.heartbeat_run(
        "run-lease-pg",
        user_id="owner",
        execution_token="worker-old",
        current_node="execute_tools",
    ) is None
    assert lease_ledger.finish_run(
        "run-lease-pg",
        user_id="owner",
        execution_token="worker-old",
        status="succeeded",
        current_node="__end__",
    ) is None
    assert lease_pool.state["status"] == "running"

    effect_pool = PostgresEffectCasPool()
    effect_ledger = PostgresGraphRunLedger(effect_pool)
    for effect in ("message:complete", "message:fail"):
        assert effect_ledger.claim_effect(
            "run-effect-pg",
            "send",
            effect,
            user_id="owner",
            execution_token="worker-old",
        )
    effect_pool.state["deleted"] = True

    assert effect_ledger.claim_effect(
        "run-effect-pg",
        "send",
        "message:new",
        user_id="owner",
        execution_token="worker-old",
    ) is False
    assert effect_ledger.complete_effect(
        "run-effect-pg",
        "send",
        "message:complete",
        user_id="owner",
        execution_token="worker-old",
        result={"sent": True},
    ) is None
    assert effect_ledger.fail_effect(
        "run-effect-pg",
        "send",
        "message:fail",
        user_id="owner",
        execution_token="worker-old",
        error="must remain claimed",
    ) is None
    assert all(
        record["status"] == "claimed"
        for record in effect_pool.state["effects"].values()
    )

    statements = lease_pool.statements + effect_pool.statements
    guarded_writes = [
        statement
        for statement, _params in statements
        if statement.startswith(("UPDATE graph_runs", "INSERT INTO graph_effects", "UPDATE graph_effects"))
    ]
    assert guarded_writes
    assert all("graph_session_deletions" in statement for statement in guarded_writes)
    sql = "\n".join(statement for statement, _params in statements)
    assert "SELECT session_id FROM graph_runs" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "BEGIN" in sql and "COMMIT" in sql


def test_postgres_effect_claim_is_owner_scoped_and_atomic():
    pool = PostgresEffectCasPool()
    ledger = PostgresGraphRunLedger(pool)

    assert ledger.claim_effect(
        "run-effect-pg", "send", "message:1", user_id="other"
    ) is False
    assert ledger.claim_effect(
        "run-effect-pg", "send", "message:1", user_id="owner"
    ) is True
    assert ledger.claim_effect(
        "run-effect-pg", "send", "message:1", user_id="owner"
    ) is False

    sql = "\n".join(statement for statement, _params in pool.statements)
    assert "SELECT run_id" in sql
    assert "FROM graph_runs" in sql
    assert "WHERE run_id = %s AND user_id = %s" in sql
    assert "ON CONFLICT(run_id, node, effect) DO NOTHING" in sql


def test_postgres_effect_lease_fence_reclaims_only_failed_or_stale_claims():
    pool = PostgresEffectCasPool()
    ledger = PostgresGraphRunLedger(pool)
    claimed_at = datetime(2026, 8, 9, 9, 0, tzinfo=timezone.utc)

    assert ledger.claim_effect(
        "run-effect-pg",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-old",
        claimed_at=claimed_at,
    )
    pool.state["execution_token"] = "worker-new"

    assert ledger.complete_effect(
        "run-effect-pg",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-old",
        result={"stale": True},
    ) is None
    assert ledger.claim_effect(
        "run-effect-pg",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-new",
        allow_reclaim=True,
        stale_before=claimed_at,
    ) is False
    assert ledger.claim_effect(
        "run-effect-pg",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-new",
        allow_reclaim=True,
        stale_before=claimed_at + timedelta(seconds=1),
    ) is True
    completed = ledger.complete_effect(
        "run-effect-pg",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-new",
        result={"message_id": "message-new"},
    )
    assert completed is not None
    assert completed.status == "completed"
    assert completed.execution_token == "worker-new"
    assert ledger.claim_effect(
        "run-effect-pg",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-new",
        allow_reclaim=True,
        stale_before=claimed_at + timedelta(days=1),
    ) is False

    assert ledger.claim_effect(
        "run-effect-pg",
        "charge",
        "billing:fenced",
        user_id="owner",
        execution_token="worker-new",
    )
    failed = ledger.fail_effect(
        "run-effect-pg",
        "charge",
        "billing:fenced",
        user_id="owner",
        execution_token="worker-new",
        error="provider timeout",
    )
    assert failed is not None
    assert failed.status == "failed"
    pool.state["execution_token"] = "worker-newer"
    assert ledger.claim_effect(
        "run-effect-pg",
        "charge",
        "billing:fenced",
        user_id="owner",
        execution_token="worker-newer",
        allow_reclaim=True,
    )

    sql = "\n".join(statement for statement, _params in pool.statements)
    assert "status = 'running' AND execution_token = %s" in sql
    assert "graph_effects.execution_token = %s" in sql
    assert "graph_effects.status = 'failed'" in sql
    assert "graph_effects.status = 'claimed'" in sql


def test_setup_script_calls_full_postgres_setup(tmp_path, monkeypatch, capsys):
    from scripts import setup_langgraph_checkpointer as setup_script

    config = make_config(
        tmp_path,
        backend="postgres",
        environment="production",
        dsn="postgresql://secret@example.invalid/graph",
    )
    calls = []

    class FakeConfigLoader:
        @classmethod
        def from_env(cls):
            return config

    monkeypatch.setattr(setup_script, "GraphPersistenceConfig", FakeConfigLoader)
    monkeypatch.setattr(
        setup_script,
        "setup_postgres_persistence",
        lambda resolved: calls.append(resolved),
    )

    exit_code = setup_script.main(
        ["--env-file", str(tmp_path / "missing.env")]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [config]
    assert "graph_runs schemas are ready" in captured.out
    assert "secret" not in captured.out


def test_production_factory_uses_shared_postgres_run_ledger(tmp_path, monkeypatch):
    FakePool.instances.clear()
    FakePostgresSaver.instances.clear()
    monkeypatch.setattr(
        persistence,
        "_load_postgres_dependencies",
        lambda: (FakePostgresSaver, FakePool, object()),
    )
    monkeypatch.setattr(PostgresGraphRunLedger, "verify_schema", lambda self: None)
    config = make_config(
        tmp_path,
        backend="postgres",
        environment="production",
        dsn="postgresql://secret@example.invalid/graph",
    )

    runtime = persistence.create_graph_persistence(config)

    assert isinstance(runtime.run_ledger, PostgresGraphRunLedger)
    assert runtime.run_ledger.pool is runtime.checkpointer_handle.resource
    runtime.close()


def test_sqlite_run_ledger_records_and_updates_runs(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    started_at = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)

    started = ledger.start_run(
        run_id="run-1",
        thread_id="thread-1",
        graph_name="chat",
        current_node="prepare",
        durability="async",
        metadata={"route": "knowledge_qa"},
        started_at=started_at,
        **run_identity("1"),
    )
    finished = ledger.update_run(
        "run-1",
        status="succeeded",
        current_node="__end__",
        updated_at=started_at + timedelta(minutes=2),
    )

    assert started.status == "running"
    assert started.metadata == {"route": "knowledge_qa"}
    assert started.user_id == "user-1"
    assert started.session_id == "session-1"
    assert started.request_id == "request-1"
    assert started.idempotency_key == "request-1"
    assert started.workflow_version == "workflow-v1"
    assert started.current_node == "prepare"
    assert started.execution_token
    assert finished is not None
    assert finished.status == "succeeded"
    assert finished.current_node == "__end__"
    assert finished.finished_at == "2026-08-01T09:02:00.000000+00:00"
    assert ledger.list_thread_runs("thread-1") == [finished]


def test_sqlite_runtime_claim_resume_is_owner_token_scoped_and_exactly_once(
    tmp_path,
):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    ledger.start_run(
        run_id="run-resume",
        thread_id="thread-resume",
        graph_name="chat",
        **run_identity("resume", user_id="owner"),
    )
    interrupted = ledger.update_run(
        "run-resume",
        status="interrupted",
        current_node="approval",
        checkpoint_id="checkpoint-1",
        interrupt_id="interrupt-1",
    )
    runtime = GraphPersistenceRuntime(
        config=make_config(tmp_path),
        checkpointer_handle=CheckpointerHandle(
            saver=FakeDeletingSaver(), backend="fake"
        ),
        run_ledger=ledger,
    )

    assert interrupted is not None
    assert interrupted.checkpoint_id == "checkpoint-1"
    assert interrupted.interrupt_id == "interrupt-1"
    with pytest.raises(ValueError, match="requires expected_checkpoint_id"):
        runtime.claim_resume("run-resume", user_id="owner")
    assert (
        runtime.claim_resume(
            "run-resume",
            user_id="other",
            expected_checkpoint_id="checkpoint-1",
        )
        is False
    )
    assert (
        runtime.claim_resume(
            "run-resume",
            user_id="owner",
            expected_checkpoint_id="wrong-checkpoint",
        )
        is False
    )
    assert (
        runtime.claim_resume(
            "run-resume",
            user_id="owner",
            expected_interrupt_id="wrong-interrupt",
        )
        is False
    )
    assert (
        runtime.claim_resume(
            "run-resume",
            user_id="owner",
            expected_checkpoint_id="checkpoint-1",
            new_execution_token="worker-resume",
        )
        is True
    )
    assert (
        runtime.claim_resume(
            "run-resume",
            user_id="owner",
            expected_checkpoint_id="checkpoint-1",
        )
        is False
    )
    claimed = ledger.get_run("run-resume")
    assert claimed is not None
    assert claimed.status == "running"
    assert claimed.finished_at is None
    assert claimed.checkpoint_id == "checkpoint-1"
    assert claimed.interrupt_id == "interrupt-1"
    assert claimed.execution_token == "worker-resume"

    ledger.start_run(
        run_id="run-resume-interrupt",
        thread_id="thread-resume-interrupt",
        graph_name="chat",
        **run_identity("resume-interrupt", user_id="owner"),
    )
    ledger.update_run(
        "run-resume-interrupt",
        status="interrupted",
        checkpoint_id="checkpoint-2",
        interrupt_id="interrupt-2",
    )
    assert (
        runtime.claim_resume(
            "run-resume-interrupt",
            user_id="owner",
            expected_interrupt_id="interrupt-2",
            new_execution_token="worker-resume-2",
        )
        is True
    )


def test_sqlite_execution_lease_invalidates_old_worker_after_recovery(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    initial = datetime(2026, 8, 9, 10, 0, tzinfo=timezone.utc)
    started = ledger.start_run(
        run_id="run-lease",
        thread_id="thread-lease",
        graph_name="chat",
        execution_token="worker-old",
        started_at=initial,
        **run_identity("lease", user_id="owner"),
    )
    heartbeat = ledger.heartbeat_run(
        "run-lease",
        user_id="owner",
        execution_token="worker-old",
        current_node="plan",
        updated_at=initial + timedelta(seconds=1),
    )

    assert started.execution_token == "worker-old"
    assert heartbeat is not None
    assert heartbeat.current_node == "plan"
    expected_updated_at = heartbeat.updated_at
    assert (
        ledger.claim_recovery(
            "run-lease",
            user_id="other",
            expected_updated_at=expected_updated_at,
            new_execution_token="worker-other",
        )
        is False
    )
    assert (
        ledger.claim_recovery(
            "run-lease",
            user_id="owner",
            expected_updated_at=expected_updated_at,
            new_execution_token="worker-new",
        )
        is True
    )
    assert (
        ledger.claim_recovery(
            "run-lease",
            user_id="owner",
            expected_updated_at=expected_updated_at,
            new_execution_token="worker-duplicate",
        )
        is False
    )

    assert (
        ledger.heartbeat_run(
            "run-lease",
            user_id="owner",
            execution_token="worker-old",
            current_node="stale-node",
        )
        is None
    )
    assert (
        ledger.finish_run(
            "run-lease",
            user_id="owner",
            execution_token="worker-old",
            status="succeeded",
            current_node="__end__",
        )
        is None
    )
    new_heartbeat = ledger.heartbeat_run(
        "run-lease",
        user_id="owner",
        execution_token="worker-new",
        current_node="execute_tools",
    )
    assert new_heartbeat is not None
    finished = ledger.finish_run(
        "run-lease",
        user_id="owner",
        execution_token="worker-new",
        status="succeeded",
        current_node="__end__",
    )
    assert finished is not None
    assert finished.status == "succeeded"
    assert finished.execution_token == "worker-new"


def test_sqlite_forward_migrates_execution_token_column(tmp_path):
    path = tmp_path / "old-runs.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE graph_runs (
                run_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL,
                graph_name TEXT NOT NULL, user_id TEXT NOT NULL,
                session_id TEXT NOT NULL, request_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL, workflow_version TEXT NOT NULL,
                current_node TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL DEFAULT '',
                interrupt_id TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
                durability TEXT NOT NULL, started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, finished_at TEXT,
                error TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE graph_thread_deletions (
                deletion_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT NOT NULL DEFAULT ''
            )
            """
        )

    ledger = SqliteGraphRunLedger(path)
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(graph_runs)")
        }
        deletion_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(graph_thread_deletions)"
            )
        }
        session_deletion_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(graph_session_deletions)"
            )
        }
        session_deletion_indexes = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list(graph_session_deletions)"
            )
        }

    assert "execution_token" in columns
    assert "artifact_run_ids_json" in deletion_columns
    assert {
        "deletion_id",
        "user_id",
        "session_id",
        "reason",
        "requested_at",
        "updated_at",
    }.issubset(session_deletion_columns)
    assert "uq_graph_session_deletions_owner" in session_deletion_indexes
    record = ledger.start_run(
        run_id="run-migrated",
        thread_id="thread-migrated",
        graph_name="chat",
        execution_token="worker-migrated",
        **run_identity("migrated"),
    )
    assert record.execution_token == "worker-migrated"


def test_sqlite_failed_run_can_be_claimed_for_recovery(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    ledger.start_run(
        run_id="run-failed-recovery",
        thread_id="thread-failed-recovery",
        graph_name="chat",
        execution_token="worker-failed",
        **run_identity("failed-recovery", user_id="owner"),
    )
    failed = ledger.finish_run(
        "run-failed-recovery",
        user_id="owner",
        execution_token="worker-failed",
        status="failed",
        current_node="error",
        error="worker crashed",
    )
    assert failed is not None
    assert failed.finished_at is not None

    assert ledger.claim_recovery(
        "run-failed-recovery",
        user_id="owner",
        expected_updated_at=failed.updated_at,
        new_execution_token="worker-recovered",
    )
    recovered = ledger.get_run("run-failed-recovery")
    assert recovered is not None
    assert recovered.status == "running"
    assert recovered.execution_token == "worker-recovered"
    assert recovered.finished_at is None
    assert recovered.error == ""


def test_sqlite_run_ledger_rejects_non_json_metadata(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")

    with pytest.raises(ValueError, match="JSON serializable"):
        ledger.start_run(
            run_id="run-1",
            thread_id="thread-1",
            graph_name="chat",
            metadata={"bad": object()},
            **run_identity("1"),
        )

    assert ledger.get_run("run-1") is None


def test_sqlite_run_ledger_enforces_idempotency_and_owner_queries(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    first = ledger.start_run(
        run_id="run-original",
        thread_id="thread-owned",
        graph_name="chat",
        idempotency_key="idem-1",
        **run_identity("original", user_id="owner", session_id="session-owned"),
    )
    replay = ledger.start_run(
        run_id="run-replay",
        thread_id="thread-other",
        graph_name="chat",
        idempotency_key="idem-1",
        **run_identity("replay", user_id="owner", session_id="session-owned"),
    )

    assert replay == first
    assert ledger.get_owned_run("run-original", user_id="owner") == first
    assert ledger.get_owned_run("run-original", user_id="other") is None
    assert ledger.get_run_by_idempotency_key("owner", "chat", "idem-1") == first
    assert ledger.list_session_runs("session-owned", user_id="owner") == [first]
    assert ledger.thread_belongs_to_user("thread-owned", user_id="owner") is True
    assert ledger.thread_belongs_to_user("thread-owned", user_id="other") is False


def test_sqlite_run_ledger_rejects_replayed_payload_digest_mismatch(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    first = ledger.start_run(
        run_id="run-digest-original",
        thread_id="thread-digest-original",
        graph_name="chat",
        idempotency_key="idem-digest",
        metadata={"request_digest": "sha256:first"},
        **run_identity("digest-original", user_id="owner"),
    )

    replay = ledger.start_run(
        run_id="run-digest-replay",
        thread_id="thread-digest-replay",
        graph_name="chat",
        idempotency_key="idem-digest",
        metadata={"request_digest": "sha256:first"},
        **run_identity("digest-replay", user_id="owner"),
    )
    assert replay == first

    with pytest.raises(GraphRunIdentityConflictError) as exc_info:
        ledger.start_run(
            run_id="run-digest-conflict",
            thread_id="thread-digest-conflict",
            graph_name="chat",
            idempotency_key="idem-digest",
            metadata={"request_digest": "sha256:different"},
            **run_identity("digest-conflict", user_id="owner"),
        )

    assert exc_info.value.conflict_type == "request_digest_mismatch"
    assert exc_info.value.existing_run_id == first.run_id


def test_sqlite_run_ledger_reports_independent_identity_conflicts(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    common = {
        "graph_name": "chat",
        "user_id": "owner",
        "session_id": "session-identities",
        "workflow_version": "workflow-v1",
    }
    first = ledger.start_run(
        run_id="run-a",
        thread_id="thread-a",
        request_id="request-a",
        idempotency_key="idem-a",
        **common,
    )
    second = ledger.start_run(
        run_id="run-b",
        thread_id="thread-b",
        request_id="request-b",
        idempotency_key="idem-b",
        **common,
    )

    with pytest.raises(GraphRunIdentityConflictError) as dual_exc:
        ledger.start_run(
            run_id="run-dual-conflict",
            thread_id="thread-dual-conflict",
            request_id="request-a",
            idempotency_key="idem-b",
            **common,
        )
    assert dual_exc.value.conflict_type == (
        "request_id_idempotency_key_mismatch"
    )
    assert dual_exc.value.existing_run_id == first.run_id

    with pytest.raises(GraphRunIdentityConflictError) as request_exc:
        ledger.start_run(
            run_id="run-request-conflict",
            thread_id="thread-request-conflict",
            request_id="request-a",
            idempotency_key="idem-new",
            **common,
        )
    assert request_exc.value.conflict_type == "request_id_conflict"
    assert request_exc.value.existing_run_id == first.run_id

    with pytest.raises(GraphRunIdentityConflictError) as run_exc:
        ledger.start_run(
            run_id="run-a",
            thread_id="thread-run-conflict",
            request_id="request-new",
            idempotency_key="idem-b",
            **common,
        )
    assert run_exc.value.conflict_type == "run_id_identity_mismatch"
    assert run_exc.value.existing_run_id == first.run_id
    assert ledger.get_run("run-b") == second


def test_sqlite_session_deletion_barrier_is_permanent_and_owner_scoped(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    first_barrier = ledger.enqueue_session_deletion(
        "session-deleted",
        user_id="owner",
        reason="user_delete",
    )
    repeated_barrier = ledger.enqueue_session_deletion(
        "session-deleted",
        user_id="owner",
        reason="retry_delete",
    )

    assert repeated_barrier.deletion_id == first_barrier.deletion_id
    assert repeated_barrier.reason == "retry_delete"
    assert ledger.is_session_deletion_requested(
        "session-deleted", user_id="owner"
    )
    with pytest.raises(GraphSessionDeletionRequestedError):
        ledger.start_run(
            run_id="run-rejected",
            thread_id="thread-rejected",
            graph_name="chat",
            **run_identity(
                "rejected", user_id="owner", session_id="session-deleted"
            ),
        )

    other_owner = ledger.start_run(
        run_id="run-other-owner",
        thread_id="thread-other-owner",
        graph_name="chat",
        **run_identity(
            "other-owner", user_id="other", session_id="session-deleted"
        ),
    )
    replacement_session = ledger.start_run(
        run_id="run-new-session",
        thread_id="thread-new-session",
        graph_name="chat",
        **run_identity(
            "new-session", user_id="owner", session_id="session-replacement"
        ),
    )
    assert other_owner.user_id == "other"
    assert replacement_session.session_id == "session-replacement"


def test_sqlite_session_deletion_barrier_rejects_recovery_and_resume(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    failed = ledger.start_run(
        run_id="run-deleted-recovery",
        thread_id="run-deleted-recovery",
        graph_name="chat",
        execution_token="worker-failed",
        **run_identity("deleted-recovery", user_id="owner", session_id="deleted"),
    )
    ledger.finish_run(
        failed.run_id,
        user_id="owner",
        execution_token=failed.execution_token,
        status="failed",
        current_node="error",
    )
    interrupted = ledger.start_run(
        run_id="run-deleted-resume",
        thread_id="run-deleted-resume",
        graph_name="chat",
        execution_token="worker-interrupted",
        **run_identity("deleted-resume", user_id="owner", session_id="deleted"),
    )
    ledger.finish_run(
        interrupted.run_id,
        user_id="owner",
        execution_token=interrupted.execution_token,
        status="interrupted",
        current_node="interrupt",
        checkpoint_id="checkpoint-deleted",
        interrupt_id="interrupt-deleted",
    )
    failed_record = ledger.get_run(failed.run_id)
    ledger.enqueue_session_deletion("deleted", user_id="owner", reason="user_delete")

    assert failed_record is not None
    assert ledger.claim_recovery(
        failed.run_id,
        user_id="owner",
        expected_updated_at=failed_record.updated_at,
        new_execution_token="worker-recovery",
    ) is False
    assert ledger.claim_resume(
        interrupted.run_id,
        user_id="owner",
        expected_checkpoint_id="checkpoint-deleted",
        expected_interrupt_id="interrupt-deleted",
        new_execution_token="worker-resume",
    ) is False


def test_pending_session_barrier_rebuilds_thread_tombstones_after_failure(tmp_path):
    class FailEnumerationOnceLedger(SqliteGraphRunLedger):
        fail_once = True

        def list_session_thread_ids(self, session_id, *, user_id):
            if self.fail_once:
                self.fail_once = False
                raise OSError("worker stopped after barrier")
            return super().list_session_thread_ids(session_id, user_id=user_id)

    ledger = FailEnumerationOnceLedger(tmp_path / "runs.sqlite")
    ledger.start_run(
        run_id="run-rebuild",
        thread_id="run-rebuild",
        graph_name="chat",
        **run_identity("rebuild", user_id="owner", session_id="session-rebuild"),
    )
    runtime = GraphPersistenceRuntime(
        config=make_config(tmp_path),
        checkpointer_handle=CheckpointerHandle(
            saver=FakeDeletingSaver(), backend="fake"
        ),
        run_ledger=ledger,
    )

    first = runtime.request_session_deletion(
        "session-rebuild",
        user_id="owner",
    )
    assert set(first.failures) == {"session:session-rebuild"}
    assert len(ledger.list_pending_session_deletions()) == 1
    assert ledger.get_run("run-rebuild") is not None

    retried = runtime.retry_pending_deletions()
    assert retried.deleted_thread_ids == ("run-rebuild",)
    assert ledger.list_pending_session_deletions() == []
    assert ledger.get_run("run-rebuild") is None


def test_sqlite_session_barrier_fences_run_and_effect_writes(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    run = ledger.start_run(
        run_id="run-write-fence",
        thread_id="run-write-fence",
        graph_name="chat",
        execution_token="worker-fenced",
        **run_identity(
            "write-fence",
            user_id="owner",
            session_id="session-write-fence",
        ),
    )
    for effect in ("message:complete", "message:fail"):
        assert ledger.claim_effect(
            run.run_id,
            "send",
            effect,
            user_id="owner",
            execution_token=run.execution_token,
        )

    ledger.enqueue_session_deletion(
        "session-write-fence",
        user_id="owner",
        reason="user_delete",
    )

    assert ledger.update_run(
        run.run_id,
        status="succeeded",
        current_node="__end__",
    ) is None
    assert ledger.heartbeat_run(
        run.run_id,
        user_id="owner",
        execution_token=run.execution_token,
        current_node="execute_tools",
    ) is None
    assert ledger.finish_run(
        run.run_id,
        user_id="owner",
        execution_token=run.execution_token,
        status="succeeded",
        current_node="__end__",
    ) is None
    assert ledger.claim_effect(
        run.run_id,
        "send",
        "message:new",
        user_id="owner",
        execution_token=run.execution_token,
    ) is False
    assert ledger.complete_effect(
        run.run_id,
        "send",
        "message:complete",
        user_id="owner",
        execution_token=run.execution_token,
        result={"sent": True},
    ) is None
    assert ledger.fail_effect(
        run.run_id,
        "send",
        "message:fail",
        user_id="owner",
        execution_token=run.execution_token,
        error="must remain claimed",
    ) is None

    current = ledger.get_run(run.run_id)
    assert current is not None
    assert current.status == "running"
    assert current.current_node == ""
    assert {
        item.effect: item.status
        for item in ledger.list_run_effects(run.run_id, user_id="owner")
    } == {
        "message:complete": "claimed",
        "message:fail": "claimed",
    }


def test_session_delete_barrier_blocks_concurrent_start_before_enumeration(
    tmp_path,
):
    enumeration_started = threading.Event()
    release_enumeration = threading.Event()

    class CoordinatedLedger(SqliteGraphRunLedger):
        def list_session_thread_ids(self, session_id, *, user_id):
            assert self.is_session_deletion_requested(
                session_id, user_id=user_id
            )
            enumeration_started.set()
            if not release_enumeration.wait(timeout=5):
                raise AssertionError("test did not release session enumeration")
            return super().list_session_thread_ids(
                session_id, user_id=user_id
            )

    ledger = CoordinatedLedger(tmp_path / "runs.sqlite")
    ledger.start_run(
        run_id="run-before-delete",
        thread_id="thread-before-delete",
        graph_name="chat",
        **run_identity(
            "before-delete", user_id="owner", session_id="session-race"
        ),
    )
    runtime = GraphPersistenceRuntime(
        config=make_config(tmp_path),
        checkpointer_handle=CheckpointerHandle(
            saver=FakeDeletingSaver(), backend="fake"
        ),
        run_ledger=ledger,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        deletion = executor.submit(
            runtime.request_session_deletion,
            "session-race",
            user_id="owner",
        )
        assert enumeration_started.wait(timeout=5)
        try:
            with pytest.raises(GraphSessionDeletionRequestedError):
                ledger.start_run(
                    run_id="run-during-delete",
                    thread_id="thread-during-delete",
                    graph_name="chat",
                    **run_identity(
                        "during-delete",
                        user_id="owner",
                        session_id="session-race",
                    ),
                )
        finally:
            release_enumeration.set()
        result = deletion.result(timeout=5)

    assert result.requested_thread_ids == ("thread-before-delete",)
    assert ledger.get_run("run-before-delete") is None
    assert ledger.get_run("run-during-delete") is None


def test_runtime_owned_thread_delete_rejects_other_users(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    ledger.start_run(
        run_id="run-owned",
        thread_id="thread-owned",
        graph_name="chat",
        **run_identity("owned", user_id="owner"),
    )
    saver = FakeDeletingSaver()
    runtime = GraphPersistenceRuntime(
        config=make_config(tmp_path),
        checkpointer_handle=CheckpointerHandle(saver=saver, backend="fake"),
        run_ledger=ledger,
    )

    with pytest.raises(GraphPersistenceOwnershipError):
        runtime.delete_owned_thread("thread-owned", user_id="other")

    assert ledger.get_run("run-owned") is not None
    assert runtime.delete_owned_thread("thread-owned", user_id="owner") == 1
    assert saver.deleted == ["thread-owned"]


def test_session_delete_persists_tombstone_and_retries_failures(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    for suffix in ("a", "b"):
        ledger.start_run(
            run_id=f"run-{suffix}",
            thread_id=f"thread-{suffix}",
            graph_name="chat",
            **run_identity(
                suffix, user_id="owner", session_id="session-delete"
            ),
        )
    saver = FakeDeletingSaver(failing_threads={"thread-b"})
    runtime = GraphPersistenceRuntime(
        config=make_config(tmp_path),
        checkpointer_handle=CheckpointerHandle(saver=saver, backend="fake"),
        run_ledger=ledger,
    )
    artifact_failures = {"run-b"}
    deleted_artifacts = []

    def delete_artifact(run_id):
        if run_id in artifact_failures:
            raise OSError("temporary artifact lock")
        deleted_artifacts.append(run_id)
        return True

    first = runtime.request_session_deletion(
        "session-delete",
        user_id="owner",
        delete_artifact_run=delete_artifact,
    )

    assert first.requested_thread_ids == ("thread-a", "thread-b")
    assert first.deleted_thread_ids == ("thread-a",)
    assert set(first.failures) == {"thread-b"}
    pending = ledger.list_pending_deletions()
    assert [item.thread_id for item in pending] == ["thread-b"]
    assert pending[0].artifact_run_ids == ("run-b",)
    assert "temporary checkpoint failure" in pending[0].last_error
    assert deleted_artifacts == ["run-a"]
    assert runtime.is_thread_deletion_pending("thread-b", user_id="owner") is True
    assert runtime.is_thread_deletion_pending("thread-b", user_id="other") is False
    assert runtime.is_thread_deletion_requested("thread-b", user_id="owner") is True
    reapplied = ledger.enqueue_thread_deletion(
        "thread-b",
        user_id="owner",
        session_id="session-delete",
        reason="active_run_cancelled",
    )
    assert reapplied.artifact_run_ids == ("run-b",)

    saver.failing_threads.clear()
    missing_callback = runtime.retry_pending_deletions()

    assert missing_callback.deleted_thread_ids == ()
    assert "Artifact deletion callback is required" in missing_callback.failures[
        "thread-b"
    ]
    assert ledger.get_run("run-b") is None
    assert runtime.is_thread_deletion_pending("thread-b", user_id="owner") is True

    artifact_retry = runtime.retry_pending_deletions(
        delete_artifact_run=delete_artifact
    )

    assert artifact_retry.deleted_thread_ids == ()
    assert set(artifact_retry.failures) == {"thread-b"}
    assert "temporary artifact lock" in artifact_retry.failures["thread-b"]
    pending = ledger.list_pending_deletions()
    assert [item.thread_id for item in pending] == ["thread-b"]
    assert pending[0].artifact_run_ids == ("run-b",)
    assert "temporary artifact lock" in pending[0].last_error

    artifact_failures.clear()
    retried = runtime.retry_pending_deletions(
        delete_artifact_run=delete_artifact
    )

    assert retried.deleted_thread_ids == ("thread-b",)
    assert retried.failures == {}
    assert ledger.list_pending_deletions() == []
    assert runtime.is_thread_deletion_pending("thread-b", user_id="owner") is False
    assert runtime.is_thread_deletion_requested("thread-b", user_id="owner") is True
    assert deleted_artifacts == ["run-a", "run-b"]


class FakeDeletingSaver:
    def __init__(self, *, failing_threads=()):
        self.deleted = []
        self.failing_threads = set(failing_threads)

    def delete_thread(self, thread_id):
        if thread_id in self.failing_threads:
            raise RuntimeError("temporary checkpoint failure")
        self.deleted.append(thread_id)


def test_retention_deletes_whole_threads_and_keeps_failed_rows_for_retry(tmp_path):
    now = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    for run_id, thread_id, age_days in (
        ("run-old", "thread-old", 100),
        ("run-fail", "thread-fail", 95),
        ("run-new", "thread-new", 1),
    ):
        ledger.start_run(
            run_id=run_id,
            thread_id=thread_id,
            graph_name="chat",
            started_at=now - timedelta(days=age_days),
            **run_identity(run_id, session_id=thread_id),
        )

    saver = FakeDeletingSaver(failing_threads={"thread-fail"})
    runtime = GraphPersistenceRuntime(
        config=make_config(tmp_path),
        checkpointer_handle=CheckpointerHandle(saver=saver, backend="fake"),
        run_ledger=ledger,
    )

    result = runtime.delete_expired_threads(now=now, retention_days=90)

    assert result.candidate_count == 2
    assert result.deleted_thread_ids == ("thread-old",)
    assert set(result.failures) == {"thread-fail"}
    assert saver.deleted == ["thread-old"]
    assert ledger.get_run("run-old") is None
    assert ledger.get_run("run-fail") is not None
    assert ledger.get_run("run-new") is not None


def test_delete_thread_does_not_drop_retry_ledger_when_checkpoint_delete_fails(
    tmp_path,
):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    ledger.start_run(
        run_id="run-1",
        thread_id="thread-1",
        graph_name="document",
        **run_identity("1"),
    )
    runtime = GraphPersistenceRuntime(
        config=make_config(tmp_path),
        checkpointer_handle=CheckpointerHandle(
            saver=FakeDeletingSaver(failing_threads={"thread-1"}),
            backend="fake",
        ),
        run_ledger=ledger,
    )

    with pytest.raises(RuntimeError, match="temporary checkpoint failure"):
        runtime.delete_thread("thread-1")

    assert ledger.get_run("run-1") is not None


def test_runtime_run_helpers_use_configured_default_durability(tmp_path):
    config = GraphPersistenceConfig(
        backend="memory",
        sqlite_path=tmp_path / "checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "runs.sqlite",
        default_durability="sync",
    )
    ledger = SqliteGraphRunLedger(config.run_ledger_sqlite_path)
    runtime = GraphPersistenceRuntime(
        config=config,
        checkpointer_handle=CheckpointerHandle(
            saver=FakeDeletingSaver(), backend="fake"
        ),
        run_ledger=ledger,
    )

    started = runtime.start_run(
        run_id="run-sync",
        thread_id="thread-sync",
        graph_name="document",
        execution_token="worker-runtime",
        **run_identity("sync"),
    )
    heartbeat = runtime.heartbeat_run(
        "run-sync",
        user_id="user-1",
        execution_token="worker-runtime",
        current_node="write",
    )
    finished = runtime.finish_run(
        "run-sync",
        user_id="user-1",
        execution_token="worker-runtime",
        status="failed",
        current_node="error",
        error="boom",
    )

    assert started.durability == "sync"
    assert heartbeat is not None
    assert heartbeat.current_node == "write"
    assert finished is not None
    assert finished.status == "failed"
    assert finished.error == "boom"


def test_sqlite_effect_ledger_claim_complete_fail_and_owner_isolation(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    ledger.start_run(
        run_id="run-effects",
        thread_id="thread-effects",
        graph_name="chat",
        **run_identity("effects", user_id="owner"),
    )
    claimed_at = datetime(2026, 8, 9, 9, 0, tzinfo=timezone.utc)
    finished_at = claimed_at + timedelta(seconds=5)

    assert (
        ledger.claim_effect(
            "run-effects",
            "send_message",
            "message:primary",
            user_id="other",
            claimed_at=claimed_at,
        )
        is False
    )
    assert ledger.get_effect(
        "run-effects", "send_message", "message:primary", user_id="other"
    ) is None
    assert (
        ledger.claim_effect(
            "run-effects",
            "send_message",
            "message:primary",
            user_id="owner",
            claimed_at=claimed_at,
        )
        is True
    )
    assert (
        ledger.claim_effect(
            "run-effects",
            "send_message",
            "message:primary",
            user_id="owner",
        )
        is False
    )

    claimed = ledger.get_effect(
        "run-effects", "send_message", "message:primary", user_id="owner"
    )
    assert claimed is not None
    assert claimed.status == "claimed"
    assert claimed.result is None
    assert claimed.claimed_at == "2026-08-09T09:00:00.000000+00:00"
    assert claimed.finished_at is None

    completed = ledger.complete_effect(
        "run-effects",
        "send_message",
        "message:primary",
        user_id="owner",
        result={"message_id": "message-1", "parts": [1, 2]},
        updated_at=finished_at,
    )
    assert completed is not None
    assert completed.status == "completed"
    assert completed.result == {"message_id": "message-1", "parts": [1, 2]}
    assert completed.error == ""
    assert completed.finished_at == "2026-08-09T09:00:05.000000+00:00"

    assert ledger.claim_effect(
        "run-effects", "charge", "billing:primary", user_id="owner"
    )
    failed = ledger.fail_effect(
        "run-effects",
        "charge",
        "billing:primary",
        user_id="owner",
        error="provider timeout",
    )
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error == "provider timeout"
    assert failed.result is None

    effects = ledger.list_run_effects("run-effects", user_id="owner")
    assert [(item.node, item.effect, item.status) for item in effects] == [
        ("send_message", "message:primary", "completed"),
        ("charge", "billing:primary", "failed"),
    ]
    assert ledger.list_run_effects("run-effects", user_id="other") == []


def test_sqlite_effect_claim_is_atomic_across_connections(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    ledger.start_run(
        run_id="run-race",
        thread_id="thread-race",
        graph_name="chat",
        **run_identity("race", user_id="owner"),
    )

    def claim_once(_attempt):
        return ledger.claim_effect(
            "run-race", "external_action", "provider:request", user_id="owner"
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(claim_once, range(24)))

    assert results.count(True) == 1
    assert results.count(False) == 23


def test_sqlite_effect_lease_fence_rejects_old_worker_and_controls_reclaim(
    tmp_path,
):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    run = ledger.start_run(
        run_id="run-effect-fence",
        thread_id="thread-effect-fence",
        graph_name="chat",
        execution_token="worker-old",
        **run_identity("effect-fence", user_id="owner"),
    )
    claimed_at = datetime(2026, 8, 9, 9, 0, tzinfo=timezone.utc)
    assert ledger.claim_effect(
        "run-effect-fence",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-old",
        claimed_at=claimed_at,
    )
    assert ledger.claim_effect(
        "run-effect-fence",
        "send",
        "message:wrong-token",
        user_id="owner",
        execution_token="worker-new",
    ) is False
    assert ledger.claim_recovery(
        "run-effect-fence",
        user_id="owner",
        expected_updated_at=run.updated_at,
        new_execution_token="worker-new",
    )

    assert ledger.complete_effect(
        "run-effect-fence",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-old",
        result={"stale": True},
    ) is None
    old_claim = ledger.get_effect(
        "run-effect-fence", "send", "message:fenced", user_id="owner"
    )
    assert old_claim is not None
    assert old_claim.status == "claimed"
    assert old_claim.execution_token == "worker-old"

    assert ledger.claim_effect(
        "run-effect-fence",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-new",
        allow_reclaim=True,
        stale_before=claimed_at,
    ) is False
    assert ledger.claim_effect(
        "run-effect-fence",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-new",
        allow_reclaim=True,
        stale_before=claimed_at + timedelta(seconds=1),
    ) is True
    reclaimed = ledger.get_effect(
        "run-effect-fence", "send", "message:fenced", user_id="owner"
    )
    assert reclaimed is not None
    assert reclaimed.execution_token == "worker-new"
    assert reclaimed.status == "claimed"

    assert ledger.fail_effect(
        "run-effect-fence",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-old",
        error="stale worker",
    ) is None
    completed = ledger.complete_effect(
        "run-effect-fence",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-new",
        result={"message_id": "message-new"},
    )
    assert completed is not None
    assert completed.status == "completed"
    assert completed.execution_token == "worker-new"
    assert ledger.claim_effect(
        "run-effect-fence",
        "send",
        "message:fenced",
        user_id="owner",
        execution_token="worker-new",
        allow_reclaim=True,
        stale_before=claimed_at + timedelta(days=1),
    ) is False

    assert ledger.claim_effect(
        "run-effect-fence",
        "charge",
        "billing:fenced",
        user_id="owner",
        execution_token="worker-new",
    )
    failed = ledger.fail_effect(
        "run-effect-fence",
        "charge",
        "billing:fenced",
        user_id="owner",
        execution_token="worker-new",
        error="provider timeout",
    )
    assert failed is not None
    assert failed.status == "failed"
    current_run = ledger.get_run("run-effect-fence")
    assert current_run is not None
    assert ledger.claim_recovery(
        "run-effect-fence",
        user_id="owner",
        expected_updated_at=current_run.updated_at,
        new_execution_token="worker-newer",
    )
    assert ledger.claim_effect(
        "run-effect-fence",
        "charge",
        "billing:fenced",
        user_id="owner",
        execution_token="worker-newer",
        allow_reclaim=True,
    )
    failed_reclaimed = ledger.get_effect(
        "run-effect-fence", "charge", "billing:fenced", user_id="owner"
    )
    assert failed_reclaimed is not None
    assert failed_reclaimed.status == "claimed"
    assert failed_reclaimed.execution_token == "worker-newer"
    assert failed_reclaimed.error == ""


def test_sqlite_effect_stale_seconds_and_reclaim_validation(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    run = ledger.start_run(
        run_id="run-effect-stale-seconds",
        thread_id="thread-effect-stale-seconds",
        graph_name="chat",
        execution_token="worker-old",
        **run_identity("effect-stale-seconds", user_id="owner"),
    )
    claimed_at = datetime(2026, 8, 9, 10, 0, tzinfo=timezone.utc)
    assert ledger.claim_effect(
        "run-effect-stale-seconds",
        "upload",
        "file:1",
        user_id="owner",
        execution_token="worker-old",
        claimed_at=claimed_at,
    )
    assert ledger.claim_recovery(
        "run-effect-stale-seconds",
        user_id="owner",
        expected_updated_at=run.updated_at,
        new_execution_token="worker-new",
    )

    with pytest.raises(ValueError, match="require allow_reclaim"):
        ledger.claim_effect(
            "run-effect-stale-seconds",
            "upload",
            "file:1",
            user_id="owner",
            stale_seconds=5,
        )
    with pytest.raises(ValueError, match="not both"):
        ledger.claim_effect(
            "run-effect-stale-seconds",
            "upload",
            "file:1",
            user_id="owner",
            allow_reclaim=True,
            stale_before=claimed_at,
            stale_seconds=5,
        )
    with pytest.raises(ValueError, match="positive number"):
        ledger.claim_effect(
            "run-effect-stale-seconds",
            "upload",
            "file:1",
            user_id="owner",
            allow_reclaim=True,
            stale_seconds=0,
        )

    assert ledger.claim_effect(
        "run-effect-stale-seconds",
        "upload",
        "file:1",
        user_id="owner",
        execution_token="worker-new",
        claimed_at=claimed_at + timedelta(seconds=10),
        allow_reclaim=True,
        stale_seconds=5,
    )


def test_effect_completion_validates_result_and_delete_cascades(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    ledger.start_run(
        run_id="run-cascade",
        thread_id="thread-cascade",
        graph_name="chat",
        **run_identity("cascade", user_id="owner"),
    )
    assert ledger.claim_effect(
        "run-cascade", "write_file", "file:report", user_id="owner"
    )

    with pytest.raises(ValueError, match="JSON serializable"):
        ledger.complete_effect(
            "run-cascade",
            "write_file",
            "file:report",
            user_id="owner",
            result=object(),
        )
    still_claimed = ledger.get_effect(
        "run-cascade", "write_file", "file:report", user_id="owner"
    )
    assert still_claimed is not None
    assert still_claimed.status == "claimed"

    assert ledger.delete_thread("thread-cascade") == 1
    assert ledger.list_run_effects("run-cascade", user_id="owner") == []


def test_sqlite_effect_schema_forward_migration_adds_payload_columns(tmp_path):
    path = tmp_path / "runs.sqlite"
    SqliteGraphRunLedger(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE graph_effects")
        connection.execute(
            """
            CREATE TABLE graph_effects (
                run_id TEXT NOT NULL,
                node TEXT NOT NULL,
                effect TEXT NOT NULL,
                PRIMARY KEY (run_id, node, effect)
            )
            """
        )

    SqliteGraphRunLedger(path)

    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(graph_effects)")
        }
    assert {
        "user_id",
        "execution_token",
        "status",
        "result_json",
        "error",
        "claimed_at",
        "updated_at",
        "finished_at",
    }.issubset(columns)


def test_runtime_exposes_effect_ledger_api(tmp_path):
    ledger = SqliteGraphRunLedger(tmp_path / "runs.sqlite")
    started = ledger.start_run(
        run_id="run-runtime-effect",
        thread_id="thread-runtime-effect",
        graph_name="chat",
        execution_token="worker-runtime-effect",
        **run_identity("runtime-effect", user_id="owner"),
    )
    runtime = GraphPersistenceRuntime(
        config=make_config(tmp_path),
        checkpointer_handle=CheckpointerHandle(
            saver=FakeDeletingSaver(), backend="fake"
        ),
        run_ledger=ledger,
    )

    assert runtime.claim_effect(
        "run-runtime-effect",
        "send",
        "message:1",
        user_id="owner",
        execution_token=started.execution_token,
    )
    completed = runtime.complete_effect(
        "run-runtime-effect",
        "send",
        "message:1",
        user_id="owner",
        execution_token=started.execution_token,
        result={"ok": True},
    )

    assert completed is not None
    assert completed.status == "completed"
    assert runtime.get_effect(
        "run-runtime-effect", "send", "message:1", user_id="owner"
    ) == completed
    assert runtime.list_run_effects("run-runtime-effect", user_id="owner") == [
        completed
    ]

    assert runtime.claim_effect(
        "run-runtime-effect",
        "charge",
        "billing:1",
        user_id="owner",
        execution_token=started.execution_token,
    )
    assert runtime.fail_effect(
        "run-runtime-effect",
        "charge",
        "billing:1",
        user_id="owner",
        execution_token=started.execution_token,
        error="retryable",
    ) is not None
    assert runtime.claim_effect(
        "run-runtime-effect",
        "charge",
        "billing:1",
        user_id="owner",
        execution_token=started.execution_token,
        allow_reclaim=True,
    )
