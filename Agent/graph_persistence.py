"""Standalone LangGraph persistence infrastructure.

This module intentionally has no dependency on the Flask application context or
the current chat runtime.  Callers own the lifecycle of the returned runtime and
can later pass ``runtime.checkpointer`` to ``StateGraph.compile``.

The PostgreSQL checkpointer schema is never created implicitly.  Run
``scripts/setup_langgraph_checkpointer.py`` (or call
``setup_postgres_checkpointer``) as a separate deployment migration first.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_APP_ENVIRONMENTS = frozenset({"development", "production"})
PRODUCTION_ENVIRONMENTS = frozenset({"production"})
SUPPORTED_CHECKPOINTER_BACKENDS = frozenset({"memory", "sqlite", "postgres"})
SUPPORTED_RUN_LEDGER_BACKENDS = frozenset({"sqlite", "postgres"})
SUPPORTED_DURABILITY_MODES = frozenset({"sync", "async", "exit"})
MAX_POSTGRES_CONNECTIONS = 8
POSTGRES_REQUIRED_UNIQUE_INDEXES: dict[str, tuple[str, ...]] = {
    "uq_graph_runs_request": ("user_id", "graph_name", "request_id"),
    "uq_graph_runs_idempotency": (
        "user_id",
        "graph_name",
        "idempotency_key",
    ),
    "uq_graph_session_deletions_owner": ("user_id", "session_id"),
    "uq_graph_effects_key": ("run_id", "node", "effect"),
}
SUPPORTED_RUN_STATUSES = frozenset(
    {"pending", "running", "succeeded", "failed", "interrupted", "cancelled"}
)
TERMINAL_RUN_STATUSES = frozenset({"succeeded", "failed", "interrupted", "cancelled"})


class GraphPersistenceError(RuntimeError):
    """Base error for persistence configuration and lifecycle failures."""


class GraphPersistenceConfigurationError(GraphPersistenceError):
    """Raised when persistence configuration is invalid or unsafe."""


class GraphPersistenceDependencyError(GraphPersistenceError):
    """Raised when an optional saver dependency is not installed."""


class GraphPersistenceConnectionError(GraphPersistenceError):
    """Raised when a configured persistence backend cannot be opened."""


class GraphPersistenceOwnershipError(GraphPersistenceError):
    """Raised when a caller attempts to access another user's graph thread."""


class GraphRunIdentityConflictError(GraphPersistenceError):
    """Raised when one request identity resolves to incompatible run rows."""

    def __init__(
        self,
        message: str,
        *,
        conflict_type: str,
        existing_run_id: str = "",
    ) -> None:
        super().__init__(message)
        self.conflict_type = str(conflict_type or "run_identity_conflict")
        self.existing_run_id = str(existing_run_id or "")


class GraphSessionDeletionRequestedError(GraphPersistenceError):
    """Raised when a deleted session attempts to create another graph run."""

    def __init__(self, *, user_id: str, session_id: str) -> None:
        super().__init__("graph session deletion has already been requested")
        self.user_id = str(user_id or "")
        self.session_id = str(session_id or "")


def _environment_value(
    environ: Mapping[str, str], key: str, default: str = ""
) -> str:
    value = environ.get(key)
    return default if value is None else str(value).strip()


def _parse_bool(environ: Mapping[str, str], key: str, default: bool) -> bool:
    raw = _environment_value(environ, key)
    if not raw:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise GraphPersistenceConfigurationError(
        f"{key} must be a boolean value, got {raw!r}"
    )


def _parse_int(
    environ: Mapping[str, str], key: str, default: int, *, minimum: int
) -> int:
    raw = _environment_value(environ, key)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise GraphPersistenceConfigurationError(
            f"{key} must be an integer, got {raw!r}"
        ) from exc
    if value < minimum:
        raise GraphPersistenceConfigurationError(
            f"{key} must be at least {minimum}, got {value}"
        )
    return value


def _parse_float(
    environ: Mapping[str, str], key: str, default: float, *, minimum: float
) -> float:
    raw = _environment_value(environ, key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise GraphPersistenceConfigurationError(
            f"{key} must be a number, got {raw!r}"
        ) from exc
    if value < minimum:
        raise GraphPersistenceConfigurationError(
            f"{key} must be at least {minimum}, got {value}"
        )
    return value


def _resolve_project_path(raw: str, default: Path) -> Path:
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


@dataclass(frozen=True)
class GraphPersistenceConfig:
    """Validated persistence settings independent from Flask configuration."""

    environment: str = "development"
    backend: str = "sqlite"
    sqlite_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "langgraph.sqlite"
    )
    run_ledger_sqlite_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "langgraph.sqlite"
    )
    postgres_dsn: str = field(default="", repr=False)
    run_ledger_backend: str = "sqlite"
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 2
    postgres_pool_timeout_seconds: float = 5.0
    postgres_pool_max_idle_seconds: float = 300.0
    postgres_pool_max_lifetime_seconds: float = 1800.0
    sqlite_busy_timeout_ms: int = 5000
    worker_processes: int = 1
    default_durability: str = "async"
    retention_days: int = 90
    retention_batch_size: int = 100
    strict_msgpack: bool = True

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in PRODUCTION_ENVIRONMENTS

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "GraphPersistenceConfig":
        source = os.environ if environ is None else environ
        environment = (
            _environment_value(source, "APP_ENV")
            or _environment_value(source, "ENVIRONMENT")
            or _environment_value(source, "FLASK_ENV")
            or "development"
        ).lower()
        backend = _environment_value(
            source, "LANGGRAPH_CHECKPOINTER_BACKEND", "sqlite"
        ).lower()
        backend = {"inmemory": "memory", "postgresql": "postgres"}.get(
            backend, backend
        )
        run_ledger_backend = _environment_value(
            source, "LANGGRAPH_RUN_LEDGER_BACKEND"
        ).lower()
        if not run_ledger_backend:
            run_ledger_backend = "postgres" if backend == "postgres" else "sqlite"
        run_ledger_backend = {"postgresql": "postgres"}.get(
            run_ledger_backend, run_ledger_backend
        )

        sqlite_path = _resolve_project_path(
            _environment_value(source, "LANGGRAPH_SQLITE_PATH"),
            PROJECT_ROOT / "data" / "langgraph.sqlite",
        )
        config = cls(
            environment=environment,
            backend=backend,
            sqlite_path=sqlite_path,
            run_ledger_sqlite_path=_resolve_project_path(
                _environment_value(source, "LANGGRAPH_RUN_LEDGER_SQLITE_PATH"),
                sqlite_path,
            ),
            postgres_dsn=_environment_value(source, "LANGGRAPH_POSTGRES_DSN"),
            run_ledger_backend=run_ledger_backend,
            postgres_pool_min_size=_parse_int(
                source, "LANGGRAPH_POOL_MIN_SIZE", 1, minimum=1
            ),
            postgres_pool_max_size=_parse_int(
                source, "LANGGRAPH_POOL_MAX_SIZE", 2, minimum=1
            ),
            postgres_pool_timeout_seconds=_parse_float(
                source, "LANGGRAPH_POOL_TIMEOUT_SECONDS", 5.0, minimum=0.1
            ),
            postgres_pool_max_idle_seconds=_parse_float(
                source, "LANGGRAPH_POOL_MAX_IDLE_SECONDS", 300.0, minimum=1.0
            ),
            postgres_pool_max_lifetime_seconds=_parse_float(
                source, "LANGGRAPH_POOL_MAX_LIFETIME_SECONDS", 1800.0, minimum=1.0
            ),
            sqlite_busy_timeout_ms=_parse_int(
                source, "LANGGRAPH_SQLITE_BUSY_TIMEOUT_MS", 5000, minimum=1
            ),
            # ``WEB_CONCURRENCY`` is the standard process-count knob used by
            # Gunicorn/container platforms.  Recording it here lets the
            # persistence boundary reject a process-local SQLite saver before
            # the application graph is constructed.
            worker_processes=_parse_int(
                source, "WEB_CONCURRENCY", 1, minimum=1
            ),
            default_durability=_environment_value(
                source, "LANGGRAPH_DURABILITY", "async"
            ).lower(),
            retention_days=_parse_int(
                source, "LANGGRAPH_CHECKPOINT_RETENTION_DAYS", 90, minimum=1
            ),
            retention_batch_size=_parse_int(
                source, "LANGGRAPH_RETENTION_BATCH_SIZE", 100, minimum=1
            ),
            strict_msgpack=_parse_bool(
                source, "LANGGRAPH_STRICT_MSGPACK", True
            ),
        )
        config.validate()
        if backend == "memory":
            raise GraphPersistenceConfigurationError(
                "LANGGRAPH_CHECKPOINTER_BACKEND must be sqlite or postgres; "
                "unit tests inject InMemorySaver directly"
            )
        return config

    def validate(self) -> None:
        if self.environment not in SUPPORTED_APP_ENVIRONMENTS:
            choices = ", ".join(sorted(SUPPORTED_APP_ENVIRONMENTS))
            raise GraphPersistenceConfigurationError(
                f"APP_ENV must be one of {choices}"
            )
        if self.backend not in SUPPORTED_CHECKPOINTER_BACKENDS:
            choices = ", ".join(sorted(SUPPORTED_CHECKPOINTER_BACKENDS))
            raise GraphPersistenceConfigurationError(
                f"LANGGRAPH_CHECKPOINTER_BACKEND must be one of {choices}"
            )
        if self.run_ledger_backend not in SUPPORTED_RUN_LEDGER_BACKENDS:
            choices = ", ".join(sorted(SUPPORTED_RUN_LEDGER_BACKENDS))
            raise GraphPersistenceConfigurationError(
                f"LANGGRAPH_RUN_LEDGER_BACKEND must be one of {choices}"
            )
        if self.is_production and self.backend != "postgres":
            raise GraphPersistenceConfigurationError(
                "Production requires LANGGRAPH_CHECKPOINTER_BACKEND=postgres; "
                "in-memory and SQLite checkpointers are not process-safe production defaults"
            )
        if self.backend == "postgres" and not self.postgres_dsn:
            raise GraphPersistenceConfigurationError(
                "LANGGRAPH_POSTGRES_DSN is required for the postgres checkpointer"
            )
        if self.run_ledger_backend == "postgres" and self.backend != "postgres":
            raise GraphPersistenceConfigurationError(
                "The postgres run ledger requires the postgres checkpointer and shared pool"
            )
        if self.is_production and self.run_ledger_backend != "postgres":
            raise GraphPersistenceConfigurationError(
                "Production requires LANGGRAPH_RUN_LEDGER_BACKEND=postgres; "
                "a local SQLite run ledger cannot coordinate multiple workers"
            )
        if self.worker_processes != 1 and (
            self.backend != "postgres" or self.run_ledger_backend != "postgres"
        ):
            raise GraphPersistenceConfigurationError(
                "Multi-process LangGraph requires both the postgres checkpointer "
                "and postgres run ledger; set WEB_CONCURRENCY=1 for local SQLite"
            )
        if self.is_production and not self.strict_msgpack:
            raise GraphPersistenceConfigurationError(
                "Production requires LANGGRAPH_STRICT_MSGPACK=true"
            )
        if self.postgres_pool_min_size < 1:
            raise GraphPersistenceConfigurationError(
                "LANGGRAPH_POOL_MIN_SIZE must be at least 1"
            )
        if self.postgres_pool_max_size < self.postgres_pool_min_size:
            raise GraphPersistenceConfigurationError(
                "LANGGRAPH_POOL_MAX_SIZE must be greater than or equal to "
                "LANGGRAPH_POOL_MIN_SIZE"
            )
        if self.postgres_pool_max_size > 2:
            raise GraphPersistenceConfigurationError(
                "LANGGRAPH_POOL_MAX_SIZE must not exceed 2 per worker"
            )
        if (
            self.backend == "postgres"
            and self.run_ledger_backend == "postgres"
            and self.worker_processes * self.postgres_pool_max_size
            > MAX_POSTGRES_CONNECTIONS
        ):
            raise GraphPersistenceConfigurationError(
                "WEB_CONCURRENCY * LANGGRAPH_POOL_MAX_SIZE must not exceed "
                f"{MAX_POSTGRES_CONNECTIONS} total PostgreSQL connections"
            )
        if self.postgres_pool_timeout_seconds <= 0:
            raise GraphPersistenceConfigurationError(
                "LANGGRAPH_POOL_TIMEOUT_SECONDS must be positive"
            )
        if self.postgres_pool_max_idle_seconds <= 0:
            raise GraphPersistenceConfigurationError(
                "LANGGRAPH_POOL_MAX_IDLE_SECONDS must be positive"
            )
        if self.postgres_pool_max_lifetime_seconds <= 0:
            raise GraphPersistenceConfigurationError(
                "LANGGRAPH_POOL_MAX_LIFETIME_SECONDS must be positive"
            )
        if self.sqlite_busy_timeout_ms < 1:
            raise GraphPersistenceConfigurationError(
                "LANGGRAPH_SQLITE_BUSY_TIMEOUT_MS must be positive"
            )
        if self.default_durability not in SUPPORTED_DURABILITY_MODES:
            choices = ", ".join(sorted(SUPPORTED_DURABILITY_MODES))
            raise GraphPersistenceConfigurationError(
                f"LANGGRAPH_DURABILITY must be one of {choices}"
            )
        if self.retention_days < 1 or self.retention_batch_size < 1:
            raise GraphPersistenceConfigurationError(
                "Retention days and batch size must be positive"
            )


def _apply_msgpack_security(config: GraphPersistenceConfig) -> None:
    """Apply the setting before importing or constructing any saver."""

    os.environ["LANGGRAPH_STRICT_MSGPACK"] = (
        "true" if config.strict_msgpack else "false"
    )


def _create_serializer(config: GraphPersistenceConfig) -> Any:
    """Build an explicit serializer so safety does not depend on import order."""

    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError as exc:  # pragma: no cover - installed with langgraph-checkpoint
        raise GraphPersistenceDependencyError(
            "The configured checkpointer requires langgraph-checkpoint serialization"
        ) from exc
    return JsonPlusSerializer(
        allowed_msgpack_modules=None if config.strict_msgpack else True
    )


def _load_memory_saver() -> type[Any]:
    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as exc:  # pragma: no cover - langgraph is a core dependency
        raise GraphPersistenceDependencyError(
            "The memory checkpointer requires the langgraph package"
        ) from exc
    return InMemorySaver


def _load_sqlite_saver() -> type[Any]:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise GraphPersistenceDependencyError(
            "The sqlite checkpointer requires langgraph-checkpoint-sqlite"
        ) from exc
    return SqliteSaver


def _load_postgres_dependencies() -> tuple[type[Any], type[Any], Any]:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError as exc:
        raise GraphPersistenceDependencyError(
            "The postgres checkpointer requires langgraph-checkpoint-postgres, "
            "psycopg and psycopg-pool"
        ) from exc
    return PostgresSaver, ConnectionPool, dict_row


class CheckpointerHandle:
    """Own a saver and the connection resource backing it."""

    def __init__(
        self,
        *,
        saver: Any,
        backend: str,
        resource: Any = None,
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        self.saver = saver
        self.backend = backend
        self.resource = resource
        self._close_callback = close_callback
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close_callback is not None:
            self._close_callback()

    def __enter__(self) -> "CheckpointerHandle":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def create_checkpointer(
    config: GraphPersistenceConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> CheckpointerHandle:
    """Create a saver without running backend schema migrations.

    ``PostgresSaver.setup()`` is deliberately excluded from this factory so
    concurrent Gunicorn workers never race deployment migrations.
    """

    if config is not None and environ is not None:
        raise GraphPersistenceConfigurationError(
            "Pass either config or environ, not both"
        )
    resolved = config or GraphPersistenceConfig.from_env(environ)
    resolved.validate()
    _apply_msgpack_security(resolved)
    serializer = _create_serializer(resolved)

    if resolved.backend == "memory":
        saver_type = _load_memory_saver()
        return CheckpointerHandle(
            saver=saver_type(serde=serializer), backend="memory"
        )

    if resolved.backend == "sqlite":
        saver_type = _load_sqlite_saver()
        resolved.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(resolved.sqlite_path),
            check_same_thread=False,
            timeout=resolved.sqlite_busy_timeout_ms / 1000,
        )
        try:
            connection.execute(
                f"PRAGMA busy_timeout={int(resolved.sqlite_busy_timeout_ms)}"
            )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            saver = saver_type(connection, serde=serializer)
        except Exception:
            connection.close()
            raise
        return CheckpointerHandle(
            saver=saver,
            backend="sqlite",
            resource=connection,
            close_callback=connection.close,
        )

    saver_type, pool_type, dict_row = _load_postgres_dependencies()
    pool = None
    try:
        pool = pool_type(
            conninfo=resolved.postgres_dsn,
            min_size=resolved.postgres_pool_min_size,
            max_size=resolved.postgres_pool_max_size,
            timeout=resolved.postgres_pool_timeout_seconds,
            max_idle=resolved.postgres_pool_max_idle_seconds,
            max_lifetime=resolved.postgres_pool_max_lifetime_seconds,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            check=pool_type.check_connection,
            open=False,
            name="langgraph-checkpointer",
        )
        pool.open(wait=True, timeout=resolved.postgres_pool_timeout_seconds)
        saver = saver_type(pool, serde=serializer)
    except Exception as exc:
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass
        raise GraphPersistenceConnectionError(
            "Could not open the configured PostgreSQL checkpointer pool"
        ) from exc
    return CheckpointerHandle(
        saver=saver,
        backend="postgres",
        resource=pool,
        close_callback=pool.close,
    )


def setup_postgres_checkpointer(
    config: GraphPersistenceConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Create or migrate official PostgresSaver tables as an explicit step."""

    if config is not None and environ is not None:
        raise GraphPersistenceConfigurationError(
            "Pass either config or environ, not both"
        )
    resolved = config or GraphPersistenceConfig.from_env(environ)
    resolved.validate()
    if resolved.backend != "postgres":
        raise GraphPersistenceConfigurationError(
            "Checkpointer setup requires LANGGRAPH_CHECKPOINTER_BACKEND=postgres"
        )
    with create_checkpointer(resolved) as handle:
        setup = getattr(handle.saver, "setup", None)
        if not callable(setup):  # pragma: no cover - defensive package guard
            raise GraphPersistenceDependencyError(
                "The installed PostgresSaver does not expose setup()"
            )
        setup()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalized_datetime(value: datetime | None = None) -> datetime:
    resolved = value or _utc_now()
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return _normalized_datetime(value).isoformat(timespec="microseconds")


def _required_identifier(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _validate_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized not in SUPPORTED_RUN_STATUSES:
        choices = ", ".join(sorted(SUPPORTED_RUN_STATUSES))
        raise ValueError(f"status must be one of {choices}")
    return normalized


def _validate_durability(durability: str) -> str:
    normalized = str(durability or "").strip().lower()
    if normalized not in SUPPORTED_DURABILITY_MODES:
        choices = ", ".join(sorted(SUPPORTED_DURABILITY_MODES))
        raise ValueError(f"durability must be one of {choices}")
    return normalized


def _validate_execution_token(execution_token: str) -> str:
    """Validate the opaque worker lease token used by run-level CAS writes."""

    return _required_identifier(execution_token, "execution_token")


def _validate_finish_status(status: str) -> str:
    normalized = _validate_status(status)
    if normalized not in TERMINAL_RUN_STATUSES:
        choices = ", ".join(sorted(TERMINAL_RUN_STATUSES))
        raise ValueError(f"finish status must be one of {choices}")
    return normalized


def _expected_updated_at(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return _timestamp(value)
    return _required_identifier(value, "expected_updated_at")


def _new_cas_timestamp(expected_updated_at: str) -> str:
    """Return a version timestamp guaranteed to differ from the CAS input."""

    timestamp = _timestamp()
    if timestamp != expected_updated_at:
        return timestamp
    try:
        previous = datetime.fromisoformat(expected_updated_at)
    except ValueError:  # pragma: no cover - ledger timestamps are ISO formatted
        return f"{timestamp}:claimed"
    return _timestamp(previous + timedelta(microseconds=1))


def _effect_identity(
    run_id: str, node: str, effect: str, user_id: str
) -> tuple[str, str, str, str]:
    return (
        _required_identifier(run_id, "run_id"),
        _required_identifier(node, "node"),
        _required_identifier(effect, "effect"),
        _required_identifier(user_id, "user_id"),
    )


def _effect_claim_window(
    *,
    claimed_at: datetime | None,
    allow_reclaim: bool,
    stale_before: datetime | None,
    stale_seconds: float | None,
) -> tuple[str, str | None]:
    """Resolve one claim timestamp and an optional stale-lease cutoff."""

    if stale_before is not None and stale_seconds is not None:
        raise ValueError("Pass stale_before or stale_seconds, not both")
    if (stale_before is not None or stale_seconds is not None) and not allow_reclaim:
        raise ValueError("stale reclaim parameters require allow_reclaim=True")

    claim_time = _normalized_datetime(claimed_at)
    cutoff: datetime | None = None
    if stale_before is not None:
        if not isinstance(stale_before, datetime):
            raise ValueError("stale_before must be a datetime")
        cutoff = _normalized_datetime(stale_before)
    elif stale_seconds is not None:
        try:
            seconds = float(stale_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("stale_seconds must be a positive number") from exc
        if seconds <= 0:
            raise ValueError("stale_seconds must be a positive number")
        cutoff = claim_time - timedelta(seconds=seconds)
    return _timestamp(claim_time), None if cutoff is None else _timestamp(cutoff)


def _metadata_json(metadata: Mapping[str, Any] | None) -> str:
    try:
        return json.dumps(
            dict(metadata or {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("run metadata must be JSON serializable") from exc


def _result_json(result: Any) -> str:
    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("effect result must be JSON serializable") from exc


def _artifact_run_ids_json(values: tuple[str, ...]) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        run_id = _required_identifier(value, "artifact_run_id")
        if run_id not in seen:
            seen.add(run_id)
            unique.append(run_id)
    return json.dumps(unique, ensure_ascii=False, separators=(",", ":"))


def _resume_claim_expectations(
    expected_checkpoint_id: str | None,
    expected_interrupt_id: str | None,
) -> tuple[str | None, str | None]:
    """Validate the immutable checkpoint/interrupt token used by resume CAS."""

    if expected_checkpoint_id is None and expected_interrupt_id is None:
        raise ValueError(
            "claim_resume requires expected_checkpoint_id or "
            "expected_interrupt_id"
        )
    checkpoint_id = (
        None
        if expected_checkpoint_id is None
        else _required_identifier(expected_checkpoint_id, "expected_checkpoint_id")
    )
    interrupt_id = (
        None
        if expected_interrupt_id is None
        else _required_identifier(expected_interrupt_id, "expected_interrupt_id")
    )
    return checkpoint_id, interrupt_id


@dataclass(frozen=True)
class GraphRunRecord:
    run_id: str
    thread_id: str
    graph_name: str
    user_id: str
    session_id: str
    request_id: str
    idempotency_key: str
    workflow_version: str
    current_node: str
    checkpoint_id: str
    interrupt_id: str
    execution_token: str
    status: str
    durability: str
    started_at: str
    updated_at: str
    finished_at: str | None
    error: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GraphThreadDeletionRecord:
    deletion_id: str
    thread_id: str
    user_id: str
    session_id: str
    artifact_run_ids: tuple[str, ...]
    reason: str
    status: str
    requested_at: str
    updated_at: str
    last_error: str


@dataclass(frozen=True)
class GraphSessionDeletionRecord:
    deletion_id: str
    user_id: str
    session_id: str
    reason: str
    status: str
    requested_at: str
    updated_at: str
    last_error: str


@dataclass(frozen=True)
class GraphEffectRecord:
    """Durable ownership record for one externally visible graph side effect."""

    run_id: str
    node: str
    effect: str
    user_id: str
    execution_token: str
    status: str
    result: Any
    error: str
    claimed_at: str
    updated_at: str
    finished_at: str | None


def _effect_record(row: Any) -> GraphEffectRecord | None:
    if row is None:
        return None
    raw_result = row["result_json"]
    if raw_result is None:
        result = None
    elif isinstance(raw_result, (dict, list, int, float, bool)):
        result = raw_result
    else:
        try:
            result = json.loads(raw_result)
        except (TypeError, ValueError, json.JSONDecodeError):
            result = None
    return GraphEffectRecord(
        run_id=str(row["run_id"]),
        node=str(row["node"]),
        effect=str(row["effect"]),
        user_id=str(row["user_id"]),
        execution_token=str(row["execution_token"] or ""),
        status=str(row["status"]),
        result=result,
        error=str(row["error"] or ""),
        claimed_at=str(row["claimed_at"]),
        updated_at=str(row["updated_at"]),
        finished_at=(
            None if row["finished_at"] is None else str(row["finished_at"])
        ),
    )


def _thread_deletion_record(row: Any) -> GraphThreadDeletionRecord | None:
    if row is None:
        return None
    try:
        raw_artifact_run_ids = json.loads(row["artifact_run_ids_json"] or "[]")
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        raw_artifact_run_ids = []
    artifact_run_ids = tuple(
        str(item).strip()
        for item in raw_artifact_run_ids
        if str(item).strip()
    ) if isinstance(raw_artifact_run_ids, list) else ()
    return GraphThreadDeletionRecord(
        deletion_id=str(row["deletion_id"]),
        thread_id=str(row["thread_id"]),
        user_id=str(row["user_id"]),
        session_id=str(row["session_id"]),
        artifact_run_ids=artifact_run_ids,
        reason=str(row["reason"]),
        status=str(row["status"]),
        requested_at=str(row["requested_at"]),
        updated_at=str(row["updated_at"]),
        last_error=str(row["last_error"] or ""),
    )


def _session_deletion_record(row: Any) -> GraphSessionDeletionRecord | None:
    if row is None:
        return None
    return GraphSessionDeletionRecord(
        deletion_id=str(row["deletion_id"]),
        user_id=str(row["user_id"]),
        session_id=str(row["session_id"]),
        reason=str(row["reason"]),
        status=str(row["status"]),
        requested_at=str(row["requested_at"]),
        updated_at=str(row["updated_at"]),
        last_error=str(row["last_error"] or ""),
    )


def _request_digest(metadata: Mapping[str, Any] | None) -> str:
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("request_digest") or "").strip()


def _validate_replayed_run(
    record: GraphRunRecord,
    *,
    request_digest: str,
) -> GraphRunRecord:
    """Reject reuse of an idempotency key for a different request body."""

    existing_digest = _request_digest(record.metadata)
    if request_digest and existing_digest and request_digest != existing_digest:
        raise GraphRunIdentityConflictError(
            "idempotency key was already used for a different request payload",
            conflict_type="request_digest_mismatch",
            existing_run_id=record.run_id,
        )
    return record


def _resolve_run_start_conflict(
    *,
    run_id: str,
    request_id: str,
    idempotency_key: str,
    request_digest: str,
    by_run_id: GraphRunRecord | None,
    by_request_id: GraphRunRecord | None,
    by_idempotency_key: GraphRunRecord | None,
) -> GraphRunRecord:
    """Resolve a failed insert without confusing independent unique keys."""

    if (
        by_request_id is not None
        and by_idempotency_key is not None
        and by_request_id.run_id != by_idempotency_key.run_id
    ):
        raise GraphRunIdentityConflictError(
            "request_id and idempotency_key belong to different graph runs",
            conflict_type="request_id_idempotency_key_mismatch",
            existing_run_id=by_request_id.run_id,
        )

    resolved_identity = by_idempotency_key or by_request_id
    if (
        by_run_id is not None
        and resolved_identity is not None
        and by_run_id.run_id != resolved_identity.run_id
    ):
        raise GraphRunIdentityConflictError(
            "run_id and request identity belong to different graph runs",
            conflict_type="run_id_identity_mismatch",
            existing_run_id=by_run_id.run_id,
        )

    if by_request_id is not None and by_idempotency_key is None:
        raise GraphRunIdentityConflictError(
            "request_id was already used with a different idempotency key",
            conflict_type="request_id_conflict",
            existing_run_id=by_request_id.run_id,
        )

    replay = resolved_identity
    if replay is not None:
        return _validate_replayed_run(
            replay,
            request_digest=request_digest,
        )

    if by_run_id is not None:
        raise GraphRunIdentityConflictError(
            "run_id already exists with different request identities",
            conflict_type="run_id_conflict",
            existing_run_id=by_run_id.run_id,
        )

    raise GraphRunIdentityConflictError(
        "graph run could not be created because its request identity conflicted",
        conflict_type="unknown_identity_conflict",
    )


@runtime_checkable
class GraphRunLedger(Protocol):
    """Backend-neutral run, deletion-tombstone, and effect-ledger contract."""

    def start_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        graph_name: str,
        user_id: str,
        session_id: str,
        request_id: str,
        workflow_version: str,
        current_node: str = "",
        execution_token: str | None = None,
        idempotency_key: str | None = None,
        durability: str = "async",
        metadata: Mapping[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> GraphRunRecord: ...

    def update_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str = "",
        current_node: str | None = None,
        checkpoint_id: str | None = None,
        interrupt_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None: ...

    def heartbeat_run(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        current_node: str,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None: ...

    def finish_run(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        status: str,
        current_node: str,
        error: str = "",
        checkpoint_id: str | None = None,
        interrupt_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None: ...

    def claim_recovery(
        self,
        run_id: str,
        *,
        user_id: str,
        expected_updated_at: str | datetime,
        new_execution_token: str,
    ) -> bool: ...

    def claim_resume(
        self,
        run_id: str,
        *,
        user_id: str,
        expected_checkpoint_id: str | None = None,
        expected_interrupt_id: str | None = None,
        new_execution_token: str | None = None,
        updated_at: datetime | None = None,
    ) -> bool: ...

    def get_run(self, run_id: str) -> GraphRunRecord | None: ...

    def get_owned_run(
        self, run_id: str, *, user_id: str
    ) -> GraphRunRecord | None: ...

    def get_run_by_idempotency_key(
        self, user_id: str, graph_name: str, idempotency_key: str
    ) -> GraphRunRecord | None: ...

    def get_run_by_request_id(
        self, user_id: str, graph_name: str, request_id: str
    ) -> GraphRunRecord | None: ...

    def list_thread_runs(
        self, thread_id: str, *, limit: int = 100
    ) -> list[GraphRunRecord]: ...

    def list_session_runs(
        self, session_id: str, *, user_id: str, limit: int = 100
    ) -> list[GraphRunRecord]: ...

    def thread_belongs_to_user(self, thread_id: str, *, user_id: str) -> bool: ...

    def list_session_thread_ids(
        self, session_id: str, *, user_id: str
    ) -> list[str]: ...

    def claim_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        claimed_at: datetime | None = None,
        allow_reclaim: bool = False,
        stale_before: datetime | None = None,
        stale_seconds: float | None = None,
    ) -> bool: ...

    def complete_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        result: Any = None,
        updated_at: datetime | None = None,
    ) -> GraphEffectRecord | None: ...

    def fail_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        error: str,
        updated_at: datetime | None = None,
    ) -> GraphEffectRecord | None: ...

    def get_effect(
        self, run_id: str, node: str, effect: str, *, user_id: str
    ) -> GraphEffectRecord | None: ...

    def list_run_effects(
        self, run_id: str, *, user_id: str
    ) -> list[GraphEffectRecord]: ...

    def enqueue_session_deletion(
        self,
        session_id: str,
        *,
        user_id: str,
        reason: str,
        requested_at: datetime | None = None,
    ) -> GraphSessionDeletionRecord: ...

    def is_session_deletion_requested(
        self, session_id: str, *, user_id: str
    ) -> bool: ...

    def list_pending_session_deletions(
        self, *, limit: int = 100
    ) -> list[GraphSessionDeletionRecord]: ...

    def update_session_deletion(
        self,
        deletion_id: str,
        *,
        status: str,
        last_error: str = "",
        updated_at: datetime | None = None,
    ) -> GraphSessionDeletionRecord | None: ...

    def enqueue_thread_deletion(
        self,
        thread_id: str,
        *,
        user_id: str,
        session_id: str,
        artifact_run_ids: tuple[str, ...] = (),
        reason: str,
        requested_at: datetime | None = None,
    ) -> GraphThreadDeletionRecord: ...

    def is_thread_deletion_pending(
        self, thread_id: str, *, user_id: str
    ) -> bool: ...

    def is_thread_deletion_requested(
        self, thread_id: str, *, user_id: str
    ) -> bool: ...

    def list_pending_deletions(
        self, *, limit: int = 100
    ) -> list[GraphThreadDeletionRecord]: ...

    def update_thread_deletion(
        self,
        deletion_id: str,
        *,
        status: str,
        last_error: str = "",
        updated_at: datetime | None = None,
    ) -> GraphThreadDeletionRecord | None: ...

    def list_expired_thread_ids(
        self, cutoff: datetime, *, limit: int = 100
    ) -> list[str]: ...

    def delete_thread(self, thread_id: str) -> int: ...

    def close(self) -> None: ...


class SqliteGraphRunLedger:
    """Small multi-process-safe run ledger using one SQLite connection per call."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=self.busy_timeout_ms / 1000
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    graph_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    workflow_version TEXT NOT NULL,
                    current_node TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL DEFAULT '',
                    interrupt_id TEXT NOT NULL DEFAULT '',
                    execution_token TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    durability TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS graph_thread_deletions (
                    deletion_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    artifact_run_ids_json TEXT NOT NULL DEFAULT '[]',
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS graph_session_deletions (
                    deletion_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    requested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT NOT NULL DEFAULT '',
                    UNIQUE (user_id, session_id)
                );

                CREATE TABLE IF NOT EXISTS graph_effects (
                    run_id TEXT NOT NULL,
                    node TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    execution_token TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    claimed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    PRIMARY KEY (run_id, node, effect),
                    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
                        ON DELETE CASCADE
                );

                """
            )

            # Forward-only compatibility for databases created by an earlier
            # local preview of this module.
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(graph_runs)")
            }
            additions = {
                "user_id": "TEXT NOT NULL DEFAULT ''",
                "session_id": "TEXT NOT NULL DEFAULT ''",
                "request_id": "TEXT NOT NULL DEFAULT ''",
                "idempotency_key": "TEXT NOT NULL DEFAULT ''",
                "workflow_version": "TEXT NOT NULL DEFAULT ''",
                "current_node": "TEXT NOT NULL DEFAULT ''",
                "checkpoint_id": "TEXT NOT NULL DEFAULT ''",
                "interrupt_id": "TEXT NOT NULL DEFAULT ''",
                "execution_token": "TEXT NOT NULL DEFAULT ''",
            }
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE graph_runs ADD COLUMN {name} {definition}"
                    )

            deletion_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(graph_thread_deletions)"
                )
            }
            if "artifact_run_ids_json" not in deletion_columns:
                connection.execute(
                    "ALTER TABLE graph_thread_deletions "
                    "ADD COLUMN artifact_run_ids_json TEXT NOT NULL DEFAULT '[]'"
                )

            session_deletion_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(graph_session_deletions)"
                )
            }
            for name, definition in {
                "status": "TEXT NOT NULL DEFAULT 'pending'",
                "last_error": "TEXT NOT NULL DEFAULT ''",
            }.items():
                if name not in session_deletion_columns:
                    connection.execute(
                        f"ALTER TABLE graph_session_deletions "
                        f"ADD COLUMN {name} {definition}"
                    )

            # Keep databases created by early previews forward-compatible. The
            # identity columns are part of every published schema; the mutable
            # payload columns can safely be added with conservative defaults.
            effect_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(graph_effects)")
            }
            effect_additions = {
                "user_id": "TEXT NOT NULL DEFAULT ''",
                "execution_token": "TEXT NOT NULL DEFAULT ''",
                "status": "TEXT NOT NULL DEFAULT 'claimed'",
                "result_json": "TEXT",
                "error": "TEXT NOT NULL DEFAULT ''",
                "claimed_at": "TEXT NOT NULL DEFAULT ''",
                "updated_at": "TEXT NOT NULL DEFAULT ''",
                "finished_at": "TEXT",
            }
            for name, definition in effect_additions.items():
                if name not in effect_columns:
                    connection.execute(
                        f"ALTER TABLE graph_effects ADD COLUMN {name} {definition}"
                    )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_runs_thread_updated
                    ON graph_runs(thread_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_graph_runs_user_session
                    ON graph_runs(user_id, session_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_graph_runs_updated
                    ON graph_runs(updated_at);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_runs_request
                    ON graph_runs(user_id, graph_name, request_id)
                    WHERE request_id <> '';
                CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_runs_idempotency
                    ON graph_runs(user_id, graph_name, idempotency_key)
                    WHERE idempotency_key <> '';
                CREATE INDEX IF NOT EXISTS idx_graph_thread_deletions_pending
                    ON graph_thread_deletions(status, updated_at);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_session_deletions_owner
                    ON graph_session_deletions(user_id, session_id);
                CREATE INDEX IF NOT EXISTS idx_graph_session_deletions_pending
                    ON graph_session_deletions(status, updated_at);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_effects_key
                    ON graph_effects(run_id, node, effect);
                CREATE INDEX IF NOT EXISTS idx_graph_effects_run_claimed
                    ON graph_effects(run_id, claimed_at, node, effect);
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row | None) -> GraphRunRecord | None:
        if row is None:
            return None
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return GraphRunRecord(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            graph_name=row["graph_name"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            request_id=row["request_id"],
            idempotency_key=row["idempotency_key"],
            workflow_version=row["workflow_version"],
            current_node=row["current_node"] or "",
            checkpoint_id=row["checkpoint_id"] or "",
            interrupt_id=row["interrupt_id"] or "",
            execution_token=row["execution_token"] or "",
            status=row["status"],
            durability=row["durability"],
            started_at=row["started_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
            error=row["error"] or "",
            metadata=metadata,
        )

    def start_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        graph_name: str,
        user_id: str,
        session_id: str,
        request_id: str,
        workflow_version: str,
        current_node: str = "",
        execution_token: str | None = None,
        idempotency_key: str | None = None,
        durability: str = "async",
        metadata: Mapping[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> GraphRunRecord:
        run_id = _required_identifier(run_id, "run_id")
        thread_id = _required_identifier(thread_id, "thread_id")
        graph_name = _required_identifier(graph_name, "graph_name")
        user_id = _required_identifier(user_id, "user_id")
        session_id = _required_identifier(session_id, "session_id")
        request_id = _required_identifier(request_id, "request_id")
        workflow_version = _required_identifier(
            workflow_version, "workflow_version"
        )
        idempotency_key = _required_identifier(
            idempotency_key or request_id, "idempotency_key"
        )
        durability = _validate_durability(durability)
        execution_token = _validate_execution_token(
            execution_token or uuid4().hex
        )
        timestamp = _timestamp(started_at)
        metadata_json = _metadata_json(metadata)
        request_digest = _request_digest(metadata)
        with self._connect() as connection:
            # Both run creation and session-barrier creation begin with a
            # write transaction. SQLite then serializes their NOT EXISTS / UPSERT
            # decisions, so whichever commits first is visible to the other.
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO graph_runs (
                    run_id, thread_id, graph_name, user_id, session_id,
                    request_id, idempotency_key, workflow_version, current_node,
                    execution_token, status, durability,
                    started_at, updated_at, finished_at, error, metadata_json
                )
                SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, NULL, '', ?
                 WHERE NOT EXISTS (
                    SELECT 1 FROM graph_session_deletions
                     WHERE user_id = ? AND session_id = ?
                 )
                ON CONFLICT DO NOTHING
                """,
                (
                    run_id,
                    thread_id,
                    graph_name,
                    user_id,
                    session_id,
                    request_id,
                    idempotency_key,
                    workflow_version,
                    str(current_node or ""),
                    execution_token,
                    durability,
                    timestamp,
                    timestamp,
                    metadata_json,
                    user_id,
                    session_id,
                ),
            )
            inserted = cursor.rowcount == 1
        if inserted:
            record = self.get_run(run_id)
            if record is None:  # pragma: no cover - committed INSERT invariant
                raise GraphPersistenceError("created graph run could not be reloaded")
            return record

        if self.is_session_deletion_requested(session_id, user_id=user_id):
            raise GraphSessionDeletionRequestedError(
                user_id=user_id,
                session_id=session_id,
            )
        return _resolve_run_start_conflict(
            run_id=run_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            request_digest=request_digest,
            by_run_id=self.get_run(run_id),
            by_request_id=self.get_run_by_request_id(
                user_id, graph_name, request_id
            ),
            by_idempotency_key=self.get_run_by_idempotency_key(
                user_id, graph_name, idempotency_key
            ),
        )

    def update_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str = "",
        current_node: str | None = None,
        checkpoint_id: str | None = None,
        interrupt_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None:
        run_id = _required_identifier(run_id, "run_id")
        status = _validate_status(status)
        timestamp = _timestamp(updated_at)
        finished_at = timestamp if status in TERMINAL_RUN_STATUSES else None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE graph_runs
                   SET status = ?, updated_at = ?, finished_at = ?, error = ?,
                       current_node = COALESCE(?, current_node),
                       checkpoint_id = COALESCE(?, checkpoint_id),
                       interrupt_id = COALESCE(?, interrupt_id)
                 WHERE run_id = ?
                   AND NOT EXISTS (
                       SELECT 1 FROM graph_session_deletions AS deletion
                        WHERE deletion.user_id = graph_runs.user_id
                          AND deletion.session_id = graph_runs.session_id
                   )
                """,
                (
                    status,
                    timestamp,
                    finished_at,
                    str(error or ""),
                    None if current_node is None else str(current_node),
                    None if checkpoint_id is None else str(checkpoint_id),
                    None if interrupt_id is None else str(interrupt_id),
                    run_id,
                ),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_run(run_id)

    def heartbeat_run(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        current_node: str,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None:
        """Advance a running lease only while this worker still owns it."""

        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        execution_token = _validate_execution_token(execution_token)
        current_node = _required_identifier(current_node, "current_node")
        timestamp = _timestamp(updated_at)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE graph_runs
                   SET current_node = ?, updated_at = ?
                 WHERE run_id = ? AND user_id = ? AND status = 'running'
                   AND execution_token = ?
                   AND NOT EXISTS (
                       SELECT 1 FROM graph_session_deletions AS deletion
                        WHERE deletion.user_id = graph_runs.user_id
                          AND deletion.session_id = graph_runs.session_id
                   )
                """,
                (current_node, timestamp, run_id, user_id, execution_token),
            )
            if cursor.rowcount != 1:
                return None
        return self.get_owned_run(run_id, user_id=user_id)

    def finish_run(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        status: str,
        current_node: str,
        error: str = "",
        checkpoint_id: str | None = None,
        interrupt_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None:
        """Finish a run only if the caller still owns its execution lease."""

        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        execution_token = _validate_execution_token(execution_token)
        status = _validate_finish_status(status)
        current_node = _required_identifier(current_node, "current_node")
        timestamp = _timestamp(updated_at)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE graph_runs
                   SET status = ?, current_node = ?, updated_at = ?,
                       finished_at = ?, error = ?,
                       checkpoint_id = COALESCE(?, checkpoint_id),
                       interrupt_id = COALESCE(?, interrupt_id)
                 WHERE run_id = ? AND user_id = ? AND status = 'running'
                   AND execution_token = ?
                   AND NOT EXISTS (
                       SELECT 1 FROM graph_session_deletions AS deletion
                        WHERE deletion.user_id = graph_runs.user_id
                          AND deletion.session_id = graph_runs.session_id
                   )
                """,
                (
                    status,
                    current_node,
                    timestamp,
                    timestamp,
                    str(error or ""),
                    None if checkpoint_id is None else str(checkpoint_id),
                    None if interrupt_id is None else str(interrupt_id),
                    run_id,
                    user_id,
                    execution_token,
                ),
            )
            if cursor.rowcount != 1:
                return None
        return self.get_owned_run(run_id, user_id=user_id)

    def claim_recovery(
        self,
        run_id: str,
        *,
        user_id: str,
        expected_updated_at: str | datetime,
        new_execution_token: str,
    ) -> bool:
        """Take over a failed/stale run with an owner-and-version CAS."""

        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        expected = _expected_updated_at(expected_updated_at)
        new_execution_token = _validate_execution_token(new_execution_token)
        timestamp = _new_cas_timestamp(expected)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE graph_runs
                   SET status = 'running', execution_token = ?, updated_at = ?,
                       finished_at = NULL, error = ''
                 WHERE run_id = ? AND user_id = ?
                   AND status IN ('failed', 'running') AND updated_at = ?
                   AND NOT EXISTS (
                       SELECT 1 FROM graph_session_deletions AS deletion
                        WHERE deletion.user_id = graph_runs.user_id
                          AND deletion.session_id = graph_runs.session_id
                   )
                """,
                (new_execution_token, timestamp, run_id, user_id, expected),
            )
            return cursor.rowcount == 1

    def claim_resume(
        self,
        run_id: str,
        *,
        user_id: str,
        expected_checkpoint_id: str | None = None,
        expected_interrupt_id: str | None = None,
        new_execution_token: str | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        """Atomically claim one interrupted run for resume exactly once."""

        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        checkpoint_id, interrupt_id = _resume_claim_expectations(
            expected_checkpoint_id, expected_interrupt_id
        )
        execution_token = _validate_execution_token(
            new_execution_token or uuid4().hex
        )
        conditions = ["run_id = ?", "user_id = ?", "status = 'interrupted'"]
        params: list[Any] = [execution_token, _timestamp(updated_at), run_id, user_id]
        if checkpoint_id is not None:
            conditions.append("checkpoint_id = ?")
            params.append(checkpoint_id)
        if interrupt_id is not None:
            conditions.append("interrupt_id = ?")
            params.append(interrupt_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"""
                UPDATE graph_runs
                   SET status = 'running', execution_token = ?, updated_at = ?,
                       finished_at = NULL, error = ''
                 WHERE {' AND '.join(conditions)}
                   AND NOT EXISTS (
                       SELECT 1 FROM graph_session_deletions AS deletion
                        WHERE deletion.user_id = graph_runs.user_id
                          AND deletion.session_id = graph_runs.session_id
                   )
                """,
                tuple(params),
            )
            return cursor.rowcount == 1

    def get_run(self, run_id: str) -> GraphRunRecord | None:
        run_id = _required_identifier(run_id, "run_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM graph_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._record(row)

    def get_owned_run(self, run_id: str, *, user_id: str) -> GraphRunRecord | None:
        user_id = _required_identifier(user_id, "user_id")
        run_id = _required_identifier(run_id, "run_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM graph_runs WHERE run_id = ? AND user_id = ?",
                (run_id, user_id),
            ).fetchone()
        return self._record(row)

    def get_run_by_idempotency_key(
        self, user_id: str, graph_name: str, idempotency_key: str
    ) -> GraphRunRecord | None:
        user_id = _required_identifier(user_id, "user_id")
        graph_name = _required_identifier(graph_name, "graph_name")
        idempotency_key = _required_identifier(
            idempotency_key, "idempotency_key"
        )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM graph_runs
                 WHERE user_id = ? AND graph_name = ? AND idempotency_key = ?
                """,
                (user_id, graph_name, idempotency_key),
            ).fetchone()
        return self._record(row)

    def get_run_by_request_id(
        self, user_id: str, graph_name: str, request_id: str
    ) -> GraphRunRecord | None:
        user_id = _required_identifier(user_id, "user_id")
        graph_name = _required_identifier(graph_name, "graph_name")
        request_id = _required_identifier(request_id, "request_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM graph_runs
                 WHERE user_id = ? AND graph_name = ? AND request_id = ?
                """,
                (user_id, graph_name, request_id),
            ).fetchone()
        return self._record(row)

    def list_thread_runs(
        self, thread_id: str, *, limit: int = 100
    ) -> list[GraphRunRecord]:
        thread_id = _required_identifier(thread_id, "thread_id")
        limit = max(1, int(limit))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_runs
                 WHERE thread_id = ?
                 ORDER BY started_at DESC, run_id DESC
                 LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [record for row in rows if (record := self._record(row)) is not None]

    def list_session_runs(
        self, session_id: str, *, user_id: str, limit: int = 100
    ) -> list[GraphRunRecord]:
        user_id = _required_identifier(user_id, "user_id")
        session_id = _required_identifier(session_id, "session_id")
        limit = max(1, int(limit))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_runs
                 WHERE user_id = ? AND session_id = ?
                 ORDER BY started_at DESC, run_id DESC
                 LIMIT ?
                """,
                (user_id, session_id, limit),
            ).fetchall()
        return [record for row in rows if (record := self._record(row)) is not None]

    def thread_belongs_to_user(self, thread_id: str, *, user_id: str) -> bool:
        user_id = _required_identifier(user_id, "user_id")
        thread_id = _required_identifier(thread_id, "thread_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM graph_runs
                 WHERE user_id = ? AND thread_id = ?
                 LIMIT 1
                """,
                (user_id, thread_id),
            ).fetchone()
        return row is not None

    def list_session_thread_ids(
        self, session_id: str, *, user_id: str
    ) -> list[str]:
        session_id = _required_identifier(session_id, "session_id")
        user_id = _required_identifier(user_id, "user_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT thread_id FROM graph_runs
                 WHERE user_id = ? AND session_id = ?
                 ORDER BY thread_id
                """,
                (user_id, session_id),
            ).fetchall()
        return [str(row["thread_id"]) for row in rows]

    def claim_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        claimed_at: datetime | None = None,
        allow_reclaim: bool = False,
        stale_before: datetime | None = None,
        stale_seconds: float | None = None,
    ) -> bool:
        """Atomically reserve or explicitly reclaim a fenced side-effect key."""

        run_id, node, effect, user_id = _effect_identity(
            run_id, node, effect, user_id
        )
        token = (
            ""
            if execution_token is None
            else _validate_execution_token(execution_token)
        )
        timestamp, stale_cutoff = _effect_claim_window(
            claimed_at=claimed_at,
            allow_reclaim=allow_reclaim,
            stale_before=stale_before,
            stale_seconds=stale_seconds,
        )
        conflict_clause = "ON CONFLICT(run_id, node, effect) DO NOTHING"
        reclaim_params: tuple[Any, ...] = ()
        if allow_reclaim:
            conflict_clause = """
                ON CONFLICT(run_id, node, effect) DO UPDATE SET
                    user_id = excluded.user_id,
                    execution_token = excluded.execution_token,
                    status = 'claimed',
                    result_json = NULL,
                    error = '',
                    claimed_at = excluded.claimed_at,
                    updated_at = excluded.updated_at,
                    finished_at = NULL
                WHERE graph_effects.status = 'failed'
                   OR (
                        graph_effects.status = 'claimed'
                        AND ? IS NOT NULL
                        AND graph_effects.updated_at < ?
                   )
            """
            reclaim_params = (stale_cutoff, stale_cutoff)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"""
                INSERT INTO graph_effects (
                    run_id, node, effect, user_id, execution_token, status,
                    result_json, error, claimed_at, updated_at, finished_at
                )
                SELECT run_id, ?, ?, user_id, ?, 'claimed', NULL, '', ?, ?, NULL
                  FROM graph_runs
                 WHERE run_id = ? AND user_id = ?
                   AND (
                        ? = '' OR (
                            status = 'running' AND execution_token = ?
                        )
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM graph_session_deletions AS deletion
                        WHERE deletion.user_id = graph_runs.user_id
                          AND deletion.session_id = graph_runs.session_id
                   )
                {conflict_clause}
                """,
                (
                    node,
                    effect,
                    token,
                    timestamp,
                    timestamp,
                    run_id,
                    user_id,
                    token,
                    token,
                    *reclaim_params,
                ),
            )
            return cursor.rowcount == 1

    def complete_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        result: Any = None,
        updated_at: datetime | None = None,
    ) -> GraphEffectRecord | None:
        run_id, node, effect, user_id = _effect_identity(
            run_id, node, effect, user_id
        )
        result_json = _result_json(result)
        timestamp = _timestamp(updated_at)
        token = (
            ""
            if execution_token is None
            else _validate_execution_token(execution_token)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE graph_effects
                   SET status = 'completed', result_json = ?, error = '',
                       updated_at = ?, finished_at = ?
                 WHERE run_id = ? AND node = ? AND effect = ? AND user_id = ?
                   AND status = 'claimed'
                   AND EXISTS (
                       SELECT 1 FROM graph_runs AS current_run
                        WHERE current_run.run_id = graph_effects.run_id
                          AND current_run.user_id = ?
                          AND current_run.status = 'running'
                          AND (
                               ? = '' OR (
                                   graph_effects.execution_token = ?
                                   AND current_run.execution_token = ?
                               )
                          )
                          AND NOT EXISTS (
                              SELECT 1
                                FROM graph_session_deletions AS deletion
                               WHERE deletion.user_id = current_run.user_id
                                 AND deletion.session_id = current_run.session_id
                          )
                   )
                """,
                (
                    result_json,
                    timestamp,
                    timestamp,
                    run_id,
                    node,
                    effect,
                    user_id,
                    user_id,
                    token,
                    token,
                    token,
                ),
            )
            if cursor.rowcount != 1:
                return None
        return self.get_effect(run_id, node, effect, user_id=user_id)

    def fail_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        error: str,
        updated_at: datetime | None = None,
    ) -> GraphEffectRecord | None:
        run_id, node, effect, user_id = _effect_identity(
            run_id, node, effect, user_id
        )
        error = _required_identifier(error, "error")[:4000]
        timestamp = _timestamp(updated_at)
        token = (
            ""
            if execution_token is None
            else _validate_execution_token(execution_token)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE graph_effects
                   SET status = 'failed', result_json = NULL, error = ?,
                       updated_at = ?, finished_at = ?
                 WHERE run_id = ? AND node = ? AND effect = ? AND user_id = ?
                   AND status = 'claimed'
                   AND EXISTS (
                       SELECT 1 FROM graph_runs AS current_run
                        WHERE current_run.run_id = graph_effects.run_id
                          AND current_run.user_id = ?
                          AND current_run.status = 'running'
                          AND (
                               ? = '' OR (
                                   graph_effects.execution_token = ?
                                   AND current_run.execution_token = ?
                               )
                          )
                          AND NOT EXISTS (
                              SELECT 1
                                FROM graph_session_deletions AS deletion
                               WHERE deletion.user_id = current_run.user_id
                                 AND deletion.session_id = current_run.session_id
                          )
                   )
                """,
                (
                    error,
                    timestamp,
                    timestamp,
                    run_id,
                    node,
                    effect,
                    user_id,
                    user_id,
                    token,
                    token,
                    token,
                ),
            )
            if cursor.rowcount != 1:
                return None
        return self.get_effect(run_id, node, effect, user_id=user_id)

    def get_effect(
        self, run_id: str, node: str, effect: str, *, user_id: str
    ) -> GraphEffectRecord | None:
        run_id, node, effect, user_id = _effect_identity(
            run_id, node, effect, user_id
        )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM graph_effects
                 WHERE run_id = ? AND node = ? AND effect = ? AND user_id = ?
                """,
                (run_id, node, effect, user_id),
            ).fetchone()
        return _effect_record(row)

    def list_run_effects(
        self, run_id: str, *, user_id: str
    ) -> list[GraphEffectRecord]:
        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_effects
                 WHERE run_id = ? AND user_id = ?
                 ORDER BY claimed_at ASC, node ASC, effect ASC
                """,
                (run_id, user_id),
            ).fetchall()
        return [
            record
            for row in rows
            if (record := _effect_record(row)) is not None
        ]

    def enqueue_session_deletion(
        self,
        session_id: str,
        *,
        user_id: str,
        reason: str,
        requested_at: datetime | None = None,
    ) -> GraphSessionDeletionRecord:
        """Install a permanent owner/session barrier before enumerating runs."""

        session_id = _required_identifier(session_id, "session_id")
        user_id = _required_identifier(user_id, "user_id")
        reason = _required_identifier(reason, "reason")
        timestamp = _timestamp(requested_at)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO graph_session_deletions (
                    deletion_id, user_id, session_id, reason, status,
                    requested_at, updated_at, last_error
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, '')
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                    reason = excluded.reason,
                    status = 'pending',
                    updated_at = excluded.updated_at,
                    last_error = ''
                """,
                (str(uuid4()), user_id, session_id, reason, timestamp, timestamp),
            )
            row = connection.execute(
                """
                SELECT * FROM graph_session_deletions
                 WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        record = _session_deletion_record(row)
        assert record is not None
        return record

    def is_session_deletion_requested(
        self, session_id: str, *, user_id: str
    ) -> bool:
        session_id = _required_identifier(session_id, "session_id")
        user_id = _required_identifier(user_id, "user_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM graph_session_deletions
                 WHERE user_id = ? AND session_id = ?
                 LIMIT 1
                """,
                (user_id, session_id),
            ).fetchone()
        return row is not None

    def list_pending_session_deletions(
        self, *, limit: int = 100
    ) -> list[GraphSessionDeletionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_session_deletions
                 WHERE status = 'pending'
                 ORDER BY updated_at ASC, deletion_id ASC
                 LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            record
            for row in rows
            if (record := _session_deletion_record(row)) is not None
        ]

    def update_session_deletion(
        self,
        deletion_id: str,
        *,
        status: str,
        last_error: str = "",
        updated_at: datetime | None = None,
    ) -> GraphSessionDeletionRecord | None:
        deletion_id = _required_identifier(deletion_id, "deletion_id")
        status = str(status or "").strip().lower()
        if status not in {"pending", "completed"}:
            raise ValueError("deletion status must be pending or completed")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE graph_session_deletions
                   SET status = ?, updated_at = ?, last_error = ?
                 WHERE deletion_id = ?
                """,
                (
                    status,
                    _timestamp(updated_at),
                    str(last_error or "")[:2000],
                    deletion_id,
                ),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM graph_session_deletions WHERE deletion_id = ?",
                (deletion_id,),
            ).fetchone()
        return _session_deletion_record(row)

    def enqueue_thread_deletion(
        self,
        thread_id: str,
        *,
        user_id: str,
        session_id: str,
        artifact_run_ids: tuple[str, ...] = (),
        reason: str,
        requested_at: datetime | None = None,
    ) -> GraphThreadDeletionRecord:
        thread_id = _required_identifier(thread_id, "thread_id")
        user_id = _required_identifier(user_id, "user_id")
        session_id = _required_identifier(session_id, "session_id")
        reason = _required_identifier(reason, "reason")
        artifact_run_ids_json = _artifact_run_ids_json(artifact_run_ids)
        timestamp = _timestamp(requested_at)
        deletion_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO graph_thread_deletions (
                    deletion_id, thread_id, user_id, session_id,
                    artifact_run_ids_json, reason, status, requested_at,
                    updated_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, '')
                ON CONFLICT(thread_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    session_id = excluded.session_id,
                    artifact_run_ids_json = CASE
                        WHEN excluded.artifact_run_ids_json = '[]'
                        THEN graph_thread_deletions.artifact_run_ids_json
                        ELSE excluded.artifact_run_ids_json
                    END,
                    reason = excluded.reason,
                    status = 'pending',
                    updated_at = excluded.updated_at,
                    last_error = ''
                """,
                (
                    deletion_id,
                    thread_id,
                    user_id,
                    session_id,
                    artifact_run_ids_json,
                    reason,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM graph_thread_deletions WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        record = _thread_deletion_record(row)
        assert record is not None
        return record

    def is_thread_deletion_pending(
        self, thread_id: str, *, user_id: str
    ) -> bool:
        thread_id = _required_identifier(thread_id, "thread_id")
        user_id = _required_identifier(user_id, "user_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM graph_thread_deletions
                 WHERE thread_id = ? AND user_id = ? AND status = 'pending'
                 LIMIT 1
                """,
                (thread_id, user_id),
            ).fetchone()
        return row is not None

    def is_thread_deletion_requested(
        self, thread_id: str, *, user_id: str
    ) -> bool:
        thread_id = _required_identifier(thread_id, "thread_id")
        user_id = _required_identifier(user_id, "user_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM graph_thread_deletions
                 WHERE thread_id = ? AND user_id = ?
                 LIMIT 1
                """,
                (thread_id, user_id),
            ).fetchone()
        return row is not None

    def list_pending_deletions(
        self, *, limit: int = 100
    ) -> list[GraphThreadDeletionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_thread_deletions
                 WHERE status = 'pending'
                 ORDER BY updated_at ASC, deletion_id ASC
                 LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            record
            for row in rows
            if (record := _thread_deletion_record(row)) is not None
        ]

    def update_thread_deletion(
        self,
        deletion_id: str,
        *,
        status: str,
        last_error: str = "",
        updated_at: datetime | None = None,
    ) -> GraphThreadDeletionRecord | None:
        deletion_id = _required_identifier(deletion_id, "deletion_id")
        status = str(status or "").strip().lower()
        if status not in {"pending", "completed"}:
            raise ValueError("deletion status must be pending or completed")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE graph_thread_deletions
                   SET status = ?, updated_at = ?, last_error = ?
                 WHERE deletion_id = ?
                """,
                (
                    status,
                    _timestamp(updated_at),
                    str(last_error or "")[:2000],
                    deletion_id,
                ),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM graph_thread_deletions WHERE deletion_id = ?",
                (deletion_id,),
            ).fetchone()
        return _thread_deletion_record(row)

    def list_expired_thread_ids(
        self, cutoff: datetime, *, limit: int = 100
    ) -> list[str]:
        cutoff_timestamp = _timestamp(cutoff)
        limit = max(1, int(limit))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT thread_id, MAX(updated_at) AS last_updated_at
                  FROM graph_runs
                 GROUP BY thread_id
                HAVING MAX(updated_at) < ?
                 ORDER BY last_updated_at ASC, thread_id ASC
                 LIMIT ?
                """,
                (cutoff_timestamp, limit),
            ).fetchall()
        return [str(row["thread_id"]) for row in rows]

    def delete_thread(self, thread_id: str) -> int:
        thread_id = _required_identifier(thread_id, "thread_id")
        with self._connect() as connection:
            # Explicit cleanup also covers databases created by a preview that
            # did not yet declare the ON DELETE CASCADE foreign key.
            connection.execute(
                """
                DELETE FROM graph_effects
                 WHERE run_id IN (
                    SELECT run_id FROM graph_runs WHERE thread_id = ?
                 )
                """,
                (thread_id,),
            )
            cursor = connection.execute(
                "DELETE FROM graph_runs WHERE thread_id = ?", (thread_id,)
            )
            return int(cursor.rowcount)

    def close(self) -> None:
        # Connections are intentionally short-lived and close after each call.
        return None


class PostgresGraphRunLedger:
    """Shared graph run ledger backed by the checkpointer's PostgreSQL pool."""

    def __init__(self, pool: Any) -> None:
        if pool is None or not callable(getattr(pool, "connection", None)):
            raise GraphPersistenceConfigurationError(
                "PostgresGraphRunLedger requires a psycopg ConnectionPool"
            )
        self.pool = pool

    @staticmethod
    @contextmanager
    def _transaction(connection: Any):
        """Open a real transaction even when the shared saver pool autocommits.

        ``PostgresSaver`` requires ``autocommit=True`` on its pool.  An
        advisory *transaction* lock would otherwise be released at the end of
        the SELECT statement and provide no protection to the following
        absence check.  Lightweight test doubles without ``transaction()``
        retain their existing context-manager behavior.
        """

        transaction = getattr(connection, "transaction", None)
        if callable(transaction):
            with transaction():
                yield
            return
        yield

    @staticmethod
    def _acquire_session_lock(
        cursor: Any,
        *,
        user_id: str,
        session_id: str,
    ) -> None:
        """Serialize run creation with deletion-barrier creation.

        The run and barrier rows live in different tables, so PostgreSQL row
        locks cannot protect the initial absence check. A transaction-scoped
        advisory lock supplies the shared linearization point. Hash collisions
        only serialize unrelated sessions; they cannot weaken correctness.
        """

        lock_identity = json.dumps(
            [user_id, session_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 730201))",
            (lock_identity,),
        )

    def _acquire_run_session_lock(
        self,
        cursor: Any,
        *,
        run_id: str,
        user_id: str,
    ) -> bool:
        """Resolve a run's immutable session, then take the deletion lock."""

        cursor.execute(
            "SELECT session_id FROM graph_runs WHERE run_id = %s AND user_id = %s",
            (run_id, user_id),
        )
        row = cursor.fetchone()
        if row is None:
            return False
        session_id = (
            row.get("session_id")
            if isinstance(row, Mapping)
            else row[0]
        )
        self._acquire_session_lock(
            cursor,
            user_id=user_id,
            session_id=_required_identifier(session_id, "session_id"),
        )
        return True

    def _acquire_unscoped_run_session_lock(
        self,
        cursor: Any,
        *,
        run_id: str,
    ) -> bool:
        """Take the same deletion fence for internal run-id-only mutations."""

        cursor.execute(
            "SELECT user_id, session_id FROM graph_runs WHERE run_id = %s",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return False
        if isinstance(row, Mapping):
            user_id = row.get("user_id")
            session_id = row.get("session_id")
        else:
            user_id, session_id = row[0], row[1]
        self._acquire_session_lock(
            cursor,
            user_id=_required_identifier(user_id, "user_id"),
            session_id=_required_identifier(session_id, "session_id"),
        )
        return True

    def setup(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS graph_runs (
                run_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                graph_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                workflow_version TEXT NOT NULL,
                current_node TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL DEFAULT '',
                interrupt_id TEXT NOT NULL DEFAULT '',
                execution_token TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                durability TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT,
                error TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS graph_thread_deletions (
                deletion_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                artifact_run_ids_json TEXT NOT NULL DEFAULT '[]',
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT NOT NULL DEFAULT ''
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS graph_session_deletions (
                deletion_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                requested_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT NOT NULL DEFAULT '',
                UNIQUE (user_id, session_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS graph_effects (
                run_id TEXT NOT NULL REFERENCES graph_runs(run_id)
                    ON DELETE CASCADE,
                node TEXT NOT NULL,
                effect TEXT NOT NULL,
                user_id TEXT NOT NULL,
                execution_token TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                result_json TEXT,
                error TEXT NOT NULL DEFAULT '',
                claimed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT,
                PRIMARY KEY (run_id, node, effect)
            )
            """,
            "ALTER TABLE graph_runs ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_runs ADD COLUMN IF NOT EXISTS session_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_runs ADD COLUMN IF NOT EXISTS request_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_runs ADD COLUMN IF NOT EXISTS idempotency_key TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_runs ADD COLUMN IF NOT EXISTS workflow_version TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_runs ADD COLUMN IF NOT EXISTS current_node TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_runs ADD COLUMN IF NOT EXISTS checkpoint_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_runs ADD COLUMN IF NOT EXISTS interrupt_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_runs ADD COLUMN IF NOT EXISTS execution_token TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_thread_deletions ADD COLUMN IF NOT EXISTS artifact_run_ids_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE graph_session_deletions ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending'",
            "ALTER TABLE graph_session_deletions ADD COLUMN IF NOT EXISTS last_error TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_effects ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_effects ADD COLUMN IF NOT EXISTS execution_token TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_effects ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'claimed'",
            "ALTER TABLE graph_effects ADD COLUMN IF NOT EXISTS result_json TEXT",
            "ALTER TABLE graph_effects ADD COLUMN IF NOT EXISTS error TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_effects ADD COLUMN IF NOT EXISTS claimed_at TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_effects ADD COLUMN IF NOT EXISTS updated_at TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE graph_effects ADD COLUMN IF NOT EXISTS finished_at TEXT",
            "CREATE INDEX IF NOT EXISTS idx_graph_runs_thread_updated ON graph_runs(thread_id, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_graph_runs_user_session ON graph_runs(user_id, session_id, started_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_graph_runs_updated ON graph_runs(updated_at)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_runs_request ON graph_runs(user_id, graph_name, request_id) WHERE request_id <> ''",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_runs_idempotency ON graph_runs(user_id, graph_name, idempotency_key) WHERE idempotency_key <> ''",
            "CREATE INDEX IF NOT EXISTS idx_graph_thread_deletions_pending ON graph_thread_deletions(status, updated_at)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_session_deletions_owner ON graph_session_deletions(user_id, session_id)",
            "CREATE INDEX IF NOT EXISTS idx_graph_session_deletions_pending ON graph_session_deletions(status, updated_at)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_effects_key ON graph_effects(run_id, node, effect)",
            "CREATE INDEX IF NOT EXISTS idx_graph_effects_run_claimed ON graph_effects(run_id, claimed_at, node, effect)",
        )
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)

    def verify_schema(self) -> None:
        """Fail fast when the independent migration step was not executed."""

        try:
            from langgraph.checkpoint.postgres.base import MIGRATIONS

            required_checkpoint_migration = len(MIGRATIONS) - 1
            with self.pool.connection() as connection:
                with connection.cursor() as cursor:
                    # Verify the official PostgresSaver schema as well as the
                    # application ledger. Worker startup must fail before the
                    # first request if the deployment migration was skipped.
                    cursor.execute(
                        "SELECT v FROM checkpoint_migrations LIMIT 0"
                    )
                    cursor.execute(
                        "SELECT MAX(v) AS latest_v FROM checkpoint_migrations"
                    )
                    migration_row = cursor.fetchone()
                    if isinstance(migration_row, Mapping):
                        latest_checkpoint_migration = migration_row.get("latest_v")
                    elif migration_row:
                        latest_checkpoint_migration = migration_row[0]
                    else:
                        latest_checkpoint_migration = None
                    if (
                        latest_checkpoint_migration is None
                        or int(latest_checkpoint_migration)
                        < required_checkpoint_migration
                    ):
                        raise RuntimeError(
                            "PostgresSaver checkpoint migrations are outdated"
                        )
                    cursor.execute(
                        """
                        SELECT thread_id, checkpoint_ns, checkpoint_id,
                               parent_checkpoint_id, checkpoint, metadata
                          FROM checkpoints
                         LIMIT 0
                        """
                    )
                    cursor.execute(
                        """
                        SELECT thread_id, checkpoint_ns, channel, version,
                               type, blob
                          FROM checkpoint_blobs
                         LIMIT 0
                        """
                    )
                    cursor.execute(
                        """
                        SELECT thread_id, checkpoint_ns, checkpoint_id,
                               task_id, task_path, idx, channel, type, blob
                          FROM checkpoint_writes
                         LIMIT 0
                        """
                    )
                    cursor.execute(
                        """
                        SELECT run_id, thread_id, user_id, session_id, request_id,
                               idempotency_key, workflow_version, current_node,
                               checkpoint_id, interrupt_id, execution_token,
                               status, durability, started_at, updated_at,
                               finished_at, error, metadata_json
                          FROM graph_runs
                         LIMIT 0
                        """
                    )
                    cursor.execute(
                        """
                        SELECT deletion_id, thread_id, user_id, session_id,
                               artifact_run_ids_json, reason, status,
                               requested_at, updated_at, last_error
                          FROM graph_thread_deletions
                         LIMIT 0
                        """
                    )
                    cursor.execute(
                        """
                        SELECT deletion_id, user_id, session_id, reason,
                               status, requested_at, updated_at, last_error
                          FROM graph_session_deletions
                         LIMIT 0
                        """
                    )
                    cursor.execute(
                        """
                        SELECT run_id, node, effect, user_id, execution_token,
                               status, result_json, error, claimed_at, updated_at,
                               finished_at
                          FROM graph_effects
                         LIMIT 0
                        """
                    )
                    required_indexes = tuple(
                        sorted(POSTGRES_REQUIRED_UNIQUE_INDEXES)
                    )
                    cursor.execute(
                        """
                        SELECT indexname, indexdef
                          FROM pg_indexes
                         WHERE schemaname = current_schema()
                           AND indexname = ANY(%s)
                        """,
                        (list(required_indexes),),
                    )
                    index_rows = cursor.fetchall()
                    index_definitions: dict[str, str] = {}
                    for row in index_rows:
                        if isinstance(row, Mapping):
                            name = str(row.get("indexname") or "")
                            definition = str(row.get("indexdef") or "")
                        else:
                            name = str(row[0] or "")
                            definition = str(row[1] or "")
                        if name:
                            index_definitions[name] = definition
                    for name, columns in POSTGRES_REQUIRED_UNIQUE_INDEXES.items():
                        definition = " ".join(
                            index_definitions.get(name, "").replace('"', "").split()
                        ).lower()
                        expected_columns = f"({', '.join(columns)})"
                        if (
                            "create unique index" not in definition
                            or expected_columns not in definition
                        ):
                            raise RuntimeError(
                                f"Required unique index {name} is missing or invalid"
                            )
        except Exception as exc:
            raise GraphPersistenceConfigurationError(
                "PostgreSQL graph persistence schema is missing or outdated; "
                "run scripts/setup_langgraph_checkpointer.py first"
            ) from exc

    @staticmethod
    def _record(row: Mapping[str, Any] | None) -> GraphRunRecord | None:
        if row is None:
            return None
        raw_metadata = row["metadata_json"] or "{}"
        if isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)
        else:
            try:
                metadata = json.loads(raw_metadata)
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return GraphRunRecord(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            graph_name=row["graph_name"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            request_id=row["request_id"],
            idempotency_key=row["idempotency_key"],
            workflow_version=row["workflow_version"],
            current_node=row["current_node"] or "",
            checkpoint_id=row["checkpoint_id"] or "",
            interrupt_id=row["interrupt_id"] or "",
            execution_token=row["execution_token"] or "",
            status=row["status"],
            durability=row["durability"],
            started_at=str(row["started_at"]),
            updated_at=str(row["updated_at"]),
            finished_at=(
                None if row["finished_at"] is None else str(row["finished_at"])
            ),
            error=row["error"] or "",
            metadata=metadata,
        )

    def start_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        graph_name: str,
        user_id: str,
        session_id: str,
        request_id: str,
        workflow_version: str,
        current_node: str = "",
        execution_token: str | None = None,
        idempotency_key: str | None = None,
        durability: str = "async",
        metadata: Mapping[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> GraphRunRecord:
        run_id = _required_identifier(run_id, "run_id")
        thread_id = _required_identifier(thread_id, "thread_id")
        graph_name = _required_identifier(graph_name, "graph_name")
        user_id = _required_identifier(user_id, "user_id")
        session_id = _required_identifier(session_id, "session_id")
        request_id = _required_identifier(request_id, "request_id")
        workflow_version = _required_identifier(
            workflow_version, "workflow_version"
        )
        idempotency_key = _required_identifier(
            idempotency_key or request_id, "idempotency_key"
        )
        durability = _validate_durability(durability)
        execution_token = _validate_execution_token(
            execution_token or uuid4().hex
        )
        timestamp = _timestamp(started_at)
        metadata_json = _metadata_json(metadata)
        request_digest = _request_digest(metadata)
        with self.pool.connection() as connection:
            # The saver pool is autocommit=True; keep the xact lock and INSERT
            # in one explicit transaction.
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    self._acquire_session_lock(
                        cursor,
                        user_id=user_id,
                        session_id=session_id,
                    )
                    cursor.execute(
                    """
                    INSERT INTO graph_runs (
                        run_id, thread_id, graph_name, user_id, session_id,
                        request_id, idempotency_key, workflow_version, current_node,
                        execution_token, status, durability, started_at, updated_at, finished_at,
                        error, metadata_json
                    )
                    SELECT
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'running', %s, %s, %s, NULL, '', %s
                     WHERE NOT EXISTS (
                        SELECT 1 FROM graph_session_deletions
                         WHERE user_id = %s AND session_id = %s
                     )
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        run_id,
                        thread_id,
                        graph_name,
                        user_id,
                        session_id,
                        request_id,
                        idempotency_key,
                        workflow_version,
                        str(current_node or ""),
                        execution_token,
                        durability,
                        timestamp,
                        timestamp,
                        metadata_json,
                        user_id,
                        session_id,
                    ),
                )
                    inserted = cursor.rowcount == 1
        if inserted:
            record = self.get_run(run_id)
            if record is None:  # pragma: no cover - committed INSERT invariant
                raise GraphPersistenceError("created graph run could not be reloaded")
            return record

        if self.is_session_deletion_requested(session_id, user_id=user_id):
            raise GraphSessionDeletionRequestedError(
                user_id=user_id,
                session_id=session_id,
            )
        return _resolve_run_start_conflict(
            run_id=run_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            request_digest=request_digest,
            by_run_id=self.get_run(run_id),
            by_request_id=self.get_run_by_request_id(
                user_id, graph_name, request_id
            ),
            by_idempotency_key=self.get_run_by_idempotency_key(
                user_id, graph_name, idempotency_key
            ),
        )

    def update_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str = "",
        current_node: str | None = None,
        checkpoint_id: str | None = None,
        interrupt_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None:
        run_id = _required_identifier(run_id, "run_id")
        status = _validate_status(status)
        timestamp = _timestamp(updated_at)
        finished_at = timestamp if status in TERMINAL_RUN_STATUSES else None
        with self.pool.connection() as connection:
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    if not self._acquire_unscoped_run_session_lock(
                        cursor,
                        run_id=run_id,
                    ):
                        return None
                    cursor.execute(
                        """
                        UPDATE graph_runs
                           SET status = %s, updated_at = %s, finished_at = %s,
                               error = %s, current_node = COALESCE(%s, current_node),
                               checkpoint_id = COALESCE(%s, checkpoint_id),
                               interrupt_id = COALESCE(%s, interrupt_id)
                         WHERE run_id = %s
                           AND NOT EXISTS (
                               SELECT 1
                                 FROM graph_session_deletions AS deletion
                                WHERE deletion.user_id = graph_runs.user_id
                                  AND deletion.session_id = graph_runs.session_id
                           )
                        """,
                        (
                            status,
                            timestamp,
                            finished_at,
                            str(error or ""),
                            None if current_node is None else str(current_node),
                            None if checkpoint_id is None else str(checkpoint_id),
                            None if interrupt_id is None else str(interrupt_id),
                            run_id,
                        ),
                    )
                    if cursor.rowcount == 0:
                        return None
        return self.get_run(run_id)

    def heartbeat_run(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        current_node: str,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None:
        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        execution_token = _validate_execution_token(execution_token)
        current_node = _required_identifier(current_node, "current_node")
        timestamp = _timestamp(updated_at)
        with self.pool.connection() as connection:
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    if not self._acquire_run_session_lock(
                        cursor,
                        run_id=run_id,
                        user_id=user_id,
                    ):
                        return None
                    cursor.execute(
                        """
                        UPDATE graph_runs
                           SET current_node = %s, updated_at = %s
                         WHERE run_id = %s AND user_id = %s
                           AND status = 'running' AND execution_token = %s
                           AND NOT EXISTS (
                               SELECT 1
                                 FROM graph_session_deletions AS deletion
                                WHERE deletion.user_id = graph_runs.user_id
                                  AND deletion.session_id = graph_runs.session_id
                           )
                        """,
                        (
                            current_node,
                            timestamp,
                            run_id,
                            user_id,
                            execution_token,
                        ),
                    )
                    if cursor.rowcount != 1:
                        return None
        return self.get_owned_run(run_id, user_id=user_id)

    def finish_run(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        status: str,
        current_node: str,
        error: str = "",
        checkpoint_id: str | None = None,
        interrupt_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None:
        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        execution_token = _validate_execution_token(execution_token)
        status = _validate_finish_status(status)
        current_node = _required_identifier(current_node, "current_node")
        timestamp = _timestamp(updated_at)
        with self.pool.connection() as connection:
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    if not self._acquire_run_session_lock(
                        cursor,
                        run_id=run_id,
                        user_id=user_id,
                    ):
                        return None
                    cursor.execute(
                        """
                        UPDATE graph_runs
                           SET status = %s, current_node = %s, updated_at = %s,
                               finished_at = %s, error = %s,
                               checkpoint_id = COALESCE(%s, checkpoint_id),
                               interrupt_id = COALESCE(%s, interrupt_id)
                         WHERE run_id = %s AND user_id = %s
                           AND status = 'running' AND execution_token = %s
                           AND NOT EXISTS (
                               SELECT 1
                                 FROM graph_session_deletions AS deletion
                                WHERE deletion.user_id = graph_runs.user_id
                                  AND deletion.session_id = graph_runs.session_id
                           )
                        """,
                        (
                            status,
                            current_node,
                            timestamp,
                            timestamp,
                            str(error or ""),
                            None if checkpoint_id is None else str(checkpoint_id),
                            None if interrupt_id is None else str(interrupt_id),
                            run_id,
                            user_id,
                            execution_token,
                        ),
                    )
                    if cursor.rowcount != 1:
                        return None
        return self.get_owned_run(run_id, user_id=user_id)

    def claim_recovery(
        self,
        run_id: str,
        *,
        user_id: str,
        expected_updated_at: str | datetime,
        new_execution_token: str,
    ) -> bool:
        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        expected = _expected_updated_at(expected_updated_at)
        new_execution_token = _validate_execution_token(new_execution_token)
        timestamp = _new_cas_timestamp(expected)
        with self.pool.connection() as connection:
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    if not self._acquire_run_session_lock(
                        cursor,
                        run_id=run_id,
                        user_id=user_id,
                    ):
                        return False
                    cursor.execute(
                        """
                        UPDATE graph_runs
                           SET status = 'running', execution_token = %s,
                               updated_at = %s, finished_at = NULL, error = ''
                         WHERE run_id = %s AND user_id = %s
                           AND status IN ('failed', 'running') AND updated_at = %s
                           AND NOT EXISTS (
                               SELECT 1 FROM graph_session_deletions AS deletion
                                WHERE deletion.user_id = graph_runs.user_id
                                  AND deletion.session_id = graph_runs.session_id
                           )
                        """,
                        (new_execution_token, timestamp, run_id, user_id, expected),
                    )
                    return cursor.rowcount == 1

    def claim_resume(
        self,
        run_id: str,
        *,
        user_id: str,
        expected_checkpoint_id: str | None = None,
        expected_interrupt_id: str | None = None,
        new_execution_token: str | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        """Atomically claim one interrupted run for resume exactly once."""

        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        checkpoint_id, interrupt_id = _resume_claim_expectations(
            expected_checkpoint_id, expected_interrupt_id
        )
        execution_token = _validate_execution_token(
            new_execution_token or uuid4().hex
        )
        conditions = ["run_id = %s", "user_id = %s", "status = 'interrupted'"]
        params: list[Any] = [execution_token, _timestamp(updated_at), run_id, user_id]
        if checkpoint_id is not None:
            conditions.append("checkpoint_id = %s")
            params.append(checkpoint_id)
        if interrupt_id is not None:
            conditions.append("interrupt_id = %s")
            params.append(interrupt_id)
        with self.pool.connection() as connection:
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    if not self._acquire_run_session_lock(
                        cursor,
                        run_id=run_id,
                        user_id=user_id,
                    ):
                        return False
                    cursor.execute(
                        f"""
                        UPDATE graph_runs
                           SET status = 'running', execution_token = %s, updated_at = %s,
                               finished_at = NULL, error = ''
                         WHERE {' AND '.join(conditions)}
                           AND NOT EXISTS (
                               SELECT 1 FROM graph_session_deletions AS deletion
                                WHERE deletion.user_id = graph_runs.user_id
                                  AND deletion.session_id = graph_runs.session_id
                           )
                        """,
                        tuple(params),
                    )
                    return cursor.rowcount == 1

    def _fetch_one(self, query: str, params: tuple[Any, ...]) -> GraphRunRecord | None:
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
        return self._record(row)

    def get_run(self, run_id: str) -> GraphRunRecord | None:
        run_id = _required_identifier(run_id, "run_id")
        return self._fetch_one(
            "SELECT * FROM graph_runs WHERE run_id = %s", (run_id,)
        )

    def get_owned_run(self, run_id: str, *, user_id: str) -> GraphRunRecord | None:
        user_id = _required_identifier(user_id, "user_id")
        run_id = _required_identifier(run_id, "run_id")
        return self._fetch_one(
            "SELECT * FROM graph_runs WHERE run_id = %s AND user_id = %s",
            (run_id, user_id),
        )

    def get_run_by_idempotency_key(
        self, user_id: str, graph_name: str, idempotency_key: str
    ) -> GraphRunRecord | None:
        user_id = _required_identifier(user_id, "user_id")
        graph_name = _required_identifier(graph_name, "graph_name")
        idempotency_key = _required_identifier(
            idempotency_key, "idempotency_key"
        )
        return self._fetch_one(
            """
            SELECT * FROM graph_runs
             WHERE user_id = %s AND graph_name = %s AND idempotency_key = %s
            """,
            (user_id, graph_name, idempotency_key),
        )

    def get_run_by_request_id(
        self, user_id: str, graph_name: str, request_id: str
    ) -> GraphRunRecord | None:
        user_id = _required_identifier(user_id, "user_id")
        graph_name = _required_identifier(graph_name, "graph_name")
        request_id = _required_identifier(request_id, "request_id")
        return self._fetch_one(
            """
            SELECT * FROM graph_runs
             WHERE user_id = %s AND graph_name = %s AND request_id = %s
            """,
            (user_id, graph_name, request_id),
        )

    def _fetch_many(
        self, query: str, params: tuple[Any, ...]
    ) -> list[GraphRunRecord]:
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
        return [record for row in rows if (record := self._record(row)) is not None]

    def list_thread_runs(
        self, thread_id: str, *, limit: int = 100
    ) -> list[GraphRunRecord]:
        thread_id = _required_identifier(thread_id, "thread_id")
        return self._fetch_many(
            """
            SELECT * FROM graph_runs
             WHERE thread_id = %s
             ORDER BY started_at DESC, run_id DESC
             LIMIT %s
            """,
            (thread_id, max(1, int(limit))),
        )

    def list_session_runs(
        self, session_id: str, *, user_id: str, limit: int = 100
    ) -> list[GraphRunRecord]:
        user_id = _required_identifier(user_id, "user_id")
        session_id = _required_identifier(session_id, "session_id")
        return self._fetch_many(
            """
            SELECT * FROM graph_runs
             WHERE user_id = %s AND session_id = %s
             ORDER BY started_at DESC, run_id DESC
             LIMIT %s
            """,
            (user_id, session_id, max(1, int(limit))),
        )

    def thread_belongs_to_user(self, thread_id: str, *, user_id: str) -> bool:
        user_id = _required_identifier(user_id, "user_id")
        thread_id = _required_identifier(thread_id, "thread_id")
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1 FROM graph_runs
                     WHERE user_id = %s AND thread_id = %s
                     LIMIT 1
                    """,
                    (user_id, thread_id),
                )
                return cursor.fetchone() is not None

    def list_session_thread_ids(
        self, session_id: str, *, user_id: str
    ) -> list[str]:
        session_id = _required_identifier(session_id, "session_id")
        user_id = _required_identifier(user_id, "user_id")
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT thread_id FROM graph_runs
                     WHERE user_id = %s AND session_id = %s
                     ORDER BY thread_id
                    """,
                    (user_id, session_id),
                )
                rows = cursor.fetchall()
        return [str(row["thread_id"]) for row in rows]

    def claim_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        claimed_at: datetime | None = None,
        allow_reclaim: bool = False,
        stale_before: datetime | None = None,
        stale_seconds: float | None = None,
    ) -> bool:
        """Atomically reserve or explicitly reclaim a fenced side-effect key."""

        run_id, node, effect, user_id = _effect_identity(
            run_id, node, effect, user_id
        )
        token = (
            ""
            if execution_token is None
            else _validate_execution_token(execution_token)
        )
        timestamp, stale_cutoff = _effect_claim_window(
            claimed_at=claimed_at,
            allow_reclaim=allow_reclaim,
            stale_before=stale_before,
            stale_seconds=stale_seconds,
        )
        conflict_clause = "ON CONFLICT(run_id, node, effect) DO NOTHING"
        reclaim_params: tuple[Any, ...] = ()
        if allow_reclaim:
            conflict_clause = """
                ON CONFLICT(run_id, node, effect) DO UPDATE SET
                    user_id = excluded.user_id,
                    execution_token = excluded.execution_token,
                    status = 'claimed',
                    result_json = NULL,
                    error = '',
                    claimed_at = excluded.claimed_at,
                    updated_at = excluded.updated_at,
                    finished_at = NULL
                WHERE graph_effects.status = 'failed'
                   OR (
                        graph_effects.status = 'claimed'
                        AND CAST(%s AS TEXT) IS NOT NULL
                        AND graph_effects.updated_at < CAST(%s AS TEXT)
                   )
            """
            reclaim_params = (stale_cutoff, stale_cutoff)
        with self.pool.connection() as connection:
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    if not self._acquire_run_session_lock(
                        cursor,
                        run_id=run_id,
                        user_id=user_id,
                    ):
                        return False
                    cursor.execute(
                        f"""
                        INSERT INTO graph_effects (
                            run_id, node, effect, user_id, execution_token, status,
                            result_json, error, claimed_at, updated_at, finished_at
                        )
                        SELECT run_id, %s, %s, user_id, %s, 'claimed', NULL, '',
                               %s, %s, NULL
                          FROM graph_runs
                         WHERE run_id = %s AND user_id = %s
                           AND (
                                %s = '' OR (
                                    status = 'running' AND execution_token = %s
                                )
                           )
                           AND NOT EXISTS (
                               SELECT 1
                                 FROM graph_session_deletions AS deletion
                                WHERE deletion.user_id = graph_runs.user_id
                                  AND deletion.session_id = graph_runs.session_id
                           )
                        {conflict_clause}
                        """,
                        (
                            node,
                            effect,
                            token,
                            timestamp,
                            timestamp,
                            run_id,
                            user_id,
                            token,
                            token,
                            *reclaim_params,
                        ),
                    )
                    return cursor.rowcount == 1

    def complete_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        result: Any = None,
        updated_at: datetime | None = None,
    ) -> GraphEffectRecord | None:
        run_id, node, effect, user_id = _effect_identity(
            run_id, node, effect, user_id
        )
        result_json = _result_json(result)
        timestamp = _timestamp(updated_at)
        token = (
            ""
            if execution_token is None
            else _validate_execution_token(execution_token)
        )
        with self.pool.connection() as connection:
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    if not self._acquire_run_session_lock(
                        cursor,
                        run_id=run_id,
                        user_id=user_id,
                    ):
                        return None
                    cursor.execute(
                        """
                        UPDATE graph_effects
                           SET status = 'completed', result_json = %s, error = '',
                               updated_at = %s, finished_at = %s
                         WHERE run_id = %s AND node = %s AND effect = %s
                           AND user_id = %s AND status = 'claimed'
                           AND EXISTS (
                               SELECT 1 FROM graph_runs AS current_run
                                WHERE current_run.run_id = graph_effects.run_id
                                  AND current_run.user_id = %s
                                  AND current_run.status = 'running'
                                  AND (
                                       %s = '' OR (
                                           graph_effects.execution_token = %s
                                           AND current_run.execution_token = %s
                                       )
                                  )
                                  AND NOT EXISTS (
                                      SELECT 1
                                        FROM graph_session_deletions AS deletion
                                       WHERE deletion.user_id = current_run.user_id
                                         AND deletion.session_id = current_run.session_id
                                  )
                           )
                        """,
                        (
                            result_json,
                            timestamp,
                            timestamp,
                            run_id,
                            node,
                            effect,
                            user_id,
                            user_id,
                            token,
                            token,
                            token,
                        ),
                    )
                    if cursor.rowcount != 1:
                        return None
        return self.get_effect(run_id, node, effect, user_id=user_id)

    def fail_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        error: str,
        updated_at: datetime | None = None,
    ) -> GraphEffectRecord | None:
        run_id, node, effect, user_id = _effect_identity(
            run_id, node, effect, user_id
        )
        error = _required_identifier(error, "error")[:4000]
        timestamp = _timestamp(updated_at)
        token = (
            ""
            if execution_token is None
            else _validate_execution_token(execution_token)
        )
        with self.pool.connection() as connection:
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    if not self._acquire_run_session_lock(
                        cursor,
                        run_id=run_id,
                        user_id=user_id,
                    ):
                        return None
                    cursor.execute(
                        """
                        UPDATE graph_effects
                           SET status = 'failed', result_json = NULL, error = %s,
                               updated_at = %s, finished_at = %s
                         WHERE run_id = %s AND node = %s AND effect = %s
                           AND user_id = %s AND status = 'claimed'
                           AND EXISTS (
                               SELECT 1 FROM graph_runs AS current_run
                                WHERE current_run.run_id = graph_effects.run_id
                                  AND current_run.user_id = %s
                                  AND current_run.status = 'running'
                                  AND (
                                       %s = '' OR (
                                           graph_effects.execution_token = %s
                                           AND current_run.execution_token = %s
                                       )
                                  )
                                  AND NOT EXISTS (
                                      SELECT 1
                                        FROM graph_session_deletions AS deletion
                                       WHERE deletion.user_id = current_run.user_id
                                         AND deletion.session_id = current_run.session_id
                                  )
                           )
                        """,
                        (
                            error,
                            timestamp,
                            timestamp,
                            run_id,
                            node,
                            effect,
                            user_id,
                            user_id,
                            token,
                            token,
                            token,
                        ),
                    )
                    if cursor.rowcount != 1:
                        return None
        return self.get_effect(run_id, node, effect, user_id=user_id)

    def get_effect(
        self, run_id: str, node: str, effect: str, *, user_id: str
    ) -> GraphEffectRecord | None:
        run_id, node, effect, user_id = _effect_identity(
            run_id, node, effect, user_id
        )
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM graph_effects
                     WHERE run_id = %s AND node = %s AND effect = %s
                       AND user_id = %s
                    """,
                    (run_id, node, effect, user_id),
                )
                row = cursor.fetchone()
        return _effect_record(row)

    def list_run_effects(
        self, run_id: str, *, user_id: str
    ) -> list[GraphEffectRecord]:
        run_id = _required_identifier(run_id, "run_id")
        user_id = _required_identifier(user_id, "user_id")
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM graph_effects
                     WHERE run_id = %s AND user_id = %s
                     ORDER BY claimed_at ASC, node ASC, effect ASC
                    """,
                    (run_id, user_id),
                )
                rows = cursor.fetchall()
        return [
            record
            for row in rows
            if (record := _effect_record(row)) is not None
        ]

    def enqueue_session_deletion(
        self,
        session_id: str,
        *,
        user_id: str,
        reason: str,
        requested_at: datetime | None = None,
    ) -> GraphSessionDeletionRecord:
        session_id = _required_identifier(session_id, "session_id")
        user_id = _required_identifier(user_id, "user_id")
        reason = _required_identifier(reason, "reason")
        timestamp = _timestamp(requested_at)
        with self.pool.connection() as connection:
            with self._transaction(connection):
                with connection.cursor() as cursor:
                    self._acquire_session_lock(
                        cursor,
                        user_id=user_id,
                        session_id=session_id,
                    )
                    cursor.execute(
                    """
                    INSERT INTO graph_session_deletions (
                        deletion_id, user_id, session_id, reason, status,
                        requested_at, updated_at, last_error
                    ) VALUES (%s, %s, %s, %s, 'pending', %s, %s, '')
                    ON CONFLICT(user_id, session_id) DO UPDATE SET
                        reason = excluded.reason,
                        status = 'pending',
                        updated_at = excluded.updated_at,
                        last_error = ''
                    RETURNING *
                    """,
                    (
                        str(uuid4()),
                        user_id,
                        session_id,
                        reason,
                        timestamp,
                        timestamp,
                    ),
                )
                    row = cursor.fetchone()
        record = _session_deletion_record(row)
        assert record is not None
        return record

    def is_session_deletion_requested(
        self, session_id: str, *, user_id: str
    ) -> bool:
        session_id = _required_identifier(session_id, "session_id")
        user_id = _required_identifier(user_id, "user_id")
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1 FROM graph_session_deletions
                     WHERE user_id = %s AND session_id = %s
                     LIMIT 1
                    """,
                    (user_id, session_id),
                )
                return cursor.fetchone() is not None

    def list_pending_session_deletions(
        self, *, limit: int = 100
    ) -> list[GraphSessionDeletionRecord]:
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM graph_session_deletions
                     WHERE status = 'pending'
                     ORDER BY updated_at ASC, deletion_id ASC
                     LIMIT %s
                    """,
                    (max(1, int(limit)),),
                )
                rows = cursor.fetchall()
        return [
            record
            for row in rows
            if (record := _session_deletion_record(row)) is not None
        ]

    def update_session_deletion(
        self,
        deletion_id: str,
        *,
        status: str,
        last_error: str = "",
        updated_at: datetime | None = None,
    ) -> GraphSessionDeletionRecord | None:
        deletion_id = _required_identifier(deletion_id, "deletion_id")
        status = str(status or "").strip().lower()
        if status not in {"pending", "completed"}:
            raise ValueError("deletion status must be pending or completed")
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE graph_session_deletions
                       SET status = %s, updated_at = %s, last_error = %s
                     WHERE deletion_id = %s
                    RETURNING *
                    """,
                    (
                        status,
                        _timestamp(updated_at),
                        str(last_error or "")[:2000],
                        deletion_id,
                    ),
                )
                row = cursor.fetchone()
        return _session_deletion_record(row)

    def enqueue_thread_deletion(
        self,
        thread_id: str,
        *,
        user_id: str,
        session_id: str,
        artifact_run_ids: tuple[str, ...] = (),
        reason: str,
        requested_at: datetime | None = None,
    ) -> GraphThreadDeletionRecord:
        thread_id = _required_identifier(thread_id, "thread_id")
        user_id = _required_identifier(user_id, "user_id")
        session_id = _required_identifier(session_id, "session_id")
        reason = _required_identifier(reason, "reason")
        artifact_run_ids_json = _artifact_run_ids_json(artifact_run_ids)
        timestamp = _timestamp(requested_at)
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO graph_thread_deletions (
                        deletion_id, thread_id, user_id, session_id,
                        artifact_run_ids_json, reason, status, requested_at,
                        updated_at, last_error
                    ) VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s, %s, '')
                    ON CONFLICT(thread_id) DO UPDATE SET
                        user_id = excluded.user_id,
                        session_id = excluded.session_id,
                        artifact_run_ids_json = CASE
                            WHEN excluded.artifact_run_ids_json = '[]'
                            THEN graph_thread_deletions.artifact_run_ids_json
                            ELSE excluded.artifact_run_ids_json
                        END,
                        reason = excluded.reason,
                        status = 'pending',
                        updated_at = excluded.updated_at,
                        last_error = ''
                    RETURNING *
                    """,
                    (
                        str(uuid4()),
                        thread_id,
                        user_id,
                        session_id,
                        artifact_run_ids_json,
                        reason,
                        timestamp,
                        timestamp,
                    ),
                )
                row = cursor.fetchone()
        record = _thread_deletion_record(row)
        assert record is not None
        return record

    def is_thread_deletion_pending(
        self, thread_id: str, *, user_id: str
    ) -> bool:
        thread_id = _required_identifier(thread_id, "thread_id")
        user_id = _required_identifier(user_id, "user_id")
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1 FROM graph_thread_deletions
                     WHERE thread_id = %s AND user_id = %s
                       AND status = 'pending'
                     LIMIT 1
                    """,
                    (thread_id, user_id),
                )
                return cursor.fetchone() is not None

    def is_thread_deletion_requested(
        self, thread_id: str, *, user_id: str
    ) -> bool:
        thread_id = _required_identifier(thread_id, "thread_id")
        user_id = _required_identifier(user_id, "user_id")
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1 FROM graph_thread_deletions
                     WHERE thread_id = %s AND user_id = %s
                     LIMIT 1
                    """,
                    (thread_id, user_id),
                )
                return cursor.fetchone() is not None

    def list_pending_deletions(
        self, *, limit: int = 100
    ) -> list[GraphThreadDeletionRecord]:
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM graph_thread_deletions
                     WHERE status = 'pending'
                     ORDER BY updated_at ASC, deletion_id ASC
                     LIMIT %s
                    """,
                    (max(1, int(limit)),),
                )
                rows = cursor.fetchall()
        return [
            record
            for row in rows
            if (record := _thread_deletion_record(row)) is not None
        ]

    def update_thread_deletion(
        self,
        deletion_id: str,
        *,
        status: str,
        last_error: str = "",
        updated_at: datetime | None = None,
    ) -> GraphThreadDeletionRecord | None:
        deletion_id = _required_identifier(deletion_id, "deletion_id")
        status = str(status or "").strip().lower()
        if status not in {"pending", "completed"}:
            raise ValueError("deletion status must be pending or completed")
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE graph_thread_deletions
                       SET status = %s, updated_at = %s, last_error = %s
                     WHERE deletion_id = %s
                    RETURNING *
                    """,
                    (
                        status,
                        _timestamp(updated_at),
                        str(last_error or "")[:2000],
                        deletion_id,
                    ),
                )
                row = cursor.fetchone()
        return _thread_deletion_record(row)

    def list_expired_thread_ids(
        self, cutoff: datetime, *, limit: int = 100
    ) -> list[str]:
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT thread_id, MAX(updated_at) AS last_updated_at
                      FROM graph_runs
                     GROUP BY thread_id
                    HAVING MAX(updated_at) < %s
                     ORDER BY last_updated_at ASC, thread_id ASC
                     LIMIT %s
                    """,
                    (_timestamp(cutoff), max(1, int(limit))),
                )
                rows = cursor.fetchall()
        return [str(row["thread_id"]) for row in rows]

    def delete_thread(self, thread_id: str) -> int:
        thread_id = _required_identifier(thread_id, "thread_id")
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM graph_effects
                     WHERE run_id IN (
                        SELECT run_id FROM graph_runs WHERE thread_id = %s
                     )
                    """,
                    (thread_id,),
                )
                cursor.execute(
                    "DELETE FROM graph_runs WHERE thread_id = %s", (thread_id,)
                )
                return int(cursor.rowcount)

    def close(self) -> None:
        # The CheckpointerHandle owns and closes the shared connection pool.
        return None


def setup_postgres_persistence(
    config: GraphPersistenceConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Set up both official checkpoint tables and the shared run ledger."""

    if config is not None and environ is not None:
        raise GraphPersistenceConfigurationError(
            "Pass either config or environ, not both"
        )
    resolved = config or GraphPersistenceConfig.from_env(environ)
    resolved.validate()
    if resolved.backend != "postgres" or resolved.run_ledger_backend != "postgres":
        raise GraphPersistenceConfigurationError(
            "PostgreSQL persistence setup requires postgres checkpointer and run ledger"
        )
    with create_checkpointer(resolved) as handle:
        setup = getattr(handle.saver, "setup", None)
        if not callable(setup):
            raise GraphPersistenceDependencyError(
                "The installed PostgresSaver does not expose setup()"
            )
        setup()
        PostgresGraphRunLedger(handle.resource).setup()


@dataclass(frozen=True)
class RetentionResult:
    cutoff_at: str
    candidate_count: int
    deleted_thread_ids: tuple[str, ...]
    failures: dict[str, str]


@dataclass(frozen=True)
class SessionDeletionResult:
    requested_thread_ids: tuple[str, ...]
    deleted_thread_ids: tuple[str, ...]
    failures: dict[str, str]


class GraphPersistenceRuntime:
    """Bundle a checkpointer with a backend-neutral run ledger."""

    def __init__(
        self,
        *,
        config: GraphPersistenceConfig,
        checkpointer_handle: CheckpointerHandle,
        run_ledger: GraphRunLedger,
    ) -> None:
        self.config = config
        self.checkpointer_handle = checkpointer_handle
        self.run_ledger = run_ledger
        self._closed = False

    @property
    def checkpointer(self) -> Any:
        return self.checkpointer_handle.saver

    def start_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        graph_name: str,
        user_id: str,
        session_id: str,
        request_id: str,
        workflow_version: str,
        current_node: str = "",
        execution_token: str | None = None,
        idempotency_key: str | None = None,
        durability: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> GraphRunRecord:
        return self.run_ledger.start_run(
            run_id=run_id,
            thread_id=thread_id,
            graph_name=graph_name,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            workflow_version=workflow_version,
            current_node=current_node,
            execution_token=execution_token,
            idempotency_key=idempotency_key,
            durability=durability or self.config.default_durability,
            metadata=metadata,
            started_at=started_at,
        )

    def update_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str = "",
        current_node: str | None = None,
        checkpoint_id: str | None = None,
        interrupt_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None:
        return self.run_ledger.update_run(
            run_id,
            status=status,
            error=error,
            current_node=current_node,
            checkpoint_id=checkpoint_id,
            interrupt_id=interrupt_id,
            updated_at=updated_at,
        )

    def heartbeat_run(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        current_node: str,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None:
        return self.run_ledger.heartbeat_run(
            run_id,
            user_id=user_id,
            execution_token=execution_token,
            current_node=current_node,
            updated_at=updated_at,
        )

    def finish_run(
        self,
        run_id: str,
        *,
        user_id: str,
        execution_token: str,
        status: str,
        current_node: str,
        error: str = "",
        checkpoint_id: str | None = None,
        interrupt_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> GraphRunRecord | None:
        return self.run_ledger.finish_run(
            run_id,
            user_id=user_id,
            execution_token=execution_token,
            status=status,
            current_node=current_node,
            error=error,
            checkpoint_id=checkpoint_id,
            interrupt_id=interrupt_id,
            updated_at=updated_at,
        )

    def claim_recovery(
        self,
        run_id: str,
        *,
        user_id: str,
        expected_updated_at: str | datetime,
        new_execution_token: str,
    ) -> bool:
        return self.run_ledger.claim_recovery(
            run_id,
            user_id=user_id,
            expected_updated_at=expected_updated_at,
            new_execution_token=new_execution_token,
        )

    def claim_resume(
        self,
        run_id: str,
        *,
        user_id: str,
        expected_checkpoint_id: str | None = None,
        expected_interrupt_id: str | None = None,
        new_execution_token: str | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        return self.run_ledger.claim_resume(
            run_id,
            user_id=user_id,
            expected_checkpoint_id=expected_checkpoint_id,
            expected_interrupt_id=expected_interrupt_id,
            new_execution_token=new_execution_token,
            updated_at=updated_at,
        )

    def get_owned_run(self, run_id: str, *, user_id: str) -> GraphRunRecord | None:
        return self.run_ledger.get_owned_run(run_id, user_id=user_id)

    def get_run_by_idempotency_key(
        self, user_id: str, graph_name: str, idempotency_key: str
    ) -> GraphRunRecord | None:
        return self.run_ledger.get_run_by_idempotency_key(
            user_id, graph_name, idempotency_key
        )

    def list_session_runs(
        self, session_id: str, *, user_id: str, limit: int = 100
    ) -> list[GraphRunRecord]:
        return self.run_ledger.list_session_runs(
            session_id, user_id=user_id, limit=limit
        )

    def is_session_deletion_requested(
        self, session_id: str, *, user_id: str
    ) -> bool:
        return self.run_ledger.is_session_deletion_requested(
            session_id,
            user_id=user_id,
        )

    def claim_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        claimed_at: datetime | None = None,
        allow_reclaim: bool = False,
        stale_before: datetime | None = None,
        stale_seconds: float | None = None,
    ) -> bool:
        return self.run_ledger.claim_effect(
            run_id,
            node,
            effect,
            user_id=user_id,
            execution_token=execution_token,
            claimed_at=claimed_at,
            allow_reclaim=allow_reclaim,
            stale_before=stale_before,
            stale_seconds=stale_seconds,
        )

    def complete_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        result: Any = None,
        updated_at: datetime | None = None,
    ) -> GraphEffectRecord | None:
        return self.run_ledger.complete_effect(
            run_id,
            node,
            effect,
            user_id=user_id,
            execution_token=execution_token,
            result=result,
            updated_at=updated_at,
        )

    def fail_effect(
        self,
        run_id: str,
        node: str,
        effect: str,
        *,
        user_id: str,
        execution_token: str | None = None,
        error: str,
        updated_at: datetime | None = None,
    ) -> GraphEffectRecord | None:
        return self.run_ledger.fail_effect(
            run_id,
            node,
            effect,
            user_id=user_id,
            execution_token=execution_token,
            error=error,
            updated_at=updated_at,
        )

    def get_effect(
        self, run_id: str, node: str, effect: str, *, user_id: str
    ) -> GraphEffectRecord | None:
        return self.run_ledger.get_effect(
            run_id, node, effect, user_id=user_id
        )

    def list_run_effects(
        self, run_id: str, *, user_id: str
    ) -> list[GraphEffectRecord]:
        return self.run_ledger.list_run_effects(run_id, user_id=user_id)

    def delete_thread(self, thread_id: str) -> int:
        """Delete checkpoints first, then remove the retry ledger rows."""

        thread_id = _required_identifier(thread_id, "thread_id")
        delete = getattr(self.checkpointer, "delete_thread", None)
        if not callable(delete):
            raise GraphPersistenceDependencyError(
                "The configured checkpointer does not support delete_thread()"
            )
        delete(thread_id)
        return self.run_ledger.delete_thread(thread_id)

    def delete_owned_thread(self, thread_id: str, *, user_id: str) -> int:
        if not self.run_ledger.thread_belongs_to_user(
            thread_id, user_id=user_id
        ):
            raise GraphPersistenceOwnershipError(
                "The graph thread does not belong to the requesting user"
            )
        return self.delete_thread(thread_id)

    def request_session_deletion(
        self,
        session_id: str,
        *,
        user_id: str,
        reason: str = "user_delete",
        delete_artifact_run: Callable[[str], Any] | None = None,
    ) -> SessionDeletionResult:
        """Persist tombstones before deleting every graph thread and artifact."""

        # This permanent owner/session barrier is the linearization point for
        # deletion. A start_run committed before it is visible to the following
        # enumeration; a start_run after it is atomically rejected by the INSERT.
        # Session IDs are never recycled: an explicit new session must use a new ID.
        barrier = self.run_ledger.enqueue_session_deletion(
            session_id,
            user_id=user_id,
            reason=reason,
        )
        try:
            return self._continue_session_deletion(
                barrier,
                delete_artifact_run=delete_artifact_run,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:2000]
            self.run_ledger.update_session_deletion(
                barrier.deletion_id,
                status="pending",
                last_error=error,
            )
            return SessionDeletionResult(
                requested_thread_ids=(),
                deleted_thread_ids=(),
                failures={f"session:{session_id}": error},
            )

    def _continue_session_deletion(
        self,
        barrier: GraphSessionDeletionRecord,
        *,
        delete_artifact_run: Callable[[str], Any] | None,
    ) -> SessionDeletionResult:
        """Materialize every thread tombstone for one durable session barrier."""

        session_id = barrier.session_id
        user_id = barrier.user_id
        thread_ids = self.run_ledger.list_session_thread_ids(
            session_id, user_id=user_id
        )
        deleted: list[str] = []
        failures: dict[str, str] = {}
        for thread_id in thread_ids:
            artifact_run_ids = (
                tuple(
                    record.run_id
                    for record in self.run_ledger.list_thread_runs(
                        thread_id, limit=10000
                    )
                    if record.user_id == user_id
                )
                if delete_artifact_run is not None
                else ()
            )
            tombstone = self.run_ledger.enqueue_thread_deletion(
                thread_id,
                user_id=user_id,
                session_id=session_id,
                artifact_run_ids=artifact_run_ids,
                reason=barrier.reason,
            )
            try:
                self._delete_tombstoned_thread(
                    tombstone,
                    delete_artifact_run=delete_artifact_run,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:2000]
                failures[thread_id] = error
                self.run_ledger.update_thread_deletion(
                    tombstone.deletion_id,
                    status="pending",
                    last_error=error,
                )
            else:
                deleted.append(thread_id)
                self.run_ledger.update_thread_deletion(
                    tombstone.deletion_id,
                    status="completed",
                )
        # Physical failures already have their own durable thread tombstones.
        # Completing the session job here prevents an ever-growing scan while
        # preserving independent retries for checkpoints and artifacts.
        self.run_ledger.update_session_deletion(
            barrier.deletion_id,
            status="completed",
        )
        return SessionDeletionResult(
            requested_thread_ids=tuple(thread_ids),
            deleted_thread_ids=tuple(deleted),
            failures=failures,
        )

    def is_thread_deletion_pending(
        self, thread_id: str, *, user_id: str
    ) -> bool:
        return self.run_ledger.is_thread_deletion_pending(
            thread_id, user_id=user_id
        )

    def is_thread_deletion_requested(
        self, thread_id: str, *, user_id: str
    ) -> bool:
        return self.run_ledger.is_thread_deletion_requested(
            thread_id, user_id=user_id
        )

    def retry_pending_deletions(
        self,
        *,
        limit: int = 100,
        delete_artifact_run: Callable[[str], Any] | None = None,
    ) -> SessionDeletionResult:
        requested: list[str] = []
        deleted: list[str] = []
        failures: dict[str, str] = {}
        session_jobs = self.run_ledger.list_pending_session_deletions(limit=limit)
        for barrier in session_jobs:
            try:
                result = self._continue_session_deletion(
                    barrier,
                    delete_artifact_run=delete_artifact_run,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:2000]
                failures[f"session:{barrier.session_id}"] = error
                self.run_ledger.update_session_deletion(
                    barrier.deletion_id,
                    status="pending",
                    last_error=error,
                )
            else:
                requested.extend(result.requested_thread_ids)
                deleted.extend(result.deleted_thread_ids)
                failures.update(result.failures)

        tombstones = self.run_ledger.list_pending_deletions(limit=limit)
        for tombstone in tombstones:
            requested.append(tombstone.thread_id)
            try:
                self._delete_tombstoned_thread(
                    tombstone,
                    delete_artifact_run=delete_artifact_run,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:2000]
                failures[tombstone.thread_id] = error
                self.run_ledger.update_thread_deletion(
                    tombstone.deletion_id,
                    status="pending",
                    last_error=error,
                )
            else:
                deleted.append(tombstone.thread_id)
                self.run_ledger.update_thread_deletion(
                    tombstone.deletion_id,
                    status="completed",
                )
        return SessionDeletionResult(
            requested_thread_ids=tuple(dict.fromkeys(requested)),
            deleted_thread_ids=tuple(dict.fromkeys(deleted)),
            failures=failures,
        )

    def _delete_tombstoned_thread(
        self,
        tombstone: GraphThreadDeletionRecord,
        *,
        delete_artifact_run: Callable[[str], Any] | None,
    ) -> None:
        """Run all physical phases before a tombstone may be completed."""

        self.delete_thread(tombstone.thread_id)
        if not tombstone.artifact_run_ids:
            return
        if delete_artifact_run is None:
            raise GraphPersistenceConfigurationError(
                "Artifact deletion callback is required for this pending tombstone"
            )

        first_error: Exception | None = None
        for run_id in tombstone.artifact_run_ids:
            try:
                delete_artifact_run(run_id)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def delete_expired_threads(
        self,
        *,
        now: datetime | None = None,
        retention_days: int | None = None,
        limit: int | None = None,
    ) -> RetentionResult:
        days = self.config.retention_days if retention_days is None else int(retention_days)
        batch_size = self.config.retention_batch_size if limit is None else int(limit)
        if days < 1 or batch_size < 1:
            raise ValueError("retention_days and limit must be positive")
        cutoff = _normalized_datetime(now) - timedelta(days=days)
        candidates = self.run_ledger.list_expired_thread_ids(
            cutoff, limit=batch_size
        )
        deleted: list[str] = []
        failures: dict[str, str] = {}
        for thread_id in candidates:
            try:
                self.delete_thread(thread_id)
            except Exception as exc:  # keep the ledger row so a later sweep retries
                failures[thread_id] = f"{type(exc).__name__}: {exc}"[:500]
            else:
                deleted.append(thread_id)
        return RetentionResult(
            cutoff_at=_timestamp(cutoff),
            candidate_count=len(candidates),
            deleted_thread_ids=tuple(deleted),
            failures=failures,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.run_ledger.close()
        finally:
            self.checkpointer_handle.close()

    def __enter__(self) -> "GraphPersistenceRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


GraphRunLedgerFactory = Callable[
    [GraphPersistenceConfig, CheckpointerHandle], GraphRunLedger
]


def create_graph_persistence(
    config: GraphPersistenceConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    run_ledger_factory: GraphRunLedgerFactory | None = None,
) -> GraphPersistenceRuntime:
    """Create the independent persistence runtime.

    Development defaults to SQLite. PostgreSQL checkpointers default to a
    PostgreSQL ledger sharing the same bounded pool. A custom implementation
    can still be injected through ``run_ledger_factory``.
    """

    if config is not None and environ is not None:
        raise GraphPersistenceConfigurationError(
            "Pass either config or environ, not both"
        )
    resolved = config or GraphPersistenceConfig.from_env(environ)
    resolved.validate()
    handle = create_checkpointer(resolved)
    try:
        if run_ledger_factory is None:
            if resolved.run_ledger_backend == "postgres":
                postgres_ledger = PostgresGraphRunLedger(handle.resource)
                postgres_ledger.verify_schema()
                ledger: GraphRunLedger = postgres_ledger
            else:
                ledger = SqliteGraphRunLedger(
                    resolved.run_ledger_sqlite_path,
                    busy_timeout_ms=resolved.sqlite_busy_timeout_ms,
                )
        else:
            ledger = run_ledger_factory(resolved, handle)
            if not isinstance(ledger, GraphRunLedger):
                raise TypeError(
                    "run_ledger_factory must return a GraphRunLedger implementation"
                )
    except Exception:
        handle.close()
        raise
    return GraphPersistenceRuntime(
        config=resolved,
        checkpointer_handle=handle,
        run_ledger=ledger,
    )


__all__ = [
    "CheckpointerHandle",
    "GraphPersistenceConfig",
    "GraphPersistenceConfigurationError",
    "GraphPersistenceConnectionError",
    "GraphPersistenceDependencyError",
    "GraphPersistenceError",
    "GraphPersistenceOwnershipError",
    "GraphRunIdentityConflictError",
    "GraphSessionDeletionRequestedError",
    "GraphPersistenceRuntime",
    "GraphEffectRecord",
    "PostgresGraphRunLedger",
    "GraphRunLedger",
    "GraphRunLedgerFactory",
    "GraphRunRecord",
    "GraphSessionDeletionRecord",
    "GraphThreadDeletionRecord",
    "RetentionResult",
    "SessionDeletionResult",
    "SqliteGraphRunLedger",
    "create_checkpointer",
    "create_graph_persistence",
    "setup_postgres_checkpointer",
    "setup_postgres_persistence",
]
