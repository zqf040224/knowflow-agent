#!/usr/bin/env python3
"""Start the web server with a persistence-safe worker count.

The repository's development defaults use SQLite, which is intentionally a
single-process LangGraph backend.  Production defaults to four Gunicorn
workers, but only after :class:`GraphPersistenceConfig` has verified the
PostgreSQL checkpointer, shared run ledger, and DSN configuration.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_PORT = 5003
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from graph_persistence import (
    GraphPersistenceConfig,
    GraphPersistenceConfigurationError,
)


def _positive_int(
    environ: Mapping[str, str], key: str, default: int
) -> int:
    raw = str(environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise GraphPersistenceConfigurationError(
            f"{key} must be an integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise GraphPersistenceConfigurationError(
            f"{key} must be at least 1, got {value}"
        )
    return value


def build_gunicorn_command(
    environ: Mapping[str, str] | None = None,
) -> tuple[list[str], int]:
    """Return the validated Gunicorn command and effective process count."""

    source = os.environ if environ is None else environ
    explicit_workers = str(source.get("WEB_CONCURRENCY") or "").strip()

    # Validate the persistence backend before importing ``app``.  This is what
    # makes a missing production DSN or local run ledger a container startup
    # failure instead of a request-time fallback.
    config = GraphPersistenceConfig.from_env(source)
    default_workers = 4 if config.is_production else 1
    workers = _positive_int(source, "WEB_CONCURRENCY", default_workers)
    if not explicit_workers and config.is_production:
        # ``GraphPersistenceConfig`` saw its conservative single-process
        # default.  Validate the effective production default as well.
        config = replace(config, worker_processes=workers)
        config.validate()

    threads = _positive_int(source, "GUNICORN_THREADS", 8)
    timeout = _positive_int(source, "GUNICORN_TIMEOUT", 120)
    host = str(source.get("HOST") or "0.0.0.0").strip() or "0.0.0.0"
    worker_class = (
        str(source.get("GUNICORN_WORKER_CLASS") or "gthread").strip()
        or "gthread"
    )
    command = [
        "gunicorn",
        "--workers",
        str(workers),
        "--worker-class",
        worker_class,
        "--threads",
        str(threads),
        "--timeout",
        str(timeout),
        "--bind",
        f"{host}:{CONTAINER_PORT}",
        "app:app",
    ]
    return command, workers


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        command, workers = build_gunicorn_command()
    except GraphPersistenceConfigurationError as exc:
        print(f"Server startup configuration error: {exc}", file=sys.stderr)
        return 2

    # Make the effective process boundary visible to application startup and
    # health/config diagnostics.  No secret values are logged.
    os.environ["WEB_CONCURRENCY"] = str(workers)
    print(
        "Starting Gunicorn with "
        f"{workers} worker(s); APP_ENV={os.getenv('APP_ENV', 'development')}; "
        "LangGraph backend="
        f"{os.getenv('LANGGRAPH_CHECKPOINTER_BACKEND', 'sqlite')}",
        flush=True,
    )
    os.execvp(command[0], command)
    return 0  # pragma: no cover - os.execvp replaces this process


if __name__ == "__main__":
    raise SystemExit(main())
