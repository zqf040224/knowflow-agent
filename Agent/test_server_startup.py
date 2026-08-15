from __future__ import annotations

from pathlib import Path

import pytest

from graph_persistence import GraphPersistenceConfigurationError
from scripts.start_server import build_gunicorn_command


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_development_sqlite_launcher_is_single_process():
    command, workers = build_gunicorn_command({
        "APP_ENV": "development",
        "LANGGRAPH_CHECKPOINTER_BACKEND": "sqlite",
        # PUBLIC_PORT belongs to Docker's host-side mapping and must never
        # change the port listened to inside the container.  A stale legacy
        # PORT value must not change it either.
        "PUBLIC_PORT": "5010",
        "PORT": "5999",
    })

    assert workers == 1
    assert _option(command, "--workers") == "1"
    assert _option(command, "--bind") == "0.0.0.0:5003"
    assert command[-1] == "app:app"


def test_production_postgres_launcher_defaults_to_four_workers():
    command, workers = build_gunicorn_command({
        "APP_ENV": "production",
        "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
        "LANGGRAPH_POSTGRES_DSN": "postgresql://example.invalid/langgraph",
    })

    assert workers == 4
    assert _option(command, "--workers") == "4"
    assert _option(command, "--threads") == "8"


def test_launcher_rejects_explicit_multi_process_sqlite():
    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="Multi-process LangGraph requires",
    ):
        build_gunicorn_command({
            "APP_ENV": "development",
            "LANGGRAPH_CHECKPOINTER_BACKEND": "sqlite",
            "WEB_CONCURRENCY": "4",
        })


def test_launcher_honors_safe_explicit_postgres_worker_count():
    command, workers = build_gunicorn_command({
        "APP_ENV": "production",
        "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
        "LANGGRAPH_POSTGRES_DSN": "postgresql://example.invalid/langgraph",
        "WEB_CONCURRENCY": "2",
        "GUNICORN_THREADS": "3",
        "GUNICORN_TIMEOUT": "90",
    })

    assert workers == 2
    assert _option(command, "--workers") == "2"
    assert _option(command, "--threads") == "3"
    assert _option(command, "--timeout") == "90"


def test_launcher_rejects_postgres_worker_pool_budget_above_eight():
    with pytest.raises(
        GraphPersistenceConfigurationError,
        match="must not exceed 8 total PostgreSQL connections",
    ):
        build_gunicorn_command({
            "APP_ENV": "production",
            "LANGGRAPH_CHECKPOINTER_BACKEND": "postgres",
            "LANGGRAPH_POSTGRES_DSN": "postgresql://example.invalid/langgraph",
            "LANGGRAPH_POOL_MAX_SIZE": "2",
            "WEB_CONCURRENCY": "5",
        })


def test_production_compose_reserves_connection_for_cleanup_pool():
    compose = (Path(__file__).parent / "docker-compose.production.yml").read_text(
        encoding="utf-8"
    )
    app_block = compose.split("  langgraph-migrate:", 1)[0]
    cleanup_block = compose.split("  langgraph-cleanup:", 1)[1]

    assert 'WEB_CONCURRENCY: "${WEB_CONCURRENCY:-4}"' in app_block
    assert 'LANGGRAPH_POOL_MAX_SIZE: "1"' in app_block
    assert 'LANGGRAPH_POOL_MAX_SIZE: "1"' in cleanup_block
