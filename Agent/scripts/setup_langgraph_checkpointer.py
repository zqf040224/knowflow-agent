#!/usr/bin/env python3
"""Create LangGraph checkpoint tables and the shared graph-runs ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from graph_persistence import (
    GraphPersistenceConfig,
    GraphPersistenceError,
    setup_postgres_persistence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize the PostgreSQL tables used by LangGraph PostgresSaver. "
            "It also creates the shared graph_runs and deletion-tombstone tables. "
            "This command is intended for a deployment migration job, not each worker."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="dotenv file to load without overriding already-exported variables",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file, override=False)
    try:
        config = GraphPersistenceConfig.from_env()
        setup_postgres_persistence(config)
    except GraphPersistenceError as exc:
        print(f"LangGraph checkpointer setup failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Do not echo the DSN; database exceptions can be inspected from the
        # chained traceback in deployment logs when needed.
        print(
            "LangGraph checkpointer setup failed with an unexpected "
            f"{type(exc).__name__}",
            file=sys.stderr,
        )
        return 1

    print(
        "LangGraph PostgreSQL checkpointer and graph_runs schemas are ready "
        f"(pool {config.postgres_pool_min_size}-{config.postgres_pool_max_size})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
