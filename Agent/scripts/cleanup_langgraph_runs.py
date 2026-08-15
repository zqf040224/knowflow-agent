#!/usr/bin/env python3
"""Retry graph deletions and remove runs past the retention window.

The checkpointer and run-ledger deletion is delegated to
``GraphPersistenceRuntime``.  Attachment snapshots are deleted only after the
corresponding thread deletion succeeds, so a failed checkpoint deletion keeps
everything needed for the next retry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from graph_artifacts import GraphArtifactStore
from graph_persistence import (
    GraphPersistenceConfig,
    GraphPersistenceError,
    create_graph_persistence,
)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retry pending LangGraph tombstones, then delete checkpoints, "
            "run-ledger rows, and attachment snapshots past retention."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="dotenv file to load without overriding exported variables",
    )
    parser.add_argument(
        "--retention-days",
        type=_positive_int,
        default=None,
        help="override LANGGRAPH_CHECKPOINT_RETENTION_DAYS for this sweep",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="maximum pending tombstones and expired threads per phase",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="run continuously, opening and closing persistence once per sweep",
    )
    parser.add_argument(
        "--interval-seconds",
        type=_positive_int,
        default=None,
        help=(
            "delay after a successful sweep (default: "
            "LANGGRAPH_CLEANUP_INTERVAL_SECONDS or 21600)"
        ),
    )
    parser.add_argument(
        "--retry-seconds",
        type=_positive_int,
        default=None,
        help=(
            "delay after a failed sweep (default: "
            "LANGGRAPH_CLEANUP_RETRY_SECONDS or 60)"
        ),
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=_positive_int,
        default=None,
        help=(
            "exit the loop after this many failed sweeps so the container "
            "runtime can restart it (default: 3)"
        ),
    )
    parser.add_argument(
        "--health-file",
        type=Path,
        default=Path("/tmp/langgraph-cleanup-health.json"),
        help="path used by loop mode and --healthcheck",
    )
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="validate the loop health file and exit without opening persistence",
    )
    parser.add_argument(
        "--health-max-age-seconds",
        type=_positive_int,
        default=None,
        help=(
            "maximum health-file age (default: "
            "LANGGRAPH_CLEANUP_HEALTH_MAX_AGE_SECONDS or 25200)"
        ),
    )
    return parser


def _resolve_storage_root() -> Path:
    # Import after load_dotenv(): storage_config resolves its paths at import
    # time and must see the same environment as the persistence runtime.
    from storage_config import STORAGE_ROOT

    return Path(STORAGE_ROOT)


def _result_payload(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "deleted_thread_ids": list(result.deleted_thread_ids),
        "failures": dict(result.failures),
    }
    if hasattr(result, "requested_thread_ids"):
        payload["requested_thread_ids"] = list(result.requested_thread_ids)
    if hasattr(result, "cutoff_at"):
        payload["cutoff_at"] = result.cutoff_at
    if hasattr(result, "candidate_count"):
        payload["candidate_count"] = int(result.candidate_count)
    return payload


def _delete_artifacts(
    artifact_store: GraphArtifactStore,
    thread_ids: list[str],
    *,
    expired_run_ids: list[str] | None = None,
) -> dict[str, Any]:
    requested: list[str] = []
    deleted: list[str] = []
    not_found: list[str] = []
    failures: dict[str, str] = {}
    seen: set[str] = set()

    for thread_id in [*thread_ids, *(expired_run_ids or [])]:
        if thread_id in seen:
            continue
        seen.add(thread_id)
        requested.append(thread_id)
        try:
            removed = artifact_store.delete_run(thread_id)
        except Exception as exc:
            failures[thread_id] = f"{type(exc).__name__}: {exc}"[:500]
        else:
            (deleted if removed else not_found).append(thread_id)

    return {
        "requested_run_ids": requested,
        "expired_scan_run_ids": list(expired_run_ids or []),
        "deleted_run_ids": deleted,
        "not_found_run_ids": not_found,
        "failures": failures,
    }


def run_cleanup(
    runtime: Any,
    artifact_store: GraphArtifactStore,
    *,
    retention_days: int,
    limit: int,
) -> dict[str, Any]:
    """Run both deletion phases in the required order and summarize them."""

    retried = runtime.retry_pending_deletions(
        limit=limit,
        delete_artifact_run=artifact_store.delete_run,
    )
    retained = runtime.delete_expired_threads(
        retention_days=retention_days,
        limit=limit,
    )
    cutoff = _parse_cutoff(retained.cutoff_at)
    expired_artifacts = artifact_store.list_expired_run_ids(cutoff=cutoff)
    # Age alone cannot authorize artifact deletion: a long-lived run can own
    # an old snapshot. Only delete aged directories that are now ledger
    # orphans, plus threads whose checkpoint/ledger deletion just succeeded.
    expired_orphans = [
        run_id
        for run_id in expired_artifacts
        if runtime.run_ledger.get_run(run_id) is None
    ]
    deleted_thread_ids = [
        *retried.deleted_thread_ids,
        *retained.deleted_thread_ids,
    ]
    artifacts = _delete_artifacts(
        artifact_store,
        deleted_thread_ids,
        expired_run_ids=expired_orphans,
    )
    retry_payload = _result_payload(retried)
    retention_payload = _result_payload(retained)
    ok = not (
        retry_payload["failures"]
        or retention_payload["failures"]
        or artifacts["failures"]
    )
    return {
        "ok": ok,
        "retention_days": retention_days,
        "limit": limit,
        "retry_pending_deletions": retry_payload,
        "retention": retention_payload,
        "artifacts": artifacts,
    }


def _parse_cutoff(value: str) -> datetime:
    normalized = str(value or "").strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    cutoff = datetime.fromisoformat(normalized)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    return cutoff.astimezone(timezone.utc)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _positive_env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _run_once(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Execute one isolated sweep and always release its database pool."""

    runtime = None
    try:
        config = GraphPersistenceConfig.from_env()
        retention_days = args.retention_days or config.retention_days
        limit = args.limit or config.retention_batch_size
        runtime = create_graph_persistence(config)
        artifact_store = GraphArtifactStore(
            _resolve_storage_root() / "graph_runs",
            upload_manager=None,
        )
        summary = run_cleanup(
            runtime,
            artifact_store,
            retention_days=retention_days,
            limit=limit,
        )
        return (0 if summary["ok"] else 1), summary
    except GraphPersistenceError as exc:
        return 2, {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    except Exception as exc:
        return 1, {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    finally:
        if runtime is not None:
            runtime.close()


def _write_health(
    path: Path,
    *,
    exit_code: int,
    consecutive_failures: int,
    last_success_at_epoch: float | None,
) -> dict[str, Any]:
    """Publish a small atomic heartbeat for the container health probe."""

    now_epoch = time.time()
    payload = {
        "ok": exit_code == 0,
        "updated_at": datetime.fromtimestamp(
            now_epoch, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "updated_at_epoch": now_epoch,
        "last_success_at_epoch": last_success_at_epoch,
        "last_exit_code": int(exit_code),
        "consecutive_failures": int(consecutive_failures),
        "pid": os.getpid(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return payload


def _healthcheck(path: Path, *, max_age_seconds: int) -> tuple[int, dict[str, Any]]:
    """Return unhealthy for missing, stale, malformed, or failed heartbeats."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("health payload must be an object")
        updated_at_epoch = float(payload["updated_at_epoch"])
        age_seconds = max(0.0, time.time() - updated_at_epoch)
        healthy = bool(payload.get("ok")) and age_seconds <= max_age_seconds
        result = {
            **payload,
            "healthy": healthy,
            "age_seconds": round(age_seconds, 3),
            "max_age_seconds": int(max_age_seconds),
        }
        return (0 if healthy else 1), result
    except Exception as exc:
        return 1, {
            "healthy": False,
            "health_file": str(path),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def _run_loop(args: argparse.Namespace) -> int:
    interval_seconds = args.interval_seconds or _positive_env_int(
        "LANGGRAPH_CLEANUP_INTERVAL_SECONDS", 21600
    )
    retry_seconds = args.retry_seconds or _positive_env_int(
        "LANGGRAPH_CLEANUP_RETRY_SECONDS", 60
    )
    max_failures = args.max_consecutive_failures or _positive_env_int(
        "LANGGRAPH_CLEANUP_MAX_CONSECUTIVE_FAILURES", 3
    )
    consecutive_failures = 0
    last_success_at_epoch: float | None = None

    while True:
        exit_code, summary = _run_once(args)
        _print_json(summary)
        if exit_code == 0:
            consecutive_failures = 0
            last_success_at_epoch = time.time()
        else:
            consecutive_failures += 1
        try:
            _write_health(
                args.health_file,
                exit_code=exit_code,
                consecutive_failures=consecutive_failures,
                last_success_at_epoch=last_success_at_epoch,
            )
        except Exception as exc:
            _print_json(
                {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": f"could not publish cleanup health: {exc}",
                    },
                }
            )
            return 1

        if exit_code != 0 and consecutive_failures >= max_failures:
            return exit_code
        delay = interval_seconds if exit_code == 0 else retry_seconds
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file, override=False)
    if args.healthcheck:
        max_age_seconds = args.health_max_age_seconds or _positive_env_int(
            "LANGGRAPH_CLEANUP_HEALTH_MAX_AGE_SECONDS", 25200
        )
        exit_code, payload = _healthcheck(
            args.health_file,
            max_age_seconds=max_age_seconds,
        )
        _print_json(payload)
        return exit_code

    if args.loop:
        return _run_loop(args)
    exit_code, payload = _run_once(args)
    _print_json(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
