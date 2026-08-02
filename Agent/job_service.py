"""Lightweight persistent background job service."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"
TERMINAL_STATUSES = {JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED}


@dataclass
class JobService:
    db_path: Path | str
    max_workers: Optional[int] = None
    now_factory: Callable[[], datetime] = datetime.now

    def __post_init__(self):
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_workers = self.max_workers or int(os.getenv("JOB_WORKERS", "2") or "2")
        self.owner_pid = os.getpid()
        self.owner_id = f"{self.owner_pid}:{uuid.uuid4().hex[:12]}"
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(self.max_workers)))
        self._lock = threading.RLock()
        self._init_db()
        self.mark_interrupted_jobs_failed()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS background_jobs (
                    job_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT DEFAULT '',
                    payload_json TEXT DEFAULT '{}',
                    result_json TEXT DEFAULT '{}',
                    error TEXT DEFAULT '',
                    owner_id TEXT DEFAULT '',
                    owner_pid INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT DEFAULT '',
                    finished_at TEXT DEFAULT '',
                    updated_at TEXT NOT NULL
                )
            """)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(background_jobs)").fetchall()}
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE background_jobs ADD COLUMN owner_id TEXT DEFAULT ''")
            if "owner_pid" not in columns:
                conn.execute("ALTER TABLE background_jobs ADD COLUMN owner_pid INTEGER DEFAULT 0")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_background_jobs_user ON background_jobs(user_id, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_background_jobs_status ON background_jobs(status, updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_background_jobs_owner ON background_jobs(owner_pid, status)")
            conn.commit()

    def mark_interrupted_jobs_failed(self):
        now = self.now_factory().isoformat()
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT job_id, owner_pid
                FROM background_jobs
                WHERE status IN (?, ?)
            """, (JOB_STATUS_QUEUED, JOB_STATUS_RUNNING)).fetchall()
            interrupted_ids = [
                row["job_id"]
                for row in rows
                if not self._owner_process_alive(row["owner_pid"])
            ]
            for job_id in interrupted_ids:
                conn.execute("""
                    UPDATE background_jobs
                    SET status = ?, message = ?, error = ?, finished_at = ?, updated_at = ?
                    WHERE job_id = ?
                """, (
                    JOB_STATUS_FAILED,
                    "服务重启，任务未完成",
                    "服务重启，任务未完成",
                    now,
                    now,
                    job_id,
                ))
            conn.commit()

    def submit(
        self,
        job_type: str,
        user_id: str,
        payload: Optional[dict],
        runner: Callable[[], Any],
        *,
        message: str = "任务已提交",
    ) -> dict:
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        now = self.now_factory().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO background_jobs (
                    job_id, type, user_id, status, message,
                    payload_json, result_json, error, owner_id, owner_pid,
                    created_at, started_at, finished_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, '{}', '', ?, ?, ?, '', '', ?)
            """, (
                job_id,
                job_type,
                user_id or "",
                JOB_STATUS_QUEUED,
                message,
                self._to_json(payload or {}),
                self.owner_id,
                self.owner_pid,
                now,
                now,
            ))
            conn.commit()

        self._executor.submit(self._run_job, job_id, runner)
        return {
            "success": True,
            "job_id": job_id,
            "status": JOB_STATUS_QUEUED,
            "message": message,
        }

    def get_job(self, job_id: str, *, user_id: str, role: str = "user") -> tuple[dict, int]:
        row = self._get_job_row(job_id)
        if not row:
            return {"success": False, "message": "任务不存在"}, 404
        if role != "admin" and row["user_id"] != user_id:
            return {"success": False, "message": "无权限查看该任务"}, 403
        return {"success": True, "job": self._row_to_payload(row)}, 200

    def _run_job(self, job_id: str, runner: Callable[[], Any]):
        self._mark_running(job_id)
        try:
            raw_result = runner()
            result_payload, result_status = self._normalize_runner_result(raw_result)
            success = bool(result_payload.get("success")) and int(result_status or 200) < 400
            if success:
                self._mark_finished(
                    job_id,
                    status=JOB_STATUS_SUCCEEDED,
                    message=result_payload.get("message") or "任务已完成",
                    result=result_payload,
                    error="",
                )
            else:
                error = result_payload.get("message") or result_payload.get("error") or "任务失败"
                self._mark_finished(
                    job_id,
                    status=JOB_STATUS_FAILED,
                    message=str(error)[:500],
                    result=result_payload,
                    error=str(error)[:1000],
                )
        except Exception as exc:
            logger.exception("后台任务执行失败: %s", exc)
            self._mark_finished(
                job_id,
                status=JOB_STATUS_FAILED,
                message=f"任务执行失败: {str(exc)[:160]}",
                result={},
                error=str(exc)[:1000],
            )

    def _mark_running(self, job_id: str):
        now = self.now_factory().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE background_jobs
                SET status = ?, message = ?, started_at = ?, updated_at = ?
                WHERE job_id = ?
            """, (JOB_STATUS_RUNNING, "任务运行中", now, now, job_id))
            conn.commit()

    def _mark_finished(self, job_id: str, *, status: str, message: str, result: dict, error: str):
        now = self.now_factory().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE background_jobs
                SET status = ?, message = ?, result_json = ?, error = ?, finished_at = ?, updated_at = ?
                WHERE job_id = ?
            """, (
                status,
                message or "",
                self._to_json(result or {}),
                error or "",
                now,
                now,
                job_id,
            ))
            conn.commit()

    def _get_job_row(self, job_id: str):
        with self._connect() as conn:
            return conn.execute("SELECT * FROM background_jobs WHERE job_id = ?", (job_id,)).fetchone()

    @staticmethod
    def _normalize_runner_result(raw_result: Any) -> tuple[dict, int]:
        if isinstance(raw_result, tuple) and len(raw_result) == 2:
            payload, status = raw_result
            return (payload or {}, int(status or 200))
        if isinstance(raw_result, dict):
            return raw_result, int(raw_result.get("status", 200) or 200)
        return {"success": True, "message": "任务已完成", "result": raw_result}, 200

    @staticmethod
    def _to_json(value: dict) -> str:
        return json.dumps(value or {}, ensure_ascii=False)

    @staticmethod
    def _from_json(value: str) -> dict:
        try:
            parsed = json.loads(value or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _owner_process_alive(owner_pid: Any) -> bool:
        try:
            pid = int(owner_pid or 0)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def _row_to_payload(self, row) -> dict:
        return {
            "job_id": row["job_id"],
            "type": row["type"],
            "status": row["status"],
            "message": row["message"] or "",
            "result": self._from_json(row["result_json"]),
            "error": row["error"] or "",
            "created_at": row["created_at"] or "",
            "started_at": row["started_at"] or "",
            "finished_at": row["finished_at"] or "",
        }
