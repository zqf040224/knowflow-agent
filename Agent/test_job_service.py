import sqlite3
import time
from datetime import datetime

from job_service import JOB_STATUS_FAILED, JOB_STATUS_SUCCEEDED, JobService


def wait_for_terminal(service, job_id, *, user_id="u1", role="user", timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload, status = service.get_job(job_id, user_id=user_id, role=role)
        assert status == 200
        if payload["job"]["status"] in {"succeeded", "failed"}:
            return payload["job"]
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_job_service_runs_successful_job(tmp_path):
    service = JobService(tmp_path / "jobs.sqlite", max_workers=1)

    submitted = service.submit(
        "demo",
        "u1",
        {"input": 1},
        lambda: {"success": True, "message": "done", "value": 2},
    )

    job = wait_for_terminal(service, submitted["job_id"])

    assert job["type"] == "demo"
    assert job["status"] == JOB_STATUS_SUCCEEDED
    assert job["message"] == "done"
    assert job["result"]["value"] == 2


def test_job_service_marks_exceptions_failed(tmp_path):
    service = JobService(tmp_path / "jobs.sqlite", max_workers=1)

    def explode():
        raise RuntimeError("boom")

    submitted = service.submit("demo", "u1", {}, explode)
    job = wait_for_terminal(service, submitted["job_id"])

    assert job["status"] == JOB_STATUS_FAILED
    assert "boom" in job["error"]


def test_job_service_enforces_read_permissions(tmp_path):
    service = JobService(tmp_path / "jobs.sqlite", max_workers=1)
    submitted = service.submit("demo", "owner", {}, lambda: {"success": True})
    wait_for_terminal(service, submitted["job_id"], user_id="owner")

    denied, denied_status = service.get_job(submitted["job_id"], user_id="other", role="user")
    admin, admin_status = service.get_job(submitted["job_id"], user_id="admin", role="admin")

    assert denied_status == 403
    assert denied["success"] is False
    assert admin_status == 200
    assert admin["job"]["job_id"] == submitted["job_id"]


def test_job_service_marks_interrupted_jobs_failed_on_startup(tmp_path):
    db_path = tmp_path / "jobs.sqlite"
    first = JobService(db_path, max_workers=1, now_factory=lambda: datetime(2026, 6, 7, 9, 0, 0))
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO background_jobs (
                job_id, type, user_id, status, message, payload_json,
                result_json, error, created_at, updated_at
            )
            VALUES ('job_old', 'demo', 'u1', 'running', '', '{}', '{}', '', 't0', 't0')
        """)
        conn.commit()

    first.mark_interrupted_jobs_failed()
    payload, status = first.get_job("job_old", user_id="u1")

    assert status == 200
    assert payload["job"]["status"] == JOB_STATUS_FAILED
    assert payload["job"]["message"] == "服务重启，任务未完成"


def test_job_service_does_not_fail_jobs_owned_by_live_process(tmp_path):
    db_path = tmp_path / "jobs.sqlite"
    first = JobService(db_path, max_workers=1)
    submitted = first.submit("slow", "u1", {}, lambda: (time.sleep(0.2) or {"success": True, "message": "done"}))

    second = JobService(db_path, max_workers=1)
    payload, status = second.get_job(submitted["job_id"], user_id="u1")

    assert status == 200
    assert payload["job"]["status"] != JOB_STATUS_FAILED

    job = wait_for_terminal(second, submitted["job_id"])
    assert job["status"] == JOB_STATUS_SUCCEEDED
