import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import cleanup_langgraph_runs as cleanup_script


class FakeRuntime:
    def __init__(
        self,
        *,
        retry,
        retention,
        fail_on_retry=False,
        active_run_ids=(),
    ):
        self.retry = retry
        self.retention = retention
        self.fail_on_retry = fail_on_retry
        self.calls = []
        self.closed = False
        active = set(active_run_ids)
        self.run_ledger = SimpleNamespace(
            get_run=lambda run_id: (
                SimpleNamespace(run_id=run_id) if run_id in active else None
            )
        )

    def retry_pending_deletions(self, *, limit, delete_artifact_run=None):
        self.calls.append(("retry", limit))
        self.delete_artifact_run = delete_artifact_run
        if self.fail_on_retry:
            raise RuntimeError("retry exploded")
        return self.retry

    def delete_expired_threads(self, *, retention_days, limit):
        self.calls.append(("retention", retention_days, limit))
        return self.retention

    def close(self):
        self.closed = True


class FakeArtifactStore:
    instances = []
    outcomes = {}
    expired_ids = []

    def __init__(self, root, upload_manager):
        self.root = Path(root)
        self.upload_manager = upload_manager
        self.deleted = []
        self.__class__.instances.append(self)

    def delete_run(self, run_id):
        self.deleted.append(run_id)
        outcome = self.__class__.outcomes.get(run_id, False)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def list_expired_run_ids(self, *, cutoff):
        self.cutoff = cutoff
        return list(self.__class__.expired_ids)


def deletion_result(*, requested=(), deleted=(), failures=None):
    return SimpleNamespace(
        requested_thread_ids=tuple(requested),
        deleted_thread_ids=tuple(deleted),
        failures=dict(failures or {}),
    )


def retention_result(*, deleted=(), failures=None):
    return SimpleNamespace(
        cutoff_at="2026-05-11T00:00:00+00:00",
        candidate_count=len(deleted) + len(failures or {}),
        deleted_thread_ids=tuple(deleted),
        failures=dict(failures or {}),
    )


def configure_main(monkeypatch, tmp_path, runtime, *, retention_days=90, limit=100):
    config = SimpleNamespace(
        retention_days=retention_days,
        retention_batch_size=limit,
    )
    monkeypatch.setattr(
        cleanup_script.GraphPersistenceConfig,
        "from_env",
        lambda: config,
    )
    monkeypatch.setattr(
        cleanup_script,
        "create_graph_persistence",
        lambda resolved: runtime,
    )
    monkeypatch.setattr(cleanup_script, "GraphArtifactStore", FakeArtifactStore)
    monkeypatch.setattr(cleanup_script, "_resolve_storage_root", lambda: tmp_path)
    monkeypatch.setattr(cleanup_script, "load_dotenv", lambda *args, **kwargs: None)
    FakeArtifactStore.instances.clear()
    FakeArtifactStore.outcomes = {}
    FakeArtifactStore.expired_ids = []
    return config


def test_main_retries_before_retention_and_deletes_artifacts(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime(
        retry=deletion_result(
            requested=("run-retry",),
            deleted=("run-retry",),
        ),
        retention=retention_result(deleted=("run-old", "run-retry")),
    )
    config = configure_main(monkeypatch, tmp_path, runtime)
    created_with = []
    monkeypatch.setattr(
        cleanup_script,
        "create_graph_persistence",
        lambda resolved: created_with.append(resolved) or runtime,
    )
    FakeArtifactStore.outcomes = {"run-retry": True, "run-old": False}

    exit_code = cleanup_script.main(
        ["--env-file", str(tmp_path / "missing.env"), "--retention-days", "30", "--limit", "7"]
    )

    payload = cleanup_script.json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert created_with == [config]
    assert runtime.calls == [("retry", 7), ("retention", 30, 7)]
    assert runtime.closed is True
    store = FakeArtifactStore.instances[0]
    assert getattr(runtime.delete_artifact_run, "__self__", None) is store
    assert store.root == tmp_path / "graph_runs"
    assert store.upload_manager is None
    assert store.deleted == ["run-retry", "run-old"]
    assert payload["artifacts"] == {
        "requested_run_ids": ["run-retry", "run-old"],
        "expired_scan_run_ids": [],
        "deleted_run_ids": ["run-retry"],
        "not_found_run_ids": ["run-old"],
        "failures": {},
    }


def test_main_uses_config_defaults_and_reports_partial_failures(
    monkeypatch, tmp_path, capsys
):
    runtime = FakeRuntime(
        retry=deletion_result(
            requested=("run-pending",),
            failures={"run-pending": "checkpoint busy"},
        ),
        retention=retention_result(deleted=("run-old",)),
    )
    configure_main(monkeypatch, tmp_path, runtime, retention_days=45, limit=11)
    FakeArtifactStore.outcomes = {"run-old": OSError("artifact busy")}

    exit_code = cleanup_script.main([])

    payload = cleanup_script.json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert runtime.calls == [("retry", 11), ("retention", 45, 11)]
    assert runtime.closed is True
    assert payload["ok"] is False
    assert payload["retry_pending_deletions"]["failures"] == {
        "run-pending": "checkpoint busy"
    }
    assert payload["artifacts"]["failures"] == {
        "run-old": "OSError: artifact busy"
    }


def test_main_closes_runtime_when_cleanup_raises(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime(
        retry=deletion_result(),
        retention=retention_result(),
        fail_on_retry=True,
    )
    configure_main(monkeypatch, tmp_path, runtime)

    exit_code = cleanup_script.main([])

    payload = cleanup_script.json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert runtime.closed is True
    assert payload["error"] == {
        "type": "RuntimeError",
        "message": "retry exploded",
    }


def test_failed_artifact_delete_is_retried_by_next_expired_scan(tmp_path):
    root = tmp_path / "graph_runs"
    run_dir = root / "run-old"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    old_timestamp = 1_700_000_000
    os.utime(manifest, (old_timestamp, old_timestamp))
    store = cleanup_script.GraphArtifactStore(root, upload_manager=None)

    class FailOnceStore:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.failed = False

        def list_expired_run_ids(self, *, cutoff):
            return self.wrapped.list_expired_run_ids(cutoff=cutoff)

        def delete_run(self, run_id):
            if not self.failed:
                self.failed = True
                raise OSError("temporary artifact lock")
            return self.wrapped.delete_run(run_id)

    flaky_store = FailOnceStore(store)
    first_runtime = FakeRuntime(
        retry=deletion_result(),
        retention=retention_result(deleted=("run-old",)),
    )
    first = cleanup_script.run_cleanup(
        first_runtime,
        flaky_store,
        retention_days=90,
        limit=10,
    )

    assert first["ok"] is False
    assert run_dir.exists()
    assert first["artifacts"]["failures"] == {
        "run-old": "OSError: temporary artifact lock"
    }

    second_runtime = FakeRuntime(
        retry=deletion_result(),
        retention=retention_result(),
    )
    second = cleanup_script.run_cleanup(
        second_runtime,
        flaky_store,
        retention_days=90,
        limit=10,
    )

    assert second["ok"] is True
    assert second["artifacts"]["expired_scan_run_ids"] == ["run-old"]
    assert second["artifacts"]["deleted_run_ids"] == ["run-old"]
    assert not run_dir.exists()


def test_expired_scan_does_not_delete_artifact_owned_by_live_ledger_run(tmp_path):
    root = tmp_path / "graph_runs"
    run_dir = root / "run-live"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    os.utime(manifest, (1_700_000_000, 1_700_000_000))
    runtime = FakeRuntime(
        retry=deletion_result(),
        retention=retention_result(),
        active_run_ids=("run-live",),
    )

    result = cleanup_script.run_cleanup(
        runtime,
        cleanup_script.GraphArtifactStore(root, upload_manager=None),
        retention_days=90,
        limit=10,
    )

    assert result["artifacts"]["expired_scan_run_ids"] == []
    assert run_dir.exists()


def test_parser_rejects_non_positive_limits():
    parser = cleanup_script.build_parser()

    try:
        parser.parse_args(["--retention-days", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("zero retention must be rejected")

    try:
        parser.parse_args(["--limit", "-1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("negative limit must be rejected")

    for option in (
        "--interval-seconds",
        "--retry-seconds",
        "--max-consecutive-failures",
        "--health-max-age-seconds",
    ):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([option, "0"])
        assert exc_info.value.code == 2


def test_loop_retries_failures_then_exits_for_container_restart(
    monkeypatch, tmp_path, capsys
):
    health_file = tmp_path / "cleanup-health.json"
    outcomes = iter(
        (
            (1, {"ok": False, "error": {"message": "first"}}),
            (2, {"ok": False, "error": {"message": "second"}}),
        )
    )
    calls = []
    sleeps = []
    monkeypatch.setattr(cleanup_script, "load_dotenv", lambda *args, **kwargs: None)

    def run_once(args):
        calls.append(args)
        return next(outcomes)

    monkeypatch.setattr(cleanup_script, "_run_once", run_once)
    monkeypatch.setattr(cleanup_script.time, "sleep", sleeps.append)

    exit_code = cleanup_script.main(
        [
            "--loop",
            "--interval-seconds",
            "30",
            "--retry-seconds",
            "4",
            "--max-consecutive-failures",
            "2",
            "--health-file",
            str(health_file),
        ]
    )

    output = [
        cleanup_script.json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    health = cleanup_script.json.loads(health_file.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert len(calls) == 2
    assert sleeps == [4]
    assert [item["error"]["message"] for item in output] == [
        "first",
        "second",
    ]
    assert health["ok"] is False
    assert health["last_exit_code"] == 2
    assert health["consecutive_failures"] == 2
    assert health["last_success_at_epoch"] is None


def test_healthcheck_accepts_fresh_success_and_rejects_stale_or_failed(
    monkeypatch, tmp_path, capsys
):
    health_file = tmp_path / "cleanup-health.json"
    monkeypatch.setattr(cleanup_script.time, "time", lambda: 100.0)
    cleanup_script._write_health(
        health_file,
        exit_code=0,
        consecutive_failures=0,
        last_success_at_epoch=100.0,
    )

    monkeypatch.setattr(cleanup_script.time, "time", lambda: 105.0)
    assert cleanup_script.main(
        [
            "--healthcheck",
            "--health-file",
            str(health_file),
            "--health-max-age-seconds",
            "10",
        ]
    ) == 0
    fresh = cleanup_script.json.loads(capsys.readouterr().out)
    assert fresh["healthy"] is True
    assert fresh["age_seconds"] == 5.0

    monkeypatch.setattr(cleanup_script.time, "time", lambda: 111.0)
    assert cleanup_script.main(
        [
            "--healthcheck",
            "--health-file",
            str(health_file),
            "--health-max-age-seconds",
            "10",
        ]
    ) == 1
    stale = cleanup_script.json.loads(capsys.readouterr().out)
    assert stale["healthy"] is False
    assert stale["age_seconds"] == 11.0

    cleanup_script._write_health(
        health_file,
        exit_code=1,
        consecutive_failures=1,
        last_success_at_epoch=100.0,
    )
    assert cleanup_script.main(
        [
            "--healthcheck",
            "--health-file",
            str(health_file),
            "--health-max-age-seconds",
            "10",
        ]
    ) == 1
    failed = cleanup_script.json.loads(capsys.readouterr().out)
    assert failed["healthy"] is False
    assert failed["last_exit_code"] == 1
