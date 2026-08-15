import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from graph_artifacts import GraphArtifactStore


class FakeUploads:
    def get_temp_content(self, file_id, user_id):
        if user_id == "u1" and file_id == "f1":
            return "附件正文"
        return ""

    def get_temp_file_info(self, file_id, user_id):
        return {"filename": "预算.xlsx", "content": "附件正文"}


class SlowMultiUploads:
    def get_temp_content(self, file_id, user_id):
        if user_id != "u1" or file_id not in {"f1", "f2"}:
            return ""
        # Without the per-run lock both workers read the empty manifest before
        # either atomic replace, making the lost-update race reproducible.
        time.sleep(0.05)
        return f"附件正文-{file_id}"

    def get_temp_file_info(self, file_id, user_id):
        content = self.get_temp_content(file_id, user_id)
        return {"filename": f"{file_id}.txt", "content": content}


def test_snapshot_hydrate_and_delete(tmp_path):
    store = GraphArtifactStore(tmp_path / "graph_runs", FakeUploads())

    refs = store.snapshot_attachments(run_id="run_1", user_id="u1", file_ids=["f1"])
    store.snapshot_document_runtime_context(
        run_id="run_1",
        user_id="u1",
        payload={"previous_context": "冻结上下文"},
    )

    assert refs[0]["filename"] == "预算.xlsx"
    assert refs[0]["is_spreadsheet"] is True
    assert "附件正文" in store.hydrate_message(
        run_id="run_1",
        user_id="u1",
        message="请分析",
        attachments=refs,
    )
    assert store.load_content(
        run_id="run_1",
        artifact_id=refs[0]["artifact_id"],
        user_id="u2",
    ) == ""
    assert store.delete_run("run_1") is True
    assert not (tmp_path / "graph_runs" / "run_1").exists()
    assert store.delete_run("run_1") is False


def test_snapshot_is_idempotent_for_same_run_and_file(tmp_path):
    store = GraphArtifactStore(tmp_path / "graph_runs", FakeUploads())

    first = store.snapshot_attachments(run_id="run_1", user_id="u1", file_ids=["f1"])
    second = store.snapshot_attachments(run_id="run_1", user_id="u1", file_ids=["f1"])

    assert second == first


def test_concurrent_attachment_snapshots_merge_manifest_entries(tmp_path):
    store = GraphArtifactStore(tmp_path / "graph_runs", SlowMultiUploads())

    def save(file_id):
        return store.snapshot_attachments(
            run_id="run_concurrent",
            user_id="u1",
            file_ids=[file_id],
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(save, ["f1", "f2"]))

    for item in (first, second):
        assert store.load_content(
            run_id="run_concurrent",
            artifact_id=item["artifact_id"],
            user_id="u1",
        ) == f"附件正文-{item['file_id']}"


def test_artifact_files_are_owner_only(tmp_path):
    root = tmp_path / "graph_runs"
    store = GraphArtifactStore(root, FakeUploads())
    refs = store.snapshot_attachments(
        run_id="run_private",
        user_id="u1",
        file_ids=["f1"],
    )
    store.snapshot_document_runtime_context(
        run_id="run_private",
        user_id="u1",
        payload={"previous_context": "private"},
    )

    paths = [
        root / "run_private" / "owner.json",
        root / "run_private" / "manifest.json",
        root / "run_private" / "document_runtime.json",
        root / "run_private" / f"{refs[0]['artifact_id']}.txt",
        root / "run_private" / ".artifacts.lock",
    ]
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in paths)


def test_rejects_cross_user_run_reuse(tmp_path):
    store = GraphArtifactStore(tmp_path / "graph_runs", FakeUploads())
    store.snapshot_attachments(run_id="run_1", user_id="u1", file_ids=["f1"])

    try:
        store.snapshot_attachments(run_id="run_1", user_id="u2", file_ids=["f1"])
    except PermissionError:
        pass
    else:
        raise AssertionError("cross-user graph artifact access must be rejected")


def test_document_runtime_context_round_trip_is_json_and_detached(tmp_path):
    store = GraphArtifactStore(tmp_path / "graph_runs", upload_manager=None)
    payload = {
        "memory_context": {"recent": ["第一轮"]},
        "profile": {"department": "研发"},
        "last_document": "旧文档",
        "last_plan": {"steps": [{"name": "检索"}]},
        "previous_context": "固定在本次 run 的上下文",
    }

    stored = store.snapshot_document_runtime_context(
        run_id="run_context_1",
        user_id="user@example.com",
        payload=payload,
    )
    payload["memory_context"]["recent"].append("后来写入")

    assert stored["memory_context"]["recent"] == ["第一轮"]
    assert store.load_document_runtime_context(
        run_id="run_context_1",
        user_id="user@example.com",
    ) == stored


def test_document_runtime_context_is_first_write_wins(tmp_path):
    store = GraphArtifactStore(tmp_path / "graph_runs", upload_manager=None)
    original = {"previous_context": "第一次", "last_plan": {"version": 1}}
    changed = {"previous_context": "第二次", "last_plan": {"version": 2}}

    first = store.snapshot_document_runtime_context(
        run_id="run_immutable",
        user_id="u1",
        payload=original,
    )
    retry = store.snapshot_document_runtime_context(
        run_id="run_immutable",
        user_id="u1",
        payload=changed,
    )

    assert first == original
    assert retry == original
    assert store.load_document_runtime_context(
        run_id="run_immutable",
        user_id="u1",
    ) == original


def test_document_runtime_context_concurrent_first_write_is_atomic(tmp_path):
    root = tmp_path / "graph_runs"
    store = GraphArtifactStore(root, upload_manager=None)
    candidates = [
        {"previous_context": "worker-a", "profile": {"worker": "a"}},
        {"previous_context": "worker-b", "profile": {"worker": "b"}},
    ]

    def save(payload):
        return store.snapshot_document_runtime_context(
            run_id="run_race",
            user_id="u1",
            payload=payload,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, candidates))

    saved = store.load_document_runtime_context(run_id="run_race", user_id="u1")
    assert saved in candidates
    assert results == [saved, saved]
    assert not list((root / "run_race").glob("*.tmp"))
    assert not list((root / "run_race").glob(".*.tmp"))


def test_document_runtime_context_enforces_shared_run_owner(tmp_path):
    store = GraphArtifactStore(tmp_path / "graph_runs", FakeUploads())
    store.snapshot_document_runtime_context(
        run_id="run_owned",
        user_id="u1",
        payload={"profile": {"name": "owner"}},
    )

    with pytest.raises(PermissionError):
        store.load_document_runtime_context(run_id="run_owned", user_id="u2")
    with pytest.raises(PermissionError):
        store.snapshot_document_runtime_context(
            run_id="run_owned",
            user_id="u2",
            payload={"profile": {"name": "intruder"}},
        )
    with pytest.raises(PermissionError):
        store.snapshot_attachments(run_id="run_owned", user_id="u2", file_ids=["f1"])


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "", "."])
def test_document_runtime_context_rejects_path_traversal(tmp_path, run_id):
    store = GraphArtifactStore(tmp_path / "graph_runs", upload_manager=None)

    with pytest.raises(ValueError):
        store.snapshot_document_runtime_context(
            run_id=run_id,
            user_id="u1",
            payload={"previous_context": "unsafe"},
        )
    with pytest.raises(ValueError):
        store.load_document_runtime_context(run_id=run_id, user_id="u1")


def test_document_runtime_context_rejects_symlink_run_directory(tmp_path):
    root = tmp_path / "graph_runs"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "linked_run").symlink_to(outside, target_is_directory=True)
    store = GraphArtifactStore(root, upload_manager=None)

    with pytest.raises(ValueError):
        store.snapshot_document_runtime_context(
            run_id="linked_run",
            user_id="u1",
            payload={"previous_context": "unsafe"},
        )
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"agent": object()},
        {1: "non-string-key"},
        {"score": float("nan")},
        {"items": ("tuple",)},
    ],
)
def test_document_runtime_context_rejects_non_json_payload(tmp_path, payload):
    store = GraphArtifactStore(tmp_path / "graph_runs", upload_manager=None)

    with pytest.raises(TypeError):
        store.snapshot_document_runtime_context(
            run_id="run_json",
            user_id="u1",
            payload=payload,
        )


def test_expired_scan_ignores_unsafe_names_files_symlinks_and_recent_runs(tmp_path):
    root = tmp_path / "graph_runs"
    store = GraphArtifactStore(root, upload_manager=None)
    old_timestamp = 1_700_000_000
    cutoff = datetime.fromtimestamp(old_timestamp + 10, tz=timezone.utc)

    old_run = root / "run-old"
    old_run.mkdir()
    old_manifest = old_run / "manifest.json"
    old_manifest.write_text("{}", encoding="utf-8")
    os.utime(old_manifest, (old_timestamp, old_timestamp))

    recent_run = root / "run-recent"
    recent_run.mkdir()
    recent_manifest = recent_run / "manifest.json"
    recent_manifest.write_text("{}", encoding="utf-8")
    os.utime(recent_manifest, (old_timestamp + 20, old_timestamp + 20))

    unsafe = root / "unsafe.name"
    unsafe.mkdir()
    os.utime(unsafe, (old_timestamp, old_timestamp))
    legal_file = root / "not_a_directory"
    legal_file.write_text("keep", encoding="utf-8")
    os.utime(legal_file, (old_timestamp, old_timestamp))
    symlink = root / "linked_run"
    symlink.symlink_to(old_run, target_is_directory=True)

    assert store.list_expired_run_ids(cutoff=cutoff) == ["run-old"]
    assert store.delete_run("not_a_directory") is False
    assert legal_file.read_text(encoding="utf-8") == "keep"
