#!/usr/bin/env python3
"""Monitor knowledge-base upload/delete/vector-index activity."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path


DEFAULT_KEYWORDS = (
    "knowledge",
    "upload",
    "delete",
    "reindex",
    "archive",
    "vector",
    "faiss",
    "spreadsheet",
    "manifest",
    "知识库",
    "上传",
    "删除",
    "归档",
    "重建",
    "向量",
    "表格",
    "失败",
    "error",
    "exception",
)

DEFAULT_IGNORED_LOG_PATTERNS = (
    r'"GET /api/admin/knowledge-health',
    r'"GET /api/admin/knowledge-audit',
    r'"GET /api/admin/knowledge-files',
    r'"GET /api/admin/spreadsheets',
    r'"GET /api/admin/vector-map',
    r'"GET /api/upload/categories',
)


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def compact_json(value: str, limit: int = 220) -> str:
    if not value:
        return ""
    try:
        parsed = json.loads(value)
        text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def connect_manifest(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        print(f"[{now()}] manifest missing: {path}", flush=True)
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def load_upload_state(conn: sqlite3.Connection) -> dict[str, tuple]:
    rows = conn.execute("""
        SELECT content_hash, filename, status, chunk_count, spreadsheet_row_count,
               updated_at, error_message, archived_path, vector_indexed_at, reindexed_at
        FROM knowledge_uploads
    """).fetchall()
    return {
        row["content_hash"]: (
            row["filename"],
            row["status"],
            row["chunk_count"],
            row["spreadsheet_row_count"],
            row["updated_at"],
            row["error_message"],
            row["archived_path"],
            row["vector_indexed_at"],
            row["reindexed_at"],
        )
        for row in rows
    }


def latest_audit_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM knowledge_admin_audit").fetchone()
    return int(row["max_id"] or 0)


def print_upload_event(content_hash: str, state: tuple) -> None:
    filename, status, chunks, rows, updated_at, error, archived, vector_at, reindexed = state
    short_hash = content_hash[:10]
    details = [f"file={filename}", f"hash={short_hash}", f"status={status}", f"chunks={chunks}", f"sheet_rows={rows}"]
    if vector_at:
        details.append(f"vector_at={vector_at}")
    if reindexed:
        details.append(f"reindexed_at={reindexed}")
    if archived:
        details.append(f"archived={archived}")
    if error:
        details.append(f"error={error}")
    print(f"[{now()}] MANIFEST {updated_at} " + " | ".join(details), flush=True)


def poll_manifest(conn: sqlite3.Connection, upload_state: dict[str, tuple], last_audit_id: int) -> tuple[dict[str, tuple], int]:
    current = load_upload_state(conn)
    for content_hash, state in sorted(current.items(), key=lambda item: item[1][4] or ""):
        if upload_state.get(content_hash) != state:
            print_upload_event(content_hash, state)

    rows = conn.execute("""
        SELECT id, content_hash, filename, action, actor_id, actor_name, status,
               message, backup_path, before_json, after_json, created_at
        FROM knowledge_admin_audit
        WHERE id > ?
        ORDER BY id ASC
    """, (last_audit_id,)).fetchall()
    for row in rows:
        last_audit_id = max(last_audit_id, int(row["id"]))
        actor = row["actor_name"] or row["actor_id"] or "-"
        before = compact_json(row["before_json"])
        after = compact_json(row["after_json"])
        details = [
            f"id={row['id']}",
            f"action={row['action']}",
            f"status={row['status']}",
            f"file={row['filename']}",
            f"hash={str(row['content_hash'])[:10]}",
            f"actor={actor}",
        ]
        if row["message"]:
            details.append(f"message={row['message']}")
        if row["backup_path"]:
            details.append(f"backup={row['backup_path']}")
        if before:
            details.append(f"before={before}")
        if after:
            details.append(f"after={after}")
        print(f"[{now()}] AUDIT {row['created_at']} " + " | ".join(details), flush=True)
    return current, last_audit_id


class Tailer:
    def __init__(
        self,
        path: Path,
        pattern: re.Pattern[str],
        ignored_pattern: re.Pattern[str] | None,
        from_start: bool,
    ) -> None:
        self.path = path
        self.pattern = pattern
        self.ignored_pattern = ignored_pattern
        self.offset = 0
        self.warned_missing = False
        if path.exists() and not from_start:
            self.offset = path.stat().st_size

    def poll(self) -> None:
        if not self.path.exists():
            if not self.warned_missing:
                print(f"[{now()}] log missing: {self.path}", flush=True)
                self.warned_missing = True
            return
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            for line in handle:
                text = line.rstrip()
                if self.ignored_pattern and self.ignored_pattern.search(text):
                    continue
                if self.pattern.search(text):
                    print(f"[{now()}] LOG {self.path.name}: {text}", flush=True)
            self.offset = handle.tell()


def health_snapshot(root: Path, manifest_path: Path) -> None:
    index_path = root / "knowledge_base" / "faiss_local.index"
    pkl_path = root / "knowledge_base" / "faiss_local_index.pkl"
    spreadsheet_path = root / "knowledge_base" / "spreadsheets.sqlite"
    paths = [manifest_path, index_path, pkl_path, spreadsheet_path]
    for path in paths:
        if path.exists():
            stamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
            print(f"[{now()}] FILE {path.name} size={path.stat().st_size} mtime={stamp}", flush=True)
        else:
            print(f"[{now()}] FILE {path.name} missing", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Agent project root")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--from-start", action="store_true", help="Read matching existing log lines before following")
    parser.add_argument("--no-logs", action="store_true", help="Only monitor SQLite manifest/audit tables")
    parser.add_argument("--snapshot", action="store_true", help="Print index file sizes before monitoring")
    parser.add_argument("--keyword", action="append", default=[], help="Extra log keyword to match")
    parser.add_argument("--show-read-requests", action="store_true", help="Do not suppress admin GET/read endpoints")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest_path = root / "knowledge_base" / "ingestion_manifest.sqlite"
    keywords = tuple(DEFAULT_KEYWORDS) + tuple(args.keyword)
    pattern = re.compile("|".join(re.escape(word) for word in keywords), flags=re.IGNORECASE)
    ignored_pattern = None
    if not args.show_read_requests:
        ignored_pattern = re.compile("|".join(DEFAULT_IGNORED_LOG_PATTERNS), flags=re.IGNORECASE)

    print(f"[{now()}] monitoring knowledge operations under {root}", flush=True)
    if args.snapshot:
        health_snapshot(root, manifest_path)

    conn = connect_manifest(manifest_path)
    upload_state: dict[str, tuple] = {}
    last_audit_id = 0
    if conn:
        upload_state = load_upload_state(conn)
        last_audit_id = latest_audit_id(conn)
        print(f"[{now()}] loaded baseline: uploads={len(upload_state)} audit_last_id={last_audit_id}", flush=True)

    tailers = []
    if not args.no_logs:
        for name in ("app.log", "agent.log", "知识库/agent.log"):
            tailers.append(Tailer(root / name, pattern, ignored_pattern, args.from_start))

    try:
        while True:
            if conn is None:
                conn = connect_manifest(manifest_path)
                if conn:
                    upload_state = load_upload_state(conn)
                    last_audit_id = latest_audit_id(conn)
            else:
                try:
                    upload_state, last_audit_id = poll_manifest(conn, upload_state, last_audit_id)
                except sqlite3.Error as exc:
                    print(f"[{now()}] manifest read error: {exc}", flush=True)
                    conn.close()
                    conn = None
            for tailer in tailers:
                tailer.poll()
            time.sleep(max(args.interval, 0.2))
    except KeyboardInterrupt:
        print(f"\n[{now()}] monitor stopped", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
