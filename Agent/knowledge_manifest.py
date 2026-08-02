"""
Knowledge ingestion manifest.

Tracks each knowledge upload across the original file, vector index, and
structured spreadsheet store. This gives admins a deterministic way to inspect
half-success states instead of relying only on logs.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from spreadsheet_store import SpreadsheetStore


class KnowledgeIngestionManifest:
    """SQLite manifest for knowledge-base ingestion status."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_uploads (
                    content_hash TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    category TEXT NOT NULL,
                    access_level TEXT NOT NULL,
                    department TEXT DEFAULT '',
                    uploaded_by TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    parser_type TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    spreadsheet_row_count INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT DEFAULT '',
                    structured_indexed_at TEXT DEFAULT '',
                    vector_indexed_at TEXT DEFAULT '',
                    archived_path TEXT DEFAULT '',
                    archived_at TEXT DEFAULT '',
                    reindexed_at TEXT DEFAULT '',
                    updated_at TEXT NOT NULL
                )
            """)
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(knowledge_uploads)").fetchall()
            }
            for column, ddl in {
                "archived_path": "ALTER TABLE knowledge_uploads ADD COLUMN archived_path TEXT DEFAULT ''",
                "archived_at": "ALTER TABLE knowledge_uploads ADD COLUMN archived_at TEXT DEFAULT ''",
                "reindexed_at": "ALTER TABLE knowledge_uploads ADD COLUMN reindexed_at TEXT DEFAULT ''",
            }.items():
                if column not in columns:
                    conn.execute(ddl)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_uploads_status
                ON knowledge_uploads(status, updated_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_uploads_scope
                ON knowledge_uploads(category, access_level, department)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_admin_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL,
                    filename TEXT DEFAULT '',
                    action TEXT NOT NULL,
                    actor_id TEXT DEFAULT '',
                    actor_name TEXT DEFAULT '',
                    status TEXT NOT NULL,
                    message TEXT DEFAULT '',
                    backup_path TEXT DEFAULT '',
                    before_json TEXT DEFAULT '{}',
                    after_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_admin_audit_recent
                ON knowledge_admin_audit(content_hash, created_at)
            """)
            conn.commit()

    def record_started(self, *, content_hash: str, filename: str, source_path: str,
                       category: str, access_level: str, department: str,
                       uploaded_by: str, uploaded_at: str, source_type: str,
                       parser_type: str, chunk_count: int) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_uploads
                (content_hash, filename, source_path, category, access_level,
                 department, uploaded_by, uploaded_at, source_type, parser_type,
                 chunk_count, spreadsheet_row_count, status, error_message,
                 structured_indexed_at, vector_indexed_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'started', '', '', '', ?)
            """, (
                content_hash, filename, source_path, category, access_level,
                department, uploaded_by, uploaded_at, source_type, parser_type,
                int(chunk_count), now,
            ))
            conn.commit()

    def mark_structured_indexed(self, content_hash: str, row_count: int) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE knowledge_uploads
                SET spreadsheet_row_count = ?, status = 'structured_indexed',
                    structured_indexed_at = ?, updated_at = ?
                WHERE content_hash = ?
            """, (int(row_count), now, now, content_hash))
            conn.commit()

    def mark_vector_indexed(self, content_hash: str) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE knowledge_uploads
                SET status = 'completed', error_message = '',
                    vector_indexed_at = ?, updated_at = ?
                WHERE content_hash = ?
            """, (now, now, content_hash))
            conn.commit()

    def mark_failed(self, content_hash: str, error_message: str) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE knowledge_uploads
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE content_hash = ?
            """, (str(error_message)[:500], now, content_hash))
            conn.commit()

    def update_metadata(self, content_hash: str, *, category: str,
                        access_level: str, department: str) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE knowledge_uploads
                SET category = ?, access_level = ?, department = ?, updated_at = ?
                WHERE content_hash = ?
            """, (category, access_level, department, now, content_hash))
            conn.commit()

    def restore_record(self, record: Optional[Dict]) -> None:
        if not record:
            return
        with self._connect() as conn:
            columns = [
                row["name"]
                for row in conn.execute("PRAGMA table_info(knowledge_uploads)").fetchall()
            ]
            placeholders = ", ".join("?" for _ in columns)
            assignments = ", ".join(f"{column} = excluded.{column}" for column in columns)
            conn.execute(f"""
                INSERT INTO knowledge_uploads ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(content_hash) DO UPDATE SET {assignments}
            """, tuple(record.get(column, "") for column in columns))
            conn.commit()

    def mark_archived(self, content_hash: str, archived_path: str) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE knowledge_uploads
                SET status = 'archived', archived_path = ?, archived_at = ?,
                    error_message = '', updated_at = ?
                WHERE content_hash = ?
            """, (archived_path, now, now, content_hash))
            conn.commit()

    def mark_reindexed(self, content_hash: str, *, source_path: str,
                       source_type: str, parser_type: str, chunk_count: int,
                       spreadsheet_row_count: int) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE knowledge_uploads
                SET source_path = ?, source_type = ?, parser_type = ?,
                    chunk_count = ?, spreadsheet_row_count = ?,
                    status = 'completed', error_message = '',
                    vector_indexed_at = ?, reindexed_at = ?, updated_at = ?
                WHERE content_hash = ?
            """, (
                source_path, source_type, parser_type, int(chunk_count),
                int(spreadsheet_row_count), now, now, now, content_hash,
            ))
            conn.commit()

    def get_record(self, content_hash: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_uploads WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return dict(row) if row else None

    def recent(self, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM knowledge_uploads
                ORDER BY updated_at DESC
                LIMIT ?
            """, (int(limit),)).fetchall()
        return [dict(row) for row in rows]

    def all_records(self, limit: int = 500) -> List[Dict]:
        return self.recent(limit=limit)

    def record_audit(self, *, content_hash: str, filename: str = "",
                     action: str, actor_id: str = "", actor_name: str = "",
                     status: str, message: str = "", backup_path: str = "",
                     before: Optional[Dict] = None, after: Optional[Dict] = None) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO knowledge_admin_audit
                (content_hash, filename, action, actor_id, actor_name, status,
                 message, backup_path, before_json, after_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content_hash,
                filename,
                action,
                actor_id,
                actor_name,
                status,
                str(message)[:800],
                backup_path,
                json.dumps(before or {}, ensure_ascii=False, default=str),
                json.dumps(after or {}, ensure_ascii=False, default=str),
                now,
            ))
            conn.commit()

    def recent_audit(self, limit: int = 80, content_hash: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM knowledge_admin_audit"
        params: List[Any] = []
        if content_hash:
            query += " WHERE content_hash = ?"
            params.append(content_hash)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("before_json", "after_json"):
                try:
                    item[key.replace("_json", "")] = json.loads(item.get(key) or "{}")
                except json.JSONDecodeError:
                    item[key.replace("_json", "")] = {}
                item.pop(key, None)
            result.append(item)
        return result

    def consistency_report(self, knowledge_base, spreadsheet_db_path: Union[str, Path]) -> Dict:
        records = self.recent(limit=500)
        metadatas = getattr(knowledge_base, "metadatas", []) or []
        vector_counts = {}
        for meta in metadatas:
            content_hash = meta.get("content_hash")
            if content_hash:
                vector_counts[content_hash] = vector_counts.get(content_hash, 0) + 1

        spreadsheet_store = None
        spreadsheet_path = Path(spreadsheet_db_path)
        if spreadsheet_path.exists():
            spreadsheet_store = SpreadsheetStore(spreadsheet_path)

        issues = []
        checked = []
        for record in records:
            row = dict(record)
            content_hash = row["content_hash"]
            vector_count = vector_counts.get(content_hash, 0)
            file_exists = Path(row["source_path"]).exists()
            archived_exists = bool(row.get("archived_path")) and Path(row["archived_path"]).exists()
            spreadsheet_rows = 0
            if row.get("source_type") == "spreadsheet" and spreadsheet_store:
                spreadsheet_rows = len(spreadsheet_store.get_rows_by_source(content_hash))

            row.update({
                "file_exists": file_exists,
                "archived_exists": archived_exists,
                "vector_chunk_count": vector_count,
                "spreadsheet_rows_found": spreadsheet_rows,
            })
            row_issues = self._record_issues(row)
            row["issues"] = row_issues
            if row_issues:
                issues.append({
                    "content_hash": content_hash,
                    "filename": row["filename"],
                    "issues": row_issues,
                })
            checked.append(row)

        return {
            "ok": not issues,
            "checked_count": len(checked),
            "issue_count": len(issues),
            "issues": issues,
            "recent": checked[:50],
        }

    @staticmethod
    def _record_issues(record: Dict[str, Any]) -> List[str]:
        issues = []
        status = record.get("status")
        if status == "completed":
            if not record.get("file_exists"):
                issues.append("原文件不存在")
            if record.get("vector_chunk_count", 0) < record.get("chunk_count", 0):
                issues.append("向量索引片段数少于入库记录")
            if record.get("source_type") == "spreadsheet":
                expected = record.get("spreadsheet_row_count", 0)
                found = record.get("spreadsheet_rows_found", 0)
                if expected and found < expected:
                    issues.append("结构化表格行数少于入库记录")
        elif status in {"started", "structured_indexed"}:
            issues.append(f"入库未完成: {status}")
        elif status == "failed":
            issues.append("入库失败")
        elif status == "archived":
            if record.get("vector_chunk_count", 0):
                issues.append("已归档文件仍存在向量片段")
            if record.get("source_type") == "spreadsheet" and record.get("spreadsheet_rows_found", 0):
                issues.append("已归档表格仍存在结构化行")
            if not record.get("archived_exists"):
                issues.append("归档原文件不存在")
        else:
            issues.append(f"未知入库状态: {status}")
        return issues
