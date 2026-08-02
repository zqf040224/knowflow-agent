"""Audit, backup, and cache helpers for knowledge-admin write flows."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeAdminAuditDependencies:
    knowledge_agent: Any
    knowledge_base: Any
    knowledge_manifest: Any
    spreadsheet_store: Any
    admin_backup_dir: Path
    actor_provider: Callable[[], dict]
    now_factory: Callable[[], Any]


class KnowledgeAdminAuditService:
    def __init__(self, deps: KnowledgeAdminAuditDependencies):
        self.deps = deps

    def clear_runtime_cache(self) -> None:
        self.deps.knowledge_agent.refresh()
        self.deps.knowledge_agent.cache_manager.clear_knowledge_cache()

    def backup_storage(self, action: str, content_hash: str) -> str:
        timestamp = self.deps.now_factory().strftime("%Y%m%d%H%M%S")
        safe_action = re.sub(r"[^a-zA-Z0-9_-]+", "_", action)[:32] or "admin"
        short_hash = (content_hash or "unknown")[:12]
        backup_dir = self.deps.admin_backup_dir / f"{timestamp}_{safe_action}_{short_hash}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            getattr(self.deps.knowledge_base, "index_path", None),
            getattr(self.deps.knowledge_base, "pkl_path", None),
            getattr(self.deps.knowledge_base, "config_path", None),
            getattr(self.deps.knowledge_manifest, "db_path", None),
            getattr(self.deps.spreadsheet_store, "db_path", None),
        ]
        copied = 0
        for path in candidates:
            if not path:
                continue
            source = Path(path)
            if source.exists():
                shutil.copy2(source, backup_dir / source.name)
                copied += 1
        if copied == 0:
            try:
                backup_dir.rmdir()
            except OSError:
                pass
            return ""
        return str(backup_dir)

    def record_audit(self, action: str, record: dict, *, status: str,
                     message: str = "", backup_path: str = "",
                     before: dict | None = None, after: dict | None = None,
                     actor: dict | None = None) -> None:
        actor_payload = actor or self.deps.actor_provider()
        self.deps.knowledge_manifest.record_audit(
            content_hash=(record or {}).get("content_hash", ""),
            filename=(record or {}).get("filename", ""),
            action=action,
            status=status,
            message=message,
            backup_path=backup_path,
            before=before,
            after=after,
            **actor_payload,
        )

    def record_upload_audit(self, result: dict, *, filename: str,
                            category: str, department: str = "",
                            actor: dict | None = None) -> None:
        try:
            actor_payload = actor or self.deps.actor_provider()
            content_hash = result.get("content_hash") or hashlib.sha1(
                f"{actor_payload.get('actor_id', '')}:{filename}:{self.deps.now_factory().isoformat()}".encode("utf-8")
            ).hexdigest()
            record = {
                "content_hash": content_hash,
                "filename": result.get("filename") or filename,
                "category": result.get("category") or category,
                "department": department,
                "uploaded_by": actor_payload.get("actor_id", ""),
            }
            audit_status = "success" if result.get("success") else "failed"
            if result.get("duplicate"):
                audit_status = "duplicate"
            self.record_audit(
                "upload",
                record,
                status=audit_status,
                message=result.get("message", ""),
                after=result if result.get("success") else None,
                actor=actor_payload,
            )
        except Exception as exc:
            logger.warning("知识库上传审计记录失败: %s", exc)
