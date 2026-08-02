"""Write-side admin services for knowledge-file management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeAdminWriteDependencies:
    knowledge_base: Any
    spreadsheet_store: Any
    knowledge_manifest: Any
    upload_manager: Any
    admin_lock: Any
    access_levels: set[str]
    department_dirs: set[str]
    parse_spreadsheet: Callable[..., list]
    now_factory: Callable[[], Any]
    audit_service: Any


class KnowledgeAdminWriteService:
    def __init__(self, deps: KnowledgeAdminWriteDependencies):
        self.deps = deps

    def delete_knowledge_file(self, content_hash: str, payload: dict) -> tuple[dict, int]:
        request_data = payload or {}
        delete_reason = (request_data.get("reason") or "").strip()
        if request_data.get("confirm") is not True:
            return {"success": False, "message": "请确认删除操作"}, 400
        if len(delete_reason) < 2:
            return {"success": False, "message": "请填写删除原因"}, 400

        record = self.deps.knowledge_manifest.get_record(content_hash)
        if not record:
            return {"success": False, "message": "未找到该知识库文件"}, 404

        with self.deps.admin_lock:
            kb_snapshot = self.deps.knowledge_base.snapshot_state()
            sheet_snapshot = self.deps.spreadsheet_store.snapshot_file(content_hash)
            manifest_snapshot = dict(record)
            backup_path = self.deps.audit_service.backup_storage("delete", content_hash)
            archived_path = record.get("archived_path") or ""
            source_path = record.get("source_path") or ""
            try:
                if source_path and Path(source_path).exists():
                    archived_path = self.deps.upload_manager.archive_knowledge_file(source_path)
                    if not archived_path:
                        raise RuntimeError("原文件归档失败")

                vector_removed = self.deps.knowledge_base.delete_by_content_hash(content_hash)
                self.deps.spreadsheet_store.delete_file(content_hash)
                self.deps.knowledge_manifest.mark_archived(content_hash, archived_path)
                updated_record = self.deps.knowledge_manifest.get_record(content_hash) or {}
                self.deps.audit_service.record_audit(
                    "delete",
                    record,
                    status="success",
                    message=f"软删除并归档完成；原因：{delete_reason}",
                    backup_path=backup_path,
                    before=manifest_snapshot,
                    after=updated_record,
                )
                self.deps.audit_service.clear_runtime_cache()
                return {
                    "success": True,
                    "message": "已软删除并归档",
                    "vector_removed": vector_removed,
                    "archived_path": archived_path,
                    "backup_path": backup_path,
                }, 200
            except Exception as exc:
                rollback_errors = self._restore_snapshots(
                    content_hash,
                    kb_snapshot=kb_snapshot,
                    sheet_snapshot=sheet_snapshot,
                    manifest_snapshot=manifest_snapshot,
                )
                try:
                    self._restore_archived_file(archived_path, source_path)
                except Exception as rollback_exc:
                    rollback_errors.append(f"原文件回滚失败: {rollback_exc}")
                self.deps.audit_service.record_audit(
                    "delete",
                    record,
                    status="failed",
                    message=f"原因：{delete_reason}; {exc}; rollback_errors={rollback_errors}",
                    backup_path=backup_path,
                    before=manifest_snapshot,
                    after=self.deps.knowledge_manifest.get_record(content_hash) or {},
                )
                return {
                    "success": False,
                    "message": f"删除失败，已尝试回滚: {str(exc)[:160]}",
                    "rolled_back": not rollback_errors,
                    "rollback_errors": rollback_errors,
                }, 500

    def update_knowledge_file_metadata(self, content_hash: str, payload: dict) -> tuple[dict, int]:
        record = self.deps.knowledge_manifest.get_record(content_hash)
        if not record:
            return {"success": False, "message": "未找到该知识库文件"}, 404

        request_data = payload or {}
        category = (request_data.get("category") or record.get("category") or "").strip()
        access_level = (request_data.get("access_level") or record.get("access_level") or "public").strip()
        department = (request_data.get("department") or "").strip()
        if not category:
            return {"success": False, "message": "分类不能为空"}, 400
        if access_level not in self.deps.access_levels:
            return {"success": False, "message": "访问级别无效"}, 400
        if access_level == "restricted" and department not in self.deps.department_dirs:
            return {"success": False, "message": "restricted 权限必须选择有效部门"}, 400
        if access_level != "restricted":
            department = ""

        with self.deps.admin_lock:
            kb_snapshot = self.deps.knowledge_base.snapshot_state()
            sheet_snapshot = self.deps.spreadsheet_store.snapshot_file(content_hash)
            manifest_snapshot = dict(record)
            backup_path = self.deps.audit_service.backup_storage("metadata", content_hash)
            requested_after = {
                "category": category,
                "access_level": access_level,
                "department": department,
            }
            try:
                vector_updated = self.deps.knowledge_base.update_metadata_by_content_hash(content_hash, requested_after)
                sheet_updated = self.deps.spreadsheet_store.update_file_metadata(
                    content_hash,
                    category=category,
                    access_level=access_level,
                    department=department,
                )
                self.deps.knowledge_manifest.update_metadata(
                    content_hash,
                    category=category,
                    access_level=access_level,
                    department=department,
                )
                updated_record = self.deps.knowledge_manifest.get_record(content_hash) or {}
                self.deps.audit_service.record_audit(
                    "metadata",
                    record,
                    status="success",
                    message="分类/部门/权限已同步",
                    backup_path=backup_path,
                    before=manifest_snapshot,
                    after=updated_record,
                )
                self.deps.audit_service.clear_runtime_cache()
                return {
                    "success": True,
                    "message": "元数据已更新",
                    "vector_updated": vector_updated,
                    "spreadsheet_updated": sheet_updated,
                    "backup_path": backup_path,
                }, 200
            except Exception as exc:
                rollback_errors = self._restore_snapshots(
                    content_hash,
                    kb_snapshot=kb_snapshot,
                    sheet_snapshot=sheet_snapshot,
                    manifest_snapshot=manifest_snapshot,
                )
                self.deps.audit_service.record_audit(
                    "metadata",
                    record,
                    status="failed",
                    message=f"{exc}; requested={requested_after}; rollback_errors={rollback_errors}",
                    backup_path=backup_path,
                    before=manifest_snapshot,
                    after=self.deps.knowledge_manifest.get_record(content_hash) or {},
                )
                return {
                    "success": False,
                    "message": f"更新失败，已尝试回滚: {str(exc)[:160]}",
                    "rolled_back": not rollback_errors,
                    "rollback_errors": rollback_errors,
                }, 500

    def reindex_knowledge_file(self, content_hash: str, *, user_id: str) -> tuple[dict, int]:
        record = self.deps.knowledge_manifest.get_record(content_hash)
        if not record:
            return {"success": False, "message": "未找到该知识库文件"}, 404

        source_raw = record.get("source_path") or ""
        archived_raw = record.get("archived_path") or ""
        source_path = Path(source_raw) if source_raw else None
        archived_path = Path(archived_raw) if archived_raw else None
        active_path = (
            source_path if source_path and source_path.exists()
            else archived_path if archived_path and archived_path.exists()
            else None
        )
        if active_path is None:
            return {"success": False, "message": "原文件和归档文件都不存在，无法重建"}, 400

        try:
            actual_hash = self.deps.upload_manager._file_sha256(str(active_path))
            if actual_hash != content_hash:
                return {
                    "success": False,
                    "message": "文件内容与入库哈希不一致，已阻止重建以避免数据错配",
                }, 409

            prepared = self.deps.upload_manager.build_knowledge_documents(
                file_path=str(active_path),
                filename=record.get("filename") or active_path.name,
                category=record.get("category") or "公共资料",
                user_id=record.get("uploaded_by") or user_id,
                department=record.get("department") or "",
                access_level=record.get("access_level") or "public",
                content_hash=content_hash,
                uploaded_at=record.get("uploaded_at") or self.deps.now_factory().isoformat(),
            )
            if not prepared.get("success"):
                return {"success": False, "message": prepared.get("message", "重建解析失败")}, 400

            spreadsheet_rows = []
            if prepared.get("is_spreadsheet"):
                spreadsheet_rows = self.deps.parse_spreadsheet(
                    active_path,
                    display_filename=record.get("filename") or active_path.name,
                )
        except Exception as exc:
            logger.exception("知识库文件重建预检查失败: %s", exc)
            return {"success": False, "message": f"重建预检查失败: {str(exc)[:160]}"}, 500

        with self.deps.admin_lock:
            kb_snapshot = self.deps.knowledge_base.snapshot_state()
            sheet_snapshot = self.deps.spreadsheet_store.snapshot_file(content_hash)
            manifest_snapshot = dict(record)
            backup_path = self.deps.audit_service.backup_storage("reindex", content_hash)
            try:
                removed = self.deps.knowledge_base.replace_by_content_hash(content_hash, prepared["documents"])
                if prepared.get("is_spreadsheet"):
                    row_count = self.deps.spreadsheet_store.upsert_file_rows(
                        content_hash=content_hash,
                        filename=record.get("filename") or active_path.name,
                        source_path=str(active_path),
                        category=record.get("category") or "公共资料",
                        access_level=prepared.get("access_level") or record.get("access_level") or "public",
                        department=prepared.get("department") or "",
                        uploaded_by=record.get("uploaded_by") or user_id,
                        uploaded_at=record.get("uploaded_at") or self.deps.now_factory().isoformat(),
                        rows=spreadsheet_rows,
                    )
                else:
                    self.deps.spreadsheet_store.delete_file(content_hash)
                    row_count = 0

                self.deps.knowledge_manifest.mark_reindexed(
                    content_hash,
                    source_path=str(active_path),
                    source_type=prepared.get("source_type", "document"),
                    parser_type=prepared.get("parser_type", "plain_text"),
                    chunk_count=len(prepared["documents"]),
                    spreadsheet_row_count=row_count,
                )
                updated_record = self.deps.knowledge_manifest.get_record(content_hash) or {}
                self.deps.audit_service.record_audit(
                    "reindex",
                    record,
                    status="success",
                    message=f"重建完成: chunks={len(prepared['documents'])}, rows={row_count}",
                    backup_path=backup_path,
                    before=manifest_snapshot,
                    after=updated_record,
                    actor={"actor_id": user_id, "actor_name": user_id},
                )
                self.deps.audit_service.clear_runtime_cache()
                return {
                    "success": True,
                    "message": "重建索引完成",
                    "removed_previous_vectors": removed,
                    "chunks": len(prepared["documents"]),
                    "spreadsheet_rows": row_count,
                    "backup_path": backup_path,
                }, 200
            except Exception as exc:
                rollback_errors = self._restore_snapshots(
                    content_hash,
                    kb_snapshot=kb_snapshot,
                    sheet_snapshot=sheet_snapshot,
                    manifest_snapshot=manifest_snapshot,
                )
                self.deps.audit_service.record_audit(
                    "reindex",
                    record,
                    status="failed",
                    message=f"{exc}; rollback_errors={rollback_errors}",
                    backup_path=backup_path,
                    before=manifest_snapshot,
                    after=self.deps.knowledge_manifest.get_record(content_hash) or {},
                    actor={"actor_id": user_id, "actor_name": user_id},
                )
                logger.exception("知识库文件重建失败: %s", exc)
                return {
                    "success": False,
                    "message": f"重建失败，已尝试回滚: {str(exc)[:160]}",
                    "rolled_back": not rollback_errors,
                    "rollback_errors": rollback_errors,
                }, 500

    def _restore_snapshots(self, content_hash: str, *, kb_snapshot=None, sheet_snapshot=None, manifest_snapshot=None):
        errors = []
        if kb_snapshot is not None:
            try:
                self.deps.knowledge_base.restore_state(kb_snapshot)
            except Exception as exc:
                errors.append(f"向量回滚失败: {exc}")
        if sheet_snapshot is not None:
            try:
                self.deps.spreadsheet_store.restore_file_snapshot(sheet_snapshot)
            except Exception as exc:
                errors.append(f"表格回滚失败: {exc}")
        if manifest_snapshot is not None:
            try:
                self.deps.knowledge_manifest.restore_record(manifest_snapshot)
            except Exception as exc:
                errors.append(f"manifest 回滚失败: {exc}")
        return errors

    @staticmethod
    def _restore_archived_file(archived_path: str, original_path: str):
        if not archived_path or not original_path:
            return
        archived = Path(archived_path)
        original = Path(original_path)
        if archived.exists() and not original.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            archived.replace(original)
