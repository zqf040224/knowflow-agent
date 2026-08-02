"""Upload services for temporary files and knowledge-base ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class UploadServiceDependencies:
    upload_manager: Any
    memory: Any
    knowledge_base: Any
    knowledge_agent: Any
    knowledge_manifest: Any
    knowledge_source_dir: Path
    spreadsheet_db_path: Path
    department_dirs: set[str]
    record_knowledge_upload_audit: Callable[..., None]


class UploadService:
    def __init__(self, deps: UploadServiceDependencies):
        self.deps = deps

    def allowed_categories(self, *, user_role: str, user_department: str) -> list[dict]:
        kb_root = self.deps.knowledge_source_dir
        categories = []

        center_dir = kb_root / "公共资料"
        if center_dir.is_dir():
            categories.append({
                "id": "公共资料",
                "name": "公共资料",
                "icon": "🏛️",
                "desc": "全院共享，所有人可见",
                "access_level": "public",
                "department": "",
            })

        for department in sorted(self.deps.department_dirs):
            dept_path = kb_root / department
            if not dept_path.is_dir():
                continue
            if user_role == "admin" or user_department == department:
                categories.append({
                    "id": department,
                    "name": department,
                    "icon": "📂",
                    "desc": "部门内部文件，仅本部门可见",
                    "access_level": "restricted",
                    "department": department,
                })

        return categories

    def upload_file(
        self,
        file,
        *,
        user_id: str,
        user_role: str,
        user_department: str,
        mode: str = "temp",
        category: str = "",
    ) -> tuple[dict, int]:
        if not file:
            return {"success": False, "message": "请选择文件"}, 400

        is_valid, error_msg = self.deps.upload_manager.validate_file(file)
        if not is_valid:
            return {"success": False, "message": error_msg}, 400

        file_path, filename = self.deps.upload_manager.save_temp_file(file, user_id)

        if mode == "knowledge":
            prepared, status = self.prepare_knowledge_upload(
                file,
                user_id=user_id,
                user_role=user_role,
                user_department=user_department,
                category=category,
                saved_file_path=file_path,
                saved_filename=filename,
            )
            if status != 200:
                return prepared, status
            return self.process_prepared_knowledge_upload(prepared)

        return self.deps.upload_manager.process_temp_upload(
            file_path=file_path,
            filename=filename,
            user_id=user_id,
        ), 200

    def prepare_knowledge_upload(
        self,
        file,
        *,
        user_id: str,
        user_role: str,
        user_department: str,
        category: str,
        saved_file_path: str = "",
        saved_filename: str = "",
    ) -> tuple[dict, int]:
        if not file and not saved_file_path:
            return {"success": False, "message": "请选择文件"}, 400

        if not saved_file_path:
            is_valid, error_msg = self.deps.upload_manager.validate_file(file)
            if not is_valid:
                return {"success": False, "message": error_msg}, 400
            saved_file_path, saved_filename = self.deps.upload_manager.save_temp_file(file, user_id)

        validation = self._validate_knowledge_upload_request(
            saved_file_path,
            user_role=user_role,
            user_department=user_department,
            category=category,
        )
        if validation[1] != 200:
            return validation

        category_info = validation[0]["category_info"]
        return {
            "success": True,
            "file_path": saved_file_path,
            "filename": saved_filename,
            "user_id": user_id,
            "category": category_info["id"],
            "department": category_info.get("department", ""),
        }, 200

    def process_prepared_knowledge_upload(self, prepared: dict) -> tuple[dict, int]:
        return self._process_knowledge_upload(
            prepared["file_path"],
            prepared["filename"],
            user_id=prepared["user_id"],
            category=prepared["category"],
            department=prepared.get("department", ""),
        )

    def cleanup_prepared_upload(self, prepared: dict) -> None:
        self._remove_temp_file((prepared or {}).get("file_path", ""))

    def _validate_knowledge_upload_request(
        self,
        file_path: str,
        *,
        user_role: str,
        user_department: str,
        category: str,
    ) -> tuple[dict, int]:
        if not category:
            self._remove_temp_file(file_path)
            return {"success": False, "message": "请选择知识库分类"}, 400

        category_info = self._resolve_allowed_category(
            category,
            user_role=user_role,
            user_department=user_department,
        )
        if not category_info:
            self._remove_temp_file(file_path)
            return {"success": False, "message": "无权限上传到该知识库分类"}, 403

        return {"success": True, "category_info": category_info}, 200

    def _process_knowledge_upload(
        self,
        file_path: str,
        filename: str,
        *,
        user_id: str,
        category: str,
        department: str = "",
    ) -> tuple[dict, int]:
        result = self.deps.upload_manager.process_knowledge_upload(
            file_path=file_path,
            filename=filename,
            category=category,
            user_id=user_id,
            knowledge_base=self.deps.knowledge_base,
            department=department,
        )
        audit_result = dict(result)
        audit_result.setdefault("uploaded_by", user_id)
        self.deps.record_knowledge_upload_audit(
            audit_result,
            filename=filename,
            category=category,
            department=department,
        )

        if result.get("success"):
            self.deps.knowledge_agent.refresh()
            self.deps.knowledge_agent.cache_manager.clear_knowledge_cache()
            self.deps.memory.add_message(
                session_id=self.deps.memory.get_or_create_session(user_id),
                role="system",
                content=f"上传了文档到知识库「{category}」: {filename}",
                metadata={"type": "upload", "category": category, "filename": filename},
            )

        return result, 200

    def get_temp_file_content(self, file_id: str, *, user_id: str) -> tuple[dict, int]:
        content = self.deps.upload_manager.get_temp_content(file_id, user_id)
        if content is None:
            return {"success": False, "message": "文件不存在或已过期"}, 404
        return {
            "success": True,
            "file_id": file_id,
            "content": content[:500] + "..." if len(content) > 500 else content,
            "char_count": len(content),
        }, 200

    def upload_categories(self, *, user_role: str, user_department: str) -> dict:
        return {
            "success": True,
            "categories": self.allowed_categories(
                user_role=user_role,
                user_department=user_department,
            ),
        }

    def upload_stats(self) -> dict:
        stats = self.deps.upload_manager.get_upload_stats()
        consistency = self.deps.knowledge_manifest.consistency_report(
            self.deps.knowledge_base,
            self.deps.spreadsheet_db_path,
        )
        return {
            "success": True,
            "stats": stats,
            "ingestion_consistency": {
                "ok": consistency.get("ok", True),
                "checked_count": consistency.get("checked_count", 0),
                "issue_count": consistency.get("issue_count", 0),
                "issues": consistency.get("issues", [])[:10],
            },
        }

    def _resolve_allowed_category(self, category_id: str, *, user_role: str, user_department: str):
        return next((
            category
            for category in self.allowed_categories(user_role=user_role, user_department=user_department)
            if category["id"] == category_id
        ), None)

    @staticmethod
    def _remove_temp_file(file_path: str):
        try:
            os.remove(file_path)
        except OSError:
            pass
