"""Service wrapper for transforming temporary spreadsheet uploads."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

XLSX_MIMETYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@dataclass
class SpreadsheetTransformResult:
    filename: str = ""
    payload: Optional[bytes] = None
    summary: Optional[dict] = None
    mimetype: str = XLSX_MIMETYPE
    error: Optional[dict] = None
    status: int = 200

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class SpreadsheetTransformDependencies:
    upload_manager: Any
    is_spreadsheet_file: Callable[[str], bool]
    transform_spreadsheet_file: Callable[..., dict]
    deepseek_api_key: str = ""
    llm_timeout: float = 120
    llm_client_factory: Optional[Callable[..., Any]] = None


class SpreadsheetTransformService:
    def __init__(self, deps: SpreadsheetTransformDependencies):
        self.deps = deps

    def transform(self, data: dict, *, user_id: str) -> SpreadsheetTransformResult:
        request_data = data or {}
        file_id = (request_data.get("file_id") or "").strip()
        instruction = (request_data.get("instruction") or "").strip()

        if not file_id:
            return self._error("缺少表格文件", 400)
        if not instruction:
            return self._error("请填写筛选或排序规则", 400)

        info = self.deps.upload_manager.get_temp_file_info(file_id, user_id)
        if not info:
            return self._error("文件不存在或已过期，请重新上传", 404)

        file_path = info.get("file_path")
        filename = info.get("filename") or "上传表格.xlsx"
        if not self.deps.is_spreadsheet_file(file_path or filename):
            return self._error("当前文件不是 Excel/CSV 表格", 400)

        try:
            result = self.deps.transform_spreadsheet_file(
                file_path,
                filename,
                instruction,
                client=self._build_llm_client(),
                model="deepseek-v4-flash",
            )
        except Exception as exc:
            logger.exception("表格处理失败: %s", exc)
            return self._error(f"表格处理失败: {str(exc)[:120]}", 500)

        if not result.get("success"):
            return SpreadsheetTransformResult(error=result, status=422)

        return SpreadsheetTransformResult(
            filename=result["filename"],
            payload=result["content"],
            summary=result.get("summary", {}),
        )

    def _build_llm_client(self):
        if not self.deps.deepseek_api_key:
            return None

        factory = self.deps.llm_client_factory
        if factory is None:
            try:
                from openai import OpenAI
                factory = OpenAI
            except Exception as exc:
                logger.warning("表格规则 AI 客户端初始化失败，将使用规则兜底: %s", exc)
                return None

        try:
            return factory(
                api_key=self.deps.deepseek_api_key,
                base_url="https://api.deepseek.com/v1",
                timeout=self.deps.llm_timeout,
            )
        except Exception as exc:
            logger.warning("表格规则 AI 客户端初始化失败，将使用规则兜底: %s", exc)
            return None

    @staticmethod
    def _error(message: str, status: int) -> SpreadsheetTransformResult:
        return SpreadsheetTransformResult(
            error={"success": False, "message": message},
            status=status,
        )
