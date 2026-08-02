"""
文件上传处理模块

功能：
1. 接收 Word/PDF 文件上传
2. 解析文本内容
3. 模式 A：存入知识库（长期共享）
4. 模式 B：临时参考（30分钟）
5. 进度反馈
"""

import os
import re
import uuid
import json
import shutil
import threading
import hashlib
import fcntl
from contextlib import suppress, contextmanager
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from werkzeug.utils import secure_filename
from flask import request, jsonify, g

from app_config import DEPARTMENT_DIRS
from document_parser import parse_document, parse_document_with_format
from knowledge_manifest import KnowledgeIngestionManifest
from semantic_chunker import StructuralChunker
from spreadsheet_store import SpreadsheetStore, is_spreadsheet_file, parse_spreadsheet
from storage_config import UPLOADS_DIR


class UploadManager:
    """文件上传管理器"""

    # 允许的文件类型
    ALLOWED_EXTENSIONS = {'.docx', '.doc', '.pdf', '.txt', '.md', '.xlsx', '.xls', '.csv'}
    DEPARTMENT_SCOPES = set(DEPARTMENT_DIRS)

    # 文件大小限制 (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # 临时文件有效期 (30分钟)
    TEMP_FILE_TTL = 30 * 60

    def __init__(self, base_path: str = "./uploads"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.temp_path = self.base_path / "temp"
        self.knowledge_path = self.base_path / "knowledge"
        self.temp_path.mkdir(exist_ok=True)
        self.knowledge_path.mkdir(exist_ok=True)

        # 内存中的临时文件索引
        self._temp_files: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._knowledge_ingest_lock_path = self.knowledge_path / ".ingest.lock"

        # 启动清理线程
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """启动定时清理线程"""
        def cleanup_loop():
            import time
            while True:
                time.sleep(300)  # 每5分钟清理一次
                self._cleanup_expired_files()

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    def _cleanup_expired_files(self):
        """清理过期的临时文件"""
        now = datetime.now()
        expired = []

        with self._lock:
            for file_id, info in list(self._temp_files.items()):
                if now - info['created_at'] > timedelta(seconds=self.TEMP_FILE_TTL):
                    expired.append(file_id)
            for meta_path in self.temp_path.glob("*.json"):
                info = self._load_temp_file_info_unlocked(meta_path.stem, cache=False)
                if info and now > info['expires_at']:
                    expired.append(info['file_id'])

        for file_id in set(expired):
            self._delete_temp_file(file_id)

    def _delete_temp_file(self, file_id: str):
        """删除临时文件"""
        with self._lock:
            info = self._temp_files.get(file_id) or self._load_temp_file_info_unlocked(file_id, cache=False)
            if info:
                try:
                    if os.path.exists(info['file_path']):
                        os.remove(info['file_path'])
                except OSError:
                    pass
            self._temp_files.pop(file_id, None)
            with suppress(OSError):
                self._temp_meta_path(file_id).unlink()

    def _temp_meta_path(self, file_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", file_id or "")
        return self.temp_path / f"{safe_id}.json"

    def _serialize_temp_info(self, info: Dict) -> Dict:
        payload = dict(info)
        for key in ("created_at", "expires_at"):
            if isinstance(payload.get(key), datetime):
                payload[key] = payload[key].isoformat()
        return payload

    def _parse_temp_info(self, payload: Dict) -> Optional[Dict]:
        try:
            info = dict(payload)
            info['created_at'] = datetime.fromisoformat(info['created_at'])
            info['expires_at'] = datetime.fromisoformat(info['expires_at'])
            return info
        except Exception:
            return None

    def _persist_temp_file_info(self, info: Dict):
        meta_path = self._temp_meta_path(info['file_id'])
        tmp_path = meta_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._serialize_temp_info(info), f, ensure_ascii=False)
        os.replace(tmp_path, meta_path)

    def _load_temp_file_info_unlocked(self, file_id: str, cache: bool = True) -> Optional[Dict]:
        meta_path = self._temp_meta_path(file_id)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                info = self._parse_temp_info(json.load(f))
        except Exception:
            return None
        if not info:
            return None
        if cache:
            self._temp_files[file_id] = info
        return info

    def allowed_file(self, filename: str) -> bool:
        """检查文件类型是否允许"""
        ext = Path(filename).suffix.lower()
        return ext in self.ALLOWED_EXTENSIONS

    def validate_file(self, file) -> tuple[bool, str]:
        """验证文件"""
        # 检查文件名
        if not file or not file.filename:
            return False, "请选择文件"

        # 检查文件类型
        if not self.allowed_file(file.filename):
            return False, f"不支持的文件类型，仅支持：{', '.join(self.ALLOWED_EXTENSIONS)}"

        # 检查文件大小
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > self.MAX_FILE_SIZE:
            return False, f"文件过大（{size/1024/1024:.1f}MB），请压缩后重试（限制10MB）"

        if size == 0:
            return False, "文件为空"

        return True, "OK"

    def save_temp_file(self, file, user_id: str) -> tuple[str, str]:
        """保存上传的临时文件"""
        # 生成唯一文件名
        file_id = str(uuid.uuid4())[:8]
        ext = Path(file.filename).suffix.lower()
        safe_filename = self._safe_filename(file.filename, ext)

        # 保存路径
        save_path = self.temp_path / f"{user_id}_{file_id}{ext}"

        # 保存文件
        file.save(save_path)

        return str(save_path), safe_filename

    def _safe_filename(self, filename: str, ext: str = "") -> str:
        """生成可落盘的安全文件名，兼容中文文件名被 secure_filename 清空的情况。"""
        raw_name = Path(filename or "").name
        raw_name = re.sub(r"[\x00-\x1f/\\:]+", "_", raw_name).strip(" .")
        if raw_name and (Path(raw_name).suffix or not ext):
            return raw_name[:180]

        safe_filename = secure_filename(filename or "")
        if safe_filename and (Path(safe_filename).suffix or not ext):
            return safe_filename[:180]
        fallback_ext = ext or Path(filename or "").suffix.lower()
        return f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}{fallback_ext}"

    def _delete_file_quietly(self, file_path: str):
        with suppress(Exception):
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    @contextmanager
    def _knowledge_ingest_guard(self):
        """Serialize knowledge ingestion across threads/processes."""
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        self._knowledge_ingest_lock_path.touch(exist_ok=True)
        with open(self._knowledge_ingest_lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _file_sha256(self, file_path: str) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _chunk_text_hash(self, content: str) -> str:
        normalized = re.sub(r"\s+", " ", content or "").strip()
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

    def _heading_level(self, line: str) -> int:
        line = (line or "").strip()
        if re.match(r"^#{1,6}\s+", line):
            return min(len(line) - len(line.lstrip("#")), 6)
        if re.match(r"^[一二三四五六七八九十]+[、，,]", line):
            return 1
        if re.match(r"^第[一二三四五六七八九十\d]+[章节条]", line):
            return 1
        if re.match(r"^（[一二三四五六七八九十]+）", line):
            return 2
        if re.match(r"^\d+[\.\、](?!\d)", line):
            return 3
        if re.match(r"^\(\d+\)", line):
            return 4
        return 0

    def _detect_section_title(self, content: str) -> str:
        for line in (content or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if len(line) <= 100 and self._heading_level(line):
                return re.sub(r"^#{1,6}\s+", "", line)
            return ""
        return ""

    def _attach_document_structure(self, items: List[Dict]) -> List[Dict]:
        structured = []
        heading_path: List[str] = []
        for item in items:
            content = (item.get("content") or "").strip()
            section_title = item.get("section_title") or self._detect_section_title(content)
            if section_title:
                level = self._heading_level(content.splitlines()[0] if content else section_title) or 1
                heading_path = heading_path[:max(level - 1, 0)]
                heading_path.append(section_title)

            enriched = dict(item)
            enriched["section_title"] = section_title or (heading_path[-1] if heading_path else "")
            enriched["heading_path"] = list(heading_path)
            structured.append(enriched)
        return structured

    def _prepare_raw_items(self, parsed_items: List[Dict], ext: str,
                           is_spreadsheet_parsed: bool) -> List[Dict]:
        if is_spreadsheet_parsed:
            return parsed_items

        chunker = StructuralChunker(max_chunk_size=800, min_chunk_size=10)
        raw_items = []
        for item in parsed_items:
            content = (item.get('content') or '').strip()
            if not content:
                continue
            base = {
                "format": item.get("format", {}),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "parse_warnings": item.get("parse_warnings", []),
            }
            if len(content) > 800:
                for sub_chunk in chunker.split(content):
                    raw_items.append({"content": sub_chunk, **base})
            else:
                raw_items.append({"content": content, **base})

        return self._attach_document_structure(raw_items)

    def _parser_type_for(self, ext: str, is_spreadsheet_parsed: bool, is_docx_parsed: bool) -> str:
        if is_spreadsheet_parsed:
            return "spreadsheet_row"
        if is_docx_parsed:
            return "docx_paragraph"
        if ext == ".pdf":
            return "pdf_page"
        if ext in {".txt", ".md"}:
            return "text_structural"
        return "plain_text"

    def _is_review_proposal_template(self, filename: str, category: str = "") -> bool:
        text = f"{filename} {category}"
        return any(marker in text for marker in ["议案", "审议", "办公会"])

    def _review_proposal_template_summary(self) -> Dict:
        return {
            "content": (
                "会议审议案/审议模板完整结构："
                "1. 标题，通常为“关于审议XXX的有关事宜”，标题使用方正小标宋简体二号（22pt）居中；"
                "2. 主送称谓“各位领导：”；"
                "3. 正文先说明事由，再引用制度、办法、规定或细则的具体条款，随后写明需提请审议或决议的事项；"
                "4. 正文结尾使用“以上，请审议。”；"
                "5. 如有附件，使用“附件：1.XXX”“2.XXX”等格式列明；"
                "6. 单独写明“此议案需决议：是否通过XXX并XXX。”；"
                "7. 末尾为提案部门和日期。正文、附件、决议行和落款一般使用仿宋三号（16pt）。"
            ),
            "format": {},
            "section_title": "会议审议案模板结构摘要",
            "heading_path": ["会议审议案模板结构摘要"],
        }

    def parse_file(self, file_path: str, display_filename: str = None) -> tuple[bool, str, List[Dict]]:
        """
        解析文件
        返回: (是否成功, 错误信息, 解析结果列表)
        """
        from pathlib import Path

        path = Path(file_path)

        try:
            # 根据文件类型选择解析方式
            if is_spreadsheet_file(path):
                rows = parse_spreadsheet(path, display_filename=display_filename)
                if rows:
                    return True, "", [
                        {
                            "content": row.text,
                            "format": {},
                            "spreadsheet": {
                                "sheet_name": row.sheet_name,
                                "row_number": row.row_number,
                                "row_type": row.row_type,
                                "headers": row.headers,
                                "values": row.values,
                            }
                        }
                        for row in rows
                    ]
            elif path.suffix.lower() == '.docx':
                # Word 文档 - 尝试提取格式
                formatted = parse_document_with_format(path)
                if formatted:
                    return True, "", formatted
                # 回退到纯文本
                text = parse_document(path)
                if text:
                    return True, "", [{"content": text, "format": {}}]
            elif path.suffix.lower() in {'.pdf', '.txt', '.md'}:
                formatted = parse_document_with_format(path)
                if formatted:
                    return True, "", formatted
            else:
                # PDF 或其他 - 纯文本解析
                text = parse_document(path)
                if text:
                    return True, "", [{"content": text, "format": {}}]

            return False, "文件解析失败，可能是加密、损坏或表格为空", []

        except Exception as e:
            return False, f"解析错误: {str(e)}", []

    def build_knowledge_documents(self, file_path: str, filename: str,
                                  category: str, user_id: str,
                                  department: str = "",
                                  access_level: Optional[str] = None,
                                  content_hash: Optional[str] = None,
                                  uploaded_at: Optional[str] = None) -> Dict:
        """解析已落盘文件并构造向量入库文档，不移动或删除原文件。"""
        success, error, parsed_items = self.parse_file(file_path, display_filename=filename)
        if not success:
            return {"success": False, "message": error}

        source_path = Path(file_path)
        ext = source_path.suffix.lower()
        content_hash = content_hash or self._file_sha256(file_path)
        uploaded_at = uploaded_at or datetime.now().isoformat()
        if access_level is None:
            access_level = "restricted" if department in self.DEPARTMENT_SCOPES else "public"
        metadata_department = department if access_level == "restricted" else ""

        is_spreadsheet_parsed = is_spreadsheet_file(source_path)
        is_docx_parsed = ext == '.docx' and len(parsed_items) > 1

        raw_items = self._prepare_raw_items(parsed_items, ext, is_spreadsheet_parsed)

        valid_items = [
            item for item in raw_items
            if len(item.get('content', '').strip()) >= 10
        ]
        if self._is_review_proposal_template(filename, category):
            valid_items.append(self._review_proposal_template_summary())
        if not valid_items:
            return {"success": False, "message": "未能提取有效文本内容"}

        documents = []
        total_chunks = len(valid_items)
        doc_type = "报表数据" if is_spreadsheet_parsed else self._detect_doc_type(filename, category)
        parser_type = self._parser_type_for(ext, is_spreadsheet_parsed, is_docx_parsed)
        source_type = "spreadsheet" if is_spreadsheet_parsed else "document"

        for chunk_index, item in enumerate(valid_items):
            content = item.get('content', '').strip()
            spreadsheet_meta = item.get("spreadsheet", {}) or {}
            format_info = item.get('format', {})
            if format_info and format_info.get('font'):
                format_desc = f"[格式: {format_info.get('font', '')} {format_info.get('size', '')}]"
                enhanced_content = f"{content}\n{format_desc}"
            else:
                enhanced_content = content

            documents.append({
                "content": enhanced_content,
                "metadata": {
                    "source": str(source_path),
                    "source_path": str(source_path),
                    "filename": filename,
                    "original_filename": filename,
                    "file_size": source_path.stat().st_size if source_path.exists() else 0,
                    "content_hash": content_hash,
                    "category": category,
                    "access_level": access_level,
                    "department": metadata_department,
                    "uploaded_by": user_id,
                    "uploaded_at": uploaded_at,
                    "format": format_info,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "doc_type": doc_type,
                    "parser_type": parser_type,
                    "source_type": source_type,
                    "sheet_name": spreadsheet_meta.get("sheet_name", ""),
                    "row_start": spreadsheet_meta.get("row_number"),
                    "row_end": spreadsheet_meta.get("row_number"),
                    "row_type": spreadsheet_meta.get("row_type", "data"),
                    "column_headers": spreadsheet_meta.get("headers", []),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "section_title": item.get("section_title", ""),
                    "heading_path": item.get("heading_path", []),
                    "chunk_text_hash": self._chunk_text_hash(content),
                    "parse_warnings": item.get("parse_warnings", []),
                }
            })

        return {
            "success": True,
            "documents": documents,
            "content_hash": content_hash,
            "uploaded_at": uploaded_at,
            "source_type": source_type,
            "parser_type": parser_type,
            "is_spreadsheet": is_spreadsheet_parsed,
            "access_level": access_level,
            "department": metadata_department,
            "chunk_count": len(documents),
        }

    def archive_knowledge_file(self, file_path: str) -> str:
        """软删除原文件：移动到 uploads/knowledge_archive/YYYYMMDD。"""
        source = Path(file_path)
        if not source.exists():
            return ""

        archive_dir = self.base_path / "knowledge_archive" / datetime.now().strftime("%Y%m%d")
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / source.name
        counter = 1
        while target.exists():
            target = archive_dir / f"{source.stem}_{counter}{source.suffix}"
            counter += 1
        shutil.move(str(source), target)
        return str(target)

    def process_knowledge_upload(self, file_path: str, filename: str,
                                  category: str, user_id: str,
                                  knowledge_base, department: str = "") -> Dict:
        """
        处理知识库上传

        流程:
        1. 解析文件
        2. 移动到知识库目录
        3. 生成 embedding 并索引

        Args:
            department: 部门名称，如果在8个部门列表中则设为 restricted 权限
        """
        # 1. 解析文件
        success, error, parsed_items = self.parse_file(file_path, display_filename=filename)
        if not success:
            self._delete_file_quietly(file_path)
            return {"success": False, "message": error}

        content_hash = self._file_sha256(file_path)

        with self._knowledge_ingest_guard():
            return self._process_knowledge_upload_locked(
                file_path=file_path,
                filename=filename,
                category=category,
                user_id=user_id,
                knowledge_base=knowledge_base,
                department=department,
                parsed_items=parsed_items,
                content_hash=content_hash,
            )

    def _process_knowledge_upload_locked(self, *, file_path: str, filename: str,
                                         category: str, user_id: str,
                                         knowledge_base, department: str,
                                         parsed_items: List[Dict],
                                         content_hash: str) -> Dict:
        """Finish knowledge ingestion while the cross-process ingest lock is held."""
        # 2. 移动到知识库目录
        category_path = self.knowledge_path / category
        category_path.mkdir(exist_ok=True)

        ext = Path(file_path).suffix.lower()
        safe_name = self._safe_filename(filename, ext)
        final_path = category_path / f"{datetime.now().strftime('%Y%m%d')}_{safe_name}"

        for meta in getattr(knowledge_base, "metadatas", []) or []:
            same_hash = meta.get("content_hash") == content_hash
            same_scope = meta.get("category") == category and meta.get("department", "") == department
            if same_hash and same_scope:
                self._delete_file_quietly(file_path)
                return {
                    "success": False,
                    "message": f"该文件内容已存在于知识库「{category}」，未重复入库",
                    "duplicate": True,
                    "filename": filename,
                    "category": category,
                    "content_hash": content_hash,
                    "existing_source": meta.get("source_path") or meta.get("source", "")
                }

        # 如果文件已存在，添加序号
        counter = 1
        while final_path.exists():
            stem = Path(safe_name).stem
            final_path = category_path / f"{datetime.now().strftime('%Y%m%d')}_{stem}_{counter}{ext}"
            counter += 1

        try:
            shutil.move(file_path, final_path)
        except Exception as e:
            self._delete_file_quietly(file_path)
            return {"success": False, "message": f"保存文件失败: {str(e)}"}

        # 3. 准备添加到知识库的文档
        access_level = "restricted" if department in self.DEPARTMENT_SCOPES else "public"

        is_spreadsheet_parsed = is_spreadsheet_file(final_path)

        # 判断是否是 docx 段落级解析（多个段落且有格式信息）
        is_docx_parsed = ext == '.docx' and len(parsed_items) > 1

        raw_items = self._prepare_raw_items(parsed_items, ext, is_spreadsheet_parsed)

        valid_items = [
            item for item in raw_items
            if len(item.get('content', '').strip()) >= 10
        ]
        if self._is_review_proposal_template(filename, category):
            valid_items.append(self._review_proposal_template_summary())

        documents = []
        total_chunks = len(valid_items)
        doc_type = "报表数据" if is_spreadsheet_parsed else self._detect_doc_type(filename, category)
        uploaded_at = datetime.now().isoformat()
        for chunk_index, item in enumerate(valid_items):
            content = item.get('content', '').strip()
            spreadsheet_meta = item.get("spreadsheet", {}) or {}

            # 添加格式信息
            format_info = item.get('format', {})
            if format_info and format_info.get('font'):
                format_desc = f"[格式: {format_info.get('font', '')} {format_info.get('size', '')}]"
                enhanced_content = f"{content}\n{format_desc}"
            else:
                enhanced_content = content

            documents.append({
                "content": enhanced_content,
                "metadata": {
                    "source": str(final_path),
                    "source_path": str(final_path),
                    "filename": filename,
                    "original_filename": filename,
                    "file_size": final_path.stat().st_size if final_path.exists() else 0,
                    "content_hash": content_hash,
                    "category": category,
                    "access_level": access_level,
                    "department": department if department in self.DEPARTMENT_SCOPES else "",
                    "uploaded_by": user_id,
                    "uploaded_at": uploaded_at,
                    "format": format_info,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "doc_type": doc_type,
                    "parser_type": self._parser_type_for(ext, is_spreadsheet_parsed, is_docx_parsed),
                    "source_type": "spreadsheet" if is_spreadsheet_parsed else "document",
                    "sheet_name": spreadsheet_meta.get("sheet_name", ""),
                    "row_start": spreadsheet_meta.get("row_number"),
                    "row_end": spreadsheet_meta.get("row_number"),
                    "row_type": spreadsheet_meta.get("row_type", "data"),
                    "column_headers": spreadsheet_meta.get("headers", []),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "section_title": item.get("section_title", ""),
                    "heading_path": item.get("heading_path", []),
                    "chunk_text_hash": self._chunk_text_hash(content),
                    "parse_warnings": item.get("parse_warnings", []),
                }
            })

        # 4. 添加到知识库索引
        if documents:
            spreadsheet_store = None
            manifest = None
            try:
                manifest = KnowledgeIngestionManifest(Path(knowledge_base.index_dir) / "ingestion_manifest.sqlite")
                manifest.record_started(
                    content_hash=content_hash,
                    filename=filename,
                    source_path=str(final_path),
                    category=category,
                    access_level=access_level,
                    department=department if department in self.DEPARTMENT_SCOPES else "",
                    uploaded_by=user_id,
                    uploaded_at=uploaded_at,
                    source_type="spreadsheet" if is_spreadsheet_parsed else "document",
                    parser_type=documents[0]["metadata"].get("parser_type", ""),
                    chunk_count=len(documents),
                )

                if is_spreadsheet_parsed:
                    spreadsheet_store = SpreadsheetStore(Path(knowledge_base.index_dir) / "spreadsheets.sqlite")
                    rows = parse_spreadsheet(final_path, display_filename=filename)
                    row_count = spreadsheet_store.upsert_file_rows(
                        content_hash=content_hash,
                        filename=filename,
                        source_path=str(final_path),
                        category=category,
                        access_level=access_level,
                        department=department if department in self.DEPARTMENT_SCOPES else "",
                        uploaded_by=user_id,
                        uploaded_at=uploaded_at,
                        rows=rows,
                    )
                    manifest.mark_structured_indexed(content_hash, row_count)

                knowledge_base.add_documents(documents)
                manifest.mark_vector_indexed(content_hash)
                return {
                    "success": True,
                    "message": f"已添加到知识库「{category}」分类，共 {len(documents)} 个片段",
                    "file_id": str(final_path),
                    "filename": filename,
                    "category": category,
                    "content_hash": content_hash,
                    "chunks": len(documents)
                }
            except Exception as e:
                if manifest is not None:
                    manifest.mark_failed(content_hash, str(e))
                if spreadsheet_store is not None:
                    spreadsheet_store.delete_file(content_hash)
                self._delete_file_quietly(str(final_path))
                return {
                    "success": False,
                    "message": f"添加到索引失败: {str(e)}",
                    "filename": filename,
                    "category": category,
                    "content_hash": content_hash,
                }
        else:
            self._delete_file_quietly(str(final_path))
            return {
                "success": False,
                "message": "未能提取有效文本内容",
                "filename": filename,
                "category": category,
                "content_hash": content_hash,
            }

    def _detect_doc_type(self, filename: str, category: str = "") -> str:
        text = f"{filename} {category}"
        for doc_type in ["议案", "审议", "通知", "请示", "报告", "函", "对策建议", "情况反映", "会议纪要", "方案"]:
            if doc_type in text:
                return "会议审议案" if doc_type in {"议案", "审议"} else doc_type
        return "参考材料"

    def process_temp_upload(self, file_path: str, filename: str,
                            user_id: str) -> Dict:
        """
        处理临时上传

        流程:
        1. 解析文件
        2. 提取全文（最多5000字）
        3. 保存到内存索引
        4. 30分钟后自动清理
        """
        # 1. 解析文件
        success, error, parsed_items = self.parse_file(file_path, display_filename=filename)
        if not success:
            self._delete_file_quietly(file_path)
            return {"success": False, "message": error}

        # 2. 合并文本并截断
        full_text = "\n\n".join([item.get('content', '') for item in parsed_items])

        # 截断到5000字
        MAX_CHARS = 5000
        was_truncated = len(full_text) > MAX_CHARS
        if was_truncated:
            full_text = full_text[:MAX_CHARS] + "\n\n[内容已截断...]"

        # 3. 生成临时文件ID
        file_id = str(uuid.uuid4())[:12]

        # 4. 保存到共享临时索引，支持多 worker 读取
        info = {
            'file_id': file_id,
            'user_id': user_id,
            'filename': filename,
            'file_path': file_path,
            'content': full_text,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=self.TEMP_FILE_TTL)
        }
        with self._lock:
            self._temp_files[file_id] = info
            self._persist_temp_file_info(info)

        return {
            "success": True,
            "message": f"已提取 {len(full_text)} 字符{'（已截断）' if was_truncated else ''}，30分钟后自动删除",
            "file_id": file_id,
            "filename": filename,
            "char_count": len(full_text),
            "truncated": was_truncated,
            "expires_in": self.TEMP_FILE_TTL
        }

    def get_temp_content(self, file_id: str, user_id: str) -> Optional[str]:
        """获取临时文件内容（验证归属）"""
        with self._lock:
            if file_id not in self._temp_files:
                self._load_temp_file_info_unlocked(file_id)
            if file_id not in self._temp_files:
                return None

            info = self._temp_files[file_id]

            # 验证归属
            if info['user_id'] != user_id:
                return None

            # 检查是否过期
            if datetime.now() > info['expires_at']:
                self._delete_temp_file(file_id)
                return None

            return info['content']

    def get_temp_file_info(self, file_id: str, user_id: str) -> Optional[Dict]:
        """获取临时文件元信息（验证归属），用于需要读取原文件的工具。"""
        with self._lock:
            if file_id not in self._temp_files:
                self._load_temp_file_info_unlocked(file_id)
            if file_id not in self._temp_files:
                return None

            info = self._temp_files[file_id]
            if info['user_id'] != user_id:
                return None

            if datetime.now() > info['expires_at']:
                self._delete_temp_file(file_id)
                return None

            return dict(info)

    def get_upload_stats(self) -> Dict:
        """获取上传统计"""
        temp_size = 0
        with self._lock:
            for f in self._temp_files.values():
                try:
                    if os.path.exists(f['file_path']):
                        temp_size += os.path.getsize(f['file_path']) / 1024 / 1024
                except OSError:
                    pass
            return {
                "temp_files": len(self._temp_files),
                "temp_storage_mb": round(temp_size, 2),
                "knowledge_categories": len(list(self.knowledge_path.iterdir())),
                "knowledge_files": sum(
                    1 for _ in self.knowledge_path.rglob('*')
                    if _.is_file()
                )
            }


# ========== 全局实例 ==========
_upload_manager: Optional[UploadManager] = None


def get_upload_manager() -> UploadManager:
    """获取全局上传管理实例"""
    global _upload_manager
    if _upload_manager is None:
        _upload_manager = UploadManager(str(UPLOADS_DIR))
    return _upload_manager
