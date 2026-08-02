"""Runtime dependency context for registered Flask routes."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path

from flask import g, jsonify, request, send_file

from agent_generate_service import AgentGenerateDependencies, AgentGenerateService
from agents.knowledge_agent import KnowledgeAgent
from agents.orchestrator import AgentOrchestrator
from auth_route_service import AuthRouteService
from beta_ops_service import BetaActor, BetaOpsService, BetaRequestMeta
from chat_container import ChatContainerDependencies, ChatServiceContainer
from embedding_config import ACCESS_LEVELS, UserInfo, build_access_filter
from export_service import ExportService, ExportServiceDependencies
from job_service import JobService
from knowledge_admin_audit_service import KnowledgeAdminAuditDependencies, KnowledgeAdminAuditService
from knowledge_admin_read_service import KnowledgeAdminReadDependencies, KnowledgeAdminReadService
from knowledge_admin_write_service import KnowledgeAdminWriteDependencies, KnowledgeAdminWriteService
from knowledge_base import KnowledgeBase
from knowledge_manifest import KnowledgeIngestionManifest
from runtime_query_service import RuntimeQueryDependencies, RuntimeQueryService
from session_service import SessionService, SessionServiceDependencies
from spreadsheet_export import (
    safe_filename_stem,
    workbook_from_structured_rows,
    workbook_from_text,
    workbook_to_bytes,
)
from spreadsheet_store import SpreadsheetStore, is_spreadsheet_file, parse_spreadsheet
from spreadsheet_transform_service import SpreadsheetTransformDependencies, SpreadsheetTransformService
from spreadsheet_transformer import transform_spreadsheet_file
from storage_config import ADMIN_BACKUP_DIR, KNOWLEDGE_SOURCE_DIR, SPREADSHEET_DB_PATH, storage_health
from upload_service import UploadService, UploadServiceDependencies
from vector_map import build_vector_map
from word_export_service import (
    create_docx as build_docx,
    detect_export_template,
    detect_reimbursement_template,
    explicit_document_request,
    reimbursement_template_path as resolve_reimbursement_template_path,
    resolve_export_template,
    resolve_reimbursement_template,
)

from app_config import DEPARTMENT_DIRS, MAX_REQUEST_SIZE, RATE_LIMITS, new_rate_limit_store


logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    deepseek_api_key: str
    enable_llm_intent_classifier: bool
    auth_manager: object
    memory: object
    upload_manager: object
    knowledge_agent: KnowledgeAgent
    kb_instance: KnowledgeBase
    knowledge_manifest: KnowledgeIngestionManifest
    spreadsheet_store: SpreadsheetStore
    beta_ops_service: BetaOpsService
    auth_route_service: AuthRouteService
    account_admin_service: object
    project_root: Path
    source_materials_dir: Path
    review_proposal_template_path: Path
    reimbursement_template_dir: Path
    reimbursement_template_files: dict
    cookie_secure: bool
    max_request_size: int = MAX_REQUEST_SIZE
    rate_limits: dict = field(default_factory=new_rate_limit_store)
    knowledge_admin_lock: threading.Lock = field(default_factory=threading.Lock)
    chat_container_instance: ChatServiceContainer | None = None
    agent_generate_service_instance: AgentGenerateService | None = None
    export_service_instance: ExportService | None = None
    session_service_instance: SessionService | None = None
    upload_service_instance: UploadService | None = None
    spreadsheet_transform_service_instance: SpreadsheetTransformService | None = None
    knowledge_admin_read_service_instance: KnowledgeAdminReadService | None = None
    knowledge_admin_write_service_instance: KnowledgeAdminWriteService | None = None
    runtime_query_service_instance: RuntimeQueryService | None = None
    knowledge_admin_audit_service_instance: KnowledgeAdminAuditService | None = None
    job_service_instance: JobService | None = None

    def rate_limit(self, f):
        """简单的频率限制装饰器（IP 级别，内存计数）"""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            ip = request.remote_addr or "127.0.0.1"
            endpoint = request.endpoint or f.__name__
            now = _time.time()
            window = 60

            limit = RATE_LIMITS.get(endpoint, 60)
            key = f"{ip}:{endpoint}"
            timestamps = self.rate_limits[key]
            self.rate_limits[key] = [t for t in timestamps if now - t < window]

            if len(self.rate_limits[key]) >= limit:
                return jsonify({"error": "请求过于频繁，请稍后再试"}), 429

            self.rate_limits[key].append(now)
            return f(*args, **kwargs)

        return decorated

    def get_user_info(self):
        """从 Flask g 对象构建 UserInfo，用于知识库权限过滤"""
        if g.user_id:
            try:
                user = self.auth_manager.get_user_by_id(g.user_id)
                if user:
                    return UserInfo(
                        user_id=g.user_id,
                        username=user.get("username", g.username),
                        role=user.get("role", g.role or "user"),
                        department=user.get("department", ""),
                    )
            except Exception:
                pass
            return UserInfo(
                user_id=g.user_id,
                username=g.username or "",
                role=g.role or "user",
                department=getattr(g, "department", ""),
            )
        return UserInfo()

    def record_beta_feedback(self, data: dict) -> dict:
        user_info = self.get_user_info()
        return self.beta_ops_service.record_feedback(
            data or {},
            actor=BetaActor(
                user_id=getattr(g, "user_id", ""),
                username=getattr(g, "username", ""),
                department=user_info.department,
            ),
            request_meta=BetaRequestMeta(
                ip_address=request.remote_addr or "",
                user_agent=request.headers.get("User-Agent", ""),
            ),
        )

    def get_beta_dashboard(self, limit: int = 80) -> dict:
        return self.beta_ops_service.feedback_dashboard(limit)

    def update_beta_feedback_status(self, feedback_id: int, data: dict) -> dict:
        return self.beta_ops_service.update_feedback_status(
            feedback_id,
            data or {},
            handled_by=getattr(g, "username", ""),
        )

    def record_token_usage(self, **kwargs) -> None:
        try:
            self.beta_ops_service.record_token_usage(**kwargs)
        except Exception as exc:
            logger.warning("Token 使用记录失败: %s", exc)

    def record_agent_run_token_usage(self, run_records: list, *, user_id: str,
                                     user_info: UserInfo = None, session_id: str = "",
                                     mode: str = "agent") -> None:
        try:
            self.beta_ops_service.record_agent_run_token_usage(
                run_records,
                user_id=user_id,
                user_info=user_info,
                session_id=session_id,
                mode=mode,
            )
        except Exception as exc:
            logger.warning("Agent token 使用记录失败: %s", exc)

    def get_token_usage_dashboard(self, limit: int = 120) -> dict:
        return self.beta_ops_service.token_usage_dashboard(limit)

    def reimbursement_template_path(self, template_key: str) -> Path:
        return resolve_reimbursement_template_path(
            template_key,
            template_dir=self.reimbursement_template_dir,
            template_files=self.reimbursement_template_files,
        )

    def explicit_document_request(self, text: str, *, has_file_content: bool = False) -> bool:
        return explicit_document_request(text, has_file_content=has_file_content)

    def create_docx(self, text, template_type: str = "default"):
        return build_docx(text, template_type, review_template_path=self.review_proposal_template_path)

    def get_existing_departments(self) -> list:
        return self.account_admin_service.existing_departments()

    def validate_account_department(self, role: str, department: str) -> tuple[bool, str, str]:
        return self.account_admin_service.validate_department(role, department)

    def register_user_from_admin_payload(self, data: dict) -> dict:
        return self.account_admin_service.register_from_admin_payload(data or {})

    def knowledge_actor(self):
        try:
            return {
                "actor_id": getattr(g, "user_id", ""),
                "actor_name": getattr(g, "username", "") or getattr(g, "name", ""),
            }
        except RuntimeError:
            return {"actor_id": "", "actor_name": ""}

    def record_knowledge_upload_audit(self, result: dict, *, filename: str,
                                      category: str, department: str = ""):
        actor = self.knowledge_actor()
        if not actor.get("actor_id") and result.get("uploaded_by"):
            actor = {
                "actor_id": result.get("uploaded_by", ""),
                "actor_name": result.get("uploaded_by", ""),
            }
        self.knowledge_admin_audit_service().record_upload_audit(
            result,
            filename=filename,
            category=category,
            department=department,
            actor=actor,
        )

    def llm_intent_classifier(self, payload):
        """Optional route-only LLM classifier; never generates answer content."""
        if not self.deepseek_api_key:
            return None

        rule_result = payload.get("rule_result") or {}
        compact_payload = {
            "message": payload.get("message", ""),
            "attachments": [
                {
                    "filename": item.get("filename", ""),
                    "is_spreadsheet": bool(item.get("is_spreadsheet")),
                }
                for item in payload.get("attachments", [])
            ],
            "has_last_document": bool(payload.get("has_last_document")),
            "rule_result": rule_result,
            "allowed_intents": payload.get("allowed_intents", []),
        }
        prompt = f"""你只负责对用户请求做意图分类，不生成正文。

请从 allowed_intents 中选择一个 intent，并只输出 JSON：
{{
  "intent": "knowledge_qa|doc_drafting|doc_formatting|form_template_export|spreadsheet_transform|identity_help|clarify",
  "confidence": 0.0,
  "reason": "一句话理由",
  "document_type": "",
  "template_key": "travel|meeting|labor_expert|other|"
}}

边界：
- 普通咨询默认 knowledge_qa。
- 只有明确要求写通知/请示/报告/函/议案/审议材料等，才是 doc_drafting。
- 只有明确要求改为/套用公文格式且已有材料，才是 doc_formatting。
- 报销表单导出必须能判断具体 template_key，否则 clarify。
- 表格筛选、排序、导出等上传表格处理才是 spreadsheet_transform。

输入：
{json.dumps(compact_payload, ensure_ascii=False)}
"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.deepseek_api_key, base_url="https://api.deepseek.com/v1")
            response = client.chat.completions.create(
                model=os.getenv("INTENT_CLASSIFIER_MODEL", "deepseek-v4-flash"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                stream=False,
            )
            content = response.choices[0].message.content or "{}"
            match = re.search(r"\{.*\}", content, re.S)
            return json.loads(match.group(0) if match else content)
        except Exception as exc:
            logger.warning("LLM intent classifier failed, falling back to rules: %s", exc)
            return None

    def writer_factory(self):
        from agents.writer_agent import WriterAgent
        return WriterAgent()

    def chat_container(self) -> ChatServiceContainer:
        if self.chat_container_instance is None:
            self.chat_container_instance = ChatServiceContainer(ChatContainerDependencies(
                memory=self.memory,
                upload_manager=self.upload_manager,
                knowledge_agent=self.knowledge_agent,
                deepseek_api_key=self.deepseek_api_key,
                spreadsheet_db_path=SPREADSHEET_DB_PATH,
                reimbursement_template_files=self.reimbursement_template_files,
                reimbursement_detector=detect_reimbursement_template,
                intent_classifier=self.llm_intent_classifier if self.enable_llm_intent_classifier else None,
                orchestrator_factory=self.request_orchestrator,
                writer_factory=self.writer_factory,
                resolve_export_template=resolve_export_template,
                record_token_usage=self.record_token_usage,
                record_agent_run_token_usage=self.record_agent_run_token_usage,
            ))
        return self.chat_container_instance

    def chat_runtime(self):
        return self.chat_container().chat_runtime()

    def agent_generate_service(self) -> AgentGenerateService:
        if self.agent_generate_service_instance is None:
            self.agent_generate_service_instance = AgentGenerateService(AgentGenerateDependencies(
                memory=self.memory,
                orchestrator_factory=self.request_orchestrator,
            ))
        return self.agent_generate_service_instance

    def export_service(self) -> ExportService:
        if self.export_service_instance is None:
            self.export_service_instance = ExportService(ExportServiceDependencies(
                create_docx=self.create_docx,
                detect_export_template=detect_export_template,
                workbook_from_text=workbook_from_text,
                workbook_to_bytes=workbook_to_bytes,
                safe_filename_stem=safe_filename_stem,
                detect_reimbursement_template=detect_reimbursement_template,
                resolve_reimbursement_template=resolve_reimbursement_template,
                reimbursement_template_path=self.reimbursement_template_path,
                reimbursement_template_files=self.reimbursement_template_files,
            ))
        return self.export_service_instance

    def session_service(self) -> SessionService:
        if self.session_service_instance is None:
            self.session_service_instance = SessionService(SessionServiceDependencies(
                memory=self.memory,
            ))
        return self.session_service_instance

    def upload_service(self) -> UploadService:
        if self.upload_service_instance is None:
            self.upload_service_instance = UploadService(UploadServiceDependencies(
                upload_manager=self.upload_manager,
                memory=self.memory,
                knowledge_base=self.kb_instance,
                knowledge_agent=self.knowledge_agent,
                knowledge_manifest=self.knowledge_manifest,
                knowledge_source_dir=KNOWLEDGE_SOURCE_DIR,
                spreadsheet_db_path=SPREADSHEET_DB_PATH,
                department_dirs=DEPARTMENT_DIRS,
                record_knowledge_upload_audit=self.record_knowledge_upload_audit,
            ))
        return self.upload_service_instance

    def spreadsheet_transform_service(self) -> SpreadsheetTransformService:
        if self.spreadsheet_transform_service_instance is None:
            self.spreadsheet_transform_service_instance = SpreadsheetTransformService(SpreadsheetTransformDependencies(
                upload_manager=self.upload_manager,
                is_spreadsheet_file=is_spreadsheet_file,
                transform_spreadsheet_file=transform_spreadsheet_file,
                deepseek_api_key=self.deepseek_api_key,
                llm_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            ))
        return self.spreadsheet_transform_service_instance

    def runtime_query_service(self) -> RuntimeQueryService:
        if self.runtime_query_service_instance is None:
            self.runtime_query_service_instance = RuntimeQueryService(RuntimeQueryDependencies(
                knowledge_agent=self.knowledge_agent,
                knowledge_base=self.kb_instance,
                knowledge_manifest=self.knowledge_manifest,
                spreadsheet_store=self.spreadsheet_store,
                spreadsheet_db_path=SPREADSHEET_DB_PATH,
                storage_health=storage_health,
                build_access_filter=build_access_filter,
                build_vector_map=build_vector_map,
                now_factory=datetime.now,
            ))
        return self.runtime_query_service_instance

    def knowledge_admin_read_service(self) -> KnowledgeAdminReadService:
        if self.knowledge_admin_read_service_instance is None:
            self.knowledge_admin_read_service_instance = KnowledgeAdminReadService(KnowledgeAdminReadDependencies(
                knowledge_base=self.kb_instance,
                knowledge_manifest=self.knowledge_manifest,
                spreadsheet_store=self.spreadsheet_store,
                workbook_from_structured_rows=workbook_from_structured_rows,
                workbook_to_bytes=workbook_to_bytes,
                safe_filename_stem=safe_filename_stem,
            ))
        return self.knowledge_admin_read_service_instance

    def knowledge_admin_audit_service(self) -> KnowledgeAdminAuditService:
        if self.knowledge_admin_audit_service_instance is None:
            self.knowledge_admin_audit_service_instance = KnowledgeAdminAuditService(KnowledgeAdminAuditDependencies(
                knowledge_agent=self.knowledge_agent,
                knowledge_base=self.kb_instance,
                knowledge_manifest=self.knowledge_manifest,
                spreadsheet_store=self.spreadsheet_store,
                admin_backup_dir=ADMIN_BACKUP_DIR,
                actor_provider=self.knowledge_actor,
                now_factory=datetime.now,
            ))
        return self.knowledge_admin_audit_service_instance

    def knowledge_admin_write_service(self) -> KnowledgeAdminWriteService:
        if self.knowledge_admin_write_service_instance is None:
            self.knowledge_admin_write_service_instance = KnowledgeAdminWriteService(KnowledgeAdminWriteDependencies(
                knowledge_base=self.kb_instance,
                spreadsheet_store=self.spreadsheet_store,
                knowledge_manifest=self.knowledge_manifest,
                upload_manager=self.upload_manager,
                admin_lock=self.knowledge_admin_lock,
                access_levels=ACCESS_LEVELS,
                department_dirs=DEPARTMENT_DIRS,
                parse_spreadsheet=parse_spreadsheet,
                now_factory=datetime.now,
                audit_service=self.knowledge_admin_audit_service(),
            ))
        return self.knowledge_admin_write_service_instance

    def job_service(self) -> JobService:
        if self.job_service_instance is None:
            self.job_service_instance = JobService(
                getattr(self.memory, "db_path", self.project_root / "data" / "agent_memory.db"),
                now_factory=datetime.now,
            )
        return self.job_service_instance

    def send_export_result(self, result):
        if not result.success:
            return jsonify({"error": result.error}), result.status
        file_obj = result.path if result.path else BytesIO(result.payload or b"")
        return send_file(
            file_obj,
            mimetype=result.mimetype,
            as_attachment=True,
            download_name=result.filename,
        )

    def profile_payload(self, profile) -> dict:
        if not profile:
            return {}
        return {
            "preferred_font": profile.preferred_font,
            "preferred_size": profile.preferred_size,
            "writing_style": profile.writing_style,
            "common_doc_types": profile.common_doc_types,
            "name": profile.name,
            "department": profile.department,
        }

    def request_orchestrator(self, session_id: str, *, profile=None, user_info=None) -> AgentOrchestrator:
        """Create an isolated orchestrator for one request to avoid cross-user state bleed."""
        runner = AgentOrchestrator(memory=self.memory, session_id=session_id)
        payload = self.profile_payload(profile)
        if payload:
            runner.set_user_profile(payload)
        if user_info:
            runner.set_user_info(user_info)
        return runner
