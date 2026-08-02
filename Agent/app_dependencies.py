"""Application dependency assembly.

Keep this module focused on startup wiring. Runtime helpers and service factories
live on AppContext in app_context.py so this file does not become a new app.py.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from account_admin_service import AccountAdminDependencies, AccountAdminService
from agents.knowledge_agent import KnowledgeAgent
from app_config import DEPARTMENT_DIRS
from app_context import AppContext
from auth import get_auth_manager
from auth_route_service import AuthRouteDependencies, AuthRouteService
from beta_ops_service import BetaOpsDependencies, BetaOpsService
from knowledge_base import KnowledgeBase
from knowledge_manifest import KnowledgeIngestionManifest
from memory_v2 import get_memory
from spreadsheet_store import SpreadsheetStore
from storage_config import (
    INGESTION_MANIFEST_DB,
    KNOWLEDGE_BASE_DIR,
    KNOWLEDGE_SOURCE_DIR,
    SPREADSHEET_DB_PATH,
    ensure_storage_dirs,
    storage_summary,
)
from upload_manager import get_upload_manager


logger = logging.getLogger(__name__)


def create_app_context() -> AppContext:
    bocha_api_key = os.getenv("BOCHA_API_KEY")
    if bocha_api_key:
        print(f"✓ 博查 API 已配置: {bocha_api_key[:10]}...")
    else:
        print("⚠️ 博查 API 未配置，联网搜索功能将不可用")

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    enable_llm_intent_classifier = os.getenv("ENABLE_LLM_INTENT_CLASSIFIER", "").lower() in {"1", "true", "yes", "on"}
    if deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
    else:
        logger.warning("DEEPSEEK_API_KEY 未配置，模型调用接口将不可用")

    print("=" * 60)
    print("初始化多 Agent 公文写作系统...")
    print("=" * 60)
    ensure_storage_dirs()
    print(f"✓ 存储后端: {storage_summary()['backend']} ({storage_summary()['root']})")

    auth_manager = get_auth_manager()
    print("✓ 认证系统已初始化")

    memory = get_memory("./data/agent_memory.db", db_type="sqlite")
    print("✓ 团队记忆系统已初始化（SQLite）")

    upload_manager = get_upload_manager()
    print("✓ 文件上传系统已初始化")

    knowledge_agent = KnowledgeAgent()
    kb_instance = KnowledgeBase(str(KNOWLEDGE_BASE_DIR), lazy_load=True)
    knowledge_manifest = KnowledgeIngestionManifest(INGESTION_MANIFEST_DB)
    spreadsheet_store = SpreadsheetStore(SPREADSHEET_DB_PATH)

    project_root = Path(__file__).resolve().parent
    source_materials_dir = project_root / "资料原件"
    review_template_override = os.getenv("REVIEW_PROPOSAL_TEMPLATE_PATH", "").strip()
    review_proposal_template_path = (
        Path(review_template_override).expanduser()
        if review_template_override
        else source_materials_dir
        / "通用模板"
        / "公文与通知"
        / "【模板】【第X次审议会议】议案X：关于审议XXX的有关事宜.docx"
    )
    reimbursement_template_dir = source_materials_dir / "通用模板" / "审批流程"
    reimbursement_template_files = {
        "travel": "差旅费.xlsx",
        "meeting": "会议费.xlsx",
        "labor_expert": "劳务费&专家咨询费.xlsx",
        "other": "其他费用报销.xlsx",
    }

    beta_ops_service = BetaOpsService(BetaOpsDependencies(memory=memory, now_factory=datetime.now))
    beta_ops_service.init_tables()

    context = AppContext(
        deepseek_api_key=deepseek_api_key,
        enable_llm_intent_classifier=enable_llm_intent_classifier,
        auth_manager=auth_manager,
        memory=memory,
        upload_manager=upload_manager,
        knowledge_agent=knowledge_agent,
        kb_instance=kb_instance,
        knowledge_manifest=knowledge_manifest,
        spreadsheet_store=spreadsheet_store,
        beta_ops_service=beta_ops_service,
        auth_route_service=AuthRouteService(AuthRouteDependencies(
            auth_manager=auth_manager,
            now_factory=datetime.now,
        )),
        account_admin_service=AccountAdminService(AccountAdminDependencies(
            auth_manager=auth_manager,
            memory=memory,
            knowledge_source_dir=KNOWLEDGE_SOURCE_DIR,
            department_dirs=DEPARTMENT_DIRS,
        )),
        project_root=project_root,
        source_materials_dir=source_materials_dir,
        review_proposal_template_path=review_proposal_template_path,
        reimbursement_template_dir=reimbursement_template_dir,
        reimbursement_template_files=reimbursement_template_files,
        cookie_secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
    )
    context.job_service()
    print("系统初始化完成！")
    return context
