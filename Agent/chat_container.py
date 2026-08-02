"""Service container for the production chat endpoint dependencies.

The Flask app owns HTTP concerns; this module owns lazy construction of the
ChatGraphRuntime and the stream services behind each routed intent. New chat
capabilities should be wired here instead of adding alternate /api/chat paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from chat_draft import DocumentDraftDependencies, DocumentDraftStreamService
from chat_format import DocumentFormatDependencies, DocumentFormatStreamService
from chat_lightweight import LightweightChatDependencies, LightweightChatStreamService
from chat_rag import RagQaDependencies, RagQaStreamService
from chat_runtime import ChatGraphRuntime, ChatRuntimeDependencies
from task_planner import TaskPlanner
from tool_runtime import ChatTool, ToolOrchestrator, ToolRegistry


@dataclass
class ChatContainerDependencies:
    memory: Any
    upload_manager: Any
    knowledge_agent: Any
    deepseek_api_key: str
    spreadsheet_db_path: Path
    reimbursement_template_files: dict[str, str]
    reimbursement_detector: Callable[[str, str], str]
    orchestrator_factory: Callable[..., Any]
    writer_factory: Callable[[], Any]
    resolve_export_template: Callable[[str, dict, str], str]
    record_token_usage: Callable[..., None]
    record_agent_run_token_usage: Callable[..., None]
    intent_classifier: Optional[Callable[[dict[str, Any]], Any]] = None


class ChatServiceContainer:
    """Lazy service wiring for the chat runtime.

    Runtime construction is cached, but invalidated when CHAT_RUNTIME changes so
    tests and operations can switch between LangGraph and legacy fallback.
    """

    def __init__(self, deps: ChatContainerDependencies):
        self.deps = deps
        self._runtime = None
        self._runtime_signature = None
        self._rag_qa_service = None
        self._document_format_service = None
        self._document_draft_service = None
        self._lightweight_chat_service = None
        self._tool_registry = None
        self._tool_orchestrator = None
        self._task_planner = None

    def rag_qa_service(self) -> RagQaStreamService:
        if self._rag_qa_service is None:
            self._rag_qa_service = RagQaStreamService(RagQaDependencies(
                memory=self.deps.memory,
                knowledge_agent=self.deps.knowledge_agent,
                deepseek_api_key=self.deps.deepseek_api_key,
                record_token_usage=self.deps.record_token_usage,
            ))
        return self._rag_qa_service

    def document_format_service(self) -> DocumentFormatStreamService:
        if self._document_format_service is None:
            self._document_format_service = DocumentFormatStreamService(DocumentFormatDependencies(
                memory=self.deps.memory,
                knowledge_agent=self.deps.knowledge_agent,
                spreadsheet_db_path=self.deps.spreadsheet_db_path,
                writer_factory=self.deps.writer_factory,
                resolve_export_template=self.deps.resolve_export_template,
                record_token_usage=self.deps.record_token_usage,
            ))
        return self._document_format_service

    def document_draft_service(self) -> DocumentDraftStreamService:
        if self._document_draft_service is None:
            self._document_draft_service = DocumentDraftStreamService(DocumentDraftDependencies(
                memory=self.deps.memory,
                orchestrator_factory=self.deps.orchestrator_factory,
                resolve_export_template=self.deps.resolve_export_template,
                record_agent_run_token_usage=self.deps.record_agent_run_token_usage,
                record_token_usage=self.deps.record_token_usage,
            ))
        return self._document_draft_service

    def lightweight_chat_service(self) -> LightweightChatStreamService:
        if self._lightweight_chat_service is None:
            self._lightweight_chat_service = LightweightChatStreamService(LightweightChatDependencies(
                memory=self.deps.memory,
                reimbursement_template_files=self.deps.reimbursement_template_files,
            ))
        return self._lightweight_chat_service

    def task_planner(self) -> TaskPlanner:
        if self._task_planner is None:
            self._task_planner = TaskPlanner(
                reimbursement_detector=self.deps.reimbursement_detector,
                planner_classifier=self.deps.intent_classifier,
                intent_classifier=None,
            )
        return self._task_planner

    def tool_registry(self) -> ToolRegistry:
        if self._tool_registry is None:
            registry = ToolRegistry()
            registry.register(ChatTool(
                name="knowledge_qa",
                description="基于知识库检索和证据校验回答问题",
                risk_level="low",
                input_schema={"message": "string"},
                stream=self.rag_qa_service().stream,
            ))
            registry.register(ChatTool(
                name="draft_document",
                description="完整公文或材料写作，多 agent 规划、检索、写作、审核",
                risk_level="low",
                input_schema={"message": "string"},
                stream=self.document_draft_service().stream,
            ))
            registry.register(ChatTool(
                name="format_document",
                description="将上传材料整理或转换为标准公文格式",
                risk_level="low",
                input_schema={"message": "string", "attachments": "array"},
                stream=self.document_format_service().stream,
            ))
            registry.register(ChatTool(
                name="prepare_form_export",
                description="准备报销表模板导出动作，等待用户确认",
                risk_level="confirm",
                input_schema={"template_key": "string"},
                stream=self.lightweight_chat_service().stream,
            ))
            registry.register(ChatTool(
                name="prepare_spreadsheet_transform",
                description="准备表格处理和导出动作，等待用户确认",
                risk_level="confirm",
                input_schema={"file_id": "string", "instruction": "string"},
                stream=self.lightweight_chat_service().stream,
            ))
            registry.register(ChatTool(
                name="clarify",
                description="在信息不足时生成澄清问题",
                risk_level="low",
                input_schema={"message": "string"},
                stream=self.lightweight_chat_service().stream,
            ))
            registry.register(ChatTool(
                name="identity_help",
                description="回答助手身份和能力边界",
                risk_level="low",
                input_schema={"message": "string"},
                stream=self.lightweight_chat_service().stream,
            ))
            self._tool_registry = registry
        return self._tool_registry

    def tool_orchestrator(self) -> ToolOrchestrator:
        if self._tool_orchestrator is None:
            self._tool_orchestrator = ToolOrchestrator(self.tool_registry())
        return self._tool_orchestrator

    def chat_runtime(self) -> ChatGraphRuntime:
        signature = (
            os.getenv("CHAT_RUNTIME", "langgraph"),
            bool(self.deps.intent_classifier),
        )
        if self._runtime is None or self._runtime_signature != signature:
            self._runtime = ChatGraphRuntime(ChatRuntimeDependencies(
                memory=self.deps.memory,
                upload_manager=self.deps.upload_manager,
                reimbursement_detector=self.deps.reimbursement_detector,
                intent_classifier=self.deps.intent_classifier,
                lightweight_stream=self.lightweight_chat_service().stream,
                document_format_stream=self.document_format_service().stream,
                document_draft_stream=self.document_draft_service().stream,
                rag_qa_stream=self.rag_qa_service().stream,
                task_planner=self.task_planner(),
                tool_orchestrator=self.tool_orchestrator(),
            ))
            self._runtime_signature = signature
        return self._runtime
