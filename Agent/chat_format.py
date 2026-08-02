"""Streaming document formatting pipeline for uploaded-material conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from chat_architecture import INTENT_DOC_FORMATTING
from chat_events import route_actions, route_event, route_intent, route_payload, source_details_from_results, sse
from spreadsheet_auditor import SpreadsheetFactAuditor

logger = logging.getLogger(__name__)


@dataclass
class DocumentFormatDependencies:
    memory: Any
    knowledge_agent: Any
    spreadsheet_db_path: Path
    writer_factory: Callable[[], Any]
    resolve_export_template: Callable[[str, dict, str], str]
    record_token_usage: Callable[..., None]


class DocumentFormatStreamService:
    def __init__(self, deps: DocumentFormatDependencies):
        self.deps = deps

    def stream(
        self,
        message,
        session_id,
        user_id,
        user_info=None,
        display_message=None,
        user_metadata=None,
        route=None,
    ):
        """快速模式 - 格式转换/排版处理，保留原文内容仅改格式."""
        stored_user_message = display_message or message
        self.deps.memory.add_message(session_id, "user", stored_user_message, metadata=user_metadata or {})

        yield sse({"type": "start"})
        yield sse({"type": "session", "session_id": session_id})
        if route:
            yield route_event(route)
        yield sse({"type": "thinking_start", "message": "开始分析待转换材料"})

        try:
            logger.info("快速模式生成，用户消息: %s...", message[:50])
            has_file_content = "[文件内容]" in message

            knowledge_context = ""
            knowledge_sources = []
            yield sse({"type": "think", "agent": "KnowledgeAgent", "emoji": "📚", "message": "正在检索知识库..."})
            try:
                kb_result = self.deps.knowledge_agent.process({
                    "user_request": message,
                    "knowledge_queries": [message],
                    "user_info": user_info.to_dict() if user_info else None,
                })
                if kb_result.success and kb_result.content:
                    knowledge_context = kb_result.content
                    knowledge_sources = kb_result.metadata.get("results", [])
                    logger.info("知识库检索完成，找到上下文")
                    yield sse({
                        "type": "think",
                        "agent": "KnowledgeAgent",
                        "emoji": "✅",
                        "message": f"知识库检索完成，命中 {len(knowledge_sources)} 条参考",
                    })
                else:
                    yield sse({"type": "think", "agent": "KnowledgeAgent", "emoji": "ℹ️", "message": "未命中高相关资料，将直接生成"})
            except Exception as exc:
                logger.warning("知识库检索失败，继续生成: %s", exc)
                yield sse({"type": "think", "agent": "KnowledgeAgent", "emoji": "⚠️", "message": "知识库暂不可用，将继续生成"})

            writer = self.deps.writer_factory()
            plan = {
                "document_type": "格式转换" if has_file_content else "通用公文",
                "task_type": "格式转换" if has_file_content else "公文生成",
                "need_web_search": False,
            }
            input_data = {
                "user_request": message,
                "search_context": "",
                "knowledge_context": knowledge_context,
                "document_type": plan["document_type"],
                "task_type": plan["task_type"],
                "is_format_conversion": has_file_content,
                "knowledge_sources": knowledge_sources,
            }

            stream = writer.process_stream(input_data)
            yield sse({"type": "plan_start", "message": "正在整理生成任务..."})
            yield sse({"type": "plan", "data": plan})

            think_msg = "正在将文档转换为标准公文格式..." if has_file_content else "正在处理公文格式和排版..."
            yield sse({"type": "write_start", "message": think_msg})
            yield sse({"type": "think", "agent": "Writer", "emoji": "📝", "message": think_msg})
            yield sse({"type": "thinking_done", "summary": "检索和格式规划完成，开始输出正文"})
            yield sse({"type": "answer_start", "message": "开始输出正文", "session_id": session_id})

            doc_content = ""
            chunk_count = 0
            for chunk in stream:
                doc_content += chunk
                chunk_count += 1
                yield sse({"type": "answer_delta", "data": chunk, "session_id": session_id})
                yield sse({"type": "content", "data": chunk})
            yield sse({"type": "answer_done", "answer": doc_content, "session_id": session_id})

            logger.info("快速模式生成完成，共 %s 个片段", chunk_count)
            audit_summary = self._audit_spreadsheet_facts(doc_content, knowledge_sources)
            if audit_summary.get("spreadsheet_evidence_count", 0):
                if audit_summary.get("passed"):
                    yield sse({"type": "think", "agent": "Reviewer", "emoji": "✅", "message": "报表数值精确校验通过"})
                else:
                    issue_preview = "；".join(audit_summary.get("issues", [])[:1])
                    yield sse({"type": "think", "agent": "Reviewer", "emoji": "⚠️", "message": issue_preview or "报表数值精确校验未通过"})

            self.deps.memory.add_message(
                session_id,
                "assistant",
                doc_content,
                metadata={"plan": plan, "type": "document", "audit_summary": audit_summary},
            )
            self._record_writer_usage(writer, user_id, user_info, session_id, has_file_content)

            source_filenames = list(dict.fromkeys(
                s.get("filename") or Path(s.get("source", "")).name
                for s in knowledge_sources
                if s.get("filename") or s.get("source")
            ))[:8]
            source_details = source_details_from_results(knowledge_sources)
            self.deps.memory.set_context(session_id, "last_request", stored_user_message)
            self.deps.memory.set_context(session_id, "last_document", doc_content)
            self.deps.memory.set_context(session_id, "last_plan", plan)
            self.deps.memory.update_rolling_summary(session_id, stored_user_message, doc_content, plan, source_filenames)

            yield sse({"type": "run_done", "session_id": session_id, "intent": route_intent(route, INTENT_DOC_FORMATTING)})
            yield sse({
                "type": "done",
                "intent": route_intent(route, INTENT_DOC_FORMATTING),
                "answer": doc_content,
                "document": doc_content,
                "session_id": session_id,
                "plan": plan,
                "route": route_payload(route),
                "actions": route_actions(route),
                "export_template": self.deps.resolve_export_template(doc_content, plan, stored_user_message),
                "export_spreadsheet_template": "",
                "source_filenames": source_filenames,
                "source_details": source_details,
                "audit_summary": audit_summary,
            })
        except Exception as exc:
            logger.exception("快速模式生成过程中发生错误: %s", exc)
            self.deps.record_token_usage(
                user_id=user_id,
                user_info=user_info,
                session_id=session_id,
                mode="quick",
                agent="Writer",
                model="deepseek-v4-flash",
                prompt_chars=len(message or ""),
                status="failed",
                error_message=str(exc),
            )
            yield sse({"type": "error", "message": "生成失败，请稍后重试"})

    def _audit_spreadsheet_facts(self, document_content: str, evidence_items: list) -> dict:
        try:
            auditor = SpreadsheetFactAuditor(self.deps.spreadsheet_db_path)
            return auditor.audit(document_content, evidence_items)
        except Exception as exc:
            logger.warning("quick 模式报表数值校验失败: %s", exc)
            return {
                "passed": False,
                "issues": [f"报表数值校验过程失败：{str(exc)[:120]}"],
                "verified_claims": [],
                "unverified_claims": [],
                "spreadsheet_evidence_count": 0,
            }

    def _record_writer_usage(self, writer, user_id, user_info, session_id, has_file_content: bool) -> None:
        usage = getattr(writer, "last_usage", {}) or {}
        if not usage:
            return
        self.deps.record_token_usage(
            user_id=user_id,
            user_info=user_info,
            session_id=session_id,
            mode="quick" if not has_file_content else "document",
            agent=usage.get("agent", "Writer"),
            model=usage.get("model", ""),
            stream=bool(usage.get("stream")),
            prompt_chars=usage.get("prompt_chars", 0),
            completion_chars=usage.get("completion_chars", 0),
            reasoning_chars=usage.get("reasoning_chars", 0),
            estimated_prompt_tokens=usage.get("estimated_prompt_tokens"),
            estimated_completion_tokens=usage.get("estimated_completion_tokens"),
            estimated_total_tokens=usage.get("estimated_total_tokens"),
            duration_ms=usage.get("duration_ms", 0),
            max_tokens=usage.get("max_tokens", 0),
            temperature=usage.get("temperature", 0),
            status="success",
        )
