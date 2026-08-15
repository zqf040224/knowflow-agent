import json
import hashlib
import logging
import sys
import os
import time
from math import isfinite
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Callable, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding_config import UserInfo

from agents.base_agent import AgentResult
from agents.context_agent import ContextAgent
from agents.planner_agent import PlannerAgent
from agents.search_agent import SearchAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.writer_agent import WriterAgent
from agents.reviewer_agent import ReviewerAgent
from agents.reflection_agent import ReflectionAgent
from agents.document_linear_runner import DocumentLinearRunner
from agents.document_stream_runner import DocumentStreamRunner
from agents.document_graph_runner import (
    DocumentGraphRunner,
    LANGGRAPH_AVAILABLE,
    LANGGRAPH_IMPORT_ERROR,
)

logger = logging.getLogger(__name__)


@dataclass
class ContextPacket:
    """在 Agent 之间传递的累积上下文"""
    user_request: str
    context_analysis: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    search_context: str = ""
    knowledge_context: str = ""
    knowledge_sources: list = field(default_factory=list)
    search_sources: list = field(default_factory=list)
    evidence_items: list = field(default_factory=list)
    compact_evidence: list = field(default_factory=list)
    revision_history: list = field(default_factory=list)
    run_records: list = field(default_factory=list)
    last_document: str = ""
    last_plan: dict = field(default_factory=dict)
    user_constraints: list = field(default_factory=list)
    unresolved_questions: list = field(default_factory=list)
    user_profile: Optional[dict] = None
    memory_context: str = ""
    audit_summary: dict = field(default_factory=dict)


@dataclass
class PreparedDocumentRun:
    user_request: str
    request_with_context: str
    previous_context: str
    session_id: Optional[str]
    run_id: str
    persisted_user_message: str = ""
    effect_scope: str = ""


class AgentOrchestrator:
    MAX_TOTAL_ROUNDS = 3
    MAX_REVISION_ROUNDS = MAX_TOTAL_ROUNDS - 1

    def __init__(
        self,
        memory=None,
        session_id: Optional[str] = None,
        *,
        graph_checkpointer=None,
        graph_durability: str = "async",
    ):
        self.context_agent = ContextAgent()
        self.planner = PlannerAgent()
        self.search_agent = SearchAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()
        self.reflection = ReflectionAgent(model="deepseek-reasoner")
        self.think_log: list[dict] = []
        self.memory = memory
        self.session_id = session_id
        self.user_profile: Optional[dict] = None
        self.user_info: Optional[UserInfo] = None
        self._current_agent_memory_context = ""
        self._current_effect_run_id = ""
        self._current_effect_scope = ""
        self._current_persisted_user_message: Optional[str] = None
        self._document_runtime_snapshot: dict = {}
        self._document_runtime = self.validate_runtime_configuration()
        self._prefer_langgraph = self._document_runtime == "graph"
        self._document_graph_runner = None
        self._graph_checkpointer = graph_checkpointer
        self._graph_durability = graph_durability

        if memory and session_id:
            self._setup_agents_session()

    @classmethod
    def resolve_document_runtime(cls, explicit: Optional[str] = None) -> str:
        value = (
            explicit
            if explicit is not None
            else os.getenv("AGENT_ORCHESTRATOR", "graph")
        )
        normalized = str(value or "graph").strip().lower()
        if normalized in {"langgraph", "graph", "1", "true", "yes", "on"}:
            return "graph"
        if normalized in {"linear", "legacy", "0", "false", "no", "off"}:
            return "linear"
        raise ValueError(
            f"Unsupported AGENT_ORCHESTRATOR value {value!r}; "
            "expected graph/langgraph/on or linear/legacy/off"
        )

    @classmethod
    def validate_runtime_configuration(cls) -> str:
        runtime = cls.resolve_document_runtime()
        if runtime == "graph" and not LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "AGENT_ORCHESTRATOR=graph requires LangGraph, but it could not "
                f"be imported: {LANGGRAPH_IMPORT_ERROR or 'unknown import error'}"
            )
        return runtime

    @classmethod
    def _langgraph_requested(cls) -> bool:
        """Compatibility predicate retained for one rollback release."""
        return cls.resolve_document_runtime() == "graph"

    def _should_use_langgraph(self) -> bool:
        return self._prefer_langgraph

    def _setup_agents_session(self):
        for agent in [self.context_agent, self.planner, self.search_agent,
                      self.knowledge_agent, self.writer, self.reviewer,
                      self.reflection]:
            agent.set_session(self.session_id, self.memory, self.user_profile)

    def set_session(self, session_id: str, memory=None):
        self.session_id = session_id
        if memory:
            self.memory = memory
        self._setup_agents_session()

    def set_user_profile(self, user_profile: dict):
        self.user_profile = user_profile
        for agent in [self.context_agent, self.planner, self.search_agent,
                      self.knowledge_agent, self.writer, self.reviewer,
                      self.reflection]:
            agent.user_profile = user_profile

    def set_user_info(self, user_info: UserInfo):
        """设置用户信息，用于知识库权限过滤"""
        self.user_info = user_info

    def _on_think(self, agent_name: str, emoji: str, message: str):
        entry = {"agent": agent_name, "emoji": emoji, "message": message}
        self.think_log.append(entry)

    def _think_handler(self, on_think: Optional[Callable] = None):
        def handler(agent_name, emoji, message):
            self._on_think(agent_name, emoji, message)
            if on_think:
                on_think(agent_name, emoji, message)

        return handler

    def _prepare_document_run(
        self,
        user_request: str,
        session_id: Optional[str] = None,
        *,
        run_id: str = "",
        persisted_user_message: Optional[str] = None,
        effect_scope: str = "",
    ) -> PreparedDocumentRun:
        self.think_log = []
        provided_run_id = str(run_id or "").strip()
        normalized_effect_scope = str(effect_scope or "").strip()
        safe_user_message = str(
            user_request
            if persisted_user_message is None
            else persisted_user_message
        )
        self._current_effect_run_id = provided_run_id
        self._current_effect_scope = normalized_effect_scope
        self._current_persisted_user_message = safe_user_message

        if session_id and session_id != self.session_id:
            self.set_session(session_id, self.memory)

        if self.memory and self.session_id:
            metadata = (
                {
                    "run_id": provided_run_id,
                    "effect_key": self._document_effect_key(
                        "message_user",
                        run_id=provided_run_id,
                        effect_scope=normalized_effect_scope,
                    ),
                }
                if provided_run_id
                else None
            )
            self.memory.add_message(
                self.session_id,
                "user",
                safe_user_message,
                metadata=metadata,
            )

        self._current_agent_memory_context = self._recall_agent_context(user_request)

        previous_context = ""
        if self.memory and self.session_id:
            snapshot_key = (
                (
                    f"document_prepare:{provided_run_id}:{normalized_effect_scope}"
                    if normalized_effect_scope
                    else f"document_prepare:{provided_run_id}"
                )
                if provided_run_id
                else ""
            )
            request_digest = hashlib.sha256(
                str(user_request or "").encode("utf-8")
            ).hexdigest()
            snapshot = (
                self.memory.get_context(self.session_id, snapshot_key, None)
                if snapshot_key
                else None
            )
            if isinstance(snapshot, dict):
                if snapshot.get("request_digest") != request_digest:
                    raise RuntimeError(
                        "document run_id was reused with a different request"
                    )
                previous_context = str(snapshot.get("previous_context") or "")
            else:
                previous_context = self.memory.get_context(
                    self.session_id,
                    "last_request",
                    "",
                )
                if snapshot_key:
                    # Persist the prior turn before mutating last_request. A
                    # crash anywhere after this write can replay preparation
                    # without treating the current request as its predecessor.
                    self.memory.set_context(
                        self.session_id,
                        snapshot_key,
                        {
                            "request_digest": request_digest,
                            "previous_context": previous_context,
                        },
                    )
            self.memory.set_context(
                self.session_id,
                "last_request",
                safe_user_message,
            )

        request_with_context = user_request
        if previous_context:
            request_with_context = f"之前的需求：{previous_context}\n\n当前需求：{user_request}"

        return PreparedDocumentRun(
            user_request=user_request,
            request_with_context=request_with_context,
            previous_context=previous_context,
            session_id=self.session_id,
            run_id=provided_run_id or str(uuid4()),
            persisted_user_message=safe_user_message,
            effect_scope=normalized_effect_scope,
        )

    def _document_effect_key(
        self,
        effect: str,
        *,
        run_id: str = "",
        effect_scope: str = "",
    ) -> Optional[str]:
        resolved_run_id = str(
            run_id or getattr(self, "_current_effect_run_id", "") or ""
        ).strip()
        if not resolved_run_id:
            return None
        resolved_scope = str(
            effect_scope
            or getattr(self, "_current_effect_scope", "")
            or ""
        ).strip()
        prefix = f"{resolved_run_id}:tool_draft_document"
        if resolved_scope:
            prefix = f"{prefix}:{resolved_scope}"
        return f"{prefix}:{effect}"

    def _recall_agent_context(self, user_request: str) -> str:
        """Build one bounded, user-scoped recall packet for the whole run.

        The packet is created once after the current user message is persisted,
        then passed explicitly through Context/Planner/Writer/Reviewer/Reflection.
        It deliberately does not alter retrieval queries: historical preferences
        should guide writing, not broaden knowledge-base permissions or searches.
        """
        if not self.memory or not self.session_id:
            return ""
        try:
            get_agent_context = getattr(self.memory, "get_agent_context_for_prompt", None)
            if callable(get_agent_context):
                context = get_agent_context(
                    self.session_id,
                    memory_query=user_request,
                    max_messages=6,
                    max_chars=3600,
                )
            else:
                try:
                    context = self.memory.get_context_for_prompt(
                        self.session_id,
                        max_messages=6,
                        memory_query=user_request,
                    )
                except TypeError:
                    # Keep compatibility with lightweight adapters and older
                    # test doubles that have not adopted memory_query yet.
                    context = self.memory.get_context_for_prompt(self.session_id, max_messages=6)
            context = str(context or "").strip()
            return context if len(context) <= 3600 else context[:3597] + "..."
        except Exception as exc:
            logger.warning("Agent 上下文召回失败，继续执行当前请求: %s", exc)
            return ""

    def _build_document_run_result(self, ctx: ContextPacket, document_content: str,
                                   user_request: str, *, runtime: str = "",
                                   quality_status: str = "passed",
                                   revisions_applied: Optional[int] = None,
                                   run_id: str = "",
                                   effect_scope: str = "") -> dict:
        final_confidence = ctx.revision_history[-1]["confidence"] if ctx.revision_history else 0.8
        source_filenames = self._source_filenames(ctx)
        source_details = self._source_details(ctx)
        stored_user_message = getattr(
            self,
            "_current_persisted_user_message",
            None,
        )
        persisted_user_message = str(
            user_request if stored_user_message is None else stored_user_message
        )

        if self.memory and self.session_id:
            message_metadata = {
                "type": "document",
                "plan": ctx.plan,
                "run_records": ctx.run_records,
                "run_id": run_id,
                "quality_status": quality_status,
                "source_filenames": source_filenames,
                "source_details": source_details,
                "context_snapshot": self._context_snapshot(ctx),
            }
            if str(
                run_id or getattr(self, "_current_effect_run_id", "") or ""
            ).strip():
                message_metadata["effect_key"] = AgentOrchestrator._document_effect_key(
                    self,
                    "message_assistant",
                    run_id=run_id,
                    effect_scope=effect_scope,
                )
            self.memory.add_message(self.session_id, "assistant", document_content,
                                   metadata=message_metadata)
            self.memory.set_context(self.session_id, "last_document", document_content)
            self.memory.set_context(self.session_id, "last_plan", ctx.plan)
            if hasattr(self.memory, "update_rolling_summary"):
                self.memory.update_rolling_summary(
                    self.session_id,
                    persisted_user_message,
                    document_content,
                    ctx.plan,
                    source_filenames,
                    effect_key=AgentOrchestrator._document_effect_key(
                        self,
                        "rolling_summary",
                        run_id=run_id,
                        effect_scope=effect_scope,
                    ),
                )

        result = {
            "document": document_content,
            "plan": ctx.plan,
            "think_log": self.think_log,
            "confidence": final_confidence,
            "revision_rounds": (
                revisions_applied
                if revisions_applied is not None
                else len({
                    h.get("round")
                    for h in ctx.revision_history
                    if h.get("needs_revision") and h.get("round") is not None
                })
            ),
            "session_id": self.session_id,
            "run_records": ctx.run_records,
            "source_filenames": source_filenames,
            "source_details": source_details,
            "audit_summary": ctx.audit_summary,
            "quality_status": quality_status,
            "run_id": run_id,
        }
        if runtime:
            result["runtime"] = runtime
        return result

    # ========== 非流式运行 ==========

    def run(self, user_request: str, on_think: Optional[Callable] = None,
            session_id: Optional[str] = None, *, run_id: str = "",
            persisted_user_message: Optional[str] = None,
            effect_scope: str = "") -> dict:
        if self._should_use_langgraph():
            return self._run_langgraph(
                user_request,
                on_think=on_think,
                session_id=session_id,
                run_id=run_id,
                persisted_user_message=persisted_user_message,
                effect_scope=effect_scope,
            )
        prepared_run = self._prepare_document_run(
            user_request,
            session_id=session_id,
            run_id=run_id,
            persisted_user_message=persisted_user_message,
            effect_scope=effect_scope,
        )
        think_handler = self._think_handler(on_think)
        run_result = DocumentLinearRunner(self).run(prepared_run, think_handler=think_handler)
        return self._build_document_run_result(
            run_result.ctx,
            run_result.document_content,
            prepared_run.persisted_user_message,
            quality_status=run_result.quality_status,
            revisions_applied=run_result.revisions_applied,
            run_id=prepared_run.run_id,
            effect_scope=prepared_run.effect_scope,
        )

    # ========== LangGraph 运行 ==========

    def _run_langgraph(self, user_request: str, on_think: Optional[Callable] = None,
                       session_id: Optional[str] = None, *, run_id: str = "",
                       persisted_user_message: Optional[str] = None,
                       effect_scope: str = "") -> dict:
        """Use LangGraph for the non-streaming orchestration path.

        The node bodies intentionally reuse the existing Agent steps so the
        inner prompts, model calls, memory behavior and review metadata remain
        compatible with the current Flask endpoints.
        """
        prepared_run = self._prepare_document_run(
            user_request,
            session_id=session_id,
            run_id=run_id,
            persisted_user_message=persisted_user_message,
            effect_scope=effect_scope,
        )
        think_handler = self._think_handler(on_think)

        graph_result = self._get_document_graph_runner().run(
            prepared_run,
            think_handler=think_handler,
            thread_id=prepared_run.run_id,
        )
        return self._build_document_run_result(
            graph_result.ctx,
            graph_result.document_content,
            prepared_run.persisted_user_message,
            runtime="langgraph",
            quality_status=graph_result.quality_status,
            revisions_applied=graph_result.revisions_applied,
            run_id=graph_result.run_id,
            effect_scope=prepared_run.effect_scope,
        )

    def _get_document_graph_runner(self) -> DocumentGraphRunner:
        if self._document_graph_runner is None:
            self._document_graph_runner = DocumentGraphRunner(
                self,
                checkpointer=self._graph_checkpointer,
                durability=self._graph_durability,
            )
        return self._document_graph_runner

    # ========== 流式运行 ==========

    def run_stream(self, user_request: str, on_think: Optional[Callable] = None,
                   session_id: Optional[str] = None, *, run_id: str = "",
                   persisted_user_message: Optional[str] = None,
                   effect_scope: str = ""):
        prepared_run = self._prepare_document_run(
            user_request,
            session_id=session_id,
            run_id=run_id,
            persisted_user_message=persisted_user_message,
            effect_scope=effect_scope,
        )
        if self._should_use_langgraph():
            think_handler = self._think_handler(on_think)
            yield from self._get_document_graph_runner().stream(
                prepared_run,
                think_handler=think_handler,
                thread_id=prepared_run.run_id,
                user_request=user_request,
            )
            return
        yield from DocumentStreamRunner(self).run(
            prepared_run,
            user_request=user_request,
            on_think=on_think,
        )

    # ========== Pipeline 步骤 ==========

    def _step_context(self, user_request: str, previous_context: str,
                      on_think) -> ContextPacket:
        """阶段1: 上下文分析"""
        try:
            conversation_history = []
            last_document = ""
            last_plan = {}
            if self.memory and self.session_id:
                conversation_history = self.memory.get_session_history(
                    self.session_id, limit=10
                )
                last_document = self.memory.get_context(self.session_id, "last_document", "") or ""
                last_plan = self.memory.get_context(self.session_id, "last_plan", {}) or {}

            context_result = self.context_agent.process({
                "conversation_history": conversation_history,
                "user_request": user_request,
                "user_profile": self.user_profile or {},
                "memory_context": self._current_agent_memory_context,
                "previous_context": {
                    "last_request": previous_context,
                    "last_plan": last_plan,
                    "last_document_excerpt": last_document[:1200] if last_document else "",
                },
            }, on_think=on_think)

            context_analysis = context_result.metadata if context_result.success else {}
            user_constraints = self._extract_user_constraints(context_analysis, self.user_profile)
            unresolved_questions = context_analysis.get("context_quality", {}).get("issues", [])

            if context_result.success:
                on_think("ContextAgent", "✅", "上下文分析完成")
            else:
                logger.warning("ContextAgent 分析失败，使用空上下文继续")

            return ContextPacket(
                user_request=user_request,
                context_analysis=context_analysis,
                user_profile=self.user_profile,
                memory_context=self._current_agent_memory_context,
                last_document=last_document,
                last_plan=last_plan,
                user_constraints=user_constraints,
                unresolved_questions=unresolved_questions,
            )
        except Exception as e:
            logger.warning(f"ContextAgent 异常，降级为空上下文: {e}")
            on_think("ContextAgent", "⚠️", f"上下文分析跳过: {str(e)[:50]}")
            return ContextPacket(
                user_request=user_request,
                context_analysis={"key_points": [], "user_intent": user_request[:100],
                                 "document_type": "通用公文", "confidence": 0.5},
                user_profile=self.user_profile,
                memory_context=self._current_agent_memory_context,
                user_constraints=self._extract_user_constraints({}, self.user_profile),
            )

    def _step_context_plan(self, user_request: str, previous_context: str,
                           on_think) -> ContextPacket:
        """合并 ContextAgent + PlannerAgent，一次模型调用生成上下文和计划。"""
        try:
            conversation_history = []
            last_document = ""
            last_plan = {}
            runtime_snapshot = getattr(self, "_document_runtime_snapshot", {})
            if isinstance(runtime_snapshot, dict) and runtime_snapshot:
                conversation_history = list(
                    runtime_snapshot.get("conversation_history") or []
                )
                last_document = str(
                    runtime_snapshot.get("last_document") or ""
                )
                last_plan = dict(runtime_snapshot.get("last_plan") or {})
            elif self.memory and self.session_id:
                conversation_history = self.memory.get_session_history(
                    self.session_id, limit=10
                )
                last_document = self.memory.get_context(self.session_id, "last_document", "") or ""
                last_plan = self.memory.get_context(self.session_id, "last_plan", {}) or {}

            result = self.planner.process_with_context({
                "conversation_history": conversation_history,
                "user_request": user_request,
                "user_profile": self.user_profile or {},
                "memory_context": self._current_agent_memory_context,
                "previous_context": {
                    "last_request": previous_context,
                    "last_plan": last_plan,
                    "last_document_excerpt": last_document[:1200] if last_document else "",
                },
            }, on_think=on_think)

            metadata = result.metadata if result.success else {}
            context_analysis = metadata.get("context_analysis", {})
            plan = metadata.get("plan", {})
            user_constraints = self._extract_user_constraints(context_analysis, self.user_profile)
            unresolved_questions = context_analysis.get("context_quality", {}).get("issues", [])

            ctx = ContextPacket(
                user_request=user_request,
                context_analysis=context_analysis,
                plan=plan,
                user_profile=self.user_profile,
                memory_context=self._current_agent_memory_context,
                last_document=last_document,
                last_plan=last_plan,
                user_constraints=user_constraints,
                unresolved_questions=unresolved_questions,
            )
            return self._merge_plan_key_points(ctx)
        except Exception as e:
            logger.warning(f"合并上下文规划失败，回退到旧链路: {e}")
            ctx = self._step_context(user_request, previous_context, on_think)
            return self._step_plan(ctx, on_think)

    def _step_plan(self, ctx: ContextPacket, on_think) -> ContextPacket:
        """阶段2: 任务规划"""
        # 用 context_analysis 丰富 Planner 的输入
        enriched_request = ctx.user_request
        key_points = ctx.context_analysis.get("key_points", [])
        user_intent = ctx.context_analysis.get("user_intent", "")

        if key_points or user_intent or ctx.last_plan or ctx.last_document or ctx.user_constraints:
            parts = [ctx.user_request]
            if user_intent:
                parts.append(f"\n\n用户意图：{user_intent}")
            if key_points:
                parts.append(f"\n\n上下文关键信息：{'；'.join(key_points[:5])}")
            if ctx.user_constraints:
                parts.append(f"\n\n用户约束：{'；'.join(ctx.user_constraints[:6])}")
            if ctx.last_plan:
                parts.append(f"\n\n上一轮计划：{json.dumps(ctx.last_plan, ensure_ascii=False)[:800]}")
            if ctx.last_document:
                parts.append(f"\n\n上一版文档摘要：{ctx.last_document[:1000]}")
            enriched_request = "".join(parts)

        plan_result = self.planner.process(
            {
                "user_request": enriched_request,
                "context_analysis": ctx.context_analysis,
                "last_plan": ctx.last_plan,
                "last_document": ctx.last_document,
                "user_constraints": ctx.user_constraints,
                "memory_context": ctx.memory_context,
            },
            on_think=on_think
        )
        ctx.plan = plan_result.metadata

        return self._merge_plan_key_points(ctx)

    def _merge_plan_key_points(self, ctx: ContextPacket) -> ContextPacket:
        plan_key_points = ctx.plan.get("key_points", []) if isinstance(ctx.plan, dict) else []
        context_key_points = ctx.context_analysis.get("key_points", []) if isinstance(ctx.context_analysis, dict) else []
        merged = list(dict.fromkeys(context_key_points + plan_key_points))[:8]
        ctx.plan["key_points"] = merged
        return ctx

    def _step_search(self, ctx: ContextPacket, on_think) -> ContextPacket:
        """阶段3: 联网搜索"""
        if not ctx.plan.get("need_web_search"):
            return ctx

        search_result = self.search_agent.process({
            "user_request": ctx.user_request,
            "search_queries": ctx.plan.get("search_queries", []),
            "user_info": self.user_info.to_dict() if self.user_info else None,
        }, on_think=on_think)

        ctx.search_context = search_result.content if search_result.success else ""
        ctx.search_sources = search_result.metadata.get("search_results", []) if search_result.success else []
        return ctx

    def _step_knowledge(self, ctx: ContextPacket, on_think) -> ContextPacket:
        """阶段4: 知识库检索（利用搜索结果优化检索）"""
        knowledge_result = self.knowledge_agent.process({
            "user_request": ctx.user_request,
            "knowledge_queries": ctx.plan.get("knowledge_queries", [ctx.user_request]),
            "search_context": ctx.search_context,
            "key_points": ctx.plan.get("key_points", []),
            "user_info": self.user_info.to_dict() if self.user_info else None,
        }, on_think=on_think)

        ctx.knowledge_context = knowledge_result.content if knowledge_result.success else ""
        ctx.knowledge_sources = knowledge_result.metadata.get("results", []) if knowledge_result.success else []
        return ctx

    def _step_write(self, ctx: ContextPacket, on_think) -> AgentResult:
        """阶段5a: 生成草稿"""
        input_data = {
            "user_request": ctx.user_request,
            "search_context": ctx.search_context,
            "knowledge_context": ctx.knowledge_context,
            "document_type": ctx.plan.get("document_type", "通用公文"),
            "task_type": ctx.plan.get("task_type", "公文生成"),
            "key_points": self._merged_key_points(ctx),
            "revision_history": ctx.revision_history,
            "knowledge_sources": ctx.knowledge_sources,
            "context_analysis": ctx.context_analysis,
            "memory_context": ctx.memory_context,
            "last_document": ctx.last_document,
            "last_plan": ctx.last_plan,
            "user_constraints": ctx.user_constraints,
            "unresolved_questions": ctx.unresolved_questions,
            "evidence_items": ctx.compact_evidence,
            "revision_mode": bool(ctx.revision_history),
            "draft_document": ctx.last_document if ctx.revision_history else "",
        }
        return self.writer.process(input_data, on_think=on_think)

    def _step_write_stream(self, ctx: ContextPacket, revision_round: int, on_think) -> str:
        """Collect one complete streamed draft before exposing it to review."""
        chunks = self.writer.process_stream({
            "user_request": ctx.user_request,
            "search_context": self._writer_search_context(ctx, revision_round),
            "knowledge_context": self._writer_knowledge_context(ctx, revision_round),
            "document_type": ctx.plan.get("document_type", "通用公文"),
            "task_type": ctx.plan.get("task_type", "公文生成"),
            "key_points": self._merged_key_points(ctx),
            "revision_history": ctx.revision_history,
            "knowledge_sources": ctx.knowledge_sources,
            "context_analysis": ctx.context_analysis,
            "memory_context": ctx.memory_context,
            "last_document": ctx.last_document,
            "draft_document": "" if revision_round == 0 else ctx.last_document,
            "last_plan": ctx.last_plan,
            "user_constraints": ctx.user_constraints,
            "unresolved_questions": ctx.unresolved_questions,
            "evidence_items": ctx.compact_evidence,
            "revision_mode": revision_round > 0,
        }, on_think=on_think)
        return "".join(chunks)

    @staticmethod
    def _sanitize_document_output(text: str, user_request: str) -> str:
        return DocumentStreamRunner._sanitize_unsupported_specifics(text, user_request)

    def _step_review(self, ctx: ContextPacket, document_content: str,
                     on_think) -> AgentResult:
        """阶段5b: 审查"""
        failures = []
        model_responses = []
        original_call = self.reviewer.call_llm
        original_parse = self.reviewer._parse_json_response
        had_call_override = "call_llm" in self.reviewer.__dict__
        had_parse_override = "_parse_json_response" in self.reviewer.__dict__
        previous_call_override = self.reviewer.__dict__.get("call_llm")
        previous_parse_override = self.reviewer.__dict__.get("_parse_json_response")

        def tracked_call(*args, **kwargs):
            try:
                response = original_call(*args, **kwargs)
                model_responses.append(response)
                return response
            except Exception as exc:
                failures.append(exc)
                raise

        def tracked_parse(*args, **kwargs):
            try:
                return original_parse(*args, **kwargs)
            except Exception as exc:
                failures.append(exc)
                raise

        self.reviewer.call_llm = tracked_call
        self.reviewer._parse_json_response = tracked_parse
        try:
            result = self.reviewer.process({
                "user_request": ctx.user_request,
                "document_content": document_content,
                "document_type": ctx.plan.get("document_type", "通用公文"),
                "task_type": ctx.plan.get("task_type", "公文生成"),
                "key_points": ctx.plan.get("key_points", []),
                "source_filenames": self._source_filenames(ctx),
                "context_analysis": ctx.context_analysis,
                "memory_context": ctx.memory_context,
                "user_constraints": ctx.user_constraints,
                "evidence_items": ctx.evidence_items,
            }, on_think=on_think)
        finally:
            if had_call_override:
                self.reviewer.call_llm = previous_call_override
            else:
                self.reviewer.__dict__.pop("call_llm", None)
            if had_parse_override:
                self.reviewer._parse_json_response = previous_parse_override
            else:
                self.reviewer.__dict__.pop("_parse_json_response", None)

        raw_response_error = None
        for response in model_responses:
            try:
                self._parse_reviewer_model_response_strict(response)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raw_response_error = exc
                break

        if failures or raw_response_error is not None:
            failure = failures[-1] if failures else raw_response_error
            error = str(failure)[:500] or "Reviewer model call failed"
            logger.error("Reviewer 不可用，文档流程按 fail-closed 终止: %s", error)
            return AgentResult(
                success=False,
                content="",
                agent_name=self.reviewer.name,
                confidence=0.0,
                metadata={"review_unavailable": True},
                error_info={"error": error},
            )
        return result

    @staticmethod
    def _parse_reviewer_model_response_strict(response_text: str) -> dict:
        """Strictly validate the raw model response before trusting Reviewer fallbacks."""
        if not isinstance(response_text, str) or not response_text.strip():
            raise TypeError("Reviewer model returned an empty or non-text response")

        text = response_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) < 3 or not lines[-1].strip().startswith("```"):
                raise ValueError("Reviewer model returned an incomplete JSON code block")
            text = "\n".join(lines[1:-1]).strip()

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise TypeError("Reviewer model response must be a JSON object")
        return parsed

    @staticmethod
    def _validated_review_metadata(review_result: AgentResult) -> dict:
        if not getattr(review_result, "success", False):
            error_info = getattr(review_result, "error_info", {}) or {}
            raise RuntimeError(error_info.get("error") or "Reviewer returned an unsuccessful result")
        metadata = getattr(review_result, "metadata", None)
        if not isinstance(metadata, dict) or not metadata:
            raise RuntimeError("Reviewer returned no review metadata")
        if not isinstance(metadata.get("needs_revision"), bool):
            raise RuntimeError("Reviewer metadata is missing a boolean needs_revision")
        confidence = metadata.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise RuntimeError("Reviewer metadata contains invalid confidence")
        has_review_issue = False
        for field_name in (
            "format_check", "content_check", "logic_check",
            "language_check", "fact_check",
        ):
            check = metadata.get(field_name)
            if not isinstance(check, dict):
                raise RuntimeError(f"Reviewer metadata is missing {field_name}")
            if not isinstance(check.get("passed"), bool):
                raise RuntimeError(f"Reviewer metadata is missing {field_name}.passed")
            if not isinstance(check.get("issues"), list):
                raise RuntimeError(f"Reviewer metadata is missing {field_name}.issues")
            if check["passed"] and check["issues"]:
                raise RuntimeError(f"Reviewer metadata contains contradictory {field_name}")
            has_review_issue = has_review_issue or not check["passed"] or bool(check["issues"])
        for field_name in ("suggestions", "revision_focus"):
            if not isinstance(metadata.get(field_name), list):
                raise RuntimeError(f"Reviewer metadata is missing {field_name}")
        if has_review_issue and not metadata["needs_revision"]:
            raise RuntimeError("Reviewer metadata contains issues but does not request revision")
        return dict(metadata)

    def _step_reflection(self, ctx: ContextPacket, document_content: str,
                         on_think) -> AgentResult:
        """阶段5c: R1深度反思（仅 Reviewer 通过后运行一次）"""
        return self.reflection.process({
            "user_request": ctx.user_request,
            "document_content": document_content,
            "document_type": ctx.plan.get("document_type", "通用公文"),
            "context_analysis": ctx.context_analysis,
            "memory_context": ctx.memory_context,
            "evidence_items": ctx.evidence_items,
        }, on_think=on_think)

    def _merged_key_points(self, ctx: ContextPacket) -> list:
        """合并所有来源的关键点"""
        seen = set()
        merged = []
        for kp in (ctx.context_analysis.get("key_points", []) +
                   ctx.plan.get("key_points", [])):
            if kp not in seen:
                seen.add(kp)
                merged.append(kp)
        return merged[:8]

    def _source_filenames(self, ctx: ContextPacket) -> list:
        """提取知识库命中文件名，用于引用校验和最终记录。"""
        names = []
        for item in ctx.knowledge_sources:
            name = item.get("filename") or os.path.basename(item.get("source", ""))
            if name and name not in names:
                names.append(name)
        return names

    def _source_details(self, ctx: ContextPacket) -> list:
        """Return traceable source metadata for the frontend."""
        details = []
        seen = set()
        for item in ctx.knowledge_sources[:10]:
            filename = item.get("filename") or os.path.basename(item.get("source", ""))
            if not filename:
                continue
            source_type = item.get("source_type", "document")
            key = (
                filename,
                source_type,
                item.get("sheet_name", ""),
                item.get("row_start"),
                item.get("row_end"),
                item.get("chunk_index", -1),
            )
            if key in seen:
                continue
            seen.add(key)
            details.append({
                "filename": filename,
                "source_type": source_type,
                "source_path": item.get("source_path") or item.get("source", ""),
                "category": item.get("category", ""),
                "department": item.get("department", ""),
                "chunk_index": item.get("chunk_index", -1),
                "total_chunks": item.get("total_chunks", -1),
                "sheet_name": item.get("sheet_name", ""),
                "row_start": item.get("row_start"),
                "row_end": item.get("row_end"),
                "row_type": item.get("row_type", "data"),
                "column_headers": item.get("column_headers", []),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "section_title": item.get("section_title", ""),
                "heading_path": item.get("heading_path", []),
                "chunk_text_hash": item.get("chunk_text_hash", ""),
                "parse_warnings": item.get("parse_warnings", []),
            })
        return details

    def _build_evidence_items(self, ctx: ContextPacket) -> list:
        """统一检索证据，避免 Writer/Reviewer 分别解释不同来源。"""
        evidence = []
        for item in ctx.knowledge_sources[:10]:
            filename = item.get("filename") or os.path.basename(item.get("source", ""))
            evidence.append({
                "type": "spreadsheet" if item.get("source_type") == "spreadsheet" else "knowledge",
                "title": filename or item.get("source", "知识库片段"),
                "source": item.get("source_path") or item.get("source") or filename,
                "category": item.get("category", ""),
                "department": item.get("department", ""),
                "chunk_index": item.get("chunk_index", -1),
                "content_hash": item.get("content_hash", ""),
                "source_type": item.get("source_type", "document"),
                "sheet_name": item.get("sheet_name", ""),
                "row_start": item.get("row_start"),
                "row_end": item.get("row_end"),
                "row_type": item.get("row_type", "data"),
                "column_headers": item.get("column_headers", []),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "section_title": item.get("section_title", ""),
                "heading_path": item.get("heading_path", []),
                "chunk_text_hash": item.get("chunk_text_hash", ""),
                "parse_warnings": item.get("parse_warnings", []),
                "spreadsheet_values": item.get("spreadsheet_values", []),
                "score": item.get("rerank_score", item.get("similarity", 0)),
            })

        for item in ctx.search_sources[:5]:
            evidence.append({
                "type": "web" if item.get("source_type") == "bocha" else "search",
                "title": item.get("title", "搜索结果"),
                "source": item.get("url", ""),
                "score": item.get("score", 0),
            })
        return evidence

    def _compact_evidence_items(self, evidence_items: list, limit: int = 6) -> list:
        """压缩证据给 Writer，保留可追溯信息和关键值，减少重复上下文。"""
        compact = []
        for item in (evidence_items or [])[:limit]:
            source_type = item.get("source_type") or item.get("type", "knowledge")
            spreadsheet_values = item.get("spreadsheet_values") or []
            entry = {
                "type": item.get("type", source_type),
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "source_type": source_type,
                "sheet_name": item.get("sheet_name", ""),
                "row_start": item.get("row_start"),
                "row_end": item.get("row_end"),
                "row_type": item.get("row_type", "data"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "section_title": item.get("section_title", ""),
                "heading_path": item.get("heading_path", []),
                "score": item.get("score", 0),
            }
            if spreadsheet_values:
                entry["spreadsheet_values"] = spreadsheet_values[:8]
            headers = item.get("column_headers") or []
            if headers:
                entry["column_headers"] = headers[:12]
            compact.append(entry)
        return compact

    def _writer_knowledge_context(self, ctx: ContextPacket, revision_round: int = 0) -> str:
        """首轮给压缩后的知识库文本；修订轮只给结构化证据，避免反复塞全文。"""
        if revision_round > 0:
            return ""
        text = ctx.knowledge_context or ""
        return text[:2600]

    def _writer_search_context(self, ctx: ContextPacket, revision_round: int = 0) -> str:
        if revision_round > 0:
            return ""
        return (ctx.search_context or "")[:1400]

    def _extract_user_constraints(self, context_analysis: dict, user_profile: Optional[dict]) -> list:
        """从上下文分析和用户画像里提取稳定约束。"""
        constraints = []
        key_info = context_analysis.get("key_info", {}) if isinstance(context_analysis, dict) else {}
        preferences = key_info.get("preferences", {}) if isinstance(key_info, dict) else {}

        if user_profile:
            if user_profile.get("preferred_font"):
                constraints.append(f"字体偏好：{user_profile['preferred_font']}")
            if user_profile.get("preferred_size"):
                constraints.append(f"字号偏好：{user_profile['preferred_size']}")
            if user_profile.get("writing_style"):
                constraints.append(f"写作风格：{user_profile['writing_style']}")

        if isinstance(preferences, dict):
            for key, value in preferences.items():
                if value:
                    constraints.append(f"{key}：{value}")

        writing_style = context_analysis.get("writing_style") if isinstance(context_analysis, dict) else ""
        if writing_style:
            constraints.append(f"上下文风格：{writing_style}")

        return list(dict.fromkeys(constraints))[:10]

    def _context_snapshot(self, ctx: ContextPacket) -> dict:
        """保存轻量上下文快照，便于后续调试和续写。"""
        return {
            "user_intent": ctx.context_analysis.get("user_intent", ""),
            "document_type": ctx.plan.get("document_type", ctx.context_analysis.get("document_type", "")),
            "task_type": ctx.plan.get("task_type", ""),
            "key_points": self._merged_key_points(ctx),
            "user_constraints": ctx.user_constraints,
            "unresolved_questions": ctx.unresolved_questions,
            "evidence_count": len(ctx.evidence_items),
            "has_last_document": bool(ctx.last_document),
            "has_recalled_context": bool(ctx.memory_context),
        }

    @staticmethod
    def _graph_json_safe(value):
        """Validate graph state as JSON-like data; never stringify objects."""
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            if not isfinite(value):
                raise TypeError("Graph state cannot contain non-finite floats")
            return value
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Graph state mapping keys must be strings")
            return {
                key: AgentOrchestrator._graph_json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [AgentOrchestrator._graph_json_safe(item) for item in value]
        raise TypeError(
            "Graph state contains unsupported runtime object: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )

    def _graph_context_to_state(self, ctx: ContextPacket) -> dict:
        fields = (
            "user_request", "context_analysis", "plan", "search_context",
            "knowledge_context", "knowledge_sources", "search_sources",
            "evidence_items", "compact_evidence", "revision_history",
            "run_records", "last_document", "last_plan", "user_constraints",
            "unresolved_questions", "user_profile", "memory_context",
            "audit_summary",
        )
        return {
            field_name: self._graph_json_safe(getattr(ctx, field_name))
            for field_name in fields
        }

    @staticmethod
    def _graph_state_to_context(state: dict) -> ContextPacket:
        return ContextPacket(
            user_request=state.get("user_request", state.get("request_with_context", "")),
            context_analysis=dict(state.get("context_analysis", {}) or {}),
            plan=dict(state.get("plan", {}) or {}),
            search_context=state.get("search_context", "") or "",
            knowledge_context=state.get("knowledge_context", "") or "",
            knowledge_sources=list(state.get("knowledge_sources", []) or []),
            search_sources=list(state.get("search_sources", []) or []),
            evidence_items=list(state.get("evidence_items", []) or []),
            compact_evidence=list(state.get("compact_evidence", []) or []),
            revision_history=list(state.get("revision_history", []) or []),
            run_records=list(state.get("run_records", []) or []),
            last_document=state.get("last_document", "") or "",
            last_plan=dict(state.get("last_plan", {}) or {}),
            user_constraints=list(state.get("user_constraints", []) or []),
            unresolved_questions=list(state.get("unresolved_questions", []) or []),
            user_profile=state.get("user_profile"),
            memory_context=state.get("memory_context", "") or "",
            audit_summary=dict(state.get("audit_summary", {}) or {}),
        )

    def _record_step(self, ctx: ContextPacket, step: str, start_time: float, **extra):
        usage = self._last_agent_usage_for_step(step)
        record = {
            "step": step,
            "duration_ms": int((time.time() - start_time) * 1000),
            **extra,
        }
        if usage:
            record["llm_usage"] = usage
        ctx.run_records.append(record)

    def _last_agent_usage_for_step(self, step: str) -> dict:
        agent = {
            "context_plan": self.planner,
            "plan": self.planner,
            "write": self.writer,
            "review": self.reviewer,
            "reflection": self.reflection,
        }.get(step)
        usage = getattr(agent, "last_usage", {}) if agent else {}
        return dict(usage) if usage else {}

    def _combined_revision_focus(self, review_meta: dict, reflection_meta: Optional[dict] = None) -> list:
        """合并 Reviewer 与 R1 的修订重点，避免两套审查分别驱动返工。"""
        reflection_meta = reflection_meta or {}
        focus = []
        focus.extend(review_meta.get("revision_focus", []) or [])
        focus.extend(review_meta.get("suggestions", []) or [])
        focus.extend(reflection_meta.get("revision_suggestions", []) or [])
        focus.extend(reflection_meta.get("weaknesses", []) or [])
        return list(dict.fromkeys(str(item) for item in focus if item))[:6]

    def _should_reflect(self, ctx: ContextPacket, review_meta: dict, revision_round: int) -> bool:
        """复杂或低置信度任务才触发 R1，减少简单任务的等待时间。"""
        task_type = ctx.plan.get("task_type", "公文生成")
        confidence = review_meta.get("confidence", 0.8)
        high_risk = task_type in {"公文生成", "材料改写", "续写修改"}
        low_confidence = confidence < 0.82
        has_logic_or_fact_issue = bool(
            review_meta.get("logic_check", {}).get("issues")
            or review_meta.get("fact_check", {}).get("issues")
        )
        return high_risk or low_confidence or has_logic_or_fact_issue
