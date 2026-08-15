"""Pure node updates for the LangGraph document subgraph."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

try:
    from langgraph.config import get_stream_writer
    from langgraph.runtime import Runtime
except Exception:  # Optional in planner/legacy rollback deployments.
    get_stream_writer = None
    Runtime = Any

from agents.document_graph_state import DocumentGraphRuntime, DocumentGraphState
from agents.document_run_history import reflection_history_entry, review_history_entry
from graph_state_validation import validate_graph_update


class DocumentGraphSteps:
    """Document nodes backed by per-run services from ``Runtime.context``."""

    @staticmethod
    def _services(
        runtime: Runtime[DocumentGraphRuntime],
        state: DocumentGraphState,
    ):
        orchestrator = getattr(runtime.context, "orchestrator", None)
        if orchestrator is None:
            resolver = getattr(runtime.context, "get_document_orchestrator", None)
            if callable(resolver):
                orchestrator = resolver(state)
        if orchestrator is None:
            raise RuntimeError("Document graph runtime has no orchestrator")
        if not getattr(orchestrator, "think_log", None) and state.get("think_log"):
            orchestrator.think_log = list(state.get("think_log") or [])
        return orchestrator

    @staticmethod
    def _json_safe(orchestrator, value):
        converter = getattr(orchestrator, "_graph_json_safe", None)
        return converter(value) if callable(converter) else value

    @classmethod
    def context_to_state(cls, orchestrator, ctx) -> dict:
        converter = getattr(orchestrator, "_graph_context_to_state", None)
        if callable(converter):
            return converter(ctx)
        fields = (
            "user_request", "context_analysis", "plan", "search_context",
            "knowledge_context", "knowledge_sources", "search_sources",
            "evidence_items", "compact_evidence", "revision_history",
            "run_records", "last_document", "last_plan", "user_constraints",
            "unresolved_questions", "user_profile", "memory_context",
            "audit_summary",
        )
        defaults = {
            "user_request": "", "context_analysis": {}, "plan": {},
            "search_context": "", "knowledge_context": "",
            "knowledge_sources": [], "search_sources": [], "evidence_items": [],
            "compact_evidence": [], "revision_history": [], "run_records": [],
            "last_document": "", "last_plan": {}, "user_constraints": [],
            "unresolved_questions": [], "user_profile": None,
            "memory_context": "", "audit_summary": {},
        }
        return {
            field_name: cls._json_safe(
                orchestrator,
                getattr(ctx, field_name, defaults[field_name]),
            )
            for field_name in fields
        }

    @staticmethod
    def state_to_context(orchestrator, state: DocumentGraphState):
        converter = getattr(orchestrator, "_graph_state_to_context", None)
        if callable(converter):
            return converter(state)
        return SimpleNamespace(
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

    @staticmethod
    def _resolved_request(
        runtime: Runtime[DocumentGraphRuntime],
        state: DocumentGraphState,
    ) -> str:
        resolver = getattr(runtime.context, "resolve_document_request", None)
        if callable(resolver):
            return str(resolver(state) or "")
        return str(
            state.get("request_with_context")
            or state.get("input_user_request")
            or ""
        )

    @staticmethod
    def _resolved_previous_context(
        runtime: Runtime[DocumentGraphRuntime],
        state: DocumentGraphState,
    ) -> str:
        resolver = getattr(
            runtime.context,
            "resolve_document_previous_context",
            None,
        )
        if callable(resolver):
            return str(resolver(state) or "")
        return str(state.get("previous_context") or "")

    @classmethod
    def _context_from_state(
        cls,
        orchestrator,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ):
        ctx = cls.state_to_context(orchestrator, state)
        resolver = getattr(runtime.context, "resolve_document_request", None)
        if callable(resolver):
            # Hydrated attachment text and recalled prompt memory live only in
            # Runtime.context.  Reconstruct them for the node, then discard
            # them before returning a checkpoint update.
            ctx.user_request = cls._resolved_request(runtime, state)
            snapshot_resolver = getattr(
                runtime.context,
                "resolve_document_runtime_snapshot",
                None,
            )
            snapshot = (
                snapshot_resolver(state)
                if callable(snapshot_resolver)
                else {}
            )
            if not isinstance(snapshot, dict):
                snapshot = {}
            ctx.memory_context = str(
                snapshot.get("memory_context")
                or getattr(orchestrator, "_current_agent_memory_context", "")
                or ""
            )
            ctx.user_profile = snapshot.get(
                "user_profile",
                getattr(orchestrator, "user_profile", None),
            )
            ctx.last_plan = dict(
                snapshot.get("last_plan") or state.get("last_plan") or {}
            )
            if int(state.get("revision_round", 0) or 0) > 0:
                ctx.last_document = str(
                    state.get("last_document")
                    or state.get("document_content")
                    or ""
                )
            else:
                ctx.last_document = str(snapshot.get("last_document") or "")
        return ctx

    @classmethod
    def _context_update(
        cls,
        orchestrator,
        ctx,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ) -> dict:
        update = cls.context_to_state(orchestrator, ctx)
        update["think_log"] = cls._json_safe(
            orchestrator,
            list(getattr(orchestrator, "think_log", []) or []),
        )
        resolver = getattr(runtime.context, "resolve_document_request", None)
        if callable(resolver):
            safe_request = str(
                state.get("working_message")
                or state.get("request_message")
                or state.get("input_user_request")
                or ""
            )
            update["user_request"] = safe_request
            update["memory_context"] = ""
            update["user_profile"] = None
            update["last_plan"] = {}
            # Historical documents stay in the run-scoped runtime snapshot.
            # Once revision begins, retaining the active previous draft preserves
            # the established Writer contract without exposing hydrated inputs.
            candidate_last_document = str(update.get("last_document") or "")
            active_document = str(state.get("document_content") or "")
            if not (
                int(state.get("revision_round", 0) or 0) > 0
                or (
                    active_document
                    and candidate_last_document == active_document
                )
            ):
                update["last_document"] = ""
        return update

    @classmethod
    def _finalize_update(
        cls,
        state: DocumentGraphState,
        update: dict,
        runtime: Runtime[DocumentGraphRuntime],
        *,
        node: str,
    ) -> DocumentGraphState:
        """Sanitize the complete node delta, then enforce strict JSON State."""

        sanitizer = getattr(runtime.context, "sanitize_document_update", None)
        if callable(sanitizer):
            update = sanitizer(state, update)
        return validate_graph_update(update, node=f"document.{node}")

    @staticmethod
    def _emit(event: dict, runtime: Runtime[DocumentGraphRuntime]) -> None:
        encoder = getattr(runtime.context, "encode_document_event", None)
        if get_stream_writer is None:
            raise RuntimeError("LangGraph streaming dependencies are unavailable")
        get_stream_writer()(encoder(event) if callable(encoder) else event)

    @classmethod
    def _think_handler(
        cls,
        runtime: Runtime[DocumentGraphRuntime],
        state: DocumentGraphState,
    ):
        base_handler = getattr(runtime.context, "think_handler", None)
        if not callable(base_handler):
            orchestrator = cls._services(runtime, state)
            base_handler = orchestrator._think_handler()

        def handler(agent_name, emoji, message):
            base_handler(agent_name, emoji, message)
            cls._emit({
                "type": "think",
                "agent": agent_name,
                "emoji": emoji,
                "message": message,
            }, runtime)

        return handler

    @staticmethod
    def _mark_current_node(
        runtime: Runtime[DocumentGraphRuntime],
        state: DocumentGraphState,
        node: str,
    ) -> None:
        observer = getattr(runtime.context, "observe_document_node", None)
        if callable(observer):
            observer(state, node)

    @classmethod
    def _raise_native_failure(
        cls,
        runtime: Runtime[DocumentGraphRuntime],
        state: DocumentGraphState,
        step: str,
        exc: Exception,
    ) -> None:
        if not bool(getattr(runtime.context, "raise_document_node_errors", False)):
            return
        handler = getattr(runtime.context, "handle_document_failure", None)
        if callable(handler):
            handler(state, step, exc)
        cls._emit({
            "type": "run_failed",
            "message": str(exc)[:500],
            "step": step,
        }, runtime)
        cls._emit({"type": "error", "message": "生成失败，请稍后重试"}, runtime)
        raise exc

    @staticmethod
    def _failure(state: DocumentGraphState, step: str, error: object) -> DocumentGraphState:
        message = str(error)[:500] or f"{step} failed"
        errors = list(state.get("errors", []) or [])
        errors.append({"step": step, "message": message})
        return {
            "run_status": "failed",
            "quality_status": "failed",
            "continue_revision": False,
            "error_step": step,
            "error_message": message,
            "errors": errors,
        }

    def context_plan(
        self,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ) -> DocumentGraphState:
        self._mark_current_node(runtime, state, "context_plan")
        orchestrator = self._services(runtime, state)
        think_handler = self._think_handler(runtime, state)
        self._emit({"type": "context_start", "message": "开始分析上下文并制定计划..."}, runtime)
        step_start = time.time()
        try:
            ctx = orchestrator._step_context_plan(
                self._resolved_request(runtime, state),
                self._resolved_previous_context(runtime, state),
                think_handler,
            )
            orchestrator._record_step(
                ctx,
                "context_plan",
                step_start,
                task_type=ctx.plan.get("task_type"),
            )
        except Exception as exc:
            self._raise_native_failure(runtime, state, "context_plan", exc)
            return self._finalize_update(
                state,
                self._failure(state, "context_plan", exc),
                runtime,
                node="context_plan",
            )
        update = self._context_update(orchestrator, ctx, state, runtime)
        update.update({"run_status": "running", "quality_status": "pending"})
        self._emit({"type": "context_end", "data": getattr(ctx, "context_analysis", {})}, runtime)
        self._emit({"type": "plan_start", "message": "任务计划已生成"}, runtime)
        self._emit({"type": "plan", "data": ctx.plan}, runtime)
        return self._finalize_update(
            state, update, runtime, node="context_plan"
        )

    def retrieval(
        self,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ) -> DocumentGraphState:
        self._mark_current_node(runtime, state, "retrieval")
        orchestrator = self._services(runtime, state)
        think_handler = self._think_handler(runtime, state)
        ctx = self._context_from_state(orchestrator, state, runtime)
        step_start = time.time()
        try:
            need_web_search = bool(ctx.plan.get("need_web_search"))
            if need_web_search:
                think_handler("Orchestrator", "⚡", "先联网搜索，再增强知识库检索...")
                ctx = orchestrator._step_search(ctx, think_handler)
            else:
                think_handler("Orchestrator", "⚡", "知识库检索中...")
            ctx = orchestrator._step_knowledge(ctx, think_handler)
            ctx.evidence_items = orchestrator._build_evidence_items(ctx)
            ctx.compact_evidence = orchestrator._compact_evidence_items(ctx.evidence_items)
            orchestrator._record_step(
                ctx,
                "retrieval",
                step_start,
                source_count=len(ctx.knowledge_sources),
                evidence_count=len(ctx.evidence_items),
                need_web_search=need_web_search,
            )
        except Exception as exc:
            self._raise_native_failure(runtime, state, "retrieval", exc)
            update = self._context_update(orchestrator, ctx, state, runtime)
            update.update(self._failure(state, "retrieval", exc))
            return self._finalize_update(
                state, update, runtime, node="retrieval"
            )
        return self._finalize_update(
            state,
            self._context_update(orchestrator, ctx, state, runtime),
            runtime,
            node="retrieval",
        )

    def write(
        self,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ) -> DocumentGraphState:
        self._mark_current_node(runtime, state, "write")
        orchestrator = self._services(runtime, state)
        think_handler = self._think_handler(runtime, state)
        ctx = self._context_from_state(orchestrator, state, runtime)
        revision_round = int(state.get("revision_round", 0))
        document_type = ctx.plan.get("document_type", "公文")
        self._emit({"type": "write_start", "message": f"开始生成{document_type}..."}, runtime)
        step_start = time.time()

        try:
            # invoke() and stream() must execute identical business logic.  The
            # stream API changes only how custom graph events are consumed; it
            # must not select a different Writer prompt or model entrypoint.
            writer_result = orchestrator._step_write(ctx, think_handler)
            if not getattr(writer_result, "success", True):
                error_info = getattr(writer_result, "error_info", {}) or {}
                raise RuntimeError(error_info.get("error") or "Writer returned an unsuccessful result")
            document_content = getattr(writer_result, "content", "")
        except Exception as exc:
            orchestrator._record_step(
                ctx,
                "write",
                step_start,
                round=revision_round + 1,
                success=False,
                error=str(exc)[:500],
            )
            self._raise_native_failure(runtime, state, "write", exc)
            update = self._context_update(orchestrator, ctx, state, runtime)
            update.update(self._failure(state, "write", exc))
            return self._finalize_update(state, update, runtime, node="write")

        document_content = str(document_content or "").strip()
        if not document_content:
            exc = RuntimeError("Writer returned empty document content")
            orchestrator._record_step(
                ctx,
                "write",
                step_start,
                round=revision_round + 1,
                success=False,
                error=str(exc),
            )
            self._raise_native_failure(runtime, state, "write", exc)
            update = self._context_update(orchestrator, ctx, state, runtime)
            update.update(self._failure(state, "write", exc))
            return self._finalize_update(state, update, runtime, node="write")

        sanitizer = getattr(orchestrator, "_sanitize_document_output", None)
        if callable(sanitizer):
            document_content = sanitizer(
                document_content,
                ctx.user_request,
            )
        orchestrator._record_step(
            ctx,
            "write",
            step_start,
            round=revision_round + 1,
            success=True,
        )
        update = self._context_update(orchestrator, ctx, state, runtime)
        update.update({
            "document_content": document_content,
            "continue_revision": False,
            "review_meta": {},
            "reflection_meta": {},
            "run_status": "running",
            "error_step": "",
            "error_message": "",
        })
        return self._finalize_update(state, update, runtime, node="write")

    def review(
        self,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ) -> DocumentGraphState:
        self._mark_current_node(runtime, state, "review")
        orchestrator = self._services(runtime, state)
        think_handler = self._think_handler(runtime, state)
        ctx = self._context_from_state(orchestrator, state, runtime)
        document_content = state.get("document_content", "")
        revision_round = int(state.get("revision_round", 0))
        step_start = time.time()

        try:
            review_result = orchestrator._step_review(ctx, document_content, think_handler)
            validator = getattr(orchestrator, "_validated_review_metadata", None)
            if callable(validator):
                review_meta = validator(review_result)
            else:
                if not getattr(review_result, "success", True):
                    error_info = getattr(review_result, "error_info", {}) or {}
                    raise RuntimeError(error_info.get("error") or "Reviewer returned an unsuccessful result")
                review_meta = dict(getattr(review_result, "metadata", {}) or {})
                if not review_meta:
                    raise RuntimeError("Reviewer returned no review metadata")
        except Exception as exc:
            orchestrator._record_step(
                ctx,
                "review",
                step_start,
                round=revision_round + 1,
                success=False,
                error=str(exc)[:500],
            )
            self._raise_native_failure(runtime, state, "review", exc)
            update = self._context_update(orchestrator, ctx, state, runtime)
            update.update(self._failure(state, "review", exc))
            return self._finalize_update(state, update, runtime, node="review")

        ctx.audit_summary = review_meta.get("spreadsheet_audit", {}) or {}
        orchestrator._record_step(
            ctx,
            "review",
            step_start,
            round=revision_round + 1,
            success=True,
            needs_revision=review_meta.get("needs_revision", False),
            confidence=review_meta.get("confidence", 0.8),
        )
        ctx.revision_history.append(review_history_entry(review_meta, revision_round))
        update = self._context_update(orchestrator, ctx, state, runtime)
        update["review_meta"] = self._json_safe(orchestrator, review_meta)
        update["should_reflect"] = bool(
            orchestrator._should_reflect(ctx, review_meta, revision_round)
        )
        return self._finalize_update(state, update, runtime, node="review")

    def reflection(
        self,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ) -> DocumentGraphState:
        self._mark_current_node(runtime, state, "reflection")
        orchestrator = self._services(runtime, state)
        think_handler = self._think_handler(runtime, state)
        ctx = self._context_from_state(orchestrator, state, runtime)
        document_content = state.get("document_content", "")
        revision_round = int(state.get("revision_round", 0))
        step_start = time.time()

        try:
            reflection_result = orchestrator._step_reflection(ctx, document_content, think_handler)
            if not getattr(reflection_result, "success", True):
                raise RuntimeError("Reflection returned an unsuccessful result")
            reflection_meta = dict(getattr(reflection_result, "metadata", {}) or {})
        except Exception as exc:
            orchestrator._record_step(
                ctx,
                "reflection",
                step_start,
                round=revision_round + 1,
                success=False,
                error=str(exc)[:500],
            )
            errors = list(state.get("errors", []) or [])
            errors.append({"step": "reflection", "message": str(exc)[:500]})
            update = self._context_update(orchestrator, ctx, state, runtime)
            update.update({"reflection_done": True, "reflection_meta": {}, "errors": errors})
            return self._finalize_update(
                state, update, runtime, node="reflection"
            )

        orchestrator._record_step(
            ctx,
            "reflection",
            step_start,
            round=revision_round + 1,
            success=True,
            needs_revision=reflection_meta.get("needs_revision", False),
        )
        ctx.revision_history.append(reflection_history_entry(reflection_meta, revision_round))
        if reflection_meta.get("needs_revision", False):
            weaknesses = reflection_meta.get("weaknesses", [])
            think_handler(
                "Orchestrator",
                "🧠",
                f"R1深度反思发现问题：{'；'.join(weaknesses[:2])}",
            )
        update = self._context_update(orchestrator, ctx, state, runtime)
        update.update({
            "reflection_done": True,
            "reflection_meta": self._json_safe(orchestrator, reflection_meta),
        })
        return self._finalize_update(state, update, runtime, node="reflection")

    def decide(
        self,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ) -> DocumentGraphState:
        self._mark_current_node(runtime, state, "decide")
        orchestrator = self._services(runtime, state)
        think_handler = self._think_handler(runtime, state)
        ctx = self._context_from_state(orchestrator, state, runtime)
        review_meta = state.get("review_meta", {}) or {}
        reflection_meta = state.get("reflection_meta", {}) or {}
        revision_round = int(state.get("revision_round", 0))
        needs_revision = bool(
            review_meta.get("needs_revision", False)
            or reflection_meta.get("needs_revision", False)
        )

        if needs_revision and revision_round < orchestrator.MAX_TOTAL_ROUNDS - 1:
            focus = orchestrator._combined_revision_focus(review_meta, reflection_meta)
            think_handler(
                "Orchestrator",
                "🔄",
                f"第{revision_round + 1}轮已汇总审核意见，重点：{'；'.join(focus[:3])}",
            )
            ctx.last_document = state.get("document_content", "")
            update = self._context_update(orchestrator, ctx, state, runtime)
            update.update({
                "revision_round": revision_round + 1,
                "continue_revision": True,
                "quality_status": "pending",
            })
            return self._finalize_update(state, update, runtime, node="decide")

        if needs_revision:
            think_handler("Orchestrator", "⚠️", "已达最大修订轮次，输出当前最优版本")
            return self._finalize_update(state, {
                "continue_revision": False,
                "quality_status": "max_revisions",
                "run_status": "completed",
            }, runtime, node="decide")

        think_handler("Reviewer", "✅", "审核通过，无需修改")
        return self._finalize_update(state, {
            "continue_revision": False,
            "quality_status": "passed",
            "run_status": "completed",
        }, runtime, node="decide")

    def finalize(
        self,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ) -> DocumentGraphState:
        self._mark_current_node(runtime, state, "finalize")
        return self._finalize_update(state, {
            "run_status": "completed",
            "document_content": state.get("document_content", ""),
        }, runtime, node="finalize")

    def error_terminal(
        self,
        state: DocumentGraphState,
        runtime: Runtime[DocumentGraphRuntime],
    ) -> DocumentGraphState:
        self._mark_current_node(runtime, state, "error_terminal")
        message = state.get("error_message", "文档生成失败")
        self._emit({"type": "run_failed", "message": message, "step": state.get("error_step", "")}, runtime)
        return self._finalize_update(
            state,
            {"run_status": "failed", "quality_status": "failed"},
            runtime,
            node="error_terminal",
        )

    @staticmethod
    def route_after_write(state: DocumentGraphState) -> str:
        return "error" if state.get("run_status") == "failed" else "review"

    @staticmethod
    def route_after_context_plan(state: DocumentGraphState) -> str:
        return "error" if state.get("run_status") == "failed" else "retrieval"

    @staticmethod
    def route_after_retrieval(state: DocumentGraphState) -> str:
        return "error" if state.get("run_status") == "failed" else "write"

    @staticmethod
    def route_after_review(state: DocumentGraphState) -> str:
        if state.get("run_status") == "failed":
            return "error"
        if (
            not state.get("reflection_done", False)
            and state.get("should_reflect", False)
        ):
            return "reflection"
        return "decide"

    @staticmethod
    def route_after_decide(state: DocumentGraphState) -> str:
        return "write" if state.get("continue_revision", False) else "finalize"
