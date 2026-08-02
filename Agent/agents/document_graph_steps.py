"""Node implementations for the LangGraph document subgraph."""

from __future__ import annotations

import time
from typing import Any

from agents.document_run_history import reflection_history_entry, review_history_entry
from agents.document_graph_state import DocumentGraphState


class DocumentGraphSteps:
    def __init__(self, orchestrator: Any, think_handler):
        self.orchestrator = orchestrator
        self.think_handler = think_handler

    def context_plan(self, state: DocumentGraphState) -> DocumentGraphState:
        step_start = time.time()
        ctx = self.orchestrator._step_context_plan(
            state["request_with_context"],
            state.get("previous_context", ""),
            self.think_handler,
        )
        self.orchestrator._record_step(ctx, "context_plan", step_start, task_type=ctx.plan.get("task_type"))
        return {"ctx": ctx}

    def retrieval(self, state: DocumentGraphState) -> DocumentGraphState:
        ctx = state["ctx"]
        step_start = time.time()
        if ctx.plan.get("need_web_search"):
            ctx = self.orchestrator._step_search(ctx, self.think_handler)
        ctx = self.orchestrator._step_knowledge(ctx, self.think_handler)
        ctx.evidence_items = self.orchestrator._build_evidence_items(ctx)
        ctx.compact_evidence = self.orchestrator._compact_evidence_items(ctx.evidence_items)
        self.orchestrator._record_step(
            ctx,
            "retrieval",
            step_start,
            source_count=len(ctx.knowledge_sources),
            evidence_count=len(ctx.evidence_items),
            need_web_search=bool(ctx.plan.get("need_web_search")),
        )
        return {"ctx": ctx}

    def write(self, state: DocumentGraphState) -> DocumentGraphState:
        ctx = state["ctx"]
        revision_round = state.get("revision_round", 0)
        step_start = time.time()
        writer_result = self.orchestrator._step_write(ctx, self.think_handler)
        document_content = writer_result.content
        self.orchestrator._record_step(ctx, "write", step_start, round=revision_round + 1)
        return {
            "ctx": ctx,
            "document_content": document_content,
            "continue_revision": False,
            "review_meta": {},
            "reflection_meta": {},
        }

    def review(self, state: DocumentGraphState) -> DocumentGraphState:
        ctx = state["ctx"]
        document_content = state.get("document_content", "")
        revision_round = state.get("revision_round", 0)
        step_start = time.time()
        review_result = self.orchestrator._step_review(ctx, document_content, self.think_handler)
        review_meta = review_result.metadata
        ctx.audit_summary = review_meta.get("spreadsheet_audit", {}) or {}
        self.orchestrator._record_step(
            ctx,
            "review",
            step_start,
            round=revision_round + 1,
            needs_revision=review_meta.get("needs_revision", False),
            confidence=review_meta.get("confidence", 0.8),
        )
        ctx.revision_history.append(review_history_entry(review_meta, revision_round))
        return {"ctx": ctx, "review_meta": review_meta}

    def reflection(self, state: DocumentGraphState) -> DocumentGraphState:
        ctx = state["ctx"]
        document_content = state.get("document_content", "")
        revision_round = state.get("revision_round", 0)
        self.orchestrator._reflection_done = True
        step_start = time.time()
        reflection_result = self.orchestrator._step_reflection(ctx, document_content, self.think_handler)
        reflection_meta = reflection_result.metadata
        self.orchestrator._record_step(
            ctx,
            "reflection",
            step_start,
            round=revision_round + 1,
            needs_revision=reflection_meta.get("needs_revision", False),
        )
        ctx.revision_history.append(reflection_history_entry(reflection_meta, revision_round))
        if reflection_meta.get("needs_revision", False):
            weaknesses = reflection_meta.get("weaknesses", [])
            self.think_handler("Orchestrator", "🧠",
                               f"R1深度反思发现问题：{'；'.join(weaknesses[:2])}")
        return {"ctx": ctx, "reflection_meta": reflection_meta}

    def decide(self, state: DocumentGraphState) -> DocumentGraphState:
        ctx = state["ctx"]
        review_meta = state.get("review_meta", {}) or {}
        reflection_meta = state.get("reflection_meta", {}) or {}
        revision_round = state.get("revision_round", 0)
        needs_revision = (
            review_meta.get("needs_revision", False)
            or reflection_meta.get("needs_revision", False)
        )
        if needs_revision and revision_round < self.orchestrator.MAX_TOTAL_ROUNDS - 1:
            focus = self.orchestrator._combined_revision_focus(review_meta, reflection_meta)
            self.think_handler("Orchestrator", "🔄",
                               f"第{revision_round + 1}轮已汇总审核意见，重点：{'；'.join(focus[:3])}")
            ctx.last_document = state.get("document_content", "")
            return {"ctx": ctx, "revision_round": revision_round + 1, "continue_revision": True}

        if not needs_revision:
            self.think_handler("Reviewer", "✅", "审核通过，无需修改")
        return {"ctx": ctx, "continue_revision": False}

    def route_after_review(self, state: DocumentGraphState) -> str:
        ctx = state["ctx"]
        review_meta = state.get("review_meta", {}) or {}
        revision_round = state.get("revision_round", 0)
        if (
            not self.orchestrator._reflection_done
            and self.orchestrator._should_reflect(ctx, review_meta, revision_round)
        ):
            return "reflection"
        return "decide"

    @staticmethod
    def route_after_decide(state: DocumentGraphState) -> str:
        if state.get("continue_revision", False):
            return "write"
        return "end"
