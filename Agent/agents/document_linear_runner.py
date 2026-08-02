"""Legacy linear document runner used when LangGraph is disabled."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agents.document_run_history import reflection_history_entry, review_history_entry


@dataclass
class DocumentLinearRunResult:
    ctx: Any
    document_content: str


class DocumentLinearRunner:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator

    def run(self, prepared_run: Any, *, think_handler) -> DocumentLinearRunResult:
        step_start = time.time()
        ctx = self.orchestrator._step_context_plan(
            prepared_run.request_with_context,
            prepared_run.previous_context,
            think_handler,
        )
        self.orchestrator._record_step(ctx, "context_plan", step_start, task_type=ctx.plan.get("task_type"))

        step_start = time.time()
        if ctx.plan.get("need_web_search"):
            ctx = self.orchestrator._step_search(ctx, think_handler)
        ctx = self.orchestrator._step_knowledge(ctx, think_handler)
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

        self.orchestrator._reflection_done = False
        document_content = ""
        for revision_round in range(self.orchestrator.MAX_TOTAL_ROUNDS):
            step_start = time.time()
            writer_result = self.orchestrator._step_write(ctx, think_handler)
            document_content = writer_result.content
            self.orchestrator._record_step(ctx, "write", step_start, round=revision_round + 1)

            step_start = time.time()
            review_result = self.orchestrator._step_review(ctx, document_content, think_handler)
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

            reflection_meta = {}
            if (
                not self.orchestrator._reflection_done
                and self.orchestrator._should_reflect(ctx, review_meta, revision_round)
            ):
                self.orchestrator._reflection_done = True
                step_start = time.time()
                reflection_result = self.orchestrator._step_reflection(ctx, document_content, think_handler)
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
                    think_handler("Orchestrator", "🧠",
                                  f"R1深度反思发现问题：{'；'.join(weaknesses[:2])}")

            needs_revision = review_meta.get("needs_revision", False) or reflection_meta.get("needs_revision", False)
            if needs_revision:
                focus = self.orchestrator._combined_revision_focus(review_meta, reflection_meta)
                think_handler("Orchestrator", "🔄",
                              f"第{revision_round + 1}轮已汇总审核意见，重点：{'；'.join(focus[:3])}")
                ctx.last_document = document_content
                continue

            think_handler("Reviewer", "✅", "审核通过，无需修改")
            break

        return DocumentLinearRunResult(ctx=ctx, document_content=document_content)
