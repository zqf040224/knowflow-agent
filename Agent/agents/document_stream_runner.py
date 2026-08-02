"""Streaming document runner for the public SSE document flow."""

from __future__ import annotations

import time
import re
from queue import Queue
from threading import Thread
from typing import Any, Optional

from agents.document_run_history import reflection_history_entry, review_history_entry


class DocumentStreamRunner:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator

    def run(self, prepared_run: Any, *, user_request: str, on_think: Optional[Any] = None):
        def think_yield(agent_name, emoji, message):
            self.orchestrator._on_think(agent_name, emoji, message)
            if on_think:
                on_think(agent_name, emoji, message)
            return {"type": "think", "agent": agent_name, "emoji": emoji, "message": message}

        yield {"type": "context_start", "message": "开始分析上下文并制定计划..."}
        step_start = time.time()
        ctx = yield from self._run_blocking_step_stream(
            lambda cb: self.orchestrator._step_context_plan(
                prepared_run.request_with_context,
                prepared_run.previous_context,
                cb,
            ),
            on_think=on_think,
        )
        self.orchestrator._record_step(ctx, "context_plan", step_start, task_type=ctx.plan.get("task_type"))
        yield {"type": "context_end", "data": ctx.context_analysis}
        yield {"type": "plan_start", "message": "任务计划已生成"}
        yield {"type": "plan", "data": ctx.plan}

        need_search = ctx.plan.get("need_web_search")
        if need_search:
            yield think_yield("Orchestrator", "⚡", "先联网搜索，再增强知识库检索...")
        else:
            yield think_yield("Orchestrator", "⚡", "知识库检索中...")

        step_start = time.time()
        if need_search:
            ctx = yield from self._run_blocking_step_stream(
                lambda cb: self.orchestrator._step_search(ctx, cb),
                on_think=on_think,
            )
            if ctx.search_context:
                yield think_yield("SearchAgent", "📡", f"搜索到 {len(ctx.search_context)} 字符的参考信息")

        ctx = yield from self._run_blocking_step_stream(
            lambda cb: self.orchestrator._step_knowledge(ctx, cb),
            on_think=on_think,
        )
        ctx.evidence_items = self.orchestrator._build_evidence_items(ctx)
        ctx.compact_evidence = self.orchestrator._compact_evidence_items(ctx.evidence_items)
        self.orchestrator._record_step(
            ctx,
            "retrieval",
            step_start,
            source_count=len(ctx.knowledge_sources),
            evidence_count=len(ctx.evidence_items),
            need_web_search=bool(need_search),
        )

        self.orchestrator._reflection_done = False
        doc_content = ""
        best_doc_content = ""

        for revision_round in range(self.orchestrator.MAX_TOTAL_ROUNDS):
            yield think_yield("Writer", "📝", f"正在生成{ctx.plan.get('document_type', '公文')}草稿...")
            yield {"type": "write_start", "message": f"开始生成{ctx.plan.get('document_type', '公文')}..."}

            step_start = time.time()
            stream = self.orchestrator.writer.process_stream({
                "user_request": ctx.user_request,
                "search_context": self.orchestrator._writer_search_context(ctx, revision_round),
                "knowledge_context": self.orchestrator._writer_knowledge_context(ctx, revision_round),
                "document_type": ctx.plan.get("document_type", "通用公文"),
                "task_type": ctx.plan.get("task_type", "公文生成"),
                "key_points": self.orchestrator._merged_key_points(ctx),
                "revision_history": ctx.revision_history,
                "knowledge_sources": ctx.knowledge_sources,
                "context_analysis": ctx.context_analysis,
                "memory_context": getattr(ctx, "memory_context", ""),
                "last_document": ctx.last_document,
                "draft_document": doc_content if revision_round > 0 else "",
                "last_plan": ctx.last_plan,
                "user_constraints": ctx.user_constraints,
                "unresolved_questions": ctx.unresolved_questions,
                "evidence_items": ctx.compact_evidence,
                "revision_mode": revision_round > 0,
            })

            draft_content = ""
            try:
                for chunk in stream:
                    draft_content += chunk
            except Exception as exc:
                self.orchestrator._record_step(
                    ctx,
                    "write",
                    step_start,
                    round=revision_round + 1,
                    recovered=bool(best_doc_content),
                    error=str(exc),
                )
                if best_doc_content:
                    doc_content = best_doc_content
                    yield think_yield(
                        "Writer",
                        "⚠️",
                        "修订生成连接中断，已保留上一版可用结果继续输出",
                    )
                    break
                raise

            doc_content = self._sanitize_unsupported_specifics(draft_content, ctx.user_request)
            best_doc_content = doc_content or best_doc_content
            self.orchestrator._record_step(ctx, "write", step_start, round=revision_round + 1)

            step_start = time.time()
            review_result = yield from self._run_blocking_step_stream(
                lambda cb: self.orchestrator._step_review(ctx, doc_content, cb),
                on_think=on_think,
            )
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
            is_last_round = (revision_round == self.orchestrator.MAX_TOTAL_ROUNDS - 1)

            reflection_meta = {}
            if (
                not self.orchestrator._reflection_done
                and self.orchestrator._should_reflect(ctx, review_meta, revision_round)
            ):
                reflection_meta = yield from self._run_reflection_stream(
                    ctx,
                    doc_content,
                    revision_round,
                    think_yield,
                )

                if reflection_meta.get("needs_revision", False) and not is_last_round:
                    weaknesses = reflection_meta.get("weaknesses", [])
                    yield think_yield("Orchestrator", "🧠",
                                      f"R1反思发现问题：{'；'.join(weaknesses[:2])}")

            needs_revision = review_meta.get("needs_revision", False) or reflection_meta.get("needs_revision", False)
            if needs_revision:
                if is_last_round:
                    yield think_yield("Orchestrator", "⚠️", "已达最大修订轮次，输出当前最优版本")
                else:
                    focus = self.orchestrator._combined_revision_focus(review_meta, reflection_meta)
                    yield think_yield("Orchestrator", "🔄",
                                      f"第{revision_round + 1}轮已汇总审核意见，重点：{'；'.join(focus[:3])}")
                    continue

            yield think_yield("Reviewer", "✅", "审核通过，无需修改")
            break

        if doc_content:
            yield think_yield("Orchestrator", "📄", "最终版本已确认，正在输出正文")
            yield {"type": "thinking_done", "summary": "写作、审核和反思完成，开始输出最终正文"}
            yield {"type": "answer_start", "message": "开始输出正文"}
            for chunk in self._iter_final_content_chunks(doc_content):
                yield {"type": "answer_delta", "data": chunk}
                yield {"type": "content", "data": chunk}
                time.sleep(0.04)
            yield {"type": "answer_done", "answer": doc_content}

        self._save_stream_result(ctx, doc_content, user_request)

        yield think_yield("Orchestrator", "✅", f"文档生成完成，共{len(self.orchestrator.think_log)}个思考步骤")
        yield {
            "type": "done",
            "document": doc_content,
            "plan": ctx.plan,
            "think_log": self.orchestrator.think_log,
            "run_records": ctx.run_records,
            "source_filenames": self.orchestrator._source_filenames(ctx),
            "source_details": self.orchestrator._source_details(ctx),
            "audit_summary": ctx.audit_summary,
        }

    def _run_reflection_stream(self, ctx: Any, doc_content: str, revision_round: int, think_yield):
        self.orchestrator._reflection_done = True
        step_start = time.time()
        reflection_result = None
        for event in self.orchestrator.reflection.process_stream({
            "user_request": ctx.user_request,
            "document_content": doc_content,
            "document_type": ctx.plan.get("document_type", "通用公文"),
            "context_analysis": ctx.context_analysis,
            "memory_context": getattr(ctx, "memory_context", ""),
            "evidence_items": ctx.evidence_items,
        }, on_think=think_yield):
            if event["type"] == "think":
                yield event
            elif event["type"] == "reasoning":
                continue
            elif event["type"] == "result":
                reflection_result = event["data"]

        reflection_meta = reflection_result.metadata if reflection_result else {}
        public_reflection_meta = dict(reflection_meta)
        public_reflection_meta.pop("reasoning_content", None)
        self.orchestrator._record_step(
            ctx,
            "reflection",
            step_start,
            round=revision_round + 1,
            needs_revision=reflection_meta.get("needs_revision", False),
        )

        yield {
            "type": "reflection",
            "data": public_reflection_meta,
        }

        ctx.revision_history.append(reflection_history_entry(reflection_meta, revision_round))
        return reflection_meta

    @staticmethod
    def _iter_final_content_chunks(text: str, chunk_size: int = 45):
        if not text:
            return
        for start in range(0, len(text), chunk_size):
            yield text[start:start + chunk_size]

    @staticmethod
    def _sanitize_unsupported_specifics(text: str, user_request: str) -> str:
        """Remove common invented meeting specifics that were not supplied by the user."""
        if not text:
            return text
        request = user_request or ""
        sanitized = text

        if not re.search(r"A栋|a栋|301|三楼|3楼", request):
            sanitized = re.sub(r"示例单位A栋(?:3楼|三楼)?301(?:室|会议室)", "会议室", sanitized)
            sanitized = re.sub(r"示例单位A栋会议室", "会议室", sanitized)
            sanitized = re.sub(r"A栋(?:3楼|三楼)?301(?:室|会议室)", "会议室", sanitized)
            sanitized = re.sub(r"A栋会议室", "会议室", sanitized)

        if "星期" not in request and "周" not in request:
            sanitized = re.sub(r"(20\d{2}年\d{1,2}月\d{1,2}日)（星期[一二三四五六日天]）", r"\1", sanitized)
            sanitized = re.sub(r"(\d{1,2}月\d{1,2}日)（星期[一二三四五六日天]）", r"\1", sanitized)

        return sanitized

    def _run_blocking_step_stream(self, step_callable, on_think: Optional[Any] = None):
        events = Queue()
        sentinel = object()
        result_holder = {}

        def stream_think(agent_name, emoji, message):
            self.orchestrator._on_think(agent_name, emoji, message)
            if on_think:
                on_think(agent_name, emoji, message)
            events.put({
                "type": "think",
                "agent": agent_name,
                "emoji": emoji,
                "message": message,
            })

        def target():
            try:
                result_holder["result"] = step_callable(stream_think)
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                events.put(sentinel)

        thread = Thread(target=target, daemon=True)
        thread.start()

        while True:
            item = events.get()
            if item is sentinel:
                break
            yield item

        thread.join()
        if "error" in result_holder:
            raise result_holder["error"]
        return result_holder.get("result")

    def _save_stream_result(self, ctx: Any, doc_content: str, user_request: str) -> None:
        if not self.orchestrator.memory or not self.orchestrator.session_id:
            return

        source_filenames = self.orchestrator._source_filenames(ctx)
        self.orchestrator.memory.add_message(
            self.orchestrator.session_id,
            "assistant",
            doc_content,
            metadata={
                "type": "document",
                "plan": ctx.plan,
                "run_records": ctx.run_records,
                "source_filenames": source_filenames,
                "source_details": self.orchestrator._source_details(ctx),
                "context_snapshot": self.orchestrator._context_snapshot(ctx),
            },
        )
        self.orchestrator.memory.set_context(self.orchestrator.session_id, "last_document", doc_content)
        self.orchestrator.memory.set_context(self.orchestrator.session_id, "last_plan", ctx.plan)
        if hasattr(self.orchestrator.memory, "update_rolling_summary"):
            self.orchestrator.memory.update_rolling_summary(
                self.orchestrator.session_id,
                user_request,
                doc_content,
                ctx.plan,
                source_filenames,
            )
