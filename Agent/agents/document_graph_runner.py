"""LangGraph document subgraph runner.

The runner owns graph topology and node transitions. AgentOrchestrator still
owns the concrete step implementations for now, which keeps this extraction
behavior-preserving while creating a cleaner seam for future node classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.document_graph_state import DocumentGraphState
from agents.document_graph_steps import DocumentGraphSteps

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = ""
except Exception as exc:
    END = "__end__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(exc)


@dataclass
class DocumentGraphRunResult:
    ctx: Any
    document_content: str


class DocumentGraphRunner:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator

    def run(self, prepared_run: Any, *, think_handler, thread_id: str) -> DocumentGraphRunResult:
        graph = self._build_graph(think_handler)
        initial_state: DocumentGraphState = {
            "user_request": prepared_run.user_request,
            "request_with_context": prepared_run.request_with_context,
            "previous_context": prepared_run.previous_context,
            "document_content": "",
            "revision_round": 0,
            "continue_revision": False,
            "review_meta": {},
            "reflection_meta": {},
        }
        config = {"configurable": {"thread_id": thread_id or "default"}}
        state = graph.invoke(initial_state, config=config)

        ctx = state["ctx"]
        ctx.run_records.append({
            "step": "orchestrator_runtime",
            "runtime": "langgraph",
            "stream": False,
        })
        return DocumentGraphRunResult(
            ctx=ctx,
            document_content=state.get("document_content", ""),
        )

    def _build_graph(self, think_handler):
        if StateGraph is None:
            raise RuntimeError(f"LangGraph is not available: {LANGGRAPH_IMPORT_ERROR}")

        workflow = StateGraph(DocumentGraphState)
        steps = DocumentGraphSteps(self.orchestrator, think_handler)

        workflow.add_node("context_plan", steps.context_plan)
        workflow.add_node("retrieval", steps.retrieval)
        workflow.add_node("write", steps.write)
        workflow.add_node("review", steps.review)
        workflow.add_node("reflection", steps.reflection)
        workflow.add_node("decide", steps.decide)

        workflow.set_entry_point("context_plan")
        workflow.add_edge("context_plan", "retrieval")
        workflow.add_edge("retrieval", "write")
        workflow.add_edge("write", "review")
        workflow.add_conditional_edges("review", steps.route_after_review, {
            "reflection": "reflection",
            "decide": "decide",
        })
        workflow.add_edge("reflection", "decide")
        workflow.add_conditional_edges("decide", steps.route_after_decide, {
            "write": "write",
            "end": END,
        })
        return workflow.compile()
