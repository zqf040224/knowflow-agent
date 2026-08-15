"""Agent package exports loaded on demand.

Importing a lightweight rollback service must not import the document
LangGraph stack.  Lazy package exports keep the historical ``from agents
import ...`` API while leaving optional graph dependencies behind the graph
runtime boundary.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BaseAgent": ("agents.base_agent", "BaseAgent"),
    "AgentResult": ("agents.base_agent", "AgentResult"),
    "AgentMessage": ("agents.base_agent", "AgentMessage"),
    "ContextAgent": ("agents.context_agent", "ContextAgent"),
    "PlannerAgent": ("agents.planner_agent", "PlannerAgent"),
    "SearchAgent": ("agents.search_agent", "SearchAgent"),
    "KnowledgeAgent": ("agents.knowledge_agent", "KnowledgeAgent"),
    "WriterAgent": ("agents.writer_agent", "WriterAgent"),
    "ReviewerAgent": ("agents.reviewer_agent", "ReviewerAgent"),
    "AgentOrchestrator": ("agents.orchestrator", "AgentOrchestrator"),
    "ContextPacket": ("agents.orchestrator", "ContextPacket"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - normal Python attribute contract
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
