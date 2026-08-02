"""Shared state types for the LangGraph document subgraph."""

from __future__ import annotations

from typing import Any, TypedDict


class DocumentGraphState(TypedDict, total=False):
    user_request: str
    request_with_context: str
    previous_context: str
    ctx: Any
    document_content: str
    revision_round: int
    continue_revision: bool
    review_meta: dict
    reflection_meta: dict
