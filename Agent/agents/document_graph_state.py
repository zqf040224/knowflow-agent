"""Serializable state and runtime context for the document subgraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict


class DocumentGraphState(TypedDict, total=False):
    """Checkpoint-safe document workflow state.

    Only JSON-like values belong here. Runtime services and callbacks are
    injected through ``DocumentGraphRuntime`` and are never checkpointed.
    """

    input_user_request: str
    run_id: str
    session_id: str
    user_id: str
    display_message: str
    parent_step_index: int
    request_with_context: str
    previous_context: str

    user_request: str
    context_analysis: dict
    plan: dict
    search_context: str
    knowledge_context: str
    knowledge_sources: list
    search_sources: list
    evidence_items: list
    compact_evidence: list
    revision_history: list
    run_records: list
    last_document: str
    last_plan: dict
    user_constraints: list
    unresolved_questions: list
    user_profile: Optional[dict]
    memory_context: str
    audit_summary: dict
    think_log: list

    document_content: str
    revision_round: int
    continue_revision: bool
    review_meta: dict
    reflection_meta: dict
    reflection_done: bool
    should_reflect: bool
    quality_status: str
    run_status: str
    error_step: str
    error_message: str
    errors: list


@dataclass(frozen=True)
class DocumentGraphRuntime:
    """Per-invocation services that must stay outside graph state."""

    orchestrator: object
    think_handler: Callable[[str, str, str], None]
    streaming: bool = False
    hydrated_request: Optional[str] = None
    hydrated_previous_context: Optional[str] = None
    runtime_snapshot: Optional[dict] = None
    redaction_fragments: tuple[str, ...] = ()

    def resolve_document_request(self, state: DocumentGraphState) -> str:
        if self.hydrated_request is not None:
            return str(self.hydrated_request)
        return str(
            state.get("request_with_context")
            or state.get("input_user_request")
            or ""
        )

    def resolve_document_previous_context(self, state: DocumentGraphState) -> str:
        if self.hydrated_previous_context is not None:
            return str(self.hydrated_previous_context)
        return str(state.get("previous_context") or "")

    def resolve_document_runtime_snapshot(
        self,
        _state: DocumentGraphState,
    ) -> dict:
        return dict(self.runtime_snapshot or {})

    def sanitize_document_update(
        self,
        _state: DocumentGraphState,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep hydrated attachment fragments out of standalone checkpoints."""

        fragments = tuple(fragment for fragment in self.redaction_fragments if fragment)
        if not fragments:
            return update

        def redact(value: Any) -> Any:
            if isinstance(value, str):
                sanitized = value
                for fragment in fragments:
                    sanitized = sanitized.replace(
                        fragment,
                        "[附件正文见受控引用]",
                    )
                return sanitized
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {
                    (redact(key) if isinstance(key, str) else key): redact(item)
                    for key, item in value.items()
                }
            return value

        return {
            field_name: (
                value if field_name == "document_content" else redact(value)
            )
            for field_name, value in update.items()
        }
