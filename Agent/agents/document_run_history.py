"""Shared revision-history record builders for document generation."""

from __future__ import annotations


def review_history_entry(review_meta: dict, revision_round: int) -> dict:
    return {
        "round": revision_round + 1,
        "needs_revision": review_meta.get("needs_revision", False),
        "revision_focus": review_meta.get("revision_focus", []),
        "suggestions": review_meta.get("suggestions", []),
        "format_issues": review_meta.get("format_check", {}).get("issues", []),
        "content_issues": review_meta.get("content_check", {}).get("issues", []),
        "logic_issues": review_meta.get("logic_check", {}).get("issues", []),
        "language_issues": review_meta.get("language_check", {}).get("issues", []),
        "fact_issues": review_meta.get("fact_check", {}).get("issues", []),
        "spreadsheet_audit": review_meta.get("spreadsheet_audit", {}),
        "confidence": review_meta.get("confidence", 0.8),
    }


def reflection_history_entry(reflection_meta: dict, revision_round: int) -> dict:
    return {
        "round": revision_round + 1,
        "needs_revision": reflection_meta.get("needs_revision", False),
        "revision_focus": reflection_meta.get("revision_suggestions", []),
        "suggestions": reflection_meta.get("revision_suggestions", []),
        "format_issues": [],
        "content_issues": reflection_meta.get("weaknesses", []),
        "logic_issues": reflection_meta.get("counter_arguments", []),
        "language_issues": [],
        "confidence": reflection_meta.get("logic_score", 0.8),
        "source": "reflection",
    }
