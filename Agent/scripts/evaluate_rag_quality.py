#!/usr/bin/env python3
"""Offline rule-based quality report for RAG answer samples.

The script intentionally avoids heavy eval dependencies. It can read hand-made
JSON cases and low-rated beta feedback, then reports simple grounding checks.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


INTERNAL_PATTERNS = (
    r"\[文档\d+\]",
    r"\bchunk\b",
    r"片段[:：]?\s*\d*",
    r"行号[:：]?\s*\d*",
    r"Sheet[:：]",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG answer quality with lightweight rules.")
    parser.add_argument("--cases", default="", help="JSON file with eval cases.")
    parser.add_argument("--feedback-db", default="agent_memory.db", help="SQLite DB containing beta_feedback.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    cases = []
    if args.cases:
        cases.extend(load_json_cases(Path(args.cases)))
    cases.extend(load_feedback_cases(Path(args.feedback_db), args.limit))

    report = build_report(cases)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
    return 0


def load_json_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    return [normalize_case(item, source="json") for item in data if isinstance(item, dict)]


def load_feedback_cases(db_path: Path, limit: int) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, category, rating, content, context_json, created_at
            FROM beta_feedback
            WHERE rating BETWEEN 1 AND 3 OR category IN ('bug', 'answer', 'generation')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    cases = []
    for row in rows:
        context = parse_json(row["context_json"], {})
        cases.append(normalize_case({
            "id": f"feedback-{row['id']}",
            "question": context.get("message") or context.get("question") or row["content"],
            "answer": context.get("answer", ""),
            "expected_sources": context.get("source_filenames") or context.get("expected_sources") or [],
            "actual_sources": context.get("source_filenames") or [],
            "intent": context.get("intent", ""),
            "rating": row["rating"],
            "notes": row["content"],
            "created_at": row["created_at"],
        }, source="feedback"))
    return cases


def normalize_case(item: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "id": str(item.get("id") or item.get("name") or f"{source}-{abs(hash(json.dumps(item, ensure_ascii=False, sort_keys=True))) % 100000}"),
        "source": source,
        "question": str(item.get("question") or item.get("input") or ""),
        "answer": str(item.get("answer") or item.get("actual_answer") or ""),
        "intent": str(item.get("intent") or ""),
        "expected_sources": as_list(item.get("expected_sources")),
        "actual_sources": as_list(item.get("actual_sources") or item.get("source_filenames")),
        "answer_requirements": as_list(item.get("answer_requirements")),
        "notes": str(item.get("notes") or ""),
    }


def build_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [evaluate_case(case) for case in cases]
    runnable = [item for item in evaluated if item["has_answer"]]
    passed = [item for item in runnable if item["passed"]]
    return {
        "summary": {
            "case_count": len(evaluated),
            "answered_case_count": len(runnable),
            "passed_count": len(passed),
            "pass_rate": round(len(passed) / len(runnable), 3) if runnable else 0,
        },
        "cases": evaluated,
    }


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    answer = case["answer"]
    issues = []
    if not answer:
        issues.append("缺少 answer，作为待补充样例保留")
    if contains_internal_coordinates(answer):
        issues.append("回答泄露内部检索坐标")
    unknown = unknown_sources(answer, case["actual_sources"] or case["expected_sources"])
    if unknown:
        issues.append("回答包含未知来源：" + "；".join(unknown[:3]))
    missing_expected = [
        src for src in case["expected_sources"]
        if answer and src not in answer and src not in case["actual_sources"]
    ]
    if missing_expected:
        issues.append("未命中期望来源：" + "；".join(missing_expected[:3]))

    return {
        **case,
        "has_answer": bool(answer),
        "passed": bool(answer) and not issues,
        "issues": issues,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# RAG Quality Report",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Answered cases: {summary['answered_case_count']}",
        f"- Passed: {summary['passed_count']}",
        f"- Pass rate: {summary['pass_rate']}",
        "",
        "## Issues",
    ]
    issue_cases = [case for case in report["cases"] if case["issues"]]
    if not issue_cases:
        lines.append("- No issues found.")
    for case in issue_cases[:50]:
        lines.append(f"- `{case['id']}` ({case['intent'] or 'unknown'}): {'；'.join(case['issues'])}")
    return "\n".join(lines)


def contains_internal_coordinates(answer: str) -> bool:
    return any(re.search(pattern, answer or "", flags=re.IGNORECASE) for pattern in INTERNAL_PATTERNS)


def unknown_sources(answer: str, allowed_sources: list[str]) -> list[str]:
    allowed = set(allowed_sources or [])
    if not allowed:
        return []
    filenames = re.findall(r"[\w\u4e00-\u9fff（）()《》、，,\-—\s]+?\.(?:docx?|pdf|xlsx|csv)", answer or "", flags=re.IGNORECASE)
    unknown = []
    for filename in filenames:
        name = filename.strip(" 　，,。；;：:《》")
        if name and name not in allowed and name not in unknown:
            unknown.append(name)
    return unknown


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[;,；、\n]+", value) if part.strip()]
    return [str(value)]


def parse_json(value: str, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
