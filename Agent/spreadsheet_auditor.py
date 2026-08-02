"""
Spreadsheet fact auditing.

This module checks numeric claims in generated text against structured
spreadsheet evidence. It is intentionally rule-based: spreadsheet data should
not rely on another model pass for factual verification.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from spreadsheet_store import SpreadsheetStore


DATA_UNIT_PATTERN = (
    r"亿元|万元|元|万人次|人次|万人|人|百分比|百分点|%|％|项|个|家|倍|"
    r"平方米|㎡|亩|吨|亿元以上|亿元左右"
)


class SpreadsheetFactAuditor:
    """Rule-based verifier for spreadsheet-backed numeric facts."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else None

    def audit(self, document_content: str, evidence_items: List[Dict]) -> Dict:
        spreadsheet_refs = [
            item for item in evidence_items or []
            if item.get("source_type") == "spreadsheet" and item.get("content_hash")
        ]
        if not spreadsheet_refs:
            return self._empty_result()

        trusted_values = self._collect_trusted_values(spreadsheet_refs)
        if not trusted_values:
            return self._empty_result()

        claims = self._extract_numeric_claims(document_content)
        unverified = []
        verified = []
        for claim in claims:
            if self._claim_is_supported(claim, trusted_values):
                verified.append(claim)
            else:
                unverified.append(claim)

        issues = []
        if unverified:
            sample = "；".join(unverified[:5])
            issues.append(
                f"报表数据校验未通过，以下数值未在命中表格行中找到精确依据：{sample}"
            )

        return {
            "passed": not issues,
            "issues": issues,
            "verified_claims": verified,
            "unverified_claims": unverified,
            "spreadsheet_evidence_count": len(spreadsheet_refs),
        }

    def _collect_trusted_values(self, spreadsheet_refs: List[Dict]) -> List[str]:
        values = []
        if self.db_path and self.db_path.exists():
            store = SpreadsheetStore(self.db_path)
            for item in spreadsheet_refs:
                rows = store.get_rows_by_source(
                    item["content_hash"],
                    sheet_name=item.get("sheet_name") or None,
                    row_start=item.get("row_start"),
                    row_end=item.get("row_end") or item.get("row_start"),
                )
                for row in rows:
                    values.extend(self._values_from_row(row))

        for item in spreadsheet_refs:
            for row_values in item.get("spreadsheet_values", []) or []:
                if isinstance(row_values, dict):
                    values.extend(str(v) for v in row_values.values() if str(v).strip())

        normalized = []
        seen = set()
        for value in values:
            for variant in self._value_variants(str(value)):
                if variant and variant not in seen:
                    seen.add(variant)
                    normalized.append(variant)
        return normalized

    @staticmethod
    def _values_from_row(row: Dict) -> Iterable[str]:
        values = row.get("values", {}) or {}
        for value in values.values():
            text = str(value).strip()
            if text:
                yield text
        row_text = str(row.get("row_text", "")).strip()
        if row_text:
            yield row_text

    def _extract_numeric_claims(self, document_content: str) -> List[str]:
        claims = []
        seen = set()
        for line in document_content.splitlines():
            if self._line_is_signature_date(line):
                continue

            patterns = [
                rf"(?<![\d.])\d+(?:,\d{{3}})*(?:\.\d+)?\s*(?:{DATA_UNIT_PATTERN})",
                r"(?<![\d.])(?:19|20)\d{2}\s*年?",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, line):
                    claim = match.group(0).strip()
                    if self._looks_like_heading_number(line, claim):
                        continue
                    normalized = self._normalize(claim)
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        claims.append(claim)
        return claims

    @staticmethod
    def _line_is_signature_date(line: str) -> bool:
        text = line.strip()
        return bool(re.fullmatch(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日", text))

    @staticmethod
    def _looks_like_heading_number(line: str, claim: str) -> bool:
        stripped = line.strip()
        return stripped.startswith(claim + ".") or stripped.startswith(claim + "、")

    def _claim_is_supported(self, claim: str, trusted_values: List[str]) -> bool:
        claim_variants = self._value_variants(claim)
        for claim_variant in claim_variants:
            for trusted in trusted_values:
                if claim_variant == trusted or claim_variant in trusted:
                    return True
        return False

    def _value_variants(self, value: str) -> List[str]:
        normalized = self._normalize(value)
        variants = [normalized]
        year_match = re.fullmatch(r"((?:19|20)\d{2})年?", normalized)
        if year_match:
            variants.append(year_match.group(1))
        numeric_match = re.search(r"\d+(?:\.\d+)?", normalized)
        if numeric_match:
            variants.append(numeric_match.group(0))
        return list(dict.fromkeys(v for v in variants if v))

    @staticmethod
    def _normalize(value: str) -> str:
        return (
            str(value)
            .replace("，", ",")
            .replace("％", "%")
            .replace(",", "")
            .replace(" ", "")
            .replace("\t", "")
            .strip()
        )

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "passed": True,
            "issues": [],
            "verified_claims": [],
            "unverified_claims": [],
            "spreadsheet_evidence_count": 0,
        }
