"""
Natural-language spreadsheet transformation.

The model translates an instruction into a small operation spec. This module
keeps execution deterministic: filters, sorting, and export are applied by code.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import cmp_to_key
from typing import Any, Dict, Iterable, List, Optional

from spreadsheet_export import safe_filename_stem, workbook_to_bytes
from spreadsheet_store import SpreadsheetRow, parse_spreadsheet


FILTER_OPERATORS = {
    "eq",
    "neq",
    "contains",
    "not_contains",
    "gt",
    "gte",
    "lt",
    "lte",
    "empty",
    "not_empty",
}


@dataclass
class FilterSpec:
    column: str
    operator: str
    value: Any = ""


@dataclass
class SortSpec:
    column: str
    direction: str = "asc"


@dataclass
class OperationSpec:
    filters: List[FilterSpec] = field(default_factory=list)
    sorts: List[SortSpec] = field(default_factory=list)
    limit: Optional[int] = None
    output_sheet_name: str = "处理结果"
    explanation: str = ""
    used_ai: bool = False

    def to_dict(self) -> Dict:
        return {
            "filters": [item.__dict__ for item in self.filters],
            "sorts": [item.__dict__ for item in self.sorts],
            "limit": self.limit,
            "output_sheet_name": self.output_sheet_name,
            "explanation": self.explanation,
            "used_ai": self.used_ai,
        }


def collect_headers(rows: Iterable[SpreadsheetRow]) -> List[str]:
    headers = []
    for row in rows:
        for header in row.headers or row.values.keys():
            if header not in headers:
                headers.append(header)
    return headers


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", _normalize_text(value)).lower()


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None

    number = float(match.group())
    suffix = text[match.end(): match.end() + 2]
    if "亿" in suffix:
        number *= 100000000
    elif "万" in suffix:
        number *= 10000
    return number


def _parse_date(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value

    text = _normalize_text(value)
    candidates = [
        (r"((?:19|20)\d{2})[-/\.](\d{1,2})[-/\.](\d{1,2})", "%Y-%m-%d"),
        (r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日?", "%Y-%m-%d"),
    ]
    for pattern, fmt in candidates:
        match = re.search(pattern, text)
        if not match:
            continue
        normalized = "-".join(match.groups())
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            return None
    return None


def _resolve_column(column: str, headers: List[str]) -> Optional[str]:
    wanted = _compact(column)
    if not wanted:
        return None

    for header in headers:
        if _compact(header) == wanted:
            return header
    for header in headers:
        if wanted in _compact(header) or _compact(header) in wanted:
            return header
    return None


def _coerce_filter(item: Dict, headers: List[str]) -> Optional[FilterSpec]:
    column = _resolve_column(item.get("column", ""), headers)
    operator = _normalize_text(item.get("operator", "")).lower()
    if not column or operator not in FILTER_OPERATORS:
        return None
    return FilterSpec(column=column, operator=operator, value=item.get("value", ""))


def _coerce_sort(item: Dict, headers: List[str]) -> Optional[SortSpec]:
    column = _resolve_column(item.get("column", ""), headers)
    direction = _normalize_text(item.get("direction", "asc")).lower()
    if direction in {"descending", "descend", "down", "降序"}:
        direction = "desc"
    if direction in {"ascending", "ascend", "up", "升序"}:
        direction = "asc"
    if not column or direction not in {"asc", "desc"}:
        return None
    return SortSpec(column=column, direction=direction)


def operation_spec_from_dict(raw: Dict, headers: List[str], *, used_ai: bool = False) -> OperationSpec:
    filters = []
    for item in raw.get("filters", []) or []:
        if isinstance(item, dict):
            coerced = _coerce_filter(item, headers)
            if coerced:
                filters.append(coerced)

    sorts = []
    for item in raw.get("sorts", []) or []:
        if isinstance(item, dict):
            coerced = _coerce_sort(item, headers)
            if coerced:
                sorts.append(coerced)

    try:
        limit = int(raw.get("limit")) if raw.get("limit") not in (None, "") else None
    except (TypeError, ValueError):
        limit = None
    if limit is not None:
        limit = max(1, min(limit, 10000))

    output_sheet_name = _normalize_text(raw.get("output_sheet_name")) or "处理结果"
    return OperationSpec(
        filters=filters,
        sorts=sorts,
        limit=limit,
        output_sheet_name=output_sheet_name[:31],
        explanation=_normalize_text(raw.get("explanation")),
        used_ai=used_ai,
    )


def infer_operation_spec(instruction: str, headers: List[str]) -> OperationSpec:
    """Small deterministic fallback for common Chinese spreadsheet instructions."""
    text = _normalize_text(instruction)
    filters: List[FilterSpec] = []
    sorts: List[SortSpec] = []

    for header in headers:
        escaped = re.escape(header)
        patterns = [
            (rf"{escaped}\s*(?:大于等于|不少于|至少)\s*([^\s，。；,;]+)", "gte"),
            (rf"{escaped}\s*(?:大于|超过|高于)\s*([^\s，。；,;]+)", "gt"),
            (rf"{escaped}\s*(?:小于等于|不超过|至多)\s*([^\s，。；,;]+)", "lte"),
            (rf"{escaped}\s*(?:小于|低于|少于)\s*([^\s，。；,;]+)", "lt"),
            (rf"{escaped}\s*(?:等于|为|是)\s*([^\s，。；,;]+)", "eq"),
            (rf"{escaped}\s*(?:包含|含有)\s*([^\s，。；,;]+)", "contains"),
        ]
        for pattern, operator in patterns:
            match = re.search(pattern, text)
            if match:
                filters.append(FilterSpec(column=header, operator=operator, value=match.group(1)))

        compact_header = _compact(header)
        if compact_header and compact_header in _compact(text):
            if re.search(rf"{escaped}[^，。；,;]*(?:从高到低|降序|倒序|由大到小|由高到低)", text):
                sorts.append(SortSpec(column=header, direction="desc"))
            elif re.search(rf"{escaped}[^，。；,;]*(?:从低到高|升序|正序|由小到大|由低到高|由早到晚)", text):
                sorts.append(SortSpec(column=header, direction="asc"))
            elif re.search(rf"(?:按|根据|依照)\s*{escaped}\s*(?:排序|排列|分组)", text):
                sorts.append(SortSpec(column=header, direction="asc"))

    if not sorts:
        for header in headers:
            if _compact(header) in _compact(text) and re.search(r"从高到低|降序|倒序|由大到小|由高到低", text):
                sorts.append(SortSpec(column=header, direction="desc"))
                break
            if _compact(header) in _compact(text) and re.search(r"从低到高|升序|正序|由小到大|由低到高|由早到晚", text):
                sorts.append(SortSpec(column=header, direction="asc"))
                break

    limit = None
    limit_match = re.search(r"(?:前|只要|保留)\s*(\d+)\s*(?:条|行|名|个)?", text)
    if limit_match:
        limit = int(limit_match.group(1))

    return OperationSpec(
        filters=filters,
        sorts=sorts,
        limit=limit,
        explanation="根据规则关键词自动解析" if (filters or sorts or limit) else "",
    )


def build_operation_spec_with_ai(instruction: str, headers: List[str], client, model: str) -> OperationSpec:
    prompt = f"""你是 Excel 数据处理规划器。请把用户自然语言需求转换成 JSON 操作规格。

可用列名：
{json.dumps(headers, ensure_ascii=False)}

用户需求：
{instruction}

只输出 JSON，不要 Markdown。字段：
{{
  "filters": [{{"column": "列名", "operator": "eq|neq|contains|not_contains|gt|gte|lt|lte|empty|not_empty", "value": "值"}}],
  "sorts": [{{"column": "列名", "direction": "asc|desc"}}],
  "limit": 数字或 null,
  "output_sheet_name": "处理结果",
  "explanation": "一句中文说明"
}}

要求：
1. column 必须使用可用列名里的原文。
2. 不确定的条件不要编造，留空对应数组。
3. “从高到低/降序”用 desc，“从低到高/升序”用 asc。
4. “放一起/分组”可以转成对应列的 asc 排序。"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你只返回可解析 JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=700,
    )
    content = response.choices[0].message.content or "{}"
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    return operation_spec_from_dict(json.loads(content), headers, used_ai=True)


def _row_matches(row: SpreadsheetRow, filters: List[FilterSpec]) -> bool:
    for item in filters:
        raw_value = row.values.get(item.column)
        text_value = _normalize_text(raw_value)
        expected = _normalize_text(item.value)

        if item.operator == "empty" and text_value:
            return False
        if item.operator == "not_empty" and not text_value:
            return False
        if item.operator == "eq" and _compact(text_value) != _compact(expected):
            return False
        if item.operator == "neq" and _compact(text_value) == _compact(expected):
            return False
        if item.operator == "contains" and _compact(expected) not in _compact(text_value):
            return False
        if item.operator == "not_contains" and _compact(expected) in _compact(text_value):
            return False
        if item.operator in {"gt", "gte", "lt", "lte"}:
            left = _parse_number(raw_value)
            right = _parse_number(expected)
            if left is None or right is None:
                return False
            if item.operator == "gt" and not left > right:
                return False
            if item.operator == "gte" and not left >= right:
                return False
            if item.operator == "lt" and not left < right:
                return False
            if item.operator == "lte" and not left <= right:
                return False
    return True


def _compare_values(left: Any, right: Any) -> int:
    left_date = _parse_date(left)
    right_date = _parse_date(right)
    if left_date is not None and right_date is not None:
        return (left_date > right_date) - (left_date < right_date)

    left_num = _parse_number(left)
    right_num = _parse_number(right)
    if left_num is not None and right_num is not None:
        return (left_num > right_num) - (left_num < right_num)

    left_text = _normalize_text(left)
    right_text = _normalize_text(right)
    return (left_text > right_text) - (left_text < right_text)


def apply_operation(rows: List[SpreadsheetRow], spec: OperationSpec) -> List[SpreadsheetRow]:
    result = [row for row in rows if _row_matches(row, spec.filters)]

    if spec.sorts:
        def compare(left: SpreadsheetRow, right: SpreadsheetRow) -> int:
            for sort in spec.sorts:
                outcome = _compare_values(left.values.get(sort.column), right.values.get(sort.column))
                if outcome:
                    return -outcome if sort.direction == "desc" else outcome
            return (left.row_number > right.row_number) - (left.row_number < right.row_number)

        result = sorted(result, key=cmp_to_key(compare))

    if spec.limit:
        result = result[:spec.limit]
    return result


def workbook_from_transformed_rows(
    rows: List[SpreadsheetRow],
    headers: List[str],
    spec: OperationSpec,
    *,
    original_filename: str,
    original_count: int,
):
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError("缺少 openpyxl，无法导出 Excel 文件，请先安装依赖") from exc

    from spreadsheet_export import _excel_safe_text, _style_sheet  # internal styling helper shared by exports

    wb = Workbook()
    ws = wb.active
    ws.title = (spec.output_sheet_name or "处理结果")[:31]
    ws.append(["来源Sheet", "原始行号", *headers])
    for row in rows:
        ws.append([
            row.sheet_name,
            row.row_number,
            *[_excel_safe_text(row.values.get(header)) for header in headers],
        ])
    _style_sheet(ws)

    info = wb.create_sheet("处理说明")
    info.append(["项目", "内容"])
    info.append(["原文件", original_filename])
    info.append(["导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    info.append(["原始有效行数", original_count])
    info.append(["导出行数", len(rows)])
    info.append(["AI解析", "是" if spec.used_ai else "否"])
    info.append(["筛选条件", json.dumps([item.__dict__ for item in spec.filters], ensure_ascii=False)])
    info.append(["排序规则", json.dumps([item.__dict__ for item in spec.sorts], ensure_ascii=False)])
    info.append(["数量限制", spec.limit or "无"])
    info.append(["说明", spec.explanation or ""])
    _style_sheet(info)
    return wb


def transform_spreadsheet_file(
    file_path: str,
    filename: str,
    instruction: str,
    *,
    client=None,
    model: str = "deepseek-v4-flash",
) -> Dict[str, Any]:
    rows = parse_spreadsheet(file_path, display_filename=filename)
    if not rows:
        return {
            "success": False,
            "message": "未解析到可处理的表格数据",
        }

    headers = collect_headers(rows)
    spec = None
    ai_error = ""
    if client is not None:
        try:
            spec = build_operation_spec_with_ai(instruction, headers, client, model)
        except Exception as exc:
            ai_error = str(exc)[:160]

    fallback_spec = infer_operation_spec(instruction, headers)
    if spec is None or (not spec.filters and not spec.sorts and not spec.limit):
        spec = fallback_spec
    elif fallback_spec.limit and not spec.limit:
        spec.limit = fallback_spec.limit

    if not spec.filters and not spec.sorts and not spec.limit:
        return {
            "success": False,
            "message": "没有识别到明确的筛选、排序或数量限制规则，请补充列名和条件",
            "headers": headers,
            "ai_error": ai_error,
        }

    transformed = apply_operation(rows, spec)
    workbook = workbook_from_transformed_rows(
        transformed,
        headers,
        spec,
        original_filename=filename,
        original_count=len(rows),
    )
    output_name = f"{safe_filename_stem(filename, '表格')}_处理结果.xlsx"
    return {
        "success": True,
        "filename": output_name,
        "content": workbook_to_bytes(workbook),
        "summary": {
            "original_count": len(rows),
            "output_count": len(transformed),
            "headers": headers,
            "operation": spec.to_dict(),
            "ai_error": ai_error,
        },
    }
