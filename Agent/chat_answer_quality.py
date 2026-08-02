"""Planning and quality checks for knowledge QA answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ANSWER_INTENT_FACT_LOOKUP = "fact_lookup"
ANSWER_INTENT_FILE_DISCOVERY = "file_discovery"
ANSWER_INTENT_SPREADSHEET_FACT = "spreadsheet_fact"
ANSWER_INTENT_COMPARE_SUMMARIZE = "compare_summarize"
ANSWER_INTENT_PROCEDURE_HELP = "procedure_help"
ANSWER_INTENT_FOLLOWUP_REFINE = "followup_refine"
ANSWER_INTENT_OPEN_ENDED = "open_ended"

VALID_ANSWER_INTENTS = {
    ANSWER_INTENT_FACT_LOOKUP,
    ANSWER_INTENT_FILE_DISCOVERY,
    ANSWER_INTENT_SPREADSHEET_FACT,
    ANSWER_INTENT_COMPARE_SUMMARIZE,
    ANSWER_INTENT_PROCEDURE_HELP,
    ANSWER_INTENT_FOLLOWUP_REFINE,
    ANSWER_INTENT_OPEN_ENDED,
}

INTERNAL_COORDINATE_PATTERNS = (
    r"\[文档\d+\]",
    r"\bchunk\b",
    r"片段[:：]?\s*\d*",
    r"行号[:：]?\s*\d*",
    r"Sheet[:：]",
    r"页码[:：]?\s*\d*",
)


@dataclass
class AnswerPlan:
    answer_intent: str
    queries: list[str]
    expected_evidence: list[str]
    needs_clarification: bool = False
    clarification_question: str = ""
    strict_grounding: bool = True
    confidence: float = 0.6
    planner_source: str = "rules"
    normalized_task: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceReport:
    passed: bool
    score: float
    reason: str
    missing: list[str]
    source_count: int
    top_sources: list[str]


@dataclass
class VerificationReport:
    passed: bool
    issues: list[str] = field(default_factory=list)
    severe: bool = False
    sanitized_answer: str = ""


def build_answer_plan(message: str, conversation_context: str = "") -> AnswerPlan:
    """Classify the user's QA need and expand retrieval queries without LLMs."""
    text = _compact(message)
    raw_message = (message or "").strip()
    has_context = bool((conversation_context or "").strip())

    normalized_task = normalize_question(raw_message, conversation_context)
    if normalized_task:
        return AnswerPlan(
            answer_intent=normalized_task["answer_intent"],
            queries=normalized_task["queries"],
            expected_evidence=normalized_task["expected_evidence"],
            strict_grounding=True,
            confidence=normalized_task.get("confidence", 0.95),
            planner_source="normalizer",
            normalized_task=normalized_task,
        )

    weak_followup = _is_weak_followup(text)
    if weak_followup and not has_context:
        return AnswerPlan(
            answer_intent=ANSWER_INTENT_FOLLOWUP_REFINE,
            queries=[raw_message] if raw_message else [],
            expected_evidence=["上一轮可引用来源"],
            needs_clarification=True,
            clarification_question="我需要先确认你说的是哪份文件或哪一轮内容，请补充文件名、主题或重新贴一下相关材料。",
            confidence=0.86,
        )

    answer_intent = _classify_intent(text, weak_followup)
    expected_evidence = _expected_evidence_for(answer_intent)
    queries = _expand_queries(raw_message, answer_intent)

    return AnswerPlan(
        answer_intent=answer_intent,
        queries=queries,
        expected_evidence=expected_evidence,
        strict_grounding=answer_intent != ANSWER_INTENT_OPEN_ENDED,
        confidence=_rule_confidence(answer_intent, text),
    )


def should_use_llm_planner(message: str, rule_plan: AnswerPlan, mode: str = "auto") -> bool:
    """Decide whether the extra planner model call is worth the latency."""
    mode = (mode or "auto").strip().lower()
    if mode in {"rules", "rule", "off", "false", "0"}:
        return False
    if mode in {"llm", "deepseek", "on", "true", "1"}:
        return not rule_plan.needs_clarification
    if rule_plan.needs_clarification:
        return False
    if rule_plan.planner_source == "normalizer" and rule_plan.confidence >= 0.9:
        return False

    text = _compact(message)
    raw = message or ""
    complex_markers = (
        "结合", "同时", "分别", "并且", "以及", "梳理", "分析", "研判", "提炼",
        "形成", "根据", "围绕", "依据", "对策", "建议", "多个", "几份",
    )
    has_multiple_targets = sum(1 for marker in ("和", "与", "及", "、", "，", "；") if marker in raw) >= 2
    return (
        len(text) >= 60
        or rule_plan.answer_intent == ANSWER_INTENT_OPEN_ENDED
        or (rule_plan.answer_intent == ANSWER_INTENT_COMPARE_SUMMARIZE and len(text) >= 24)
        or has_multiple_targets
        or any(marker in text for marker in complex_markers)
    )


def build_llm_planner_prompt(message: str, conversation_context: str, fallback_plan: AnswerPlan) -> str:
    intents = "、".join(sorted(VALID_ANSWER_INTENTS))
    return f"""你是智能知识库平台的问答规划器。

请判断用户真正想做什么，并输出严格 JSON。不要输出 Markdown，不要解释。

可选 answer_intent 只能是：{intents}

字段要求：
{{
  "answer_intent": "上面的枚举之一",
  "queries": ["2到5条用于知识库检索的中文查询，第一条必须尽量保留用户原问题"],
  "expected_evidence": ["回答前必须核对的证据类型"],
  "needs_clarification": false,
  "clarification_question": "",
  "strict_grounding": true,
  "confidence": 0.0到1.0
}}

判断原则：
1. 如果用户要找文件或列材料，用 file_discovery。
2. 如果涉及金额、收费、标准、明细、清单、表格数字，用 spreadsheet_fact。
3. 如果要比较、归纳、总结多份材料，用 compare_summarize。
4. 如果问流程、网址、账号、路径、操作方法，用 procedure_help。
5. 如果问制度、办法、规定、依据、时间地点等事实，用 fact_lookup。
6. 如果明显依赖上一轮内容，用 followup_refine；但上下文不足时 needs_clarification=true。
7. 不确定时用 open_ended，但仍要给出可检索 queries。

规则初判：
{{
  "answer_intent": "{fallback_plan.answer_intent}",
  "queries": {fallback_plan.queries},
  "expected_evidence": {fallback_plan.expected_evidence},
  "needs_clarification": {str(fallback_plan.needs_clarification).lower()}
}}

对话上下文：
{(conversation_context or "")[:1800]}

用户问题：
{message}
"""


def parse_llm_answer_plan(text: str, fallback_plan: AnswerPlan, original_message: str) -> AnswerPlan:
    payload = _extract_json_object(text)
    if not isinstance(payload, dict):
        return fallback_plan

    answer_intent = str(payload.get("answer_intent") or fallback_plan.answer_intent).strip()
    if answer_intent not in VALID_ANSWER_INTENTS:
        answer_intent = fallback_plan.answer_intent

    queries = _normalize_queries(payload.get("queries"), original_message)
    if not queries:
        queries = fallback_plan.queries
    expected = _normalize_string_list(payload.get("expected_evidence")) or fallback_plan.expected_evidence

    try:
        confidence = float(payload.get("confidence", fallback_plan.confidence))
    except (TypeError, ValueError):
        confidence = fallback_plan.confidence
    confidence = max(0.0, min(confidence, 1.0))
    if confidence < 0.45:
        return fallback_plan

    needs_clarification = bool(payload.get("needs_clarification", False))
    clarification = str(payload.get("clarification_question") or "").strip()
    if needs_clarification and not clarification:
        clarification = fallback_plan.clarification_question or "我需要再确认一下你的具体需求，请补充文件名、主题或时间范围。"

    strict_grounding = bool(payload.get("strict_grounding", answer_intent != ANSWER_INTENT_OPEN_ENDED))
    return AnswerPlan(
        answer_intent=answer_intent,
        queries=queries[:5],
        expected_evidence=expected[:6],
        needs_clarification=needs_clarification,
        clarification_question=clarification,
        strict_grounding=strict_grounding,
        confidence=confidence,
        planner_source="llm",
        normalized_task=fallback_plan.normalized_task,
    )


def normalize_question(message: str, conversation_context: str = "") -> dict[str, Any]:
    """Map common equivalent user phrasings to stable structured tasks."""
    text = _compact(message)
    if not text:
        return {}

    storage_terms = (
        "nas",
        "网盘",
        "网络硬盘",
        "存储服务器",
        "文件服务器",
        "私有云",
        "共享盘",
        "共享文件",
        "虚拟盘",
    )
    storage_action_terms = (
        "怎么",
        "如何",
        "使用",
        "登录",
        "访问",
        "入口",
        "地址",
        "网址",
        "路径",
        "账号",
        "密码",
        "连接",
        "快捷方式",
        "里面",
    )
    if any(term in text for term in storage_terms) and any(term in text for term in storage_action_terms):
        canonical_query = (
            "示例单位存储服务器运营方案 NAS服务器 网盘 私有云 存储服务器 "
            "访问方式 访问地址 账号 密码 Windows+R 快捷方式"
        )
        queries = [
            canonical_query,
            "存储服务器 访问方式 访问地址 账号 密码",
            "NAS服务器 网盘 私有云 Windows+R 快捷方式",
        ]
        if message and message not in queries:
            queries.append(message)

        return {
            "task_type": "storage_server_usage",
            "answer_intent": ANSWER_INTENT_PROCEDURE_HELP,
            "entity": "示例单位存储服务器/NAS网盘",
            "target": "access_instructions",
            "fields": ["访问地址", "登录方式", "账号密码", "快捷方式", "使用规范"],
            "queries": queries[:5],
            "expected_evidence": ["存储服务器运营方案", "访问地址/路径", "账号密码或登录说明"],
            "canonical_query": canonical_query,
            "confidence": 0.96,
        }

    venue_source_terms = (
        "场地使用收费表",
        "场地收费表",
        "收费表",
        "收费标准",
        "场地费用",
        "场地收费",
        "教室收费",
        "教室价格",
        "教室费用",
        "会议室收费",
        "报告厅收费",
    )
    venue_target_terms = (
        "场地",
        "教室",
        "会议室",
        "报告厅",
        "贵宾厅",
        "智慧教室",
        "多功能",
        "房间",
        "门牌",
    )
    fee_terms = ("收费", "费用", "金额", "价格", "标准", "多少钱", "多少", "一天", "日租", "租")
    listing_terms = ("有哪些", "哪些", "什么", "内容", "列出", "清单", "明细", "分别", "怎么", "如何")

    has_source = any(term in text for term in venue_source_terms)
    has_target = any(term in text for term in venue_target_terms)
    has_fee = any(term in text for term in fee_terms)
    asks_listing = any(term in text for term in listing_terms)
    has_room_number = bool(re.search(r"\b\d{3,4}\b", message or ""))

    if (has_source and (has_fee or asks_listing or has_target)) or (has_target and has_fee) or (has_room_number and has_fee):
        target = "venue_fee"
        if any(term in text for term in ("公寓", "宿舍", "单间", "标间", "套间")):
            target = "apartment_fee"
        elif any(term in text for term in ("工位", "共享工作", "开放工位")):
            target = "workspace_fee"
        elif has_target or has_room_number:
            target = "classroom_fee"

        canonical_query = "场地使用收费表 教室 会议室 报告厅 贵宾厅 门牌号 可容纳人数 计费方式 金额 收费标准"
        if target == "apartment_fee":
            canonical_query = "场地使用收费表 公寓 单间 标间 套间 日租 月租 金额 收费标准"
        elif target == "workspace_fee":
            canonical_query = "场地使用收费表 共享工位 开放工位 工作室 日租 小时 金额 收费标准"

        queries = [
            canonical_query,
            "附件：场地使用收费表.xlsx 场地收费表 金额 计费方式",
            "场地收费表 门牌号 名称 可容纳人数 金额",
        ]
        if message and message not in queries:
            queries.append(message)

        return {
            "task_type": "spreadsheet_fee_lookup",
            "answer_intent": ANSWER_INTENT_SPREADSHEET_FACT,
            "entity": "场地使用收费表",
            "target": target,
            "fields": ["楼层", "门牌号", "名称", "可容纳人数", "计费方式", "金额", "备注"],
            "queries": queries[:5],
            "expected_evidence": ["场地使用收费表", "门牌号/名称字段", "计费方式/金额字段"],
            "canonical_query": canonical_query,
            "confidence": 0.96,
        }

    return {}


def evaluate_evidence(plan: AnswerPlan, results: list[dict[str, Any]], context: str) -> EvidenceReport:
    """Decide whether retrieved evidence is strong enough for grounded answering."""
    results = results or []
    top_sources = _source_names(results)
    source_count = len(top_sources)
    missing: list[str] = []
    score = 0.0

    if source_count == 0:
        if plan.answer_intent == ANSWER_INTENT_OPEN_ENDED:
            return EvidenceReport(
                passed=True,
                score=0.35,
                reason="开放式问题未命中知识库来源，将以通用建议回答并标明未引用本地资料",
                missing=[],
                source_count=0,
                top_sources=[],
            )
        return EvidenceReport(
            passed=False,
            score=0.0,
            reason="知识库没有命中可用来源",
            missing=plan.expected_evidence or ["相关知识库来源"],
            source_count=0,
            top_sources=[],
        )

    score += min(source_count, 4) * 0.12
    if _query_terms_covered(plan.queries, results, context):
        score += 0.34
    else:
        missing.append("与问题主体直接相关的正文或元数据")

    if plan.answer_intent == ANSWER_INTENT_SPREADSHEET_FACT:
        has_spreadsheet = any(r.get("source_type") == "spreadsheet" for r in results)
        has_numeric_context = bool(re.search(r"\d", context or ""))
        if has_spreadsheet or has_numeric_context:
            score += 0.34
        else:
            missing.append("表格、金额、标准或清单数据")
    elif plan.answer_intent == ANSWER_INTENT_FILE_DISCOVERY:
        score += 0.30
    elif plan.answer_intent == ANSWER_INTENT_COMPARE_SUMMARIZE:
        if source_count >= 2:
            score += 0.26
        else:
            missing.append("至少两个不同来源用于对比")
            score += 0.08
    elif plan.answer_intent in {ANSWER_INTENT_FACT_LOOKUP, ANSWER_INTENT_PROCEDURE_HELP}:
        if _has_filename_or_metadata_hit(results):
            score += 0.24
        else:
            missing.append("明确的文件名、章节或元数据命中")
    else:
        score += 0.16

    score = min(score, 1.0)
    passed = score >= _pass_threshold(plan.answer_intent)
    if plan.answer_intent == ANSWER_INTENT_FILE_DISCOVERY and source_count:
        passed = True

    reason = "检索证据足够支撑回答" if passed else "检索证据不足，直接回答可能产生猜测"
    return EvidenceReport(
        passed=passed,
        score=round(score, 2),
        reason=reason,
        missing=missing,
        source_count=source_count,
        top_sources=top_sources[:8],
    )


def build_evidence_fallback_answer(plan: AnswerPlan, report: EvidenceReport) -> str:
    if plan.needs_clarification:
        return plan.clarification_question
    if plan.answer_intent == ANSWER_INTENT_FILE_DISCOVERY and report.top_sources:
        names = "\n".join(f"- {name}" for name in report.top_sources)
        return f"我在知识库里找到这些可能相关的文件：\n{names}\n\n目前只适合先做文件定位，若需要我继续概括内容，请指定其中一份或几份。"
    if plan.answer_intent == ANSWER_INTENT_SPREADSHEET_FACT:
        return "当前未检索到足够的表格、金额或标准依据，暂不能可靠回答这个数值问题。请补充具体文件名、表格名称或时间范围。"
    missing = "、".join(report.missing[:3]) if report.missing else "关键依据"
    return f"目前知识库检索到的资料不足以支撑可靠回答，缺少：{missing}。请补充更具体的文件名、主题、时间范围或部门。"


def verify_answer(answer: str, plan: AnswerPlan, results: list[dict[str, Any]], context: str) -> VerificationReport:
    allowed_sources = set(_source_names(results))
    issues: list[str] = []
    severe = False

    unknown_sources = [
        name for name in _extract_filenames(answer)
        if not _filename_is_allowed(name, allowed_sources)
    ]
    if unknown_sources:
        severe = True
        issues.append(f"回答引用了未由知识库返回的来源：{'；'.join(unknown_sources[:3])}")

    internal_hits = [
        pattern for pattern in INTERNAL_COORDINATE_PATTERNS
        if re.search(pattern, answer or "", flags=re.IGNORECASE)
    ]
    sanitized = sanitize_answer(answer)
    if internal_hits:
        issues.append("回答包含内部检索坐标，已清理")

    answer_numbers = set(_extract_numbers(sanitized))
    context_numbers = set(_extract_numbers(context))
    unsupported_numbers = sorted(answer_numbers - context_numbers)
    if unsupported_numbers and context_numbers:
        issues.append(f"回答中存在未在证据中出现的数字：{'、'.join(unsupported_numbers[:5])}")

    if plan.strict_grounding and allowed_sources and not _mentions_allowed_source(sanitized, allowed_sources):
        issues.append("回答未明确引用已检索到的来源文件名")

    return VerificationReport(
        passed=not issues,
        issues=issues,
        severe=severe,
        sanitized_answer=sanitized,
    )


def sanitize_answer(answer: str) -> str:
    text = answer or ""
    for pattern in INTERNAL_COORDINATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"/[\w./\-\u4e00-\u9fff ]+\.(?:docx?|pdf|xlsx|csv)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_audit_summary(plan: AnswerPlan, evidence: EvidenceReport, verification: VerificationReport | None = None) -> dict:
    verification = verification or VerificationReport(passed=True, sanitized_answer="")
    return {
        "evidence_passed": evidence.passed,
        "evidence_score": evidence.score,
        "evidence_reason": evidence.reason,
        "missing": evidence.missing,
        "source_count": evidence.source_count,
        "top_sources": evidence.top_sources,
        "verifier_passed": verification.passed,
        "verifier_issues": verification.issues,
        "answer_intent": plan.answer_intent,
        "planner_source": plan.planner_source,
        "planner_confidence": plan.confidence,
    }


def _classify_intent(text: str, weak_followup: bool) -> str:
    if weak_followup:
        return ANSWER_INTENT_FOLLOWUP_REFINE
    if any(term in text for term in ("哪些文件", "相关文件", "有什么材料", "有哪些材料", "找一下", "帮我找", "文件列表")):
        return ANSWER_INTENT_FILE_DISCOVERY
    if any(term in text for term in ("金额", "费用", "收费", "标准", "价格", "清单", "明细", "总计", "合计", "多少")):
        return ANSWER_INTENT_SPREADSHEET_FACT
    if any(term in text for term in ("对比", "比较", "区别", "异同", "归纳", "总结", "提炼", "概括")):
        return ANSWER_INTENT_COMPARE_SUMMARIZE
    if any(term in text for term in ("怎么", "如何", "流程", "步骤", "入口", "网址", "账号", "路径", "命令", "操作")):
        return ANSWER_INTENT_PROCEDURE_HELP
    if any(term in text for term in ("制度", "办法", "规定", "是什么", "什么时候", "哪里", "哪份", "依据", "文件名")):
        return ANSWER_INTENT_FACT_LOOKUP
    return ANSWER_INTENT_OPEN_ENDED


def _rule_confidence(answer_intent: str, text: str) -> float:
    if answer_intent == ANSWER_INTENT_OPEN_ENDED:
        return 0.45
    if answer_intent == ANSWER_INTENT_FOLLOWUP_REFINE:
        return 0.72
    if len(text) >= 60:
        return 0.58
    return 0.78


def _expand_queries(message: str, answer_intent: str) -> list[str]:
    queries: list[str] = []
    if message:
        queries.append(message)

    extras = {
        ANSWER_INTENT_FILE_DISCOVERY: ["相关文件 材料 文件名", "主题 文件 清单"],
        ANSWER_INTENT_SPREADSHEET_FACT: ["收费 标准 金额 表格 明细", "费用 标准 清单 合计"],
        ANSWER_INTENT_COMPARE_SUMMARIZE: ["对比 比较 归纳 要点", "总结 差异 共性"],
        ANSWER_INTENT_PROCEDURE_HELP: ["流程 步骤 操作 网址 账号 路径", "办理 方法 注意事项"],
        ANSWER_INTENT_FACT_LOOKUP: ["制度 办法 规定 依据", "标准 要求 文件"],
        ANSWER_INTENT_FOLLOWUP_REFINE: ["上一轮 来源 文件 内容"],
        ANSWER_INTENT_OPEN_ENDED: ["相关资料 要点"],
    }.get(answer_intent, [])
    for query in extras:
        if query not in queries:
            queries.append(query)

    keywords = [w for w in re.split(r"[\s，。；、,.!?！？：:（）()]+", message or "") if len(w) >= 2]
    if keywords:
        keyword_query = " ".join(keywords[:6])
        if keyword_query and keyword_query not in queries:
            queries.append(keyword_query)

    return queries[:5]


def _expected_evidence_for(answer_intent: str) -> list[str]:
    return {
        ANSWER_INTENT_FILE_DISCOVERY: ["相关文件名或来源"],
        ANSWER_INTENT_SPREADSHEET_FACT: ["表格数据", "金额/标准/清单字段"],
        ANSWER_INTENT_COMPARE_SUMMARIZE: ["多个可对比来源"],
        ANSWER_INTENT_PROCEDURE_HELP: ["流程说明", "网址/账号/路径等操作信息"],
        ANSWER_INTENT_FACT_LOOKUP: ["制度、办法、规定或正文依据"],
        ANSWER_INTENT_FOLLOWUP_REFINE: ["上一轮上下文或来源"],
    }.get(answer_intent, ["相关知识库来源"])


def _is_weak_followup(text: str) -> bool:
    if not text:
        return False
    if any(term in text for term in ("怎么", "如何", "流程", "步骤", "登录", "网址", "账号", "路径", "制度", "标准")):
        return False
    markers = ("那个", "这个", "刚才", "上面", "上一轮", "上一条", "继续", "再说", "再改", "它")
    return len(text) <= 30 and any(marker in text for marker in markers)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()


def _source_names(results: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in results or []:
        filename = item.get("filename") or Path(str(item.get("source", ""))).name
        if filename and filename not in names:
            names.append(filename)
    return names


def _filename_alias_key(value: str) -> str:
    name = _clean_filename(value)
    name = re.sub(r"^(?:附件|附表|来源|文件|参考|根据)\s*[：:、-]*\s*", "", name)
    return _compact(name)


def _filename_alias_keys(value: str) -> set[str]:
    key = _filename_alias_key(value)
    keys = {key} if key else set()
    stem = re.sub(r"\.(?:docx?|pdf|xlsx|csv)$", "", key, flags=re.IGNORECASE)
    if stem:
        keys.add(stem)
    return keys


def _filename_is_allowed(name: str, allowed_sources: set[str] | list[str]) -> bool:
    key = _filename_alias_key(name)
    return bool(key) and any(key in _filename_alias_keys(source) for source in allowed_sources)


def _mentions_allowed_source(text: str, allowed_sources: set[str] | list[str]) -> bool:
    compact_text = _compact(text)
    for source in allowed_sources:
        exact_key = _compact(source)
        if exact_key and exact_key in compact_text:
            return True
        if any(alias_key and alias_key in compact_text for alias_key in _filename_alias_keys(source)):
            return True
    return False


def _extract_filenames(text: str) -> list[str]:
    pattern = r"[\w\u4e00-\u9fff（）()《》、，,\-—\s]+?\.(?:docx?|pdf|xlsx|csv)"
    names = []
    for match in re.findall(pattern, text or "", flags=re.IGNORECASE):
        name = _clean_filename(match)
        if name and name not in names:
            names.append(name)
    return names


def _clean_filename(value: str) -> str:
    name = (value or "").strip(" 　，,。；;：:《》")
    name = re.sub(r"^(参考|引用|来源|文件|根据)\s*", "", name)
    if " " in name:
        name = name.split()[-1]
    return name.strip(" 　，,。；;：:《》")


def _extract_numbers(text: str) -> list[str]:
    matches = re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+(?:\.\d+)?%?", text or "")
    return [match.replace(",", "") for match in matches]


def _extract_json_object(text: str) -> dict | None:
    import json

    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _normalize_queries(value: Any, original_message: str) -> list[str]:
    queries = _normalize_string_list(value)
    original = (original_message or "").strip()
    if original:
        queries = [original] + [q for q in queries if q != original]
    return list(dict.fromkeys(queries))[:5]


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[;\n；]+", value) if part.strip()]
    return []


def _query_terms_covered(queries: list[str], results: list[dict[str, Any]], context: str) -> bool:
    combined = _compact(context + "\n" + "\n".join(str(r.get("filename", "")) + str(r.get("text", "")) for r in results))
    tokens: list[str] = []
    for query in queries[:2]:
        tokens.extend([t for t in re.split(r"[\s，。；、,.!?！？：:（）()]+", query or "") if len(t) >= 2])
    if not tokens:
        return bool(combined)
    hits = sum(1 for token in tokens[:8] if _compact(token) in combined)
    return hits >= 1


def _has_filename_or_metadata_hit(results: list[dict[str, Any]]) -> bool:
    for item in results or []:
        if item.get("filename") or item.get("section_title") or item.get("heading_path") or item.get("category"):
            return True
    return False


def _pass_threshold(answer_intent: str) -> float:
    if answer_intent == ANSWER_INTENT_FILE_DISCOVERY:
        return 0.25
    if answer_intent == ANSWER_INTENT_COMPARE_SUMMARIZE:
        return 0.50
    if answer_intent == ANSWER_INTENT_SPREADSHEET_FACT:
        return 0.60
    return 0.50
