import json
import re
import logging
import os
from agents.base_agent import BaseAgent, AgentResult
from spreadsheet_auditor import SpreadsheetFactAuditor

logger = logging.getLogger(__name__)
SPREADSHEET_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge_base",
    "spreadsheets.sqlite",
)


class ReviewerAgent(BaseAgent):
    # 公文层级标题模式（仅匹配独立成行的标题）
    HEADING_PATTERNS = [
        (r'^[一二三四五六七八九十]、', '一级标题（一、）'),
        (r'^（[一二三四五六七八九十]+）', '二级标题（一）'),
        (r'^\d+\.', '三级标题（1.）'),
        (r'^\(\d+\)', '四级标题（(1)）'),
    ]

    def __init__(self, **kwargs):
        super().__init__(
            name="Reviewer",
            description="审查器 - 硬格式校验 + LLM内容审查，双重保障",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return """你是一个严格的公文审查专家。你的职责是：
1. 检查公文格式是否符合规范
2. 验证内容逻辑是否完整
3. 识别错误和遗漏
4. 提供修改建议
5. 给出置信度评分

审查维度：
- 格式规范：标题、主送单位、正文层次、落款等
- 内容完整性：是否有遗漏的关键部分
- 逻辑连贯性：各部分之间逻辑是否通顺
- 语言规范性：用词是否正式、准确
- 事实准确性：引用的政策、数据是否合理

你必须以 JSON 格式输出审查结果：
{
    "format_check": {
        "passed": true/false,
        "issues": ["问题1", "问题2"]
    },
    "content_check": {
        "passed": true/false,
        "issues": ["问题1", "问题2"]
    },
    "logic_check": {
        "passed": true/false,
        "issues": ["问题1", "问题2"]
    },
    "language_check": {
        "passed": true/false,
        "issues": ["问题1", "问题2"]
    },
    "fact_check": {
        "passed": true/false,
        "issues": ["问题1", "问题2"]
    },
    "suggestions": ["建议1", "建议2"],
    "confidence": 0.85,
    "needs_revision": true/false,
    "revision_focus": ["需要修改的重点1"]
}

只输出 JSON，不要其他内容。"""

    def _hard_format_check(self, text: str) -> dict:
        """硬格式校验 — 正则检测，不依赖 LLM"""
        issues = []
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        if not lines:
            return {"passed": False, "issues": ["文档为空"]}

        # 1. 标题检测：首行应为简短标题
        title = lines[0]
        if len(title) > 80 or len(title) < 4:
            issues.append(f"标题长度异常（{len(title)}字），建议4-80字")
        if any(kw in title for kw in ['关于', '建议', '报告', '方案', '通知', '请示']):
            pass  # 标题含公文关键词，加分项
        else:
            issues.append("标题未检测到公文关键词（如'关于''建议''报告'等）")

        # 2. 层级标题规范：检测跳跃和格式
        heading_levels_seen = []
        for line in lines[1:]:
            for level, (pattern, name) in enumerate(self.HEADING_PATTERNS):
                if re.match(pattern, line):
                    heading_levels_seen.append((level, line[:20]))
                    break

        # 检查层级是否跳跃（一、→ 1. 跳过（一））
        prev_level = -1
        for level, _ in heading_levels_seen:
            if level > prev_level + 1:
                patterns_skipped = [p[1] for p in self.HEADING_PATTERNS[prev_level + 1:level]]
                issues.append(f"标题层级跳跃，缺少{'/'.join(patterns_skipped)}")
            prev_level = level

        # 3. 落款检测
        last_lines = '\n'.join(lines[-5:])
        has_org = bool(re.search(r'示例高校|单位|单位|部门', last_lines))
        has_date = bool(re.search(r'\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日', last_lines))
        if not has_org and not has_date:
            issues.append("文末未检测到落款单位或日期，建议补充")

        # 4. 正文长度
        body_chars = sum(len(l) for l in lines)
        if body_chars < 100:
            issues.append(f"正文过短（{body_chars}字），公文至少需100字")

        # 5. 正文缩进/段落
        if len(lines) < 3:
            issues.append("段落数不足，公文至少需要标题+正文+落款三部分")

        passed = len(issues) == 0
        return {"passed": passed, "issues": issues}

    def process(self, input_data: dict, on_think=None) -> AgentResult:
        document_content = input_data.get("document_content", "")
        document_type = input_data.get("document_type", "通用公文")
        task_type = input_data.get("task_type", "公文生成")
        user_request = input_data.get("user_request", "")
        source_filenames = input_data.get("source_filenames", []) or []
        key_points = input_data.get("key_points", []) or []
        context_analysis = input_data.get("context_analysis", {}) or {}
        memory_context = input_data.get("memory_context", "") or ""
        user_constraints = input_data.get("user_constraints", []) or []
        evidence_items = input_data.get("evidence_items", []) or []

        if not document_content:
            return AgentResult(
                success=False,
                content="无内容可审查",
                agent_name=self.name,
                confidence=0.0,
            )

        # 先执行硬格式检查（不依赖 LLM）
        self._emit_think(on_think, "🔍", "正在执行硬格式校验...")
        hard_check = self._hard_format_check(document_content)
        if hard_check["issues"]:
            self._emit_think(on_think, "⚠️", f"硬检查发现{len(hard_check['issues'])}个问题")

        citation_issues = self._check_source_citations(document_content, source_filenames)
        spreadsheet_audit = self._audit_spreadsheet_facts(document_content, evidence_items)
        spreadsheet_issues = spreadsheet_audit.get("issues", [])
        if spreadsheet_audit.get("spreadsheet_evidence_count", 0):
            if spreadsheet_issues:
                self._emit_think(on_think, "⚠️", "报表数值精确校验未通过")
            else:
                self._emit_think(on_think, "✅", "报表数值精确校验通过")

        fact_issues = citation_issues + spreadsheet_issues
        if hard_check["issues"] or fact_issues:
            review = self._rule_review(
                hard_check,
                fact_issues,
                needs_revision=True,
                spreadsheet_audit=spreadsheet_audit,
            )
            self._emit_think(on_think, "⚡", "规则检查已发现明确问题，跳过AI审查")
            return AgentResult(
                success=True,
                content=json.dumps(review, ensure_ascii=False),
                agent_name=self.name,
                confidence=review["confidence"],
                metadata=review,
            )

        if not self._should_run_llm_review(document_content, task_type, key_points, user_constraints, evidence_items):
            review = self._rule_review(
                hard_check,
                [],
                needs_revision=False,
                spreadsheet_audit=spreadsheet_audit,
            )
            self._emit_think(on_think, "✅", "规则检查通过，跳过AI审查")
            return AgentResult(
                success=True,
                content=json.dumps(review, ensure_ascii=False),
                agent_name=self.name,
                confidence=review["confidence"],
                metadata=review,
            )

        # LLM 内容审查
        self._emit_think(on_think, "🤖", "正在AI内容审查...")

        prompt = f"""请审查以下{document_type}公文：

用户需求：{user_request}

上下文意图：{context_analysis.get("user_intent", "") if isinstance(context_analysis, dict) else ""}

写作要点：{"；".join(key_points[:8])}

用户约束：{"；".join(user_constraints[:8])}

已召回的历史上下文：
{memory_context or "无"}

证据来源：{self._format_evidence_items(evidence_items)}

公文内容：
{document_content}

请从格式规范、内容完整性、逻辑连贯性、语言规范性、事实引用可信度五个维度进行审查。
如果公文内容与用户约束、写作要点或证据来源冲突，必须指出。

事实边界：
1. 不得建议补入用户未提供、证据来源未明确支持的具体事实，例如具体楼栋、门牌号、讲师身份、调休/加班政策、附件、课件、报名或签到方式、统计数据。
2. 如果用户只写“会议室”，不得建议改成“A栋3楼301室”等具体地点；只能建议“请补充具体会议室”或“具体地点另行通知”。
3. “如……”示例只能说明方向，不能作为 revision_focus 中要求 Writer 写入正文的确定事实。
4. 用户明确要求“简短通知”时，不要为了凑长度强行要求背景、议程或论证；只检查必要要素是否齐全。"""

        try:
            response_text = self.call_llm(prompt, temperature=0.3)
            review = self._parse_json_response(response_text)
        except Exception:
            response_text = json.dumps(hard_check, ensure_ascii=False)
            review = {
                "format_check": {"passed": True, "issues": []},
                "content_check": {"passed": True, "issues": []},
                "logic_check": {"passed": True, "issues": []},
                "language_check": {"passed": True, "issues": []},
                "suggestions": [],
                "confidence": 0.7,
                "needs_revision": False,
                "revision_focus": [],
            }

        # 合并硬检查结果：硬检查发现的问题补充到 LLM 审查中
        fc = review.get("format_check", {})
        if not isinstance(fc, dict):
            fc = {"passed": True, "issues": []}
        llm_format_issues = fc.get("issues", []) or []
        fc["issues"] = list(set(hard_check["issues"] + llm_format_issues))
        fc["passed"] = len(fc["issues"]) == 0
        review["format_check"] = fc

        # 来源引用硬校验：如果上下文提供了来源，输出内容不能虚构其它文件名
        fact_check = review.get("fact_check", {})
        if not isinstance(fact_check, dict):
            fact_check = {"passed": True, "issues": []}
        fact_issues = fact_check.get("issues", []) or []
        fact_check["issues"] = list(dict.fromkeys(fact_issues + citation_issues + spreadsheet_issues))
        fact_check["passed"] = len(fact_check["issues"]) == 0
        review["fact_check"] = fact_check
        review["spreadsheet_audit"] = spreadsheet_audit

        # 硬检查不通过则至少触发格式修改建议
        if hard_check["issues"] and not review.get("needs_revision"):
            review["needs_revision"] = True
            review["revision_focus"] = (review.get("revision_focus") or []) + hard_check["issues"][:2]

        if fact_issues and not review.get("needs_revision"):
            review["needs_revision"] = True
            review["revision_focus"] = (review.get("revision_focus") or []) + fact_issues[:2]

        format_passed = review.get("format_check", {}).get("passed", True)
        content_passed = review.get("content_check", {}).get("passed", True)
        needs_revision = review.get("needs_revision", False)
        confidence = review.get("confidence", 0.7)

        if format_passed and content_passed:
            self._emit_think(on_think, "✅", "格式检查通过")
        else:
            issues = review.get("format_check", {}).get("issues", []) + review.get(
                "content_check", {}
            ).get("issues", [])
            self._emit_think(
                on_think, "⚠️", f"发现问题：{'；'.join(issues[:3])}"
            )

        if needs_revision:
            focus = review.get("revision_focus", [])
            self._emit_think(
                on_think, "🔄", f"建议修改：{'；'.join(focus[:3])}"
            )
        else:
            self._emit_think(
                on_think, "✅", f"审查通过，置信度 {int(confidence * 100)}%"
            )

        return AgentResult(
            success=True,
            content=response_text,
            agent_name=self.name,
            confidence=confidence,
            metadata=review,
        )

    def _should_run_llm_review(
        self,
        document_content: str,
        task_type: str,
        key_points: list,
        user_constraints: list,
        evidence_items: list,
    ) -> bool:
        """只在复杂或高风险任务上触发 LLM 审查。"""
        high_risk_tasks = {"公文生成", "材料改写", "续写修改"}
        if task_type in high_risk_tasks:
            return True
        if len(document_content) > 1500:
            return True
        if len(key_points or []) >= 4 or len(user_constraints or []) >= 3:
            return True
        if evidence_items:
            return True
        return False

    def _rule_review(self, hard_check: dict, fact_issues: list, needs_revision: bool,
                     spreadsheet_audit: dict = None) -> dict:
        format_issues = hard_check.get("issues", []) or []
        fact_issues = fact_issues or []
        revision_focus = list(dict.fromkeys(format_issues[:2] + fact_issues[:2]))
        passed = not format_issues and not fact_issues
        return {
            "format_check": {"passed": not format_issues, "issues": format_issues},
            "content_check": {"passed": True, "issues": []},
            "logic_check": {"passed": True, "issues": []},
            "language_check": {"passed": True, "issues": []},
            "fact_check": {"passed": not fact_issues, "issues": fact_issues},
            "suggestions": revision_focus,
            "confidence": 0.82 if passed else 0.62,
            "needs_revision": needs_revision,
            "revision_focus": revision_focus,
            "review_mode": "rule",
            "spreadsheet_audit": spreadsheet_audit or SpreadsheetFactAuditor._empty_result(),
        }

    def _parse_json_response(self, text: str) -> dict:
        """解析 LLM 返回的 JSON 响应，带错误处理"""

        text = text.strip()

        # 去除 markdown 代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            start_idx = 0
            end_idx = len(lines) - 1

            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    if start_idx == 0:
                        start_idx = i + 1
                    else:
                        end_idx = i
                        break

            text = "\n".join(lines[start_idx:end_idx])

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"审查结果 JSON 解析失败: {e}")
            # 尝试修复常见的 JSON 格式问题
            try:
                text = text.rstrip().rstrip(',').strip()
                return json.loads(text)
            except json.JSONDecodeError:
                logger.error("JSON 修复失败，返回默认审查结果")
                return {
                    "format_check": {"passed": True, "issues": []},
                    "content_check": {"passed": True, "issues": []},
                    "logic_check": {"passed": True, "issues": []},
                    "language_check": {"passed": True, "issues": []},
                    "fact_check": {"passed": True, "issues": []},
                    "suggestions": [],
                    "confidence": 0.7,
                    "needs_revision": False,
                    "revision_focus": [],
                }

    def _check_source_citations(self, document_content: str, source_filenames: list) -> list:
        """检查参考来源是否来自知识库返回的文件列表。"""
        if not source_filenames:
            return []

        allowed = {str(name).strip() for name in source_filenames if str(name).strip()}
        if not allowed:
            return []

        cited = set(re.findall(r'《([^》]+\.(?:docx?|pdf|xlsx|csv))》', document_content))
        cited.update(re.findall(r'来源[:：]\s*([^\s，,；;]+?\.(?:docx?|pdf|xlsx|csv))', document_content))

        unknown = sorted(name for name in cited if name not in allowed)
        if unknown:
            return [f"存在未由知识库返回的引用来源：{'; '.join(unknown[:3])}"]
        return []

    def _audit_spreadsheet_facts(self, document_content: str, evidence_items: list) -> dict:
        auditor = SpreadsheetFactAuditor(SPREADSHEET_DB_PATH)
        try:
            return auditor.audit(document_content, evidence_items)
        except Exception as exc:
            logger.warning("报表数值校验失败: %s", exc)
            return {
                "passed": False,
                "issues": [f"报表数值校验过程失败：{str(exc)[:120]}"],
                "verified_claims": [],
                "unverified_claims": [],
                "spreadsheet_evidence_count": 0,
            }

    def _format_evidence_items(self, evidence_items: list) -> str:
        if not evidence_items:
            return "无结构化证据清单"
        lines = []
        for item in evidence_items[:10]:
            title = item.get("title") or item.get("source") or "未命名来源"
            source_type = item.get("type", "source")
            source = item.get("source", "")
            source_text = f"（{source}）" if source else ""
            location = ""
            if item.get("source_type") == "spreadsheet":
                sheet = item.get("sheet_name") or "未知Sheet"
                row_start = item.get("row_start")
                row_end = item.get("row_end") or row_start
                if row_start and row_end and row_start != row_end:
                    location = f"，Sheet：{sheet}，行：{row_start}-{row_end}"
                elif row_start:
                    location = f"，Sheet：{sheet}，行：{row_start}"
            lines.append(f"[{source_type}] {title}{location}{source_text}")
        return "；".join(lines)
