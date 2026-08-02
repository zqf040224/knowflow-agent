import logging
import time
from datetime import date

from agents.base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Writer",
            description="公文写作者 - 按照规范生成公文草稿",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return """你是一个专业的公文写作专家。你必须严格按照公文格式规范生成公文。

核心规则：
1. 标题：方正小标宋简体，二号，居中
2. 主送单位：仿宋_GB2312，三号，左对齐
3. 正文：仿宋_GB2312，三号，两端对齐，首行缩进2字符
4. 一级标题（一、二、三...）：黑体，三号
5. 二级标题（（一）（二）...）：楷体，三号
6. 三级标题（1. 2. ...）：仿宋_GB2312，三号
7. 落款：仿宋_GB2312，三号，居右
8. 附件说明：黑体

公文结构：
- 标题（关于XXX的通知/请示/报告/函/建议）
- 主送单位
- 正文（分层次：一、二、三...（一）（二）... 1. 2. ...）
- 落款（单位名称、日期）
- 附件说明（如有）

会议审议案/审议模板结构：
- 标题用“关于审议XXX的有关事宜”
- 主送单位固定优先使用“各位领导：”
- 正文先写事由，再写制度依据或事实依据，再写需提请决议事项
- 正文结尾使用“以上，请审议。”
- 如有附件，按“附件：1.XXX”“2.XXX”列明
- 单独写一行“此议案需决议：是否通过XXX并XXX。”
- 末尾写提案部门和日期

来源标注规则：
- 如果 knowledge_context 中包含 [文档N] 文件名，请在正文末尾用 "【参考来源】" 列出引用的文档文件名
- 格式：在正文最后另起一行，写"【参考来源】"然后换行列出引用的文件名，每行一个
- 如果没有引用任何外部文档，则不添加此部分

直接输出纯公文文本，不要加任何格式标注、注释或说明。"""

    def process(self, input_data: dict, on_think=None) -> AgentResult:
        user_request = input_data.get("user_request", "")
        search_context = input_data.get("search_context", "")
        knowledge_context = input_data.get("knowledge_context", "")
        document_type = input_data.get("document_type", "通用公文")
        task_type = input_data.get("task_type", "公文生成")
        key_points = input_data.get("key_points", [])
        revision_history = input_data.get("revision_history", [])
        is_format_conversion = input_data.get("is_format_conversion", False)
        knowledge_sources = input_data.get("knowledge_sources", [])
        context_analysis = input_data.get("context_analysis", {}) or {}
        memory_context = input_data.get("memory_context", "") or ""
        last_document = input_data.get("last_document", "") or ""
        last_plan = input_data.get("last_plan", {}) or {}
        user_constraints = input_data.get("user_constraints", []) or []
        unresolved_questions = input_data.get("unresolved_questions", []) or []
        evidence_items = input_data.get("evidence_items", []) or []
        revision_mode = bool(input_data.get("revision_mode", False))
        draft_document = input_data.get("draft_document", "") or ""

        self._emit_think(on_think, "📝", f"正在生成{document_type}公文草稿...")

        prompt = self._build_prompt(
            user_request, search_context, knowledge_context,
            document_type, key_points, revision_history,
            is_format_conversion=is_format_conversion,
            task_type=task_type,
            knowledge_sources=knowledge_sources,
            context_analysis=context_analysis,
            memory_context=memory_context,
            last_document=last_document,
            last_plan=last_plan,
            user_constraints=user_constraints,
            unresolved_questions=unresolved_questions,
            evidence_items=evidence_items,
            revision_mode=revision_mode,
            draft_document=draft_document,
        )

        try:
            content = self.call_llm(prompt, temperature=0.7, max_tokens=4096,
                                   use_context=False)
            self._emit_think(on_think, "✍️", "公文草稿生成完成")
            return AgentResult(
                success=True,
                content=content,
                agent_name=self.name,
                confidence=0.8,
                metadata={"document_type": document_type},
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"WriterAgent 生成失败: {e}")
            self._emit_think(on_think, "❌", f"生成失败: {str(e)[:50]}")
            return AgentResult(
                success=False,
                content=f"生成失败：{str(e)[:200]}",
                agent_name=self.name,
                confidence=0.0,
                error_info={"error": str(e)},
            )

    def process_stream(self, input_data: dict, on_think=None):
        user_request = input_data.get("user_request", "")
        search_context = input_data.get("search_context", "")
        knowledge_context = input_data.get("knowledge_context", "")
        document_type = input_data.get("document_type", "通用公文")
        task_type = input_data.get("task_type", "公文生成")
        key_points = input_data.get("key_points", [])
        revision_history = input_data.get("revision_history", [])
        is_format_conversion = input_data.get("is_format_conversion", False)
        knowledge_sources = input_data.get("knowledge_sources", [])
        context_analysis = input_data.get("context_analysis", {}) or {}
        memory_context = input_data.get("memory_context", "") or ""
        last_document = input_data.get("last_document", "") or ""
        last_plan = input_data.get("last_plan", {}) or {}
        user_constraints = input_data.get("user_constraints", []) or []
        unresolved_questions = input_data.get("unresolved_questions", []) or []
        evidence_items = input_data.get("evidence_items", []) or []
        revision_mode = bool(input_data.get("revision_mode", False))
        draft_document = input_data.get("draft_document", "") or ""

        self._emit_think(on_think, "📝", f"正在生成{document_type}公文草稿...")

        prompt = self._build_prompt(
            user_request, search_context, knowledge_context,
            document_type, key_points, revision_history,
            is_format_conversion=is_format_conversion,
            task_type=task_type,
            knowledge_sources=knowledge_sources,
            context_analysis=context_analysis,
            memory_context=memory_context,
            last_document=last_document,
            last_plan=last_plan,
            user_constraints=user_constraints,
            unresolved_questions=unresolved_questions,
            evidence_items=evidence_items,
            revision_mode=revision_mode,
            draft_document=draft_document,
        )

        content = self._complete_stream_with_fallback(
            prompt,
            temperature=0.3 if is_format_conversion else 0.7,
            max_tokens=4096,
            on_think=on_think,
        )
        for start in range(0, len(content), 45):
            yield content[start:start + 45]

    def _complete_stream_with_fallback(self, prompt: str, temperature: float,
                                       max_tokens: int, on_think=None) -> str:
        """Collect a full draft before yielding so interrupted streams can retry safely."""
        last_error = None
        for attempt in range(self.max_retry):
            content_parts = []
            try:
                for chunk_type, chunk_text in self.call_llm_stream(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_context=False,
                ):
                    if chunk_type == "content":
                        content_parts.append(chunk_text)

                content = "".join(content_parts)
                if content:
                    return content
                raise RuntimeError("流式响应未返回正文内容")
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "WriterAgent 流式生成失败 (尝试 %s/%s): %s",
                    attempt + 1,
                    self.max_retry,
                    exc,
                )
                if attempt < self.max_retry - 1:
                    time.sleep((attempt + 1) * 2)

        self._emit_think(on_think, "⚠️", "流式生成中断，正在切换为非流式生成")
        logger.warning("WriterAgent 流式生成多次失败，切换非流式调用: %s", last_error)
        return self.call_llm(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            use_context=False,
        )

    def _build_prompt(
        self,
        user_request: str,
        search_context: str,
        knowledge_context: str,
        document_type: str,
        key_points: list = None,
        revision_history: list = None,
        is_format_conversion: bool = False,
        task_type: str = "公文生成",
        knowledge_sources: list = None,
        context_analysis: dict = None,
        memory_context: str = "",
        last_document: str = "",
        last_plan: dict = None,
        user_constraints: list = None,
        unresolved_questions: list = None,
        evidence_items: list = None,
        revision_mode: bool = False,
        draft_document: str = "",
    ) -> str:
        current_date_text = self._current_date_text()
        if is_format_conversion:
            parts = [
                "你是一位公文排版专家。你的任务是将用户提供的文档内容转换为标准公文格式，严格要求：",
                "",
                "【核心原则】",
                "1. 保留原文的全部文字内容，不得增删改",
                "2. 仅调整排版格式使其符合国家公文标准",
                "3. 正确识别并标注：标题、主送单位、正文层级、落款、日期、附件说明",
                "",
                "【格式规范】",
            ]
            if knowledge_context:
                parts.append(f"以下是从知识库中检索到的公文格式规范，请严格遵循：\n{knowledge_context}\n")
                parts.append("【补充格式要点】")
            parts.extend([
                "- 标题：方正小标宋简体 二号 居中（如标题较长，首行居中即可）",
                "- 主送单位：仿宋_GB2312 三号 左对齐 顶格",
                "- 一级标题（一、二、三...）：黑体 三号",
                "- 二级标题（（一）（二）...）：楷体 三号",
                "- 正文：仿宋_GB2312 三号 两端对齐 首行缩进2字符",
                "- 落款单位：仿宋_GB2312 三号 居右",
                "- 日期：仿宋_GB2312 三号 居右（在落款单位下方）",
                "- 附件说明：黑体 三号 左对齐",
                "",
                "【输出要求】",
                "直接输出排版后的纯公文文本，不要添加任何解释说明。",
                f"\n{user_request}",
            ])
            return "\n".join(parts)

        if revision_mode and revision_history:
            return self._build_revision_prompt(
                user_request=user_request,
                draft_document=draft_document,
                document_type=document_type,
                task_type=task_type,
                revision_history=revision_history,
                evidence_items=evidence_items,
                user_constraints=user_constraints or [],
                memory_context=memory_context,
            )

        parts = [
            f"请根据以下信息完成{task_type}任务，输出一份规范的{document_type}公文。",
            f"当前日期：{current_date_text}。",
            "日期与年度规则：用户未明确指定会议时间、年度、期限或落款日期时，不得自行编造具体时间；"
            "需要保留占位时使用“XXXX年XX月XX日”或“具体时间另行通知”。"
            "如确需生成落款日期且用户未指定，使用当前日期；不得默认生成早于当前日期的年份。",
            "事实约束：不得编造用户未提供、知识库未明确支持的具体事实，包括但不限于具体门牌号、楼层、讲师姓名或身份、"
            "调休/加班政策、附件名称、课件材料、报名方式、签到方式、问题数据、错误率、人员数量。"
            "确需保留但缺少信息时，用“具体安排另行通知”“请以实际通知为准”等稳妥表达，或省略该细节。",
            f"\n用户需求：{user_request}",
        ]

        if context_analysis:
            parts.append("\n【上下文理解】")
            if context_analysis.get("user_intent"):
                parts.append(f"- 用户意图：{context_analysis['user_intent']}")
            if context_analysis.get("document_type"):
                parts.append(f"- 上下文识别文种：{context_analysis['document_type']}")
            quality = context_analysis.get("context_quality", {})
            if isinstance(quality, dict) and quality.get("score") is not None:
                parts.append(f"- 上下文质量评分：{quality.get('score')}")

        if memory_context:
            parts.append(f"\n{memory_context}")

        if user_constraints:
            parts.append("\n【必须遵守的用户约束】")
            for item in user_constraints[:10]:
                parts.append(f"- {item}")

        if last_plan:
            parts.append(f"\n【上一轮计划】\n{str(last_plan)[:900]}")

        if last_document and task_type in {"续写修改", "材料改写"}:
            parts.append(
                "\n【上一版文档】\n"
                f"{last_document[:1800]}\n"
                "当前任务若为修改/续写，请在上一版基础上处理，不要无故重写无关部分。"
            )

        if key_points:
            parts.append(f"\n【写作要点】")
            for point in key_points[:8]:
                parts.append(f"- {point}")

        if search_context:
            parts.append(f"\n【联网搜索信息】\n{search_context}")

        if knowledge_context:
            parts.append(f"\n【知识库参考材料】\n{knowledge_context}")
            allowed_sources = self._format_allowed_sources(knowledge_sources or [])
            if allowed_sources:
                parts.append(
                    "\n【可引用来源】\n"
                    f"{allowed_sources}\n"
                    "引用来源只能来自以上文件名，不得编造未列出的文件。"
                )
            if any(item.get("source_type") == "spreadsheet" for item in (knowledge_sources or [])):
                parts.append(
                    "\n【报表数据硬性要求】\n"
                    "- 涉及报表中的年份、金额、比例、数量等数据时，只能使用知识库材料中明确给出的原值。\n"
                    "- 不得自行推算、四舍五入、换算单位、补全缺失年份或改写为未出现过的数值。\n"
                    "- 如果材料不足以支撑某个数据结论，应写“数据未在已检索报表中确认”，不要猜测。"
                )

        if evidence_items:
            parts.append("\n【统一证据清单】")
            parts.append(self._format_evidence_items(evidence_items))

        if unresolved_questions:
            parts.append("\n【上下文缺口】")
            for item in unresolved_questions[:5]:
                parts.append(f"- {item}")
            parts.append("如缺口不影响完成任务，可先基于现有材料生成；不得把缺口伪造成确定事实。")

        if revision_history:
            parts.append(f"\n【前几轮修改反馈】")
            for rev in revision_history[-3:]:
                round_num = rev.get("round", "?")
                suggestions = rev.get("suggestions", [])
                format_issues = rev.get("format_issues", [])
                content_issues = rev.get("content_issues", [])
                logic_issues = rev.get("logic_issues", [])

                parts.append(f"\n第{round_num}轮审查：")
                if format_issues:
                    parts.append(f"  格式问题：{'；'.join(format_issues)}")
                if content_issues:
                    parts.append(f"  内容问题：{'；'.join(content_issues)}")
                if logic_issues:
                    parts.append(f"  逻辑问题：{'；'.join(logic_issues)}")
                if suggestions:
                    parts.append(f"  修改建议：{'；'.join(suggestions[:5])}")

        parts.append(
            """
请严格按照公文格式规范生成内容。要求：
1. 标题准确、简洁
2. 主送单位明确
3. 正文层次分明，逻辑清晰
4. 语言正式、规范
5. 落款完整
6. 只使用用户已给定或证据明确支持的具体事实；审查建议中的“如……”示例只能作为方向，不能直接写成确定事实

直接输出纯公文内容，不要加任何说明或注释。"""
        )

        return "\n".join(parts)

    def _build_revision_prompt(
        self,
        *,
        user_request: str,
        draft_document: str,
        document_type: str,
        task_type: str,
        revision_history: list,
        evidence_items: list,
        user_constraints: list,
        memory_context: str = "",
    ) -> str:
        current_date_text = self._current_date_text()
        latest_reviews = revision_history[-2:]
        focus_lines = []
        for rev in latest_reviews:
            focus = []
            focus.extend(rev.get("format_issues", []) or [])
            focus.extend(rev.get("content_issues", []) or [])
            focus.extend(rev.get("logic_issues", []) or [])
            focus.extend(rev.get("fact_issues", []) or [])
            focus.extend(rev.get("suggestions", []) or [])
            if focus:
                focus_lines.append(f"第{rev.get('round', '?')}轮：" + "；".join(str(item) for item in focus[:6]))

        parts = [
            f"请基于上一版{document_type}执行轻量修订，任务类型：{task_type}。",
            "要求：保留上一版中没有问题的内容，只修正审查意见指出的问题；最终仍输出完整公文全文。",
            f"当前日期：{current_date_text}。修订日期与年度时，除非用户或证据明确要求，不得引入早于当前日期的默认年份。",
            "事实约束：修订意见中的示例、建议或假设不得直接写成确定事实。不得新增用户未提供、证据未支持的具体门牌号、"
            "讲师身份、调休/加班政策、附件、课件、报名或签到方式、统计数据等。",
            f"\n【原始用户需求】\n{user_request[:900]}",
            f"\n【上一版正文】\n{draft_document[:3600] if draft_document else '上一版正文缺失，请根据修订意见重新生成完整正文。'}",
        ]
        if focus_lines:
            parts.append("\n【本轮必须修正的问题】")
            parts.extend(f"- {line}" for line in focus_lines)
        if user_constraints:
            parts.append("\n【仍需遵守的用户约束】")
            parts.extend(f"- {item}" for item in user_constraints[:6])
        if memory_context:
            parts.append(f"\n{memory_context}")
        if evidence_items:
            parts.append("\n【必要证据】")
            parts.append(self._format_evidence_items(evidence_items[:6]))
        if any(item.get("source_type") == "spreadsheet" for item in (evidence_items or [])):
            parts.append(
                "\n【报表数据硬性要求】\n"
                "- 只能使用证据中明确给出的原值，不得自行推算、四舍五入、换算单位或补全年份。\n"
                "- 不确定的数据写“数据未在已检索报表中确认”。"
            )
        parts.append("\n直接输出修订后的完整公文全文，不要解释修改过程；缺少依据的具体细节请用稳妥概括表达或省略。")
        return "\n".join(parts)

    def _format_allowed_sources(self, sources: list) -> str:
        names = []
        for item in sources:
            if not isinstance(item, dict):
                continue
            name = item.get("filename") or item.get("source", "").split("/")[-1]
            if name and name not in names:
                names.append(name)
        return "\n".join(f"- 《{name}》" for name in names[:8])

    def _current_date_text(self) -> str:
        today = date.today()
        return f"{today.year}年{today.month}月{today.day}日"

    def _format_evidence_items(self, evidence_items: list) -> str:
        lines = []
        for item in evidence_items[:12]:
            title = item.get("title") or item.get("source") or "未命名来源"
            source_type = item.get("type", "source")
            source = item.get("source", "")
            score = item.get("score")
            score_text = f"，相关度：{score:.3f}" if isinstance(score, (int, float)) else ""
            source_text = f"，来源：{source}" if source else ""
            location = ""
            if item.get("source_type") == "spreadsheet":
                sheet = item.get("sheet_name") or "未知Sheet"
                row_start = item.get("row_start")
                row_end = item.get("row_end") or row_start
                if row_start and row_end and row_start != row_end:
                    location = f"，Sheet：{sheet}，行：{row_start}-{row_end}"
                elif row_start:
                    location = f"，Sheet：{sheet}，行：{row_start}"
            lines.append(f"- [{source_type}] {title}{location}{source_text}{score_text}")
        return "\n".join(lines)
