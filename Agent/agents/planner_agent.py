import json
import logging
from agents.base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Planner",
            description="任务规划器 - 分析用户需求、拆解任务步骤、制定执行计划",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return """你是一个公文写作任务规划器。你的职责是：
1. 分析用户需求，判断公文类型（通知、请示、报告、函、对策建议、情况反映等）
2. 判断是否需要联网搜索最新信息（涉及最新政策、数据、动态时需要联网）
3. 拆解知识库检索关键词，聚焦内容而非格式

你必须以 JSON 格式输出规划结果，格式如下：
{
    "task_type": "问答检索|公文生成|材料改写|格式转换|续写修改",
    "document_type": "公文类型",
    "need_web_search": true/false,
    "search_queries": ["搜索关键词1", "搜索关键词2"],
    "knowledge_queries": ["内容检索关键词1", "内容检索关键词2"],
    "plan_steps": [
        {"step": 1, "agent": "Search", "action": "搜索XXX最新政策"},
        {"step": 2, "agent": "Knowledge", "action": "检索XXX相关内容"},
        {"step": 3, "agent": "Writer", "action": "生成XXX公文草稿"},
        {"step": 4, "agent": "Reviewer", "action": "审查校验"}
    ],
    "key_points": ["要点1", "要点2"],
    "confidence": 0.85
}

判断是否需要联网搜索的规则：
- 涉及最新政策、法规、数据 → 需要联网
- 涉及时事热点、行业动态 → 需要联网
- 纯模板、格式参考 → 不需要联网
- 涉及具体人名、机构最新信息 → 需要联网

注意：knowledge_queries 应专注于内容检索（政策背景、参考范文等），不要检索公文格式规范（格式由系统内置处理）。

任务类型判断规则：
- 问答检索：用户主要是在问知识库、文件内容、政策依据或历史材料
- 公文生成：用户要求起草一份新的通知、报告、请示、函、建议等
- 材料改写：用户要求润色、改写、压缩、扩写已有材料
- 格式转换：用户上传材料后要求套用公文格式或规范排版
- 续写修改：用户基于上一轮结果要求继续修改、补充、重写某部分

只输出 JSON，不要其他内容。"""

    def process(self, input_data: dict, on_think=None) -> AgentResult:
        user_request = input_data.get("user_request", "")
        context_analysis = input_data.get("context_analysis", {}) or {}
        last_plan = input_data.get("last_plan", {}) or {}
        last_document = input_data.get("last_document", "") or ""
        user_constraints = input_data.get("user_constraints", []) or []
        memory_context = input_data.get("memory_context", "") or ""
        if not user_request:
            return AgentResult(
                success=False,
                content="",
                agent_name=self.name,
                confidence=0.0,
            )

        self._emit_think(on_think, "🤔", "正在分析您的需求...")

        prompt_parts = [f"请分析以下公文写作需求并制定执行计划：\n\n{user_request}"]
        if memory_context:
            prompt_parts.append(f"\n{memory_context}")
        if context_analysis:
            prompt_parts.append("\n【上下文分析】")
            prompt_parts.append(json.dumps({
                "user_intent": context_analysis.get("user_intent", ""),
                "document_type": context_analysis.get("document_type", ""),
                "key_points": context_analysis.get("key_points", []),
                "context_quality": context_analysis.get("context_quality", {}),
            }, ensure_ascii=False))
        if user_constraints:
            prompt_parts.append("\n【用户约束】\n" + "\n".join(f"- {item}" for item in user_constraints[:8]))
        if last_plan:
            prompt_parts.append("\n【上一轮计划】\n" + json.dumps(last_plan, ensure_ascii=False)[:1000])
        if last_document:
            prompt_parts.append("\n【上一版文档摘录】\n" + last_document[:1200])
            prompt_parts.append("如果当前需求包含继续、上一版、修改、补充等表达，优先判断为续写修改或材料改写。")

        prompt = "\n".join(prompt_parts)

        try:
            response_text = self.call_llm(prompt, temperature=0.3)
            self._emit_think(on_think, "📊", "需求分析完成，正在制定执行计划...")

            plan = self._parse_json_response(response_text)
            plan = self._normalize_plan(plan, user_request)
            doc_type = plan.get("document_type", "未识别")
            need_search = plan.get("need_web_search", False)
            task_type = plan.get("task_type", "公文生成")
            self._emit_think(
                on_think,
                "📋",
                f"任务类型：{task_type} / {doc_type}，{'需要' if need_search else '不需要'}联网搜索",
            )

            return AgentResult(
                success=True,
                content=response_text,
                agent_name=self.name,
                confidence=plan.get("confidence", 0.8),
                metadata=plan,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"PlannerAgent 降级: {e}")
            default_plan = {
                "task_type": "公文生成",
                "document_type": "通用公文",
                "need_web_search": False,
                "search_queries": [],
                "knowledge_queries": [user_request],
                "plan_steps": [
                    {"step": 1, "agent": "Knowledge", "action": "检索格式规范"},
                    {"step": 2, "agent": "Writer", "action": "生成公文草稿"},
                    {"step": 3, "agent": "Reviewer", "action": "审查校验"},
                ],
                "key_points": [],
                "confidence": 0.6,
            }
            return AgentResult(
                success=True,
                content=json.dumps(default_plan, ensure_ascii=False),
                agent_name=self.name,
                confidence=0.6,
                metadata=default_plan,
            )

    def process_with_context(self, input_data: dict, on_think=None) -> AgentResult:
        """一次 LLM 调用同时完成上下文分析和任务规划，减少 Agent 链路延迟。"""
        user_request = input_data.get("user_request", "")
        conversation_history = input_data.get("conversation_history", []) or []
        user_profile = input_data.get("user_profile", {}) or {}
        memory_context = input_data.get("memory_context", "") or ""
        previous_context = input_data.get("previous_context", {}) or {}

        if hasattr(user_profile, "to_dict"):
            user_profile = user_profile.to_dict()

        if not user_request:
            metadata = {
                "context_analysis": self._fallback_context(user_request, user_profile),
                "plan": self._normalize_plan({}, user_request),
            }
            return AgentResult(
                success=False,
                content=json.dumps(metadata, ensure_ascii=False),
                agent_name=self.name,
                confidence=0.0,
                metadata=metadata,
            )

        self._emit_think(on_think, "🤔", "正在分析上下文并制定任务计划...")
        prompt = self._build_context_plan_prompt(
            user_request,
            conversation_history,
            user_profile,
            previous_context,
            memory_context,
        )

        try:
            response_text = self.call_llm(prompt, temperature=0.3)
            payload = self._parse_json_response(response_text)
            context_analysis = payload.get("context_analysis", {}) if isinstance(payload, dict) else {}
            plan = payload.get("plan", payload if isinstance(payload, dict) else {})
            if not isinstance(context_analysis, dict):
                context_analysis = {}
            plan = self._normalize_plan(plan, user_request)
            context_analysis = self._normalize_context_analysis(context_analysis, user_request, user_profile)

            task_type = plan.get("task_type", "公文生成")
            doc_type = plan.get("document_type", "通用公文")
            self._emit_think(on_think, "📋", f"任务类型：{task_type} / {doc_type}")

            metadata = {
                "context_analysis": context_analysis,
                "plan": plan,
            }
            return AgentResult(
                success=True,
                content=json.dumps(metadata, ensure_ascii=False),
                agent_name=self.name,
                confidence=min(plan.get("confidence", 0.8), context_analysis.get("confidence", 0.8)),
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("PlannerAgent 合并规划降级: %s", e)
            context_analysis = self._fallback_context(user_request, user_profile)
            plan = self._normalize_plan({}, user_request)
            metadata = {
                "context_analysis": context_analysis,
                "plan": plan,
            }
            return AgentResult(
                success=True,
                content=json.dumps(metadata, ensure_ascii=False),
                agent_name=self.name,
                confidence=0.55,
                metadata=metadata,
            )

    def _build_context_plan_prompt(
        self,
        user_request: str,
        conversation_history: list,
        user_profile: dict,
        previous_context: dict,
        memory_context: str = "",
    ) -> str:
        history_lines = []
        for msg in conversation_history[-6:]:
            if hasattr(msg, "role"):
                role = msg.role
                content = msg.content
            else:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            history_lines.append(f"{role}: {str(content)[:500]}")

        return f"""请一次性完成上下文分析和任务规划，输出严格 JSON。

当前用户请求：
{user_request}

用户画像：
{json.dumps(user_profile, ensure_ascii=False)}

上一轮上下文：
{json.dumps(previous_context, ensure_ascii=False)[:1800]}

最近对话：
{chr(10).join(history_lines) if history_lines else "无"}

已召回的历史上下文：
{memory_context or "无"}

输出格式：
{{
  "context_analysis": {{
    "key_points": ["关键信息1"],
    "user_intent": "用户核心意图",
    "document_type": "文档或任务类型",
    "writing_style": "写作风格",
    "key_info": {{
      "preferences": {{}},
      "important_dates": [],
      "names": [],
      "organizations": []
    }},
    "context_quality": {{
      "score": 0.8,
      "issues": [],
      "suggestions": []
    }},
    "confidence": 0.8
  }},
  "plan": {{
    "task_type": "问答检索|公文生成|材料改写|格式转换|续写修改",
    "document_type": "公文类型或知识库问答",
    "need_web_search": false,
    "search_queries": [],
    "knowledge_queries": ["知识库检索关键词"],
    "plan_steps": [
      {{"step": 1, "agent": "Knowledge", "action": "检索相关资料"}},
      {{"step": 2, "agent": "Writer", "action": "生成或回答"}}
    ],
    "key_points": ["要点1"],
    "confidence": 0.8
  }}
}}

规则：
- 用户主要询问知识库、文件内容、政策依据或历史材料时，task_type 为“问答检索”。
- 用户要求起草通知、报告、请示、函、建议、议案、审议事项等新文档时，task_type 为“公文生成”；涉及“议案/审议/办公会”时，document_type 使用“会议审议案”。
- 用户要求润色、压缩、扩写、优化已有材料时，task_type 为“材料改写”。
- 用户上传材料后要求套用格式或规范排版时，task_type 为“格式转换”。
- 用户基于上一轮结果继续修改、补充、重写时，task_type 为“续写修改”。
- 涉及最新政策、数据、时事动态、人物或机构最新信息时，need_web_search 为 true。
- 只输出 JSON，不要输出解释。"""

    def _normalize_context_analysis(self, context: dict, user_request: str, user_profile: dict) -> dict:
        context = context if isinstance(context, dict) else {}
        fallback = self._fallback_context(user_request, user_profile)
        for key, value in fallback.items():
            context.setdefault(key, value)
        if not isinstance(context.get("key_points"), list):
            context["key_points"] = []
        if not isinstance(context.get("key_info"), dict):
            context["key_info"] = {"preferences": user_profile or {}}
        if not isinstance(context.get("context_quality"), dict):
            context["context_quality"] = {"score": 0.6, "issues": [], "suggestions": []}
        context.setdefault("confidence", 0.6)
        return context

    def _fallback_context(self, user_request: str, user_profile: dict) -> dict:
        return {
            "key_points": [],
            "user_intent": user_request[:100],
            "document_type": "通用公文",
            "writing_style": "简洁正式",
            "key_info": {"preferences": user_profile or {}},
            "context_quality": {"score": 0.6, "issues": [], "suggestions": []},
            "confidence": 0.5,
        }

    def _parse_json_response(self, text: str) -> dict:
        """解析 LLM 返回的 JSON 响应，带错误处理"""
        import logging
        logger = logging.getLogger(__name__)

        text = text.strip()

        # 去除 markdown 代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            # 找到第一个和最后一个 ``` 的位置
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
            logger.warning(f"JSON 解析失败: {e}, 原始文本: {text[:200]}...")
            # 尝试修复常见的 JSON 格式问题
            try:
                # 使用正则表达式去除对象和数组中的尾部逗号
                import re
                # 去除对象尾部逗号: {"a":1,} -> {"a":1}
                text = re.sub(r',\s*([}\]])', r'\1', text)
                # 去除数组尾部逗号: [1,2,] -> [1,2]
                text = re.sub(r',\s*$', '', text)
                return json.loads(text)
            except json.JSONDecodeError:
                logger.error("JSON 修复失败，返回默认规划")
                # 返回默认规划而不是抛出异常
                return {
                    "task_type": "公文生成",
                    "document_type": "通用公文",
                    "need_web_search": False,
                    "search_queries": [],
                    "knowledge_queries": [],
                    "plan_steps": [],
                    "key_points": [],
                    "confidence": 0.5,
                }

    def _normalize_plan(self, plan: dict, user_request: str) -> dict:
        """补齐旧模型输出缺失的结构化字段，保证下游流程稳定。"""
        if not isinstance(plan, dict):
            plan = {}

        allowed_task_types = {"问答检索", "公文生成", "材料改写", "格式转换", "续写修改"}
        task_type = plan.get("task_type")
        if task_type not in allowed_task_types:
            task_type = self._infer_task_type(user_request)
        plan["task_type"] = task_type

        if any(k in user_request for k in ["议案", "审议", "办公会"]):
            plan["document_type"] = "会议审议案"
        else:
            plan.setdefault("document_type", "通用公文")
        plan.setdefault("need_web_search", False)
        plan.setdefault("search_queries", [])
        plan.setdefault("knowledge_queries", [user_request] if user_request else [])
        plan.setdefault("plan_steps", [])
        plan.setdefault("key_points", [])
        plan.setdefault("confidence", 0.6)
        return plan

    def _infer_task_type(self, user_request: str) -> str:
        text = user_request or ""
        if "[文件内容]" in text or any(k in text for k in ["格式", "排版", "规范公文"]):
            return "格式转换"
        if any(k in text for k in ["修改", "润色", "改写", "优化", "压缩", "扩写"]):
            return "材料改写"
        if any(k in text for k in ["继续", "上一版", "上一次", "补充", "重写"]):
            return "续写修改"
        if any(k in text for k in ["查询", "检索", "有哪些", "是什么", "依据", "来源"]):
            return "问答检索"
        return "公文生成"
