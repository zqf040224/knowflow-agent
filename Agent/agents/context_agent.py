
"""
上下文管理Agent - 负责智能上下文收集、整理、摘要和提供

特性：
1. 智能上下文压缩与摘要
2. 关键词提取与重要性排序
3. 多轮对话上下文管理
4. 用户画像与偏好整合
5. 上下文质量评估
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from agents.base_agent import BaseAgent, AgentResult


@dataclass
class ContextSummary:
    """上下文摘要"""
    key_points: List[str] = field(default_factory=list)
    user_intent: str = ""
    document_type: str = ""
    writing_style: str = ""
    key_info: Dict = field(default_factory=dict)
    confidence: float = 0.8


class ContextAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="ContextManager",
            description="上下文管理器 - 智能收集、整理、摘要和提供上下文信息",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return """你是一个专业的上下文管理专家。你的职责是：
1. 分析对话历史，提取关键信息
2. 理解用户意图和需求
3. 识别重要的写作规范和格式要求
4. 摘要历史信息，避免上下文过长
5. 评估上下文质量

你必须以 JSON 格式输出上下文分析结果，格式如下：
{
    "key_points": ["关键信息1", "关键信息2"],
    "user_intent": "用户的核心意图",
    "document_type": "公文类型",
    "writing_style": "写作风格偏好",
    "key_info": {
        "preferences": {"font": "字体", "size": "字号"},
        "important_dates": [],
        "names": [],
        "organizations": []
    },
    "context_quality": {
        "score": 0.8,
        "issues": ["缺少的信息1"],
        "suggestions": ["建议1"]
    },
    "confidence": 0.85
}

只输出 JSON，不要其他内容。"""

    def process(self, input_data: dict, on_think=None) -> AgentResult:
        conversation_history = input_data.get("conversation_history", [])
        user_request = input_data.get("user_request", "")
        user_profile = input_data.get("user_profile", {})
        memory_context = input_data.get("memory_context", "")
        previous_context = input_data.get("previous_context", {})

        # 处理 UserProfile 对象，转换为字典
        if hasattr(user_profile, 'to_dict'):
            user_profile = user_profile.to_dict()

        self._emit_think(on_think, "🧠", "正在分析对话历史...")

        prompt = self._build_prompt(
            conversation_history, user_request, user_profile, previous_context, memory_context
        )

        try:
            response_text = self.call_llm(prompt, temperature=0.3)
            context_analysis = self._parse_json_response(response_text)
            self._emit_think(on_think, "📊", "上下文分析完成")

            return AgentResult(
                success=True,
                content=response_text,
                agent_name=self.name,
                confidence=context_analysis.get("confidence", 0.8),
                metadata=context_analysis,
            )
        except Exception as e:
            self._emit_think(on_think, "⚠️", f"上下文分析出错: {str(e)}")
            import logging
            logging.getLogger(__name__).warning(f"ContextAgent 降级: {e}")
            default_analysis = {
                "key_points": [],
                "user_intent": user_request[:100],
                "document_type": "通用公文",
                "writing_style": "简洁正式",
                "key_info": {"preferences": user_profile},
                "context_quality": {
                    "score": 0.6,
                    "issues": [],
                    "suggestions": []
                },
                "confidence": 0.5
            }
            return AgentResult(
                success=True,
                content=json.dumps(default_analysis, ensure_ascii=False),
                agent_name=self.name,
                confidence=0.5,
                metadata=default_analysis,
            )

    def compress_context(self, context: str, max_length: int = 2000, on_think=None) -> str:
        """智能压缩上下文"""
        if len(context) <= max_length:
            return context

        self._emit_think(on_think, "📝", "正在压缩上下文...")

        prompt = f"""请智能压缩以下上下文，保留所有关键信息，控制在{max_length}字符以内：

{context}

要求：
1. 保留用户需求、关键决策、重要信息
2. 保持逻辑连贯性
3. 去除冗余和重复信息
4. 保留公文格式要求"""

        compressed = self.call_llm(prompt, temperature=0.2, max_tokens=max_length)
        return compressed

    def extract_keywords(self, text: str, on_think=None) -> List[str]:
        """提取关键词"""
        self._emit_think(on_think, "🔑", "正在提取关键词...")

        prompt = f"""请从以下文本中提取关键词（最多10个），按重要性排序：

{text}

以 JSON 数组格式输出：["关键词1", "关键词2"]"""

        response = self.call_llm(prompt, temperature=0.2)
        try:
            keywords = self._parse_json_response(response)
            if isinstance(keywords, list):
                return keywords
            return []
        except Exception:
            return []

    def build_enhanced_prompt(
        self,
        user_request: str,
        conversation_history: List,
        user_profile: Dict = None,
        on_think=None
    ) -> str:
        """构建增强的提示词"""
        self._emit_think(on_think, "✨", "正在构建增强提示词...")

        # 先分析上下文
        analysis_result = self.process({
            "conversation_history": conversation_history,
            "user_request": user_request,
            "user_profile": user_profile or {}
        }, on_think=on_think)

        context_analysis = analysis_result.metadata

        # 构建增强提示
        parts = []
        parts.append(f"【核心需求】{user_request}")

        if context_analysis.get("key_points"):
            parts.append(f"\n【关键信息】\n" + "\n".join(
                f"- {point}" for point in context_analysis["key_points"]
            ))

        if context_analysis.get("user_intent"):
            parts.append(f"\n【用户意图】{context_analysis['user_intent']}")

        if user_profile:
            parts.append(f"\n【用户偏好】")
            if user_profile.get("preferred_font"):
                parts.append(f"- 字体：{user_profile['preferred_font']}")
            if user_profile.get("preferred_size"):
                parts.append(f"- 字号：{user_profile['preferred_size']}")
            if user_profile.get("writing_style"):
                parts.append(f"- 风格：{user_profile['writing_style']}")

        if conversation_history:
            # 只添加最近的几条历史
            recent_history = conversation_history[-3:]
            parts.append(f"\n【对话历史】")
            for msg in recent_history:
                role = "用户" if msg.get("role") == "user" else "助手"
                parts.append(f"{role}：{msg.get('content', '')[:100]}...")

        quality_info = context_analysis.get("context_quality", {})
        if quality_info.get("issues"):
            parts.append(f"\n【注意事项】")
            for issue in quality_info["issues"][:3]:
                parts.append(f"- {issue}")

        return "\n".join(parts)

    def _build_prompt(
        self,
        conversation_history: List,
        user_request: str,
        user_profile: Dict,
        previous_context: Dict,
        memory_context: str = "",
    ) -> str:
        parts = ["请分析以下对话历史和用户请求，生成上下文摘要："]

        if user_profile:
            parts.append(f"\n【用户画像】\n{json.dumps(user_profile, ensure_ascii=False, indent=2)}")

        if conversation_history:
            parts.append("\n【对话历史】")
            for i, msg in enumerate(conversation_history[-5:]):
                # 处理 Message 对象或字典
                if hasattr(msg, 'role'):
                    role = msg.role
                    content = msg.content
                else:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                parts.append(f"{i+1}. {role}: {content}")

        parts.append(f"\n【当前请求】\n{user_request}")

        if memory_context:
            parts.append(f"\n{memory_context}")

        if previous_context:
            parts.append(f"\n【之前的上下文分析】\n{json.dumps(previous_context, ensure_ascii=False, indent=2)}")

        return "\n".join(parts)

    def _parse_json_response(self, text: str) -> Dict:
        """解析 JSON 响应"""
        import logging
        logger = logging.getLogger(__name__)

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
            logger.warning(f"JSON 解析失败: {e}")
            # 尝试修复
            try:
                import re
                text = re.sub(r',\s*([}\]])', r'\1', text)
                return json.loads(text)
            except Exception:
                logger.error("JSON 修复失败")
                return {}
