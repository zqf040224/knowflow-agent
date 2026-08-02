
"""
基础Agent类 - 优化版

提供Agent的基础功能：
1. 增强的LLM调用
2. 智能记忆管理
3. 上下文感知
4. 用户画像集成
5. 错误处理增强
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 加载环境变量 - 从项目根目录加载
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class AgentMessage:
    """Agent消息"""
    role: str
    content: str
    agent_name: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__('time').time())

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "agent_name": self.agent_name,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentResult:
    """Agent执行结果 - 增强版"""
    success: bool
    content: str
    agent_name: str
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)
    execution_time: float = 0.0
    error_info: Optional[Dict] = None

    def to_dict(self):
        return {
            "success": self.success,
            "content": self.content,
            "agent_name": self.agent_name,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "error_info": self.error_info,
        }


class BaseAgent(ABC):
    """基础Agent类 - 优化版"""

    def __init__(
        self,
        name: str,
        description: str,
        model: str = "deepseek-v4-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retry: int = 3,
        context_window: int = 1024000,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = base_url or DEEPSEEK_BASE_URL
        self.timeout = float(os.getenv("LLM_TIMEOUT", "120"))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.history: list[AgentMessage] = []
        self.session_id: Optional[str] = None
        self.short_term_memory: Optional[Any] = None
        self.user_profile: Optional[Dict] = None
        self.max_retry = max_retry
        self.context_window = context_window
        self._call_count = 0
        self.last_reasoning = ""  # deepseek-reasoner 推理链
        self.last_usage: Dict[str, Any] = {}

    @abstractmethod
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        pass

    @abstractmethod
    def process(self, input_data: dict, on_think=None) -> AgentResult:
        """
        处理输入数据并返回结果
        
        Args:
            input_data: 输入数据
            on_think: 思考回调函数 (agent_name, emoji, message)
        """
        pass

    def _emit_think(self, on_think, emoji, message):
        """发送思考消息"""
        if on_think:
            return on_think(self.name, emoji, message)
        return None

    def set_session(self, session_id: str, memory=None, user_profile=None):
        """设置当前会话和用户画像"""
        self.session_id = session_id
        self.short_term_memory = memory
        if user_profile:
            self.user_profile = user_profile

    def save_to_memory(self, key: str, value: Any):
        """保存数据到短期记忆"""
        if self.short_term_memory and self.session_id:
            self.short_term_memory.set_agent_state(
                self.session_id, self.name, {key: value}
            )

    def load_from_memory(self) -> Optional[Dict]:
        """从短期记忆加载数据"""
        if self.short_term_memory and self.session_id:
            return self.short_term_memory.get_agent_state(self.session_id, self.name)
        return None

    def get_conversation_context(self, max_messages: int = 10, 
                                   summarize: bool = False) -> str:
        """获取对话上下文 - 增强版"""
        if not (self.short_term_memory and self.session_id):
            return ""
        
        if summarize:
            # 智能摘要模式
            messages = self.short_term_memory.get_session_history(
                self.session_id, limit=max_messages
            )
            if len(messages) > 5:
                # 历史较长时进行摘要
                raw_context = self._format_messages(messages)
                return self._summarize_context(raw_context)
            else:
                return self._format_messages(messages)
        else:
            # 原始模式
            return self.short_term_memory.get_conversation_context(
                self.session_id, max_messages
            )

    def _format_messages(self, messages: List) -> str:
        """格式化消息列表"""
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
            else:
                role = getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', '')
            
            role_label = "用户" if role == "user" else "助手"
            formatted.append(f"{role_label}：{content}")
        
        return "\n\n".join(formatted)

    def _summarize_context(self, context: str) -> str:
        """智能摘要上下文"""
        if len(context) < 500:
            return context
        
        prompt = f"""请摘要以下对话历史，保留关键信息，控制在300字以内：

{context}

要求：
1. 保留用户核心需求
2. 保留重要的决策和共识
3. 保留格式要求
4. 保持语言简洁"""

        try:
            return self.call_llm(prompt, temperature=0.3, max_tokens=500, use_context=False)
        except Exception as e:
            logger.warning(f"摘要失败: {e}")
            return context[:500] + "..."

    def build_enhanced_prompt(
        self,
        user_content: str,
        context_analysis: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """构建增强的提示词"""
        sys_prompt = system_prompt or self.get_system_prompt()
        parts = [sys_prompt]

        # 添加用户画像
        if self.user_profile:
            parts.append("\n【用户偏好】")
            if self.user_profile.get("preferred_font"):
                parts.append(f"- 字体：{self.user_profile['preferred_font']}")
            if self.user_profile.get("preferred_size"):
                parts.append(f"- 字号：{self.user_profile['preferred_size']}")
            if self.user_profile.get("writing_style"):
                parts.append(f"- 风格：{self.user_profile['writing_style']}")

        # 添加上下文分析
        if context_analysis:
            if context_analysis.get("key_points"):
                parts.append("\n【关键信息】")
                for point in context_analysis["key_points"][:5]:
                    parts.append(f"- {point}")
            
            if context_analysis.get("user_intent"):
                parts.append(f"\n【用户意图】{context_analysis['user_intent']}")
            
            quality_info = context_analysis.get("context_quality", {})
            if quality_info.get("issues"):
                parts.append("\n【注意事项】")
                for issue in quality_info["issues"][:3]:
                    parts.append(f"- {issue}")

        # 添加当前需求
        parts.append(f"\n【当前需求】{user_content}")

        return "\n".join(parts)

    def build_prompt_with_context(self, user_content: str,
                                   system_prompt: Optional[str] = None) -> str:
        """构建包含上下文的 Prompt - 保留向后兼容"""
        context = self.get_conversation_context()
        sys_prompt = system_prompt or self.get_system_prompt()

        if context:
            return f"{sys_prompt}\n\n## 对话历史\n{context}\n\n## 当前需求\n{user_content}"
        return f"{sys_prompt}\n\n{user_content}"

    def _build_messages(self, user_content: str, system_prompt: Optional[str],
                        use_context: bool, context_analysis: Optional[Dict]) -> List[Dict[str, str]]:
        if use_context and context_analysis:
            sys_prompt = self.build_enhanced_prompt(
                user_content, context_analysis, system_prompt
            )
            return [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": sys_prompt},
            ]
        if use_context and self.session_id:
            sys_prompt = self.build_prompt_with_context(user_content, system_prompt)
            return [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": sys_prompt},
            ]
        sys_prompt = system_prompt or self.get_system_prompt()
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

    def _message_chars(self, messages: List[Dict[str, str]]) -> int:
        return sum(len(item.get("content", "")) for item in messages)

    def _record_usage(self, *, prompt_chars: int, completion_chars: int,
                      duration_ms: int, max_tokens: int, temperature: float,
                      stream: bool, reasoning_chars: int = 0):
        estimated_prompt_tokens = max(1, int(prompt_chars / 1.8)) if prompt_chars else 0
        estimated_completion_tokens = max(1, int(completion_chars / 1.8)) if completion_chars else 0
        self.last_usage = {
            "agent": self.name,
            "model": self.model,
            "stream": stream,
            "prompt_chars": prompt_chars,
            "completion_chars": completion_chars,
            "reasoning_chars": reasoning_chars,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "estimated_completion_tokens": estimated_completion_tokens,
            "estimated_total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
            "duration_ms": duration_ms,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "call_count": self._call_count,
        }

    def call_llm(
        self,
        user_content: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_context: bool = True,
        context_analysis: Optional[Dict] = None,
    ):
        """调用 LLM（非流式），带重试机制"""
        start_time = time.time()
        self._call_count += 1

        logger.info(f"[{self.name}] 开始调用 LLM (#{self._call_count}) - 模型: {self.model}")

        messages = self._build_messages(user_content, system_prompt, use_context, context_analysis)
        prompt_chars = self._message_chars(messages)

        last_error = None
        for retry in range(self.max_retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                msg = response.choices[0].message
                content = msg.content
                # 捕获 deepseek-reasoner 的推理链
                if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                    self.last_reasoning = msg.reasoning_content
                execution_time = time.time() - start_time
                reasoning_chars = len(self.last_reasoning or "")
                self._record_usage(
                    prompt_chars=prompt_chars,
                    completion_chars=len(content or ""),
                    reasoning_chars=reasoning_chars,
                    duration_ms=int(execution_time * 1000),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
                logger.info(
                    f"[{self.name}] LLM 调用完成 - prompt: {prompt_chars} 字符, "
                    f"响应: {len(content)} 字符, 耗时: {execution_time:.2f}s"
                )

                return content

            except Exception as e:
                last_error = e
                wait_time = (retry + 1) * 2
                logger.warning(f"[{self.name}] LLM 调用失败 (尝试 {retry + 1}/{self.max_retry}): {e}")

                if retry < self.max_retry - 1:
                    logger.info(f"[{self.name}] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        raise Exception(f"LLM 调用失败，已重试 {self.max_retry} 次: {last_error}")

    def call_llm_stream(
        self,
        user_content: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_context: bool = True,
        context_analysis: Optional[Dict] = None,
    ):
        """流式调用 LLM - 增强版"""
        start_time = time.time()
        self._call_count += 1
        logger.info(f"[{self.name}] 开始流式调用 LLM - 模型: {self.model}")
        logger.info(f"[{self.name}] 温度: {temperature}, 最大tokens: {max_tokens}")
        logger.info(f"[{self.name}] 用户输入: {user_content[:100]}...")
        
        messages = self._build_messages(user_content, system_prompt, use_context, context_analysis)
        prompt_chars = self._message_chars(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        total_chars = 0
        reasoning_total = 0
        for chunk in response:
            delta = chunk.choices[0].delta
            # deepseek-reasoner 的推理链
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_total += len(delta.reasoning_content)
                yield ("reasoning", delta.reasoning_content)
            if delta.content:
                chunk_content = delta.content
                total_chars += len(chunk_content)
                yield ("content", chunk_content)

        duration_ms = int((time.time() - start_time) * 1000)
        self._record_usage(
            prompt_chars=prompt_chars,
            completion_chars=total_chars,
            reasoning_chars=reasoning_total,
            duration_ms=duration_ms,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        logger.info(f"[{self.name}] 流式 LLM 调用完成 - prompt: {prompt_chars} 字符, 响应: {total_chars} 字符, 推理: {reasoning_total} 字符")

    def get_stats(self) -> Dict:
        """获取Agent统计信息"""
        return {
            "name": self.name,
            "call_count": self._call_count,
            "session_id": self.session_id,
            "has_user_profile": self.user_profile is not None,
        }
