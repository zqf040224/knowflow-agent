import json
import logging
from agents.base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class ReflectionAgent(BaseAgent):
    """
    反思 Agent — 用 DeepSeek-R1 从对立视角深度审视文档

    与 ReviewerAgent 的区别：
    - Reviewer: 格式规范 + 内容完整性（硬检查 + LLM）
    - Reflection: 论证质量 + 逻辑深度 + 对立观点（仅 R1）
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="Reflection",
            description="反思器 - R1深度推理，从对立视角审视论证质量和逻辑完整性",
            model=kwargs.pop("model", "deepseek-reasoner"),
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return """你是一位严格的评审专家。你的任务是从对立视角审视公文的论证质量。

审查方式：
1. 假设你是评审委员会中最挑剔的一员
2. 找出论证中最薄弱的环节
3. 提出反对意见：如果你持相反观点，你会怎么反驳？
4. 评估论据是否充分支撑结论

你必须以 JSON 格式输出：
{
    "weaknesses": ["论证弱点1", "弱点2"],
    "counter_arguments": ["如果持相反观点，可以这样反驳：..."],
    "missing_evidence": ["缺少的数据或案例支撑"],
    "better_angle": "有没有更好的切入角度或替代方案？",
    "logic_score": 0.8,
    "needs_revision": false,
    "revision_suggestions": ["改进建议1"]
}

只输出 JSON，不要其他内容。"""

    def process(self, input_data: dict, on_think=None) -> AgentResult:
        document_content = input_data.get("document_content", "")
        document_type = input_data.get("document_type", "通用公文")
        user_request = input_data.get("user_request", "")
        context_analysis = input_data.get("context_analysis", {}) or {}
        memory_context = input_data.get("memory_context", "") or ""
        evidence_items = input_data.get("evidence_items", []) or []

        if not document_content:
            return AgentResult(
                success=False,
                content="无内容可反思",
                agent_name=self.name,
                confidence=0.0,
            )

        self._emit_think(on_think, "🧠", "R1 深度反思中，从对立视角审视论证质量...")

        prompt = f"""请从对立视角审查以下{document_type}。

原始需求：{user_request}

上下文意图：{context_analysis.get("user_intent", "") if isinstance(context_analysis, dict) else ""}

已召回的历史上下文：
{memory_context or "无"}

证据来源：{self._format_evidence_items(evidence_items)}

公文内容：
{document_content}

请重点审查：
1. 论证链条是否完整？有没有跳跃？
2. 论据是否充分支撑结论？
3. 如果你是对立观点的一方，怎么反驳？
4. 有没有更好的分析角度或替代方案？
5. 数据和事实引用是否合理？"""

        try:
            response_text = self.call_llm(prompt, temperature=0.3)
            reflection = self._parse_json_response(response_text)
            reflection["reasoning_content"] = self.last_reasoning or ""
        except Exception:
            logger.exception("R1 反思调用失败，使用默认结果")
            reflection = {
                "weaknesses": [],
                "counter_arguments": [],
                "missing_evidence": [],
                "better_angle": "",
                "logic_score": 0.7,
                "needs_revision": False,
                "revision_suggestions": [],
                "reasoning_content": "",
            }
            response_text = json.dumps(reflection, ensure_ascii=False)

        logic_score = reflection.get("logic_score", 0.7)
        needs_revision = reflection.get("needs_revision", False)
        weaknesses = reflection.get("weaknesses", [])

        if logic_score >= 0.8 and not needs_revision:
            self._emit_think(on_think, "✅", f"R1 反思通过，逻辑评分 {int(logic_score * 100)}")
        elif weaknesses:
            self._emit_think(on_think, "⚠️", f"R1 发现问题：{'；'.join(weaknesses[:2])}")
        else:
            self._emit_think(on_think, "⚡", f"R1 反思完成，逻辑评分 {int(logic_score * 100)}")

        return AgentResult(
            success=True,
            content=response_text,
            agent_name=self.name,
            confidence=logic_score,
            metadata=reflection,
        )

    def process_stream(self, input_data: dict, on_think=None):
        """流式反思 — 实时输出 R1 推理链"""
        document_content = input_data.get("document_content", "")
        document_type = input_data.get("document_type", "通用公文")
        user_request = input_data.get("user_request", "")
        context_analysis = input_data.get("context_analysis", {}) or {}
        memory_context = input_data.get("memory_context", "") or ""
        evidence_items = input_data.get("evidence_items", []) or []

        if not document_content:
            yield {"type": "result", "data": AgentResult(
                success=False,
                content="无内容可反思",
                agent_name=self.name,
                confidence=0.0,
            )}
            return

        think_event = self._emit_think(on_think, "🧠", "R1 深度反思中，从对立视角审视论证质量...")
        if think_event:
            yield think_event

        prompt = f"""请从对立视角审查以下{document_type}。

原始需求：{user_request}

上下文意图：{context_analysis.get("user_intent", "") if isinstance(context_analysis, dict) else ""}

已召回的历史上下文：
{memory_context or "无"}

证据来源：{self._format_evidence_items(evidence_items)}

公文内容：
{document_content}

请重点审查：
1. 论证链条是否完整？有没有跳跃？
2. 论据是否充分支撑结论？
3. 如果你是对立观点的一方，怎么反驳？
4. 有没有更好的分析角度或替代方案？
5. 数据和事实引用是否合理？"""

        content_parts = []
        reasoning_parts = []
        reasoning_streamed = False
        self.last_reasoning = ""
        try:
            for chunk_type, chunk_text in self.call_llm_stream(
                prompt, temperature=0.3, max_tokens=4096, use_context=False
            ):
                if chunk_type == "reasoning":
                    reasoning_parts.append(chunk_text)
                    reasoning_streamed = True
                    yield {"type": "reasoning", "data": chunk_text}
                elif chunk_type == "content":
                    content_parts.append(chunk_text)

            response_text = "".join(content_parts)
            reflection = self._parse_json_response(response_text)
            reasoning_text = "".join(reasoning_parts) or self.last_reasoning or reflection.get("reasoning_content", "")
            reflection["reasoning_content"] = reasoning_text
            reflection["reasoning_available"] = bool(reasoning_text)
            if reasoning_text and not reasoning_streamed:
                for start in range(0, len(reasoning_text), 600):
                    yield {"type": "reasoning", "data": reasoning_text[start:start + 600]}

        except Exception:
            logger.exception("R1 流式反思调用失败，使用默认结果")
            reflection = {
                "weaknesses": [],
                "counter_arguments": [],
                "missing_evidence": [],
                "better_angle": "",
                "logic_score": 0.7,
                "needs_revision": False,
                "revision_suggestions": [],
                "reasoning_content": "",
                "reasoning_available": False,
            }

        if not reflection.get("reasoning_available"):
            think_event = self._emit_think(on_think, "ℹ️", "当前模型未返回可展示的 R1 推理链，已展示结构化校验结果")
            if think_event:
                yield think_event

        logic_score = reflection.get("logic_score", 0.7)
        needs_revision = reflection.get("needs_revision", False)
        weaknesses = reflection.get("weaknesses", [])

        if logic_score >= 0.8 and not needs_revision:
            think_event = self._emit_think(on_think, "✅", f"R1 反思通过，逻辑评分 {int(logic_score * 100)}")
        elif weaknesses:
            think_event = self._emit_think(on_think, "⚠️", f"R1 发现问题：{'；'.join(weaknesses[:2])}")
        else:
            think_event = self._emit_think(on_think, "⚡", f"R1 反思完成，逻辑评分 {int(logic_score * 100)}")
        if think_event:
            yield think_event

        yield {"type": "result", "data": AgentResult(
            success=True,
            content=json.dumps(reflection, ensure_ascii=False),
            agent_name=self.name,
            confidence=logic_score,
            metadata=reflection,
        )}

    def _format_evidence_items(self, evidence_items: list) -> str:
        if not evidence_items:
            return "无结构化证据清单"
        lines = []
        for item in evidence_items[:8]:
            title = item.get("title") or item.get("source") or "未命名来源"
            source_type = item.get("type", "source")
            source = item.get("source", "")
            source_text = f"（{source}）" if source else ""
            lines.append(f"[{source_type}] {title}{source_text}")
        return "；".join(lines)

    def _parse_json_response(self, text: str) -> dict:
        text = text.strip()

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
            logger.warning(f"Reflection JSON 解析失败: {e}")
            try:
                text = text.rstrip().rstrip(",").strip()
                return json.loads(text)
            except json.JSONDecodeError:
                return {
                    "weaknesses": [],
                    "counter_arguments": [],
                    "missing_evidence": [],
                    "better_angle": "",
                    "logic_score": 0.7,
                    "needs_revision": False,
                    "revision_suggestions": [],
                }
