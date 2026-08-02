"""Streaming RAG question-answering pipeline for the chat endpoint."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from chat_answer_quality import (
    EvidenceReport,
    VerificationReport,
    build_answer_plan,
    build_audit_summary,
    build_llm_planner_prompt,
    build_evidence_fallback_answer,
    evaluate_evidence,
    parse_llm_answer_plan,
    should_use_llm_planner,
    verify_answer,
)
from chat_architecture import INTENT_KNOWLEDGE_QA
from chat_events import (
    route_actions,
    route_event,
    route_intent,
    route_payload,
    source_details_from_results,
    sse,
    text_stream_sse,
)

logger = logging.getLogger(__name__)


def chat_context_for_model(context: str) -> str:
    """Keep answer context useful while hiding internal retrieval coordinates."""
    if not context:
        return ""
    hidden_prefixes = (
        "[文档",
        "片段:",
        "页码:",
        "章节:",
        "解析提示:",
        "Sheet:",
        "行类型:",
        "文件：",
        "Sheet：",
        "行号：",
        "行类型：",
    )
    lines = []
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if any(line.startswith(prefix) for prefix in hidden_prefixes):
            continue
        unnamed = re.match(r"^未命名列\d+：\s*(.*)$", line)
        if unnamed:
            value = unnamed.group(1).strip()
            if not value or value == "0":
                continue
            lines.append(value)
            continue
        lines.append(raw_line)
    return "\n".join(lines).strip()


def build_storage_server_answer(normalized_task: dict[str, Any], context: str, results: list[dict[str, Any]]) -> str:
    """Return NAS/netdisk instructions only from permission-filtered evidence."""
    if (normalized_task or {}).get("task_type") != "storage_server_usage":
        return ""

    # Only use permission-filtered retrieval results. Conversation history is
    # user-controlled and must not be treated as an authorized infrastructure source.
    combined = "\n".join(str(item.get("text", "")) for item in results or [])
    if not combined.strip():
        return ""
    address_match = re.search(r"(?:\\\\|\\){1,2}\s*(\d{1,3}(?:\.\d{1,3}){3})", combined)
    ip_match = re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", combined)
    address = ""
    if address_match:
        address = "\\\\" + address_match.group(1)
    elif ip_match:
        address = "\\\\" + ip_match.group(0)
    if not address:
        return ""

    source = ""
    for item in results or []:
        filename = str(item.get("filename") or "").strip()
        if filename:
            source = filename
            break

    lines = [
        f"网盘/NAS 存储服务器的访问地址是：`{address}`。",
        "",
        "使用方式：",
        f"1. 在 Windows 电脑上按 `Windows + R` 打开运行窗口。",
        f"2. 输入 `{address}`，进入账户登录界面。",
        "3. 输入你的用户名和密码；如是常用电脑，可以勾选“保存此凭证”。",
        "4. 登录后即可按权限访问、上传、下载单位工作相关文件。",
        "",
        "也可以创建桌面快捷方式：在桌面右键选择“新建” -> “快捷方式”，对象地址填写 "
        f"`{address}`，然后按提示完成创建。",
        "",
        "注意：请只存放单位工作相关文件，不要上传与工作无关的个人数据或敏感信息。",
    ]
    if source:
        lines.extend(["", f"参考来源：{source}"])
    return "\n".join(lines)


def storage_server_source_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only NAS/storage-server sources for deterministic storage answers."""
    filtered = []
    for item in results or []:
        filename = str(item.get("filename") or "")
        text = str(item.get("text") or "")
        haystack = f"{filename}\n{text}".lower()
        if (
            "存储服务器" in haystack
            or "nas" in haystack
            or "私有云" in haystack
            or "虚拟盘" in haystack
        ):
            filtered.append(item)
    return filtered or (results[:1] if results else [])


@dataclass
class RagQaDependencies:
    memory: Any
    knowledge_agent: Any
    deepseek_api_key: str
    record_token_usage: Callable[..., None]


class RagQaStreamService:
    def __init__(self, deps: RagQaDependencies):
        self.deps = deps

    def stream(
        self,
        message,
        session_id,
        user_id,
        user_info=None,
        display_message=None,
        user_metadata=None,
        route=None,
    ):
        """普通聊天流式响应."""
        stored_user_message = display_message or message
        memory = self.deps.memory
        memory.add_message(session_id, "user", stored_user_message, metadata=user_metadata or {})

        yield sse({"type": "start"})
        yield sse({"type": "session", "session_id": session_id})
        if route:
            yield route_event(route)
        yield sse({"type": "thinking_start", "message": "开始理解问题并检索知识库"})
        conversation_context = memory.get_context_for_prompt(session_id, max_messages=5)
        rule_plan = build_answer_plan(message, conversation_context)
        answer_plan = rule_plan
        if self._should_call_deepseek_planner(message, rule_plan):
            yield sse({
                "type": "think",
                "agent": "AnswerPlanner",
                "emoji": "🧠",
                "message": "正在用 DeepSeek 理解你的真实需求...",
            })
            answer_plan = self._build_deepseek_answer_plan(
                message=message,
                conversation_context=conversation_context,
                fallback_plan=rule_plan,
                session_id=session_id,
                user_id=user_id,
                user_info=user_info,
            )
        yield sse({
            "type": "think",
            "agent": "AnswerPlanner",
            "emoji": "🧭",
            "message": f"已识别为{answer_plan.answer_intent}，正在规划检索...",
        })

        response_plan = {
            "document_type": "知识库问答",
            "task_type": "问答检索",
            "need_web_search": False,
            "answer_intent": answer_plan.answer_intent,
            "planner_source": answer_plan.planner_source,
            "normalized_task": answer_plan.normalized_task,
        }

        if answer_plan.needs_clarification:
            evidence = EvidenceReport(
                passed=False,
                score=0.0,
                reason="问题存在弱指代且会话中没有足够上下文",
                missing=answer_plan.expected_evidence,
                source_count=0,
                top_sources=[],
            )
            response_plan["task_type"] = "澄清需求"
            answer = build_evidence_fallback_answer(answer_plan, evidence)
            yield from self._finalize_without_model(
                answer=answer,
                memory=memory,
                session_id=session_id,
                stored_user_message=stored_user_message,
                response_plan=response_plan,
                route=route,
                audit_summary=build_audit_summary(answer_plan, evidence),
                user_id=user_id,
                user_info=user_info,
            )
            return

        yield sse({"type": "think", "agent": "KnowledgeAgent", "emoji": "📚", "message": "正在检索知识库..."})

        try:
            retrieval_request = (
                answer_plan.normalized_task.get("canonical_query")
                if answer_plan.normalized_task
                else message
            )
            result = self.deps.knowledge_agent.process({
                "user_request": retrieval_request,
                "knowledge_queries": answer_plan.queries,
                "answer_intent": answer_plan.answer_intent,
                "expected_evidence": answer_plan.expected_evidence,
                "normalized_task": answer_plan.normalized_task,
                "user_info": user_info.to_dict() if user_info else None,
            })
            context = result.content
            hit_count = len(result.metadata.get("results", [])) if result.success else 0
            yield sse({
                "type": "think",
                "agent": "KnowledgeAgent",
                "emoji": "✅",
                "message": f"知识库检索完成，命中 {hit_count} 条参考",
            })
        except Exception as exc:
            logger.warning("普通聊天知识库检索失败: %s", exc)
            result = type("ChatKbResult", (), {"success": False, "content": "", "metadata": {"results": []}})()
            context = ""
            yield sse({"type": "think", "agent": "KnowledgeAgent", "emoji": "⚠️", "message": "知识库暂不可用，将直接回答"})

        evidence = evaluate_evidence(
            answer_plan,
            result.metadata.get("results", []) if result.success else [],
            context,
        )
        yield sse({
            "type": "think",
            "agent": "EvidenceGate",
            "emoji": "✅" if evidence.passed else "⚠️",
            "message": evidence.reason,
        })

        if not evidence.passed:
            response_plan["task_type"] = "证据不足答复"
            answer = build_evidence_fallback_answer(answer_plan, evidence)
            yield from self._finalize_without_model(
                answer=answer,
                memory=memory,
                session_id=session_id,
                stored_user_message=stored_user_message,
                response_plan=response_plan,
                route=route,
                audit_summary=build_audit_summary(answer_plan, evidence),
                user_id=user_id,
                user_info=user_info,
                source_results=result.metadata.get("results", []) if result.success else [],
            )
            return

        deterministic_answer = build_storage_server_answer(
            answer_plan.normalized_task,
            context,
            result.metadata.get("results", []) if result.success else [],
        )
        if deterministic_answer:
            response_plan["task_type"] = "标准操作说明"
            verification = VerificationReport(passed=True, sanitized_answer=deterministic_answer)
            deterministic_sources = storage_server_source_results(
                result.metadata.get("results", []) if result.success else []
            )
            yield sse({"type": "think", "agent": "Chat", "emoji": "💬", "message": "已根据标准操作说明整理回答"})
            yield from self._finalize_without_model(
                answer=deterministic_answer,
                memory=memory,
                session_id=session_id,
                stored_user_message=stored_user_message,
                response_plan=response_plan,
                route=route,
                audit_summary=build_audit_summary(answer_plan, evidence, verification),
                user_id=user_id,
                user_info=user_info,
                source_results=deterministic_sources,
            )
            return

        model_context = chat_context_for_model(context)
        if not self.deps.deepseek_api_key:
            logger.error("DEEPSEEK_API_KEY 未配置，普通聊天无法调用模型")
            self.deps.record_token_usage(
                user_id=user_id,
                user_info=user_info,
                session_id=session_id,
                mode="chat",
                agent="Chat",
                model="deepseek-v4-flash",
                prompt_chars=len(message or "") + len(context or ""),
                status="failed",
                error_message="DEEPSEEK_API_KEY 未配置",
            )
            yield sse({"type": "error", "message": "模型 API Key 未配置，请先在服务端配置 DEEPSEEK_API_KEY"})
            return

        prompt = self._build_prompt(
            message=message,
            model_context=model_context,
            conversation_context=conversation_context,
            profile=memory.get_user_profile(user_id),
            user_id=user_id,
            answer_intent=answer_plan.answer_intent,
            expected_evidence=answer_plan.expected_evidence,
        )

        from openai import OpenAI

        client = OpenAI(
            api_key=self.deps.deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
        )

        chat_start = time.time()
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        yield sse({"type": "write_start", "message": "正在组织回答..."})
        yield sse({"type": "think", "agent": "Chat", "emoji": "💬", "message": "正在基于参考资料生成回答..."})
        yield sse({"type": "thinking_done", "summary": "资料检索和证据检查完成，开始输出正文"})
        yield sse({"type": "answer_start", "message": "开始输出正文", "session_id": session_id})

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                full_response += delta
                yield sse({"type": "answer_delta", "data": delta, "session_id": session_id})
                yield sse({"type": "content", "data": delta, "session_id": session_id})

        verification = verify_answer(
            full_response,
            answer_plan,
            result.metadata.get("results", []) if result.success else [],
            context,
        )
        full_response = self._answer_with_verification_notes(
            verification,
            evidence,
        )
        if full_response and full_response.startswith(verification.sanitized_answer):
            suffix = full_response[len(verification.sanitized_answer):]
            if suffix:
                yield sse({"type": "answer_delta", "data": suffix, "session_id": session_id})
                yield sse({"type": "content", "data": suffix, "session_id": session_id})
        yield sse({"type": "answer_done", "answer": full_response, "session_id": session_id})

        source_filenames = evidence.top_sources
        audit_summary = build_audit_summary(answer_plan, evidence, verification)
        memory.add_message(session_id, "assistant", full_response, metadata={
            "type": route_intent(route, INTENT_KNOWLEDGE_QA),
            "route": route_payload(route),
            "actions": route_actions(route),
            "audit_summary": audit_summary,
        })
        self.deps.record_token_usage(
            user_id=user_id,
            user_info=user_info,
            session_id=session_id,
            mode="chat",
            agent="Chat",
            model="deepseek-v4-flash",
            stream=True,
            prompt_chars=len(prompt),
            completion_chars=len(full_response),
            duration_ms=int((time.time() - chat_start) * 1000),
            status="success",
        )
        unique_sources = list(dict.fromkeys(
            f[0] if isinstance(f, tuple) else f for f in source_filenames
        ))[:8]
        memory.set_context(session_id, "last_request", stored_user_message)
        memory.set_context(session_id, "last_answer", full_response)
        memory.set_context(session_id, "last_answer_plan", response_plan)
        memory.update_rolling_summary(session_id, stored_user_message, full_response, response_plan, unique_sources)

        source_details = source_details_from_results(result.metadata.get("results", []) if result.success else [])
        yield sse({"type": "run_done", "session_id": session_id, "intent": route_intent(route, INTENT_KNOWLEDGE_QA)})
        yield sse({
            "type": "done",
            "intent": route_intent(route, INTENT_KNOWLEDGE_QA),
            "answer": full_response,
            "document": "",
            "session_id": session_id,
            "plan": response_plan,
            "route": route_payload(route),
            "actions": route_actions(route),
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": unique_sources,
            "source_details": source_details,
            "audit_summary": audit_summary,
        })

    @staticmethod
    def _build_prompt(
        *,
        message: str,
        model_context: str,
        conversation_context: str,
        profile: Optional[Any],
        user_id: str,
        answer_intent: str = "",
        expected_evidence: Optional[list[str]] = None,
    ) -> str:
        profile_prefix = ""
        if profile:
            profile_prefix = f"用户 {profile.name or user_id} ({profile.department})，"
        context_header = "## 对话历史\n" + conversation_context + "\n" if conversation_context else ""
        evidence_line = "、".join(expected_evidence or [])
        return f"""你是智能知识库助手，{profile_prefix}服务于智能知识库平台。公文写作只是你的能力之一，你还可以进行知识库检索、资料问答、材料归纳、内容改写和文档处理。

请基于以下从知识库检索到的参考材料回答用户问题。

如果用户询问你的身份或能力，请明确回答：你是智能知识库助手，并说明你可以做知识库检索与问答、资料归纳、公文写作、材料处理、上下文续写修改、导入编辑器与导出 Word 等工作。

{context_header}参考材料（每条材料只标注来源文件名）：
{model_context}

本轮回答意图：{answer_intent or "knowledge_qa"}
应优先核对的证据：{evidence_line or "相关知识库来源"}

用户问题：{message}

请根据参考材料回答。要求：
1. 用客服式自然语言回答用户问题：先直接给结论或操作步骤，再补充注意事项；不要整段照抄原文
2. 如果用户询问某个文件的内容但记不清具体文件名，列出所有可能相关的文件
3. 只能引用参考材料中“文件名:”后面的文件名作为来源；回答正文和参考来源都不要出现[文档N]、片段、chunk、页码、章节、Sheet、行号、路径等内部检索信息
4. 对敏感信息执行最小披露：密码、Token、密钥不得输出；网址、内网地址、账号、路径、端口、命令、电话等，仅在用户明确询问且该信息来自本轮权限过滤后的参考材料时才按必要范围展示，否则应掩码或提示联系管理员
5. 只有当用户明确要求“写成公文/改为公文格式/起草通知、请示、报告、议案”等时，才按公文格式输出；普通咨询不要写成公文
6. 如果本轮回答意图是 open_ended 且参考材料为空，可以给通用建议，但必须说明“当前未引用知识库资料”"""

    def _finalize_without_model(
        self,
        *,
        answer: str,
        memory,
        session_id: str,
        stored_user_message: str,
        response_plan: dict,
        route,
        audit_summary: dict,
        user_id: str,
        user_info=None,
        source_results=None,
    ):
        source_results = source_results or []
        source_details = source_details_from_results(source_results)
        source_filenames = [item["filename"] for item in source_details if item.get("filename")]

        yield sse({"type": "thinking_done", "summary": "已完成证据检查，直接返回结论"})
        yield sse({"type": "answer_start", "message": "开始输出正文", "session_id": session_id})
        yield from text_stream_sse(answer, session_id=session_id)
        yield sse({"type": "answer_done", "answer": answer, "session_id": session_id})
        memory.add_message(session_id, "assistant", answer, metadata={
            "type": route_intent(route, INTENT_KNOWLEDGE_QA),
            "route": route_payload(route),
            "actions": route_actions(route),
            "audit_summary": audit_summary,
        })
        self.deps.record_token_usage(
            user_id=user_id,
            user_info=user_info,
            session_id=session_id,
            mode="chat",
            agent="Chat",
            model="none",
            prompt_chars=len(stored_user_message or ""),
            completion_chars=len(answer or ""),
            status="skipped",
        )
        memory.set_context(session_id, "last_request", stored_user_message)
        memory.set_context(session_id, "last_answer", answer)
        memory.set_context(session_id, "last_answer_plan", response_plan)
        memory.update_rolling_summary(session_id, stored_user_message, answer, response_plan, source_filenames)

        yield sse({"type": "run_done", "session_id": session_id, "intent": route_intent(route, INTENT_KNOWLEDGE_QA)})
        yield sse({
            "type": "done",
            "intent": route_intent(route, INTENT_KNOWLEDGE_QA),
            "answer": answer,
            "document": "",
            "session_id": session_id,
            "plan": response_plan,
            "route": route_payload(route),
            "actions": route_actions(route),
            "export_template": "",
            "export_spreadsheet_template": "",
            "source_filenames": source_filenames,
            "source_details": source_details,
            "audit_summary": audit_summary,
        })

    @staticmethod
    def _answer_with_verification_notes(verification: VerificationReport, evidence: EvidenceReport) -> str:
        answer = verification.sanitized_answer
        notes = []
        if verification.issues:
            notes.extend(verification.issues[:3])
        if evidence.top_sources and "回答未明确引用已检索到的来源文件名" in verification.issues:
            notes.append("可参考来源：" + "；".join(evidence.top_sources[:5]))
        if notes:
            answer = answer.rstrip() + "\n\n【依据提示】" + "；".join(dict.fromkeys(notes))
        return answer

    def _should_call_deepseek_planner(self, message: str, rule_plan) -> bool:
        if not self.deps.deepseek_api_key:
            return False
        return should_use_llm_planner(
            message,
            rule_plan,
            mode=os.getenv("ANSWER_PLANNER", "auto"),
        )

    def _build_deepseek_answer_plan(
        self,
        *,
        message: str,
        conversation_context: str,
        fallback_plan,
        session_id: str,
        user_id: str,
        user_info=None,
    ):
        prompt = build_llm_planner_prompt(message, conversation_context, fallback_plan)
        planner_start = time.time()
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.deps.deepseek_api_key,
                base_url="https://api.deepseek.com/v1",
            )
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            text = response.choices[0].message.content or ""
            plan = parse_llm_answer_plan(text, fallback_plan, message)
            self.deps.record_token_usage(
                user_id=user_id,
                user_info=user_info,
                session_id=session_id,
                mode="chat",
                agent="AnswerPlanner",
                model="deepseek-v4-flash",
                prompt_chars=len(prompt),
                completion_chars=len(text),
                duration_ms=int((time.time() - planner_start) * 1000),
                status="success",
            )
            return plan
        except Exception as exc:
            logger.warning("DeepSeek AnswerPlanner failed, fallback to rules: %s", exc)
            self.deps.record_token_usage(
                user_id=user_id,
                user_info=user_info,
                session_id=session_id,
                mode="chat",
                agent="AnswerPlanner",
                model="deepseek-v4-flash",
                prompt_chars=len(prompt),
                duration_ms=int((time.time() - planner_start) * 1000),
                status="failed",
                error_message=str(exc)[:300],
            )
            return fallback_plan
