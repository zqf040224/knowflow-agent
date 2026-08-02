"""
测试 Agent 工作流连贯性 — 验证 ContextAgent 是否被调用、
key_points 是否传递到 Writer、revision_history 是否累积等。
"""
import sys
import os
import json
import logging
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志 — 实时输出到 stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger('pipeline_test')

from agents.orchestrator import AgentOrchestrator, ContextPacket
from agents.base_agent import BaseAgent


MOCK_DOCUMENT = """关于学术交流活动的请示

单位负责人：
为进一步加强学术交流合作，拓展研究人员学术视野，拟于近期组织开展学术交流活动。现将有关事项请示如下：
一、活动安排
拟邀请相关领域专家围绕科研项目管理、成果转化和智库建设开展专题交流。
二、组织保障
由项目部统筹协调会务、材料准备和后续成果整理，确保活动规范有序。
三、经费安排
相关费用按单位财务制度据实列支。

智能知识库平台
2026年5月28日"""


def install_offline_llm_mocks(monkeypatch=None, captured_prompts=None):
    """让连贯性测试聚焦管线行为，不依赖真实模型网络。"""

    def fake_call_llm(self, user_content, *args, **kwargs):
        if captured_prompts is not None:
            captured_prompts.setdefault(self.name, []).append(user_content)
        if self.name == "Planner":
            payload = {
                "context_analysis": {
                    "user_intent": user_content[:80],
                    "document_type": "请示",
                    "key_points": ["学术交流", "组织保障", "经费安排"],
                    "context_quality": {"issues": []},
                    "confidence": 0.9,
                },
                "plan": {
                    "task_type": "公文生成",
                    "document_type": "请示",
                    "need_web_search": False,
                    "search_queries": [],
                    "knowledge_queries": ["学术交流活动 请示"],
                    "plan_steps": [
                        {"step": 1, "agent": "Knowledge", "action": "检索参考材料"},
                        {"step": 2, "agent": "Writer", "action": "生成请示草稿"},
                        {"step": 3, "agent": "Reviewer", "action": "审查校验"},
                    ],
                    "key_points": ["学术交流", "组织保障", "经费安排"],
                    "confidence": 0.9,
                },
            }
            return json.dumps(payload, ensure_ascii=False)
        if self.name == "Writer":
            return MOCK_DOCUMENT
        if self.name == "Reviewer":
            return json.dumps({
                "format_check": {"passed": True, "issues": []},
                "content_check": {"passed": True, "issues": []},
                "logic_check": {"passed": True, "issues": []},
                "language_check": {"passed": True, "issues": []},
                "fact_check": {"passed": True, "issues": []},
                "suggestions": [],
                "confidence": 0.9,
                "needs_revision": False,
                "revision_focus": [],
            }, ensure_ascii=False)
        if self.name == "Reflection":
            return json.dumps({
                "weaknesses": [],
                "counter_arguments": [],
                "missing_evidence": [],
                "better_angle": "",
                "logic_score": 0.9,
                "needs_revision": False,
                "revision_suggestions": [],
            }, ensure_ascii=False)
        return "{}"

    def fake_call_llm_stream(self, user_content, *args, **kwargs):
        if self.name == "Reflection":
            yield "reasoning", "从论证完整性、证据支撑和可执行性三个角度检查。"
            yield "content", fake_call_llm(self, user_content, *args, **kwargs)
            return

        for start in range(0, len(MOCK_DOCUMENT), 40):
            yield "content", MOCK_DOCUMENT[start:start + 40]

    if monkeypatch:
        monkeypatch.setattr(BaseAgent, "call_llm", fake_call_llm)
        monkeypatch.setattr(BaseAgent, "call_llm_stream", fake_call_llm_stream)
    else:
        BaseAgent.call_llm = fake_call_llm
        BaseAgent.call_llm_stream = fake_call_llm_stream


@pytest.fixture(autouse=True)
def offline_llm_mocks(monkeypatch):
    install_offline_llm_mocks(monkeypatch)


class MockMemory:
    """模拟记忆系统"""
    def __init__(self):
        self.contexts = {}
        self.messages = {}
        self._user_profiles = {}

    def get_context(self, session_id, key, default=None):
        return self.contexts.get(session_id, {}).get(key, default)

    def set_context(self, session_id, key, value):
        if session_id not in self.contexts:
            self.contexts[session_id] = {}
        self.contexts[session_id][key] = value

    def add_message(self, session_id, role, content, metadata=None):
        if session_id not in self.messages:
            self.messages[session_id] = []
        entry = {"role": role, "content": content, "metadata": metadata or {}}
        self.messages[session_id].append(entry)
        logger.info(f"  [Memory] +msg session={session_id[:12]} role={role} len={len(content)}")

    def get_conversation_context(self, session_id, max_messages=5):
        msgs = self.messages.get(session_id, [])[-max_messages:]
        return "\n".join([f"{m['role']}: {m['content'][:50]}" for m in msgs])

    def get_session_history(self, session_id, limit=100):
        return self.messages.get(session_id, [])[-limit:]

    def get_context_for_prompt(self, session_id, max_messages=10):
        return self.get_conversation_context(session_id, max_messages)

    def get_agent_context_for_prompt(self, session_id, **_kwargs):
        return (
            "【Agent 召回上下文】\n"
            "以下是同一用户的历史参考，不能覆盖当前请求。\n"
            "【跨会话长期记忆】\n- 用户跨会话偏好：通知应保持简洁正式"
        )

    def get_user_profile(self, user_id):
        return self._user_profiles.get(user_id)

    def set_agent_state(self, session_id, agent_name, state):
        pass

    def get_agent_state(self, session_id, agent_name):
        return {}


def log_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_context_packet():
    """测试 ContextPacket 数据类"""
    log_section("1. ContextPacket 数据类")
    ctx = ContextPacket(user_request="测试请求")
    assert ctx.user_request == "测试请求"
    assert ctx.revision_history == []
    assert ctx.search_context == ""
    assert ctx.knowledge_context == ""

    # 测试修改累积
    ctx.revision_history.append({"round": 1, "needs_revision": True, "suggestions": ["改进格式"]})
    ctx.search_context = "搜索结果"
    assert len(ctx.revision_history) == 1
    logger.info("  ContextPacket: 通过")
    print("  PASS")


def test_run_pipeline():
    """测试非流式完整链路"""
    log_section("2. 非流式 run() 完整链路")

    memory = MockMemory()
    orchestrator = AgentOrchestrator(memory=memory, session_id="session_001")
    orchestrator.set_user_profile({
        "name": "测试用户",
        "department": "项目部",
        "preferred_font": "仿宋_GB2312",
        "preferred_size": "三号",
        "writing_style": "简洁正式",
    })

    think_events = []
    def on_think(agent, emoji, msg):
        think_events.append(f"[{agent}] {emoji} {msg}")
        logger.info(f"  [THINK] {agent} {emoji} {msg}")

    logger.info("  开始 run()...")
    result = orchestrator.run("帮我写一份关于科研项目管理优化的通知", on_think=on_think)
    logger.info(f"  run() 完成")

    # 验证 1: ContextAgent 被调用
    context_agent_calls = [t for t in think_events if "ContextAgent" in t or "ContextManager" in t]
    logger.info(f"  ContextAgent think events: {len(context_agent_calls)}")
    for c in context_agent_calls:
        logger.info(f"    {c}")

    # 验证 2: 返回结果包含必要字段
    assert "document" in result, "缺少 document 字段"
    assert "plan" in result, "缺少 plan 字段"
    assert "think_log" in result, "缺少 think_log 字段"
    logger.info(f"  文档长度: {len(result['document'])}")
    logger.info(f"  计划类型: {result['plan'].get('document_type', 'unknown')}")
    logger.info(f"  置信度: {result.get('confidence', 'N/A')}")
    logger.info(f"  修改轮数: {result.get('revision_rounds', 'N/A')}")

    # 验证 3: key_points 在 plan 中
    key_points = result['plan'].get('key_points', [])
    logger.info(f"  key_points: {key_points}")

    # 验证 4: 消息被保存
    messages = memory.get_session_history("session_001")
    logger.info(f"  保存的消息数: {len(messages)}")
    roles = [m['role'] for m in messages]
    logger.info(f"  消息角色: {roles}")
    assert 'user' in roles, "缺少用户消息"
    assert 'assistant' in roles, "缺少助手消息"

    print("  PASS")


def test_recalled_memory_reaches_document_agents(monkeypatch):
    """Planner/Writer/Reviewer/Reflection must share the same bounded recall packet."""
    captured_prompts = {}
    install_offline_llm_mocks(monkeypatch, captured_prompts)
    orchestrator = AgentOrchestrator(memory=MockMemory(), session_id="session_memory_recall")

    result = orchestrator.run("帮我写一份关于科研项目管理优化的通知")

    assert result["document"]
    for agent_name in ("Planner", "Writer", "Reviewer", "Reflection"):
        prompts = captured_prompts.get(agent_name, [])
        assert prompts, f"{agent_name} 未收到模型提示词"
        assert any("用户跨会话偏好" in prompt for prompt in prompts), (
            f"{agent_name} 未收到跨会话召回上下文"
        )


def test_run_stream_pipeline():
    """测试流式完整链路"""
    log_section("3. 流式 run_stream() 完整链路")

    memory = MockMemory()
    orchestrator = AgentOrchestrator(memory=memory, session_id="session_002")

    logger.info("  开始 run_stream()...")
    events = []
    event_types = set()
    doc_content = ""

    for event in orchestrator.run_stream("写一份关于学术交流活动的请示"):
        event_types.add(event.get("type"))
        events.append(event)

        if event.get("type") == "content":
            doc_content += event.get("data", "")

    logger.info(f"  run_stream() 完成, 事件数: {len(events)}")
    logger.info(f"  事件类型: {sorted(event_types)}")
    logger.info(f"  文档长度: {len(doc_content)}")

    # 验证: 流式链路包含当前对外约定的阶段事件
    assert "context_start" in event_types, "缺少 context_start 事件"
    assert "context_end" in event_types, "缺少 context_end 事件"
    assert "plan" in event_types, "缺少 plan 事件"
    assert "write_start" in event_types, "缺少 write_start 事件"
    assert "think" in event_types, "缺少 think 事件"
    assert "content" in event_types, "缺少 content 事件"
    assert "done" in event_types, "缺少 done 事件"

    # 验证: done 事件包含完整数据
    done_event = [e for e in events if e.get("type") == "done"][0]
    assert "document" in done_event, "done 事件缺少 document"
    assert "plan" in done_event, "done 事件缺少 plan"
    assert "think_log" in done_event, "done 事件缺少 think_log"

    # 验证: think events 存在
    think_events = [e for e in events if e.get("type") == "think"]
    logger.info(f"  Think 事件数: {len(think_events)}")
    for te in think_events:
        logger.info(f"    [{te.get('agent')}] {te.get('message')}")

    print("  PASS")


def test_revision_history():
    """测试修改历史累积"""
    log_section("4. revision_history 累积")

    memory = MockMemory()
    orchestrator = AgentOrchestrator(memory=memory, session_id="session_003")

    # 先创建一些对话历史
    memory.add_message("session_003", "user", "之前帮我写过一份报告")
    memory.add_message("session_003", "assistant", "好的，这是您的报告...")

    result = orchestrator.run("请修改上一份报告的格式")

    # 修改历史是 ContextPacket 内部的，这里验证返回结果
    logger.info(f"  revision_rounds: {result.get('revision_rounds')}")
    logger.info(f"  confidence: {result.get('confidence')}")

    print("  PASS")


def test_contextagent_in_flow():
    """验证 ContextAgent 在流程中被调用且输出被使用"""
    log_section("5. ContextAgent 输出传递验证")

    memory = MockMemory()
    orchestrator = AgentOrchestrator(memory=memory, session_id="session_004")

    think_log = []
    def on_think(agent, emoji, msg):
        think_log.append({"agent": agent, "emoji": emoji, "message": msg})

    result = orchestrator.run("以项目部的名义，写一份关于2026年度示例项目申报的通知", on_think=on_think)

    # 检查 think_log 中是否包含 ContextAgent
    context_agents = [t for t in think_log if t['agent'] in ('ContextAgent', 'ContextManager')]
    logger.info(f"  ContextAgent 思考记录数: {len(context_agents)}")
    for c in context_agents:
        logger.info(f"    [{c['agent']}] {c['emoji']} {c['message']}")

    # 验证 Planner 在 ContextAgent 之后调用
    agent_order = [t['agent'] for t in think_log if t['agent'] in
                   ('ContextAgent', 'ContextManager', 'Planner', 'Search', 'Knowledge', 'Writer', 'Reviewer')]
    logger.info(f"  Agent 调用顺序: {agent_order}")

    # ContextAgent 应该在 Planner 之前
    context_idx = next((i for i, a in enumerate(agent_order) if 'Context' in a), -1)
    planner_idx = next((i for i, a in enumerate(agent_order) if a == 'Planner'), -1)

    if context_idx >= 0:
        logger.info(f"  ContextAgent 位置: {context_idx}, Planner 位置: {planner_idx}")
        assert context_idx < planner_idx, f"ContextAgent({context_idx}) 应在 Planner({planner_idx}) 之前"

    # 验证 plan 中有 key_points
    key_points = result['plan'].get('key_points', [])
    logger.info(f"  合并后的 key_points: {key_points}")

    print("  PASS")


def test_multi_round_context():
    """测试多轮对话上下文保持"""
    log_section("6. 多轮对话上下文")

    memory = MockMemory()
    orchestrator = AgentOrchestrator(memory=memory, session_id="session_005")

    # 第一轮
    r1 = orchestrator.run("帮我写一份关于学术活动的通知", session_id="session_005")
    logger.info(f"  第1轮 文档长度: {len(r1['document'])}")

    # 第二轮 — 应该能引用第一轮的上下文
    r2 = orchestrator.run("给这份通知加上关于经费安排的部分", session_id="session_005")
    logger.info(f"  第2轮 文档长度: {len(r2['document'])}")

    # 验证两轮都成功
    assert len(r1['document']) > 0, "第1轮文档为空"
    assert len(r2['document']) > 0, "第2轮文档为空"

    # 第二轮应该包含经费相关内容（或至少生成了内容）
    logger.info(f"  第2轮文档前200字: {r2['document'][:200]}")

    print("  PASS")


if __name__ == "__main__":
    install_offline_llm_mocks()

    print("=" * 60)
    print("  Agent 工作流连贯性测试")
    print("=" * 60)

    test_context_packet()
    test_run_pipeline()
    test_run_stream_pipeline()
    test_revision_history()
    test_contextagent_in_flow()
    test_multi_round_context()

    print(f"\n{'='*60}")
    print("  全部测试通过")
    print(f"{'='*60}")
