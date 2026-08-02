import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.base_agent import BaseAgent, AgentResult
from agents.knowledge_agent import KnowledgeAgent
from agents.orchestrator import AgentOrchestrator
from agents.writer_agent import WriterAgent


MOCK_DOCUMENT = """关于人工智能发展的报告

有关单位：
近年来，人工智能技术加速演进，正在推动科研管理、产业升级和公共服务模式创新。现将有关情况报告如下：
一、总体情况
人工智能应用场景持续拓展，数据、算力和算法协同能力明显增强。
二、主要问题
部分场景仍存在数据质量不足、应用标准不统一和复合型人才短缺等问题。
三、工作建议
建议完善应用规范，加强试点示范，推动技术应用与业务需求深度结合。

智能知识库平台
2026年5月28日"""


class MockMemory:
    def __init__(self):
        self.contexts = {}
        self.messages = {}
        self._user_profiles = {}

    def get_context(self, session_id, key, default=None):
        return self.contexts.get(session_id, {}).get(key, default)

    def set_context(self, session_id, key, value):
        self.contexts.setdefault(session_id, {})[key] = value

    def add_message(self, session_id, role, content, metadata=None):
        self.messages.setdefault(session_id, []).append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
        })

    def get_conversation_context(self, session_id, max_messages=5):
        messages = self.messages.get(session_id, [])[-max_messages:]
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    def get_session_history(self, session_id, limit=100):
        return self.messages.get(session_id, [])[-limit:]

    def get_context_for_prompt(self, session_id, max_messages=10):
        return self.get_conversation_context(session_id, max_messages)

    def get_user_profile(self, user_id):
        return self._user_profiles.get(user_id)

    def set_agent_state(self, session_id, agent_name, state):
        pass

    def get_agent_state(self, session_id, agent_name):
        return {}


def install_offline_mocks(monkeypatch=None):
    def fake_load_index(self):
        self.model = None
        self.device = "cpu"
        self.faiss_index = None
        self.index_data = {"texts": [], "metadatas": []}
        self.bm25 = None

    def fake_knowledge_process(self, input_data, on_think=None):
        if on_think:
            on_think(self.name, "📚", "使用离线知识库 mock")
        return AgentResult(
            success=True,
            content="离线知识库参考：人工智能应用需关注场景、数据治理和人才保障。",
            agent_name=self.name,
            confidence=0.8,
            metadata={"results": []},
        )

    def fake_call_llm(self, user_content, *args, **kwargs):
        if self.name == "Planner":
            payload = {
                "context_analysis": {
                    "user_intent": user_content[:80],
                    "document_type": "报告",
                    "key_points": ["人工智能", "应用场景", "治理建议"],
                    "context_quality": {"issues": []},
                    "confidence": 0.9,
                },
                "plan": {
                    "task_type": "公文生成",
                    "document_type": "报告",
                    "need_web_search": False,
                    "search_queries": [],
                    "knowledge_queries": ["人工智能 应用 报告"],
                    "plan_steps": [
                        {"step": 1, "agent": "Knowledge", "action": "检索本地资料"},
                        {"step": 2, "agent": "Writer", "action": "生成报告"},
                        {"step": 3, "agent": "Reviewer", "action": "审查校验"},
                    ],
                    "key_points": ["人工智能", "应用场景", "治理建议"],
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
            yield "reasoning", "离线推理：结构完整，建议可执行。"
            yield "content", fake_call_llm(self, user_content, *args, **kwargs)
            return
        for start in range(0, len(MOCK_DOCUMENT), 50):
            yield "content", MOCK_DOCUMENT[start:start + 50]

    if monkeypatch:
        monkeypatch.setattr(KnowledgeAgent, "_load_index", fake_load_index)
        monkeypatch.setattr(KnowledgeAgent, "process", fake_knowledge_process)
        monkeypatch.setattr(BaseAgent, "call_llm", fake_call_llm)
        monkeypatch.setattr(BaseAgent, "call_llm_stream", fake_call_llm_stream)
    else:
        KnowledgeAgent._load_index = fake_load_index
        KnowledgeAgent.process = fake_knowledge_process
        BaseAgent.call_llm = fake_call_llm
        BaseAgent.call_llm_stream = fake_call_llm_stream


@pytest.fixture(autouse=True)
def offline_mocks(monkeypatch):
    install_offline_mocks(monkeypatch)


def test_env_variable_loading():
    print("=== 测试环境变量加载 ===")

    class TestAgent(BaseAgent):
        def get_system_prompt(self) -> str:
            return "Test prompt"

        def process(self, input_data: dict, on_think=None):
            return None

    agent = TestAgent(name="Test", description="Test agent")
    assert agent.base_url == "https://api.deepseek.com/v1"
    print(f"API Key 存在: {bool(agent.api_key)}")
    print(f"Base URL: {agent.base_url}\n")


def test_revision_focus_bug():
    print("=== 测试 revision_focus 变量未定义的 bug ===")
    memory = MockMemory()
    orchestrator = AgentOrchestrator(memory=memory, session_id="test_session")
    result = orchestrator.run("帮我写一份关于人工智能发展的报告")
    assert result["document"]
    assert result["revision_rounds"] == 0
    print("✓ 测试通过：revision_focus 变量问题已修复")
    print(f"生成的文档长度: {len(result['document'])}")
    print(f"修订轮数: {result['revision_rounds']}\n")


def test_json_parsing():
    print("=== 测试 JSON 解析错误处理 ===")
    from agents.planner_agent import PlannerAgent

    planner = PlannerAgent()
    test_cases = [
        '{"document_type": "报告", "need_web_search": false, "search_queries": [], "knowledge_queries": [], "plan_steps": [], "key_points": [], "confidence": 0.8}',
        '```json\n{"document_type": "报告", "need_web_search": false, "search_queries": [], "knowledge_queries": [], "plan_steps": [], "key_points": [], "confidence": 0.8}\n```',
        '{"document_type": "报告", "need_web_search": false, "search_queries": [], "knowledge_queries": [], "plan_steps": [], "key_points": [], "confidence": 0.8,}',
    ]

    for i, test_case in enumerate(test_cases, 1):
        result = planner._parse_json_response(test_case)
        assert result.get("document_type") == "报告"
        print(f"✓ 测试用例 {i} 通过: {result.get('document_type')}")
    print()


def test_streaming():
    print("=== 测试流式运行 ===")
    memory = MockMemory()
    orchestrator = AgentOrchestrator(memory=memory, session_id="test_session")

    plan = None
    think_log = []
    content = ""

    for event in orchestrator.run_stream("帮我写一份简短的通知"):
        if event.get("type") == "plan":
            plan = event.get("data", {})
        elif event.get("type") == "content":
            content += event.get("data", "")
        elif event.get("type") == "done":
            think_log = event.get("think_log", [])

    assert plan and plan.get("document_type") == "报告"
    assert "关于人工智能发展的报告" in content
    assert think_log
    print("✓ 流式运行成功")
    print(f"计划类型: {plan.get('document_type')}")
    print(f"思考日志条目数: {len(think_log)}")
    print(f"输出内容: {content[:60]}...\n")


def test_writer_stream_disconnect_fallback():
    print("=== 测试 Writer 流式断开回退 ===")
    writer = WriterAgent(max_retry=1)

    def broken_stream(*args, **kwargs):
        yield "content", "半截内容"
        raise RuntimeError("incomplete chunked read")

    writer.call_llm_stream = broken_stream
    writer.call_llm = lambda *args, **kwargs: MOCK_DOCUMENT

    content = "".join(writer.process_stream({
        "user_request": "帮我写一份关于人工智能发展的报告",
        "document_type": "报告",
    }))
    assert content == MOCK_DOCUMENT
    print("✓ 流式断开后已回退非流式生成\n")


def test_error_handling():
    print("=== 测试错误处理 ===")
    memory = MockMemory()
    orchestrator = AgentOrchestrator(memory=memory, session_id="test_session")
    assert orchestrator.run("")["document"]
    assert orchestrator.run("帮我写一份关于 &*() 特殊字符的报告")["document"]
    print("✓ 空请求测试通过")
    print("✓ 特殊字符测试通过\n")


def test_session_management():
    print("=== 测试会话管理 ===")
    memory = MockMemory()
    orchestrator = AgentOrchestrator(memory=memory, session_id="session1")

    result1 = orchestrator.run("帮我写一份关于人工智能的报告")
    result2 = orchestrator.run("帮我写一份关于环境保护的报告", session_id="session2")
    result3 = orchestrator.run("继续写之前的报告", session_id="session1")

    assert result1["session_id"] == "session1"
    assert result2["session_id"] == "session2"
    assert result3["session_id"] == "session1"
    print(f"✓ 第一次请求成功，会话ID: {result1['session_id']}")
    print(f"✓ 会话切换成功，新会话ID: {result2['session_id']}")
    print(f"✓ 会话恢复成功，会话ID: {result3['session_id']}\n")


if __name__ == "__main__":
    install_offline_mocks()

    print("开始测试 Agent 潜在 bug（离线回归）...\n")

    test_env_variable_loading()
    test_revision_focus_bug()
    test_json_parsing()
    test_streaming()
    test_writer_stream_disconnect_fallback()
    test_error_handling()
    test_session_management()

    print("测试完成！")
