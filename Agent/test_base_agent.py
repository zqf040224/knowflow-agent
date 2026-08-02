
"""
测试基础Agent功能
"""

import sys

if "pytest" in sys.modules and __name__ != "__main__":
    import pytest
    pytest.skip("test_base_agent.py calls the real LLM; run it directly for integration smoke.", allow_module_level=True)

from agents.base_agent import BaseAgent, AgentResult


class TestAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return "你是一个测试助手，简单回答用户问题"
    
    def process(self, input_data: dict, on_think=None) -> AgentResult:
        user_request = input_data.get("user_request", "")
        response = self.call_llm(user_request, use_context=False)
        return AgentResult(
            success=True,
            content=response,
            agent_name=self.name,
            confidence=0.9
        )


def test_agent():
    print("=" * 60)
    print("测试基础Agent功能")
    print("=" * 60)
    
    agent = TestAgent(name="TestAgent", description="测试Agent")
    
    # 测试LLM调用
    print("测试LLM调用...")
    try:
        result = agent.process({"user_request": "你好，测试一下"})
        print(f"成功: {result.success}")
        print(f"响应: {result.content}")
        print(f"置信度: {result.confidence}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_agent()
