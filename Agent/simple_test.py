
"""
简单测试脚本 - 测试优化后的agent系统
"""

import sys
import time
from agents import AgentOrchestrator

if "pytest" in sys.modules and __name__ != "__main__":
    import pytest
    pytest.skip("simple_test.py is an interactive LLM smoke script; run it directly.", allow_module_level=True)


def test_context_agent():
    """测试上下文管理Agent"""
    print("=" * 60)
    print("测试上下文管理Agent")
    print("=" * 60)
    
    # 初始化当前 Orchestrator（不使用内存系统以避免依赖问题）
    orchestrator = AgentOrchestrator()
    print(f"Orchestrator 初始化完成: {orchestrator.__class__.__name__}")
    
    # 测试请求
    user_request = "帮我写一份关于AI发展研讨会的会议通知，时间定在5月10日，地点在示例会议中心"
    
    def think_handler(agent_name, emoji, message):
        print(f"[{emoji} {agent_name}] {message}")
    
    start_time = time.time()
    print("\n开始处理请求...")
    
    # 模拟对话历史
    conversation_history = [
        {"role": "user", "content": "你好，我需要写一份会议通知"},
        {"role": "assistant", "content": "好的，请问是什么类型的会议？"},
    ]
    
    # 直接测试 ContextAgent
    from agents import ContextAgent
    context_agent = ContextAgent()
    
    result = context_agent.process({
        "conversation_history": conversation_history,
        "user_request": user_request,
        "user_profile": {"name": "张三", "department": "研发部"},
    }, on_think=think_handler)
    
    end_time = time.time()
    
    print(f"\n处理完成，耗时: {end_time - start_time:.2f}秒")
    print("=" * 60)
    
    if result.success:
        print("\n上下文分析结果:")
        print("-" * 60)
        print(result.content)
        print("-" * 60)
    else:
        print(f"\n执行失败: {result.content}")


def test_base_agent():
    """测试基础Agent功能"""
    print("\n" + "=" * 60)
    print("测试基础Agent功能")
    print("=" * 60)
    
    from agents import BaseAgent
    
    # 创建一个简单的测试Agent
    class TestAgent(BaseAgent):
        def get_system_prompt(self) -> str:
            return "你是一个测试助手，简单回答用户问题"
        
        def process(self, input_data: dict, on_think=None) -> str:
            user_request = input_data.get("user_request", "")
            return f"测试响应: {user_request}"
    
    agent = TestAgent(name="TestAgent", description="测试Agent")
    
    # 测试LLM调用
    response = agent.call_llm("你好，测试一下", use_context=False)
    print(f"\nLLM调用测试:")
    print(f"响应: {response}")


if __name__ == "__main__":
    try:
        # 运行测试
        test_context_agent()
        test_base_agent()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()
