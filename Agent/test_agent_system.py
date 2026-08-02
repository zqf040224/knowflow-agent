#!/usr/bin/env python3
"""
Agent 系统测试脚本
测试实际的 Agent 类调用
"""

import sys
import os
import time
import logging

if "pytest" in sys.modules and __name__ != "__main__":
    import pytest
    pytest.skip("test_agent_system.py calls the real LLM; run it directly for integration smoke.", allow_module_level=True)

# 配置日志
logging.basicConfig(level=logging.INFO)

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

from agents.base_agent import BaseAgent

class TestAgent(BaseAgent):
    """简单测试 Agent"""
    def __init__(self):
        super().__init__(
            name="TestAgent",
            description="测试 Agent"
        )
    
    def get_system_prompt(self):
        return "你是一个测试助手，回复简洁明了。"
    
    def process(self, input_data, on_think=None):
        prompt = input_data.get("prompt", "")
        print(f"[Agent] 正在处理: {prompt[:50]}...")
        
        start_time = time.time()
        response = self.call_llm(prompt)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        return {
            "success": True,
            "content": response,
            "elapsed_time": elapsed_time
        }

def test_agent():
    """测试 Agent 系统"""
    print("=" * 60)
    print("Agent 系统测试")
    print("=" * 60)
    
    try:
        # 初始化 Agent
        print("\n正在初始化 Agent...")
        agent = TestAgent()
        print(f"✓ Agent 初始化成功")
        print(f"  - 名称: {agent.name}")
        print(f"  - 模型: {agent.model}")
        print(f"  - Base URL: {agent.base_url}")
        
        # 测试用例
        test_prompts = [
            "你好，请简单介绍一下自己。",
            "1+1等于几？",
            "请用3句话描述今天的天气很好。",
        ]
        
        print("\n开始测试 Agent 调用...")
        print("-" * 60)
        
        total_time = 0
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n【Agent 测试 {i}/{len(test_prompts)}】")
            print(f"输入: {prompt}")
            
            start_time = time.time()
            result = agent.process({"prompt": prompt})
            end_time = time.time()
            
            if result["success"]:
                elapsed_time = result["elapsed_time"]
                total_time += elapsed_time
                
                print(f"✓ 调用成功!")
                print(f"输出: {result['content']}")
                print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
            else:
                print(f"❌ 调用失败!")
        
        print("\n" + "-" * 60)
        avg_time = total_time / len(test_prompts)
        print(f"\n📊 测试结果汇总:")
        print(f"  - 总测试次数: {len(test_prompts)}")
        print(f"  - 总耗时: {total_time:.2f} 秒")
        print(f"  - 平均耗时: {avg_time:.2f} 秒")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_agent()
