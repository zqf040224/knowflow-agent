#!/usr/bin/env python3
"""
DeepSeek API 测试脚本
测试 LLM 调用并测量耗时
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import pytest

# 加载环境变量
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

def test_deepseek_api():
    """测试 DeepSeek API 调用"""
    if __name__ != "__main__" and os.getenv("RUN_EXTERNAL_API_TESTS") != "1":
        pytest.skip("设置 RUN_EXTERNAL_API_TESTS=1 后再运行 DeepSeek 外部 API 烟测")

    print("=" * 60)
    print("DeepSeek API 测试")
    print("=" * 60)
    
    # 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("❌ 错误：未找到 DEEPSEEK_API_KEY，请检查 .env 文件")
        return
    
    print(f"✓ API Key 已加载: {DEEPSEEK_API_KEY[:10]}...")
    print(f"✓ Base URL: {DEEPSEEK_BASE_URL}")
    
    # 初始化客户端
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    
    print("\n开始测试...")
    print("-" * 60)
    
    # 测试用例
    test_cases = [
        "请简单介绍一下自己。",
        "请用一句话概括人工智能的定义。",
        "请写一首关于春天的四句古诗。",
    ]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n【测试 {i}/{len(test_cases)}】")
        print(f"输入: {prompt}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 调用 API
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            
            # 计算耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 获取响应
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            print(f"✓ 响应成功!")
            print(f"输出: {content}")
            print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
            print(f"🔢 Token 使用量: {tokens_used}")
            
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"❌ 调用失败!")
            print(f"错误信息: {str(e)}")
            print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_deepseek_api()
