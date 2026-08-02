#!/usr/bin/env python3
"""
测试博查 API 是否可用
"""
import os
import sys
from dotenv import load_dotenv

if "pytest" in sys.modules and __name__ != "__main__":
    import pytest
    pytest.skip("test_bocha.py is an external API smoke script; run it directly.", allow_module_level=True)

# 加载环境变量
project_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(project_env):
    load_dotenv(project_env)
    print("✓ 从项目 .env 加载环境变量")
else:
    load_dotenv()

BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")

print("\n" + "=" * 60)
print("博查 API 测试")
print("=" * 60)

if not BOCHA_API_KEY:
    print("❌ BOCHA_API_KEY 未设置")
    exit(1)

print(f"✓ API Key 已配置: {BOCHA_API_KEY[:15]}...")

# 测试 API
try:
    import requests

    url = "https://api.bochaai.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {BOCHA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": "人工智能产业发展 2024",
        "count": 3,
        "summary": True,
    }

    print("\n发送测试请求...")
    response = requests.post(url, headers=headers, json=payload, timeout=15)

    if response.status_code == 200:
        data = response.json()
        print("✓ API 调用成功！")

        results = data.get("data", {}).get("webPages", {}).get("value", [])
        print(f"\n获取到 {len(results)} 条结果:")
        for i, item in enumerate(results[:3], 1):
            print(f"  {i}. {item.get('name', '无标题')}")
            print(f"     {item.get('summary') or item.get('snippet', '')[:100]}...")

        print("\n✅ 博查 API 工作正常！")
    else:
        print(f"❌ API 错误: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ 测试失败: {e}")
