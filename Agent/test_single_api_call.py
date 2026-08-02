import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

# 加载当前项目目录下的环境变量
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"

MODELS = [
    "qwen3.5-plus",
    "qwen-turbo",
    "qwen-coder-plus",
]


def run_dashscope_smoke() -> bool:
    if not DASHSCOPE_API_KEY:
        print("❌ API Key 未配置")
        return False

    print("开始测试单次API调用...")
    print("=" * 60)

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )

    all_ok = True
    for model in MODELS:
        print(f"\n测试模型: {model}")
        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个助手"},
                    {"role": "user", "content": "请简单介绍一下你自己"},
                ],
                temperature=0.7,
                max_tokens=100,
            )
            end_time = time.time()
            response_time = end_time - start_time

            content = response.choices[0].message.content
            print(f"✅ 调用成功，响应时间: {response_time:.2f}秒")
            print(f"响应内容: {content[:100]}...")

        except Exception as e:
            all_ok = False
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ 调用失败，耗时: {response_time:.2f}秒")
            print(f"错误: {str(e)[:100]}...")

    print("\n" + "=" * 60)
    print("测试完成！")
    return all_ok


def test_dashscope_api_smoke():
    if os.getenv("RUN_EXTERNAL_API_TESTS") != "1":
        import pytest
        pytest.skip("设置 RUN_EXTERNAL_API_TESTS=1 后再运行 DashScope 外部 API 烟测")
    if not DASHSCOPE_API_KEY:
        import pytest
        pytest.skip("DASHSCOPE_API_KEY 未配置，跳过外部 API 烟测")
    assert run_dashscope_smoke()


if __name__ == "__main__":
    sys.exit(0 if run_dashscope_smoke() else 1)
