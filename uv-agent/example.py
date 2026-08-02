"""
阿里云百炼 + LangChain 代码助手示例
使用方法：
1. 复制 .env.example 为 .env，填入 API Key
2. 运行 python example.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# ^^^ 在 .env 文件中设置: DASHSCOPE_API_KEY=你的阿里云百炼API密钥

if not DASHSCOPE_API_KEY:
    raise ValueError("请在 .env 文件中设置 DASHSCOPE_API_KEY")

os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

llm = ChatOpenAI(
    model="qwen3.5-plus",
    # ^^^ Coding Plan 模型:
    #     - qwen3.6-plus   (通用对话)
    #     - qwen3.5-plus   (代码优化) ✓ 推荐
    base_url="https://coding.dashscope.aliyuncs.com/v1",
    api_key=DASHSCOPE_API_KEY,
)

def code_review(code: str) -> str:
    """代码审查示例"""
    prompt = f"""请审查以下代码，指出潜在问题和改进建议：

{code}

只返回审查结果，不要其他解释。"""
    return llm.invoke(prompt)

def explain_code(code: str) -> str:
    """代码解释示例"""
    prompt = f"""请解释以下代码的功能和工作原理：

{code}"""
    return llm.invoke(prompt)

if __name__ == "__main__":
    sample_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

    print("=== 代码审查 ===")
    review = code_review(sample_code)
    print(review)

    print("\n=== 代码解释 ===")
    explanation = explain_code(sample_code)
    print(explanation)