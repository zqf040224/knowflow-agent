"""
阿里云百炼 + LangChain 公文助手示例

使用方法：
1. 确保 .env 中已设置 DASHSCOPE_API_KEY
2. 运行 python dashscope_example.py

归档说明：本文件是早期 DashScope/LangChain 示例，不属于当前主服务。
"""

import os
from dotenv import load_dotenv
try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    ChatOpenAI = None

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

llm = None
if DASHSCOPE_API_KEY and ChatOpenAI:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://coding.dashscope.aliyuncs.com/v1",
        api_key=DASHSCOPE_API_KEY,
    )


def generate_document(requirement: str) -> str:
    """根据需求生成公文"""
    if llm is None:
        raise RuntimeError("DashScope 示例已归档，或缺少 langchain_openai/DASHSCOPE_API_KEY")
    prompt = f"""你是一个专业的公文写作助手，遵循《党政机关公文格式国家标准》。

请根据以下需求生成规范公文：

{requirement}

要求：
1. 标题准确、简洁
2. 主送单位明确
3. 正文层次分明（一、二、三...（一）（二）...）
4. 语言正式、规范
5. 落款完整

直接输出纯公文内容，不要加任何说明。"""
    return llm.invoke(prompt)


def check_format(document: str) -> str:
    """检查公文格式"""
    if llm is None:
        raise RuntimeError("DashScope 示例已归档，或缺少 langchain_openai/DASHSCOPE_API_KEY")
    prompt = f"""请检查以下公文的格式是否符合《党政机关公文格式国家标准》，如有问题请指出：

{document}

从以下维度检查：
1. 标题格式
2. 主送单位
3. 正文层次结构
4. 落款格式
5. 标点符号使用"""
    return llm.invoke(prompt)


if __name__ == "__main__":
    print("此 DashScope 示例已归档，不作为当前主服务运行入口。")
    raise SystemExit(0)
    requirement = "关于某市AI产业发展机遇的对策建议"

    print("=" * 60)
    print(f"需求：{requirement}")
    print("=" * 60)

    print("\n正在生成公文...")
    doc = generate_document(requirement)
    print("\n生成的公文：")
    print("-" * 40)
    print(doc)
    print("-" * 40)

    print("\n正在检查格式...")
    feedback = check_format(doc)
    print("\n格式检查反馈：")
    print(feedback)
