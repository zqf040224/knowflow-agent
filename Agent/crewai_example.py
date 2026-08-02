"""
CrewAI 多智能体公文生成示例（阿里云百炼）

使用方法：
1. 确保 .env 中已设置 DASHSCOPE_API_KEY
2. 运行 python crewai_example.py

归档说明：本文件是早期外部框架示例，不属于当前 Flask/LangGraph 主服务。
"""

import os
from dotenv import load_dotenv
try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    Agent = Task = Crew = ChatOpenAI = None

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


def create_document_crew():
    """创建一个公文生成团队"""
    if llm is None:
        raise RuntimeError("CrewAI 示例已归档，或缺少 crewai/langchain_openai/DASHSCOPE_API_KEY")

    researcher = Agent(
        role="政策研究员",
        goal="收集最新的政策信息和相关数据",
        backstory="你是一名政策研究员，擅长收集和整理最新的政策文件、统计数据和行业动态。",
        llm=llm,
        verbose=True
    )

    formatter = Agent(
        role="格式规范师",
        goal="确保公文格式符合国家标准",
        backstory="你是一名公文格式专家，精通《党政机关公文格式国家标准》，能够确保生成的公文格式规范。",
        llm=llm,
        verbose=True
    )

    writer = Agent(
        role="公文撰写员",
        goal="撰写规范、准确、逻辑清晰的公文",
        backstory="你是一名资深公文写作者，有多年撰写党政机关公文的经验，擅长将复杂信息组织成规范的公文文本。",
        llm=llm,
        verbose=True
    )

    research_task = Task(
        description="收集关于某市AI产业发展的最新政策、动态和数据，为撰写对策建议提供支撑材料。",
        agent=researcher
    )

    format_task = Task(
        description="提供对策建议公文的格式规范，包括标题、主送单位、正文层次结构、落款等格式要求。",
        agent=formatter
    )

    write_task = Task(
        description="基于研究员提供的材料和格式师提供的规范，撰写一份关于某市AI产业发展机遇的对策建议公文。",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, formatter, writer],
        tasks=[research_task, format_task, write_task],
        verbose=True
    )

    return crew


if __name__ == "__main__":
    print("此 CrewAI 示例已归档，不作为当前主服务运行入口。")
    raise SystemExit(0)
    print("=" * 60)
    print("启动公文生成团队...")
    print("=" * 60)

    crew = create_document_crew()
    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("生成的公文：")
    print("=" * 60)
    print(result)
