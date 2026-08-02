"""
Agent 开发环境配置
使用 uv 管理依赖
"""

import os
from dotenv import load_dotenv

load_dotenv()

# 阿里云百炼配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# ^^^ 在 .env 文件中设置: DASHSCOPE_API_KEY=你的阿里云百炼API密钥

# OpenAI 配置 (可选)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Anthropic 配置 (可选)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 验证配置
def check_config():
    """检查 API 配置"""
    if DASHSCOPE_API_KEY:
        print("✅ 阿里云百炼 API Key 已配置")
        return True
    else:
        print("⚠️ 未检测到 API Key，请配置 .env 文件")
        return False

if __name__ == "__main__":
    check_config()