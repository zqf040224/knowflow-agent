#!/usr/bin/env python3
"""
Agent CLI 工具
提供命令行界面与 Agent 交互
"""

import sys
import argparse
from config import check_config

def main():
    parser = argparse.ArgumentParser(description="Agent CLI 工具")
    parser.add_argument("--check", action="store_true", help="检查配置")
    parser.add_argument("--chat", nargs="+", help="与 Agent 对话")
    args = parser.parse_args()

    if args.check:
        check_config()
    elif args.chat:
        message = " ".join(args.chat)
        print(f"发送消息: {message}")
        # TODO: 实现聊天功能
        print("聊天功能开发中...")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()