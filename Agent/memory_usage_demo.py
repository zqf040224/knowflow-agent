"""
团队记忆系统使用示例

展示如何在 Agent 系统中集成新的记忆模块
"""

from memory_v2 import TeamMemory, get_memory

# ========== 使用方式 1: 直接实例化 ==========

memory = TeamMemory("./data/team_memory.db")

# ========== 使用方式 2: 单例模式（推荐） ==========

# 在应用启动时初始化
memory = get_memory("./data/team_memory.db")

# ========== 典型使用流程 ==========

def handle_user_request(user_id: str, user_message: str):
    """处理用户请求"""

    # 1. 获取或创建用户
    profile = memory.get_or_create_user(
        user_id=user_id,
        name="",  # 可以从登录信息获取
        department=""  # 可以从登录信息获取
    )

    # 2. 获取或创建会话
    session_id = memory.get_or_create_session(user_id)

    # 3. 获取用户偏好（用于 Prompt）
    user_prefs = {
        'font': profile.preferred_font,
        'size': profile.preferred_size,
        'style': profile.writing_style,
        'common_types': profile.common_doc_types
    }

    # 4. 获取会话上下文
    context_messages = memory.get_formatted_messages(session_id, max_messages=10)

    # 5. 构建 LLM Prompt
    system_prompt = f"""你是公文写作助手。
用户偏好：
- 字体：{user_prefs['font']}
- 字号：{user_prefs['size']}
- 风格：{user_prefs['style']}
- 常用公文：{', '.join(user_prefs['common_types'])}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        *context_messages,
        {"role": "user", "content": user_message}
    ]

    # 6. 调用 LLM（示例）
    # response = llm.invoke(messages)
    response_content = "这是模拟的助手回复"

    # 7. 保存消息到记忆
    memory.add_message(session_id, "user", user_message)
    memory.add_message(session_id, "assistant", response_content,
                      metadata={"tokens": 100})  # 可以保存额外信息

    # 8. 更新用户画像（可选）
    if "通知" in user_message:
        new_types = list(set(profile.common_doc_types + ["通知"]))
        memory.update_user_profile(user_id, {
            "common_doc_types": new_types
        })

    return response_content


# ========== 会话管理示例 ==========

def list_user_history(user_id: str):
    """列出用户的历史会话"""
    sessions = memory.list_user_sessions(user_id, limit=10)

    print(f"\n用户 {user_id} 的历史会话:")
    for session in sessions:
        print(f"  - {session['title'] or '未命名会话'}")
        print(f"    类型: {session['doc_type'] or '通用'}")
        print(f"    消息数: {session['message_count']}")
        print(f"    更新时间: {session['updated_at']}")


def continue_session(user_id: str, session_id: str):
    """继续之前的会话"""
    # 获取会话历史
    history = memory.get_session_history(session_id)

    print(f"\n继续会话 {session_id}:")
    print(f"历史消息 ({len(history)} 条):")
    for msg in history:
        role = "用户" if msg.role == "user" else "助手"
        print(f"  [{role}] {msg.content[:50]}...")


def export_and_backup(user_id: str):
    """导出用户数据备份"""
    data = memory.export_user_data(user_id)

    import json
    with open(f"./backup/user_{user_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 用户 {user_id} 数据已导出")


# ========== 系统维护示例 ==========

def maintenance():
    """系统维护"""
    # 清理过期会话
    result = memory.cleanup_expired_sessions(days=7)
    print(f"\n清理完成:")
    print(f"  - 过期会话: {result['expired_sessions']}")
    print(f"  - 删除消息: {result['deleted_messages']}")

    # 查看统计
    stats = memory.get_stats()
    print(f"\n系统统计:")
    print(f"  - 用户数: {stats['users']}")
    print(f"  - 活跃会话: {stats['active_sessions']}")
    print(f"  - 今日消息: {stats['today_messages']}")


# ========== 完整示例运行 ==========

if __name__ == "__main__":
    import os

    # 使用测试数据库
    test_db = "./demo_memory.db"

    # 如果存在旧测试数据，先删除
    if os.path.exists(test_db):
        os.remove(test_db)

    print("=" * 60)
    print("团队记忆系统使用示例")
    print("=" * 60)

    memory = TeamMemory(test_db)

    # 模拟多用户使用
    users = [
        ("user_001", "示例用户甲", "示例部门甲"),
        ("user_002", "李四", "行政部"),
        ("user_003", "示例用户丙", "示例部门乙"),
    ]

    for user_id, name, dept in users:
        print(f"\n--- 用户 {name} ({dept}) ---")

        # 创建用户
        profile = memory.get_or_create_user(user_id, name, dept)

        # 设置不同偏好
        if dept == "示例部门甲":
            memory.update_user_profile(user_id, {
                "common_doc_types": ["项目申报", "对策建议", "研究报告"],
                "writing_style": "学术严谨"
            })
        elif dept == "行政部":
            memory.update_user_profile(user_id, {
                "common_doc_types": ["会议通知", "放假通知", "行政公告"],
                "writing_style": "简洁正式"
            })

        # 模拟对话
        session_id = memory.create_session(user_id, title=f"{name}的会话", doc_type="通知")

        messages = [
            ("user", f"我是{name}，帮我写一份{dept}的通知"),
            ("assistant", "好的，请告诉我具体需求"),
            ("user", "关于下周的部门会议"),
        ]

        for role, content in messages:
            memory.add_message(session_id, role, content)

        # 显示上下文
        context = memory.get_context_for_prompt(session_id)
        print(f"会话上下文:\n{context[:100]}...")

    # 显示所有用户统计
    print("\n" + "=" * 60)
    print("系统统计")
    print("=" * 60)
    stats = memory.get_stats()
    print(f"用户数: {stats['users']}")
    print(f"总会话: {stats['total_sessions']}")
    print(f"总消息: {stats['total_messages']}")

    # 显示用户详情
    print("\n" + "=" * 60)
    print("用户详情")
    print("=" * 60)
    for user_id, name, dept in users:
        profile = memory.get_user_profile(user_id)
        print(f"\n{name} ({dept}):")
        print(f"  偏好字体: {profile.preferred_font}")
        print(f"  常用公文: {profile.common_doc_types}")
        print(f"  写作风格: {profile.writing_style}")

        sessions = memory.list_user_sessions(user_id)
        print(f"  会话数: {len(sessions)}")

    # 清理测试数据
    print("\n" + "=" * 60)
    print("清理测试数据")
    print("=" * 60)
    os.remove(test_db)
    print("✓ 完成")
