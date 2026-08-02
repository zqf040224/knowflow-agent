
"""
优化后的 Agent 系统演示

演示上下文中心架构的优势：
1. 智能上下文管理
2. 用户画像集成
3. 增强的错误处理
4. 性能监控

归档说明：本文件保留早期演示思路，不再作为可运行入口。
当前主流程请使用 agents.AgentOrchestrator、/api/chat 或 test_offline.py 验证。
"""

import time
from memory_v2 import get_memory


class ImprovedOrchestrator:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("ImprovedOrchestrator 已归档，请改用 agents.AgentOrchestrator")


def demo_basic_usage():
    """基本使用演示"""
    print("=" * 60)
    print("优化后 Agent 系统演示")
    print("=" * 60)
    
    # 初始化记忆系统
    memory = get_memory()
    
    # 创建用户ID和会话
    user_id = "demo_user_001"
    session_id = memory.create_session(
        user_id=user_id,
        title="AI发展研讨会通知",
        doc_type="会议通知"
    )
    
    print(f"用户ID: {user_id}")
    print(f"会话ID: {session_id}")
    
    # 初始化改进版 Orchestrator
    orchestrator = ImprovedOrchestrator(
        memory=memory,
        session_id=session_id,
        user_id=user_id
    )
    
    # 测试请求
    user_request = "帮我写一份关于AI发展研讨会的会议通知，时间定在5月10日，地点在示例会议中心"
    
    def think_handler(agent_name, emoji, message):
        print(f"[{emoji} {agent_name}] {message}")
    
    start_time = time.time()
    print("\n开始处理请求...")
    result = orchestrator.run(user_request, on_think=think_handler)
    end_time = time.time()
    
    print(f"\n处理完成，耗时: {end_time - start_time:.2f}秒")
    print("=" * 60)
    
    if result["success"]:
        print("\n生成的文档:")
        print("-" * 60)
        print(result["document"])
        print("-" * 60)
        
        print("\n执行统计:")
        print(f"  置信度: {result['confidence']:.2f}")
        print(f"  思考步骤: {len(result['think_log'])}")
    else:
        print(f"\n执行失败: {result['error']}")
    
    # 打印统计信息
    stats = orchestrator.get_stats()
    print("\n系统统计:")
    print(f"  总运行次数: {stats['orchestrator']['total_runs']}")
    print(f"  总耗时: {stats['orchestrator']['total_time']:.2f}秒")
    print(f"  平均置信度: {stats['orchestrator']['average_confidence']:.2f}")
    print(f"  错误次数: {stats['orchestrator']['error_count']}")


def demo_context_awareness():
    """上下文感知演示"""
    print("\n" + "=" * 60)
    print("上下文感知功能演示")
    print("=" * 60)
    
    memory = get_memory()
    user_id = "demo_user_002"
    session_id = memory.create_session(
        user_id=user_id,
        title="部门会议通知",
        doc_type="会议通知"
    )
    
    orchestrator = ImprovedOrchestrator(
        memory=memory,
        session_id=session_id,
        user_id=user_id
    )
    
    # 第一次请求
    first_request = "帮我写一份部门会议通知"
    
    def think_handler(agent_name, emoji, message):
        print(f"[{emoji} {agent_name}] {message}")
    
    print("\n第一次请求:")
    print(f"用户: {first_request}")
    result1 = orchestrator.run(first_request, on_think=think_handler)
    
    # 第二次请求（基于上下文）
    second_request = "把时间改为下周五下午2点"
    print("\n第二次请求:")
    print(f"用户: {second_request}")
    result2 = orchestrator.run(second_request, on_think=think_handler)
    
    print("\n第二次生成的文档:")
    print("-" * 60)
    print(result2["document"])
    print("-" * 60)


def demo_user_profile_integration():
    """用户画像集成演示"""
    print("\n" + "=" * 60)
    print("用户画像集成演示")
    print("=" * 60)
    
    memory = get_memory()
    user_id = "demo_user_003"
    
    # 创建用户并设置偏好
    user_profile = memory.get_or_create_user(
        user_id=user_id,
        name="张三",
        department="研发部"
    )
    
    # 更新用户偏好
    memory.update_user_profile(user_id, {
        "preferred_font": "黑体",
        "preferred_size": "四号",
        "writing_style": "简洁明了",
        "common_doc_types": ["会议通知", "工作报告"]
    })
    
    session_id = memory.create_session(
        user_id=user_id,
        title="周工作总结",
        doc_type="工作报告"
    )
    
    orchestrator = ImprovedOrchestrator(
        memory=memory,
        session_id=session_id,
        user_id=user_id
    )
    
    user_request = "帮我写一份周工作总结，重点汇报本周的项目进展"
    
    def think_handler(agent_name, emoji, message):
        print(f"[{emoji} {agent_name}] {message}")
    
    result = orchestrator.run(user_request, on_think=think_handler)
    
    print("\n生成的文档:")
    print("-" * 60)
    print(result["document"])
    print("-" * 60)


if __name__ == "__main__":
    print("此演示已归档：请使用 agents.AgentOrchestrator 或 test_offline.py。")
    raise SystemExit(0)
    try:
        # 运行演示
        demo_basic_usage()
        demo_context_awareness()
        demo_user_profile_integration()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示出错: {e}")
        import traceback
        traceback.print_exc()
