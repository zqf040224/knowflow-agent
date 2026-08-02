"""
短期记忆模块 - 为 Agent 系统提供短期记忆能力

功能：
1. 会话记忆（Session Memory）：保存多轮对话历史
2. Agent 上下文（Agent Context）：保存每个 Agent 的执行状态
3. 用户偏好（User Preference）：保存用户的设置和偏好
4. 临时上下文（Temp Context）：当前会话的临时信息

存储方式：
- 活跃会话：内存存储（字典）
- 持久化：SQLite（可选）
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import threading


@dataclass
class Message:
    """对话消息"""
    role: str  # 'user' 或 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(**data)


@dataclass
class Session:
    """会话对象"""
    session_id: str
    created_at: float
    last_active: float
    messages: List[Message] = field(default_factory=list)
    context: Dict = field(default_factory=dict)  # 会话级别的上下文
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'last_active': self.last_active,
            'messages': [m.to_dict() for m in self.messages],
            'context': self.context,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Session':
        session = cls(
            session_id=data['session_id'],
            created_at=data['created_at'],
            last_active=data['last_active'],
            context=data.get('context', {}),
            metadata=data.get('metadata', {})
        )
        session.messages = [Message.from_dict(m) for m in data.get('messages', [])]
        return session


class ShortTermMemory:
    """短期记忆管理器"""

    def __init__(self, max_sessions: int = 100, session_ttl: int = 3600):
        """
        初始化短期记忆

        Args:
            max_sessions: 最大会话数，超过会清理最早的会话
            session_ttl: 会话存活时间（秒），超过此时间未活跃的会话会被清理
        """
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, str] = {}  # user_id -> session_id 映射
        self._lock = threading.RLock()

    # ========== 会话管理 ==========

    def create_session(self, user_id: Optional[str] = None) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())[:8]  # 短ID便于使用

        with self._lock:
            # 清理过期会话
            self._cleanup_expired_sessions()

            # 如果超过最大会话数，清理最早的
            if len(self._sessions) >= self.max_sessions:
                self._cleanup_oldest_sessions(10)  # 清理10个最老的

            session = Session(
                session_id=session_id,
                created_at=time.time(),
                last_active=time.time()
            )
            self._sessions[session_id] = session

            # 绑定用户
            if user_id:
                self._user_sessions[user_id] = session_id

        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话，并更新最后活跃时间"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_active = time.time()
            return session

    def get_or_create_session(self, session_id: Optional[str] = None,
                              user_id: Optional[str] = None) -> str:
        """获取或创建会话"""
        # 尝试通过 user_id 查找
        if user_id and not session_id:
            session_id = self._user_sessions.get(user_id)

        # 检查会话是否存在
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session_id

        # 创建新会话
        return self.create_session(user_id)

    def delete_session(self, session_id: str):
        """删除会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            # 清理用户映射
            for user_id, sid in list(self._user_sessions.items()):
                if sid == session_id:
                    del self._user_sessions[user_id]

    def list_sessions(self) -> List[Dict]:
        """列出所有会话"""
        with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

    # ========== 消息管理 ==========

    def add_message(self, session_id: str, role: str, content: str,
                    metadata: Optional[Dict] = None) -> bool:
        """添加消息到会话"""
        session = self.get_session(session_id)
        if not session:
            return False

        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        session.messages.append(message)

        # 限制消息数量（保留最近50条）
        if len(session.messages) > 50:
            session.messages = session.messages[-50:]

        return True

    def get_messages(self, session_id: str, limit: int = 10) -> List[Message]:
        """获取会话的最近消息"""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.messages[-limit:]

    def get_conversation_context(self, session_id: str,
                                  max_messages: int = 5) -> str:
        """获取对话上下文，用于 Prompt"""
        messages = self.get_messages(session_id, max_messages)
        if not messages:
            return ""

        context_parts = []
        for msg in messages:
            role = "用户" if msg.role == 'user' else "助手"
            context_parts.append(f"{role}：{msg.content}")

        return "\n".join(context_parts)

    def clear_messages(self, session_id: str):
        """清空会话消息"""
        session = self.get_session(session_id)
        if session:
            session.messages = []

    # ========== 上下文管理 ==========

    def set_context(self, session_id: str, key: str, value: Any):
        """设置会话上下文"""
        session = self.get_session(session_id)
        if session:
            session.context[key] = value

    def get_context(self, session_id: str, key: str,
                    default: Any = None) -> Any:
        """获取会话上下文"""
        session = self.get_session(session_id)
        if session:
            return session.context.get(key, default)
        return default

    def update_context(self, session_id: str, updates: Dict):
        """批量更新会话上下文"""
        session = self.get_session(session_id)
        if session:
            session.context.update(updates)

    def get_all_context(self, session_id: str) -> Dict:
        """获取会话所有上下文"""
        session = self.get_session(session_id)
        return session.context if session else {}

    def clear_context(self, session_id: str):
        """清空会话上下文"""
        session = self.get_session(session_id)
        if session:
            session.context = {}

    # ========== Agent 状态管理 ==========

    def set_agent_state(self, session_id: str, agent_name: str,
                        state: Dict):
        """保存 Agent 的执行状态"""
        key = f"_agent_state_{agent_name}"
        self.set_context(session_id, key, state)

    def get_agent_state(self, session_id: str, agent_name: str) -> Optional[Dict]:
        """获取 Agent 的执行状态"""
        key = f"_agent_state_{agent_name}"
        return self.get_context(session_id, key)

    # ========== 清理机制 ==========

    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self.session_ttl
        ]
        for sid in expired:
            self.delete_session(sid)

    def _cleanup_oldest_sessions(self, count: int):
        """清理最早的会话"""
        sorted_sessions = sorted(
            self._sessions.items(),
            key=lambda x: x[1].created_at
        )
        for sid, _ in sorted_sessions[:count]:
            self.delete_session(sid)

    def cleanup_all(self):
        """清理所有会话"""
        with self._lock:
            self._sessions.clear()
            self._user_sessions.clear()

    # ========== 统计信息 ==========

    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self._lock:
            total_sessions = len(self._sessions)
            total_messages = sum(len(s.messages) for s in self._sessions.values())
            active_users = len(self._user_sessions)

            return {
                'total_sessions': total_sessions,
                'total_messages': total_messages,
                'active_users': active_users,
                'max_sessions': self.max_sessions,
                'session_ttl': self.session_ttl
            }


# ========== 全局实例 ==========

# 创建全局短期记忆实例
memory = ShortTermMemory()


# ========== 便捷函数 ==========

def get_memory() -> ShortTermMemory:
    """获取全局短期记忆实例"""
    return memory


def create_session(user_id: Optional[str] = None) -> str:
    """创建新会话"""
    return memory.create_session(user_id)


def get_session_context(session_id: str, max_messages: int = 5) -> str:
    """获取会话上下文"""
    return memory.get_conversation_context(session_id, max_messages)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("短期记忆模块测试")
    print("=" * 60)

    # 创建会话
    session_id = memory.create_session(user_id="user_001")
    print(f"\n创建会话: {session_id}")

    # 添加消息
    memory.add_message(session_id, "user", "帮我写一份会议通知")
    memory.add_message(session_id, "assistant", "好的，请告诉我会议主题和时间")
    memory.add_message(session_id, "user", "主题是关于AI发展的研讨会")

    # 获取上下文
    context = memory.get_conversation_context(session_id)
    print(f"\n对话上下文:\n{context}")

    # 设置上下文变量
    memory.set_context(session_id, "document_type", "会议通知")
    memory.set_context(session_id, "topic", "AI发展研讨会")

    # 获取上下文变量
    doc_type = memory.get_context(session_id, "document_type")
    print(f"\n文档类型: {doc_type}")

    # 保存 Agent 状态
    memory.set_agent_state(session_id, "Planner", {
        "plan": "生成会议通知",
        "steps": ["分析需求", "生成草稿", "审查"]
    })

    planner_state = memory.get_agent_state(session_id, "Planner")
    print(f"\nPlanner 状态: {planner_state}")

    # 统计
    stats = memory.get_stats()
    print(f"\n统计信息: {stats}")
