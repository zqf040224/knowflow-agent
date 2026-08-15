import sqlite3
from datetime import datetime
from types import SimpleNamespace

from beta_ops_service import BetaActor, BetaOpsDependencies, BetaOpsService, BetaRequestMeta


class FakeMemory:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE auth_users (
                user_id TEXT,
                username TEXT,
                name TEXT,
                department TEXT,
                role TEXT,
                is_active INTEGER,
                created_at TEXT,
                last_login TEXT
            );
            CREATE TABLE sessions (
                session_id TEXT,
                user_id TEXT,
                title TEXT,
                doc_type TEXT,
                message_count INTEGER,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp INTEGER
            );
        """)
        self.conn.execute("""
            INSERT INTO auth_users
            VALUES ('u1', 'alice', 'Alice', '示例部门甲', 'user', 1, '2026-06-01T09:00:00', '2026-06-07T08:00:00')
        """)
        self.conn.execute("""
            INSERT INTO sessions
            VALUES ('s1', 'u1', '测试会话', '通知', 1, '2026-06-07T08:00:00', '2026-06-07T09:00:00')
        """)
        self.conn.execute("""
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES ('s1', 'user', 'hello', 1780790400)
        """)
        self.conn.commit()

    def _get_conn(self):
        return self.conn


def build_service():
    return BetaOpsService(BetaOpsDependencies(
        memory=FakeMemory(),
        now_factory=lambda: datetime(2026, 6, 7, 10, 0, 0),
    ))


def test_feedback_lifecycle_and_dashboard():
    service = build_service()
    service.init_tables()

    result = service.record_feedback(
        {
            "content": "回答质量不错",
            "category": "quality",
            "rating": 5,
            "session_id": "s1",
            "context": {"intent": "knowledge_qa"},
        },
        actor=BetaActor(user_id="u1", username="alice", department="示例部门甲"),
        request_meta=BetaRequestMeta(ip_address="127.0.0.1", user_agent="pytest"),
    )
    update = service.update_feedback_status(result["feedback_id"], {
        "status": "closed",
        "resolution_note": "已处理",
    }, handled_by="admin")
    dashboard = service.feedback_dashboard(limit=20)

    assert result["success"] is True
    assert update == {"success": True}
    assert dashboard["summary"]["total_feedback"] == 1
    assert dashboard["summary"]["feedback_status"] == {"closed": 1}
    assert dashboard["recent_feedback"][0]["context"] == {"intent": "knowledge_qa"}
    assert dashboard["users"][0]["feedback_count"] == 1


def test_feedback_validation():
    service = build_service()
    service.init_tables()

    empty = service.record_feedback({}, actor=BetaActor(), request_meta=BetaRequestMeta())
    too_long = service.record_feedback({"content": "x" * 2001}, actor=BetaActor(), request_meta=BetaRequestMeta())
    bad_status = service.update_feedback_status(1, {"status": "done"})

    assert empty == {"success": False, "message": "请填写反馈内容"}
    assert too_long == {"success": False, "message": "反馈内容过长，请控制在 2000 字以内"}
    assert bad_status == {"success": False, "message": "反馈状态无效"}


def test_token_usage_recording_and_dashboard():
    service = build_service()
    service.init_tables()

    service.record_token_usage(
        user_id="u1",
        user_info=SimpleNamespace(username="alice", department="示例部门甲"),
        session_id="s1",
        mode="chat",
        agent="Chat",
        model="deepseek",
        stream=True,
        prompt_chars=18,
        completion_chars=36,
        duration_ms=1200,
    )
    service.record_agent_run_token_usage([{
        "step": "write",
        "duration_ms": 500,
        "llm_usage": {
            "agent": "Writer",
            "model": "deepseek",
            "prompt_chars": 10,
            "completion_chars": 20,
        },
    }], user_id="u1", session_id="s1")
    dashboard = service.token_usage_dashboard()

    assert dashboard["summary"]["call_count"] == 2
    assert dashboard["summary"]["total_tokens"] > 0
    assert {row["agent"] for row in dashboard["by_agent"]} == {"Chat", "Writer"}


def test_token_usage_effect_key_deduplicates_direct_and_agent_records():
    service = build_service()
    service.init_tables()

    direct = {
        "user_id": "u1",
        "session_id": "s1",
        "mode": "chat",
        "agent": "Chat",
        "model": "deepseek",
        "prompt_chars": 10,
        "effect_key": "run-1:tool_knowledge_qa:token_usage_chat",
    }
    service.record_token_usage(**direct)
    service.record_token_usage(**direct)

    run_records = [{
        "step": "write",
        "llm_usage": {
            "agent": "Writer",
            "model": "deepseek",
            "prompt_chars": 10,
            "completion_chars": 20,
        },
    }]
    service.record_agent_run_token_usage(
        run_records,
        user_id="u1",
        session_id="s1",
        run_id="run-1",
    )
    service.record_agent_run_token_usage(
        run_records,
        user_id="u1",
        session_id="s1",
        run_id="run-1",
    )
    service.record_token_usage(user_id="u1", agent="unkeyed")
    service.record_token_usage(user_id="u1", agent="unkeyed")

    rows = service.deps.memory.conn.execute(
        "SELECT effect_key, agent FROM beta_token_usage ORDER BY id"
    ).fetchall()
    assert [(row["effect_key"], row["agent"]) for row in rows] == [
        ("run-1:tool_knowledge_qa:token_usage_chat", "Chat"),
        ("run-1:tool_draft_document:token_usage_0_Writer", "Writer"),
        (None, "unkeyed"),
        (None, "unkeyed"),
    ]


def test_agent_token_usage_effect_keys_are_isolated_by_parent_step_scope():
    service = build_service()
    service.init_tables()
    run_records = [{
        "step": "write",
        "llm_usage": {
            "agent": "Writer",
            "model": "deepseek",
            "prompt_chars": 10,
        },
    }]

    for effect_scope in ("step:1", "step:2", "step:2"):
        service.record_agent_run_token_usage(
            run_records,
            user_id="u1",
            session_id="s1",
            run_id="run-parent",
            effect_scope=effect_scope,
        )

    rows = service.deps.memory.conn.execute(
        "SELECT effect_key FROM beta_token_usage ORDER BY id"
    ).fetchall()
    assert [row["effect_key"] for row in rows] == [
        "run-parent:tool_draft_document:step:1:token_usage_0_Writer",
        "run-parent:tool_draft_document:step:2:token_usage_0_Writer",
    ]


def test_token_usage_table_receives_forward_effect_key_migration():
    service = build_service()
    service.init_tables()
    connection = service.deps.memory.conn
    connection.execute("DROP INDEX idx_beta_token_usage_effect_key_unique")
    connection.execute("ALTER TABLE beta_token_usage DROP COLUMN effect_key")
    connection.commit()

    service.init_tables()

    columns = {
        row["name"] for row in connection.execute(
            "PRAGMA table_info(beta_token_usage)"
        ).fetchall()
    }
    indexes = {
        row["name"] for row in connection.execute(
            "PRAGMA index_list(beta_token_usage)"
        ).fetchall()
    }
    assert "effect_key" in columns
    assert "idx_beta_token_usage_effect_key_unique" in indexes
