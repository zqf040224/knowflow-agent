"""Internal beta feedback and token-usage service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional


@dataclass
class BetaOpsDependencies:
    memory: Any
    now_factory: Callable[[], datetime]


@dataclass
class BetaActor:
    user_id: str = ""
    username: str = ""
    department: str = ""


@dataclass
class BetaRequestMeta:
    ip_address: str = ""
    user_agent: str = ""


class BetaOpsService:
    ALLOWED_FEEDBACK_CATEGORIES = {"general", "bug", "quality", "export", "upload", "knowledge", "performance"}
    ALLOWED_FEEDBACK_STATUS = {"open", "in_progress", "closed", "ignored"}

    def __init__(self, deps: BetaOpsDependencies):
        self.deps = deps

    def init_tables(self) -> None:
        with self.deps.memory._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS beta_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    username TEXT,
                    department TEXT DEFAULT '',
                    session_id TEXT DEFAULT '',
                    category TEXT DEFAULT 'general',
                    rating INTEGER DEFAULT 0,
                    content TEXT NOT NULL,
                    page_url TEXT DEFAULT '',
                    context_json TEXT DEFAULT '{}',
                    ip_address TEXT DEFAULT '',
                    user_agent TEXT DEFAULT '',
                    status TEXT DEFAULT 'open',
                    resolution_note TEXT DEFAULT '',
                    handled_by TEXT DEFAULT '',
                    handled_at TEXT DEFAULT '',
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            columns = {
                row['name']
                for row in cursor.execute('PRAGMA table_info(beta_feedback)').fetchall()
            }
            for column, ddl in {
                'resolution_note': "ALTER TABLE beta_feedback ADD COLUMN resolution_note TEXT DEFAULT ''",
                'handled_by': "ALTER TABLE beta_feedback ADD COLUMN handled_by TEXT DEFAULT ''",
                'handled_at': "ALTER TABLE beta_feedback ADD COLUMN handled_at TEXT DEFAULT ''",
            }.items():
                if column not in columns:
                    cursor.execute(ddl)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_beta_feedback_user ON beta_feedback(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_beta_feedback_created ON beta_feedback(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_beta_feedback_status ON beta_feedback(status)')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS beta_token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT DEFAULT '',
                    username TEXT DEFAULT '',
                    department TEXT DEFAULT '',
                    session_id TEXT DEFAULT '',
                    mode TEXT DEFAULT '',
                    agent TEXT DEFAULT '',
                    model TEXT DEFAULT '',
                    stream INTEGER DEFAULT 0,
                    prompt_chars INTEGER DEFAULT 0,
                    completion_chars INTEGER DEFAULT 0,
                    reasoning_chars INTEGER DEFAULT 0,
                    estimated_prompt_tokens INTEGER DEFAULT 0,
                    estimated_completion_tokens INTEGER DEFAULT 0,
                    estimated_total_tokens INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    max_tokens INTEGER DEFAULT 0,
                    temperature REAL DEFAULT 0,
                    status TEXT DEFAULT 'success',
                    error_message TEXT DEFAULT '',
                    created_at TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_beta_token_usage_created ON beta_token_usage(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_beta_token_usage_user ON beta_token_usage(user_id, created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_beta_token_usage_mode ON beta_token_usage(mode, created_at)')
            conn.commit()

    def record_feedback(self, data: dict, *, actor: BetaActor, request_meta: BetaRequestMeta) -> dict:
        content = (data.get('content') or '').strip()
        if not content:
            return {'success': False, 'message': '请填写反馈内容'}
        if len(content) > 2000:
            return {'success': False, 'message': '反馈内容过长，请控制在 2000 字以内'}

        category = (data.get('category') or 'general').strip()[:32]
        if category not in self.ALLOWED_FEEDBACK_CATEGORIES:
            category = 'general'

        try:
            rating = int(data.get('rating') or 0)
        except (TypeError, ValueError):
            rating = 0
        rating = max(0, min(rating, 5))

        context = data.get('context') or {}
        if not isinstance(context, dict):
            context = {}

        now = self.deps.now_factory().isoformat()
        with self.deps.memory._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO beta_feedback (
                    user_id, username, department, session_id, category, rating,
                    content, page_url, context_json, ip_address, user_agent,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
            ''', (
                actor.user_id,
                actor.username,
                actor.department,
                (data.get('session_id') or '').strip()[:160],
                category,
                rating,
                content,
                (data.get('page_url') or '').strip()[:500],
                json.dumps(context, ensure_ascii=False),
                request_meta.ip_address,
                request_meta.user_agent[:500],
                now,
                now,
            ))
            feedback_id = cursor.lastrowid
            conn.commit()

        return {'success': True, 'feedback_id': feedback_id}

    def feedback_dashboard(self, limit: int = 80) -> dict:
        try:
            limit = int(limit or 80)
        except (TypeError, ValueError):
            limit = 80
        limit = min(max(limit, 10), 300)
        today = self.deps.now_factory().strftime('%Y-%m-%d')

        with self.deps.memory._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, name, department, role, is_active, created_at, last_login
                FROM auth_users
                ORDER BY created_at DESC
            ''')
            users = [dict(row) for row in cursor.fetchall()]

            cursor.execute('''
                SELECT user_id, COUNT(*) AS session_count, MAX(updated_at) AS last_session_at
                FROM sessions
                GROUP BY user_id
            ''')
            sessions_by_user = {row['user_id']: dict(row) for row in cursor.fetchall()}

            cursor.execute('''
                SELECT s.user_id,
                       COUNT(m.id) AS message_count,
                       SUM(CASE WHEN datetime(m.timestamp, 'unixepoch') LIKE ? THEN 1 ELSE 0 END) AS today_messages,
                       MAX(m.timestamp) AS last_message_ts
                FROM messages m
                JOIN sessions s ON s.session_id = m.session_id
                GROUP BY s.user_id
            ''', (f'{today}%',))
            messages_by_user = {row['user_id']: dict(row) for row in cursor.fetchall()}

            cursor.execute('''
                SELECT user_id, COUNT(*) AS feedback_count, MAX(created_at) AS last_feedback_at
                FROM beta_feedback
                GROUP BY user_id
            ''')
            feedback_by_user = {row['user_id']: dict(row) for row in cursor.fetchall()}

            cursor.execute('SELECT COUNT(*) FROM beta_feedback')
            total_feedback = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM beta_feedback WHERE created_at LIKE ?', (f'{today}%',))
            today_feedback = cursor.fetchone()[0]
            cursor.execute('SELECT AVG(NULLIF(rating, 0)) FROM beta_feedback')
            avg_rating = cursor.fetchone()[0] or 0
            cursor.execute('SELECT status, COUNT(*) AS count FROM beta_feedback GROUP BY status')
            feedback_status_counts = {row['status']: int(row['count'] or 0) for row in cursor.fetchall()}

            cursor.execute('''
                SELECT id, user_id, username, department, session_id, category, rating,
                       content, page_url, context_json, status, resolution_note,
                       handled_by, handled_at, created_at
                FROM beta_feedback
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            recent_feedback = []
            for row in cursor.fetchall():
                item = dict(row)
                try:
                    item['context'] = json.loads(item.pop('context_json') or '{}')
                except Exception:
                    item['context'] = {}
                recent_feedback.append(item)

            cursor.execute('''
                SELECT s.session_id, s.user_id, u.username, u.name, u.department,
                       s.title, s.doc_type, s.message_count, s.created_at, s.updated_at
                FROM sessions s
                LEFT JOIN auth_users u ON u.user_id = s.user_id
                ORDER BY s.updated_at DESC
                LIMIT 12
            ''')
            recent_sessions = [dict(row) for row in cursor.fetchall()]

        per_user = self._feedback_users(users, sessions_by_user, messages_by_user, feedback_by_user)
        return {
            'success': True,
            'summary': {
                'total_users': len(users),
                'active_users': sum(1 for item in per_user if item.get('_active')),
                'total_messages': sum(int(item.get('message_count') or 0) for item in per_user),
                'today_messages': sum(int(item.get('today_messages') or 0) for item in per_user),
                'total_feedback': total_feedback,
                'today_feedback': today_feedback,
                'avg_rating': round(float(avg_rating), 1),
                'feedback_status': feedback_status_counts,
            },
            'users': [{k: v for k, v in item.items() if k != '_active'} for item in per_user],
            'recent_feedback': recent_feedback,
            'recent_sessions': recent_sessions,
        }

    def update_feedback_status(self, feedback_id: int, data: dict, *, handled_by: str = "") -> dict:
        status = (data.get('status') or '').strip()
        if status not in self.ALLOWED_FEEDBACK_STATUS:
            return {'success': False, 'message': '反馈状态无效'}
        note = (data.get('resolution_note') or '').strip()[:500]
        now = self.deps.now_factory().isoformat()
        with self.deps.memory._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM beta_feedback WHERE id = ?', (feedback_id,))
            if not cursor.fetchone():
                return {'success': False, 'message': '未找到该反馈'}
            cursor.execute('''
                UPDATE beta_feedback
                SET status = ?, resolution_note = ?, handled_by = ?,
                    handled_at = ?, updated_at = ?
                WHERE id = ?
            ''', (status, note, handled_by, now, now, feedback_id))
            conn.commit()
        return {'success': True}

    def record_token_usage(
        self,
        *,
        user_id: str,
        user_info: Optional[Any] = None,
        session_id: str = "",
        mode: str = "",
        agent: str = "",
        model: str = "",
        stream: bool = False,
        prompt_chars: int = 0,
        completion_chars: int = 0,
        reasoning_chars: int = 0,
        estimated_prompt_tokens: Optional[int] = None,
        estimated_completion_tokens: Optional[int] = None,
        estimated_total_tokens: Optional[int] = None,
        duration_ms: int = 0,
        max_tokens: int = 0,
        temperature: float = 0,
        status: str = "success",
        error_message: str = "",
    ) -> None:
        if estimated_prompt_tokens is None:
            estimated_prompt_tokens = self._estimate_tokens_from_chars(prompt_chars)
        if estimated_completion_tokens is None:
            estimated_completion_tokens = self._estimate_tokens_from_chars(completion_chars)
        if estimated_total_tokens is None:
            estimated_total_tokens = int(estimated_prompt_tokens or 0) + int(estimated_completion_tokens or 0)

        now = self.deps.now_factory().isoformat()
        username = getattr(user_info, "username", "") if user_info else ""
        department = getattr(user_info, "department", "") if user_info else ""
        with self.deps.memory._get_conn() as conn:
            conn.execute('''
                INSERT INTO beta_token_usage (
                    user_id, username, department, session_id, mode, agent, model,
                    stream, prompt_chars, completion_chars, reasoning_chars,
                    estimated_prompt_tokens, estimated_completion_tokens,
                    estimated_total_tokens, duration_ms, max_tokens, temperature,
                    status, error_message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id or "",
                username,
                department,
                session_id or "",
                mode or "",
                agent or "",
                model or "",
                1 if stream else 0,
                int(prompt_chars or 0),
                int(completion_chars or 0),
                int(reasoning_chars or 0),
                int(estimated_prompt_tokens or 0),
                int(estimated_completion_tokens or 0),
                int(estimated_total_tokens or 0),
                int(duration_ms or 0),
                int(max_tokens or 0),
                float(temperature or 0),
                status or "success",
                str(error_message or "")[:500],
                now,
            ))
            conn.commit()

    def record_agent_run_token_usage(self, run_records: list, *, user_id: str, user_info: Any = None, session_id: str = "", mode: str = "agent") -> None:
        for record in run_records or []:
            usage = record.get("llm_usage") or {}
            if not usage:
                continue
            self.record_token_usage(
                user_id=user_id,
                user_info=user_info,
                session_id=session_id,
                mode=mode,
                agent=usage.get("agent") or record.get("step", ""),
                model=usage.get("model", ""),
                stream=bool(usage.get("stream")),
                prompt_chars=usage.get("prompt_chars", 0),
                completion_chars=usage.get("completion_chars", 0),
                reasoning_chars=usage.get("reasoning_chars", 0),
                estimated_prompt_tokens=usage.get("estimated_prompt_tokens"),
                estimated_completion_tokens=usage.get("estimated_completion_tokens"),
                estimated_total_tokens=usage.get("estimated_total_tokens"),
                duration_ms=usage.get("duration_ms", record.get("duration_ms", 0)),
                max_tokens=usage.get("max_tokens", 0),
                temperature=usage.get("temperature", 0),
                status="success",
            )

    def token_usage_dashboard(self, limit: int = 120) -> dict:
        try:
            limit = int(limit or 120)
        except (TypeError, ValueError):
            limit = 120
        limit = min(max(limit, 20), 500)
        today = self.deps.now_factory().strftime('%Y-%m-%d')

        with self.deps.memory._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) AS call_count,
                       COALESCE(SUM(estimated_total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(estimated_prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(estimated_completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(reasoning_chars), 0) AS reasoning_chars,
                       COALESCE(AVG(duration_ms), 0) AS avg_duration_ms
                FROM beta_token_usage
            ''')
            total = dict(cursor.fetchone())

            cursor.execute('''
                SELECT COUNT(*) AS call_count,
                       COALESCE(SUM(estimated_total_tokens), 0) AS total_tokens
                FROM beta_token_usage
                WHERE created_at LIKE ?
            ''', (f'{today}%',))
            today_row = dict(cursor.fetchone())

            cursor.execute('''
                SELECT user_id, username, department,
                       COUNT(*) AS call_count,
                       COALESCE(SUM(estimated_total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(estimated_prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(estimated_completion_tokens), 0) AS completion_tokens,
                       MAX(created_at) AS last_used_at
                FROM beta_token_usage
                GROUP BY user_id, username, department
                ORDER BY total_tokens DESC
                LIMIT 50
            ''')
            by_user = [dict(row) for row in cursor.fetchall()]

            cursor.execute('''
                SELECT mode, agent, model,
                       COUNT(*) AS call_count,
                       COALESCE(SUM(estimated_total_tokens), 0) AS total_tokens,
                       COALESCE(AVG(duration_ms), 0) AS avg_duration_ms
                FROM beta_token_usage
                GROUP BY mode, agent, model
                ORDER BY total_tokens DESC
                LIMIT 80
            ''')
            by_agent = [dict(row) for row in cursor.fetchall()]

            cursor.execute('''
                SELECT *
                FROM beta_token_usage
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            recent = [dict(row) for row in cursor.fetchall()]

        return {
            'success': True,
            'summary': {
                'call_count': int(total.get('call_count') or 0),
                'total_tokens': int(total.get('total_tokens') or 0),
                'prompt_tokens': int(total.get('prompt_tokens') or 0),
                'completion_tokens': int(total.get('completion_tokens') or 0),
                'reasoning_chars': int(total.get('reasoning_chars') or 0),
                'avg_duration_ms': round(float(total.get('avg_duration_ms') or 0)),
                'today_call_count': int(today_row.get('call_count') or 0),
                'today_tokens': int(today_row.get('total_tokens') or 0),
            },
            'by_user': by_user,
            'by_agent': by_agent,
            'recent': recent,
        }

    def _feedback_users(self, users, sessions_by_user, messages_by_user, feedback_by_user):
        per_user = []
        for user in users:
            user_id = user.get('user_id')
            session_stats = sessions_by_user.get(user_id, {})
            message_stats = messages_by_user.get(user_id, {})
            feedback_stats = feedback_by_user.get(user_id, {})
            message_count = int(message_stats.get('message_count') or 0)
            today_messages = int(message_stats.get('today_messages') or 0)
            last_message_ts = message_stats.get('last_message_ts')
            last_message_at = datetime.fromtimestamp(last_message_ts).isoformat() if last_message_ts else ''
            last_active = max(
                [value for value in [
                    user.get('last_login') or '',
                    session_stats.get('last_session_at') or '',
                    feedback_stats.get('last_feedback_at') or '',
                    last_message_at,
                ] if value],
                default='',
            )
            is_active = bool(user.get('last_login') or session_stats.get('session_count') or message_count or feedback_stats.get('feedback_count'))
            per_user.append({
                **user,
                'session_count': int(session_stats.get('session_count') or 0),
                'message_count': message_count,
                'today_messages': today_messages,
                'feedback_count': int(feedback_stats.get('feedback_count') or 0),
                'last_active': last_active,
                '_active': is_active,
            })
        per_user.sort(key=lambda item: item.get('last_active') or item.get('created_at') or '', reverse=True)
        return per_user

    @staticmethod
    def _estimate_tokens_from_chars(char_count: int) -> int:
        return max(1, int((char_count or 0) / 1.8)) if char_count else 0
