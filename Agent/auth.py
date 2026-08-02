"""
用户认证模块

特性：
1. 用户注册/登录/登出
2. 密码加密存储（bcrypt）
3. JWT Token 认证
4. 角色管理（admin/user）
5. 管理员账号初始化
"""

import os
import sys
import sqlite3
import hashlib
import secrets
import bcrypt
import jwt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
from functools import wraps
from flask import request, jsonify, g, redirect
from database import DatabaseManager

# JWT 配置
JWT_SECRET = os.getenv('JWT_SECRET')
if not JWT_SECRET:
    print("ERROR: JWT_SECRET 环境变量未设置，服务无法启动")
    sys.exit(1)
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION = 24  # 小时

# 管理员默认配置
DEFAULT_ADMIN = {
    'username': 'admin',
    'name': '系统管理员',
    'department': '系统管理部',
    'role': 'admin'
}


def _should_return_json_auth_error() -> bool:
    """API clients should receive status-coded JSON instead of login HTML."""
    accept = request.headers.get('Accept', '')
    return request.path.startswith('/api/') or request.is_json or 'application/json' in accept


class AuthManager:
    """认证管理器"""

    def __init__(self, db_path: str = "./data/agent_memory.db", db_type: str = "sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_type = db_type
        self.db_manager = DatabaseManager(db_type=db_type, db_path=str(db_path))
        self._init_db()
        self._create_default_admin()

    def _init_db(self):
        """初始化用户表"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # 用户认证表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    department TEXT DEFAULT '',
                    role TEXT DEFAULT 'user',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT,
                    last_login TEXT,
                    password_changed INTEGER DEFAULT 0  -- 是否修改过默认密码
                )
            ''')

            # 登录日志表
            if self.db_type == 'postgresql':
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS login_logs (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT,
                        username TEXT,
                        action TEXT,  -- login, logout, login_failed
                        ip_address TEXT,
                        user_agent TEXT,
                        timestamp TEXT,
                        FOREIGN KEY (user_id) REFERENCES auth_users(user_id)
                    )
                ''')
            else:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS login_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        username TEXT,
                        action TEXT,  -- login, logout, login_failed
                        ip_address TEXT,
                        user_agent TEXT,
                        timestamp TEXT,
                        FOREIGN KEY (user_id) REFERENCES auth_users(user_id)
                    )
                ''')

            conn.commit()

    def _create_default_admin(self):
        """创建默认管理员账号"""
        if not self.get_user_by_username(DEFAULT_ADMIN['username']):
            admin_password = os.getenv('ADMIN_DEFAULT_PASSWORD')
            if not admin_password:
                admin_password = secrets.token_urlsafe(16)
            self.register_user(
                username=DEFAULT_ADMIN['username'],
                password=admin_password,
                name=DEFAULT_ADMIN['name'],
                department=DEFAULT_ADMIN['department'],
                role=DEFAULT_ADMIN['role']
            )
            # 仅写入日志，不打印到控制台
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"默认管理员已创建，用户名: {DEFAULT_ADMIN['username']}")
            if not os.getenv('ADMIN_DEFAULT_PASSWORD'):
                logger.info(f"随机生成的管理员密码: {admin_password}")

    def _hash_password(self, password: str) -> str:
        """密码哈希（bcrypt）"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码，兼容旧的 SHA256 格式并自动升级"""
        # bcrypt 格式（$2b$ 前缀）
        if password_hash.startswith('$2b$') or password_hash.startswith('$2a$'):
            try:
                return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
            except Exception:
                return False

        # 旧的 SHA256 格式（salt$hash）— 向后兼容
        try:
            salt, stored_hash = password_hash.split('$')
            pwdhash = hashlib.sha256((password + salt).encode()).hexdigest()
            return pwdhash == stored_hash
        except Exception:
            return False

    def _upgrade_password_hash(self, user_id: str, password: str):
        """将旧格式密码升级为 bcrypt"""
        new_hash = self._hash_password(password)
        with self.db_manager.get_connection() as conn:
            conn.cursor().execute(
                'UPDATE auth_users SET password_hash = ? WHERE user_id = ?',
                (new_hash, user_id)
            )
            conn.commit()

    def register_user(self, username: str, password: str, name: str = "",
                     department: str = "", role: str = "user") -> Dict:
        """注册用户"""
        # 检查用户名是否已存在
        if self.get_user_by_username(username):
            return {'success': False, 'message': '用户名已存在'}

        # 生成用户ID
        user_id = f"user_{int(datetime.now().timestamp())}_{secrets.token_hex(4)}"

        # 密码哈希
        password_hash = self._hash_password(password)

        # 保存到数据库
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO auth_users
                (user_id, username, password_hash, name, department, role, created_at, password_changed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, username, password_hash, name, department, role, now,
                  1 if role == 'admin' else 0))  # 管理员标记为已修改密码

            conn.commit()

        return {
            'success': True,
            'user_id': user_id,
            'username': username,
            'role': role
        }

    def login(self, username: str, password: str, ip_address: str = "",
              user_agent: str = "") -> Dict:
        """用户登录"""
        user = self.get_user_by_username(username)

        if not user:
            self._log_login(None, username, 'login_failed', ip_address, user_agent)
            return {'success': False, 'message': '用户名或密码错误'}

        if not user.get('is_active'):
            return {'success': False, 'message': '账号已被禁用'}

        if not self._verify_password(password, user['password_hash']):
            self._log_login(user['user_id'], username, 'login_failed', ip_address, user_agent)
            return {'success': False, 'message': '用户名或密码错误'}

        # 自动升级旧格式密码哈希为 bcrypt
        if not (user['password_hash'].startswith('$2b$') or user['password_hash'].startswith('$2a$')):
            self._upgrade_password_hash(user['user_id'], password)

        # 生成 JWT Token
        token = self._generate_token(user['user_id'], user['username'], user['role'], user.get('department', ''))

        # 更新最后登录时间
        self._update_last_login(user['user_id'])

        # 记录登录日志
        self._log_login(user['user_id'], username, 'login', ip_address, user_agent)

        return {
            'success': True,
            'token': token,
            'user': {
                'user_id': user['user_id'],
                'username': user['username'],
                'name': user['name'],
                'department': user['department'],
                'role': user['role'],
                'password_changed': user.get('password_changed', 0)
            }
        }

    def _generate_token(self, user_id: str, username: str, role: str, department: str = "") -> str:
        """生成 JWT Token（含部门信息，避免每次请求查DB）"""
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'department': department,
            'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    def verify_token(self, token: str) -> Optional[Dict]:
        """验证 JWT Token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """通过用户名获取用户"""
        with self.db_manager.get_connection() as conn:
            if self.db_type == 'sqlite':
                conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM auth_users WHERE username = ?
            ''', (username,))

            row = cursor.fetchone()
            
            if row:
                if self.db_type == 'sqlite':
                    return dict(row)
                else:
                    return dict(zip([desc[0] for desc in cursor.description], row))
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """通过ID获取用户"""
        with self.db_manager.get_connection() as conn:
            if self.db_type == 'sqlite':
                conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM auth_users WHERE user_id = ?
            ''', (user_id,))

            row = cursor.fetchone()
            
            if row:
                if self.db_type == 'sqlite':
                    return dict(row)
                else:
                    return dict(zip([desc[0] for desc in cursor.description], row))
            return None

    def change_password(self, user_id: str, old_password: str, new_password: str) -> Dict:
        """修改密码"""
        user = self.get_user_by_id(user_id)

        if not user:
            return {'success': False, 'message': '用户不存在'}

        if not self._verify_password(old_password, user['password_hash']):
            return {'success': False, 'message': '原密码错误'}

        # 更新密码
        new_hash = self._hash_password(new_password)

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE auth_users
                SET password_hash = ?, password_changed = 1
                WHERE user_id = ?
            ''', (new_hash, user_id))

            conn.commit()

        return {'success': True, 'message': '密码修改成功'}

    def reset_password(self, admin_id: str, target_user_id: str, new_password: str) -> Dict:
        """管理员重置密码"""
        # 验证管理员权限
        admin = self.get_user_by_id(admin_id)
        if not admin or admin['role'] != 'admin':
            return {'success': False, 'message': '权限不足'}

        # 重置密码
        new_hash = self._hash_password(new_password)

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE auth_users
                SET password_hash = ?, password_changed = 0
                WHERE user_id = ?
            ''', (new_hash, target_user_id))

            conn.commit()

        return {'success': True, 'message': '密码重置成功'}

    def update_user(self, user_id: str, updates: Dict) -> Dict:
        """更新用户信息"""
        allowed_fields = ['name', 'department']
        updates = {k: v for k, v in updates.items() if k in allowed_fields}

        if not updates:
            return {'success': False, 'message': '没有可更新的字段'}

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [user_id]

            cursor.execute(f'''
                UPDATE auth_users SET {set_clause} WHERE user_id = ?
            ''', values)

            conn.commit()

        return {'success': True, 'message': '更新成功'}

    def list_users(self, admin_id: str) -> Dict:
        """列出所有用户（仅管理员）"""
        admin = self.get_user_by_id(admin_id)
        if not admin or admin['role'] != 'admin':
            return {'success': False, 'message': '权限不足'}

        with self.db_manager.get_connection() as conn:
            if self.db_type == 'sqlite':
                conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, username, name, department, role, is_active,
                       created_at, last_login, password_changed
                FROM auth_users
                ORDER BY created_at DESC
            ''')

            if self.db_type == 'sqlite':
                users = [dict(row) for row in cursor.fetchall()]
            else:
                users = []
                for row in cursor.fetchall():
                    users.append(dict(zip([desc[0] for desc in cursor.description], row)))

        return {'success': True, 'users': users}

    def toggle_user_status(self, admin_id: str, target_user_id: str, is_active: bool) -> Dict:
        """启用/禁用用户"""
        admin = self.get_user_by_id(admin_id)
        if not admin or admin['role'] != 'admin':
            return {'success': False, 'message': '权限不足'}

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE auth_users SET is_active = ? WHERE user_id = ?
            ''', (1 if is_active else 0, target_user_id))

            conn.commit()

        return {'success': True, 'message': '状态更新成功'}

    def _update_last_login(self, user_id: str):
        """更新最后登录时间"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            now = datetime.now().isoformat()
            cursor.execute('''
                UPDATE auth_users SET last_login = ? WHERE user_id = ?
            ''', (now, user_id))

            conn.commit()

    def _log_login(self, user_id: Optional[str], username: str, action: str,
                   ip_address: str, user_agent: str):
        """记录登录日志"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO login_logs (user_id, username, action, ip_address, user_agent, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, action, ip_address, user_agent, now))

            conn.commit()

    def get_login_logs(self, admin_id: str, limit: int = 100) -> Dict:
        """获取登录日志"""
        admin = self.get_user_by_id(admin_id)
        if not admin or admin['role'] != 'admin':
            return {'success': False, 'message': '权限不足'}

        with self.db_manager.get_connection() as conn:
            if self.db_type == 'sqlite':
                conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM login_logs ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))

            if self.db_type == 'sqlite':
                logs = [dict(row) for row in cursor.fetchall()]
            else:
                logs = []
                for row in cursor.fetchall():
                    logs.append(dict(zip([desc[0] for desc in cursor.description], row)))

        return {'success': True, 'logs': logs}


# ========== Flask 装饰器 ==========

def login_required(f):
    """登录验证装饰器 - 支持 Header 和 Cookie"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None

        # 1. 先从 Header 获取 (API 请求)
        auth_header = request.headers.get('Authorization')
        if auth_header:
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]

        # 2. 从 Cookie 获取 (页面请求)
        if not token:
            token = request.cookies.get('token')

        if not token:
            # 如果是页面请求，重定向到登录页
            if _should_return_json_auth_error():
                return jsonify({'error': '未登录'}), 401
            else:
                return redirect('/login')

        auth_manager = get_auth_manager()
        payload = auth_manager.verify_token(token)

        if not payload:
            # Token 过期
            if _should_return_json_auth_error():
                return jsonify({'error': '登录已过期'}), 401
            else:
                response = redirect('/')
                response.set_cookie('token', '', expires=0)
                return response

        # 设置当前用户
        g.user_id = payload['user_id']
        g.username = payload['username']
        g.role = payload['role']
        g.department = payload.get('department', '')

        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """管理员权限装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'role') or g.role != 'admin':
            if not request.path.startswith('/api/') and not request.is_json:
                return redirect('/chat')
            return jsonify({'error': '权限不足'}), 403
        return f(*args, **kwargs)
    return decorated_function


# ========== 全局实例 ==========
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """获取全局认证实例"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager(db_type="sqlite")
    return _auth_manager
