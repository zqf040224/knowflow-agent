import sqlite3
from contextlib import contextmanager
from config import DATABASE_CONFIG

try:
    import psycopg2
except ImportError:  # SQLite is the default runtime and must not require PostgreSQL extras.
    psycopg2 = None

class DatabaseManager:
    def __init__(self, db_type='sqlite', db_path='./data/agent_memory.db'):
        self.db_type = db_type
        self.db_path = db_path
        self.postgres_config = DATABASE_CONFIG['postgresql']
        
    @contextmanager
    def get_connection(self):
        if self.db_type == 'postgresql':
            if psycopg2 is None:
                raise RuntimeError('PostgreSQL 模式需要安装 psycopg2-binary')
            conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                database=self.postgres_config['database']
            )
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5)
            conn.row_factory = sqlite3.Row
            # These pragmas are per connection.  Foreign keys keep the SQLite
            # database honest, while a bounded busy timeout avoids immediate
            # "database is locked" failures under gthread workers.
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        if self.db_type == 'postgresql':
            self._init_postgresql()
        else:
            self._init_sqlite()
    
    def _init_postgresql(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    department TEXT DEFAULT '',
                    preferred_font TEXT DEFAULT '仿宋_GB2312',
                    preferred_size TEXT DEFAULT '三号',
                    common_doc_types TEXT DEFAULT '[]',
                    writing_style TEXT DEFAULT '简洁正式',
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # 创建会话表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT DEFAULT '',
                    doc_type TEXT DEFAULT '',
                    created_at TEXT,
                    updated_at TEXT,
                    message_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            
            # 创建消息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp REAL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            ''')
            
            # 创建会话上下文表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_context (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    context_key TEXT,
                    context_value TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    UNIQUE(session_id, context_key)
                )
            ''')

            # Long-term memories are intentionally separate from raw messages:
            # only explicit user memories and curated profile facts live here.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_items (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    source_session_id TEXT,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    FOREIGN KEY (source_session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON messages(session_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user_active_updated ON sessions(user_id, is_active, updated_at DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_context_session ON session_context(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_user_active_updated ON memory_items(user_id, is_active, updated_at DESC)')
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_active_key
                ON memory_items(user_id, memory_key)
                WHERE is_active = 1
            ''')
            
            conn.commit()
    
    def _init_sqlite(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    department TEXT DEFAULT '',
                    preferred_font TEXT DEFAULT '仿宋_GB2312',
                    preferred_size TEXT DEFAULT '三号',
                    common_doc_types TEXT DEFAULT '[]',
                    writing_style TEXT DEFAULT '简洁正式',
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # 创建会话表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT DEFAULT '',
                    doc_type TEXT DEFAULT '',
                    created_at TEXT,
                    updated_at TEXT,
                    message_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            
            # 创建消息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp REAL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            ''')
            
            # 创建会话上下文表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    context_key TEXT,
                    context_value TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    UNIQUE(session_id, context_key)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    source_session_id TEXT,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    FOREIGN KEY (source_session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON messages(session_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user_active_updated ON sessions(user_id, is_active, updated_at DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_context_session ON session_context(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_user_active_updated ON memory_items(user_id, is_active, updated_at DESC)')
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_active_key
                ON memory_items(user_id, memory_key)
                WHERE is_active = 1
            ''')
            
            conn.commit()
