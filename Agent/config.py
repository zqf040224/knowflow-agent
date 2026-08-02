import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 数据库配置
DATABASE_CONFIG = {
    'postgresql': {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'user': os.getenv('DB_USER', 'admin'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'database': os.getenv('DB_NAME', 'agent_db'),
    }
}

# Redis配置
CACHE_BACKEND = os.getenv('CACHE_BACKEND', 'memory').lower()
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': os.getenv('REDIS_PORT', '6379'),
    'db': os.getenv('REDIS_DB', '0'),
    'password': os.getenv('REDIS_PASSWORD', ''),
}

# 系统配置
SYSTEM_CONFIG = {
    'debug': os.getenv('DEBUG', 'False').lower() == 'true',
    'port': os.getenv('PORT', '5003'),
    'host': os.getenv('HOST', '0.0.0.0'),
}
