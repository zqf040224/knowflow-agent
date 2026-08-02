"""Shared application constants for Flask wiring."""

import os
from collections import defaultdict


DEFAULT_DEPARTMENT_DIRS = (
    "综合管理部",
    "人力资源部",
    "财务部",
    "运营保障部",
    "市场部",
    "业务部",
    "客户服务部",
    "项目部",
)


def _load_department_dirs():
    configured = {
        item.strip()
        for item in os.getenv("DEPARTMENT_DIRS", "").split(",")
        if item.strip()
    }
    return frozenset(configured or DEFAULT_DEPARTMENT_DIRS)


DEPARTMENT_DIRS = _load_department_dirs()
MAX_REQUEST_SIZE = 10 * 1024 * 1024
RATE_LIMITS = {
    "chat": 20,
    "agent_generate": 10,
    "api_login": 5,
}


def new_rate_limit_store():
    return defaultdict(list)
