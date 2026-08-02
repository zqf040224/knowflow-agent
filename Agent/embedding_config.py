"""
Embedding 模型 + 权限框架 统一配置

所有需要 Embedding 模型或权限控制的模块从此处 import，避免分散硬编码。
"""
import os
import sys
from typing import Dict, Optional, List
from dataclasses import dataclass, field

# ============================================================
# Embedding 模型配置
# ============================================================

MODEL_NAME = "maidalun1020/bce-embedding-base_v1"
DIM = 768
LOCAL_MODEL_CACHE = os.path.expanduser(
    "~/.cache/torch/sentence_transformers/maidalun1020_bce-embedding-base_v1"
)

# BCE 模型需要 query/document 区分前缀以达到最佳效果
# 参考: https://huggingface.co/maidalun1020/bce-embedding-base_v1
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

HF_ENDPOINT = "https://hf-mirror.com"

if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def _running_under_gunicorn() -> bool:
    executable = os.path.basename(sys.argv[0] or "")
    return executable == "gunicorn" or "gunicorn" in executable


def resolve_embedding_device(torch_module=None) -> str:
    """Resolve the embedding device with a production-safe override."""
    requested = (os.getenv("EMBEDDING_DEVICE") or "auto").strip().lower()
    if requested in {"cpu", "mps", "cuda"}:
        return requested
    if requested and requested != "auto":
        return "cpu"

    if _running_under_gunicorn():
        return "cpu"

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return "cpu"

    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_embedding_model_path() -> str:
    """Prefer the local sentence-transformers cache to avoid startup downloads."""
    if os.path.isdir(LOCAL_MODEL_CACHE):
        return LOCAL_MODEL_CACHE
    return MODEL_NAME

# ============================================================
# 知识库权限框架
# ============================================================

# 文档访问级别（由低到高）
ACCESS_PUBLIC = "public"        # 所有人可见，包括未登录用户
ACCESS_INTERNAL = "internal"    # 登录用户可见
ACCESS_RESTRICTED = "restricted"  # 特定角色/部门可见
ACCESS_ADMIN = "admin"          # 仅管理员可见

ACCESS_LEVELS = [ACCESS_PUBLIC, ACCESS_INTERNAL, ACCESS_RESTRICTED, ACCESS_ADMIN]

# 角色能看到的最高访问级别
ROLE_ACCESS_MAP = {
    "admin": ACCESS_ADMIN,
    "user": ACCESS_INTERNAL,
    "anonymous": ACCESS_PUBLIC,
}


@dataclass
class UserInfo:
    """用户信息，用于权限过滤（从 Flask g.user 提取）"""
    user_id: str = ""
    username: str = ""
    role: str = "anonymous"
    department: str = ""
    clearance: str = ACCESS_PUBLIC  # 用户可访问的最高文档级别

    def __post_init__(self):
        # 如果未指定 clearance，从角色推导
        if self.clearance == ACCESS_PUBLIC and self.role != "anonymous":
            self.clearance = ROLE_ACCESS_MAP.get(self.role, ACCESS_PUBLIC)

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "department": self.department,
            "clearance": self.clearance,
        }


def build_access_filter(user_info: Optional[UserInfo] = None,
                        extra_tags: Optional[List[str]] = None) -> Dict:
    """
    根据用户信息构建知识库检索的权限过滤条件。

    权限规则：
    - admin：可查看所有文档（无过滤）
    - 部门用户：可查看 public 文档（公共资料）+ 本部门 restricted 文档
    - 普通用户（无部门）：仅可查看 public 文档
    - 匿名用户：仅可查看 public 文档

    Returns:
        Dict: 传入 KnowledgeAgent._match_access 的过滤条件
    """
    if user_info is None:
        return {"access_level": [ACCESS_PUBLIC]}

    # 兼容 dict 和 UserInfo 对象
    if isinstance(user_info, dict):
        role = user_info.get("role", "anonymous")
        department = user_info.get("department", "")
    else:
        role = user_info.role
        department = user_info.department

    # admin 可以看所有
    if role == "admin":
        return {}

    # 构建可访问级别列表（从最低到最高，包含所有低于用户级别的）
    max_clearance = ROLE_ACCESS_MAP.get(role, ACCESS_PUBLIC)
    # 有部门的用户可以访问 restricted 级别的本部门文档
    if department and max_clearance == ACCESS_INTERNAL:
        max_clearance = ACCESS_RESTRICTED
    allowed_levels = []
    for level in ACCESS_LEVELS:
        allowed_levels.append(level)
        if level == max_clearance:
            break

    filters = {"access_level": allowed_levels}

    # 部门过滤：用户只能看本部门文档 + 无部门归属的公开文档
    if department:
        filters["department"] = [department, ""]

    return filters
