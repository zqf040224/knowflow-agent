"""
知识库模块

提供统一的知识库操作接口：
- KnowledgeBase: 核心类
- load_knowledge_base: 便捷函数
- quick_search: 快速检索
"""

from .core import KnowledgeBase, SearchResult, load_knowledge_base, quick_search

__all__ = ['KnowledgeBase', 'SearchResult', 'load_knowledge_base', 'quick_search']
