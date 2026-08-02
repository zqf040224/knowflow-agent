"""
知识库核心模块 - 统一封装知识库操作

特性：
1. 统一接口：加载、检索、添加文档
2. 本地模型：使用 sentence-transformers，无需 API
3. 元数据过滤：支持按来源、分类等过滤
4. 设备自适应：自动选择 MPS/CPU
5. 与 builder.py v3.0 和 knowledge_qa_fast.py 完全兼容

使用方法：
    from knowledge_base import KnowledgeBase

    # 加载现有知识库
    kb = KnowledgeBase("./knowledge_base")

    # 检索
    results = kb.search("会议通知格式", top_k=5)

    # 添加新文档
    kb.add_documents([{"content": "...", "metadata": {...}}])
"""

import os
import json
import pickle
import time
import warnings
import threading
import fcntl
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import faiss

from embedding_config import (
    MODEL_NAME as DEFAULT_MODEL,
    DIM as DEFAULT_DIM,
    QUERY_PREFIX,
    HF_ENDPOINT,
    resolve_embedding_model_path,
    resolve_embedding_device,
)

os.environ['HF_ENDPOINT'] = HF_ENDPOINT

# 尝试导入 torch 和 sentence-transformers
try:
    import torch
    from sentence_transformers import SentenceTransformer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    warnings.warn("sentence-transformers 未安装，知识库功能受限")


@dataclass
class SearchResult:
    """搜索结果"""
    content: str
    source: str
    similarity: float
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        return f"SearchResult(source='{self.source}', similarity={self.similarity:.4f})"


class KnowledgeBase:
    """
    统一知识库接口

    兼容：
    - builder.py v3.0 生成的索引
    - knowledge_qa_fast.py 使用的格式
    """

    def __init__(self, index_dir: Union[str, Path],
                 model_name: str = None,
                 lazy_load: bool = False):
        """
        初始化知识库

        Args:
            index_dir: 知识库目录路径
            model_name: sentence-transformers 模型名称（默认使用索引中的配置或 DEFAULT_MODEL）
            lazy_load: 是否延迟加载模型（只在需要时加载）
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.lazy_load = lazy_load
        self._write_lock = threading.Lock()  # 进程内线程安全
        self._lock_path = self.index_dir / ".write.lock"  # 跨进程文件锁

        # 文件路径
        self.index_path = self.index_dir / "faiss_local.index"
        self.pkl_path = self.index_dir / "faiss_local_index.pkl"
        self.config_path = self.index_dir / "config.json"

        # 初始化状态
        self.index = None
        self.texts = []
        self.metadatas = []
        self.config = {}
        self.model = None
        self.device = 'cpu'
        self.dim = DEFAULT_DIM

        # 加载索引（如果存在）
        if self.index_path.exists():
            self._load_index()
        else:
            print(f"⚠️ 索引文件不存在：{self.index_path}")
            print(f"  请先运行：python builder.py --input ./资料库 --output {index_dir}")

        # 加载模型（除非延迟加载）
        if not lazy_load and _HAS_TRANSFORMERS:
            self._load_model()

    def _load_index(self):
        """加载 FAISS 索引和元数据"""
        print(f"📂 加载知识库：{self.index_dir}")

        # 加载 FAISS 索引
        try:
            self.index = faiss.read_index(str(self.index_path))
            print(f"  ✓ FAISS 索引：{self.index.ntotal} 条记录")
        except Exception as e:
            print(f"  ❌ 加载索引失败：{e}")
            return

        # 加载文本和元数据
        try:
            with open(self.pkl_path, 'rb') as f:
                data = pickle.load(f)
            self.texts = data.get('texts', [])
            self.metadatas = data.get('metadatas', [])
            # 兼容旧格式：如果存储了 dim 和 model 就使用，否则使用默认值
            self.dim = data.get('dim', DEFAULT_DIM)

            # 如果指定了模型名称，使用存储的；否则使用默认值
            if self.model_name is None:
                self.model_name = data.get('model', DEFAULT_MODEL)

            print(f"  ✓ 文本映射：{len(self.texts)} 条")
            if data.get('dim'):
                print(f"  ✓ 维度：{self.dim}")
            if data.get('model'):
                print(f"  ✓ 模型：{self.model_name}")
        except Exception as e:
            print(f"  ❌ 加载元数据失败：{e}")
            return

        # 加载配置
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"  ✓ 配置信息：v{self.config.get('version', 'unknown')}")

            # 检查维度兼容性
            if self.dim != DEFAULT_DIM:
                print(f"  ⚠️ 索引维度({self.dim})与当前模型维度({DEFAULT_DIM})不匹配，请重建索引")

    def _load_model(self):
        """加载 embedding 模型"""
        if not _HAS_TRANSFORMERS:
            raise ImportError("sentence-transformers 未安装，无法加载模型")

        if self.model is not None:
            return

        print(f"🔄 加载 embedding 模型...")

        self.device = resolve_embedding_device(torch)

        model_name = self.model_name or DEFAULT_MODEL
        model_path = resolve_embedding_model_path() if model_name == DEFAULT_MODEL else model_name
        self.model = SentenceTransformer(model_path, device=self.device)
        self.model.to(self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

        print(f"  ✓ 模型：{model_name}")
        print(f"  ✓ 设备：{self.device}, 维度：{self.dim}")

    def _ensure_model_loaded(self):
        """确保模型已加载（用于延迟加载模式）"""
        if self.model is None:
            self._load_model()

    def search(self, query: str, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None,
               min_similarity: float = 0.0) -> List[SearchResult]:
        """
        检索知识库

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件，如 {"source": "xxx.docx", "category": "通知模板"}
            min_similarity: 最小相似度阈值（0-1）

        Returns:
            List[SearchResult]: 检索结果列表
        """
        if self.index is None or len(self.texts) == 0:
            print("⚠️ 知识库为空，请先构建索引")
            return []

        self._ensure_model_loaded()

        # BCE 模型需要 query 前缀
        prefixed_query = QUERY_PREFIX + query

        # 编码查询
        query_vec = self.model.encode([prefixed_query], device=self.device)
        query_vec = np.array(query_vec).astype('float32')
        faiss.normalize_L2(query_vec)

        # 检索（多取一些用于过滤）
        search_k = min(top_k * 3, len(self.texts)) if filters else min(top_k, len(self.texts))
        distances, indices = self.index.search(query_vec, search_k)

        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue

            # 计算相似度（内积 -> 余弦相似度）
            similarity = float(distances[0][i])

            # 应用最小相似度过滤
            if similarity < min_similarity:
                continue

            metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}

            # 应用元数据过滤
            if filters and not self._match_filters(metadata, filters):
                continue

            results.append(SearchResult(
                content=self.texts[idx],
                source=metadata.get('source', 'unknown'),
                similarity=similarity,
                metadata=metadata
            ))

            if len(results) >= top_k:
                break

        return results

    def _match_filters(self, metadata: Dict, filters: Dict) -> bool:
        """检查是否匹配过滤条件"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            meta_value = metadata[key]
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            elif isinstance(value, str) and isinstance(meta_value, str):
                # 支持部分匹配
                if value.lower() not in meta_value.lower():
                    return False
            elif meta_value != value:
                return False
        return True

    def add_documents(self, documents: List[Dict[str, Any]], save: bool = True):
        """
        添加文档到知识库（增量更新，线程安全）

        Args:
            documents: 文档列表，每项包含 {"content": "文本", "metadata": {...}}
            save: 是否立即保存到文件
        """
        if not documents:
            return

        # 线程锁 + 文件锁（跨进程互斥）
        with self._write_lock:
            self._ensure_lock_file()
            with open(self._lock_path, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    self._ensure_model_loaded()
                    previous_index = faiss.clone_index(self.index) if self.index is not None else None
                    previous_texts = list(self.texts)
                    previous_metadatas = list(self.metadatas)

                    print(f"📝 添加 {len(documents)} 个文档...")

                    try:
                        # 编码新文档
                        texts = [doc["content"] for doc in documents]
                        embeddings = self.model.encode(texts, show_progress_bar=True)
                        embeddings = np.array(embeddings).astype('float32')
                        faiss.normalize_L2(embeddings)

                        # 添加到索引
                        if self.index is None:
                            self.index = faiss.IndexFlatIP(self.dim)

                        self.index.add(embeddings)

                        # 保存文本和元数据
                        for doc in documents:
                            self.texts.append(doc["content"])
                            self.metadatas.append(doc.get("metadata", {}))

                        print(f"✓ 知识库现在包含 {len(self.texts)} 条记录")

                        # 保存到文件
                        if save:
                            self.save()
                    except Exception:
                        # 保存或索引追加失败时回滚内存状态，避免半成功污染检索。
                        self.index = previous_index
                        self.texts = previous_texts
                        self.metadatas = previous_metadatas
                        raise
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def delete_by_source(self, source: str) -> int:
        """
        按来源删除文档（需要重建索引）

        Args:
            source: 来源文件路径或名称

        Returns:
            int: 删除的文档数量
        """
        with self._write_lock:
            self._ensure_lock_file()
            with open(self._lock_path, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    # 找出要删除的索引
                    to_delete = [i for i, meta in enumerate(self.metadatas)
                                 if source in meta.get('source', '')]

                    if not to_delete:
                        return 0

                    print(f"🗑️ 删除 {len(to_delete)} 条来自 {source} 的记录...")

                    # FAISS 不支持直接删除，需要重建索引
                    keep_indices = [i for i in range(len(self.texts)) if i not in to_delete]

                    self.texts = [self.texts[i] for i in keep_indices]
                    self.metadatas = [self.metadatas[i] for i in keep_indices]

                    # 重建索引
                    if self.texts:
                        self._ensure_model_loaded()
                        embeddings = self.model.encode(self.texts, show_progress_bar=True)
                        embeddings = np.array(embeddings).astype('float32')
                        faiss.normalize_L2(embeddings)

                        self.index = faiss.IndexFlatIP(self.dim)
                        self.index.add(embeddings)
                    else:
                        self.index = faiss.IndexFlatIP(self.dim)

                    self.save()
                    return len(to_delete)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def delete_by_content_hash(self, content_hash: str) -> int:
        """按内容哈希精确删除文档片段。"""
        return self.replace_by_content_hash(content_hash, [])

    def snapshot_state(self) -> Dict[str, Any]:
        """创建当前索引状态快照，用于管理操作失败回滚。"""
        with self._write_lock:
            self._ensure_lock_file()
            with open(self._lock_path, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    return {
                        "index": faiss.clone_index(self.index) if self.index is not None else None,
                        "texts": list(self.texts),
                        "metadatas": pickle.loads(pickle.dumps(self.metadatas)),
                        "dim": self.dim,
                        "model_name": self.model_name,
                    }
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        """恢复 snapshot_state() 创建的快照并落盘。"""
        if not snapshot:
            return
        with self._write_lock:
            self._ensure_lock_file()
            with open(self._lock_path, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    self.index = faiss.clone_index(snapshot["index"]) if snapshot.get("index") is not None else None
                    self.texts = list(snapshot.get("texts", []))
                    self.metadatas = pickle.loads(pickle.dumps(snapshot.get("metadatas", [])))
                    self.dim = snapshot.get("dim", self.dim)
                    self.model_name = snapshot.get("model_name", self.model_name)
                    if self.index is None:
                        self.index = faiss.IndexFlatIP(self.dim)
                    self.save()
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def update_metadata_by_content_hash(self, content_hash: str, updates: Dict[str, Any]) -> int:
        """批量更新同一文件的元数据，不改变向量本身。"""
        if not content_hash or not updates:
            return 0

        allowed_updates = dict(updates)
        with self._write_lock:
            self._ensure_lock_file()
            with open(self._lock_path, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    updated = 0
                    for meta in self.metadatas:
                        if meta.get("content_hash") != content_hash:
                            continue
                        meta.update(allowed_updates)
                        updated += 1

                    if updated:
                        self.save()
                    return updated
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def replace_by_content_hash(self, content_hash: str, documents: List[Dict[str, Any]]) -> int:
        """
        原子替换某个文件的全部向量片段。

        FAISS IndexFlat 不支持按条件原地删除，因此这里先在内存中构建完整的新索引，
        全部成功后再替换当前状态并落盘，避免重建失败破坏旧索引。
        """
        if not content_hash:
            return 0

        with self._write_lock:
            self._ensure_lock_file()
            with open(self._lock_path, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    keep_indices = [
                        i for i, meta in enumerate(self.metadatas)
                        if meta.get("content_hash") != content_hash
                    ]
                    removed = len(self.texts) - len(keep_indices)
                    if removed == 0 and not documents:
                        return 0

                    new_texts = [self.texts[i] for i in keep_indices]
                    new_metadatas = [self.metadatas[i] for i in keep_indices]
                    for doc in documents:
                        new_texts.append(doc["content"])
                        new_metadatas.append(doc.get("metadata", {}))

                    previous_index = faiss.clone_index(self.index) if self.index is not None else None
                    previous_texts = list(self.texts)
                    previous_metadatas = list(self.metadatas)

                    try:
                        if new_texts:
                            self._ensure_model_loaded()
                            embeddings = self.model.encode(new_texts, show_progress_bar=True)
                            embeddings = np.array(embeddings).astype('float32')
                            faiss.normalize_L2(embeddings)
                            new_index = faiss.IndexFlatIP(self.dim)
                            new_index.add(embeddings)
                        else:
                            new_index = faiss.IndexFlatIP(self.dim)

                        self.index = new_index
                        self.texts = new_texts
                        self.metadatas = new_metadatas
                        self.save()
                    except Exception:
                        self.index = previous_index
                        self.texts = previous_texts
                        self.metadatas = previous_metadatas
                        raise

                    return removed
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _ensure_lock_file(self):
        """确保锁文件存在"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not self._lock_path.exists():
            self._lock_path.touch()

    def save(self):
        """保存知识库到文件"""
        if self.index is None:
            print("⚠️ 索引为空，无需保存")
            return

        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp_suffix = f".tmp.{os.getpid()}.{threading.get_ident()}"
        tmp_index_path = Path(str(self.index_path) + tmp_suffix)
        tmp_pkl_path = Path(str(self.pkl_path) + tmp_suffix)
        tmp_config_path = Path(str(self.config_path) + tmp_suffix)

        try:
            # 先写临时文件，再原子替换，避免进程中断留下半写文件。
            faiss.write_index(self.index, str(tmp_index_path))

            index_data = {
                "texts": self.texts,
                "metadatas": self.metadatas,
                "dim": self.dim,
                "model": self.model_name or DEFAULT_MODEL,
                "updated_at": datetime.now().isoformat()
            }
            with open(tmp_pkl_path, 'wb') as f:
                pickle.dump(index_data, f)

            config = {
                "version": "3.0",
                "updated_at": datetime.now().isoformat(),
                "embedding_model": self.model_name or DEFAULT_MODEL,
                "embedding_dim": self.dim,
                "total_documents": len(self.texts),
            }
            with open(tmp_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            os.replace(tmp_index_path, self.index_path)
            os.replace(tmp_pkl_path, self.pkl_path)
            os.replace(tmp_config_path, self.config_path)
        finally:
            for path in (tmp_index_path, tmp_pkl_path, tmp_config_path):
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass

        print(f"💾 知识库已保存到 {self.index_dir}")

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        return {
            "total_documents": len(self.texts),
            "embedding_dim": self.dim,
            "model": self.model_name or DEFAULT_MODEL,
            "device": self.device,
            "sources": list(set(meta.get('source', 'unknown') for meta in self.metadatas)),
            "categories": list(set(meta.get('category', 'unknown') for meta in self.metadatas)),
            "config": self.config
        }

    def list_sources(self) -> List[str]:
        """列出所有文档来源"""
        return sorted(set(meta.get('source', 'unknown') for meta in self.metadatas))

    def list_categories(self) -> List[str]:
        """列出所有分类"""
        return sorted(set(meta.get('category', 'unknown') for meta in self.metadatas))


# ========== 便捷函数 ==========

def load_knowledge_base(index_dir: Union[str, Path]) -> KnowledgeBase:
    """加载知识库的便捷函数"""
    return KnowledgeBase(index_dir)


def quick_search(query: str, index_dir: Union[str, Path] = "./knowledge_base",
                 top_k: int = 3) -> List[SearchResult]:
    """快速检索的便捷函数"""
    kb = KnowledgeBase(index_dir, lazy_load=False)
    return kb.search(query, top_k=top_k)


# ========== 测试代码 ==========

if __name__ == "__main__":
    import sys

    # 测试加载
    if len(sys.argv) > 1:
        index_dir = sys.argv[1]
    else:
        index_dir = "./knowledge_base"

    print("=" * 60)
    print("知识库核心模块测试")
    print("=" * 60)

    # 加载知识库
    kb = KnowledgeBase(index_dir)

    # 显示统计信息
    print(f"\n📊 知识库统计：")
    stats = kb.get_stats()
    print(f"  总文档数：{stats['total_documents']}")
    print(f"  维度：{stats['embedding_dim']}")
    print(f"  模型：{stats['model']}")
    print(f"  来源数：{len(stats['sources'])}")
    print(f"  分类数：{len(stats['categories'])}")

    # 测试检索
    if stats['total_documents'] > 0:
        test_queries = [
            "公文格式",
            "会议通知",
            "区域发展"
        ]

        for query in test_queries:
            print(f"\n🔍 查询：{query}")
            results = kb.search(query, top_k=2)
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r.source}] 相似度：{r.similarity:.4f}")
                print(f"     内容：{r.content[:80]}...")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
