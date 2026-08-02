"""
知识库 Agent - 混合检索（BM25 关键词 + FAISS 语义 + RRF 融合）
支持模型和索引的单例缓存，避免重复加载
"""
import os
if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pickle
import time
import re
import json
import threading
import torch
import faiss
import numpy as np
import jieba

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding_config import (
    MODEL_NAME,
    DIM,
    QUERY_PREFIX,
    HF_ENDPOINT,
    build_access_filter,
    UserInfo,
    ACCESS_PUBLIC,
    resolve_embedding_model_path,
    resolve_embedding_device,
)

os.environ['HF_ENDPOINT'] = HF_ENDPOINT

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from agents.base_agent import BaseAgent, AgentResult
from cache import CacheManager
from spreadsheet_store import SpreadsheetStore
from storage_config import KNOWLEDGE_BASE_DIR, SPREADSHEET_DB_PATH as CONFIG_SPREADSHEET_DB_PATH

INDEX_DIR = str(KNOWLEDGE_BASE_DIR)
INDEX_PATH = os.path.join(INDEX_DIR, 'faiss_local_index.pkl')
FAISS_PATH = os.path.join(INDEX_DIR, 'faiss_local.index')
SPREADSHEET_DB_PATH = str(CONFIG_SPREADSHEET_DB_PATH)

# 全局缓存
_model_cache = None
_index_cache = None
_index_data_cache = None
_device_cache = None
_index_mtime = 0  # 索引文件修改时间，用于跨 worker 检测更新
_bm25_cache = None  # BM25 索引缓存
_RRF_K = 60  # RRF 平滑参数
_RANKING_VERSION = "document-hybrid-v8"
_index_reload_lock = threading.RLock()


def _reranker_mode() -> str:
    mode = os.getenv("RAG_RERANKER", "off").strip().lower()
    return mode if mode in {"off", "local", "api"} else "off"


def _rerank_top_n() -> int:
    try:
        return max(1, int(os.getenv("RAG_RERANK_TOP_N", "20")))
    except ValueError:
        return 20


def _context_top_k() -> int:
    try:
        return max(1, int(os.getenv("RAG_CONTEXT_TOP_K", "8")))
    except ValueError:
        return 8


def _ranking_cache_version() -> str:
    return f"{_RANKING_VERSION}|reranker={_reranker_mode()}|rerank_top_n={_rerank_top_n()}|context_top_k={_context_top_k()}"

def get_device():
    return resolve_embedding_device(torch)


class KnowledgeAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Knowledge",
            description="知识库检索器 - 检索公文格式规范、参考范文",
            **kwargs,
        )
        self._load_index()
        self.cache_manager = CacheManager()

    def _load_index(self):
        global _model_cache, _index_cache, _index_data_cache, _device_cache, _index_mtime

        with _index_reload_lock:
            # 加载模型（仅首次）
            if _model_cache is None:
                device = get_device()
                print(f"Knowledge Agent 初始化 - 使用设备: {device}")

                print("  加载 embedding 模型...")
                t0 = time.time()
                try:
                    _model_cache = SentenceTransformer(resolve_embedding_model_path(), device=device)
                    _device_cache = device
                    _model_cache.to(device)
                    print(f"  模型加载完成: {time.time()-t0:.2f}秒")
                except Exception as e:
                    print(f"  ⚠️ 模型加载失败: {e}")
                    print("  跳过模型加载，继续启动服务")
                    _model_cache = None
                    _device_cache = device
            else:
                print("Knowledge Agent 使用缓存的模型")

            # 检查磁盘文件是否比内存中的新（跨 worker 更新检测）
            disk_mtime = os.path.getmtime(FAISS_PATH) if os.path.exists(FAISS_PATH) else 0
            needs_reload = (_index_cache is None or disk_mtime > _index_mtime)

            # 加载 FAISS 索引（首次、缓存被清除后、或磁盘文件更新后重新加载）
            if needs_reload:
                print("  加载 FAISS 索引...")
                t0 = time.time()
                if os.path.exists(FAISS_PATH) and os.path.exists(INDEX_PATH):
                    _index_cache = faiss.read_index(FAISS_PATH)
                    with open(INDEX_PATH, 'rb') as f:
                        _index_data_cache = pickle.load(f)
                    _index_mtime = disk_mtime
                    print(f"  索引加载完成: {time.time()-t0:.2f}秒, {len(_index_data_cache['texts'])} 文本块")
                else:
                    print(f"  ⚠️ 索引文件不存在: {FAISS_PATH}")
                    _index_cache = None
                    _index_data_cache = {'texts': [], 'metadatas': []}

            self.model = _model_cache
            self.device = _device_cache
            self.faiss_index = _index_cache
            self.index_data = _index_data_cache

            # 构建 BM25 索引（全局缓存，首次构建后复用）
            global _bm25_cache
            if needs_reload and self.index_data and self.index_data.get('texts'):
                texts = self.index_data['texts']
                metadatas = self.index_data.get('metadatas') or [{} for _ in texts]
                tokenized = [
                    list(jieba.cut(self._bm25_document_text(text, metadatas[i] if i < len(metadatas) else {})))
                    for i, text in enumerate(texts)
                ]
                _bm25_cache = BM25Okapi(tokenized)
                print(f"  BM25 索引构建完成: {len(texts)} 篇文本")
            self.bm25 = _bm25_cache

    @classmethod
    def reload_cache(cls):
        """清除索引缓存，下次检索自动加载最新索引"""
        global _index_cache, _index_data_cache, _index_mtime
        with _index_reload_lock:
            _index_cache = None
            _index_data_cache = None
            _index_mtime = 0

    def refresh(self, force: bool = False):
        """刷新当前实例使用的索引缓存，默认只在磁盘索引更新后重载。"""
        if force:
            self.__class__.reload_cache()
        self._load_index()

    def add_documents(self, documents: list):
        """添加文档到知识库索引"""
        from knowledge_base.core import KnowledgeBase
        kb = KnowledgeBase(str(KNOWLEDGE_BASE_DIR), lazy_load=True)
        kb.add_documents(documents)
        self.refresh()

    def get_system_prompt(self) -> str:
        return """你是一个公文规范顾问。你的职责是：
1. 根据检索到的知识库内容，提供公文格式规范指导
2. 提供参考范文的写作风格和结构
3. 确保格式规范准确无误

请以结构化方式输出，包含：
- 格式规范要点
- 参考范文的结构特点
- 写作注意事项"""

    def process(self, input_data: dict, on_think=None) -> AgentResult:
        queries = input_data.get("knowledge_queries", [])
        user_request = input_data.get("user_request", "")
        search_context = input_data.get("search_context", "")
        key_points = input_data.get("key_points", [])
        user_info = input_data.get("user_info")  # UserInfo 或 dict

        if not queries:
            queries = [user_request]

        queries = self._rewrite_queries(queries, user_request)

        # 如果有 search_context 或 key_points，优化查询词
        if search_context or key_points:
            enriched = []
            if key_points:
                enriched.extend(key_points[:3])
            if search_context:
                enriched.append(search_context[:200])
            queries = enriched + queries

        self._emit_think(on_think, "📚", "正在检索知识库...")

        access_filter = build_access_filter(user_info)
        process_cache_key = self._process_cache_key(
            queries,
            user_request,
            search_context,
            key_points,
            access_filter,
        )
        cached_payload = self.cache_manager.get(process_cache_key)
        if cached_payload:
            self._emit_think(on_think, "⚡", "使用知识库缓存结果")
            return AgentResult(
                success=True,
                content=cached_payload["content"],
                agent_name=self.name,
                confidence=cached_payload.get("confidence", 0.9),
                metadata=cached_payload.get("metadata", {}),
            )

        all_results = []
        for query in queries:
            results = self._search_hybrid(query, access_filter)
            all_results.extend(results)

        all_results = self._deduplicate(all_results)
        all_results = self._hydrate_spreadsheet_rows(all_results, user_request=user_request)
        all_results = self._rerank_results(all_results, user_request)
        all_results = self._apply_optional_reranker(all_results, user_request)

        format_results = [r for r in all_results if r.get("is_format")]
        other_results = [r for r in all_results if not r.get("is_format")]

        self._emit_think(
            on_think,
            "📖",
            f"检索到 {len(format_results)} 条格式规范，{len(other_results)} 条参考范文",
        )

        context = self._build_context(all_results)

        self._emit_think(on_think, "📋", "正在整理规范要点...")

        metadata = {
            "format_count": len(format_results),
            "reference_count": len(other_results),
            "results": [
                {
                    "source": r["source"],
                    "filename": r.get("filename", ""),
                    "source_path": r.get("source_path", r["source"]),
                    "category": r.get("category", ""),
                    "department": r.get("department", ""),
                    "access_level": r.get("access_level", ACCESS_PUBLIC),
                    "chunk_index": r.get("chunk_index", -1),
                    "total_chunks": r.get("total_chunks", -1),
                    "doc_type": r.get("doc_type", ""),
                    "content_hash": r.get("content_hash", ""),
                    "parser_type": r.get("parser_type", ""),
                    "source_type": r.get("source_type", "document"),
                    "sheet_name": r.get("sheet_name", ""),
                    "row_start": r.get("row_start"),
                    "row_end": r.get("row_end"),
                    "row_type": r.get("row_type", "data"),
                    "column_headers": r.get("column_headers", []),
                    "page_start": r.get("page_start"),
                    "page_end": r.get("page_end"),
                    "section_title": r.get("section_title", ""),
                    "heading_path": r.get("heading_path", []),
                    "chunk_text_hash": r.get("chunk_text_hash", ""),
                    "parse_warnings": r.get("parse_warnings", []),
                    "spreadsheet_values": r.get("spreadsheet_values", []),
                    "similarity": r["similarity"],
                    "rerank_score": r.get("_rerank_score", r.get("_rrf_score", r["similarity"])),
                    "is_format": r.get("is_format", False),
                }
                for r in all_results[:10]
            ],
        }
        self.cache_manager.set(
            process_cache_key,
            {"content": context, "confidence": 0.9, "metadata": metadata},
            ttl=900,
        )

        return AgentResult(
            success=True,
            content=context,
            agent_name=self.name,
            confidence=0.9,
            metadata=metadata,
        )

    def _search(self, query: str, access_filter: dict = None, top_k: int = 5) -> list:
        global _index_cache, _index_mtime

        # 跨 worker 更新检测：磁盘文件被其他进程修改
        if os.path.exists(FAISS_PATH):
            cur_mtime = os.path.getmtime(FAISS_PATH)
            if cur_mtime > _index_mtime:
                self.__class__.reload_cache()
                self._load_index()

        # 跨实例同步：同一进程内另一个 KnowledgeAgent 实例已刷新全局缓存
        if self.faiss_index is not _index_cache:
            self._load_index()

        filter_key = str(sorted(access_filter.items())) if access_filter else "no_filter"
        cache_key = self.cache_manager.get_knowledge_cache_key(
            f"dense|{_ranking_cache_version()}|{query}|{filter_key}|{_index_mtime}"
        )
        cached_results = self.cache_manager.get(cache_key)
        if cached_results:
            return cached_results

        if self.faiss_index is None or self.model is None:
            return []

        prefixed_query = QUERY_PREFIX + query
        query_vec = self.model.encode([prefixed_query], device=self.device)
        query_vec = np.array(query_vec).astype('float32')
        faiss.normalize_L2(query_vec)

        # 多取一些以应对权限过滤
        fetch_k = min(top_k * 3, len(self.index_data['texts'])) if access_filter else top_k
        distances, indices = self.faiss_index.search(query_vec, fetch_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.index_data['texts']):
                continue

            metadata = self.index_data['metadatas'][idx]
            if not self._match_access(metadata, access_filter):
                continue

            results.append({
                "text": self.index_data['texts'][idx],
                "source": metadata['source'],
                "source_path": metadata.get('source_path', metadata.get('source', '')),
                "filename": metadata.get('filename', ''),
                "category": metadata.get('category', ''),
                "department": metadata.get('department', ''),
                "access_level": metadata.get('access_level', ACCESS_PUBLIC),
                "chunk_index": metadata.get('chunk_index', -1),
                "total_chunks": metadata.get('total_chunks', -1),
                "doc_type": metadata.get('doc_type', ''),
                "content_hash": metadata.get('content_hash', ''),
                "parser_type": metadata.get('parser_type', ''),
                "source_type": metadata.get('source_type', 'document'),
                "sheet_name": metadata.get('sheet_name', ''),
                "row_start": metadata.get('row_start'),
                "row_end": metadata.get('row_end'),
                "row_type": metadata.get('row_type', 'data'),
                "column_headers": metadata.get('column_headers', []),
                "page_start": metadata.get('page_start'),
                "page_end": metadata.get('page_end'),
                "section_title": metadata.get('section_title', ''),
                "heading_path": metadata.get('heading_path', []),
                "chunk_text_hash": metadata.get('chunk_text_hash', ''),
                "parse_warnings": metadata.get('parse_warnings', []),
                "similarity": self._dense_similarity(distances[0][i]),
                "raw_score": float(distances[0][i]),
                "is_format": "公文格式" in metadata['source'],
                "source_method": "dense",
            })

            if len(results) >= top_k:
                break

        self.cache_manager.set(cache_key, results, ttl=3600)
        return results

    def _search_bm25(self, query: str, access_filter: dict = None, top_k: int = 5) -> list:
        """BM25 关键词检索"""
        if self.bm25 is None:
            return []

        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)

        # 按分数排序取 top_k × 3（应对权限过滤）
        max_score = float(np.max(scores)) if len(scores) else 0.0
        indexed = [(i, s) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed:
            if score <= 0:
                continue
            metadata = self.index_data['metadatas'][idx]
            if not self._match_access(metadata, access_filter):
                continue

            results.append({
                "text": self.index_data['texts'][idx],
                "source": metadata['source'],
                "source_path": metadata.get('source_path', metadata.get('source', '')),
                "filename": metadata.get('filename', ''),
                "category": metadata.get('category', ''),
                "department": metadata.get('department', ''),
                "access_level": metadata.get('access_level', ACCESS_PUBLIC),
                "chunk_index": metadata.get('chunk_index', -1),
                "total_chunks": metadata.get('total_chunks', -1),
                "doc_type": metadata.get('doc_type', ''),
                "content_hash": metadata.get('content_hash', ''),
                "parser_type": metadata.get('parser_type', ''),
                "source_type": metadata.get('source_type', 'document'),
                "sheet_name": metadata.get('sheet_name', ''),
                "row_start": metadata.get('row_start'),
                "row_end": metadata.get('row_end'),
                "row_type": metadata.get('row_type', 'data'),
                "column_headers": metadata.get('column_headers', []),
                "page_start": metadata.get('page_start'),
                "page_end": metadata.get('page_end'),
                "section_title": metadata.get('section_title', ''),
                "heading_path": metadata.get('heading_path', []),
                "chunk_text_hash": metadata.get('chunk_text_hash', ''),
                "parse_warnings": metadata.get('parse_warnings', []),
                "similarity": self._bm25_similarity(score, max_score),
                "raw_score": float(score),
                "is_format": "公文格式" in metadata['source'],
                "source_method": "bm25",
            })
            if len(results) >= top_k:
                break

        return results

    def _search_hybrid(self, query: str, access_filter: dict = None, top_k: int = 5) -> list:
        """混合检索：BM25 + Dense + RRF 融合"""
        cache_key = self.cache_manager.get_knowledge_cache_key(
            f"hybrid|{_ranking_cache_version()}|{query}|{self._access_filter_key(access_filter)}|{top_k}|{_index_mtime}"
        )
        cached_results = self.cache_manager.get(cache_key)
        if cached_results:
            return cached_results

        fetch_k = top_k * 2

        # 并行执行两种检索
        dense_results = self._search(query, access_filter, fetch_k)
        bm25_results = self._search_bm25(query, access_filter, fetch_k)

        if not bm25_results and not dense_results:
            return []
        if not bm25_results:
            merged = dense_results[:top_k]
            self.cache_manager.set(cache_key, merged, ttl=900)
            return merged
        if not dense_results:
            merged = bm25_results[:top_k]
            self.cache_manager.set(cache_key, merged, ttl=900)
            return merged

        # RRF (Reciprocal Rank Fusion) 融合
        rrf_scores = {}

        for rank, r in enumerate(dense_results):
            key = (r["source"], r["text"][:100])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (_RRF_K + rank + 1)

        for rank, r in enumerate(bm25_results):
            key = (r["source"], r["text"][:100])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (_RRF_K + rank + 1)

        # 构建结果字典（用首次出现的结果对象）
        all_results_map = {}
        for r in dense_results + bm25_results:
            key = (r["source"], r["text"][:100])
            if key not in all_results_map:
                r["_rrf_score"] = rrf_scores.get(key, 0)
                all_results_map[key] = r

        for r in all_results_map.values():
            r["_metadata_score"] = self._metadata_match_score(query, r)

        # 按 RRF 分数 + 文件名/Sheet/列名命中排序，避免表格重复行刷屏压过精确 Sheet。
        merged = sorted(
            all_results_map.values(),
            key=lambda x: x["_rrf_score"] + x.get("_metadata_score", 0),
            reverse=True,
        )
        merged = merged[:top_k]
        self.cache_manager.set(cache_key, merged, ttl=900)
        return merged

    def _process_cache_key(
        self,
        queries: list,
        user_request: str,
        search_context: str,
        key_points: list,
        access_filter: dict,
    ) -> str:
        payload = {
            "queries": queries[:8],
            "user_request": user_request,
            "search_context": search_context[:500] if search_context else "",
            "key_points": key_points[:8],
            "access_filter": access_filter or {},
            "index_mtime": _index_mtime,
            "ranking_version": _ranking_cache_version(),
        }
        return self.cache_manager.get_knowledge_cache_key(
            "process|" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    @staticmethod
    def _access_filter_key(access_filter: dict) -> str:
        return json.dumps(access_filter or {}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _dense_similarity(distance_or_score) -> float:
        """FAISS IndexFlatIP 返回内积分数，归一化后应落在 0-1 区间。"""
        score = float(distance_or_score)
        if score < 0:
            return 0.0
        if score > 1:
            return 1.0
        return score

    @staticmethod
    def _bm25_similarity(score, max_score) -> float:
        if max_score <= 0:
            return 0.0
        normalized = float(score) / float(max_score)
        if normalized < 0:
            return 0.0
        if normalized > 1:
            return 1.0
        return normalized

    @staticmethod
    def _normalize_compact(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).lower()

    @staticmethod
    def _spreadsheet_lookup_score(query: str, result: dict) -> float:
        if result.get("source_type") != "spreadsheet":
            return 0.0

        query_text = KnowledgeAgent._normalize_compact(query or "")
        if not query_text:
            return 0.0

        filename = str(result.get("filename", ""))
        sheet_name = str(result.get("sheet_name", ""))
        headers = " ".join(str(h) for h in result.get("column_headers", []) or [])
        metadata_text = KnowledgeAgent._normalize_compact(f"{filename} {sheet_name} {headers}")

        table_lookup_terms = ("表", "明细", "清单", "标准", "收费", "费用", "金额", "价格", "模板")
        if not any(term in query_text for term in table_lookup_terms):
            return 0.0

        score = 0.0
        for term in ("收费", "费用", "金额", "价格", "标准", "表", "明细", "清单"):
            if term in query_text and term in metadata_text:
                score += 0.035

        if any(term in query_text for term in ("哪里看", "在哪里", "查看", "查询", "包含哪些", "有哪些")):
            score += 0.06

        if "收费标准" in query_text and "收费" in metadata_text and ("表" in metadata_text or "金额" in metadata_text):
            score += 0.14

        return min(score, 0.24)

    @classmethod
    def _bm25_document_text(cls, text: str, metadata: dict) -> str:
        """Build a field-enhanced BM25 document without changing vector text."""
        filename = metadata.get("filename") or os.path.basename(metadata.get("source", ""))
        heading_path = metadata.get("heading_path") or []
        if isinstance(heading_path, str):
            heading_path = [heading_path]
        fields = [
            text or "",
            filename,
            filename,
            filename,
            metadata.get("category", ""),
            metadata.get("doc_type", ""),
            metadata.get("section_title", ""),
            metadata.get("section_title", ""),
            " ".join(str(h) for h in heading_path),
            metadata.get("sheet_name", ""),
            " ".join(str(h) for h in metadata.get("column_headers", []) or []),
        ]
        return "\n".join(str(f) for f in fields if f)

    @staticmethod
    def _metadata_match_score(query: str, result: dict) -> float:
        query_text = KnowledgeAgent._normalize_compact(query or "")
        if not query_text:
            return 0.0

        filename = str(result.get("filename", "")).lower()
        filename_stem = re.sub(r"\.[^.]+$", "", filename)
        sheet_name = str(result.get("sheet_name", "")).lower()
        headers = [str(h).lower() for h in (result.get("column_headers") or [])]
        header_text = "".join(headers)
        section_title = str(result.get("section_title", "")).lower()
        heading_path = result.get("heading_path") or []
        if isinstance(heading_path, str):
            heading_path = [heading_path]
        heading_text = "".join(str(h).lower() for h in heading_path)
        category = str(result.get("category", "")).lower()
        doc_type = str(result.get("doc_type", "")).lower()
        body_text = KnowledgeAgent._normalize_compact(result.get("text", "")[:1200])

        score = 0.0
        filename_norm = KnowledgeAgent._normalize_compact(filename_stem)
        if filename_norm and (filename_norm in query_text or query_text in filename_norm):
            score += 0.12
        if sheet_name and KnowledgeAgent._normalize_compact(sheet_name) in query_text:
            score += 0.12
        section_norm = KnowledgeAgent._normalize_compact(section_title)
        heading_norm = KnowledgeAgent._normalize_compact(heading_text)
        if section_norm and (section_norm in query_text or query_text in section_norm):
            score += 0.10
        if heading_norm and any(part and part in query_text for part in [KnowledgeAgent._normalize_compact(h) for h in heading_path]):
            score += 0.08
        if query_text and len(query_text) >= 4 and query_text in body_text:
            score += 0.06
        if doc_type and doc_type in query_text:
            score += 0.04
        if category and category in query_text:
            score += 0.03
        if result.get("source_type") == "document" and result.get("page_start"):
            score += 0.01
        score += KnowledgeAgent._spreadsheet_lookup_score(query, result)

        query_tokens = {t.lower() for t in jieba.cut(query or "") if len(t.strip()) > 1}
        filename_hits = sum(1 for token in query_tokens if token in filename)
        if filename_hits:
            score += min(filename_hits * 0.025, 0.10)
        heading_hits = sum(
            1 for token in query_tokens
            if token in section_title or token in heading_text or token in category or token in doc_type
        )
        if heading_hits:
            score += min(heading_hits * 0.025, 0.10)
        header_hits = sum(1 for token in query_tokens if token in header_text)
        if header_hits:
            score += min(header_hits * 0.025, 0.10)
        if result.get("row_type") == "summary":
            summary_terms = ("总计", "合计", "小计", "总费用", "总金额", "总额", "总收入", "总支出")
            if any(term in query_text for term in summary_terms):
                score += 0.10
        return score

    @staticmethod
    def _match_access(metadata: dict, access_filter: dict) -> bool:
        if not access_filter:
            return True
        # 检查访问级别
        doc_access = metadata.get("access_level", ACCESS_PUBLIC)
        allowed_levels = access_filter.get("access_level", [])
        if allowed_levels and doc_access not in allowed_levels:
            return False
        # public 文档所有人可见，跳过部门检查
        if doc_access == ACCESS_PUBLIC:
            return True
        # 部门过滤：restricted 文档需匹配用户部门
        allowed_depts = access_filter.get("department", [])
        if allowed_depts:
            doc_dept = metadata.get("department", "")
            if doc_dept not in allowed_depts:
                return False
        return True

    def _deduplicate(self, results: list) -> list:
        seen = set()
        unique = []
        for r in results:
            key = (r["source"], r["text"][:100])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        unique.sort(key=lambda x: x.get("_rerank_score", x.get("_rrf_score", x["similarity"])), reverse=True)
        return unique

    def _build_context(self, results: list) -> str:
        parts = []
        for i, r in enumerate(self._compress_results(results)[:_context_top_k()]):
            if r.get("source_type") == "spreadsheet":
                tag = "【报表数据】"
            else:
                tag = "【格式规范】" if r.get("is_format") else "【参考范文】"
            method = r.get("source_method", "")
            method_tag = {"bm25": "[关键词命中]", "dense": "[语义匹配]"}.get(method, "")
            filename = r.get('filename', '') or r['source'].split('/')[-1]
            location = ""
            if r.get("source_type") == "spreadsheet":
                sheet = r.get("sheet_name") or "未知Sheet"
                row_start = r.get("row_start")
                row_end = r.get("row_end")
                if row_start and row_end and row_start != row_end:
                    location = f"Sheet: {sheet}，行: {row_start}-{row_end}\n"
                elif row_start:
                    location = f"Sheet: {sheet}，行: {row_start}\n"
                else:
                    location = f"Sheet: {sheet}\n"
                if r.get("row_type") == "summary":
                    location += "行类型: 汇总行\n"
            elif r.get("chunk_index", -1) >= 0 and r.get("total_chunks", -1) >= 0:
                location = f"片段: {r.get('chunk_index') + 1}/{r.get('total_chunks')}\n"
                page_start = r.get("page_start")
                page_end = r.get("page_end")
                if page_start and page_end and page_start != page_end:
                    location += f"页码: {page_start}-{page_end}\n"
                elif page_start:
                    location += f"页码: {page_start}\n"
                heading_path = r.get("heading_path") or []
                if isinstance(heading_path, str):
                    heading_path = [heading_path]
                if heading_path:
                    location += f"章节: {' > '.join(str(h) for h in heading_path)}\n"
                elif r.get("section_title"):
                    location += f"章节: {r.get('section_title')}\n"
                warnings = r.get("parse_warnings") or []
                if warnings:
                    location += f"解析提示: {'；'.join(str(w) for w in warnings[:2])}\n"
            caution = ""
            if r.get("source_type") == "spreadsheet":
                caution = "注意: 涉及数值时必须按本行原文引用，不可自行换算、补全或推测。\n"
            parts.append(
                f"[文档{i + 1}] {tag} {method_tag}\n"
                f"文件名: {filename}\n"
                f"{location}"
                f"{caution}"
                f"---\n"
                f"{r['text'][:12000] if r.get('_expanded_spreadsheet_table') else (r['text'][:5000] if r.get('source_type') == 'spreadsheet' else r['text'][:800])}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _is_spreadsheet_table_request(user_request: str) -> bool:
        query = KnowledgeAgent._normalize_compact(user_request or "")
        if not query:
            return False
        table_terms = ("表", "收费", "费用", "金额", "价格", "标准", "清单", "明细")
        list_terms = ("有哪些", "哪些", "内容", "列出", "分别", "所有", "多少", "收费")
        return any(term in query for term in table_terms) and any(term in query for term in list_terms)

    @staticmethod
    def _spreadsheet_row_is_useful(row: dict) -> bool:
        values = row.get("values", {}) or {}
        normalized_values = [
            KnowledgeAgent._normalize_compact(value)
            for value in values.values()
            if KnowledgeAgent._normalize_compact(value)
        ]
        normalized_data_values = [
            KnowledgeAgent._normalize_compact(value)
            for key, value in values.items()
            if KnowledgeAgent._normalize_compact(key) != "序号"
            and KnowledgeAgent._normalize_compact(value)
        ]
        if not normalized_values:
            return False
        if len(set(normalized_values)) == 1:
            return False
        if normalized_data_values and len(set(normalized_data_values)) == 1:
            return False
        header_markers = {"序号", "楼层", "门牌号", "名称", "计费方式", "金额", "备注"}
        header_hits = sum(1 for value in normalized_values if value in header_markers)
        return header_hits < 4

    @staticmethod
    def _spreadsheet_row_matches_request(row: dict, user_request: str) -> bool:
        query = KnowledgeAgent._normalize_compact(user_request or "")
        if not any(term in query for term in ("教室", "会议室", "报告厅", "贵宾厅")):
            return True
        values = row.get("values", {}) or {}
        name_text = KnowledgeAgent._normalize_compact(
            KnowledgeAgent._spreadsheet_value(values, "名称", "教室")
        )
        venue_terms = ("教室", "会议室", "会议厅", "报告厅", "贵宾厅")
        return any(term in name_text for term in venue_terms)

    @classmethod
    def _spreadsheet_semantic_key(cls, filename: str) -> str:
        name = os.path.basename(filename or "")
        name = re.sub(r"\.(?:xlsx?|csv)$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^(?:\d{8}_)?(?:附件|附表)[：:_-]*", "", name)
        name = re.sub(r"[（(][^）)]*(?:版本|副本|复制|copy|Copy|COPY)[^）)]*[）)]", "", name)
        name = re.sub(r"[（(]\d+[）)]", "", name)
        name = re.sub(r"(?:20\d{6}|20\d{2}[-_.年]?\d{1,2}[-_.月]?\d{0,2}日?)", "", name)
        return cls._normalize_compact(name)

    @staticmethod
    def _spreadsheet_version_key(result: dict) -> tuple[str, str, str]:
        candidates = [
            result.get("uploaded_at", ""),
            result.get("updated_at", ""),
            result.get("vector_indexed_at", ""),
            result.get("source_path", ""),
            result.get("source", ""),
            result.get("filename", ""),
        ]
        combined = " ".join(str(item) for item in candidates if item)
        date_matches = re.findall(r"20\d{2}[-_/]?\d{2}[-_/]?\d{2}", combined)
        normalized_date = max((re.sub(r"\D", "", item) for item in date_matches), default="")
        return (
            str(result.get("uploaded_at") or result.get("updated_at") or result.get("vector_indexed_at") or ""),
            normalized_date,
            str(result.get("source_path") or result.get("source") or result.get("filename") or ""),
        )

    @classmethod
    def _prefer_latest_spreadsheet_tables(cls, expanded: list) -> list:
        latest_by_key = {}
        passthrough = []
        for item in expanded:
            key = cls._spreadsheet_semantic_key(item.get("filename") or item.get("source") or "")
            if not key:
                passthrough.append(item)
                continue
            current = latest_by_key.get(key)
            if current is None or cls._spreadsheet_version_key(item) > cls._spreadsheet_version_key(current):
                latest_by_key[key] = item
        return passthrough + list(latest_by_key.values())

    @staticmethod
    def _spreadsheet_value(values: dict, *labels: str) -> str:
        normalized_labels = {KnowledgeAgent._normalize_compact(label) for label in labels}
        for key, value in (values or {}).items():
            if KnowledgeAgent._normalize_compact(key) in normalized_labels:
                return str(value).strip()
        return ""

    @classmethod
    def _format_spreadsheet_row_brief(cls, row: dict) -> str:
        values = row.get("values", {}) or {}
        fields = [
            ("楼层", cls._spreadsheet_value(values, "楼层", "位置")),
            ("门牌号", cls._spreadsheet_value(values, "门牌号", "房间号")),
            ("名称", cls._spreadsheet_value(values, "名称", "教室", "收费项目")),
            ("面积", cls._spreadsheet_value(values, "面积（㎡）", "面积\n（㎡）", "面积")),
            ("人数", cls._spreadsheet_value(values, "可容纳人数", "剧院式", "工位")),
            ("计费方式", cls._spreadsheet_value(values, "计费方式", "日租", "单位")),
            ("金额", cls._spreadsheet_value(values, "金额", "金额（元）", "小时")),
            ("备注", cls._spreadsheet_value(values, "备注")),
        ]
        parts = [f"行{row.get('row_number')}"]
        for label, value in fields:
            if value:
                parts.append(f"{label}:{value}")
        return "；".join(parts)

    def _expand_spreadsheet_table_results(self, results: list, user_request: str, store: SpreadsheetStore) -> list:
        if not self._is_spreadsheet_table_request(user_request):
            return results

        expanded = []
        seen_hashes = set()
        for result in results:
            if result.get("source_type") != "spreadsheet" or not result.get("content_hash"):
                continue
            content_hash = result["content_hash"]
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            rows = store.get_rows_by_source(content_hash, sheet_name=result.get("sheet_name") or None)
            rows = [
                row for row in rows
                if row.get("row_type") in {"data", "summary"}
                and self._spreadsheet_row_is_useful(row)
                and self._spreadsheet_row_matches_request(row, user_request)
            ][:80]
            if not rows:
                continue
            header = (
                f"文件：{result.get('filename') or rows[0].get('filename')}\n"
                f"Sheet：{result.get('sheet_name') or rows[0].get('sheet_name')}\n"
                "以下为同一表格中可用于回答收费/清单问题的数据行："
            )
            text = header + "\n" + "\n".join(self._format_spreadsheet_row_brief(row) for row in rows)
            expanded.append({
                **result,
                "text": text,
                "row_start": rows[0].get("row_number"),
                "row_end": rows[-1].get("row_number"),
                "row_type": "data",
                "spreadsheet_values": [row.get("values", {}) for row in rows],
                "similarity": max(float(result.get("similarity", 0.0)), 0.99),
                "_rerank_score": max(float(result.get("_rerank_score", result.get("similarity", 0.0))), 1.25),
                "_expanded_spreadsheet_table": True,
            })
        if expanded:
            expanded_hashes = {item.get("content_hash") for item in expanded if item.get("content_hash")}
            expanded = self._prefer_latest_spreadsheet_tables(expanded)
            remainder = [
                result for result in results
                if result.get("source_type") != "spreadsheet"
                or result.get("content_hash") not in expanded_hashes
            ]
            return expanded + remainder
        return results

    def _hydrate_spreadsheet_rows(self, results: list, user_request: str = "") -> list:
        """Use structured spreadsheet rows as the source of truth for table data."""
        if not results or not os.path.exists(SPREADSHEET_DB_PATH):
            return results

        spreadsheet_results = [
            r for r in results
            if r.get("source_type") == "spreadsheet"
            and r.get("content_hash")
            and r.get("row_start") is not None
        ]
        if not spreadsheet_results:
            return results

        try:
            store = SpreadsheetStore(SPREADSHEET_DB_PATH)
            for result in spreadsheet_results:
                row_start = result.get("row_start")
                row_end = result.get("row_end") or row_start
                rows = store.get_rows_by_source(
                    result["content_hash"],
                    sheet_name=result.get("sheet_name") or None,
                    row_start=row_start,
                    row_end=row_end,
                )
                if rows:
                    result["text"] = "\n\n".join(row["row_text"] for row in rows)
                    result["spreadsheet_values"] = [row.get("values", {}) for row in rows]
            results = self._expand_spreadsheet_table_results(results, user_request, store)
        except Exception as exc:
            print(f"  ⚠️ 表格精确行回填失败，使用向量片段: {exc}")
        return results

    def _rewrite_queries(self, queries: list, user_request: str) -> list:
        """轻量查询改写：保留原查询，并补充公文类型/关键词组合。"""
        rewritten = []
        for q in queries:
            if q and q not in rewritten:
                rewritten.append(q)

        text = user_request or ""
        doc_terms = [t for t in ["通知", "请示", "报告", "函", "对策建议", "情况反映", "会议纪要"] if t in text]
        if doc_terms:
            for term in doc_terms[:2]:
                q = f"{term} 范文 结构 要点"
                if q not in rewritten:
                    rewritten.append(q)

        keywords = [w for w in re.split(r"[\s，。；、,.!?！？：:（）()]+", text) if len(w) >= 2]
        if keywords:
            q = " ".join(keywords[:6])
            if q and q not in rewritten:
                rewritten.append(q)

        return rewritten[:5]

    def _rerank_results(self, results: list, user_request: str) -> list:
        """无新依赖的轻量重排：RRF/相似度 + 关键词覆盖 + 格式规范加权。"""
        if not results:
            return []
        query_tokens = {t for t in jieba.cut(user_request or "") if len(t.strip()) > 1}
        for r in results:
            heading_path = r.get("heading_path") or []
            if isinstance(heading_path, str):
                heading_path = [heading_path]
            metadata_text = " ".join([
                r.get("filename", ""),
                r.get("category", ""),
                r.get("doc_type", ""),
                r.get("section_title", ""),
                " ".join(str(h) for h in heading_path),
            ])
            text = f"{r.get('text', '')}\n{metadata_text}"
            overlap = sum(1 for t in query_tokens if t in text)
            overlap_score = overlap / max(len(query_tokens), 1)
            base = r.get("_rrf_score", 0) or min(float(r.get("similarity", 0)), 1.0)
            format_boost = 0.04 if r.get("is_format") else 0
            metadata_boost = self._metadata_match_score(user_request, r)
            r["_rerank_score"] = base + overlap_score * 0.25 + format_boost + metadata_boost
        return sorted(results, key=lambda x: x.get("_rerank_score", 0), reverse=True)

    def _apply_optional_reranker(self, results: list, user_request: str) -> list:
        """Optional reranker extension point; disabled by default and safe to fail."""
        mode = _reranker_mode()
        if mode == "off" or not results:
            return results

        top_n = min(_rerank_top_n(), len(results))
        candidates = results[:top_n]
        remainder = results[top_n:]
        try:
            reranked = self._run_external_reranker(candidates, user_request, mode=mode)
            if reranked:
                return reranked + remainder
        except Exception as exc:
            print(f"  ⚠️ RAG reranker({mode}) 不可用，降级为轻量重排: {exc}")
        return results

    def _run_external_reranker(self, candidates: list, user_request: str, *, mode: str) -> list:
        """Placeholder for future local/API cross-encoder reranking."""
        raise RuntimeError(f"RAG_RERANKER={mode} has no configured backend")

    def _compress_results(self, results: list) -> list:
        """限制同一来源占比，避免上下文被单个文件刷屏。"""
        compressed = []
        per_source = {}
        for r in results:
            source = r.get("source", "")
            per_source[source] = per_source.get(source, 0) + 1
            if per_source[source] > 3:
                continue
            compressed.append(r)
            if len(compressed) >= 10:
                break
        return compressed

    def get_health(self, user_info=None) -> dict:
        """返回知识库索引健康状态，供管理端或健康检查使用。"""
        access_filter = build_access_filter(user_info)
        visible_count = 0
        if self.index_data and self.index_data.get("metadatas"):
            visible_count = sum(1 for m in self.index_data["metadatas"] if self._match_access(m, access_filter))
        return {
            "model_loaded": self.model is not None,
            "faiss_loaded": self.faiss_index is not None,
            "bm25_loaded": self.bm25 is not None,
            "text_count": len(self.index_data.get("texts", [])) if self.index_data else 0,
            "metadata_count": len(self.index_data.get("metadatas", [])) if self.index_data else 0,
            "visible_count": visible_count,
            "embedding_dim": DIM,
            "faiss_mtime": _index_mtime,
            "index_path": FAISS_PATH,
        }
