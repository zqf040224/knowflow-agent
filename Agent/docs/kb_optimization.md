# 知识库检索优化方案

本文档记录当前知识库检索能力与后续优化路线。

## 当前状态

已实现：
- ✅ 语义检索：sentence-transformers (MiniLM) + FAISS
- ✅ 向量相似度：余弦相似度
- ✅ 知识库增强：检索结果拼接到 Prompt
- ✅ 混合检索：BM25 关键词 + FAISS 语义 + RRF 融合
- ✅ 元数据加权：文件名、Sheet、章节、列名、行类型等字段增强排序
- ✅ 表格行回填：涉及 spreadsheet 来源时优先回填结构化原始行
- ✅ 回答质量闭环（第一版）：AnswerPlanner + EvidenceGate + AnswerVerifier
- ✅ DeepSeek 增强规划：`ANSWER_PLANNER=auto|rules|llm`，默认只在复杂问题上调用 DeepSeek 生成结构化检索计划

下一步路线：
- 可选 Cross-Encoder reranker：通过 `RAG_RERANKER=off|local|api` 接入，默认关闭并保留轻量 rerank fallback。
- RAG eval 数据集：从 beta feedback、典型问法、文件定位和表格数值问答沉淀 50-100 条 golden cases。
- 质量指标：先用规则指标检查来源命中、证据不足拒答、内部坐标泄露和虚构文件名，后续再接 Ragas/LangSmith 等评测。

---

## 方案 1：混合检索（Hybrid Search，已实现）

### 问题
纯向量检索对精确匹配（如文号、编号）效果不好。

例如搜索："示例单位〔2025〕1号"
- 向量检索：可能匹配到相似但不同的文号
- 需要：精确匹配该文号

### 解决方案
结合向量检索 + 关键词检索（BM25）

```python
def hybrid_search(query, alpha=0.7):
    """
    alpha: 向量检索权重 (0-1)
    1-alpha: 关键词检索权重
    """
    # 1. 向量检索（语义相似度）
    vector_results = vector_search(query, top_k=50)

    # 2. BM25 关键词检索（精确匹配）
    keyword_results = bm25_search(query, top_k=50)

    # 3. 融合排序
    fused_results = []
    all_docs = set(vector_results.keys()) | set(keyword_results.keys())

    for doc_id in all_docs:
        vector_score = vector_results.get(doc_id, 0)
        keyword_score = keyword_results.get(doc_id, 0)

        # 加权融合
        final_score = alpha * vector_score + (1 - alpha) * keyword_score
        fused_results.append((doc_id, final_score))

    # 按融合分数排序
    fused_results.sort(key=lambda x: x[1], reverse=True)
    return fused_results[:10]
```

### 依赖状态
```bash
pip install rank-bm25 jieba
```

当前 `requirements.txt` 已包含 `rank-bm25`，`jieba` 由项目环境提供。

### 实现代码参考
```python
from rank_bm25 import BM25Okapi
import jieba

class HybridRetriever:
    def __init__(self, texts):
        self.texts = texts
        # 构建 BM25 索引
        tokenized = [list(jieba.cut(text)) for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=10):
        # BM25 检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 向量检索（已有）
        vector_scores = self.vector_search(query)

        # 融合（RRF 算法）
        return self.reciprocal_rank_fusion(bm25_scores, vector_scores)
```

### 预期效果
- 语义匹配："会议通知怎么写" → 匹配到通知模板
- 精确匹配："示例单位〔2025〕1号" → 精确找到该文件

---

## 方案 2：检索后重排序（Reranking，部分实现）

### 问题
FAISS 返回的 Top-K 是按向量相似度排序，但不一定是最相关的。

### 当前实现

当前 `_rerank_results` 已实现无新增依赖的轻量重排：
- RRF/相似度基础分
- 查询关键词覆盖
- 格式规范加权
- 元数据命中加权

### 下一步解决方案
两阶段检索：召回（粗排）+ 重排序（精排）

```python
def search_with_rerank(query, top_k=5):
    # 第一阶段：召回更多候选（粗排）
    candidates = vector_search(query, top_k=20)  # 多召回一些

    # 第二阶段：精确重排序（精排）
    for doc in candidates:
        # 多因子打分
        doc.final_score = (
            doc.vector_similarity * 0.5 +      # 向量相似度
            doc.keyword_match_score * 0.2 +     # 关键词匹配
            doc.recency_score * 0.15 +          # 时效性（越新越好）
            doc.source_authority * 0.15         # 来源权威性
        )

    # 按最终分数排序
    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates[:top_k]
```

### 重排序因子详解

| 因子 | 权重 | 说明 |
|------|------|------|
| 向量相似度 | 50% | 语义相关性 |
| 关键词匹配 | 20% | 查询词在文档中出现的频率 |
| 时效性 | 15% | 文档日期越近分数越高 |
| 来源权威 | 15% | 官方文件 > 普通文档 |

### 时效性计算
```python
def recency_score(doc_date):
    """文档越新分数越高，指数衰减"""
    days_old = (now - doc_date).days
    return 1.0 / (1 + days_old / 365)  # 一年后衰减到 50%
```

---

## 方案 3：元数据过滤 + 语义检索

### 问题
用户只想搜"通知模板"，但返回了所有类型的文档。

### 解决方案
先按元数据过滤，再语义检索。

```python
def search_with_filter(query, filters=None):
    """
    filters: {
        "category": "通知模板",  # 分类
        "file_type": ".docx",     # 文件类型
        "date_range": ["2024-01", "2025-12"]  # 日期范围
    }
    """
    # 1. 元数据过滤（缩小候选集）
    candidates = []
    for doc in all_documents:
        if filters.get("category") and doc.category != filters["category"]:
            continue
        if filters.get("file_type") and not doc.source.endswith(filters["file_type"]):
            continue
        if filters.get("date_range"):
            if not (filters["date_range"][0] <= doc.date <= filters["date_range"][1]):
                continue
        candidates.append(doc)

    # 2. 在候选集中语义检索
    return vector_search_in_candidates(query, candidates)
```

### 前端交互
```html
搜索: [输入关键词]

筛选条件:
☑️ 分类: [通知模板 ▼]
☑️ 日期: [2024年 ▼]
☑️ 类型: [Word ▼]

[搜索]
```

---

## 方案 4：上下文压缩（Context Compression）

### 问题
检索到的片段太长，占用大量 Token，且包含无关信息。

### 解决方案
提取与查询最相关的句子，压缩上下文。

```python
def compress_documents(documents, query, max_length=1000):
    """
    压缩文档，只保留与查询相关的部分
    """
    compressed = []

    for doc in documents:
        # 切分句子
        sentences = doc.split('。')

        # 计算每句与查询的相关性
        relevant_sentences = []
        for sent in sentences:
            score = sentence_similarity(sent, query)
            if score > 0.5:  # 只保留相关度高的句子
                relevant_sentences.append((sent, score))

        # 按相关性排序，取 Top-N
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        compressed_doc = '。'.join([s[0] for s in relevant_sentences[:3]])

        compressed.append(compressed_doc)

    return '\n\n'.join(compressed)
```

### 节省 Token 示例
```
原始文档: 2000 字
压缩后: 300 字（只保留关键句子）
节省: 85% Token
```

---

## 方案 5：查询扩展（Query Expansion）

### 问题
用户查询太简短，检索效果不佳。

例如："通知" → 太宽泛

### 解决方案
自动扩展查询词

```python
def expand_query(query):
    """查询扩展"""
    expansions = {
        "通知": ["会议通知", "放假通知", "行政通知"],
        "报告": ["工作报告", "调研报告", "述职报告"],
        "请示": ["经费请示", "活动请示", "人事请示"]
    }

    expanded = [query]
    for keyword, synonyms in expansions.items():
        if keyword in query:
            expanded.extend(synonyms)

    # 去重
    return list(set(expanded))

# 使用
queries = expand_query("通知")
# ["通知", "会议通知", "放假通知", "行政通知"]

# 多查询检索，合并结果
all_results = []
for q in queries:
    all_results.extend(vector_search(q))

# 去重后返回
return deduplicate(all_results)
```

---

## 方案 6：检索结果反馈学习

### 问题
检索质量固定，无法根据用户行为优化。

### 解决方案
记录用户点击，调整排序权重。

```python
class FeedbackLearning:
    def __init__(self):
        self.click_history = {}  # 查询 -> 点击的文档

    def record_click(self, query, doc_id):
        """记录用户点击"""
        if query not in self.click_history:
            self.click_history[query] = []
        self.click_history[query].append(doc_id)

    def boost_popular_docs(self, query, results):
        """提升历史点击多的文档"""
        popular_docs = self.click_history.get(query, [])

        for result in results:
            if result.doc_id in popular_docs:
                # 点击次数越多，提升越大
                boost = 1 + popular_docs.count(result.doc_id) * 0.1
                result.score *= boost

        return sorted(results, key=lambda x: x.score, reverse=True)
```

---

## 实施建议

### 优先级排序

| 优先级 | 方案 | 难度 | 效果 |
|--------|------|------|------|
| P1 | 元数据过滤 | ⭐ 简单 | ⭐⭐⭐ 显著 |
| P2 | 混合检索 | ⭐⭐ 中等 | ⭐⭐⭐ 显著 |
| P3 | 重排序 | ⭐⭐ 中等 | ⭐⭐ 较好 |
| P4 | 上下文压缩 | ⭐⭐ 中等 | ⭐⭐ 省成本 |
| P5 | 查询扩展 | ⭐⭐⭐ 复杂 | ⭐⭐ 较好 |
| P6 | 反馈学习 | ⭐⭐⭐ 复杂 | ⭐⭐⭐ 长期效果好 |

### 推荐实施路径

```
Phase 1 (现在够用了)
└── 基础语义检索（已实现）

Phase 2 (发现问题后)
├── 添加元数据过滤
└── 添加混合检索

Phase 3 (优化体验)
├── 添加重排序
└── 添加上下文压缩

Phase 4 (高级功能)
├── 查询扩展
└── 反馈学习
```

---

## 参考实现代码

完整的实现代码可参考：
- LangChain Retrievers: https://python.langchain.com/docs/modules/data_connection/retrievers/
- FAISS 高级用法: https://faiss.ai/
- BM25 算法: https://github.com/dorianbrown/rank_bm25

---

**文档版本**: v1.0
**创建日期**: 2025-04-17
**状态**: 待实现（保持现状，按需启用）
