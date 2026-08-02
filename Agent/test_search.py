#!/usr/bin/env python3
"""知识库索引烟测。

当前知识库使用 FAISS + 本地 embedding 索引，元数据存放在
knowledge_base/faiss_local_index.pkl。这个脚本只做轻量级可用性检查，
避免继续依赖已废弃的 tfidf_index.pkl。
"""

import os
import pickle


KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
INDEX_DATA_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "faiss_local_index.pkl")


def load_index_data():
    if not os.path.exists(INDEX_DATA_PATH):
        raise FileNotFoundError(
            f"缺少知识库索引数据: {INDEX_DATA_PATH}\n"
            "请先运行: python builder.py --input ./知识库 --output ./knowledge_base"
        )

    with open(INDEX_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    if not data.get("texts") or not data.get("metadatas"):
        raise AssertionError("索引数据为空或缺少 metadatas")

    return data


def keyword_search(data, query, top_k=5):
    terms = [term for term in query.split() if term] or [query]
    scored = []
    for idx, text in enumerate(data["texts"]):
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append((score, idx))

    scored.sort(reverse=True)
    return [(score, idx, data["texts"][idx], data["metadatas"][idx]) for score, idx in scored[:top_k]]


def main():
    data = load_index_data()
    texts = data["texts"]
    metadatas = data["metadatas"]

    print("索引统计:")
    print(f"  文本块数: {len(texts)}")
    print(f"  元数据数: {len(metadatas)}")
    print(f"  模型: {data.get('model', 'unknown')}")
    print(f"  更新时间: {data.get('updated_at', 'unknown')}")

    assert len(texts) == len(metadatas), "texts 与 metadatas 数量不一致"

    for query in ("公文格式", "仿宋"):
        print(f"\n\n搜索'{query}'...")
        results = keyword_search(data, query)
        assert results, f"未找到关键词: {query}"

        for score, idx, text, meta in results:
            print(f"\n命中次数: {score}")
            print(f"来源: {meta.get('source', 'N/A')}")
            print(f"内容: {text[:300]}...")

    print("\n\n查找'公文格式.pdf'相关文本...")
    matched = [
        (i, text)
        for i, (text, meta) in enumerate(zip(texts, metadatas))
        if "公文格式" in meta.get("source", "")
    ]
    assert matched, "未找到公文格式.pdf 相关文本"

    for i, text in matched[:3]:
        print(f"\n文本块 {i}: {text[:500]}...")


if __name__ == "__main__":
    main()
