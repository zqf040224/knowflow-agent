#!/usr/bin/env python3
"""
知识库向量索引构建脚本 - 使用国内 HF 镜像
M系列Mac自动使用Metal GPU加速
"""

import os
import pickle
import time
import torch

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 项目根目录（scripts 的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, '知识库')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'knowledge_base')
INDEX_PATH = os.path.join(OUTPUT_DIR, 'faiss_local_index.pkl')
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'faiss_local.index')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
DIM = 384

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def load_docx_text(path):
    import docx
    doc = docx.Document(path)
    texts = []
    for para in doc.paragraphs:
        if para.text.strip():
            texts.append(para.text.strip())
    return '\n'.join(texts)

def load_pdf_text(path):
    import PyPDF2
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return '\n'.join(text_parts)
    except Exception as e:
        print(f"PDF读取错误: {e}")
        return ""

def load_documents():
    all_docs = []
    files = os.listdir(DOCS_DIR)
    for fname in files:
        path = os.path.join(DOCS_DIR, fname)
        print(f"加载: {fname}")
        try:
            if fname.endswith('.docx'):
                content = load_docx_text(path)
            elif fname.endswith('.pdf'):
                content = load_pdf_text(path)
            else:
                continue
            if content:
                all_docs.append({
                    'content': content,
                    'source': fname,
                    'type': 'docx' if fname.endswith('.docx') else 'pdf'
                })
        except Exception as e:
            print(f"  错误: {e}")
    return all_docs

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_faiss_index(documents, model, device):
    all_chunks = []
    all_metadatas = []

    for doc in documents:
        chunks = chunk_text(doc['content'])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                'source': doc['source'],
                'type': doc['type']
            })

    print(f"\n总文本块数: {len(all_chunks)}")

    print(f"\n使用设备: {device}")
    print("开始生成embeddings...")

    embeddings = model.encode(all_chunks, show_progress_bar=True, device=device)
    embeddings = np.array(embeddings).astype('float32')

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)

    print(f"FAISS索引构建完成，包含 {index.ntotal} 个向量")

    return {
        'index': index,
        'texts': all_chunks,
        'metadatas': all_metadatas
    }

def save_index(index_data, index_path, faiss_path):
    with open(index_path, 'wb') as f:
        pickle.dump({
            'texts': index_data['texts'],
            'metadatas': index_data['metadatas']
        }, f)
    faiss.write_index(index_data['index'], faiss_path)
    print(f"\n索引已保存:")
    print(f"  - 元数据: {index_path}")
    print(f"  - FAISS索引: {faiss_path}")

def main():
    print("=" * 60)
    print("构建本地向量知识库 (HF镜像 + sentence-transformers + FAISS)")
    print("=" * 60)

    device = get_device()
    print(f"\n检测到设备: {device}")
    print(f"使用HF镜像: https://hf-mirror.com")

    print(f"\n加载模型: {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    model.to(device)
    print(f"模型加载完成: {time.time()-t0:.2f}秒")

    print("\n步骤1: 加载文档...")
    documents = load_documents()
    print(f"\n共加载 {len(documents)} 个文档")

    print("\n步骤2: 构建FAISS向量索引...")
    t0 = time.time()
    index_data = build_faiss_index(documents, model, device)
    print(f"索引构建耗时: {time.time()-t0:.2f}秒")

    print("\n步骤3: 保存索引...")
    save_index(index_data, INDEX_PATH, FAISS_INDEX_PATH)

    print("\n" + "=" * 60)
    print("知识库构建完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()