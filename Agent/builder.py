"""
知识库构建工具 v3.0 - 使用本地 embedding 模型 (sentence-transformers)

使用方法：
python builder.py --input ./资料库 --output ./knowledge_base

注意：首次运行会下载模型（约 471MB），需要网络连接
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from app_config import DEPARTMENT_DIRS as APP_DEPARTMENT_DIRS
from embedding_config import (
    MODEL_NAME,
    DIM,
    HF_ENDPOINT,
    ACCESS_PUBLIC,
    ACCESS_INTERNAL,
    ACCESS_RESTRICTED,
    ACCESS_ADMIN,
    resolve_embedding_model_path,
)

os.environ['HF_ENDPOINT'] = HF_ENDPOINT

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# 语义文档切分
from semantic_chunker import SemanticChunker

# 文档解析
from document_parser import (
    parse_document,
    parse_document_with_format,
    format_to_style_spec,
    FormatFingerprint
)


class KnowledgeBaseBuilder:
    """知识库构建器 v3.0 - 本地 embedding 模型"""

    def __init__(self, input_dir: str, output_dir: str,
                 model_name: str = MODEL_NAME):
        """
        初始化构建器

        Args:
            input_dir: 输入资料目录
            output_dir: 输出知识库目录
            model_name: sentence-transformers 模型名称
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # 加载本地 embedding 模型
        print("🔄 加载本地 embedding 模型...")
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_path = resolve_embedding_model_path() if self.model_name == MODEL_NAME else self.model_name
        self.model = SentenceTransformer(model_path)
        self.model.to(device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ 模型加载完成: {self.model_name}")
        print(f"✓ 使用设备: {device}, 维度: {self.embedding_dim}")

        # 初始化语义切分器（结构切分 + embedding 相似度切分）
        self.text_splitter = SemanticChunker(
            model=self.model,
            max_chunk_size=800,
            min_chunk_size=50
        )

        # 存储文本和元数据
        self.all_texts = []
        self.all_metadata = []

        # 统计信息
        self.stats = {
            "total_files": 0,
            "total_chunks": 0,
            "file_types": {},
            "categories": {}
        }

    def scan_documents(self) -> List[Path]:
        """扫描输入目录下的所有文档"""
        print(f"📂 扫描目录：{self.input_dir}")

        file_extensions = [".pdf", ".docx", ".doc", ".txt", ".md"]
        files = []

        for ext in file_extensions:
            files.extend(self.input_dir.rglob(f"*{ext}"))

        self.stats["total_files"] = len(files)
        print(f"✅ 找到 {len(files)} 个文档")

        return files

    def parse_and_chunk(self, file_path: Path, with_format: bool = True) -> List[Dict]:
        """
        解析文档并切分为片段

        Args:
            file_path: 文件路径
            with_format: 是否提取格式信息

        Returns:
            List[Dict]: 每个元素包含 "content" 和 "metadata"
        """
        print(f"  📄 解析：{file_path.name}")

        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "category": self._detect_category(file_path),
            "file_type": file_path.suffix,
            "access_level": self._detect_access_level(file_path),
            "department": self._detect_department(file_path),
        }

        results = []

        # 根据文件类型选择解析方式
        if with_format and file_path.suffix.lower() == ".docx":
            # Word 文件使用带格式解析
            formatted_content = parse_document_with_format(file_path)

            if not formatted_content:
                print(f"    ⚠️ 解析失败，跳过")
                return []

            # 统计
            file_type = file_path.suffix
            self.stats["file_types"][file_type] = self.stats["file_types"].get(file_type, 0) + 1
            category = metadata["category"]
            self.stats["categories"][category] = self.stats["categories"].get(category, 0) + 1

            # 将带格式的内容转换为标准格式，超长段落做二次切分
            total_items = len(formatted_content)
            result_idx = 0
            for idx, item in enumerate(formatted_content):
                content = item.get('content', '').strip()
                if not content:
                    continue

                # 组合文字内容 + 格式描述
                format_desc = format_to_style_spec(item["format"])
                item_format = item["format"]

                if len(content) > 800:
                    # 超长段落按句子切分，保留格式信息
                    sub_chunks = self.text_splitter.split(content)
                    for sub in sub_chunks:
                        sub = sub.strip()
                        if len(sub) < 10:
                            continue
                        enhanced_content = f"{sub}\n[格式：{format_desc}]"
                        item_metadata = metadata.copy()
                        item_metadata["format"] = item_format
                        item_metadata["chunk_index"] = result_idx
                        result_idx += 1
                        results.append({
                            "content": enhanced_content,
                            "metadata": item_metadata
                        })
                else:
                    enhanced_content = f"{content}\n[格式：{format_desc}]"
                    item_metadata = metadata.copy()
                    item_metadata["format"] = item_format
                    item_metadata["chunk_index"] = result_idx
                    result_idx += 1
                    results.append({
                        "content": enhanced_content,
                        "metadata": item_metadata
                    })

            for item in results:
                item["metadata"]["total_chunks"] = len(results)

            print(f"    ✓ 提取 {len(formatted_content)} 个片段（含格式）")
            return results

        else:
            # 其他文件使用纯文本解析
            content = parse_document(file_path)
            if not content:
                print(f"    ⚠️ 解析失败，跳过")
                return []

            # 统计
            file_type = file_path.suffix
            self.stats["file_types"][file_type] = self.stats["file_types"].get(file_type, 0) + 1
            category = metadata["category"]
            self.stats["categories"][category] = self.stats["categories"].get(category, 0) + 1

            # 切分文档（结构切分 + 语义切分）
            chunks = self.text_splitter.split(content)
            total_chunks = len(chunks)

            for idx, chunk in enumerate(chunks):
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = idx
                chunk_meta["total_chunks"] = total_chunks
                results.append({
                    "content": chunk,
                    "metadata": chunk_meta
                })

            print(f"    ✓ 切分为 {len(chunks)} 个片段")
            return results

    def _detect_category(self, file_path: Path) -> str:
        """根据文件路径自动检测文档分类"""
        parts = file_path.parts

        try:
            idx = parts.index(self.input_dir.name)
            if idx + 1 < len(parts):
                return parts[idx + 1]
        except ValueError:
            pass

        return "其他"

    # 目录 → 访问级别映射
    # 部门目录下的文件仅本部门可见；公共资料所有人可见
    DEPARTMENT_DIRS = set(APP_DEPARTMENT_DIRS)

    ACCESS_RULES = {d: ACCESS_RESTRICTED for d in APP_DEPARTMENT_DIRS}
    ACCESS_RULES["公共资料"] = ACCESS_PUBLIC

    def _detect_department(self, file_path: Path) -> str:
        """检测文件所属部门（部门目录名或空字符串表示公共资料）"""
        parts = file_path.parts
        try:
            idx = parts.index(self.input_dir.name)
            subdirs = parts[idx + 1:-1]
            for subdir in subdirs:
                if subdir in self.DEPARTMENT_DIRS:
                    return subdir
        except ValueError:
            pass
        return ""  # 公共资料无部门归属

    def _detect_access_level(self, file_path: Path) -> str:
        """根据文件路径检测文档访问级别"""
        parts = file_path.parts
        try:
            idx = parts.index(self.input_dir.name)
            subdirs = parts[idx + 1:-1]  # 输入目录到文件名之间的子目录
            for subdir in subdirs:
                if subdir in self.ACCESS_RULES:
                    return self.ACCESS_RULES[subdir]
        except ValueError:
            pass
        return ACCESS_PUBLIC

    def build_vector_index(self):
        """构建向量索引（使用本地模型）"""
        print(f"\n🔮 开始构建向量索引...")
        print(f"  总片段数：{len(self.all_texts)}")
        print(f"  Embedding 维度：{self.embedding_dim}")

        # 使用本地模型生成 embedding
        print(f"  正在生成向量（可能需要几分钟）...")
        embeddings = self.model.encode(
            self.all_texts,
            show_progress_bar=True,
            batch_size=32
        )
        embeddings = np.array(embeddings).astype('float32')

        # L2 归一化（使得内积 = 余弦相似度）
        faiss.normalize_L2(embeddings)

        # 创建 FAISS 索引（内积 = 余弦相似度）
        self.vector_index = faiss.IndexFlatIP(self.embedding_dim)
        self.vector_index.add(embeddings)

        print(f"✅ 向量索引构建完成")
        print(f"  索引大小：{self.vector_index.ntotal} 条")

    def save_knowledge_base(self):
        """保存知识库到文件（兼容 knowledge_qa_fast.py 格式）"""
        import pickle

        print(f"\n💾 保存知识库...")

        # 保存 FAISS 索引（使用与 knowledge_qa_fast.py 相同的文件名）
        faiss_path = self.output_dir / "faiss_local.index"
        faiss.write_index(self.vector_index, str(faiss_path))
        print(f"  ✓ FAISS 索引：{faiss_path}")

        # 保存文本和元数据（使用与 knowledge_qa_fast.py 相同的格式）
        pkl_path = self.output_dir / "faiss_local_index.pkl"
        index_data = {
            "texts": self.all_texts,
            "metadatas": self.all_metadata,
            "dim": self.embedding_dim,
            "model": self.model_name
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"  ✓ 文本映射：{pkl_path}")

        # 保存配置信息
        config = {
            "version": "3.0",
            "created_at": datetime.now().isoformat(),
            "stats": self.stats,
            "embedding_model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "format_support": True,
            "note": "使用本地 sentence-transformers 模型，无 API 依赖"
        }
        config_path = self.output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 配置文件：{config_path}")

        # 输出统计信息
        print(f"\n📊 构建统计:")
        print(f"  总文件数：{self.stats['total_files']}")
        print(f"  总片段数：{self.stats['total_chunks']}")
        print(f"  Embedding 维度：{self.embedding_dim}")
        print(f"  分类统计:")
        for category, count in self.stats['categories'].items():
            print(f"    - {category}: {count} 个文件")
        print(f"  文件类型:")
        for file_type, count in self.stats['file_types'].items():
            print(f"    - {file_type}: {count} 个")

    def build(self, with_format: bool = True):
        """执行完整的构建流程"""
        print("=" * 60)
        print("🚀 知识库构建工具 v3.0（本地 embedding 模型）")
        print("=" * 60)

        # 1. 扫描文档
        files = self.scan_documents()

        # 2. 解析并切分
        for file_path in files:
            items = self.parse_and_chunk(file_path, with_format=with_format)
            for item in items:
                self.all_texts.append(item["content"])
                self.all_metadata.append(item["metadata"])
            self.stats["total_chunks"] += len(items)

        if not self.all_texts:
            print("❌ 未找到任何文档，构建失败")
            return False

        # 3. 构建向量索引
        self.build_vector_index()

        # 4. 保存知识库
        self.save_knowledge_base()

        print("\n" + "=" * 60)
        print("✅ 知识库构建完成！")
        print("=" * 60)

        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识库构建工具 v3.0 - 本地 embedding 模型")
    parser.add_argument("--input", required=True, help="输入资料目录")
    parser.add_argument("--output", required=True, help="输出知识库目录")
    parser.add_argument("--no-format", action="store_true",
                        help="禁用格式指纹提取（纯文本模式）")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"sentence-transformers 模型名称（默认：{MODEL_NAME}）")

    args = parser.parse_args()

    # 创建构建器
    builder = KnowledgeBaseBuilder(args.input, args.output, model_name=args.model)

    # 执行构建
    success = builder.build(with_format=not args.no_format)

    if success:
        print("\n💡 使用说明:")
        print("  - 输出文件：")
        print("    * faiss_local.index - FAISS 向量索引")
        print("    * faiss_local_index.pkl - 文本和元数据映射")
        print("    * config.json - 配置信息")
        print("\n  - 检索示例：")
        print("    python -c \"from knowledge_base import quick_search; print(quick_search('公文格式'))\"")
        print("\n  - 特点：")
        print("    ✓ 使用本地 BCE embedding 模型，无需 API Key")
        print("    ✓ 支持语义检索，中文效果优于 TF-IDF")


if __name__ == "__main__":
    main()
