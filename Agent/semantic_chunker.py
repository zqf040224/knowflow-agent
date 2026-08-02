"""
语义文档切分器

组合两种策略：
- Plan B（结构切分）：按公文层级标题正则切分，保留文档结构
- Plan A（语义切分）：对超长段落用 embedding 相似度找语义边界

结构切分无需模型可独立使用；语义切分需要传入 embedding 模型。
"""

import re
import numpy as np
from typing import List, Optional

# 公文层级标题正则（匹配标题开头的换行符，split 时保留标题）
_HEADING_PATTERNS = [
    # 一、二、三、... 十、 (中文数字一级标题)
    r'\n(?=[一二三四五六七八九十]+[、，,])',
    # （一）（二）... (括号中文数字二级标题)
    r'\n(?=（[一二三四五六七八九十]+）)',
    # 1. 2. 3. 或 1、2、3、 (阿拉伯数字标题，前面需有换行)
    r'\n(?=\d+[\.\、])',
    # (1) (2) (3) (括号阿拉伯数字)
    r'\n(?=\(\d+\))',
    # 第一，第二... 第X章 第X节
    r'\n(?=第[一二三四五六七八九十\d]+[章节条，,])',
    # Markdown 标题 ## ...
    r'\n(?=#{1,6}\s)',
]

_HEADING_RE = re.compile('|'.join(_HEADING_PATTERNS))

# 句子切分（中英文标点）
_SENTENCE_RE = re.compile(r'([。！？；]|[.!?;]\s)', re.MULTILINE)


class StructuralChunker:
    """基于文档结构的切分器 — 按公文标题层级 + 自然段切分，无需模型"""

    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split(self, text: str) -> List[str]:
        """按结构切分文本，返回 chunk 列表"""
        if not text or not text.strip():
            return []

        # 1. 按标题切分
        sections = self._split_by_headings(text)

        # 2. 对每个段落：合适的保留，超长的继续切
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._split_long_section(section))

        # 3. 合并太小的 chunk
        chunks = self._merge_short(chunks)

        # 4. 确保非空文本至少返回一个 chunk
        if not chunks:
            text = text.strip()
            if len(text) >= 10:
                return [text]
        return chunks

    def _split_by_headings(self, text: str) -> List[str]:
        """按层级标题正则切分"""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [text]

        sections = []
        prev_end = 0
        for m in matches:
            if m.start() > prev_end:
                section = text[prev_end:m.start()]
                if section.strip():
                    sections.append(section)
            prev_end = m.start()

        if prev_end < len(text):
            section = text[prev_end:]
            if section.strip():
                sections.append(section)

        return sections if sections else [text]

    def _split_long_section(self, text: str) -> List[str]:
        """超长段落 → 按自然段 → 按句子逐级切分"""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if not paragraphs:
            return [text] if len(text) >= self.min_chunk_size else []

        chunks = []
        for para in paragraphs:
            if len(para) <= self.max_chunk_size:
                if len(para) >= self.min_chunk_size:
                    chunks.append(para)
            else:
                chunks.extend(self._split_by_sentences(para))
        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子边界切分，合并到目标大小"""
        parts = _SENTENCE_RE.split(text)
        sentences = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # 标点附加到前一句
            if _SENTENCE_RE.match(part) and sentences:
                sentences[-1] += part
            else:
                sentences.append(part)

        if not sentences:
            return [text] if len(text) >= self.min_chunk_size else []

        chunks = []
        current = ''
        for sent in sentences:
            if len(current) + len(sent) <= self.max_chunk_size:
                current += sent
            else:
                if len(current) >= self.min_chunk_size:
                    chunks.append(current)
                    current = sent
                else:
                    current += sent

        if len(current) >= self.min_chunk_size:
            chunks.append(current)
        elif chunks:
            chunks[-1] += current

        return chunks

    def _merge_short(self, chunks: List[str]) -> List[str]:
        """将过短的 chunk 合并到相邻 chunk，保留结构边界"""
        if not chunks:
            return chunks
        merged = []
        buffer = ''
        for chunk in chunks:
            if len(chunk) < self.min_chunk_size:
                # 过短的 chunk（如孤立标题），合并到缓冲
                buffer += ('\n' if buffer else '') + chunk
            else:
                if buffer:
                    # 将缓冲内容附加到当前 chunk
                    chunk = buffer + '\n' + chunk
                    buffer = ''
                if len(chunk) > self.max_chunk_size * 1.5:
                    # 合并后超长，按句子切分
                    merged.extend(self._split_long_section(chunk))
                else:
                    merged.append(chunk)
        if buffer:
            # 末尾残余合并到最后一段
            if merged and len(merged[-1]) + len(buffer) <= self.max_chunk_size:
                merged[-1] += '\n' + buffer
            elif buffer.strip():
                merged.append(buffer)
        return merged


class SemanticChunker(StructuralChunker):
    """
    语义切分器 = 结构切分 + embedding 相似度切分

    先用标题结构切分，对仍超长的段落用相邻句 embedding 相似度找语义边界（低谷处断开）。
    """

    def __init__(self, model, max_chunk_size: int = 800,
                 min_chunk_size: int = 50, similarity_percentile: float = 15):
        super().__init__(max_chunk_size, min_chunk_size)
        self.model = model
        self.similarity_percentile = similarity_percentile

    def split(self, text: str) -> List[str]:
        """结构切分 + 语义切分（对超长段落）"""
        if not text or not text.strip():
            return []

        sections = self._split_by_headings(text)
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._semantic_split(section))

        chunks = self._merge_short(chunks)

        if not chunks:
            text = text.strip()
            if len(text) >= 10:
                return [text]
        return chunks

    def _semantic_split(self, text: str) -> List[str]:
        """基于 embedding 相似度的语义切分"""
        # 先拆句子
        parts = _SENTENCE_RE.split(text)
        sentences = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if _SENTENCE_RE.match(part) and sentences:
                sentences[-1] += part
            else:
                sentences.append(part)

        if len(sentences) <= 1:
            return self._split_long_section(text)

        # 计算句子 embedding + 相邻相似度
        embeddings = self.model.encode(sentences)
        embeddings = np.array(embeddings).astype('float32')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        similarities = [float(np.dot(embeddings[i], embeddings[i + 1]))
                        for i in range(len(sentences) - 1)]

        # 相似度低谷 = 语义边界
        threshold = np.percentile(similarities, self.similarity_percentile)

        breakpoints = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))

        # 合并句子为 chunk
        chunks = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1]
            chunk_text = ''.join(sentences[start:end]).strip()

            if not chunk_text:
                continue

            if len(chunk_text) > self.max_chunk_size * 1.5:
                # 递归语义切分
                chunks.extend(self._semantic_split(chunk_text))
            elif len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:
                # 过短则合并到上一个
                chunks[-1] += '\n' + chunk_text
            else:
                chunks.append(chunk_text)

        return chunks if chunks else [text]
