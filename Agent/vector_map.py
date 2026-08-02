"""
Vector map projection helpers.

The knowledge-base embeddings are high-dimensional. For admin inspection we
project them to 3D with PCA so files and chunks can be explored visually.
"""

from typing import Any, Dict, List

import numpy as np


def build_vector_map(embeddings: np.ndarray, metadatas: List[Dict[str, Any]],
                     texts: List[str], indices: List[int]) -> Dict[str, Any]:
    """Build a frontend-friendly 3D vector map payload."""
    if embeddings.size == 0:
        return {
            "ok": False,
            "message": "知识库暂无可视化向量",
            "points": [],
            "file_count": 0,
            "point_count": 0,
        }

    coords, explained = project_embeddings_to_3d(embeddings)
    points = []
    files = set()

    for row_idx, source_idx in enumerate(indices):
        metadata = metadatas[source_idx] if source_idx < len(metadatas) else {}
        filename = metadata.get("filename") or str(metadata.get("source", "")).split("/")[-1] or "unknown"
        files.add(filename)
        x, y, z = coords[row_idx]
        points.append({
            "id": int(source_idx),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "filename": filename,
            "category": metadata.get("category", ""),
            "department": metadata.get("department", ""),
            "source_type": metadata.get("source_type", "document"),
            "doc_type": metadata.get("doc_type", ""),
            "sheet_name": metadata.get("sheet_name", ""),
            "row_start": metadata.get("row_start"),
            "row_end": metadata.get("row_end"),
            "chunk_index": metadata.get("chunk_index", -1),
            "total_chunks": metadata.get("total_chunks", -1),
            "preview": (texts[source_idx] if source_idx < len(texts) else "")[:180],
        })

    return {
        "ok": True,
        "points": points,
        "file_count": len(files),
        "point_count": len(points),
        "explained_variance": explained,
    }


def project_embeddings_to_3d(embeddings: np.ndarray):
    """Project embeddings to normalized 3D coordinates using PCA."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")

    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    if matrix.shape[0] == 1:
        coords = np.zeros((1, 3), dtype=np.float32)
        return coords, [1.0, 0.0, 0.0]

    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    dims = min(3, vt.shape[0])
    coords = matrix @ vt[:dims].T
    if dims < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - dims)), mode="constant")

    max_abs = float(np.max(np.abs(coords))) if coords.size else 0.0
    if max_abs > 0:
        coords = coords / max_abs

    variances = singular_values ** 2
    total = float(np.sum(variances)) or 1.0
    explained = [(float(v) / total) for v in variances[:3]]
    while len(explained) < 3:
        explained.append(0.0)

    return coords.astype(np.float32), explained
