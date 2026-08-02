"""Runtime health, search, and query service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass
class RuntimeQueryDependencies:
    knowledge_agent: Any
    knowledge_base: Any
    knowledge_manifest: Any
    spreadsheet_store: Any
    spreadsheet_db_path: Any
    storage_health: Callable[[], dict]
    build_access_filter: Callable[[Any], dict]
    build_vector_map: Callable[..., dict]
    now_factory: Callable[[], datetime]


class RuntimeQueryService:
    def __init__(self, deps: RuntimeQueryDependencies):
        self.deps = deps

    def health(self) -> dict:
        return {
            "status": "ok",
            "timestamp": self.deps.now_factory().isoformat(),
            "storage": self.deps.storage_health(),
        }

    def knowledge_health(self, user_info: Any) -> dict:
        consistency = self.deps.knowledge_manifest.consistency_report(
            self.deps.knowledge_base,
            self.deps.spreadsheet_db_path,
        )
        return {
            "success": True,
            "health": self.deps.knowledge_agent.get_health(user_info.to_dict()),
            "ingestion_consistency": consistency,
            "storage": self.deps.storage_health(),
        }

    def storage_health_payload(self) -> dict:
        return {
            "success": True,
            "storage": self.deps.storage_health(),
        }

    def vector_map(self, *, limit: int = 1200) -> tuple[dict, int]:
        try:
            import numpy as np

            limit = min(max(int(limit or 1200), 10), 3000)
            kb = self.deps.knowledge_base
            if kb.index is None or not kb.texts:
                return {
                    "success": True,
                    "map": {
                        "ok": False,
                        "message": "知识库暂无可视化向量",
                        "points": [],
                        "file_count": 0,
                        "point_count": 0,
                    },
                }, 200

            total = min(kb.index.ntotal, len(kb.texts), len(kb.metadatas))
            if total <= 0:
                return {
                    "success": True,
                    "map": {
                        "ok": False,
                        "message": "知识库索引为空",
                        "points": [],
                        "file_count": 0,
                        "point_count": 0,
                    },
                }, 200

            indices = np.linspace(0, total - 1, limit, dtype=int).tolist() if total > limit else list(range(total))
            embeddings = np.asarray([kb.index.reconstruct(int(idx)) for idx in indices], dtype=np.float32)
            payload = self.deps.build_vector_map(embeddings, kb.metadatas, kb.texts, indices)
            payload["total_vectors"] = int(total)
            payload["sampled"] = int(len(indices))
            return {"success": True, "map": payload}, 200
        except Exception as exc:
            return {
                "success": False,
                "message": f"向量地图生成失败: {str(exc)[:160]}",
            }, 500

    def spreadsheet_query(self, data: dict, *, user_info: Any) -> dict:
        request_data = data or {}
        rows = self.deps.spreadsheet_store.query_rows(
            keyword=request_data.get("keyword", ""),
            content_hash=request_data.get("content_hash") or None,
            filename=request_data.get("filename") or None,
            sheet_name=request_data.get("sheet_name") or None,
            column_name=request_data.get("column_name") or None,
            cell_value=request_data.get("cell_value") or None,
            access_filter=self.deps.build_access_filter(user_info),
            limit=min(max(int(request_data.get("limit", 50)), 1), 200),
        )
        return {"success": True, "rows": rows, "count": len(rows)}

    def search(self, data: dict, *, user_info: Any) -> list:
        request_data = data or {}
        query = request_data.get("query", "")
        result = self.deps.knowledge_agent.process({
            "user_request": query,
            "knowledge_queries": [query],
            "user_info": user_info.to_dict(),
        })
        return result.metadata.get("results", [])
