from datetime import datetime

from runtime_query_service import RuntimeQueryDependencies, RuntimeQueryService


class FakeUserInfo:
    def __init__(self):
        self.calls = 0

    def to_dict(self):
        self.calls += 1
        return {"user_id": "u1", "department": "财务部"}


class FakeKnowledgeAgent:
    def __init__(self):
        self.process_calls = []

    def get_health(self, user_info):
        return {"ok": True, "user": user_info}

    def process(self, payload):
        self.process_calls.append(payload)

        class Result:
            metadata = {"results": [{"filename": "a.docx"}]}

        return Result()


class FakeManifest:
    db_path = None

    def __init__(self):
        self.consistency_calls = []

    def consistency_report(self, knowledge_base, spreadsheet_db_path):
        self.consistency_calls.append((knowledge_base, spreadsheet_db_path))
        return {"consistent": True}


class FakeSpreadsheetStore:
    db_path = None

    def __init__(self):
        self.query_calls = []

    def query_rows(self, **kwargs):
        self.query_calls.append(kwargs)
        return [{"row": 1}, {"row": 2}]


class FakeIndex:
    def __init__(self, vectors):
        self.vectors = vectors
        self.ntotal = len(vectors)

    def reconstruct(self, idx):
        return self.vectors[idx]


class FakeKnowledgeBase:
    def __init__(self, *, vectors=None):
        self.index = FakeIndex(vectors or []) if vectors is not None else None
        self.texts = ["t1", "t2", "t3"][:len(vectors or [])]
        self.metadatas = [{"filename": f"f{i}.txt"} for i in range(len(vectors or []))]


def build_service(**overrides):
    deps = RuntimeQueryDependencies(
        knowledge_agent=overrides.get("knowledge_agent") or FakeKnowledgeAgent(),
        knowledge_base=overrides.get("knowledge_base") or FakeKnowledgeBase(),
        knowledge_manifest=overrides.get("knowledge_manifest") or FakeManifest(),
        spreadsheet_store=overrides.get("spreadsheet_store") or FakeSpreadsheetStore(),
        spreadsheet_db_path="sheets.sqlite",
        storage_health=lambda: {"backend": "local"},
        build_access_filter=lambda user_info: {"department": "财务部"},
        build_vector_map=lambda embeddings, metadatas, texts, indices: {
            "ok": True,
            "points": [{"idx": int(indices[0])}],
            "file_count": len({item["filename"] for item in metadatas}),
            "point_count": len(indices),
        },
        now_factory=lambda: datetime(2026, 6, 7, 9, 0, 0),
    )
    return RuntimeQueryService(deps)


def test_health_and_knowledge_health_payloads():
    manifest = FakeManifest()
    kb = FakeKnowledgeBase()
    service = build_service(knowledge_manifest=manifest, knowledge_base=kb)
    user_info = FakeUserInfo()

    health = service.health()
    knowledge = service.knowledge_health(user_info)

    assert health == {
        "status": "ok",
        "timestamp": "2026-06-07T09:00:00",
        "storage": {"backend": "local"},
    }
    assert knowledge["success"] is True
    assert knowledge["health"]["user"] == {"user_id": "u1", "department": "财务部"}
    assert knowledge["ingestion_consistency"] == {"consistent": True}
    assert manifest.consistency_calls == [(kb, "sheets.sqlite")]


def test_spreadsheet_query_applies_access_filter_and_clamps_limit():
    store = FakeSpreadsheetStore()
    service = build_service(spreadsheet_store=store)

    payload = service.spreadsheet_query({"keyword": "预算", "limit": 999}, user_info=FakeUserInfo())

    assert payload == {"success": True, "rows": [{"row": 1}, {"row": 2}], "count": 2}
    assert store.query_calls[0]["keyword"] == "预算"
    assert store.query_calls[0]["access_filter"] == {"department": "财务部"}
    assert store.query_calls[0]["limit"] == 200


def test_search_passes_user_info_to_knowledge_agent():
    agent = FakeKnowledgeAgent()
    service = build_service(knowledge_agent=agent)

    payload = service.search({"query": "制度"}, user_info=FakeUserInfo())

    assert payload == [{"filename": "a.docx"}]
    assert agent.process_calls == [{
        "user_request": "制度",
        "knowledge_queries": ["制度"],
        "user_info": {"user_id": "u1", "department": "财务部"},
    }]


def test_vector_map_handles_empty_and_successful_index():
    empty_payload, empty_status = build_service().vector_map(limit=1200)
    kb = FakeKnowledgeBase(vectors=[[1.0, 0.0], [0.0, 1.0]])
    ok_payload, ok_status = build_service(knowledge_base=kb).vector_map(limit=1)

    assert empty_status == 200
    assert empty_payload["map"]["ok"] is False
    assert empty_payload["map"]["message"] == "知识库暂无可视化向量"
    assert ok_status == 200
    assert ok_payload["map"]["ok"] is True
    assert ok_payload["map"]["total_vectors"] == 2
    assert ok_payload["map"]["sampled"] == 2
