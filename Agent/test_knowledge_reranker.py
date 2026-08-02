from agents.knowledge_agent import KnowledgeAgent, _ranking_cache_version


def test_optional_reranker_off_preserves_lightweight_order(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER", "off")
    agent = KnowledgeAgent.__new__(KnowledgeAgent)
    results = [{"filename": "a.docx"}, {"filename": "b.docx"}]

    assert agent._apply_optional_reranker(results, "query") == results


def test_optional_reranker_falls_back_when_backend_missing(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER", "local")
    agent = KnowledgeAgent.__new__(KnowledgeAgent)
    results = [{"filename": "a.docx"}, {"filename": "b.docx"}]

    assert agent._apply_optional_reranker(results, "query") == results


def test_ranking_cache_version_includes_reranker_config(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER", "api")
    monkeypatch.setenv("RAG_RERANK_TOP_N", "12")
    monkeypatch.setenv("RAG_CONTEXT_TOP_K", "6")

    version = _ranking_cache_version()

    assert "reranker=api" in version
    assert "rerank_top_n=12" in version
    assert "context_top_k=6" in version
