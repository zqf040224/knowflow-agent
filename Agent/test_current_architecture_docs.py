from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_current_architecture_documents_production_chat_mainline():
    text = (ROOT / "docs" / "current_architecture.md").read_text(encoding="utf-8")

    assert "POST /api/chat" in text
    assert "ChatGraphRuntime" in text
    assert "IntentRouter" in text
    assert "RagQaStreamService" in text
    assert "DocumentDraftStreamService" in text
    assert "LightweightChatStreamService" in text
    assert "IntelligentRouter" in text
    assert "不再作为新功能接入点" in text
    assert "AppContext` 位于 `app_context.py" in text
    assert "app_dependencies.py` 只负责启动时装配" in text


def test_legacy_router_docs_are_marked_non_production():
    router_doc = (ROOT / "ROUTER_ARCHITECTURE.md").read_text(encoding="utf-8")
    using_doc = (ROOT / "using.md").read_text(encoding="utf-8")
    router_py = (ROOT / "intelligent_router.py").read_text(encoding="utf-8")

    for text in (router_doc, using_doc, router_py):
        assert "历史" in text
        assert "ChatGraphRuntime" in text
        assert "IntentRouter" in text

    assert "不是当前生产聊天主链路" in router_doc
    assert "不代表当前生产聊天主链路" in using_doc
