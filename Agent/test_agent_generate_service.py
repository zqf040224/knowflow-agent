from types import SimpleNamespace

from agent_generate_service import AgentGenerateDependencies, AgentGenerateService


class FakeMemory:
    def __init__(self):
        self.profile = SimpleNamespace(
            preferred_font="仿宋",
            preferred_size="三号",
            writing_style="简洁正式",
            common_doc_types=[],
        )
        self.session_requests = []

    def get_or_create_session(self, user_id, session_id=None):
        self.session_requests.append((user_id, session_id))
        return session_id or "session_1"

    def get_user_profile(self, user_id):
        return self.profile


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, message, on_think=None, session_id=None):
        self.calls.append((message, session_id))
        if on_think:
            on_think("Planner", "🧭", "已制定计划")
        return {
            "document": "正文",
            "plan": {"document_type": "通知"},
            "confidence": 0.91,
            "revision_rounds": 1,
        }


def test_agent_generate_service_preserves_legacy_response_contract():
    memory = FakeMemory()
    runner = FakeRunner()
    factory_calls = []

    def orchestrator_factory(session_id, *, profile=None, user_info=None):
        factory_calls.append((session_id, profile, user_info))
        return runner

    service = AgentGenerateService(AgentGenerateDependencies(
        memory=memory,
        orchestrator_factory=orchestrator_factory,
    ))

    result = service.generate(
        {"message": "帮我写一份通知", "session_id": "session_x"},
        user_id="user_1",
        user_info=SimpleNamespace(username="tester"),
    )

    assert memory.session_requests == [("user_1", "session_x")]
    assert factory_calls[0][0] == "session_x"
    assert factory_calls[0][1] is memory.profile
    assert factory_calls[0][2].username == "tester"
    assert runner.calls == [("帮我写一份通知\n\n[用户偏好：仿宋 三号]", "session_x")]
    assert result == {
        "document": "正文",
        "plan": {"document_type": "通知"},
        "think_log": [{"agent": "Planner", "emoji": "🧭", "message": "已制定计划"}],
        "confidence": 0.91,
        "revision_rounds": 1,
    }
