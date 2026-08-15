from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agents.document_graph_runner import DocumentGraphRunner
from agents.orchestrator import AgentOrchestrator
from chat_architecture import RouteResult
from chat_draft import DocumentDraftDependencies, DocumentDraftStreamService
from chat_events import sse
from chat_runtime import ChatGraphRuntime, ChatRunContext, ChatRuntimeDependencies
from graph_persistence import GraphPersistenceConfig, create_graph_persistence
from memory_v2 import TeamMemory
from task_planner import TaskPlan, TaskStep
from test_chat_runtime import FakeMemory, FakeUploadManager, parse_sse
from test_document_graph_runner import FakeOrchestrator
from tool_runtime import ChatTool, ToolOrchestrator, ToolRegistry


def _runtime(persistence, calls, *, run_id="run-sqlite-recovery"):
    class Planner:
        def plan(self, **_kwargs):
            calls["plan"] += 1
            return TaskPlan(
                task_type="knowledge_qa",
                steps=[TaskStep(tool="knowledge_qa", reason="sqlite recovery")],
                route=RouteResult("knowledge_qa", 1.0, "sqlite recovery"),
            )

    def tool_stream(*_args, **_kwargs):
        calls["tool"] += 1
        yield sse({"type": "start"})
        yield sse({"type": "content", "data": "ok"})
        yield sse({
            "type": "done",
            "intent": "knowledge_qa",
            "answer": "ok",
        })

    registry = ToolRegistry()
    registry.register(ChatTool(
        name="knowledge_qa",
        description="test",
        risk_level="low",
        input_schema={"message": "string"},
        stream=tool_stream,
    ))
    return ChatGraphRuntime(ChatRuntimeDependencies(
        memory=FakeMemory(),
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda *_args: "",
        lightweight_stream=lambda *_args: iter(()),
        document_format_stream=lambda *_args: iter(()),
        document_draft_stream=lambda *_args: iter(()),
        rag_qa_stream=lambda *_args: iter(()),
        task_planner=Planner(),
        tool_orchestrator=ToolOrchestrator(registry),
        checkpointer=persistence.checkpointer,
        persistence=persistence,
        run_id_factory=lambda: run_id,
    ), runtime_mode="graph")


def _draft_retry_runtime(persistence, calls, *, run_id="run-sqlite-draft-retry"):
    class Planner:
        def plan(self, **_kwargs):
            calls["plan"] += 1
            return TaskPlan(
                task_type="document_drafting",
                steps=[TaskStep(tool="draft_document", reason="retry review")],
                route=RouteResult("doc_drafting", 1.0, "retry review"),
            )

    def tool_stream(_message, session_id, *_args, **_kwargs):
        calls["tool"] += 1
        if calls["tool"] == 1:
            yield sse({"type": "error", "message": "review unavailable"})
            return
        yield sse({"type": "content", "data": "recovered document"})
        yield sse({
            "type": "done",
            "intent": "doc_drafting",
            "answer": "recovered document",
            "document": "recovered document",
            "session_id": session_id,
            "plan": {},
            "actions": [],
            "source_filenames": [],
            "source_details": [],
        })

    registry = ToolRegistry()
    registry.register(ChatTool(
        name="draft_document",
        description="test",
        risk_level="low",
        input_schema={"message": "string"},
        stream=tool_stream,
    ))
    return ChatGraphRuntime(ChatRuntimeDependencies(
        memory=FakeMemory(),
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda *_args: "",
        lightweight_stream=lambda *_args: iter(()),
        document_format_stream=lambda *_args: iter(()),
        document_draft_stream=tool_stream,
        rag_qa_stream=lambda *_args: iter(()),
        task_planner=Planner(),
        tool_orchestrator=ToolOrchestrator(registry),
        checkpointer=persistence.checkpointer,
        persistence=persistence,
        run_id_factory=lambda: run_id,
    ), runtime_mode="graph")


def test_sqlite_unstarted_http_reservation_retries_after_process_restart(tmp_path):
    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / "unstarted-http-checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "unstarted-http-runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )
    payload = {
        "message": "hello",
        "request_id": "request-unstarted-http-restart",
        "session_id": "session-1",
    }
    calls = {"plan": 0, "tool": 0}

    first_persistence = create_graph_persistence(config)
    try:
        first_runtime = _runtime(
            first_persistence,
            calls,
            run_id="run-unstarted-http",
        )
        stream = first_runtime.stream_http(
            payload,
            user_id="u1",
            user_info=None,
        )
        stream.close()

        abandoned = first_persistence.get_owned_run(
            "run-unstarted-http",
            user_id="u1",
        )
        assert abandoned.status == "failed"
        assert abandoned.error == first_runtime.UNSTARTED_HTTP_ERROR
        assert calls == {"plan": 0, "tool": 0}
    finally:
        first_persistence.close()

    second_persistence = create_graph_persistence(config)
    try:
        second_runtime = _runtime(
            second_persistence,
            calls,
            run_id="new-worker-generated-run",
        )
        events = parse_sse(second_runtime.stream_http(
            payload,
            user_id="u1",
            user_info=None,
        ))

        recovered = second_persistence.get_owned_run(
            "run-unstarted-http",
            user_id="u1",
        )
        assert events[-1]["type"] == "done"
        assert events[-1]["run_id"] == "run-unstarted-http"
        assert recovered.status == "succeeded"
        assert calls == {"plan": 1, "tool": 1}
    finally:
        second_persistence.close()


def _native_draft_runtime(
    persistence,
    calls,
    *,
    run_id="run-sqlite-native-draft",
    workflow_version="chat-v3",
    include_final_tool=False,
    runtime_mode="graph",
    business_memory=None,
    commit_fault_stage="",
):
    def fail_commit_once(stage):
        if commit_fault_stage != stage or calls.get("commit_fault_injected"):
            return
        calls["commit_fault_injected"] = stage
        raise RuntimeError(f"simulated {stage} failure")

    class Planner:
        def plan(self, **_kwargs):
            calls["plan"] += 1
            steps = [TaskStep(tool="draft_document", reason="native recovery")]
            if include_final_tool:
                steps.append(TaskStep(tool="identity_help", reason="final response"))
            return TaskPlan(
                task_type="document_drafting",
                steps=steps,
                route=RouteResult("doc_drafting", 1.0, "native recovery"),
            )

    class DraftMemory(FakeMemory):
        def get_user_profile(self, _user_id):
            return None

    class NativeOrchestrator(FakeOrchestrator):
        def __init__(self, session_id):
            super().__init__(reflect=True)
            self.session_id = session_id
            self.memory = business_memory
            self._current_effect_run_id = ""

        def _prepare_document_run(self, user_request, session_id=None, *, run_id=""):
            self.think_log = []
            self.session_id = session_id or self.session_id
            return SimpleNamespace(
                user_request=user_request,
                request_with_context=user_request,
                previous_context="",
                session_id=self.session_id,
                run_id=run_id,
            )

        def _think_handler(self, on_think=None):
            def handler(agent, emoji, message):
                self.think_log.append({
                    "agent": agent,
                    "emoji": emoji,
                    "message": message,
                })
                if on_think:
                    on_think(agent, emoji, message)

            return handler

        def _step_context_plan(self, *args, **kwargs):
            calls["context_plan"] += 1
            return super()._step_context_plan(*args, **kwargs)

        def _step_knowledge(self, *args, **kwargs):
            calls["retrieval"] += 1
            return super()._step_knowledge(*args, **kwargs)

        def _step_write(self, *args, **kwargs):
            calls["write"] += 1
            self.fail_write = bool(calls.get("fail_write", False))
            return super()._step_write(*args, **kwargs)

        def _step_review(self, *args, **kwargs):
            calls["review"] += 1
            self.fail_review = bool(calls["fail_review"])
            return super()._step_review(*args, **kwargs)

        def _step_reflection(self, *args, **kwargs):
            calls["reflection"] += 1
            return super()._step_reflection(*args, **kwargs)

        def _build_document_run_result(self, ctx, document_content, user_request, **extra):
            if business_memory is None:
                return super()._build_document_run_result(
                    ctx,
                    document_content,
                    user_request,
                    **extra,
                )
            return AgentOrchestrator._build_document_run_result(
                self,
                ctx,
                document_content,
                user_request,
                **extra,
            )

        _merged_key_points = AgentOrchestrator._merged_key_points
        _source_filenames = AgentOrchestrator._source_filenames
        _source_details = AgentOrchestrator._source_details
        _context_snapshot = AgentOrchestrator._context_snapshot

        def run_stream(self, user_request, on_think=None, session_id=None, *, run_id=""):
            prepared = self._prepare_document_run(
                user_request,
                session_id=session_id,
                run_id=run_id,
            )
            yield from DocumentGraphRunner(
                self,
                checkpointer=InMemorySaver(),
            ).stream(
                prepared,
                think_handler=self._think_handler(on_think),
                thread_id=run_id,
                user_request=user_request,
            )

    memory = business_memory or DraftMemory()

    billing_effects = calls.setdefault("billing_effects", set())

    def record_agent_run_token_usage(*_args, **kwargs):
        calls["success_usage"] += 1
        billing_key = f"{kwargs.get('run_id', '')}:document_token_usage"
        billing_effects.add(billing_key)
        fail_commit_once("billing")

    class FaultInjectingDraftService(DocumentDraftStreamService):
        def build_public_done(self, *args, **kwargs):
            fail_commit_once("build_public_done")
            return super().build_public_done(*args, **kwargs)

        def _update_common_doc_types(self, profile, user_id, stored_user_message):
            result = super()._update_common_doc_types(
                profile,
                user_id,
                stored_user_message,
            )
            fail_commit_once("profile")
            return result

    service = FaultInjectingDraftService(DocumentDraftDependencies(
        memory=memory,
        orchestrator_factory=lambda session_id, **_kwargs: NativeOrchestrator(session_id),
        resolve_export_template=lambda *_args: "official_document",
        record_agent_run_token_usage=record_agent_run_token_usage,
        record_token_usage=lambda **_kwargs: calls.__setitem__(
            "failure_usage", calls["failure_usage"] + 1
        ),
    ))
    registry = ToolRegistry()
    registry.register(ChatTool(
        name="draft_document",
        description="native test",
        risk_level="low",
        input_schema={"message": "string"},
        stream=service.stream,
    ))
    if include_final_tool:
        def final_stream(_message, session_id, *_args, **_kwargs):
            yield sse({"type": "answer_start", "message": "开始输出最终回答"})
            yield sse({"type": "answer_delta", "data": "最终回答"})
            yield sse({"type": "content", "data": "最终回答"})
            yield sse({"type": "answer_done", "answer": "最终回答"})
            yield sse({
                "type": "done",
                "intent": "identity_help",
                "answer": "最终回答",
                "document": "",
                "session_id": session_id,
                "plan": {},
                "actions": [],
                "source_filenames": [],
                "source_details": [],
            })

        registry.register(ChatTool(
            name="identity_help",
            description="deterministic final response",
            risk_level="low",
            input_schema={"message": "string"},
            stream=final_stream,
        ))
    runtime = ChatGraphRuntime(ChatRuntimeDependencies(
        memory=memory,
        upload_manager=FakeUploadManager(),
        reimbursement_detector=lambda *_args: "",
        lightweight_stream=lambda *_args: iter(()),
        document_format_stream=lambda *_args: iter(()),
        document_draft_stream=service.stream,
        rag_qa_stream=lambda *_args: iter(()),
        task_planner=Planner(),
        tool_orchestrator=ToolOrchestrator(registry),
        checkpointer=persistence.checkpointer,
        persistence=persistence,
        run_id_factory=lambda: run_id,
        workflow_version=workflow_version,
    ), runtime_mode=runtime_mode)
    if commit_fault_stage == "effect":
        complete_effect = persistence.complete_effect

        def complete_effect_with_fault(*args, **kwargs):
            node = args[1] if len(args) > 1 else kwargs.get("node")
            if node == "tool_draft_document":
                fail_commit_once("effect")
            return complete_effect(*args, **kwargs)

        persistence.complete_effect = complete_effect_with_fault
    return runtime


def test_native_nonfinal_document_step_matches_planner_sse_byte_for_byte(tmp_path):
    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / "nonfinal-checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "nonfinal-runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )

    def new_calls():
        return {
            "plan": 0,
            "context_plan": 0,
            "retrieval": 0,
            "write": 0,
            "review": 0,
            "reflection": 0,
            "fail_write": False,
            "fail_review": False,
            "failure_usage": 0,
            "success_usage": 0,
        }

    persistence = create_graph_persistence(config)
    try:
        graph_runtime = _native_draft_runtime(
            persistence,
            new_calls(),
            run_id="run-native-nonfinal-parity",
            include_final_tool=True,
            runtime_mode="graph",
        )
        planner_runtime = _native_draft_runtime(
            persistence,
            new_calls(),
            run_id="run-native-nonfinal-parity",
            include_final_tool=True,
            runtime_mode="planner",
        )
        payload = {
            "message": "先起草通知，再回答身份问题",
            "session_id": "same-session",
            "request_id": "request-native-nonfinal-parity",
        }

        graph_output = list(graph_runtime.stream(
            payload,
            user_id="u1",
            user_info=None,
        ))
        planner_output = list(planner_runtime.stream(
            payload,
            user_id="u1",
            user_info=None,
        ))

        assert graph_output == planner_output
    finally:
        persistence.close()


@pytest.mark.parametrize("failed_step", ["write", "review"])
def test_native_document_failure_callback_fails_effect_without_parent_checkpoint(
    tmp_path,
    failed_step,
):
    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / "callback-checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "callback-runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )
    calls = {
        "plan": 0,
        "context_plan": 0,
        "retrieval": 0,
        "write": 0,
        "review": 0,
        "reflection": 0,
        "fail_write": False,
        "fail_review": False,
        "failure_usage": 0,
        "success_usage": 0,
    }
    run_id = f"run-direct-{failed_step}-failure"
    execution_token = f"worker-{failed_step}"
    persistence = create_graph_persistence(config)
    try:
        runtime = _native_draft_runtime(persistence, calls, run_id=run_id)
        state = runtime._initial_state(
            {
                "message": "写通知",
                "request_id": f"request-direct-{failed_step}-failure",
            },
            user_id="u1",
        )
        persistence.start_run(
            run_id=run_id,
            thread_id=run_id,
            graph_name="chat",
            user_id="u1",
            session_id=state["session_id"],
            request_id=state["request_id"],
            workflow_version="chat-v3",
            current_node=f"document.{failed_step}",
            execution_token=execution_token,
            durability="async",
        )
        assert persistence.claim_effect(
            run_id,
            "tool_draft_document",
            "step:1:draft_document",
            user_id="u1",
            execution_token=execution_token,
        )

        context = runtime._new_run_context(
            user_info=None,
            execution_token=execution_token,
            lease=None,
            recovery=False,
            source_workflow_version="chat-v3",
        )
        context.handle_document_failure(
            {
                "run_id": run_id,
                "user_id": "u1",
                "session_id": state["session_id"],
                "input_user_request": "写通知",
                "parent_step_index": 0,
            },
            failed_step,
            RuntimeError(f"{failed_step} unavailable"),
        )

        effect = persistence.get_effect(
            run_id,
            "tool_draft_document",
            "step:1:draft_document",
            user_id="u1",
        )
        assert effect.status == "failed"
        assert calls["failure_usage"] == 1
    finally:
        persistence.close()


def test_sqlite_process_reopen_recovers_from_next_node(tmp_path):
    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / "checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )
    calls = {"plan": 0, "tool": 0}
    token = "worker-before-crash"
    run_id = "run-sqlite-recovery"

    first_persistence = create_graph_persistence(config)
    first_runtime = _runtime(first_persistence, calls, run_id=run_id)
    state = first_runtime._initial_state(
        {"message": "recover me", "request_id": "request-sqlite-recovery"},
        user_id="u1",
    )
    first_persistence.start_run(
        run_id=run_id,
        thread_id=run_id,
        graph_name="chat",
        user_id="u1",
        session_id=state["session_id"],
        request_id=state["request_id"],
        workflow_version="chat-v2",
        current_node="prepare",
        execution_token=token,
        durability="sync",
    )
    config_payload = {"configurable": {"thread_id": run_id}}
    list(first_runtime._graph.stream(
        state,
        context=ChatRunContext(user_info=None, execution_token=token),
        config=config_payload,
        stream_mode="custom",
        version="v2",
        durability="sync",
        interrupt_after=["select_step"],
    ))
    snapshot = first_runtime._graph.get_state(config_payload)
    assert snapshot.next == ("tool_knowledge_qa",)
    assert calls == {"plan": 1, "tool": 0}
    assert first_persistence.claim_effect(
        run_id,
        "tool_knowledge_qa",
        "step:1:knowledge_qa",
        user_id="u1",
        execution_token=token,
        claimed_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    assert first_persistence.finish_run(
        run_id,
        user_id="u1",
        execution_token=token,
        status="failed",
        current_node="worker_crashed",
        error="simulated process crash",
    ) is not None
    first_persistence.close()

    second_persistence = create_graph_persistence(config)
    try:
        second_runtime = _runtime(second_persistence, calls, run_id=run_id)
        events = parse_sse(second_runtime.recover(
            run_id,
            user_id="u1",
            user_info=None,
        ))

        assert events[-1]["type"] == "done"
        assert events[-1]["run_id"] == run_id
        assert calls == {"plan": 1, "tool": 1}
        record = second_persistence.get_owned_run(run_id, user_id="u1")
        assert record.status == "succeeded"
        assert record.current_node == "__end__"
    finally:
        second_persistence.close()


def test_sqlite_process_reopen_reclaims_failed_draft_effect(tmp_path):
    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / "checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )
    calls = {"plan": 0, "tool": 0}
    run_id = "run-sqlite-draft-retry"

    first_persistence = create_graph_persistence(config)
    first_runtime = _draft_retry_runtime(first_persistence, calls, run_id=run_id)
    first_events = parse_sse(first_runtime.stream(
        {"message": "写通知", "request_id": "request-sqlite-draft-retry"},
        user_id="u1",
        user_info=None,
    ))
    assert first_events[-1]["type"] == "error"
    assert first_persistence.get_effect(
        run_id,
        "tool_draft_document",
        "step:1:draft_document",
        user_id="u1",
    ).status == "failed"
    first_persistence.close()

    second_persistence = create_graph_persistence(config)
    try:
        second_runtime = _draft_retry_runtime(
            second_persistence,
            calls,
            run_id=run_id,
        )
        recovered = parse_sse(second_runtime.recover(
            run_id,
            user_id="u1",
            user_info=None,
        ))
        assert recovered[-1]["type"] == "done"
        assert recovered[-1]["document"] == "recovered document"
        assert calls == {"plan": 1, "tool": 2}
        assert second_persistence.get_effect(
            run_id,
            "tool_draft_document",
            "step:1:draft_document",
            user_id="u1",
        ).status == "completed"
        assert second_persistence.get_owned_run(run_id, user_id="u1").status == "succeeded"
    finally:
        second_persistence.close()


def test_sqlite_native_document_recovery_retries_only_failed_reviewer(tmp_path):
    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / "native-checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "native-runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )
    calls = {
        "plan": 0,
        "context_plan": 0,
        "retrieval": 0,
        "write": 0,
        "review": 0,
        "reflection": 0,
        "fail_review": True,
        "failure_usage": 0,
        "success_usage": 0,
    }
    run_id = "run-sqlite-native-draft"

    first_persistence = create_graph_persistence(config)
    first_runtime = _native_draft_runtime(first_persistence, calls, run_id=run_id)
    first_events = parse_sse(first_runtime.stream(
        {"message": "写通知", "request_id": "request-native-draft"},
        user_id="u1",
        user_info=None,
    ))
    assert first_events[-1]["type"] == "error"
    assert calls["context_plan"] == 1
    assert calls["retrieval"] == 1
    assert calls["write"] == 1
    assert calls["review"] == 1
    assert calls["failure_usage"] == 1
    effect = first_persistence.get_effect(
        run_id,
        "tool_draft_document",
        "step:1:draft_document",
        user_id="u1",
    )
    assert effect.status == "failed"
    first_persistence.close()

    calls["fail_review"] = False
    second_persistence = create_graph_persistence(config)
    try:
        second_runtime = _native_draft_runtime(second_persistence, calls, run_id=run_id)
        recovered = parse_sse(second_runtime.recover(
            run_id,
            user_id="u1",
            user_info=None,
        ))

        assert recovered[-1]["type"] == "done"
        assert recovered[-1]["document"] == "doc-v1"
        assert calls["plan"] == 1
        assert calls["context_plan"] == 1
        assert calls["retrieval"] == 1
        assert calls["write"] == 1
        assert calls["review"] == 2
        assert calls["reflection"] == 1
        assert calls["success_usage"] == 1
        effect = second_persistence.get_effect(
            run_id,
            "tool_draft_document",
            "step:1:draft_document",
            user_id="u1",
        )
        assert effect.status == "completed"
        assert second_persistence.get_owned_run(run_id, user_id="u1").status == "succeeded"
    finally:
        second_persistence.close()


def test_sqlite_native_document_recovery_retries_only_failed_writer(tmp_path):
    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / "native-writer-checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "native-writer-runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )
    calls = {
        "plan": 0,
        "context_plan": 0,
        "retrieval": 0,
        "write": 0,
        "review": 0,
        "reflection": 0,
        "fail_write": True,
        "fail_review": False,
        "failure_usage": 0,
        "success_usage": 0,
    }
    run_id = "run-sqlite-native-writer"

    first_persistence = create_graph_persistence(config)
    first_runtime = _native_draft_runtime(first_persistence, calls, run_id=run_id)
    first_events = parse_sse(first_runtime.stream(
        {"message": "写通知", "request_id": "request-native-writer"},
        user_id="u1",
        user_info=None,
    ))
    assert first_events[-1]["type"] == "error"
    assert calls["context_plan"] == 1
    assert calls["retrieval"] == 1
    assert calls["write"] == 1
    assert calls["review"] == 0
    assert first_persistence.get_effect(
        run_id,
        "tool_draft_document",
        "step:1:draft_document",
        user_id="u1",
    ).status == "failed"
    first_persistence.close()

    calls["fail_write"] = False
    second_persistence = create_graph_persistence(config)
    try:
        second_runtime = _native_draft_runtime(second_persistence, calls, run_id=run_id)
        recovered = parse_sse(second_runtime.recover(
            run_id,
            user_id="u1",
            user_info=None,
        ))

        assert recovered[-1]["type"] == "done"
        assert recovered[-1]["document"] == "doc-v1"
        assert calls["plan"] == 1
        assert calls["context_plan"] == 1
        assert calls["retrieval"] == 1
        assert calls["write"] == 2
        assert calls["review"] == 1
        assert calls["success_usage"] == 1
        assert second_persistence.get_effect(
            run_id,
            "tool_draft_document",
            "step:1:draft_document",
            user_id="u1",
        ).status == "completed"
    finally:
        second_persistence.close()


@pytest.mark.parametrize(
    "failure_stage",
    ["build_public_done", "billing", "profile", "effect"],
)
def test_native_document_commit_withholds_answer_until_durable_writes_finish(
    tmp_path,
    failure_stage,
):
    """A retryable commit failure must not expose a document as completed."""

    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / f"commit-{failure_stage}-checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / f"commit-{failure_stage}-runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )
    memory_path = tmp_path / f"commit-{failure_stage}-memory.sqlite"
    memory = TeamMemory(str(memory_path))
    session_id = memory.create_session("u1")
    run_id = f"run-native-commit-{failure_stage}"
    calls = {
        "plan": 0,
        "context_plan": 0,
        "retrieval": 0,
        "write": 0,
        "review": 0,
        "reflection": 0,
        "fail_write": False,
        "fail_review": False,
        "failure_usage": 0,
        "success_usage": 0,
    }
    payload = {
        "message": "写通知",
        "session_id": session_id,
        "request_id": f"request-native-commit-{failure_stage}",
    }
    forbidden_before_commit = {
        "answer_start",
        "answer_delta",
        "content",
        "answer_done",
    }

    first_persistence = create_graph_persistence(config)
    first_runtime = _native_draft_runtime(
        first_persistence,
        calls,
        run_id=run_id,
        business_memory=memory,
        commit_fault_stage=failure_stage,
    )
    first_events = parse_sse(first_runtime.stream(
        payload,
        user_id="u1",
        user_info=None,
    ))

    assert first_events[-1]["type"] == "error"
    assert forbidden_before_commit.isdisjoint(
        event.get("type") for event in first_events
    )
    assert first_persistence.get_effect(
        run_id,
        "tool_draft_document",
        "step:1:draft_document",
        user_id="u1",
    ).status == "failed"
    first_persistence.close()

    second_persistence = create_graph_persistence(config)
    recovered_memory = TeamMemory(str(memory_path))
    try:
        second_runtime = _native_draft_runtime(
            second_persistence,
            calls,
            run_id=run_id,
            business_memory=recovered_memory,
            commit_fault_stage=failure_stage,
        )
        recovered = parse_sse(second_runtime.recover(
            run_id,
            user_id="u1",
            user_info=None,
        ))

        assert recovered[-1]["type"] == "done"
        assert recovered[-1]["document"] == "doc-v1"
        assert [event["type"] for event in recovered].count("answer_start") == 1
        assert [event["type"] for event in recovered].count("answer_done") == 1
        assert second_persistence.get_effect(
            run_id,
            "tool_draft_document",
            "step:1:draft_document",
            user_id="u1",
        ).status == "completed"

        assistant_key = (
            f"{run_id}:tool_draft_document:step:1:message_assistant"
        )
        summary_key = (
            f"{run_id}:tool_draft_document:step:1:rolling_summary"
        )
        with recovered_memory._get_conn() as conn:
            assistant_writes = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE effect_key = ?",
                (assistant_key,),
            ).fetchone()[0]
            summary_writes = conn.execute(
                "SELECT COUNT(*) FROM memory_effects WHERE effect_key = ?",
                (summary_key,),
            ).fetchone()[0]
        assert assistant_writes == 1
        assert summary_writes == 1
        assert calls["billing_effects"] == {f"{run_id}:document_token_usage"}
        assert recovered_memory.get_user_profile("u1").common_doc_types.count("通知") == 1
    finally:
        second_persistence.close()


def test_recovery_fast_forwards_completed_effect_when_parent_checkpoint_lags(tmp_path):
    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / "lag-checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "lag-runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )
    calls = {"plan": 0, "tool": 0}
    run_id = "run-effect-ahead-of-checkpoint"
    token = "worker-before-checkpoint"
    first_persistence = create_graph_persistence(config)
    first_runtime = _runtime(first_persistence, calls, run_id=run_id)
    state = first_runtime._initial_state(
        {"message": "recover receipt", "request_id": "request-effect-receipt"},
        user_id="u1",
    )
    first_persistence.start_run(
        run_id=run_id,
        thread_id=run_id,
        graph_name="chat",
        user_id="u1",
        session_id=state["session_id"],
        request_id=state["request_id"],
        workflow_version="chat-v3",
        current_node="prepare",
        execution_token=token,
        durability="sync",
    )
    graph_config = {"configurable": {"thread_id": run_id}}
    list(first_runtime._graph.stream(
        state,
        context=ChatRunContext(user_info=None, execution_token=token),
        config=graph_config,
        stream_mode="custom",
        version="v2",
        durability="sync",
        interrupt_after=["select_step"],
    ))
    assert first_runtime._graph.get_state(graph_config).next == ("tool_knowledge_qa",)
    assert first_persistence.claim_effect(
        run_id,
        "tool_knowledge_qa",
        "step:1:knowledge_qa",
        user_id="u1",
        execution_token=token,
    )
    receipt_done = {
        "type": "done",
        "intent": "knowledge_qa",
        "answer": "durable receipt",
        "document": "",
        "session_id": state["session_id"],
        "plan": {},
        "actions": [],
        "source_filenames": [],
        "source_details": [],
    }
    assert first_persistence.complete_effect(
        run_id,
        "tool_knowledge_qa",
        "step:1:knowledge_qa",
        user_id="u1",
        execution_token=token,
        result={"step_done": receipt_done, "failed": False},
    ) is not None
    assert first_persistence.finish_run(
        run_id,
        user_id="u1",
        execution_token=token,
        status="failed",
        current_node="worker_crashed_after_effect",
        error="simulated async checkpoint lag",
    ) is not None
    first_persistence.close()

    second_persistence = create_graph_persistence(config)
    try:
        second_runtime = _runtime(second_persistence, calls, run_id=run_id)
        recovered = parse_sse(second_runtime.recover(
            run_id,
            user_id="u1",
            user_info=None,
        ))

        assert recovered[-1]["type"] == "done"
        assert recovered[-1]["answer"] == "durable receipt"
        assert calls == {"plan": 1, "tool": 0}
        assert second_persistence.get_owned_run(run_id, user_id="u1").status == "succeeded"
    finally:
        second_persistence.close()


def test_chat_v2_execute_checkpoint_recovers_with_v3_native_topology(tmp_path):
    config = GraphPersistenceConfig(
        environment="development",
        backend="sqlite",
        sqlite_path=tmp_path / "upgrade-checkpoints.sqlite",
        run_ledger_sqlite_path=tmp_path / "upgrade-runs.sqlite",
        run_ledger_backend="sqlite",
        default_durability="sync",
    )
    calls = {
        "plan": 0,
        "tool": 0,
        "context_plan": 0,
        "retrieval": 0,
        "write": 0,
        "review": 0,
        "reflection": 0,
        "fail_review": False,
        "failure_usage": 0,
        "success_usage": 0,
    }
    run_id = "run-chat-v2-draft-upgrade"

    first_persistence = create_graph_persistence(config)
    first_runtime = _draft_retry_runtime(first_persistence, calls, run_id=run_id)
    first_runtime.deps.workflow_version = "chat-v2"
    first_events = parse_sse(first_runtime.stream(
        {"message": "写通知", "request_id": "request-chat-v2-upgrade"},
        user_id="u1",
        user_info=None,
    ))
    assert first_events[-1]["type"] == "error"
    assert first_runtime._graph.get_state(
        {"configurable": {"thread_id": run_id}}
    ).next == ("tool_draft_document",)
    assert first_persistence.get_owned_run(run_id, user_id="u1").workflow_version == "chat-v2"
    first_persistence.close()

    second_persistence = create_graph_persistence(config)
    try:
        second_runtime = _native_draft_runtime(
            second_persistence,
            calls,
            run_id=run_id,
            workflow_version="chat-v3",
        )
        recovered = parse_sse(second_runtime.recover(
            run_id,
            user_id="u1",
            user_info=None,
        ))

        assert recovered[-1]["type"] == "done"
        assert recovered[-1]["document"] == "doc-v1"
        assert calls["plan"] == 1
        assert calls["tool"] == 1
        assert calls["context_plan"] == 1
        assert second_persistence.get_owned_run(run_id, user_id="u1").status == "succeeded"
    finally:
        second_persistence.close()
