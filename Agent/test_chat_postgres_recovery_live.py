"""Opt-in crash/recovery matrix for the production chat and document graphs.

Run through ``docker-compose.langgraph-test.yml`` or set
``LANGGRAPH_TEST_POSTGRES_DSN`` to a disposable PostgreSQL database.  Every
case persists a real production-topology checkpoint with ``durability=async``,
terminates the worker with ``os._exit()``, then resumes from a fresh process.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import uuid
from types import SimpleNamespace

import pytest

from agents.document_graph_runner import DocumentGraphRunner
from chat_architecture import RouteResult
from chat_events import sse
from chat_runtime import ChatGraphRuntime, ChatRunContext, ChatRuntimeDependencies
from graph_persistence import (
    GraphPersistenceConfig,
    create_graph_persistence,
    setup_postgres_persistence,
)
from task_planner import TaskPlan, TaskStep
from test_graph_persistence_postgres_live import isolated_postgres_dsn
from tool_runtime import ChatTool, ToolOrchestrator, ToolRegistry


POSTGRES_TEST_DSN = os.getenv("LANGGRAPH_TEST_POSTGRES_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not POSTGRES_TEST_DSN,
    reason="set LANGGRAPH_TEST_POSTGRES_DSN to run real PostgreSQL acceptance",
)

_USER_ID = "chat-crash-user"
_SESSION_ID = "chat-crash-session"


def _postgres_async_config(dsn: str) -> GraphPersistenceConfig:
    return GraphPersistenceConfig(
        environment="production",
        backend="postgres",
        postgres_dsn=dsn,
        run_ledger_backend="postgres",
        postgres_pool_min_size=1,
        postgres_pool_max_size=2,
        postgres_pool_timeout_seconds=3.0,
        worker_processes=4,
        default_durability="async",
    )


class _Memory:
    def get_or_create_session(self, _user_id, session_id=None):
        return str(session_id or _SESSION_ID)

    def get_context_for_prompt(self, *_args, **_kwargs):
        return ""

    def get_context(self, _session_id, _key, default=None):
        return default

    def delete_session(self, _session_id):
        return True


class _Uploads:
    def get_temp_content(self, *_args, **_kwargs):
        return ""

    def get_temp_file_info(self, *_args, **_kwargs):
        return None


class _Planner:
    def plan(self, **_kwargs):
        return TaskPlan(
            task_type="knowledge_qa",
            steps=[TaskStep(tool="knowledge_qa", reason="postgres crash matrix")],
            route=RouteResult("knowledge_qa", 1.0, "postgres crash matrix"),
        )


class _DocumentOrchestrator:
    """Deterministic services mounted behind the production document graph."""

    MAX_TOTAL_ROUNDS = 3

    def __init__(self, *, reflect: bool = True):
        self.reflect = reflect
        self.write_count = 0
        self.review_count = 0
        self.think_log: list[dict] = []

    def _step_context_plan(self, request_with_context, _previous_context, _think):
        return SimpleNamespace(
            user_request=request_with_context,
            context_analysis={},
            plan={"task_type": "document", "document_type": "notice"},
            search_context="",
            knowledge_context="",
            knowledge_sources=[],
            search_sources=[],
            evidence_items=[],
            compact_evidence=[],
            revision_history=[],
            run_records=[],
            audit_summary={},
            last_document="",
            last_plan={},
            user_constraints=[],
            unresolved_questions=[],
            user_profile=None,
            memory_context="",
        )

    @staticmethod
    def _step_search(ctx, _think):
        return ctx

    @staticmethod
    def _step_knowledge(ctx, _think):
        ctx.knowledge_sources = [{"filename": "source.docx"}]
        return ctx

    def _step_write(self, _ctx, _think):
        self.write_count += 1
        return SimpleNamespace(
            success=True,
            content=f"doc-v{self.write_count}",
            metadata={},
            error_info={},
        )

    def _step_review(self, _ctx, _document, _think):
        self.review_count += 1
        return SimpleNamespace(
            success=True,
            content="",
            error_info={},
            metadata={
                "needs_revision": False,
                "revision_focus": [],
                "suggestions": [],
                "format_check": {"issues": []},
                "content_check": {"issues": []},
                "logic_check": {"issues": []},
                "language_check": {"issues": []},
                "fact_check": {"issues": []},
                "spreadsheet_audit": {"ok": True},
                "confidence": 0.95,
            },
        )

    @staticmethod
    def _step_reflection(_ctx, _document, _think):
        return SimpleNamespace(
            success=True,
            content="",
            error_info={},
            metadata={
                "needs_revision": False,
                "revision_suggestions": [],
                "weaknesses": [],
                "counter_arguments": [],
                "logic_score": 0.9,
            },
        )

    @staticmethod
    def _record_step(ctx, step, _started_at, **extra):
        ctx.run_records.append({"step": step, **extra})

    @staticmethod
    def _build_evidence_items(_ctx):
        return [{"filename": "source.docx"}]

    @staticmethod
    def _compact_evidence_items(items):
        return [{"filename": item["filename"]} for item in items]

    @staticmethod
    def _combined_revision_focus(review_meta, _reflection_meta=None):
        return list(review_meta.get("revision_focus") or [])

    def _should_reflect(self, _ctx, _review_meta, revision_round):
        return self.reflect and revision_round == 0

    @staticmethod
    def _sanitize_document_output(text, _user_request):
        return text


def _chat_runtime(dsn: str, run_id: str):
    persistence = create_graph_persistence(_postgres_async_config(dsn))

    def tool_stream(
        _message,
        session_id,
        _user_id,
        _user_info,
        _display_message,
        user_metadata,
        _route,
    ):
        import psycopg

        metadata = dict(user_metadata or {})
        scope = str(metadata.get("effect_scope") or "step:1")
        effect_key = f"{run_id}:tool_knowledge_qa:{scope}:business_result"
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                """
                INSERT INTO chat_acceptance_effects (effect_key, answer)
                VALUES (%s, %s)
                ON CONFLICT (effect_key) DO NOTHING
                """,
                (effect_key, "durable answer"),
            )
        yield sse({"type": "content", "data": "durable answer"})
        yield sse({
            "type": "done",
            "intent": "knowledge_qa",
            "answer": "durable answer",
            "document": "",
            "session_id": session_id,
            "plan": {},
            "actions": [],
        })

    registry = ToolRegistry()
    registry.register(ChatTool(
        name="knowledge_qa",
        description="postgres acceptance tool",
        risk_level="low",
        input_schema={"message": "string"},
        stream=tool_stream,
    ))
    runtime = ChatGraphRuntime(ChatRuntimeDependencies(
        memory=_Memory(),
        upload_manager=_Uploads(),
        reimbursement_detector=lambda *_args: "",
        lightweight_stream=lambda *_args: iter(()),
        document_format_stream=lambda *_args: iter(()),
        document_draft_stream=lambda *_args: iter(()),
        rag_qa_stream=lambda *_args: iter(()),
        task_planner=_Planner(),
        tool_orchestrator=ToolOrchestrator(registry),
        checkpointer=persistence.checkpointer,
        persistence=persistence,
        run_id_factory=lambda: run_id,
    ), runtime_mode="graph")
    return runtime, persistence


def _parse_sse(chunks):
    events = []
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    return events


def _crash_chat_after_node(dsn: str, run_id: str, node: str, sender) -> None:
    persistence = None
    try:
        runtime, persistence = _chat_runtime(dsn, run_id)
        token = f"crash-{uuid.uuid4().hex}"
        state = runtime._initial_state(
            {
                "message": "recover production topology",
                "session_id": _SESSION_ID,
                "request_id": f"request-{run_id}",
            },
            user_id=_USER_ID,
        )
        persistence.start_run(
            run_id=run_id,
            thread_id=run_id,
            graph_name="chat",
            user_id=_USER_ID,
            session_id=state["session_id"],
            request_id=state["request_id"],
            idempotency_key=state.get("idempotency_key") or None,
            workflow_version=runtime.deps.workflow_version,
            current_node="prepare",
            execution_token=token,
            durability="async",
            metadata={
                "configured_runtime": "graph",
                "request_digest": state.get("request_digest", ""),
            },
        )
        config = {"configurable": {"thread_id": run_id}}
        list(runtime._graph.stream(
            state,
            context=ChatRunContext(user_info=None, execution_token=token),
            config=config,
            stream_mode="custom",
            version="v2",
            durability="async",
            interrupt_after=[node],
        ))
        snapshot = runtime._graph.get_state(config)
        sender.send({
            "ok": True,
            "next": list(snapshot.next),
            "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
        })
        sender.close()
    except BaseException as exc:
        try:
            sender.send({"ok": False, "error": repr(exc)})
            sender.close()
        finally:
            if persistence is not None:
                persistence.close()
        os._exit(91)
    os._exit(17)


def _recover_chat_in_new_worker(dsn: str, run_id: str, sender) -> None:
    persistence = None
    try:
        runtime, persistence = _chat_runtime(dsn, run_id)
        # Process startup already exceeds this lease interval, while retaining
        # the production invariant that a reclaim threshold must be positive.
        runtime.RECOVERY_LEASE_STALE_SECONDS = 0.001
        events = _parse_sse(runtime.recover(
            run_id,
            user_id=_USER_ID,
            user_info=None,
        ))
        import psycopg

        with psycopg.connect(dsn, autocommit=True) as connection:
            business_rows = connection.execute(
                "SELECT effect_key, answer FROM chat_acceptance_effects ORDER BY effect_key"
            ).fetchall()
        effects = persistence.list_run_effects(run_id, user_id=_USER_ID)
        record = persistence.get_owned_run(run_id, user_id=_USER_ID)
        sender.send({
            "ok": True,
            "events": events,
            "business_rows": business_rows,
            "effects": [
                (item.node, item.effect, item.status)
                for item in effects
            ],
            "run_status": record.status if record is not None else "missing",
        })
        sender.close()
    except BaseException as exc:
        try:
            sender.send({"ok": False, "error": repr(exc)})
            sender.close()
        finally:
            if persistence is not None:
                persistence.close()
        raise
    finally:
        if persistence is not None:
            persistence.close()


def _prepared_document(run_id: str):
    return SimpleNamespace(
        user_request="write a notice",
        request_with_context="write a notice",
        previous_context="",
        session_id=_SESSION_ID,
        user_id=_USER_ID,
        display_message="write a notice",
        run_id=run_id,
    )


def _crash_document_after_node(dsn: str, run_id: str, node: str, sender) -> None:
    persistence = None
    try:
        persistence = create_graph_persistence(_postgres_async_config(dsn))
        orchestrator = _DocumentOrchestrator(reflect=True)
        runner = DocumentGraphRunner(
            orchestrator,
            checkpointer=persistence.checkpointer,
            durability="async",
        )
        prepared = _prepared_document(run_id)
        config = runner._config(run_id)
        list(runner.graph.stream(
            runner._initial_state(prepared, run_id),
            config=config,
            context=runner._runtime(lambda *_args: None, streaming=False),
            stream_mode="values",
            durability="async",
            interrupt_after=[node],
        ))
        snapshot = runner.graph.get_state(config)
        sender.send({
            "ok": True,
            "next": list(snapshot.next),
            "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
        })
        sender.close()
    except BaseException as exc:
        try:
            sender.send({"ok": False, "error": repr(exc)})
            sender.close()
        finally:
            if persistence is not None:
                persistence.close()
        os._exit(92)
    os._exit(23)


def _recover_document_in_new_worker(dsn: str, run_id: str, sender) -> None:
    persistence = None
    try:
        persistence = create_graph_persistence(_postgres_async_config(dsn))
        runner = DocumentGraphRunner(
            _DocumentOrchestrator(reflect=True),
            checkpointer=persistence.checkpointer,
            durability="async",
        )
        result = runner.run(
            _prepared_document(run_id),
            think_handler=lambda *_args: None,
            thread_id=run_id,
        )
        sender.send({
            "ok": True,
            "document": result.document_content,
            "quality_status": result.quality_status,
            "revisions_applied": result.revisions_applied,
        })
        sender.close()
    except BaseException as exc:
        try:
            sender.send({"ok": False, "error": repr(exc)})
            sender.close()
        finally:
            if persistence is not None:
                persistence.close()
        raise
    finally:
        if persistence is not None:
            persistence.close()


def _run_crash_pair(target, recover, args, expected_exitcode):
    process_context = multiprocessing.get_context("spawn")
    crash_receiver, crash_sender = process_context.Pipe(duplex=False)
    crashing_worker = process_context.Process(
        target=target,
        args=(*args, crash_sender),
    )
    crashing_worker.start()
    crash_sender.close()
    assert crash_receiver.poll(45), "crashing worker did not report in time"
    crash_report = crash_receiver.recv()
    crashing_worker.join(timeout=45)
    assert not crashing_worker.is_alive(), "crashing worker did not exit"
    assert crash_report.get("ok") is True, crash_report
    assert crash_report["checkpoint_id"]
    assert crashing_worker.exitcode == expected_exitcode

    recovery_receiver, recovery_sender = process_context.Pipe(duplex=False)
    recovering_worker = process_context.Process(
        target=recover,
        args=(args[0], args[1], recovery_sender),
    )
    recovering_worker.start()
    recovery_sender.close()
    assert recovery_receiver.poll(45), "recovering worker did not report in time"
    recovery_report = recovery_receiver.recv()
    recovering_worker.join(timeout=45)
    assert not recovering_worker.is_alive(), "recovering worker did not exit"
    assert recovering_worker.exitcode == 0
    assert recovery_report.get("ok") is True, recovery_report
    return crash_report, recovery_report


@pytest.fixture()
def production_graph_postgres(isolated_postgres_dsn):
    scoped_dsn, schema = isolated_postgres_dsn
    setup_postgres_persistence(_postgres_async_config(scoped_dsn))
    import psycopg

    with psycopg.connect(scoped_dsn, autocommit=True) as connection:
        connection.execute(
            """
            CREATE TABLE chat_acceptance_effects (
                effect_key TEXT PRIMARY KEY,
                answer TEXT NOT NULL
            )
            """
        )
    return scoped_dsn, schema


@pytest.mark.parametrize(
    "crash_node",
    [
        "prepare",
        "plan_tools",
        "select_step",
        "tool_knowledge_qa",
        "collect_result",
        "finalize",
    ],
)
def test_real_postgres_recovers_production_chat_after_every_node(
    production_graph_postgres,
    crash_node,
):
    dsn, _schema = production_graph_postgres
    run_id = f"chat-node-{crash_node}-{uuid.uuid4().hex}"
    _crash, recovered = _run_crash_pair(
        _crash_chat_after_node,
        _recover_chat_in_new_worker,
        (dsn, run_id, crash_node),
        17,
    )

    assert recovered["events"][-1]["type"] == "done"
    assert recovered["events"][-1]["answer"] == "durable answer"
    assert recovered["run_status"] == "succeeded"
    assert recovered["business_rows"] == [
        (f"{run_id}:tool_knowledge_qa:step:1:business_result", "durable answer")
    ]
    assert set(map(tuple, recovered["effects"])) == {
        ("plan_tools", "task_plan", "completed"),
        ("tool_knowledge_qa", "step:1:knowledge_qa", "completed"),
    }


@pytest.mark.parametrize(
    "crash_node",
    [
        "context_plan",
        "retrieval",
        "write",
        "review",
        "reflection",
        "decide",
        "finalize",
    ],
)
def test_real_postgres_recovers_document_graph_after_every_node(
    production_graph_postgres,
    crash_node,
):
    dsn, _schema = production_graph_postgres
    run_id = f"document-node-{crash_node}-{uuid.uuid4().hex}"
    _crash, recovered = _run_crash_pair(
        _crash_document_after_node,
        _recover_document_in_new_worker,
        (dsn, run_id, crash_node),
        23,
    )

    assert recovered["document"] == "doc-v1"
    assert recovered["quality_status"] == "passed"
    assert recovered["revisions_applied"] == 0
