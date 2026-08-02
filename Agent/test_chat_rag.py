import json
import sys
from types import ModuleType
from types import SimpleNamespace

from chat_answer_quality import (
    ANSWER_INTENT_COMPARE_SUMMARIZE,
    ANSWER_INTENT_FILE_DISCOVERY,
    ANSWER_INTENT_FOLLOWUP_REFINE,
    ANSWER_INTENT_PROCEDURE_HELP,
    ANSWER_INTENT_SPREADSHEET_FACT,
    EvidenceReport,
    VerificationReport,
    build_answer_plan,
    should_use_llm_planner,
    verify_answer,
)
from chat_events import source_details_from_results
from chat_rag import RagQaDependencies, RagQaStreamService, build_storage_server_answer, chat_context_for_model


class FakeMemory:
    def __init__(self):
        self.messages = []
        self.context = {}
        self.summaries = []
        self.profile = SimpleNamespace(name="测试用户", department="测试部")

    def add_message(self, session_id, role, content, metadata=None):
        self.messages.append((session_id, role, content, metadata or {}))

    def get_context_for_prompt(self, session_id, max_messages=5):
        return "user: 上一轮问题"

    def get_user_profile(self, user_id):
        return self.profile

    def set_context(self, session_id, key, value):
        self.context[(session_id, key)] = value

    def update_rolling_summary(self, session_id, message, response, plan, sources):
        self.summaries.append((session_id, message, response, plan, sources))


class FakeKnowledgeAgent:
    def process(self, payload):
        self.payload = payload
        return SimpleNamespace(
            success=True,
            content="文件名: 制度.docx\n片段: 内部坐标\n正文内容\n行号：3\n网址：https://example.test",
            metadata={"results": [{"filename": "制度.docx"}, {"source": "/tmp/另一个文件.pdf"}]},
        )


class EmptyKnowledgeAgent:
    def process(self, payload):
        self.payload = payload
        return SimpleNamespace(success=True, content="", metadata={"results": []})


class DocumentOnlyKnowledgeAgent:
    def process(self, payload):
        self.payload = payload
        return SimpleNamespace(
            success=True,
            content="文件名: 会议费制度.docx\n会议费管理办法说明",
            metadata={"results": [{"filename": "会议费制度.docx", "source_type": "document", "text": "会议费管理办法说明"}]},
        )


class SpreadsheetFeeKnowledgeAgent:
    def process(self, payload):
        self.payload = payload
        return SimpleNamespace(
            success=True,
            content=(
                "文件名: 附件：场地使用收费表.xlsx\n"
                "3F A301 培训室 可容纳人数 30 计费方式 1天 金额 1200\n"
                "3F A302 会议室 可容纳人数 12 计费方式 1天 金额 800"
            ),
            metadata={
                "results": [{
                    "filename": "附件：场地使用收费表.xlsx",
                    "source_type": "spreadsheet",
                    "text": "A301 培训室 30 1天 1200",
                }]
            },
        )


class StorageServerKnowledgeAgent:
    def process(self, payload):
        self.payload = payload
        return SimpleNamespace(
            success=True,
            content=(
                "文件名: 示例单位存储服务器运营方案-2025.4.28.docx\n"
                "（一）访问方式：键盘Windows+R，运行窗口显示后输入\\\\192.0.2.10进入账户登录界面，"
                "输入用户名和密码并勾选“保存此凭证”对虚拟盘进行访问及读写。\n"
                "（二）快捷方式：桌面右键新建快捷方式，对象地址输入\\\\192.0.2.10。"
            ),
            metadata={
                "results": [
                    {
                        "filename": "示例单位存储服务器运营方案-2025.4.28.docx",
                        "source_type": "document",
                        "text": "访问方式 Windows+R 输入\\\\192.0.2.10 输入用户名和密码 保存此凭证 快捷方式",
                    },
                    {
                        "filename": "无关材料.docx",
                        "source_type": "document",
                        "text": "其他资料",
                    },
                ]
            },
        )


class RichKnowledgeAgent:
    def __init__(self):
        self.payload = {}

    def process(self, payload):
        self.payload = payload
        return SimpleNamespace(
            success=True,
            content=(
                "文件名: 示例智库申报.docx\n示例智库申报方向包括政策研究与平台建设。\n\n"
                "文件名: 教育发展三年计划.docx\n教育发展三年行动计划强调科研平台和人才培养。"
            ),
            metadata={
                "results": [
                    {"filename": "示例智库申报.docx", "source_type": "document", "text": "政策研究 平台建设"},
                    {"filename": "教育发展三年计划.docx", "source_type": "document", "text": "科研平台 人才培养"},
                ]
            },
        )


class FakeOpenAI:
    calls = []

    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.__class__.calls.append(kwargs)
        if kwargs.get("stream"):
            return [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="可结合示例智库申报.docx "))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="和教育发展三年计划.docx 梳理申报方向。"))]),
            ]
        content = json.dumps({
            "answer_intent": ANSWER_INTENT_COMPARE_SUMMARIZE,
            "queries": ["结合示例智库申报和教育发展三年计划梳理申报方向", "示例智库 申报方向", "教育发展 三年计划 科研平台"],
            "expected_evidence": ["示例智库申报材料", "教育发展三年计划"],
            "needs_clarification": False,
            "clarification_question": "",
            "strict_grounding": True,
            "confidence": 0.9,
        }, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class StreamingOnlyOpenAI:
    calls = []

    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.__class__.calls.append(kwargs)
        assert kwargs.get("stream") is True
        return [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="当前未引用知识库资料，可考虑三个报告题目。"))])
        ]


def parse_sse(chunks):
    events = []
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    return events


def test_chat_context_for_model_hides_internal_coordinates():
    cleaned = chat_context_for_model("文件名: 制度.docx\n片段: 1\n正文\n行号：3\n未命名列2：0\n未命名列3：有效值")
    assert "片段" not in cleaned
    assert "行号" not in cleaned
    assert "未命名列" not in cleaned
    assert "正文" in cleaned
    assert "有效值" in cleaned


def test_source_details_from_results_deduplicates_filenames():
    details = source_details_from_results([
        {"filename": "a.docx"},
        {"filename": "a.docx"},
        {"source": "/tmp/b.pdf"},
    ])
    assert details == [{"filename": "a.docx"}, {"filename": "b.pdf"}]


def test_storage_answer_ignores_addresses_from_conversation_history():
    answer = build_storage_server_answer(
        {"task_type": "storage_server_usage"},
        "用户上一轮提到 \\\\192.0.2.10",
        [],
    )
    assert answer == ""


def test_answer_planner_classifies_retrieval_intents():
    assert build_answer_plan("帮我找一下示例新区相关文件").answer_intent == ANSWER_INTENT_FILE_DISCOVERY
    assert build_answer_plan("会议费标准是多少").answer_intent == ANSWER_INTENT_SPREADSHEET_FACT
    assert build_answer_plan("这个系统怎么登录").answer_intent == ANSWER_INTENT_PROCEDURE_HELP
    assert build_answer_plan("对比一下两份材料的差异").answer_intent == ANSWER_INTENT_COMPARE_SUMMARIZE

    followup = build_answer_plan("刚才那个再说一下", "")
    assert followup.answer_intent == ANSWER_INTENT_FOLLOWUP_REFINE
    assert followup.needs_clarification is True


def test_answer_planner_normalizes_equivalent_venue_fee_questions():
    questions = [
        "场地收费表里面是什么内容",
        "场地使用收费表内容是什么？有哪些教室和收费？",
        "有哪些教室多少钱",
        "教室一天怎么收费",
        "智慧教室收费标准",
    ]
    plans = [build_answer_plan(question) for question in questions]

    canonical_queries = {plan.normalized_task.get("canonical_query") for plan in plans}
    task_types = {plan.normalized_task.get("task_type") for plan in plans}

    assert task_types == {"spreadsheet_fee_lookup"}
    assert canonical_queries == {"场地使用收费表 教室 会议室 报告厅 贵宾厅 门牌号 可容纳人数 计费方式 金额 收费标准"}
    assert {plan.answer_intent for plan in plans} == {ANSWER_INTENT_SPREADSHEET_FACT}
    assert all(plan.planner_source == "normalizer" for plan in plans)
    assert all(should_use_llm_planner("换个问法也不调用", plan, mode="auto") is False for plan in plans)


def test_answer_planner_normalizes_equivalent_storage_server_questions():
    questions = [
        "NAS服务器如何使用？",
        "如何使用网盘呢",
        "nas服务器里面应该有地址吧",
        "网盘地址是什么",
        "存储服务器账号和访问地址是什么？",
    ]
    plans = [build_answer_plan(question) for question in questions]

    canonical_queries = {plan.normalized_task.get("canonical_query") for plan in plans}
    task_types = {plan.normalized_task.get("task_type") for plan in plans}

    assert task_types == {"storage_server_usage"}
    assert canonical_queries == {
        "示例单位存储服务器运营方案 NAS服务器 网盘 私有云 存储服务器 访问方式 访问地址 账号 密码 Windows+R 快捷方式"
    }
    assert {plan.answer_intent for plan in plans} == {ANSWER_INTENT_PROCEDURE_HELP}
    assert all("访问地址/路径" in plan.expected_evidence for plan in plans)
    assert all(plan.planner_source == "normalizer" for plan in plans)


def test_llm_planner_heuristic_only_calls_for_complex_requests():
    assert should_use_llm_planner("会议费标准是多少", build_answer_plan("会议费标准是多少"), mode="auto") is False
    complex_message = "结合示例智库申报材料和教育发展三年行动计划，帮我梳理可申报方向并列出依据"
    assert should_use_llm_planner(complex_message, build_answer_plan(complex_message), mode="auto") is True


def test_rag_stream_records_failed_usage_when_api_key_missing():
    memory = FakeMemory()
    usage_calls = []
    service = RagQaStreamService(RagQaDependencies(
        memory=memory,
        knowledge_agent=FakeKnowledgeAgent(),
        deepseek_api_key="",
        record_token_usage=lambda **kwargs: usage_calls.append(kwargs),
    ))

    events = parse_sse(service.stream(
        "制度怎么查？",
        "session_1",
        "user_1",
        user_info=SimpleNamespace(to_dict=lambda: {"user_id": "user_1"}),
        display_message="制度怎么查？",
        route=SimpleNamespace(to_dict=lambda: {"intent": "knowledge_qa", "actions": []}),
    ))

    assert [event["type"] for event in events[:3]] == ["start", "session", "route"]
    assert any(event["type"] == "thinking_start" for event in events)
    assert events[-1]["type"] == "error"
    assert "API Key" in events[-1]["message"]
    assert memory.messages == [("session_1", "user", "制度怎么查？", {})]
    assert usage_calls[0]["status"] == "failed"
    assert usage_calls[0]["mode"] == "chat"


def test_rag_stream_uses_normalized_query_for_spreadsheet_fee(monkeypatch):
    openai_module = ModuleType("openai")
    openai_module.OpenAI = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model should not be called"))
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    knowledge = SpreadsheetFeeKnowledgeAgent()
    service = RagQaStreamService(RagQaDependencies(
        memory=FakeMemory(),
        knowledge_agent=knowledge,
        deepseek_api_key="",
        record_token_usage=lambda **kwargs: None,
    ))

    events = parse_sse(service.stream(
        "有哪些教室多少钱",
        "session_1",
        "user_1",
        display_message="有哪些教室多少钱",
    ))

    assert events[-1]["type"] == "error"
    assert knowledge.payload["user_request"] == "场地使用收费表 教室 会议室 报告厅 贵宾厅 门牌号 可容纳人数 计费方式 金额 收费标准"
    assert knowledge.payload["knowledge_queries"][0] == knowledge.payload["user_request"]
    assert knowledge.payload["normalized_task"]["task_type"] == "spreadsheet_fee_lookup"


def test_rag_stream_answers_storage_server_address_without_model(monkeypatch):
    openai_module = ModuleType("openai")
    openai_module.OpenAI = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model should not be called"))
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    knowledge = StorageServerKnowledgeAgent()
    memory = FakeMemory()
    usage_calls = []
    service = RagQaStreamService(RagQaDependencies(
        memory=memory,
        knowledge_agent=knowledge,
        deepseek_api_key="",
        record_token_usage=lambda **kwargs: usage_calls.append(kwargs),
    ))

    events = parse_sse(service.stream(
        "网盘地址是什么",
        "session_1",
        "user_1",
        display_message="网盘地址是什么",
    ))

    done = events[-1]
    assert done["type"] == "done"
    assert "\\\\192.0.2.10" in done["answer"]
    assert "Windows + R" in done["answer"]
    assert "保存此凭证" in done["answer"]
    assert done["plan"]["normalized_task"]["task_type"] == "storage_server_usage"
    assert done["source_filenames"] == ["示例单位存储服务器运营方案-2025.4.28.docx"]
    assert knowledge.payload["user_request"] == (
        "示例单位存储服务器运营方案 NAS服务器 网盘 私有云 存储服务器 访问方式 访问地址 账号 密码 Windows+R 快捷方式"
    )
    assert usage_calls[-1]["model"] == "none"


def test_rag_stream_returns_evidence_fallback_without_model_when_no_results(monkeypatch):
    openai_module = ModuleType("openai")
    openai_module.OpenAI = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model should not be called"))
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    memory = FakeMemory()
    usage_calls = []
    service = RagQaStreamService(RagQaDependencies(
        memory=memory,
        knowledge_agent=EmptyKnowledgeAgent(),
        deepseek_api_key="present",
        record_token_usage=lambda **kwargs: usage_calls.append(kwargs),
    ))

    events = parse_sse(service.stream(
        "制度依据是什么？",
        "session_1",
        "user_1",
        display_message="制度依据是什么？",
        route=SimpleNamespace(to_dict=lambda: {"intent": "knowledge_qa", "actions": []}),
    ))

    assert events[-1]["type"] == "done"
    assert events[-1]["audit_summary"]["evidence_passed"] is False
    assert events[-1]["plan"]["task_type"] == "证据不足答复"
    assert "不足" in events[-1]["answer"]
    deltas = [event["data"] for event in events if event["type"] == "answer_delta"]
    assert len(deltas) > 1
    assert "".join(deltas) == events[-1]["answer"]
    assert usage_calls[-1]["status"] == "skipped"


def test_rag_stream_spreadsheet_question_requires_table_evidence(monkeypatch):
    openai_module = ModuleType("openai")
    openai_module.OpenAI = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model should not be called"))
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    service = RagQaStreamService(RagQaDependencies(
        memory=FakeMemory(),
        knowledge_agent=DocumentOnlyKnowledgeAgent(),
        deepseek_api_key="present",
        record_token_usage=lambda **kwargs: None,
    ))

    events = parse_sse(service.stream(
        "会议费标准是多少？",
        "session_1",
        "user_1",
        display_message="会议费标准是多少？",
    ))

    assert events[-1]["audit_summary"]["answer_intent"] == ANSWER_INTENT_SPREADSHEET_FACT
    assert events[-1]["audit_summary"]["evidence_passed"] is False
    assert "表格" in events[-1]["answer"]


def test_answer_verifier_rejects_unknown_sources_and_cleans_internal_coordinates():
    report = verify_answer(
        "参考 未知制度.docx 的片段: 1，可得金额 999 元。",
        build_answer_plan("制度依据是什么？", "上一轮"),
        [{"filename": "制度.docx", "text": "金额 100 元"}],
        "文件名: 制度.docx\n金额 100 元",
    )

    assert report.severe is True
    assert "未知制度.docx" in report.sanitized_answer
    assert "不足以支持" not in report.sanitized_answer
    assert any("未由知识库返回" in issue for issue in report.issues)


def test_answer_verifier_allows_attachment_filename_alias():
    report = verify_answer(
        "根据《场地使用收费表》，A301培训室1天收费1,200元。",
        build_answer_plan("场地使用收费表有哪些教室和收费？", "上一轮"),
        [{"filename": "附件：场地使用收费表.xlsx", "text": "A301 培训室 1天 金额 1200"}],
        "文件名: 附件：场地使用收费表.xlsx\nA301 培训室 1天 金额 1200",
    )

    assert report.severe is False
    assert "不足以支持" not in report.sanitized_answer
    assert report.issues == []


def test_answer_verifier_sanitizes_internal_coordinates_without_severe_error():
    report = verify_answer(
        "参考 制度.docx 的[文档1]片段: 1，结论如下。",
        build_answer_plan("制度依据是什么？", "上一轮"),
        [{"filename": "制度.docx", "text": "结论如下"}],
        "文件名: 制度.docx\n结论如下",
    )

    assert report.severe is False
    assert "[文档1]" not in report.sanitized_answer
    assert "片段" not in report.sanitized_answer


def test_verification_notes_do_not_replace_streamed_answer_when_severe():
    answer = RagQaStreamService._answer_with_verification_notes(
        VerificationReport(
            passed=False,
            issues=["回答引用了未由知识库返回的来源：未知制度.docx"],
            severe=True,
            sanitized_answer="参考 未知制度.docx 的内容，结论如下。",
        ),
        EvidenceReport(
            passed=True,
            score=0.9,
            reason="检索证据足够支撑回答",
            missing=[],
            source_count=1,
            top_sources=["制度.docx"],
        ),
    )

    assert answer.startswith("参考 未知制度.docx 的内容")
    assert "【依据提示】" in answer
    assert "不足以支持" not in answer


def test_rag_stream_uses_deepseek_planner_for_complex_requests(monkeypatch):
    openai_module = ModuleType("openai")
    FakeOpenAI.calls = []
    openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setenv("ANSWER_PLANNER", "auto")

    knowledge = RichKnowledgeAgent()
    usage_calls = []
    service = RagQaStreamService(RagQaDependencies(
        memory=FakeMemory(),
        knowledge_agent=knowledge,
        deepseek_api_key="present",
        record_token_usage=lambda **kwargs: usage_calls.append(kwargs),
    ))

    events = parse_sse(service.stream(
        "结合示例智库申报材料和教育发展三年行动计划，帮我梳理可申报方向并列出依据",
        "session_1",
        "user_1",
        display_message="结合示例智库申报材料和教育发展三年行动计划，帮我梳理可申报方向并列出依据",
    ))

    assert len(FakeOpenAI.calls) == 2
    content_chunks = [event["data"] for event in events if event["type"] == "content"]
    answer_deltas = [event["data"] for event in events if event["type"] == "answer_delta"]
    assert content_chunks[:2] == ["可结合示例智库申报.docx ", "和教育发展三年计划.docx 梳理申报方向。"]
    assert answer_deltas[:2] == content_chunks[:2]
    assert any(event["type"] == "answer_start" for event in events)
    assert any(event["type"] == "answer_done" for event in events)
    assert knowledge.payload["answer_intent"] == ANSWER_INTENT_COMPARE_SUMMARIZE
    assert knowledge.payload["knowledge_queries"][0].startswith("结合示例智库申报")
    assert events[-1]["audit_summary"]["planner_source"] == "llm"
    assert events[-1]["plan"]["planner_source"] == "llm"
    assert any(call["agent"] == "AnswerPlanner" and call["status"] == "success" for call in usage_calls)


def test_rag_stream_does_not_call_deepseek_planner_for_simple_requests(monkeypatch):
    openai_module = ModuleType("openai")
    openai_module.OpenAI = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("planner should not be called"))
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setenv("ANSWER_PLANNER", "auto")

    service = RagQaStreamService(RagQaDependencies(
        memory=FakeMemory(),
        knowledge_agent=EmptyKnowledgeAgent(),
        deepseek_api_key="present",
        record_token_usage=lambda **kwargs: None,
    ))

    events = parse_sse(service.stream(
        "制度依据是什么？",
        "session_1",
        "user_1",
        display_message="制度依据是什么？",
    ))

    assert events[-1]["type"] == "done"
    assert events[-1]["audit_summary"]["planner_source"] == "rules"


def test_open_ended_writing_support_can_answer_without_knowledge_sources(monkeypatch):
    openai_module = ModuleType("openai")
    StreamingOnlyOpenAI.calls = []
    openai_module.OpenAI = StreamingOnlyOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setenv("ANSWER_PLANNER", "rules")

    service = RagQaStreamService(RagQaDependencies(
        memory=FakeMemory(),
        knowledge_agent=EmptyKnowledgeAgent(),
        deepseek_api_key="present",
        record_token_usage=lambda **kwargs: None,
    ))

    events = parse_sse(service.stream(
        "帮我想几个报告题目",
        "session_1",
        "user_1",
        display_message="帮我想几个报告题目",
    ))

    assert len(StreamingOnlyOpenAI.calls) == 1
    assert events[-1]["type"] == "done"
    assert events[-1]["audit_summary"]["answer_intent"] == "open_ended"
    assert events[-1]["audit_summary"]["evidence_passed"] is True
    assert "当前未引用知识库资料" in events[-1]["answer"]
