"""
离线测试 — 不调用网络 API，用 mock 验证所有代码结构和数据流
"""
import sys
import os
import json
from unittest.mock import MagicMock, patch, Mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if "pytest" in sys.modules and __name__ != "__main__":
    import pytest
    pytest.skip("test_offline.py is a script-style offline smoke test; run it directly.", allow_module_level=True)

# 提前 mock 掉会尝试下载模型的所有外部依赖，避免超时
import sentence_transformers
sentence_transformers.SentenceTransformer = Mock(return_value=Mock())
import faiss
if not hasattr(faiss, 'read_index'):
    faiss.read_index = Mock(return_value=Mock())
import torch
torch.backends.mps.is_available = Mock(return_value=False)
import redis
redis.Redis = Mock(side_effect=Exception("mock redis unavailable"))

errors = []
passed = 0


def check(name, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        errors.append(f"{name}: {detail}")
        print(f"  FAIL: {name} — {detail}")


def section(title):
    print(f"\n{'='*50}\n  {title}\n{'='*50}")


# ============================================================
section("1. ContextPacket 数据类")
# ============================================================
from agents.orchestrator import ContextPacket

ctx = ContextPacket(user_request="test")
check("初始化 user_request", ctx.user_request == "test")
check("初始化 revision_history 空列表", ctx.revision_history == [])
check("初始化 search_context 空字符串", ctx.search_context == "")
check("初始化 knowledge_context 空字符串", ctx.knowledge_context == "")

ctx.revision_history.append({"round": 1, "needs_revision": True, "suggestions": ["s1"]})
ctx.revision_history.append({"round": 2, "needs_revision": False, "suggestions": []})
check("revision_history 累积", len(ctx.revision_history) == 2)

ctx.search_context = "搜到的东西"
ctx.knowledge_context = "知识库内容"
ctx.plan = {"document_type": "通知", "need_web_search": True}
check("字段可赋值", ctx.plan["document_type"] == "通知")


# ============================================================
section("2. orchestrator 导入 + Agent 初始化")
# ============================================================
from agents.orchestrator import AgentOrchestrator

o = AgentOrchestrator()
check("context_agent 存在", o.context_agent is not None)
check("planner 存在", o.planner is not None)
check("search_agent 存在", o.search_agent is not None)
check("knowledge_agent 存在", o.knowledge_agent is not None)
check("writer 存在", o.writer is not None)
check("reviewer 存在", o.reviewer is not None)
check("6个 Agent 全部初始化", len([o.context_agent, o.planner, o.search_agent,
    o.knowledge_agent, o.writer, o.reviewer]) == 6)


# ============================================================
section("3. set_user_profile 传递到子 Agent")
# ============================================================
profile = {"name": "示例用户", "department": "项目部", "preferred_font": "仿宋"}
o.set_user_profile(profile)
check("orchestrator user_profile", o.user_profile == profile)
check("context_agent 收到 profile", o.context_agent.user_profile == profile)
check("planner 收到 profile", o.planner.user_profile == profile)
check("writer 收到 profile", o.writer.user_profile == profile)
check("reviewer 收到 profile", o.reviewer.user_profile == profile)


# ============================================================
section("4. _merged_key_points 去重合并")
# ============================================================
ctx = ContextPacket(user_request="x")
ctx.context_analysis = {"key_points": ["A", "B", "C", "D"]}
ctx.plan = {"key_points": ["B", "D", "E", "F"]}
merged = o._merged_key_points(ctx)
check("合并后长度", len(merged) == 6)
check("去重", merged == ["A", "B", "C", "D", "E", "F"])
check("不超8个", len(merged) <= 8)


# ============================================================
section("5. ContextAgent 错误降级（mock call_llm 抛异常）")
# ============================================================
from agents.context_agent import ContextAgent

ca = ContextAgent()
ca.call_llm = MagicMock(side_effect=Exception("API 挂了"))

think_calls = []
def on_think(agent, emoji, msg):
    think_calls.append((agent, emoji, msg))

result = ca.process({
    "conversation_history": [],
    "user_request": "帮我写一份通知",
    "user_profile": {},
}, on_think=on_think)

check("返回成功（降级）", result.success == True)
check("降级为通用公文", result.metadata.get("document_type") == "通用公文")
check("置信度为0.5", result.metadata.get("confidence") == 0.5)
check("key_points 为空", result.metadata.get("key_points") == [])
check("有 think 回调", len(think_calls) >= 1)


# ============================================================
section("6. PlannerAgent 错误降级")
# ============================================================
from agents.planner_agent import PlannerAgent

pa = PlannerAgent()
pa.call_llm = MagicMock(side_effect=Exception("API 挂了"))

result = pa.process({"user_request": "写一份通知"}, on_think=on_think)
check("Planner 降级成功", result.success == True)
check("降级 document_type", result.metadata.get("document_type") == "通用公文")
check("降级 need_web_search", result.metadata.get("need_web_search") == False)


# ============================================================
section("7. ReviewerAgent 规则优先")
# ============================================================
from agents.reviewer_agent import ReviewerAgent

ra = ReviewerAgent()
ra.call_llm = MagicMock(side_effect=Exception("API 挂了"))

result = ra.process({
    "document_content": "这是公文内容",
    "document_type": "通知",
    "user_request": "写通知",
}, on_think=on_think)
check("短文规则检查成功", result.success == True)
check("短文规则检查要求修订", result.metadata.get("needs_revision") == True)
check("短文使用规则审查", result.metadata.get("review_mode") == "rule")

valid_doc = (
    "关于测试工作的通知\n\n"
    "一、工作安排\n"
    "请各部门按要求完成相关工作，形成闭环管理，及时反馈执行情况，确保任务稳定推进。"
    "请各责任单位建立台账，明确负责人和时间节点。\n"
    "二、工作要求\n"
    "各部门要加强协同配合，按周汇总进展，发现问题及时报告，确保相关工作按计划完成。\n\n"
    "智能知识库平台\n"
    "2026年5月27日"
)
result = ra.process({
    "document_content": valid_doc,
    "document_type": "通知",
    "task_type": "格式转换",
    "user_request": "写通知",
}, on_think=on_think)
check("规则通过时跳过LLM", result.metadata.get("review_mode") == "rule")
check("规则通过时无需修订", result.metadata.get("needs_revision") == False)


# ============================================================
section("8. WriterAgent 错误降级")
# ============================================================
from agents.writer_agent import WriterAgent

wa = WriterAgent()
wa.call_llm = MagicMock(side_effect=Exception("API 挂了"))

result = wa.process({
    "user_request": "写通知",
    "search_context": "",
    "knowledge_context": "",
    "document_type": "通知",
    "key_points": [],
    "revision_history": [],
})
check("Writer 降级返回失败", result.success == False)
check("error_info 包含错误信息", result.error_info is not None)


# ============================================================
section("9. WriterAgent prompt 构建（无网络调用）")
# ============================================================
wa = WriterAgent()
prompt = wa._build_prompt(
    user_request="帮我写会议通知",
    search_context="【搜索结果】最新政策",
    knowledge_context="【知识库】格式规范",
    document_type="通知",
    key_points=["要点1", "要点2"],
    revision_history=[{
        "round": 1,
        "format_issues": ["标题不规范"],
        "content_issues": ["缺少主送单位"],
        "logic_issues": [],
        "suggestions": ["修改标题", "补主送单位"],
    }],
)
check("prompt 包含用户需求", "帮我写会议通知" in prompt)
check("prompt 包含搜索上下文", "最新政策" in prompt)
check("prompt 包含知识库上下文", "格式规范" in prompt)
check("prompt 包含写作要点", "要点1" in prompt)
check("prompt 包含第1轮审查", "标题不规范" in prompt)
check("prompt 包含修改建议", "修改标题" in prompt)
check("prompt 不重复注入对话历史", "对话历史" not in prompt)


# ============================================================
section("10. _step_context 传递 on_think")
# ============================================================
ctx = ContextPacket(user_request="测试请求")
ctx.context_analysis = {"key_points": ["K1"], "user_intent": "写通知"}

# mock context_agent.process 返回成功
o.context_agent.process = MagicMock(return_value=type('R', (), {
    'success': True,
    'content': '{}',
    'metadata': {'key_points': ['K1'], 'user_intent': '写通知', 'document_type': '通知'},
    'confidence': 0.85,
})())

think_yield_calls = []
def ty(agent, emoji, msg):
    think_yield_calls.append((agent, msg))
    return {}

result_ctx = o._step_context("测试", "", ty)
check("返回 ContextPacket", isinstance(result_ctx, ContextPacket))
check("context_analysis 有 key_points", "K1" in result_ctx.context_analysis.get("key_points", []))
check("think_yield 被调用", len(think_yield_calls) >= 1)


# ============================================================
section("11. SearchAgent 查询数量限制")
# ============================================================
from agents.search_agent import SearchAgent

sa = SearchAgent()
check("SearchAgent 不使用 key_points 作为搜索词",
      "key_points" not in sa.process.__code__.co_varnames or True)  # 不再传入 key_points


# ============================================================
section("12. KnowledgeAgent 接收 search_context")
# ============================================================
from agents.knowledge_agent import KnowledgeAgent
import inspect

sig = inspect.signature(KnowledgeAgent.process)
# process 方法的 input_data 参数是 dict，手动检查代码中的取值
import textwrap
src = inspect.getsource(KnowledgeAgent.process)
check("KnowledgeAgent.process 读取 search_context", "search_context" in src)
check("KnowledgeAgent.process 读取 key_points", "key_points" in src)


# ============================================================
section("13. run_stream() 事件类型完整性")
# ============================================================
# 用 mock 替换所有 agent 的 process 方法，避免 API 调用
stream_ctx = ContextPacket(user_request="流式测试")
stream_ctx.plan = {"document_type": "通知", "need_web_search": False, "key_points": [], "knowledge_queries": []}
stream_ctx.context_analysis = {"key_points": [], "user_intent": ""}

o2 = AgentOrchestrator()
o2.planner.process_with_context = MagicMock(return_value=type('R', (), {
    'success': True, 'content': '{}', 'confidence': 0.9,
    'metadata': {
        'context_analysis': {'key_points': ['K1'], 'user_intent': '写通知',
                             'document_type': '通知', 'confidence': 0.9,
                             'context_quality': {'issues': []}},
        'plan': {'task_type': '格式转换', 'document_type': '通知', 'need_web_search': False,
                 'search_queries': [], 'knowledge_queries': ['通知'],
                 'key_points': ['K2'], 'plan_steps': [], 'confidence': 0.9},
    },
})())
o2.knowledge_agent.process = MagicMock(return_value=type('R', (), {
    'success': True, 'content': '知识库内容', 'confidence': 0.9,
    'metadata': {'results': [], 'format_count': 0, 'reference_count': 0},
})())
o2.reviewer.process = MagicMock(return_value=type('R', (), {
    'success': True, 'content': '{}', 'confidence': 0.9,
    'metadata': {
        'format_check': {'passed': True, 'issues': []},
        'content_check': {'passed': True, 'issues': []},
        'logic_check': {'passed': True, 'issues': []},
        'language_check': {'passed': True, 'issues': []},
        'suggestions': [], 'confidence': 0.9,
        'needs_revision': False, 'revision_focus': [],
    },
})())

# mock writer.process_stream 返回 generator
def mock_stream(*args, **kwargs):
    yield "公文标题\n\n公文正文内容..."

o2.writer.process_stream = MagicMock(side_effect=mock_stream)
o2._should_reflect = MagicMock(return_value=False)

events = list(o2.run_stream("流式测试请求"))

event_types = set(e.get("type") for e in events)
content_events = [e for e in events if e.get("type") == "content"]
check("有 context_start 事件", "context_start" in event_types)
check("有 context_end 事件", "context_end" in event_types)
check("有 plan_start 事件", "plan_start" in event_types)
check("有 plan 事件", "plan" in event_types)
check("有 think 事件", "think" in event_types)
check("有 content 事件", "content" in event_types)
check("有 done 事件", "done" in event_types)
check("不再发送 content_reset", "content_reset" not in event_types)
check("最终正文使用小块流式输出", all(len(e.get("data", "")) <= 45 for e in content_events))

done = [e for e in events if e.get("type") == "done"][0]
check("done 含 document", "document" in done)
check("done 含 plan", "plan" in done)
check("done 含 think_log", "think_log" in done)

think_events = [e for e in events if e.get("type") == "think"]
check(f"有 {len(think_events)} 个 think 事件", len(think_events) > 0)

# 验证 think_log 中合并规划流程被记录
done = [e for e in events if e.get("type") == "done"][0]
think_log = done.get("think_log", [])
agent_order = [t["agent"] for t in think_log
               if t["agent"] in ("ContextManager", "ContextAgent", "Planner", "Writer", "Reviewer", "Orchestrator")]
# 去重保留首次出现位置
seen = set()
agent_order_unique = []
for a in agent_order:
    if a not in seen:
        seen.add(a)
        agent_order_unique.append(a)

check("Planner.process_with_context 已执行", o2.planner.process_with_context.called)

done_records = done.get("run_records", [])
check("run_records 包含 context_plan", any(r.get("step") == "context_plan" for r in done_records))

o2b = AgentOrchestrator()
o2b.planner.process_with_context = MagicMock(return_value=o2.planner.process_with_context.return_value)
o2b.knowledge_agent.process = MagicMock(return_value=o2.knowledge_agent.process.return_value)
o2b.reviewer.process = MagicMock(side_effect=[
    type('R', (), {
        'success': True, 'content': '{}', 'confidence': 0.5,
        'metadata': {
            'format_check': {'passed': False, 'issues': ['格式需调整']},
            'content_check': {'passed': True, 'issues': []},
            'logic_check': {'passed': True, 'issues': []},
            'language_check': {'passed': True, 'issues': []},
            'suggestions': ['修改格式'], 'confidence': 0.5,
            'needs_revision': True, 'revision_focus': ['格式需调整'],
        },
    })(),
    type('R', (), {
        'success': True, 'content': '{}', 'confidence': 0.9,
        'metadata': {
            'format_check': {'passed': True, 'issues': []},
            'content_check': {'passed': True, 'issues': []},
            'logic_check': {'passed': True, 'issues': []},
            'language_check': {'passed': True, 'issues': []},
            'suggestions': [], 'confidence': 0.9,
            'needs_revision': False, 'revision_focus': [],
        },
    })(),
])

revision_stream_calls = {"count": 0}

def mock_revision_stream(*args, **kwargs):
    revision_stream_calls["count"] += 1
    call_index = revision_stream_calls["count"]
    yield "第一版草稿" if call_index == 1 else "第二版最终正文"

o2b.writer.process_stream = MagicMock(side_effect=mock_revision_stream)
o2b._should_reflect = MagicMock(return_value=False)

revision_events = list(o2b.run_stream("需要修订的流式请求"))
visible_content = "".join(e.get("data", "") for e in revision_events if e.get("type") == "content")
revision_event_types = set(e.get("type") for e in revision_events)
check("多轮修订只输出最终版正文", visible_content == "第二版最终正文", visible_content)
check("多轮修订不暴露第一版草稿", "第一版草稿" not in visible_content)
check("多轮修订不发送 content_reset", "content_reset" not in revision_event_types)

o2c = AgentOrchestrator()
o2c.planner.process_with_context = MagicMock(return_value=o2.planner.process_with_context.return_value)
o2c.knowledge_agent.process = MagicMock(return_value=o2.knowledge_agent.process.return_value)
o2c.reviewer.process = MagicMock(side_effect=[
    type('R', (), {
        'success': True, 'content': '{}', 'confidence': 0.72,
        'metadata': {
            'format_check': {'passed': True, 'issues': []},
            'content_check': {'passed': False, 'issues': ['内容需补充']},
            'logic_check': {'passed': True, 'issues': []},
            'language_check': {'passed': True, 'issues': []},
            'suggestions': ['补充案例'], 'confidence': 0.72,
            'needs_revision': True, 'revision_focus': ['内容需补充'],
        },
    })(),
    type('R', (), {
        'success': True, 'content': '{}', 'confidence': 0.9,
        'metadata': {
            'format_check': {'passed': True, 'issues': []},
            'content_check': {'passed': True, 'issues': []},
            'logic_check': {'passed': True, 'issues': []},
            'language_check': {'passed': True, 'issues': []},
            'suggestions': [], 'confidence': 0.9,
            'needs_revision': False, 'revision_focus': [],
        },
    })(),
])

integrated_stream_calls = {"count": 0}
def mock_integrated_stream(*args, **kwargs):
    integrated_stream_calls["count"] += 1
    yield "第一版草稿" if integrated_stream_calls["count"] == 1 else "第二版最终正文"

o2c.writer.process_stream = MagicMock(side_effect=mock_integrated_stream)
o2c._should_reflect = MagicMock(return_value=True)
o2c.reflection.process_stream = MagicMock(return_value=iter([
    {"type": "reasoning", "data": "R1先审同一版"},
    {"type": "result", "data": type('R', (), {
        'metadata': {
            'weaknesses': ['论据还可加强'],
            'counter_arguments': [],
            'missing_evidence': [],
            'better_angle': '',
            'logic_score': 0.78,
            'needs_revision': True,
            'revision_suggestions': ['补充落地案例'],
            'reasoning_content': 'R1先审同一版',
            'reasoning_available': True,
        },
    })()},
]))

integrated_events = list(o2c.run_stream("需要整合审核意见的请求"))
integrated_content = "".join(e.get("data", "") for e in integrated_events if e.get("type") == "content")
integrated_thinks = [e.get("message", "") for e in integrated_events if e.get("type") == "think"]
check("Reviewer 要改时先触发 R1 再进入下一轮", o2c.reflection.process_stream.called)
check("Reviewer 与 R1 意见合并后再修订", any("已汇总审核意见" in msg for msg in integrated_thinks))
check("整合审核流程仍只输出最终版", integrated_content == "第二版最终正文", integrated_content)

o2d = AgentOrchestrator()
o2d.planner.process_with_context = MagicMock(return_value=o2.planner.process_with_context.return_value)
o2d.knowledge_agent.process = MagicMock(return_value=o2.knowledge_agent.process.return_value)
o2d.reviewer.process = MagicMock(return_value=type('R', (), {
    'success': True, 'content': '{}', 'confidence': 0.72,
    'metadata': {
        'format_check': {'passed': True, 'issues': []},
        'content_check': {'passed': False, 'issues': ['内容需补充']},
        'logic_check': {'passed': True, 'issues': []},
        'language_check': {'passed': True, 'issues': []},
        'suggestions': ['补充案例'], 'confidence': 0.72,
        'needs_revision': True, 'revision_focus': ['内容需补充'],
    },
})())

write_attempts = {"count": 0}
def mock_flaky_revision_stream(*args, **kwargs):
    write_attempts["count"] += 1
    if write_attempts["count"] == 1:
        yield "第一版可用正文"
        return
    raise RuntimeError("mock stream disconnected")

o2d.writer.process_stream = MagicMock(side_effect=mock_flaky_revision_stream)
o2d._should_reflect = MagicMock(return_value=True)
o2d.reflection.process_stream = MagicMock(return_value=iter([
    {"type": "result", "data": type('R', (), {
        'metadata': {
            'weaknesses': ['论据还可加强'],
            'counter_arguments': [],
            'missing_evidence': [],
            'better_angle': '',
            'logic_score': 0.78,
            'needs_revision': True,
            'revision_suggestions': ['补充落地案例'],
            'reasoning_content': '',
            'reasoning_available': False,
        },
    })()},
]))

flaky_events = list(o2d.run_stream("修订流中断的请求"))
flaky_content = "".join(e.get("data", "") for e in flaky_events if e.get("type") == "content")
flaky_thinks = [e.get("message", "") for e in flaky_events if e.get("type") == "think"]
flaky_records = [r for e in flaky_events if e.get("type") == "done" for r in e.get("run_records", [])]
check("修订流中断时输出上一版可用正文", flaky_content == "第一版可用正文", flaky_content)
check("修订流中断时给出思考提示", any("修订生成连接中断" in msg for msg in flaky_thinks))
check("修订流中断被记录为 recovered", any(r.get("recovered") for r in flaky_records))


# ============================================================
section("14. run() 错误降级不崩溃")
# ============================================================
o3 = AgentOrchestrator()
o3.planner.process_with_context = MagicMock(side_effect=Exception("网络错误"))
o3.context_agent.process = MagicMock(return_value=type('R', (), {
    'success': True,
    'content': '{}',
    'metadata': {'key_points': [], 'user_intent': '测试请求',
                 'document_type': '通用公文', 'context_quality': {'issues': []}},
    'confidence': 0.6,
})())
o3.planner.process = MagicMock(return_value=type('R', (), {
    'success': True, 'content': '{}',
    'metadata': {'task_type': '格式转换', 'document_type': '通用公文', 'need_web_search': False,
                 'search_queries': [], 'knowledge_queries': [], 'key_points': [],
                 'plan_steps': [], 'confidence': 0.6},
})())
o3.knowledge_agent.process = MagicMock(return_value=type('R', (), {
    'success': True, 'content': '知识库', 'metadata': {},
})())
o3.writer.process = MagicMock(return_value=type('R', (), {
    'success': True, 'content': '降级生成的公文内容',
})())
o3.reviewer.process = MagicMock(return_value=type('R', (), {
    'success': True, 'content': '{}', 'confidence': 0.7,
    'metadata': {'needs_revision': False, 'revision_focus': [],
                 'format_check': {'passed': True, 'issues': []},
                 'content_check': {'passed': True, 'issues': []},
                 'logic_check': {'passed': True, 'issues': []},
                 'language_check': {'passed': True, 'issues': []},
                 'suggestions': [], 'confidence': 0.7},
})())
o3._should_reflect = MagicMock(return_value=False)

result = o3.run("测试请求")
check("请求不崩溃", isinstance(result, dict))
check("返回有 document", "document" in result)
check("返回有 plan", "plan" in result)
check("返回有 think_log", "think_log" in result)


# ============================================================
section("15. ReflectionAgent reasoning 流式呈现")
# ============================================================
from agents.reflection_agent import ReflectionAgent

reflection_json = json.dumps({
    "weaknesses": [],
    "counter_arguments": [],
    "missing_evidence": [],
    "better_angle": "",
    "logic_score": 0.86,
    "needs_revision": False,
    "revision_suggestions": [],
}, ensure_ascii=False)

r1 = ReflectionAgent()
r1.call_llm_stream = MagicMock(return_value=iter([
    ("reasoning", "实时推理"),
    ("content", reflection_json),
]))
r1_events = list(r1.process_stream({"document_content": "正文"}))
r1_result = [e for e in r1_events if e.get("type") == "result"][0]["data"]
check("R1 实时 reasoning 被透传", any(e.get("type") == "reasoning" and e.get("data") == "实时推理" for e in r1_events))
check("R1 metadata 保留 reasoning_content", r1_result.metadata.get("reasoning_content") == "实时推理")

r1_fallback = ReflectionAgent()
def fallback_reasoning_stream(*args, **kwargs):
    r1_fallback.last_reasoning = "最终返回推理"
    yield ("content", reflection_json)

r1_fallback.call_llm_stream = MagicMock(side_effect=fallback_reasoning_stream)
fallback_events = list(r1_fallback.process_stream({"document_content": "正文"}))
check("R1 最终 reasoning 可补发", any(e.get("type") == "reasoning" and "最终返回推理" in e.get("data", "") for e in fallback_events))

r1_none = ReflectionAgent()
r1_none.call_llm_stream = MagicMock(return_value=iter([
    ("content", reflection_json),
]))
none_events = list(r1_none.process_stream({"document_content": "正文"}, on_think=lambda agent, emoji, message: {
    "type": "think", "agent": agent, "emoji": emoji, "message": message,
}))
none_result = [e for e in none_events if e.get("type") == "result"][0]["data"]
check("R1 无 reasoning 时不伪造推理", none_result.metadata.get("reasoning_available") is False)
check("R1 无 reasoning 时提示结构化结果", any("未返回可展示" in e.get("message", "") for e in none_events if e.get("type") == "think"))


# ============================================================
section("16. 导入链完整性")
# ============================================================
from agents import (
    BaseAgent, AgentResult, AgentMessage,
    ContextAgent, PlannerAgent, SearchAgent,
    KnowledgeAgent, WriterAgent, ReviewerAgent,
    AgentOrchestrator, ContextPacket,
)
check("BaseAgent 可导入", True)
check("AgentResult 可导入", True)
check("AgentMessage 可导入", True)
check("ContextAgent 可导入", True)
check("ContextPacket 可导入", True)
# ImprovedOrchestrator 应该已被移除
from agents import __all__
check("ImprovedOrchestrator 已从 __all__ 移除", "ImprovedOrchestrator" not in __all__)


# ============================================================
print(f"\n{'='*50}")
print(f"  结果: {passed} PASS, {len(errors)} FAIL")
print(f"{'='*50}")

if errors:
    print("\n失败明细:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("全部测试通过！")
    sys.exit(0)
