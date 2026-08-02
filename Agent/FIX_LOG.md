# Agent 代码修复日志

> 日期: 2026-04-24 | 状态: 全部修复已完成，70/70 测试通过

---

## 一、修复总览

```
修复前: User → Planner → Search → Knowledge → Writer → Reviewer
         ↑ call_llm缺失崩溃   ↑ key_points丢弃  ↑ 各自孤立  ↑ 反馈截断

修复后: User → ContextAgent → Planner → Search → Knowledge → Writer ⇄ Reviewer
         ↑ 降级不崩溃    ↑ 上下文传递  ↑ 串行共享  ↑ 全量反馈+累积历史
```

---

## 二、所有已修改文件

| 文件 | 改动类型 |
|------|---------|
| `agents/base_agent.py` | 重命名 call_llm_stream→call_llm，移除 logging.basicConfig，修复硬编码路径 |
| `agents/orchestrator.py` | 新增 ContextPacket，集成 ContextAgent，Search→Knowledge 串行化，修复 dir() 检查，_step_context 降级 |
| `agents/writer_agent.py` | use_context=False，prompt 注入 key_points + revision_history，LLM 失败降级 |
| `agents/knowledge_agent.py` | 接收 search_context 和 key_points 优化检索 |
| `agents/search_agent.py` | 限制查询数≤3，_search_local 复用 KnowledgeAgent 实例 |
| `agents/context_agent.py` | try 扩到 call_llm 外层，LLM 失败降级 |
| `agents/planner_agent.py` | try 扩到 call_llm 外层，LLM 失败降级 |
| `agents/reviewer_agent.py` | try 扩到 call_llm 外层，修复 response_text 未绑定 |
| `agents/__init__.py` | 移除 ImprovedOrchestrator，新增 ContextPacket |
| `agents/improved_orchestrator.py` | **已删除** |
| `app.py` | 移除重复 add_message，修复硬编码路径，传递 user_profile |
| `test_agent_bugs.py` | 修复 run_stream API 调用，补全 MockMemory |
| `test_offline.py` | **新增** — 70 个离线测试用例 |
| `.env` | 更新为真实 API key |

---

## 三、数据流架构

```
ContextPacket 在 Agent 之间传递:
┌─────────────────────────────────────────────────────────┐
│ user_request     ← 用户原始请求                           │
│ context_analysis ← ContextAgent 输出 (key_points, intent) │
│ plan            ← PlannerAgent 输出 (doc_type, queries)   │
│ search_context  ← SearchAgent 输出 (联网搜索结果)          │
│ knowledge_context ← KnowledgeAgent 输出 (知识库参考)        │
│ revision_history ← 每轮 Reviewer 完整反馈累积              │
│ user_profile    ← 用户画像 (字体/字号/风格)                │
└─────────────────────────────────────────────────────────┘
```

---

## 四、待办（下次继续）

- [ ] `run_stream()` 中 sub-agent 的 think 事件未转发到 SSE 流（仅存在 think_log 中）
- [ ] `run_stream()` 中的 think_yield 回调与生成器混用，用法脆弱，建议重构
- [ ] 用真实 API key 跑一次端到端测试（需网络环境）
- [ ] `memory_v2.py` 动态 SQL 拼接建议改为 ORM 或参数化列名
- [ ] `cache.py` 的 Redis 连接失败时静默降级确认工作正常
- [ ] `KnowledgeAgent` 模型加载耗时优化（离线缓存 / 预加载）

---

## 五、测试命令

```bash
# 离线测试（无需网络，mock API）
python3 test_offline.py

# Bug 回归测试（需网络）
python3 test_agent_bugs.py

# 基础 Agent 测试（需网络）
python3 test_base_agent.py

# 流水线连贯性测试（需网络）
python3 test_pipeline_coherence.py
```

---

## 六、关键决策记录

1. **修复 orchestrator.py 而非切换到 improved_orchestrator.py** — 后者缺少 run_stream()，且本身有问题
2. **ContextAgent 失败降级不阻塞流程** — 返回空上下文继续执行
3. **Search 和 Knowledge 串行而非并行** — 搜索结果可优化知识库检索
4. **Writer 禁用自动上下文注入** — 避免 orchestrator 构建的 prompt 被重复注入对话历史
