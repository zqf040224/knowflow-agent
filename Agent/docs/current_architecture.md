# 当前架构说明

本文档描述当前生产主线。早期 `IntelligentRouter` 相关文件仍保留用于参考，但不再作为新功能接入点。

## 应用入口

后端入口是 `app.py`。它只负责加载环境变量、配置 Flask/CORS/cache、创建 `AppContext`、注册路由并保留 `python app.py --port ...` 启动方式。

`AppContext` 位于 `app_context.py`，集中管理认证、memory、上传、知识库、聊天容器、导出服务、后台管理服务和 `JobService` 等运行期依赖。`app_dependencies.py` 只负责启动时装配这些依赖并返回 context。路由层应通过 `context` 获取服务，不应重新创建全局单例。

路由按功能注册在 `routes/`：

- 页面与认证：登录、聊天页、后台页、认证 API。
- 聊天与查询：`/api/chat`、知识库搜索、结构化表格查询、legacy 非流式生成。
- 后台管理：用户、知识库文件、审计、健康状态。
- 上传/会话/导出/job：文件上传、会话、导出、后台任务查询。

## 聊天主链路

生产聊天唯一入口是：

```text
POST /api/chat
  -> routes/chat_routes.py: chat()
  -> context.chat_runtime().stream(...)
  -> ChatGraphRuntime
  -> TaskPlanner
  -> ToolOrchestrator
  -> ToolRegistry 中的工具
```

`ChatGraphRuntime` 负责请求准备、附件内容拼接、会话获取、任务规划和工具编排。`TaskPlanner` 是默认决策入口，输出工具步骤；`ToolOrchestrator` 执行工具并统一 SSE 事件。`IntentRouter` 保留为规则 fallback 和 `CHAT_RUNTIME=legacy` 旧链路，不再是默认主决策层。

当前工具映射：

| Tool | 底层服务 | 说明 |
| --- | --- | --- |
| `knowledge_qa` | `RagQaStreamService` | 知识库检索问答 |
| `draft_document` | `DocumentDraftStreamService` | 进入 `AgentOrchestrator` 做完整公文/材料生成 |
| `format_document` | `DocumentFormatStreamService` | 上传材料格式转换或公文格式化 |
| `identity_help` | `LightweightChatStreamService` | 身份/能力说明 |
| `clarify` | `LightweightChatStreamService` | 需求澄清 |
| `prepare_form_export` | `LightweightChatStreamService` | 生成报销表模板导出动作，等待用户确认 |
| `prepare_spreadsheet_transform` | `LightweightChatStreamService` | 生成表格处理动作，等待用户确认 |

### SSE 事件协议

所有生产聊天 stream service 都应同时遵守两层事件协议：

- 阶段事件：`thinking_start`、`thinking_done`、`answer_start`、`answer_delta`、`answer_done`、`run_done`，用于前端稳定展示“思考展开 -> 思考折叠 -> 正文流式输出”。
- 工具事件：`tool_plan`、`tool_call`、`tool_result`、`tool_confirm_required`，用于展示 TaskPlanner 与 ToolOrchestrator 的执行轨迹。
- 兼容事件：`think`、`content`、`done` 等旧事件继续保留，旧前端和测试仍以 `done` 作为完整结果事件。

新增或修改 stream service 时，推荐顺序是：

```text
start -> session -> route?
thinking_start -> tool_plan/tool_call -> think/plan/reasoning/reflection...
thinking_done -> answer_start -> answer_delta/content... -> answer_done
tool_result/tool_confirm_required? -> run_done -> done
```

`answer_delta` 与 `content` 当前会同时发送以兼容旧客户端；前端需要对两者做去重。`done` 必须仍包含完整 `answer/document/plan/actions/source_details/audit_summary`，并尽量保持为最终完整结果事件。

### 写作与智能客服边界

`doc_drafting` 负责生成新内容，`knowledge_qa` 负责基于资料回答问题。

- 进入 `doc_drafting`：用户明确要求写、起草、撰写、生成一篇/一份/一个报告、通知、请示、方案、材料、文稿、提纲、发言稿等可编辑文稿。
- 保留在 `knowledge_qa`：用户询问制度、标准、金额、流程、依据、文件位置、已有材料主要内容，或问“怎么写/模板/范文/需要哪些材料”这类写法咨询。
- 写作链路中，知识库是参考资料，不是通行证。检索不到本地资料时可以先生成通用初稿或框架，但应提示未引用本地资料。
- 智能客服链路中，事实、制度、金额、流程、依据类回答必须受证据约束。检索不到足够材料时应说明依据不足或追问，不应编造。
- 开放式轻量写作辅助（如想题目、列思路）可以在 `knowledge_qa` 中回答；若未命中知识库，应明确说明未引用知识库资料。

新增聊天工具时，按顺序修改：

1. `task_planner.py`：新增工具常量、fallback 规划规则和必要的 LLM plan 归一化。
2. `tool_runtime.py`：注册工具接口和执行策略。
3. `chat_container.py`：把底层 service 包装成 `ChatTool`。
4. 对应 `chat_*.py`：实现或复用 SSE stream，保持阶段事件与旧事件兼容。
5. `test_task_planner.py`、`test_tool_runtime.py` 和 `test_chat_runtime.py`：补规划、执行和主链路测试。

不要为新聊天能力新增并行的 `/api/chat` 路径，也不要从 Flask 路由直接调用 Agent。

## Agent 内层链路

`AgentOrchestrator` 是复杂公文/材料生成的内层编排器。它只在 `doc_drafting` 以及 legacy `/api/agent/generate` 中使用。

当前内层可走：

- `DocumentStreamRunner`：聊天流式生成主路径。
- `DocumentLinearRunner`：非流式运行。
- `DocumentGraphRunner`：`AGENT_ORCHESTRATOR=langgraph` 时启用的内部图运行器。

这些 runner 属于 Agent 内层实现，不是 `/api/chat` 的外层路由边界。

## 知识库、上传、导出与任务

知识库检索当前由 `KnowledgeAgent` 承担，使用 FAISS + BM25 + RRF，并按 `UserInfo` 做权限过滤。后台读写通过 `KnowledgeAdminReadService`、`KnowledgeAdminWriteService` 管理文件级状态、审计、重建和删除。

`knowledge_qa` 链路先做 AnswerPlanner、KnowledgeAgent 检索和 EvidenceGate。证据不足时直接返回澄清/依据不足答复；证据足够时模型正文按 chunk 真实流式输出，再做 AnswerVerifier。若校验后需要补充依据提示或替换为安全答复，`done.answer` 会携带最终版本，前端以最终 `done` 为准。

上传入口是 `/api/upload`：

- `mode=temp`：同步解析并作为本轮对话附件使用。
- `mode=knowledge`：提交后台任务入库，前端通过 `/api/jobs/<job_id>` 轮询。

导出入口仍是同步服务：

- `/api/export_docx`
- `/api/export_xlsx`
- `/api/export_reimbursement_xlsx`

后台任务由 `JobService` 管理，当前覆盖知识库上传入库和单文件 reindex。新增后台任务时，应优先复用 `JobService.submit(...)`，并通过 `/api/jobs/<job_id>` 暴露状态。

## Legacy 与实验资产

以下文件保留用于参考或演示，不是当前生产聊天主链路：

- `intelligent_router.py`
- `router_integration_demo.py`
- `ROUTER_ARCHITECTURE.md`
- `using.md` 中的早期 router 说明

新功能不要接入 `IntelligentRouter`。如果需要改变生产聊天分派，请改 `IntentRouter + ChatGraphRuntime + stream service`。

## 测试策略

主链路改动至少运行：

```bash
.venv311/bin/python -m py_compile app.py app_dependencies.py routes/*.py chat_runtime.py chat_container.py chat_architecture.py
.venv311/bin/python -m pytest test_chat_runtime.py test_chat_container.py test_chat_architecture.py test_app_route_service_wrappers.py
```

涉及上传、导出、任务或知识库边界时，再运行：

```bash
.venv311/bin/python -m pytest test_upload_service.py test_export_service.py test_runtime_query_service.py test_job_service.py
```
