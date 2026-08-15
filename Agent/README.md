# KnowFlow Agent

智能知识库与公文写作助手。当前后端是 Flask 应用，提供登录、聊天、知识库检索、文件上传入库、后台管理、导出和轻量后台任务能力。

## Quick Start

```bash
cd Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py --port 5003
```

访问：

- 登录页：`http://localhost:5003/login`
- 聊天页：`http://localhost:5003/chat`
- 管理后台：`http://localhost:5003/admin`
- 健康检查：`http://localhost:5003/api/health`

生产环境先复制 `.env.example` 为 `.env`，并设置：

```text
APP_ENV=production
LANGGRAPH_CHECKPOINTER_BACKEND=postgres
LANGGRAPH_RUN_LEDGER_BACKEND=postgres
LANGGRAPH_POSTGRES_DSN=postgresql://user:password@host:5432/database
WEB_CONCURRENCY=4
```

直接部署时先执行独立迁移，再通过安全启动器拉起 Gunicorn：

```bash
python scripts/setup_langgraph_checkpointer.py
python scripts/start_server.py
```

也可以使用生产 Compose overlay：

```bash
docker compose -f docker-compose.yml -f docker-compose.production.yml up --build
```

`scripts/start_server.py` 会在导入 Flask 前验证持久化配置：开发 SQLite
固定单进程，生产默认 4 worker、每 worker 1–2 个 PostgreSQL
连接，且总连接上限为 8。不要绕过该启动器直接以多进程连接 SQLite。
生产 Compose 另有一个清理进程；它只在每次清理扫描期间使用 1 个维护
连接，扫描结束立即关闭连接池。
但当前 `JobService` 仍使用本进程线程池：若同一部署启用后台任务，需先引入
跨进程队列或完整的 owner/heartbeat 调度治理；本次 4 worker 保证不包含该旧任务队列。

## Configuration

常用环境变量：

| 变量 | 说明 |
| --- | --- |
| `JWT_SECRET` | 登录 token 签名密钥 |
| `DEEPSEEK_API_KEY` | 生成与问答模型调用 |
| `BOCHA_API_KEY` | 联网搜索能力 |
| `CORS_ORIGINS` | 允许的跨域来源，逗号分隔 |
| `COOKIE_SECURE` | 是否设置 secure cookie |
| `JOB_WORKERS` | 本地后台任务线程数，默认 `2` |
| `CHAT_RUNTIME` | 聊天运行时：`langgraph/graph/on/true/1` 为主图，`planner/task_planner/off/false/0` 为 Planner 回滚，`legacy/pipeline` 为 IntentRouter 回滚 |
| `AGENT_ORCHESTRATOR` | 文档编排：默认 `langgraph`，`linear` 为发布周期内回滚通道 |
| `APP_ENV` | 仅支持 `development` 或 `production` |
| `LANGGRAPH_CHECKPOINTER_BACKEND` | 开发用 `sqlite`，生产必须用 `postgres` |
| `LANGGRAPH_RUN_LEDGER_BACKEND` | 运行账本后端；默认跟随 Checkpointer，生产必须为 `postgres` |
| `LANGGRAPH_SQLITE_PATH` | 本地 SQLite checkpoint 路径 |
| `LANGGRAPH_RUN_LEDGER_SQLITE_PATH` | 可选的本地运行账本路径，默认与 checkpoint 同目录 |
| `LANGGRAPH_POSTGRES_DSN` | 生产 PostgreSQL DSN，不可用时启动失败 |
| `LANGGRAPH_DURABILITY` | 普通节点默认 `async` |
| `LANGGRAPH_STRICT_MSGPACK` | 严格 checkpoint 序列化；生产必须为 `true` |
| `LANGGRAPH_CHECKPOINT_RETENTION_DAYS` | 运行和 checkpoint 保留天数，默认 `90` |
| `LANGGRAPH_POOL_MIN_SIZE` / `LANGGRAPH_POOL_MAX_SIZE` | 每 Worker PostgreSQL 连接池范围，最大不得超过 `2` |
| `WEB_CONCURRENCY` | Gunicorn Worker 数；开发默认 `1`，生产默认 `4` |
| `LANGGRAPH_CLEANUP_INTERVAL_SECONDS` | 生产清理成功后的扫描间隔，默认 `21600`（6 小时） |
| `LANGGRAPH_CLEANUP_RETRY_SECONDS` | 清理失败后的重试间隔，默认 `60` |
| `LANGGRAPH_CLEANUP_MAX_CONSECUTIVE_FAILURES` | 连续失败达到该值后退出并由容器重启，默认 `3` |
| `LANGGRAPH_CLEANUP_HEALTH_MAX_AGE_SECONDS` | 清理健康记录允许的最大年龄，默认 `25200`（7 小时） |
| `ENABLE_LLM_INTENT_CLASSIFIER` | 是否启用 LLM 意图分类兜底 |

密钥不要提交到仓库，使用 `.env` 本地配置。
未知运行模式、生产 PostgreSQL 不可用或连接预算超限都会在启动阶段失败，
不会静默降级。

## Architecture

应用入口保持轻量：

```text
app.py
  -> create_app_context()
  -> register_routes(app, context)
  -> python app.py --port ...
```

核心模块：

| 模块 | 职责 |
| --- | --- |
| `app.py` | Flask app 创建、CORS/cache hook、路由注册、启动逻辑 |
| `app_dependencies.py` | 启动时依赖装配，返回 `AppContext` |
| `app_context.py` | 运行期依赖上下文、service factory、跨路由 helper |
| `routes/` | HTTP route wrapper，按页面、认证、聊天、后台、上传、导出、job 分区 |
| `chat_runtime.py` | TaskPlan、StateGraph、工具节点、SSE 提交闸门与恢复运行时 |
| `chat_container.py` | 聊天 stream service 的懒加载装配 |
| `graph_persistence.py` | Checkpointer、graph_runs/effects、删除屏障与保留清理 |
| `graph_artifacts.py` | 附件正文快照和受控引用 |
| `graph_state_validation.py` | checkpoint 前的严格 JSON State 校验 |
| `job_service.py` | SQLite 任务状态 + 本地 `ThreadPoolExecutor` |
| `upload_service.py` | 临时上传与知识库上传入库包装 |

更多内部边界见 `docs/current_architecture.md`。

## Chat Mainline

生产聊天唯一主入口：

```text
POST /api/chat
  -> routes/chat_routes.py: chat()
  -> context.chat_runtime().stream_http(...)
  -> ChatGraphRuntime
  -> prepare -> plan_tools -> select_step
  -> tool-specific node -> collect_result
  -> select_step / finalize / error_terminal
```

`stream_http()` 会在建立 SSE Response 前同步预占 `run_id` 和幂等身份；并发
payload 冲突会以 HTTP 409 返回，而不是在 200 流中途失败。

`draft_document` 工具边界是父图可见的 LangGraph 子图；文档内部的
`context_plan -> retrieval -> write -> review -> reflection? -> decide -> write/finalize`
共用父图的 saver 和 `thread_id`，并使用独立 `document` checkpoint namespace。
任一阶段失败进入 `error_terminal`；`max_revisions` 是 `finalize` 输出的质量
状态，不是独立成功节点。

运行恢复接口：

```text
GET  /api/chat/runs/<run_id>
POST /api/chat/runs/<run_id>/recover
POST /api/chat/runs/<run_id>/resume
```

`resume` 接口、归属校验和 `Command(resume=...)` 恢复底座已就绪。
当前表格/导出类工具仍只是“准备动作，由前端另行确认”，并没有在图内
执行不可逆外部副作，因此本发布不会为它们伪造 `interrupt()`。后续接入
真实 executor 节点时，再在副作之前加入 HITL 中断。
恢复窗口为 7 天；仍处于 `running` 的运行只有在租约超过安全阈值后才允许
其他 Worker 原子接管。

PostgreSQL 首次建表必须作为独立迁移任务执行，不在 Gunicorn worker
启动时运行。下面第二条命令是手工单次清理，适合运维排查：

```bash
python scripts/setup_langgraph_checkpointer.py
python scripts/cleanup_langgraph_runs.py
```

生产 Compose 会在迁移成功后同时启动 `langgraph-cleanup` 常驻服务：默认每
6 小时处理待删除 tombstone、90 天过期 checkpoint/run 和附件；失败时每
60 秒重试，连续 3 次失败退出并由 Compose 重启。容器健康检查读取原子更新
的 `/tmp/langgraph-cleanup-health.json`，失败或超过 7 小时未更新即不健康。

会话删除会先持久化 `(user_id, session_id)` 永久屏障，再枚举和清理已有
run，因此并发到达的新 run 不能逃过删除。已删除的 `session_id` 不得复用；
用户再次新建会话时必须生成新的 ID。屏障按用户隔离，同名 session 不会影响
其他用户。

工具节点与底层服务映射：

| Tool | Handler |
| --- | --- |
| `knowledge_qa` | `RagQaStreamService` |
| `draft_document` | `DocumentDraftStreamService -> AgentOrchestrator` |
| `format_document` | `DocumentFormatStreamService` |
| `prepare_form_export` / `prepare_spreadsheet_transform` | `LightweightChatStreamService`，只准备待确认动作 |
| `identity_help` / `clarify` | `LightweightChatStreamService` |

新增聊天能力优先走 `TaskPlanner + ChatGraphRuntime + tool node`，不要在 Flask route 中直接调用 Agent，也不要接入历史 `IntelligentRouter`。

## Background Jobs

当前异步化范围：

- 管理后台单文件 reindex：`POST /api/admin/knowledge-files/<content_hash>/reindex`
- 聊天页上传入库：`POST /api/upload` 且 `mode=knowledge`

任务查询：

```text
GET /api/jobs/<job_id>
```

任务状态：

- `queued`
- `running`
- `succeeded`
- `failed`

`mode=temp` 上传、普通聊天、导出、表格转换仍保持同步行为。

## Development Rules

- 路由层只做 HTTP 包装和认证装饰，业务逻辑放 service。
- 新 service 通过 `AppContext` 暴露给 route，不在 route 中新建全局单例。
- `app_dependencies.py` 只做启动装配；不要把运行期 helper 或业务逻辑放回这里。
- 新聊天能力必须同步更新 `task_planner.py`、`tool_runtime.py` 注册表、
  `chat_runtime.py` 图节点、严格 State schema 和持久化/恢复测试。
- Legacy/实验文件保留参考，但不是新功能接入点：`intelligent_router.py`、`router_integration_demo.py`、`ROUTER_ARCHITECTURE.md`。

## Tests

常用检查：

```bash
cd Agent
.venv/bin/python -m py_compile app.py app_config.py app_context.py app_dependencies.py routes/*.py chat_runtime.py chat_container.py graph_persistence.py graph_artifacts.py graph_state_validation.py
.venv/bin/python -m pytest
```

高价值目标测试：

```bash
.venv/bin/python -m pytest \
  test_app_route_service_wrappers.py \
  test_chat_runtime.py \
  test_chat_runtime_safety.py \
  test_chat_persistence_integration.py \
  test_chat_sqlite_recovery.py \
  test_chat_container.py \
  test_document_graph_runner.py \
  test_document_graph_steps.py \
  test_graph_persistence.py \
  test_graph_state_validation.py \
  test_graph_artifacts.py \
  test_server_startup.py
```

本次 LangGraph 发布验收（2026-08-16）：

```text
全量测试：480 passed, 31 skipped
PostgreSQL 16 真实持久化、跨 Worker 与强杀恢复：16 passed
Python compileall、git diff --check、生产 Compose 配置检查：通过
```
