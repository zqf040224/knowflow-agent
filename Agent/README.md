# Knowledge Agent

智能知识库与公文写作助手。当前后端是 Flask 应用，提供登录、聊天、知识库检索、文件上传入库、后台管理、导出和轻量后台任务能力。

## Quick Start

```bash
cd Agent
cp .env.example .env
.venv311/bin/python app.py --port 5003
```

访问：

- 登录页：`http://localhost:5003/login`
- 聊天页：`http://localhost:5003/chat`
- 管理后台：`http://localhost:5003/admin`
- 健康检查：`http://localhost:5003/api/health`

生产启动方式保持兼容：

```bash
gunicorn -w 1 -b 0.0.0.0:5003 app:app
```

当前 `JobService` 使用本进程线程池执行任务。任务状态持久化到 SQLite，但执行队列属于当前进程；多 worker 部署前需要先引入跨进程队列或 owner/heartbeat 策略的完整调度治理。

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
| `CHAT_RUNTIME` | 聊天运行时，支持 `langgraph` 或 `legacy` |
| `ENABLE_LLM_INTENT_CLASSIFIER` | 是否启用 LLM 意图分类兜底 |

密钥不要提交到仓库，使用 `.env` 本地配置。

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
| `chat_runtime.py` | `/api/chat` 外层主链路：准备请求、意图路由、分派 stream service |
| `chat_container.py` | 聊天 stream service 的懒加载装配 |
| `job_service.py` | SQLite 任务状态 + 本地 `ThreadPoolExecutor` |
| `upload_service.py` | 临时上传与知识库上传入库包装 |

更多内部边界见 `docs/current_architecture.md`。

## Chat Mainline

生产聊天唯一主入口：

```text
POST /api/chat
  -> routes/chat_routes.py: chat()
  -> context.chat_runtime().stream(...)
  -> ChatGraphRuntime
  -> IntentRouter
  -> intent stream service
```

Intent 分派：

| Intent | Handler |
| --- | --- |
| `knowledge_qa` | `RagQaStreamService` |
| `doc_drafting` | `DocumentDraftStreamService -> AgentOrchestrator` |
| `doc_formatting` | `DocumentFormatStreamService` |
| `identity_help` / `clarify` / `form_template_export` / `spreadsheet_transform` | `LightweightChatStreamService` |

新增聊天能力优先走 `IntentRouter + ChatGraphRuntime + stream service`，不要在 Flask route 中直接调用 Agent，也不要接入历史 `IntelligentRouter`。

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
- 新聊天 intent 必须同时更新 `chat_architecture.py`、`chat_runtime.py`、`chat_container.py` 和测试。
- Legacy/实验文件保留参考，但不是新功能接入点：`intelligent_router.py`、`router_integration_demo.py`、`ROUTER_ARCHITECTURE.md`。

## Tests

常用检查：

```bash
cd Agent
.venv311/bin/python -m py_compile app.py app_config.py app_context.py app_dependencies.py routes/*.py chat_runtime.py chat_container.py job_service.py upload_service.py
.venv311/bin/python -m pytest
```

高价值目标测试：

```bash
.venv311/bin/python -m pytest \
  test_app_route_service_wrappers.py \
  test_chat_runtime.py \
  test_chat_container.py \
  test_current_architecture_docs.py \
  test_job_service.py \
  test_upload_service.py \
  test_export_service.py \
  test_runtime_query_service.py
```

当前本地回归基线：

```text
198 passed, 15 skipped
```

跳过项多为外部 API 或本地资料条件不足的测试。
