# 智能知识库平台 Agent 系统

## 项目定位
本项目是面向团队知识管理与文档生成场景的智能知识库平台。系统以知识库检索、资料问答、材料归纳、内容改写、公文生成和 Word 导出为核心能力，公文写作只是平台功能之一，不应再把整体产品描述为单一的“公文写作助手”。

平台主要运行在局域网环境，优先关注可用性、检索质量、Agent 流程稳定性和前端流式体验。安全能力以账号、角色、部门权限闭环为主，不引入过重的外部安全基础设施。

## 当前技术栈
- **后端**: Flask + Jinja2 + Server-Sent Events
- **LLM**: DeepSeek API，默认模型 `deepseek-v4-flash`，反思阶段使用 `deepseek-reasoner`
- **Embedding**: `maidalun1020/bce-embedding-base_v1`，768 维，本地 `sentence-transformers`
- **知识库检索**: FAISS + BM25 + RRF 融合召回，按用户角色和部门过滤
- **联网搜索**: 博查 Web Search API，环境变量为 `BOCHA_API_KEY`
- **数据库**: SQLite，默认路径 `./data/agent_memory.db`
- **文档处理**: PDF / Word / TXT 解析，`python-docx` 导出 Word
- **前端**: 原生 HTML / CSS / JavaScript，浅色与深色模式，SSE 流式输出
- **部署**: Dockerfile + gunicorn `gthread`，`docker-compose.yml` 只启动应用和可选知识库初始化任务

## 关键环境变量
```bash
DEEPSEEK_API_KEY=your-deepseek-api-key
BOCHA_API_KEY=sk-your-bocha-key-here
JWT_SECRET=generate-a-random-secret-here
ADMIN_DEFAULT_PASSWORD=<set-a-strong-random-password>
PORT=5003
HOST=0.0.0.0
CORS_ORIGINS=http://localhost:5003
COOKIE_SECURE=False
CACHE_BACKEND=memory
```

注意：
- 主代码不再使用 `TAVILY_API_KEY`。
- `.env.example` 应保持和当前真实代码一致。
- 缺少 `DEEPSEEK_API_KEY` 时，普通聊天接口会通过 SSE 返回明确错误，不应等到 OpenAI 客户端调用时才失败。
- 局域网默认使用内存缓存；只有设置 `CACHE_BACKEND=redis` 时才连接 Redis。

## 主要路由与模式
| 模式 | 后端入口 | 用途 |
| --- | --- | --- |
| `quick` | `_quick_stream` | 默认模式。知识库检索 + WriterAgent 快速流式生成。 |
| `agent` | `_agent_stream` | 完整多 Agent 工作流，适合复杂公文、材料改写和需要审查的任务。 |
| `document` | `_quick_stream` | 与 quick 共用路径，偏向文档生成和格式转换。 |
| `chat` | `_chat_stream` | 知识库问答，基于检索材料流式回答并尽量标注来源。 |
| 身份问答 | `_identity_stream` | 用户问“你是谁/你能做什么”时，直接回答“智能知识库助手”及能力边界。 |

前端默认仍使用 `quick` 模式，不要随意改成 `agent`。`agent` 模式更重，主要用于需要完整规划、检索、审查和反思的复杂任务。

## Agent 编排流程
完整 Agent 入口以 `AgentOrchestrator` 为准，不启用独立示例路由。

```text
用户请求
  → PlannerAgent.process_with_context：一次调用完成上下文分析和结构化任务规划
  → SearchAgent：按需使用博查联网搜索
  → KnowledgeAgent：检索本地知识库并做权限过滤
  → WriterAgent：流式生成正文
  → ReviewerAgent：结构化审查格式、事实、逻辑、语言问题
  → ReflectionAgent：仅复杂或低置信度任务触发 R1 深度反思
  → 保存最终文档、运行记录、来源文件和上下文快照
```

Planner 需要稳定区分这些任务类型：
- `问答检索`
- `公文生成`
- `材料改写`
- `格式转换`
- `续写修改`

## 流式输出约定
前端通过 SSE 消费事件。修改 Agent 流程时要保持这些事件类型兼容：
- `start`
- `session`
- `context_start`
- `context_end`
- `plan_start`
- `plan`
- `think`
- `write_start`
- `content`
- `content_reset`
- `reasoning_chunk`
- `reflection`
- `done`
- `error`

当前正文和 Agent 思考过程都应尽量流式输出。`ReflectionAgent.process_stream()` 内部产生的 `think` 事件必须继续向上游 yield，不能只进入后端日志或 `think_log`。为减少首包等待，完整 Agent 链路默认合并 Context + Planner，不再为上下文分析单独发起一次 LLM 调用。

## 知识库与权限
知识库目录默认结构：

```text
知识库/
├── 公共资料/      public，所有登录用户可见
├── 综合管理部/    restricted，本部门和管理员可见
├── 人力资源部/
├── 财务部/
├── 运营保障部/
├── 市场部/
├── 业务部/
├── 客户服务部/
└── 项目部/
```

部署时可通过 `DEPARTMENT_DIRS` 环境变量覆盖这组公开示例值。

上传知识库文件时，后端必须重新校验分类权限，不能信任前端传入的 `category` / `department`。普通用户只能上传到自己有权限的目录，管理员可管理全部目录。

知识库文档元数据应尽量保持完整：
- `filename`
- `source_path`
- `category`
- `department`
- `access_level`
- `uploaded_by`
- `uploaded_at`
- `chunk_index`
- `total_chunks`
- `doc_type`
- `content_hash`

## 上传与索引稳定性
知识库上传流程需要保证失败可回滚：
- 文件解析失败时清理临时文件。
- 索引写入失败时回滚已追加的文本和元数据。
- FAISS / 文本 / 元数据文件保存应使用临时文件替换，避免半写入索引。
- 支持基于内容 hash 的重复上传检测。
- 上传成功后刷新知识库索引与相关缓存。

## 部署说明
开发启动：

```bash
cd Agent
python3 app.py --port 5003
```

重建知识库索引：

```bash
cd Agent
python3 builder.py --input ./知识库 --output ./knowledge_base
```

Docker 启动应用：

```bash
cd Agent
docker compose up --build app
```

可选初始化知识库：

```bash
cd Agent
docker compose --profile init run --rm kb-init
```

生产容器内默认使用：

```bash
gunicorn -w 4 --worker-class gthread --threads 8 --timeout 120 -b 0.0.0.0:5003 app:app
```

SSE 流式响应需要 `gthread` 或其他兼容长连接的 worker 配置。不要使用会阻塞流式体验的同步单线程部署方式。

## 常用检查命令
```bash
cd /path/to/project
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp python3 -m py_compile \
  Agent/app.py \
  Agent/agents/base_agent.py \
  Agent/agents/orchestrator.py \
  Agent/agents/reflection_agent.py \
  Agent/agents/search_agent.py \
  Agent/agents/reviewer_agent.py

python3 Agent/test_static_bugs.py
node --check Agent/static/js/chat.js
```

说明：
- `Agent/test_agent_bugs.py` 中部分用例会真实请求 DeepSeek，在无网络或沙箱网络受限时可能出现 `Connection error`。
- 静态测试、语法检查和本地 mock 验证优先用于确认基础改动没有破坏工程。

## 开发约束
- 不要把平台整体命名回“公文写作助手”，统一使用“智能知识库平台”和“智能知识库助手”。
- 不要重新引入 `TAVILY_API_KEY`，联网搜索统一走博查。
- 不要让模型凭空编造引用来源，最终来源应优先来自 KnowledgeAgent 返回的元数据。
- 修改 `/api/chat` 时保持前端 SSE 事件兼容。
- 修改会话相关接口时，不要用会创建新会话的函数来做权限校验。
- 前端默认 quick 模式保持不变，除非明确要进入复杂 Agent 流程。
- Docker 启动不应每次自动重建知识库索引，索引构建走独立初始化命令。
