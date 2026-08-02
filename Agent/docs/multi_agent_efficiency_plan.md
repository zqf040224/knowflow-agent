# 多 Agent 工作流效率与 Token 节约计划书 v1.1

版本：v1.1

日期：2026-05-29

## 目标

在不降低公文质量、知识库准确性和 Excel 数据可靠性的前提下，减少多 Agent 工作流中的重复上下文、重复审查和重复生成，降低 token 消耗与响应耗时。

## 当前状态

已完成第一阶段：

- `BaseAgent` 已记录 LLM 调用用量：prompt 字符数、completion 字符数、reasoning 字符数、估算 token、模型、耗时、是否流式。
- `Orchestrator.run_records` 已携带 `llm_usage`，可定位高消耗 Agent 和步骤。
- 已新增 `compact_evidence`，将知识库/Excel/网页证据压缩后传给 Writer。
- 修订轮已进入 `revision_mode`，避免重复传完整知识库上下文和搜索上下文。

## v1.1 范围补充：NAS 存储接入

NAS 存储接入纳入 v1.1 版本计划，作为内测前的部署与数据存储能力补充。v1.1 的 NAS 目标不是重做账号体系，而是先支持在已挂载 NAS 目录上保存和读取 Agent 运行所需的文件资产。

### 1. 目标

- 支持通过环境变量在 `local` 与 `nas` 存储后端之间切换。
- 将上传文件、知识库索引、表格结构库、导出文件和后台备份目录统一纳入可配置存储路径。
- 保持本地默认配置不变，未配置 NAS 时继续使用项目目录下的 `uploads`、`knowledge_base`、`outputs`。
- 提供健康检查接口，方便管理员确认 NAS 挂载目录是否存在、可读、可写。

### 2. 交付内容

- 新增 `storage_config.py`，集中管理本地/NAS 路径。
- `.env.example` 增加 `STORAGE_BACKEND`、`NAS_MOUNT_PATH`、`UPLOADS_DIR`、`KNOWLEDGE_BASE_DIR`、`OUTPUTS_DIR` 等配置项。
- `/api/health` 和管理员知识库健康接口返回存储健康状态。
- 知识库、表格库、上传管理和后台备份改为读取统一存储配置。
- 新增 `docs/nas_storage_setup.md`，说明 NAS 挂载、目录准备、环境变量配置和验证步骤。

### 3. v1.1 验收标准

- `STORAGE_BACKEND=local` 时，现有本地开发和内测流程不受影响。
- `STORAGE_BACKEND=nas` 且 `NAS_MOUNT_PATH` 指向已挂载目录时，上传、知识库索引、表格库、导出和后台备份均落到指定 NAS 根目录。
- `/api/health` 返回 `storage.ok=true`，并能列出关键目录读写状态。
- NAS 权限异常、目录不存在或不可写时，健康检查能给出明确错误信息。
- 离线回归测试、流式管线测试和表格上传测试通过。

### 4. 非目标

- v1.1 暂不实现完整 NAS 账号登录、SMB 权限继承和 NAS 文件浏览器。
- v1.1 暂不把 SQLite 长期生产化部署到 NAS；NAS 上的 SQLite 仅用于内测验证，后续正式生产建议迁移到 PostgreSQL/MySQL。
- v1.1 不直接修改 NAS 源文件，只管理 Agent 自身产生或维护的文件资产。

## v1.1 范围补充：智能体自我迭代闭环

智能体自我迭代纳入 v1.1 改进计划，但 v1.1 的目标不是让 Agent 自动修改代码或自动上线，而是建立“运行记录 -> 质量评估 -> 失败归因 -> 改进提案 -> 回放验证 -> 人工确认”的可控闭环。该闭环优先服务内测质量提升、Prompt 调整、检索策略优化和知识库治理。

### 1. 目标

- 将每次 Agent 执行过程结构化沉淀，支持按 session、intent、Agent 步骤、来源证据、审查结果和用户反馈追溯。
- 将低评分反馈、失败回答、证据不足、审查失败和高 token 消耗任务沉淀为可复盘样例。
- 建立离线评测入口，覆盖 RAG 问答、意图路由、公文生成、Excel 事实审计和导出链路。
- 由 Agent 生成改进报告和候选策略，但所有 Prompt、策略、代码和知识库变更必须人工确认后生效。
- 支持版本化记录 Prompt 和策略调整，便于回放验证、效果对比和回滚。

### 2. 交付内容

- 新增 Agent 运行记录能力，记录：
  - `run_id`、`session_id`、`user_id`、`intent`、`workflow_mode`。
  - 用户请求、最终输出、任务计划、来源证据、审查结果、反思结果。
  - `run_records`、`llm_usage`、耗时、token 估算和错误信息。
  - Prompt/策略版本号、用户评分、关联反馈 ID。
- 扩展反馈链路，将当前回答、intent、来源文件、审查摘要、run_id 写入 `beta_feedback.context_json`。
- 扩展 `scripts/evaluate_rag_quality.py` 为统一离线评测入口，逐步支持：
  - `rag_cases.json`
  - `routing_cases.json`
  - `doc_drafting_cases.json`
  - `spreadsheet_fact_cases.json`
- 新增 `SelfIterationService`，通过后台任务定期汇总失败案例，输出改进报告：
  - 高频失败类型。
  - 失败根因分析。
  - Prompt、检索、路由、知识库或工具链改进建议。
  - 回放评测结果和风险提示。
- 新增 Prompt/策略版本记录，至少记录 Agent 名称、版本号、配置摘要、评测分数、启用状态和创建时间。

### 3. 自我迭代流程

```text
用户请求
  -> ChatGraphRuntime / AgentOrchestrator 执行
  -> 记录运行过程、证据、审查、反思和 token 使用
  -> 用户反馈或离线评测标记问题
  -> SelfIterationService 聚类失败案例并分析根因
  -> 生成候选改进方案和回放评测报告
  -> 管理员人工确认
  -> 启用新 Prompt/策略版本或更新知识库
```

### 4. 安全边界

- v1.1 不允许 Agent 自动修改生产代码并直接上线。
- v1.1 不允许 Agent 自动扩大知识库、NAS 或用户权限。
- v1.1 不允许将个人隐私、敏感正文或受限资料写入公共长期记忆。
- Prompt/策略变更必须保留旧版本并支持回滚。
- 所有自动生成的改进建议必须展示证据来源和关联失败案例。

### 5. v1.1 验收标准

- 每次公文生成和知识库问答都能关联到可追踪的 `run_id`。
- 管理后台反馈详情能看到 intent、来源文件、审查摘要和关联运行记录。
- 离线评测能读取手工样例和低分反馈，输出通过率、失败列表和问题分类。
- SelfIterationService 能生成一份内测质量改进报告，但不会自动改代码或自动启用新策略。
- Prompt/策略版本支持记录当前启用版本，至少能手动回滚到上一版。

---

## 第二阶段：减少不必要的 LLM 调用

### 核心目标

降低 Reviewer 和 Reflection 的触发频率，让简单任务走轻量链路，高风险任务才进入深度审查。

### 1. Reviewer 三档审查模式

将 Reviewer 从“高风险任务默认 LLM 审查”改为三档策略：

- `rule_only`
  - 适用：短文、格式调整、简单润色、无知识库证据、无 Excel 数据。
  - 行为：只执行硬格式检查、来源引用检查、Excel 数值规则检查。
  - 预期：不调用 LLM。

- `fact_guard`
  - 适用：涉及知识库来源、Excel 报表数据、引用文件名、数字事实。
  - 行为：执行规则审查 + 结构化事实校验。
  - 预期：只有规则无法判断或发现复杂冲突时才调用 LLM。

- `llm_review`
  - 适用：长文、复杂公文、多个约束、低置信度、用户明确要求严格审查。
  - 行为：执行完整 LLM 内容审查。

### 2. Reviewer 触发条件调整

将 `ReviewerAgent._should_run_llm_review()` 改为按以下条件触发：

- 文档长度超过 2200 字。
- 写作要点不少于 5 条。
- 用户约束不少于 4 条。
- Excel 审计发现不确定项或冲突项。
- 规则审查置信度低于 0.78。
- `task_type` 属于重大材料类，并且存在知识库/搜索证据。
- 用户请求中包含“严格审查”“反复检查”“确保准确”等强审查意图。

否则默认 `rule_only` 或 `fact_guard`。

### 3. Reflection 触发门控

将 R1 Reflection 从“公文生成基本触发”改为显式门控：

- 只在以下情况触发：
  - Reviewer LLM 审查置信度低于 0.78。
  - Reviewer 发现逻辑问题或事实问题。
  - Excel 数据审计存在未验证数字。
  - 文档类型为对策建议、情况反映、重大报告等需要论证链的材料。
  - 用户明确要求“深度反思”“多角度论证”“风险研判”。

- 不触发情况：
  - 简单通知、函、格式整理、摘要、普通润色。
  - Reviewer 规则审查已通过且无事实风险。

### 4. 轻量 Workflow 路由

在 Planner 输出中增加或规范化 `workflow_mode`：

- `light`
  - Plan -> Knowledge 可选 -> Writer -> Rule Review
  - 不进入 Reflection。

- `standard`
  - Plan -> Retrieval -> Writer -> Reviewer
  - 仅必要时进入修订。

- `strict`
  - Plan -> Retrieval -> Writer -> Reviewer -> Reflection
  - 用于复杂、高风险、强准确性任务。

Planner 只负责建议模式，Orchestrator 最终根据规则兜底决定。

### 5. 第二阶段验收标准

- 简单任务不调用 Reviewer LLM。
- 普通公文默认不调用 Reflection。
- Excel 数字相关任务仍保留结构化审计和事实保护。
- `run_records` 能显示每个步骤是否跳过 LLM 以及跳过原因。
- 离线回归测试全部通过。

### 6. 第二阶段测试场景

- 简单通知生成：不触发 Reflection，Reviewer 为 `rule_only`。
- Excel 报表材料生成：触发 `fact_guard`，未验证数字会被拦截。
- 对策建议长文：可触发 `llm_review`，必要时触发 Reflection。
- 用户明确要求严格审查：强制进入 `strict`。

---

## 第三阶段：减少重复生成与重复上下文

### 核心目标

减少 Writer 的全量重写次数，让修订更像“基于上一版精准改稿”，而不是每轮重新生成。

### 1. 引入修订策略 `revision_strategy`

由 Orchestrator 根据 Reviewer 输出决定修订策略：

- `none`
  - 无需修订，直接输出。

- `patch_prompt`
  - 适用：格式、引用、措辞、局部事实修正。
  - Writer 输入：上一版全文 + 问题清单 + 必要证据。
  - 不再传完整知识库上下文。

- `section_rewrite`
  - 适用：某一节逻辑或内容需要重写。
  - Writer 输入：上一版全文 + 指定章节 + 修订目标。
  - 输出仍为完整公文全文。

- `full_rewrite`
  - 适用：结构严重错误、文种错误、用户需求理解错误。
  - 允许重新使用较完整上下文。

默认策略为 `patch_prompt`，只有严重问题才升级。

### 2. Reviewer 输出结构化修订范围

Reviewer metadata 增加：

- `revision_strategy`
- `affected_sections`
- `must_keep`
- `must_change`
- `evidence_needed`

示例：

```json
{
  "revision_strategy": "patch_prompt",
  "affected_sections": ["第二部分"],
  "must_keep": ["标题", "落款", "第一部分"],
  "must_change": ["补充数据来源", "修正 2024 年产值"],
  "evidence_needed": ["产业报表.xlsx Sheet1 行2"]
}
```

### 3. Writer 修订轮 Prompt 精简

修订轮只传：

- 原始用户需求摘要。
- 上一版正文。
- `must_change` 和 `affected_sections`。
- compact evidence 中与问题相关的 3-6 条。
- Excel 原值与 sheet/行号。

不再传：

- 完整 `knowledge_context`。
- 完整 `search_context`。
- 全部历史 revision。
- 全量 `context_analysis`。

### 4. 修订轮上限收紧

默认最多自动修订 1 次。

如第二次审查仍有问题：

- 输出当前最优版本。
- 在 `audit_summary` 中明确列出剩余风险。
- 不继续无限消耗 token。

对 `strict` workflow 可允许最多 2 次修订，但必须在 `run_records` 标记原因。

### 5. 生成缓存

对同一会话中的重复请求增加缓存判断：

- 如果 `user_request + compact_evidence + revision_focus` 未变化，避免重复调用 Writer。
- 对断流重试场景复用上一版可用草稿。

### 6. 第三阶段验收标准

- 第二轮 Writer prompt 明显小于第一轮。
- 修订轮不再重复注入完整知识库上下文。
- 大多数修订任务只调用一次额外 Writer。
- `run_records` 记录 `revision_strategy`、输入规模和节省比例。
- Excel 事实修订仍保留原值、sheet、行号。

### 7. 第三阶段测试场景

- 格式小问题：走 `patch_prompt`，不全量重写。
- Excel 数字错误：只传相关报表证据，修正为原值。
- 文种识别错误：升级为 `full_rewrite`。
- 长文逻辑局部问题：走 `section_rewrite`。
- 多轮仍失败：停止自动修订并输出风险摘要。

---

## 预期收益

- 第二阶段预计减少 30%-50% 的 Reviewer/R1 LLM 调用。
- 第三阶段预计减少 25%-45% 的修订轮 prompt 规模。
- 对简单任务，整体耗时预计下降 35%-50%。
- 对 Excel 报表任务，保持事实审计强度，同时减少重复上下文注入。
- 自我迭代闭环预计提升内测问题定位效率，减少重复排查时间，并为 Prompt/检索策略调整提供可回放依据。

## 实施顺序

1. 完成 v1.1 NAS 存储接入和健康检查，先满足内测部署需要。
2. 增加 Agent 运行记录和反馈上下文关联，先打通自我迭代所需数据底座。
3. 扩展离线评测脚本，接入低分反馈和手工评测样例。
4. 新增 SelfIterationService，先输出改进报告，不自动应用改动。
5. 第二阶段改 Reviewer 和 Reflection 门控。
6. 增加 `workflow_mode` 和审查模式记录。
7. 回归测试后，再进入第三阶段。
8. 第三阶段先实现 `patch_prompt`，再扩展 `section_rewrite`。
9. 最后增加生成缓存、Prompt/策略版本记录和节省比例统计。
