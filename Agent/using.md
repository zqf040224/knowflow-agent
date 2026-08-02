# Agent 智能文档处理系统使用说明

> 历史说明：本文档保留早期文档处理与 `IntelligentRouter` 实验说明，不代表当前生产聊天主链路。当前生产主链路以 `/api/chat -> ChatGraphRuntime -> IntentRouter -> intent stream service` 为准；详见 `docs/current_architecture.md`。

## 项目概述

Agent 是一个智能文档处理系统，支持：
- **文档解析**：支持 PDF、Word、TXT、Markdown 格式
- **格式指纹提取**：提取 Word 文档的字体、字号、加粗、对齐等格式信息
- **知识库构建**：构建向量索引，支持语义检索
- **历史智能路由实验**：`IntelligentRouter` 保留用于参考，生产聊天入口不再从这里接入

## 项目结构

```
Agent/
├── builder.py                  # 知识库构建器 v2.0
├── document_parser.py          # 文档解析模块
├── intelligent_router.py       # 智能路由核心模块
├── router_integration_demo.py  # 集成演示
├── ROUTER_ARCHITECTURE.md      # 架构文档
├── using.md                    # 本文档
└── 知识库/                      # 示例知识库目录
```

## 环境要求

- Python 3.8+
- 依赖包：
  - `python-docx` - Word 文档解析
  - `PyPDF2` - PDF 文档解析
  - `langchain` - 文本切分和向量索引
  - `faiss-cpu` - 向量数据库
  - `dashscope` - 阿里云灵积模型服务（嵌入模型）

## 安装依赖

```bash
pip install python-docx PyPDF2 langchain faiss-cpu dashscope
```

## 配置环境变量

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

## 模块详解与运行方法

### 1. 文档解析模块 (document_parser.py)

支持两种解析模式：
- **纯文本模式**：提取文字内容
- **带格式模式**：提取文字 + 格式指纹

#### 运行方法

```bash
# 测试单个文件
python document_parser.py /path/to/your/document.docx
```

#### Python API 使用

```python
from document_parser import parse_document, parse_document_with_format
from pathlib import Path

# 纯文本模式
content = parse_document(Path("document.docx"))
print(content)

# 带格式模式（推荐）
formatted = parse_document_with_format(Path("document.docx"))
for item in formatted:
    print(f"内容: {item['content']}")
    print(f"格式: {item['format']}")
    # 格式包含：font, size, bold, italic, alignment 等
```

### 2. 知识库构建器 (builder.py)

构建知识库的完整流程：
1. 扫描目录文档
2. 解析并切分文档
3. 构建向量索引
4. 打包为 `.agent` 格式

#### 运行方法

```bash
# 基本用法
python builder.py --input ./资料库 --output ./knowledge_base

# 禁用格式提取（纯文本模式）
python builder.py --input ./资料库 --output ./knowledge_base --no-format
```

#### 参数说明

| 参数 | 说明 | 必填 |
|------|------|------|
| `--input` | 输入资料目录路径 | 是 |
| `--output` | 输出知识库目录路径 | 是 |
| `--no-format` | 禁用格式指纹提取 | 否 |

#### 输出文件

```
knowledge_base/
├── faiss_index/          # FAISS 向量索引
│   ├── index.faiss
│   └── index.pkl
├── config.json           # 配置文件
└── knowledge_base.agent  # 打包后的知识库（分发给用户）
```

### 3. 智能路由模块 (intelligent_router.py)

核心特性：
- **智能路由**：自动分析输入，选择最佳处理路径
- **并行执行**：使用 asyncio 同时运行多个 Agent
- **超时控制**：防止单个 Agent 阻塞

#### 运行方法

```bash
# 运行演示
python intelligent_router.py
```

#### Python API 使用

```python
import asyncio
from intelligent_router import IntelligentRouter, RouteType

async def main():
    router = IntelligentRouter()

    # 简单路由 - 自动分析并执行
    result = await router.route("解析这个 Word 文档")

    print(f"主路径: {result['decision']['primary_route']}")
    print(f"置信度: {result['decision']['confidence']}")
    print(f"并行数: {result['parallel_count']}")
    print(f"总耗时: {result['total_execution_time']}s")

    # 查看各 Agent 执行结果
    for r in result['results']:
        print(f"{r['route']}: {r['success']} ({r['execution_time']}s)")

asyncio.run(main())
```

#### 支持的 Agent 类型

| Agent | 用途 | 触发关键词 |
|-------|------|-----------|
| DocumentAnalysisAgent | 文档分析 | 文档、文件、解析、docx、pdf |
| KnowledgeRetrievalAgent | 知识检索 | 查询、搜索、检索、知识 |
| FormatGenerationAgent | 格式生成 | 生成、公文、报告、格式 |
| CodeGenerationAgent | 代码生成 | 代码、编程、函数、python |
| GeneralChatAgent | 通用对话 | 你好、帮助、问题 |

#### 并行执行多个任务

```python
# 定义并行任务
tasks = [
    {"route_type": RouteType.DOCUMENT_ANALYSIS, "input": "file1.docx"},
    {"route_type": RouteType.KNOWLEDGE_RETRIEVAL, "input": "查询1"},
    {"route_type": RouteType.FORMAT_GENERATION, "input": "生成内容"},
]

# 并行执行
results = await router.execute_parallel(tasks)

for r in results:
    print(f"{r.route_type.name}: {r.success} ({r.execution_time:.3f}s)")
```

#### 自定义 Agent

```python
from intelligent_router import BaseAgent, AgentResult, RouteType

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__("我的Agent", RouteType.GENERAL_CHAT)
        self.set_timeout(10.0)  # 设置10秒超时

    async def process(self, input_data, context):
        start_time = time.time()

        # 处理逻辑
        output = await self._do_work(input_data)

        return AgentResult(
            route_type=self.route_type,
            success=True,
            output=output,
            execution_time=time.time() - start_time
        )

# 注册自定义 Agent
router = IntelligentRouter()
router.register_agent(MyAgent())
```

### 4. 集成演示 (router_integration_demo.py)

展示如何将智能路由与现有系统集成：
- 批量文档处理
- 智能工作流

#### 运行方法

```bash
python router_integration_demo.py
```

#### 演示内容

1. **简单路由演示** - 测试不同类型的输入
2. **批量文档处理** - 并行处理多个文档
3. **智能工作流** - 多阶段并行处理（公文生成示例）

## 完整工作流程示例

### 场景：从文档构建知识库并检索

```python
import asyncio
from pathlib import Path
from builder import KnowledgeBaseBuilder
from intelligent_router import IntelligentRouter

async def workflow():
    # 步骤 1: 构建知识库
    print("=" * 60)
    print("步骤 1: 构建知识库")
    print("=" * 60)

    builder = KnowledgeBaseBuilder(
        input_dir="./my_documents",
        output_dir="./my_knowledge_base"
    )
    success = builder.build(with_format=True)

    if not success:
        print("❌ 知识库构建失败")
        return

    # 步骤 2: 使用智能路由检索
    print("\n" + "=" * 60)
    print("步骤 2: 智能检索")
    print("=" * 60)

    router = IntelligentRouter()

    # 检索请求
    result = await router.route(
        "检索与人工智能相关的文档",
        context={
            "knowledge_base_path": "./my_knowledge_base",
            "top_k": 5
        }
    )

    print(f"\n路由决策: {result['decision']['primary_route']}")
    print(f"置信度: {result['decision']['confidence']:.1%}")
    print(f"执行时间: {result['total_execution_time']}s")

    # 步骤 3: 生成公文
    print("\n" + "=" * 60)
    print("步骤 3: 生成公文")
    print("=" * 60)

    result = await router.route(
        "生成一份人工智能发展报告",
        context={
            "format_spec": {
                "font": "仿宋_GB2312",
                "size": "三号",
                "alignment": "两端对齐"
            }
        }
    )

    print(f"生成结果: {result['final_output']}")

# 运行工作流
asyncio.run(workflow())
```

## 常见问题

### Q1: 解析 Word 文档时出现编码错误

**解决**: 确保安装了 `python-docx`:
```bash
pip install python-docx
```

### Q2: 构建知识库时提示缺少 API Key

**解决**: 设置环境变量:
```bash
export DASHSCOPE_API_KEY="your-api-key"
```

### Q3: 如何扩展新的 Agent 类型

**解决**: 继承 `BaseAgent` 并实现 `process` 方法，然后注册到 Router:
```python
class MyAgent(BaseAgent):
    async def process(self, input_data, context):
        # 实现处理逻辑
        pass

router.register_agent(MyAgent())
```

### Q4: 并行执行时如何控制并发数

**解决**: 修改 `IntelligentRouter` 的 ThreadPoolExecutor:
```python
self.executor = ThreadPoolExecutor(max_workers=3)  # 改为3个并发
```

## 性能优化建议

1. **批量处理**: 使用 `execute_parallel` 同时处理多个文档
2. **超时控制**: 为长时间运行的 Agent 设置合理的超时时间
3. **缓存结果**: 对重复查询使用缓存
4. **异步 I/O**: 所有网络请求使用异步方式

## 注意事项

1. **API Key 安全**: 不要将 `DASHSCOPE_API_KEY` 硬编码在代码中
2. **文件路径**: 使用 `pathlib.Path` 处理跨平台路径
3. **错误处理**: 生产环境需要添加完善的错误处理
4. **日志记录**: 建议添加日志记录以便调试

## 下一步开发

1. **Web 界面**: 可以基于 FastAPI/Flask 构建 Web 界面
2. **数据库集成**: 将知识库索引存储到数据库
3. **缓存层**: 添加 Redis 缓存提高检索速度
4. **监控**: 添加执行时间监控和告警

## 获取帮助

查看详细架构设计: [ROUTER_ARCHITECTURE.md](./ROUTER_ARCHITECTURE.md)
