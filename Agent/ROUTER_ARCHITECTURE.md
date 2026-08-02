# 智能 Router 架构文档

> 历史说明：本文档描述的是早期 `IntelligentRouter` 实验架构，不是当前生产聊天主链路。当前生产主链路以 `/api/chat -> ChatGraphRuntime -> IntentRouter -> intent stream service` 为准；详见 `docs/current_architecture.md`。

## 概述

智能 Router 是一个基于 `asyncio` 的并行执行框架，能够智能判断输入类型并选择最佳执行路径。它是早期实验调度组件，保留用于参考，不作为新功能接入点。

## 核心特性

### 1. 并行执行 (asyncio)

```python
# 并行执行多个 Agent
tasks = [
    self.execute_single(agent1, input_data, context),
    self.execute_single(agent2, input_data, context),
    self.execute_single(agent3, input_data, context),
]
results = await asyncio.gather(*tasks)
```

**优势：**
- 非阻塞 I/O 操作
- 同时运行多个 Agent，减少总等待时间
- 自动超时控制，防止单个 Agent 阻塞

### 2. 智能路由判断

```python
# 分析输入，自动选择最佳路径
decision = router.analyze_input("解析这个 Word 文档")
# 结果: RouteDecision(
#   primary_route=RouteType.DOCUMENT_ANALYSIS,
#   confidence=0.88,
#   reasoning="检测到关键词 [文档, word, 解析]"
# )
```

**路由策略：**
1. **关键词匹配** - 基于预定义关键词库匹配
2. **文件类型检测** - 根据文件扩展名判断
3. **置信度计算** - 为每个候选路径打分
4. **多路径并行** - 主路径 + 次要路径同时执行

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         用户输入                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      IntelligentRouter                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 analyze_input()                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ 关键词匹配  │  │ 文件类型检测 │  │ 置信度计算  │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RouteDecision                          │   │
│  │  primary_route: DOCUMENT_ANALYSIS                   │   │
│  │  secondary_routes: [KNOWLEDGE_RETRIEVAL]            │   │
│  │  confidence: 0.88                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              execute_parallel()                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │  Agent 1    │  │  Agent 2    │  │  Agent 3    │ │   │
│  │  │   (async)   │  │   (async)   │  │   (async)   │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │       │                │                │          │   │
│  │       └────────────────┼────────────────┘          │   │
│  │                        ▼                          │   │
│  │              asyncio.gather()                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              _aggregate_results()                   │   │
│  │         智能合并多个 Agent 的结果                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       最终输出结果                          │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. RouteType (路由类型枚举)

```python
class RouteType(Enum):
    DOCUMENT_ANALYSIS = auto()    # 文档分析
    KNOWLEDGE_RETRIEVAL = auto()  # 知识检索
    FORMAT_GENERATION = auto()    # 格式生成
    CODE_GENERATION = auto()      # 代码生成
    GENERAL_CHAT = auto()         # 通用对话
```

### 2. BaseAgent (Agent 基类)

```python
class BaseAgent(ABC):
    def __init__(self, name: str, route_type: RouteType):
        self.name = name
        self.route_type = route_type

    @abstractmethod
    async def process(self, input_data: Any, context: Dict) -> AgentResult:
        pass
```

### 3. RouteDecision (路由决策)

```python
@dataclass
class RouteDecision:
    primary_route: RouteType      # 主要路径
    secondary_routes: List[RouteType]  # 次要路径
    confidence: float             # 置信度
    reasoning: str                # 决策理由
```

### 4. AgentResult (执行结果)

```python
@dataclass
class AgentResult:
    route_type: RouteType
    success: bool
    output: Any
    execution_time: float
    error: Optional[str]
```

## 使用示例

### 基础用法

```python
import asyncio
from intelligent_router import IntelligentRouter, RouteType

async def main():
    router = IntelligentRouter()

    # 智能路由执行
    result = await router.route("解析这个 Word 文档")

    print(f"主路径: {result['decision']['primary_route']}")
    print(f"置信度: {result['decision']['confidence']}")
    print(f"执行时间: {result['total_execution_time']}s")

asyncio.run(main())
```

### 并行执行多个任务

```python
# 定义并行任务
tasks = [
    {"route_type": RouteType.DOCUMENT_ANALYSIS, "input": "file1.docx"},
    {"route_type": RouteType.KNOWLEDGE_RETRIEVAL, "input": "查询内容"},
    {"route_type": RouteType.FORMAT_GENERATION, "input": "生成报告"},
]

# 并行执行
results = await router.execute_parallel(tasks)
```

### 自定义 Agent

```python
class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("我的Agent", RouteType.GENERAL_CHAT)

    async def process(self, input_data: Any, context: Dict) -> AgentResult:
        start_time = time.time()

        # 处理逻辑...
        output = await self._do_work(input_data)

        return AgentResult(
            route_type=self.route_type,
            success=True,
            output=output,
            execution_time=time.time() - start_time
        )

# 注册自定义 Agent
router = IntelligentRouter()
router.register_agent(MyCustomAgent())
```

## 工作流程

### 1. 输入分析阶段

```
输入文本 → 关键词匹配 → 文件类型检测 → 置信度计算 → 路径选择
```

### 2. 并行执行阶段

```
主路径 Agent ─┐
              ├→ asyncio.gather() → 结果收集
次要路径 Agent┘
```

### 3. 结果聚合阶段

```
多个 Agent 结果 → 优先级排序 → 智能合并 → 最终输出
```

## 性能优化

### 1. 超时控制

```python
# 设置 Agent 超时
agent.set_timeout(30.0)  # 30秒超时

# 超时自动返回错误
AgentResult(
    success=False,
    error="执行超时（>30s）",
    ...
)
```

### 2. 线程池执行器

```python
# 使用 ThreadPoolExecutor 处理阻塞操作
self.executor = ThreadPoolExecutor(max_workers=5)

# 在单独的线程中运行阻塞操作
result = await loop.run_in_executor(
    self.executor,
    blocking_function,
    *args
)
```

### 3. 次要路径优化

- 置信度 < 0.5 时启用次要路径
- 最多并行 2 个次要路径
- 自动选择相关性最高的次要路径

## 集成现有系统

### 与知识库集成

```python
class KnowledgeBaseBuildAgent(BaseAgent):
    async def process(self, input_data, context):
        files = self._extract_files(input_data)

        # 并行解析多个文档
        tasks = [self._parse_document(f) for f in files]
        results = await asyncio.gather(*tasks)

        return AgentResult(...)
```

### 与文档解析集成

```python
from document_parser import parse_document_with_format

async def _parse_document(self, file_path: Path) -> Dict:
    # 在单独线程中运行阻塞操作
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        parse_document_with_format,
        file_path
    )
    return result
```

## 最佳实践

### 1. 合理设置超时

```python
# 根据操作复杂度设置超时
DocumentAnalysisAgent: 30s
KnowledgeRetrievalAgent: 10s
FormatGenerationAgent: 20s
CodeGenerationAgent: 60s
```

### 2. 使用上下文传递数据

```python
context = {
    "file_path": "./document.docx",
    "format_spec": {"font": "仿宋", "size": "三号"},
    "phase": "analysis"  # 用于区分不同阶段
}

result = await router.route("解析文档", context=context)
```

### 3. 错误处理

```python
results = await asyncio.gather(*tasks, return_exceptions=True)

for result in results:
    if isinstance(result, Exception):
        # 处理异常
    elif not result.success:
        # 处理 Agent 返回的错误
```

### 4. 结果聚合策略

```python
def _aggregate_results(self, decision, results):
    # 1. 优先使用主路径结果
    primary = result_map.get(decision.primary_route)
    if primary and primary.success:
        return primary.output

    # 2. 主路径失败，尝试次要路径
    for route in decision.secondary_routes:
        result = result_map.get(route)
        if result and result.success:
            return result.output

    # 3. 返回错误信息
    return {"error": "所有路径均失败"}
```

## 扩展开发

### 添加新的路由类型

```python
class RouteType(Enum):
    # 现有类型...
    IMAGE_ANALYSIS = auto()  # 新增图像分析

# 添加关键词
ROUTE_KEYWORDS = {
    # 现有关键词...
    RouteType.IMAGE_ANALYSIS: ["图片", "图像", "照片", "png", "jpg"]
}
```

### 自定义路由策略

```python
class AdvancedRouter(IntelligentRouter):
    def analyze_input(self, input_data: Any) -> RouteDecision:
        # 调用父类分析
        decision = super().analyze_input(input_data)

        # 添加自定义逻辑
        if self._is_urgent(input_data):
            decision.metadata["priority"] = "high"

        return decision

    def _is_urgent(self, input_data: Any) -> bool:
        # 判断是否为紧急请求
        pass
```

## 总结

智能 Router 提供了：

1. **智能路由** - 基于关键词和文件类型的自动路径选择
2. **并行执行** - 使用 asyncio 同时运行多个 Agent
3. **超时控制** - 防止单个 Agent 阻塞整体流程
4. **结果聚合** - 智能合并多个 Agent 的输出
5. **易于扩展** - 通过继承 BaseAgent 添加自定义 Agent

这个架构可以显著提高系统的响应速度和处理能力，特别适合需要同时处理多种类型请求的场景。
