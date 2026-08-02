"""
历史说明：本模块是早期 asyncio Router 架构实验，保留用于参考和演示。
当前生产聊天主链路不是这里；请使用 /api/chat -> ChatGraphRuntime -> IntentRouter。

智能 Router 模块 - 基于 asyncio 的并行执行与智能路径判断

特性：
1. 并行执行 - 使用 asyncio 同时运行多个 Agent
2. 智能路由 - 根据输入类型自动选择最佳执行路径
3. 结果聚合 - 智能合并多个 Agent 的输出
4. 超时控制 - 防止单个 Agent 阻塞整体流程
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, Union
from concurrent.futures import ThreadPoolExecutor
import inspect


class RouteType(Enum):
    """路由类型枚举"""
    DOCUMENT_ANALYSIS = auto()      # 文档分析
    KNOWLEDGE_RETRIEVAL = auto()  # 知识检索
    FORMAT_GENERATION = auto()    # 格式生成
    CODE_GENERATION = auto()      # 代码生成
    GENERAL_CHAT = auto()         # 通用对话


@dataclass
class RouteDecision:
    """路由决策结果"""
    primary_route: RouteType              # 主要路由路径
    secondary_routes: List[RouteType] = field(default_factory=list)  # 次要路径（用于并行）
    confidence: float = 0.0               # 置信度 0-1
    reasoning: str = ""                   # 决策理由
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Agent 执行结果"""
    route_type: RouteType
    success: bool
    output: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, name: str, route_type: RouteType):
        self.name = name
        self.route_type = route_type
        self.timeout = 30.0  # 默认超时 30 秒

    @abstractmethod
    async def process(self, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        """处理输入数据，返回结果"""
        pass

    def set_timeout(self, seconds: float):
        """设置超时时间"""
        self.timeout = seconds
        return self


class DocumentAnalysisAgent(BaseAgent):
    """文档分析 Agent - 分析文档结构和内容"""

    def __init__(self):
        super().__init__("文档分析器", RouteType.DOCUMENT_ANALYSIS)

    async def process(self, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            # 模拟文档分析处理
            await asyncio.sleep(0.5)  # 模拟处理时间

            # 实际实现：调用文档解析器
            from document_parser import parse_document_with_format, FormatFingerprint

            result = {
                "document_type": self._detect_doc_type(input_data),
                "structure": self._analyze_structure(input_data),
                "key_sections": self._extract_sections(input_data),
                "format_patterns": self._extract_format_patterns(input_data)
            }

            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=True,
                output=result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e)
            )

    def _detect_doc_type(self, input_data: Any) -> str:
        """检测文档类型"""
        if isinstance(input_data, str):
            if input_data.endswith('.docx'):
                return "word_document"
            elif input_data.endswith('.pdf'):
                return "pdf_document"
            elif input_data.endswith('.md'):
                return "markdown"
        return "unknown"

    def _analyze_structure(self, input_data: Any) -> Dict:
        """分析文档结构"""
        return {
            "has_title": True,
            "has_headers": True,
            "has_tables": False,
            "paragraph_count": 0
        }

    def _extract_sections(self, input_data: Any) -> List[str]:
        """提取关键章节"""
        return ["标题", "正文", "结尾"]

    def _extract_format_patterns(self, input_data: Any) -> List[Dict]:
        """提取格式模式"""
        return []


class KnowledgeRetrievalAgent(BaseAgent):
    """知识检索 Agent - 从知识库检索相关信息"""

    def __init__(self, knowledge_base_path: Optional[str] = None):
        super().__init__("知识检索器", RouteType.KNOWLEDGE_RETRIEVAL)
        self.knowledge_base_path = knowledge_base_path

    async def process(self, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            # 模拟知识检索
            await asyncio.sleep(0.3)

            # 实际实现：调用向量数据库检索
            query = self._extract_query(input_data)
            results = await self._search_knowledge_base(query)

            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=True,
                output={
                    "query": query,
                    "results": results,
                    "result_count": len(results)
                },
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e)
            )

    def _extract_query(self, input_data: Any) -> str:
        """从输入中提取查询关键词"""
        if isinstance(input_data, str):
            return input_data[:100]
        elif isinstance(input_data, dict):
            return input_data.get("query", "")
        return str(input_data)

    async def _search_knowledge_base(self, query: str) -> List[Dict]:
        """搜索知识库"""
        # 模拟检索结果
        return [
            {"content": f"相关知识 {i}", "similarity": 0.9 - i * 0.1}
            for i in range(3)
        ]


class FormatGenerationAgent(BaseAgent):
    """格式生成 Agent - 根据格式指纹生成规范文档"""

    def __init__(self):
        super().__init__("格式生成器", RouteType.FORMAT_GENERATION)

    async def process(self, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            # 获取格式要求
            format_spec = context.get("format_spec", {})
            content = self._extract_content(input_data)

            # 模拟生成过程
            await asyncio.sleep(0.4)

            generated_doc = self._apply_format(content, format_spec)

            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=True,
                output={
                    "generated_content": generated_doc,
                    "format_applied": format_spec,
                    "word_count": len(generated_doc)
                },
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e)
            )

    def _extract_content(self, input_data: Any) -> str:
        """提取内容"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return input_data.get("content", "")
        return str(input_data)

    def _apply_format(self, content: str, format_spec: Dict) -> str:
        """应用格式规范"""
        # 实际实现：根据格式指纹调整内容
        font = format_spec.get("font", "仿宋")
        size = format_spec.get("size", "三号")
        return f"【格式：{font} {size}】\n{content}"


class CodeGenerationAgent(BaseAgent):
    """代码生成 Agent - 生成或优化代码"""

    def __init__(self):
        super().__init__("代码生成器", RouteType.CODE_GENERATION)

    async def process(self, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            await asyncio.sleep(0.6)

            language = context.get("language", "python")
            requirement = self._extract_requirement(input_data)

            code = self._generate_code(requirement, language)

            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=True,
                output={
                    "code": code,
                    "language": language,
                    "explanation": "生成的代码说明"
                },
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e)
            )

    def _extract_requirement(self, input_data: Any) -> str:
        """提取代码需求"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return input_data.get("requirement", "")
        return str(input_data)

    def _generate_code(self, requirement: str, language: str) -> str:
        """生成代码"""
        return f"# {language}\n# {requirement[:50]}...\ndef generated_function():\n    pass"


class GeneralChatAgent(BaseAgent):
    """通用对话 Agent - 处理一般性问题"""

    def __init__(self):
        super().__init__("通用对话", RouteType.GENERAL_CHAT)

    async def process(self, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            await asyncio.sleep(0.2)

            message = self._extract_message(input_data)
            response = f"收到消息：{message[:50]}..."

            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=True,
                output={"response": response},
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e)
            )

    def _extract_message(self, input_data: Any) -> str:
        """提取消息内容"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return input_data.get("message", "")
        return str(input_data)


class IntelligentRouter:
    """
    智能 Router - 核心路由调度器

    功能：
    1. 智能分析输入，决定路由路径
    2. 并行执行多个 Agent
    3. 聚合结果，返回最优输出
    """

    # 路由关键词映射
    ROUTE_KEYWORDS = {
        RouteType.DOCUMENT_ANALYSIS: [
            "文档", "文件", "解析", "docx", "pdf", "word",
            "格式", "结构", "章节", "段落"
        ],
        RouteType.KNOWLEDGE_RETRIEVAL: [
            "查询", "搜索", "检索", "知识", "库", "相关",
            "类似", "匹配", "向量"
        ],
        RouteType.FORMAT_GENERATION: [
            "生成", "创建", "公文", "报告", "模板",
            "格式", "规范", "排版", "样式"
        ],
        RouteType.CODE_GENERATION: [
            "代码", "编程", "函数", "类", "python",
            "开发", "实现", "算法", "脚本"
        ],
        RouteType.GENERAL_CHAT: [
            "你好", "帮助", "问题", "咨询", "聊天"
        ]
    }

    def __init__(self):
        self.agents: Dict[RouteType, BaseAgent] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._register_default_agents()

    def _register_default_agents(self):
        """注册默认 Agent"""
        self.register_agent(DocumentAnalysisAgent())
        self.register_agent(KnowledgeRetrievalAgent())
        self.register_agent(FormatGenerationAgent())
        self.register_agent(CodeGenerationAgent())
        self.register_agent(GeneralChatAgent())

    def register_agent(self, agent: BaseAgent):
        """注册 Agent"""
        self.agents[agent.route_type] = agent
        return self

    def analyze_input(self, input_data: Any) -> RouteDecision:
        """
        智能分析输入，决定路由路径

        策略：
        1. 基于关键词匹配
        2. 基于文件扩展名
        3. 基于输入数据结构
        """
        text = self._extract_text(input_data).lower()

        # 计算各路由类型的匹配分数
        scores: Dict[RouteType, float] = {}
        matched_keywords: Dict[RouteType, List[str]] = {}

        for route_type, keywords in self.ROUTE_KEYWORDS.items():
            score = 0
            matched = []
            for keyword in keywords:
                if keyword.lower() in text:
                    score += 1
                    matched.append(keyword)
            scores[route_type] = score
            matched_keywords[route_type] = matched

        # 检测文件类型
        file_type_score = self._detect_file_type(input_data)
        if file_type_score:
            scores[file_type_score] = scores.get(file_type_score, 0) + 3

        # 选择最佳路径
        sorted_routes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary_route = sorted_routes[0][0] if sorted_routes else RouteType.GENERAL_CHAT
        primary_score = sorted_routes[0][1] if sorted_routes else 0

        # 选择次要路径（分数 > 0 且不是主路径）
        secondary_routes = [
            route for route, score in sorted_routes[1:]
            if score > 0 and score >= primary_score * 0.5
        ]

        # 计算置信度
        total_score = sum(scores.values())
        confidence = primary_score / total_score if total_score > 0 else 0.5

        # 生成决策理由
        reasoning = self._generate_reasoning(
            primary_route, matched_keywords[primary_route], confidence
        )

        return RouteDecision(
            primary_route=primary_route,
            secondary_routes=secondary_routes[:2],  # 最多 2 个次要路径
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "all_scores": {k.name: v for k, v in scores.items()},
                "matched_keywords": {k.name: v for k, v in matched_keywords.items() if v}
            }
        )

    def _extract_text(self, input_data: Any) -> str:
        """从输入数据中提取文本"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return json.dumps(input_data, ensure_ascii=False)
        return str(input_data)

    def _detect_file_type(self, input_data: Any) -> Optional[RouteType]:
        """检测文件类型"""
        if isinstance(input_data, str):
            lower_input = input_data.lower()
            if any(ext in lower_input for ext in ['.docx', '.pdf', '.doc', '.txt']):
                return RouteType.DOCUMENT_ANALYSIS
            elif any(ext in lower_input for ext in ['.py', '.js', '.java', '.cpp']):
                return RouteType.CODE_GENERATION
        return None

    def _generate_reasoning(self, route: RouteType, keywords: List[str], confidence: float) -> str:
        """生成决策理由"""
        if keywords:
            return f"检测到关键词 [{', '.join(keywords[:3])}]，匹配度 {confidence:.1%}"
        return f"基于默认规则选择 {route.name}，置信度 {confidence:.1%}"

    async def execute_single(self, agent: BaseAgent, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        """执行单个 Agent（带超时）"""
        try:
            return await asyncio.wait_for(
                agent.process(input_data, context),
                timeout=agent.timeout
            )
        except asyncio.TimeoutError:
            return AgentResult(
                route_type=agent.route_type,
                success=False,
                output=None,
                execution_time=agent.timeout,
                error=f"执行超时（>{agent.timeout}s）"
            )
        except Exception as e:
            return AgentResult(
                route_type=agent.route_type,
                success=False,
                output=None,
                execution_time=0,
                error=str(e)
            )

    async def route(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        主路由方法 - 智能判断并执行

        执行流程：
        1. 分析输入，决定路由路径
        2. 并行执行主路径 + 次要路径
        3. 聚合结果
        """
        context = context or {}

        # 1. 分析输入，决定路由
        decision = self.analyze_input(input_data)
        print(f"🎯 路由决策: {decision.primary_route.name}")
        print(f"   理由: {decision.reasoning}")
        if decision.secondary_routes:
            print(f"   次要路径: {[r.name for r in decision.secondary_routes]}")

        # 2. 准备并行执行
        routes_to_execute = [decision.primary_route] + decision.secondary_routes
        tasks = []

        for route_type in routes_to_execute:
            agent = self.agents.get(route_type)
            if agent:
                task = self.execute_single(agent, input_data, context)
                tasks.append(task)

        # 3. 并行执行所有 Agent
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # 4. 处理结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                route_type = routes_to_execute[i]
                processed_results.append(AgentResult(
                    route_type=route_type,
                    success=False,
                    output=None,
                    execution_time=0,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        # 5. 聚合结果
        final_output = self._aggregate_results(decision, processed_results)

        return {
            "decision": {
                "primary_route": decision.primary_route.name,
                "secondary_routes": [r.name for r in decision.secondary_routes],
                "confidence": decision.confidence,
                "reasoning": decision.reasoning
            },
            "results": [
                {
                    "route": r.route_type.name,
                    "success": r.success,
                    "execution_time": round(r.execution_time, 3),
                    "error": r.error,
                    "output": r.output if r.success else None
                }
                for r in processed_results
            ],
            "final_output": final_output,
            "total_execution_time": round(total_time, 3),
            "parallel_count": len(tasks)
        }

    def _aggregate_results(self, decision: RouteDecision, results: List[AgentResult]) -> Any:
        """
        聚合多个 Agent 的结果

        策略：
        1. 主路径成功 -> 使用主路径结果
        2. 主路径失败 -> 尝试次要路径
        3. 合并多个成功结果
        """
        # 按路由类型分组
        result_map = {r.route_type: r for r in results}

        # 优先使用主路径
        primary_result = result_map.get(decision.primary_route)
        if primary_result and primary_result.success:
            return primary_result.output

        # 主路径失败，尝试次要路径
        for route_type in decision.secondary_routes:
            result = result_map.get(route_type)
            if result and result.success:
                return result.output

        # 所有路径都失败，返回错误信息
        return {
            "error": "所有执行路径均失败",
            "details": [
                {"route": r.route_type.name, "error": r.error}
                for r in results if not r.success
            ]
        }

    async def execute_parallel(self, tasks_config: List[Dict[str, Any]]) -> List[AgentResult]:
        """
        并行执行多个自定义任务

        tasks_config: [
            {"route_type": RouteType, "input": ..., "context": {...}},
            ...
        ]
        """
        tasks = []
        for config in tasks_config:
            route_type = config["route_type"]
            input_data = config.get("input", "")
            context = config.get("context", {})

            agent = self.agents.get(route_type)
            if agent:
                task = self.execute_single(agent, input_data, context)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                route_type = tasks_config[i]["route_type"]
                processed.append(AgentResult(
                    route_type=route_type,
                    success=False,
                    output=None,
                    execution_time=0,
                    error=str(result)
                ))
            else:
                processed.append(result)

        return processed


# ==================== 使用示例 ====================

async def demo():
    """演示智能 Router 的使用"""

    print("=" * 70)
    print("🚀 智能 Router 演示 - asyncio 并行执行")
    print("=" * 70)

    router = IntelligentRouter()

    # 示例 1: 文档分析请求
    print("\n📄 示例 1: 文档分析")
    print("-" * 40)
    result = await router.route("请解析这个 .docx 文件的结构和格式")
    print(f"主路径: {result['decision']['primary_route']}")
    print(f"并行执行: {result['parallel_count']} 个 Agent")
    print(f"总耗时: {result['total_execution_time']}s")
    print(f"各路径耗时:")
    for r in result['results']:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['route']}: {r['execution_time']}s")

    # 示例 2: 知识检索请求
    print("\n🔍 示例 2: 知识检索")
    print("-" * 40)
    result = await router.route("从知识库中检索与人工智能相关的文档")
    print(f"主路径: {result['decision']['primary_route']}")
    print(f"置信度: {result['decision']['confidence']:.1%}")
    print(f"理由: {result['decision']['reasoning']}")

    # 示例 3: 格式生成请求
    print("\n📝 示例 3: 格式生成")
    print("-" * 40)
    result = await router.route(
        "生成一份符合公文格式的报告",
        context={"format_spec": {"font": "仿宋", "size": "三号"}}
    )
    print(f"主路径: {result['decision']['primary_route']}")
    print(f"次要路径: {result['decision']['secondary_routes']}")

    # 示例 4: 并行自定义任务
    print("\n⚡ 示例 4: 并行自定义任务")
    print("-" * 40)
    custom_tasks = [
        {"route_type": RouteType.DOCUMENT_ANALYSIS, "input": "文件1.docx"},
        {"route_type": RouteType.KNOWLEDGE_RETRIEVAL, "input": "查询1"},
        {"route_type": RouteType.FORMAT_GENERATION, "input": "生成内容"},
    ]
    parallel_results = await router.execute_parallel(custom_tasks)
    print(f"并行执行 {len(parallel_results)} 个任务:")
    for r in parallel_results:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.route_type.name}: {r.execution_time:.3f}s")

    print("\n" + "=" * 70)
    print("✅ 演示完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())
