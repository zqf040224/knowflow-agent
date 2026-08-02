"""
历史说明：本文件是早期 IntelligentRouter 集成演示，不属于当前 Flask 生产主线。
当前生产聊天主链路是 /api/chat -> ChatGraphRuntime -> IntentRouter -> stream service。

智能 Router 集成示例 - 结合知识库和文档解析

展示如何：
1. 并行处理多个文档
2. 智能路由到不同处理流程
3. 与现有知识库系统集成

归档说明：本文件保留早期 router 架构实验，不属于当前 Flask/LangGraph 主服务。
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from intelligent_router import (
    IntelligentRouter,
    RouteType,
    BaseAgent,
    AgentResult
)
from document_parser import parse_document_with_format, FormatFingerprint


class KnowledgeBaseBuildAgent(BaseAgent):
    """知识库构建 Agent - 批量处理文档并构建向量索引"""

    def __init__(self, output_dir: str = "./knowledge_base"):
        super().__init__("知识库构建器", RouteType.KNOWLEDGE_RETRIEVAL)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process(self, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        """
        处理知识库构建任务

        input_data: 可以是文件路径、文件列表或目录路径
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # 提取文件列表
            files = self._extract_files(input_data)
            if not files:
                return AgentResult(
                    route_type=self.route_type,
                    success=False,
                    output=None,
                    execution_time=0,
                    error="未找到有效文件"
                )

            print(f"   📂 准备处理 {len(files)} 个文件...")

            # 并行解析所有文档
            parsed_docs = await self._parse_documents_parallel(files)

            # 统计结果
            successful = sum(1 for d in parsed_docs if d.get("success"))
            failed = len(parsed_docs) - successful

            result = {
                "total_files": len(files),
                "successful": successful,
                "failed": failed,
                "documents": parsed_docs,
                "output_dir": str(self.output_dir)
            }

            # 保存解析结果
            await self._save_results(result)

            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=True,
                output=result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResult(
                route_type=self.route_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e)
            )

    def _extract_files(self, input_data: Any) -> List[Path]:
        """从输入数据中提取文件列表"""
        files = []

        if isinstance(input_data, str):
            path = Path(input_data)
            if path.is_dir():
                files = list(path.rglob("*.docx")) + list(path.rglob("*.pdf"))
            elif path.is_file():
                files = [path]

        elif isinstance(input_data, list):
            files = [Path(f) for f in input_data if Path(f).exists()]

        elif isinstance(input_data, dict):
            file_paths = input_data.get("files", [])
            files = [Path(f) for f in file_paths if Path(f).exists()]

        return files

    async def _parse_documents_parallel(self, files: List[Path]) -> List[Dict]:
        """并行解析多个文档"""
        tasks = [self._parse_single_document(f) for f in files]
        return await asyncio.gather(*tasks)

    async def _parse_single_document(self, file_path: Path) -> Dict:
        """解析单个文档（在单独的线程中运行阻塞操作）"""
        try:
            # 使用 run_in_executor 运行阻塞的文档解析
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # 使用默认 executor
                self._parse_blocking,
                file_path
            )
            return {
                "file": str(file_path),
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "file": str(file_path),
                "success": False,
                "error": str(e)
            }

    def _parse_blocking(self, file_path: Path) -> Dict:
        """阻塞的文档解析操作"""
        formatted = parse_document_with_format(file_path)
        return {
            "chunks": len(formatted),
            "has_format": any(f.get("format") for f in formatted)
        }

    async def _save_results(self, result: Dict):
        """异步保存结果"""
        output_file = self.output_dir / f"parsed_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._write_json,
            output_file,
            result
        )
        print(f"   💾 结果已保存: {output_file}")

    def _write_json(self, path: Path, data: Dict):
        """阻塞的 JSON 写入"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class BatchDocumentProcessor:
    """批量文档处理器 - 展示并行处理的高级用法"""

    def __init__(self, max_workers: int = 5):
        self.router = IntelligentRouter()
        self.max_workers = max_workers
        self._register_custom_agents()

    def _register_custom_agents(self):
        """注册自定义 Agent"""
        self.router.register_agent(KnowledgeBaseBuildAgent())

    async def process_directory(self, input_dir: str) -> Dict[str, Any]:
        """
        处理整个目录的文档

        流程：
        1. 扫描目录获取文件列表
        2. 并行分析每个文件的类型和内容
        3. 根据分析结果路由到不同的处理流程
        4. 聚合结果
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            return {"error": f"目录不存在: {input_dir}"}

        print(f"\n📂 扫描目录: {input_dir}")

        # 获取所有文档文件
        files = self._scan_documents(input_path)
        print(f"   找到 {len(files)} 个文档")

        if not files:
            return {"error": "未找到文档文件"}

        # 并行分析所有文件
        print("\n🔍 并行分析文档...")
        analysis_tasks = [
            self.router.route(
                f"分析文档: {f.name}",
                context={"file_path": str(f), "phase": "analysis"}
            )
            for f in files[:3]  # 限制示例数量为 3
        ]

        analysis_results = await asyncio.gather(*analysis_tasks)

        # 汇总分析结果
        print("\n📊 分析结果汇总:")
        route_distribution = {}
        for result in analysis_results:
            route = result["decision"]["primary_route"]
            route_distribution[route] = route_distribution.get(route, 0) + 1

        for route, count in route_distribution.items():
            print(f"   {route}: {count} 个文件")

        # 执行知识库构建
        print("\n🔨 构建知识库...")
        build_result = await self.router.route(
            {"files": [str(f) for f in files[:3]]},
            context={"phase": "build", "output_dir": "./kb_output"}
        )

        return {
            "total_files": len(files),
            "analyzed": len(analysis_results),
            "route_distribution": route_distribution,
            "build_result": build_result,
            "execution_summary": self._summarize_results(analysis_results)
        }

    def _scan_documents(self, input_path: Path) -> List[Path]:
        """扫描文档文件"""
        extensions = [".pdf", ".docx", ".doc", ".txt", ".md"]
        files = []
        for ext in extensions:
            files.extend(input_path.rglob(f"*{ext}"))
        return files

    def _summarize_results(self, results: List[Dict]) -> Dict:
        """汇总执行结果"""
        total_time = sum(r.get("total_execution_time", 0) for r in results)
        avg_time = total_time / len(results) if results else 0

        return {
            "total_execution_time": round(total_time, 3),
            "average_time_per_file": round(avg_time, 3),
            "parallel_agents_used": sum(r.get("parallel_count", 0) for r in results)
        }


class SmartWorkflow:
    """
    智能工作流 - 展示复杂的多步骤并行处理

    场景：公文生成工作流
    1. 同时检索相关知识 + 分析模板格式
    2. 等待两者完成后，生成符合格式的内容
    3. 并行验证内容 + 优化格式
    """

    def __init__(self):
        self.router = IntelligentRouter()

    async def generate_official_document(
        self,
        topic: str,
        template_file: str,
        knowledge_query: str
    ) -> Dict[str, Any]:
        """
        生成公文的工作流

        阶段 1: 并行获取模板格式和检索知识
        阶段 2: 根据结果生成内容
        阶段 3: 并行验证和优化
        """
        print(f"\n📝 公文生成工作流")
        print(f"   主题: {topic}")
        print("=" * 50)

        # 阶段 1: 并行获取输入
        print("\n📥 阶段 1: 并行获取模板格式和知识...")
        stage1_tasks = [
            self.router.route(
                f"分析模板文件: {template_file}",
                context={"file_path": template_file, "phase": "template_analysis"}
            ),
            self.router.route(
                f"检索知识: {knowledge_query}",
                context={"query": knowledge_query, "phase": "knowledge_retrieval"}
            )
        ]

        stage1_results = await asyncio.gather(*stage1_tasks)
        template_result, knowledge_result = stage1_results

        print(f"   ✓ 模板分析完成 ({template_result['total_execution_time']}s)")
        print(f"   ✓ 知识检索完成 ({knowledge_result['total_execution_time']}s)")

        # 阶段 2: 生成内容
        print("\n✍️ 阶段 2: 生成公文内容...")

        # 提取格式规范和知识内容
        format_spec = self._extract_format_spec(template_result)
        knowledge_content = self._extract_knowledge(knowledge_result)

        generation_result = await self.router.route(
            f"生成公文: {topic}",
            context={
                "format_spec": format_spec,
                "knowledge": knowledge_content,
                "topic": topic,
                "phase": "content_generation"
            }
        )

        print(f"   ✓ 内容生成完成 ({generation_result['total_execution_time']}s)")

        # 阶段 3: 并行验证和优化
        print("\n🔍 阶段 3: 并行验证和优化...")
        stage3_tasks = [
            self.router.route(
                "验证格式规范",
                context={
                    "content": generation_result["final_output"],
                    "format_spec": format_spec,
                    "phase": "format_validation"
                }
            ),
            self.router.route(
                "优化语言表达",
                context={
                    "content": generation_result["final_output"],
                    "phase": "language_optimization"
                }
            )
        ]

        stage3_results = await asyncio.gather(*stage3_tasks)
        validation_result, optimization_result = stage3_results

        print(f"   ✓ 格式验证完成 ({validation_result['total_execution_time']}s)")
        print(f"   ✓ 语言优化完成 ({optimization_result['total_execution_time']}s)")

        # 汇总工作流结果
        total_time = (
            template_result['total_execution_time'] +
            knowledge_result['total_execution_time'] +
            generation_result['total_execution_time'] +
            validation_result['total_execution_time'] +
            optimization_result['total_execution_time']
        )

        return {
            "workflow": "official_document_generation",
            "topic": topic,
            "phases": {
                "stage1_input_gathering": {
                    "template_analysis": template_result,
                    "knowledge_retrieval": knowledge_result
                },
                "stage2_generation": generation_result,
                "stage3_validation": {
                    "format_validation": validation_result,
                    "optimization": optimization_result
                }
            },
            "final_output": generation_result.get("final_output"),
            "total_workflow_time": round(total_time, 3),
            "parallel_efficiency": "66.7%"  # 3个阶段中2个使用了并行
        }

    def _extract_format_spec(self, template_result: Dict) -> Dict:
        """从模板分析结果中提取格式规范"""
        # 简化示例
        return {
            "font": "仿宋",
            "size": "三号",
            "alignment": "两端对齐"
        }

    def _extract_knowledge(self, knowledge_result: Dict) -> str:
        """从检索结果中提取知识内容"""
        return "相关知识内容..."


async def demo_batch_processing():
    """演示批量处理"""
    print("\n" + "=" * 70)
    print("📦 批量文档处理演示")
    print("=" * 70)

    processor = BatchDocumentProcessor(max_workers=5)

    # 使用当前目录作为示例（如果没有文档，会返回空结果）
    result = await processor.process_directory("./")

    print("\n📊 处理结果:")
    print(f"   总文件数: {result.get('total_files', 0)}")
    print(f"   已分析: {result.get('analyzed', 0)}")

    summary = result.get('execution_summary', {})
    print(f"\n⏱️ 执行统计:")
    print(f"   总耗时: {summary.get('total_execution_time', 0)}s")
    print(f"   平均每个文件: {summary.get('average_time_per_file', 0)}s")
    print(f"   并行 Agent 数: {summary.get('parallel_agents_used', 0)}")


async def demo_workflow():
    """演示智能工作流"""
    print("\n" + "=" * 70)
    print("🔄 智能工作流演示")
    print("=" * 70)

    workflow = SmartWorkflow()

    result = await workflow.generate_official_document(
        topic="关于加强安全生产工作的通知",
        template_file="template.docx",
        knowledge_query="安全生产规范要求"
    )

    print("\n" + "=" * 70)
    print("📋 工作流总结")
    print("=" * 70)
    print(f"主题: {result['topic']}")
    print(f"总耗时: {result['total_workflow_time']}s")
    print(f"并行效率: {result['parallel_efficiency']}")


async def demo_simple_routing():
    """演示简单路由"""
    print("\n" + "=" * 70)
    print("🎯 简单路由演示")
    print("=" * 70)

    router = IntelligentRouter()

    test_inputs = [
        "解析这个 Word 文档的结构",
        "从知识库检索人工智能相关资料",
        "生成一份符合公文格式的报告",
        "写一个 Python 函数来计算斐波那契数列",
        "你好，请帮我解答一个问题"
    ]

    print("\n🔍 路由测试:")
    for i, input_text in enumerate(test_inputs, 1):
        decision = router.analyze_input(input_text)
        print(f"\n   {i}. 输入: {input_text[:30]}...")
        print(f"      主路径: {decision.primary_route.name}")
        print(f"      置信度: {decision.confidence:.1%}")
        print(f"      理由: {decision.reasoning}")


async def main():
    """主函数"""
    # 1. 简单路由演示
    await demo_simple_routing()

    # 2. 批量处理演示
    await demo_batch_processing()

    # 3. 智能工作流演示
    await demo_workflow()

    print("\n" + "=" * 70)
    print("✅ 所有演示完成")
    print("=" * 70)


if __name__ == "__main__":
    print("此 Router 集成示例已归档，不作为当前主服务运行入口。")
    raise SystemExit(0)
    asyncio.run(main())
