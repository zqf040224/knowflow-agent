import os
import json
import logging
from typing import List
from agents.base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    搜索 Agent - 博查 Web Search API 联网搜索 + 本地知识库

    博查 API 端点：POST https://api.bochaai.com/v1/web-search
    兼容 Bing Search API 返回格式
    """

    BOCHA_URL = "https://api.bochaai.com/v1/web-search"

    def __init__(self, **kwargs):
        super().__init__(
            name="Search",
            description="信息搜索器 - 搜索最新政策和背景信息",
            **kwargs,
        )
        try:
            from memory import get_memory
            self.memory = get_memory()
        except Exception as exc:
            logger.debug("历史 memory 模块不可用，SearchAgent 将跳过旧记忆接口: %s", exc)
            self.memory = None

    def get_system_prompt(self) -> str:
        return """你是一个信息整理专家。你的职责是：
1. 整理搜索到的信息
2. 提取与公文写作相关的关键内容
3. 过滤无关信息
4. 按重要性排序

请以结构化方式输出整理后的信息。"""

    def process(self, input_data: dict, on_think=None) -> AgentResult:
        queries = input_data.get("search_queries", [])
        user_request = input_data.get("user_request", "")
        queries = queries[:3] if queries else [user_request[:50]]
        knowledge_agent = input_data.get("knowledge_agent")
        user_info = input_data.get("user_info")

        self._emit_think(on_think, "🔍", "开始搜索相关信息...")

        bocha_api_key = os.getenv("BOCHA_API_KEY", "")
        all_results = []
        web_used = False

        if bocha_api_key:
            for query in queries:
                self._emit_think(on_think, "🌐", f"联网搜索：{query}")
                try:
                    results = self._search_bocha(query, bocha_api_key)
                    if results:
                        all_results.extend(results)
                        web_used = True
                        self._emit_think(on_think, "📥", f"获取到 {len(results)} 条联网信息")
                except Exception as e:
                    self._emit_think(on_think, "⚠️", f"联网搜索失败：{str(e)[:50]}")
                    print(f"[SearchAgent] 博查错误: {e}")

        if knowledge_agent:
            self._emit_think(on_think, "📚", "搜索本地知识库...")
            local_results = self._search_local_with_agent(queries, user_request, knowledge_agent, user_info)
            all_results.extend(local_results)

        if not all_results:
            self._emit_think(on_think, "ℹ️", "未找到信息，将仅使用模型能力生成")
            return AgentResult(
                success=True,
                content="未搜索到相关信息，将基于模型能力生成。",
                agent_name=self.name,
                confidence=0.5,
                metadata={"search_results": [], "source": "fallback"},
            )

        self._emit_think(on_think, "📝", f"整理 {len(all_results)} 条搜索结果...")
        summary = self._summarize_results(all_results, user_request)

        sources = []
        if web_used:
            sources.append("bocha")
        if knowledge_agent:
            sources.append("local_kb")

        return AgentResult(
            success=True,
            content=summary,
            agent_name=self.name,
            confidence=0.8,
            metadata={
                "search_results": all_results,
                "result_count": len(all_results),
                "sources": sources,
                "web_used": web_used,
            },
        )

    def _search_local_with_agent(self, queries: List[str], user_request: str,
                                  knowledge_agent, user_info=None) -> List[dict]:
        """使用传入的 knowledge_agent 进行本地搜索，避免重复创建实例"""
        results = []
        try:
            for query in queries[:2]:  # 最多2个查询
                result = knowledge_agent.process({
                    "user_request": user_request,
                    "knowledge_queries": [query],
                    "user_info": user_info,
                })
                if result.metadata and "results" in result.metadata:
                    for item in result.metadata["results"]:
                        results.append({
                            "title": item.get("source", "本地知识"),
                            "content": item.get("text", "")[:500],
                            "url": "",
                            "score": item.get("similarity", 0.5),
                            "source_type": "local",
                        })
        except Exception as e:
            print(f"本地搜索失败: {e}")
        return results

    def _search_bocha(self, query: str, api_key: str, max_results: int = 5) -> list:
        """使用博查 Web Search API"""
        import requests

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "summary": True,
            "count": max_results,
            "freshness": "noLimit",
        }

        resp = requests.post(self.BOCHA_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
        for item in web_pages:
            results.append({
                "title": item.get("name", ""),
                "content": item.get("summary") or item.get("snippet", ""),
                "url": item.get("url", ""),
                "score": 1.0,
                "source_type": "bocha",
            })

        return results

    def _summarize_results(self, results: list, user_request: str) -> str:
        """整理搜索结果"""
        web_results = [r for r in results if r.get("source_type") == "bocha"]
        local_results = [r for r in results if r.get("source_type") == "local"]

        parts = []

        if web_results:
            parts.append("【联网搜索结果】")
            for r in web_results[:5]:
                source = f"（来源：{r['url']}）" if r.get("url") else ""
                parts.append(f"- {r['title']}：{r['content'][:300]}{source}")

        if local_results:
            parts.append("\n【知识库相关内容】")
            for r in local_results[:3]:
                parts.append(f"- [{r['title']}] {r['content'][:200]}...")

        return "\n".join(parts)
