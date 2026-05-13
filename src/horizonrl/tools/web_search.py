"""Web search tool using Brave Search API or DuckDuckGo fallback."""

from __future__ import annotations

import asyncio
import os
from typing import Any


class WebSearchTool:
    """Search the web and return results.

    Uses Brave Search API when available, falls back to DuckDuckGo.
    """

    name = "web_search"
    description = "Search the web for information on a given query."

    def __init__(self, brave_api_key: str | None = None):
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY", "")

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        """Execute a web search asynchronously.

        Args:
            query: The search query string.
            num_results: Number of results to return.

        Returns:
            List of results with 'title', 'url', 'snippet' keys.
        """
        if self.brave_api_key:
            return await self._brave_search(query, num_results)
        return await self._duckduckgo_search(query, num_results)

    async def _brave_search(self, query: str, num_results: int) -> list[dict[str, str]]:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": num_results},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.brave_api_key,
                },
            )
            data = response.json()
            results = []
            for r in data.get("web", {}).get("results", [])[:num_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("description", ""),
                })
            return results

    async def _duckduckgo_search(
        self, query: str, num_results: int
    ) -> list[dict[str, str]]:
        try:
            from duckduckgo_search import DDGS

            loop = asyncio.get_running_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: list(DDGS().text(query, max_results=num_results)),
                ),
                timeout=5.0,  # 国内网络环境快速超时
            )
            return [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in results
            ]
        except (ImportError, asyncio.TimeoutError):
            return [{"title": "搜索服务暂不可用", "url": "",
                     "snippet": "网络连接超时或 DuckDuckGo 不可用。国内用户建议配置 Brave API Key 或使用代理。"}]
        except Exception as e:
            error_msg = str(e)[:200]
            if "ConnectError" in error_msg or "Connection" in error_msg:
                return [{"title": "网络连接失败", "url": "",
                         "snippet": f"搜索服务不可用。国内用户建议配置 Brave API Key 或使用代理。"}]
            return [{"title": "搜索失败", "url": "", "snippet": f"{error_msg[:200]}"}]

    def __call__(self, query: str) -> list[dict[str, str]]:
        """Synchronous interface."""
        return asyncio.run(self.search(query))
