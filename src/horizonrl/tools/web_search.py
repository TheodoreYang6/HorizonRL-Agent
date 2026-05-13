"""
Web search tool — multi-backend with graceful fallback + provider routing.

Provider 优先级 (auto 模式):
    bocha → brave → duckduckgo → mock

通过环境变量控制:
    BOCHA_API_KEY / BRAVE_API_KEY — 对应 provider 的 Key
    HORIZON_SEARCH_PROVIDER — auto | bocha | brave | duckduckgo | mock
"""

from __future__ import annotations

import asyncio
import os
from enum import Enum


class SearchProvider(str, Enum):
    """搜索提供商枚举。"""
    AUTO = "auto"
    BOCHA = "bocha"
    BRAVE = "brave"
    DUCKDUCKGO = "duckduckgo"
    MOCK = "mock"


def resolve_search_provider() -> SearchProvider:
    """根据环境变量和可用 Key 解析当前应该使用的搜索提供商。

    优先级 (auto 模式):
        1. BOCHA_API_KEY 存在 → bocha
        2. BRAVE_API_KEY 存在 → brave
        3. 尝试 duckduckgo
        4. 回退 mock

    Returns:
        实际使用的 SearchProvider。
    """
    configured = os.getenv("HORIZON_SEARCH_PROVIDER", "auto").lower()

    if configured == "mock":
        return SearchProvider.MOCK
    if configured == "bocha" and os.getenv("BOCHA_API_KEY"):
        return SearchProvider.BOCHA
    if configured == "brave" and os.getenv("BRAVE_API_KEY"):
        return SearchProvider.BRAVE
    if configured == "duckduckgo":
        return SearchProvider.DUCKDUCKGO

    if configured == "bocha":
        return SearchProvider.DUCKDUCKGO  # 无 Key 降级
    if configured == "brave":
        return SearchProvider.DUCKDUCKGO  # 无 Key 降级

    # auto 模式: 按优先级
    if os.getenv("BOCHA_API_KEY"):
        return SearchProvider.BOCHA
    if os.getenv("BRAVE_API_KEY"):
        return SearchProvider.BRAVE
    return SearchProvider.DUCKDUCKGO  # ddgs → 失败自动到 mock


class WebSearchTool:
    """Multi-backend web search with automatic fallback + provider tracking.

    Attributes:
        actual_provider: 实际处理本次搜索的 provider (搜索后设置)。
    """

    name = "web_search"
    description = "Search the web for information on a given query."

    def __init__(self, brave_api_key: str | None = None):
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY", "")
        self.bocha_api_key = os.getenv("BOCHA_API_KEY", "")
        self.actual_provider: str = ""  # 搜索后记录实际使用的后端

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        """Execute web search with automatic backend fallback.

        搜索后 self.actual_provider 记录实际使用的后端。

        Args:
            query: Search query string.
            num_results: Number of results to return (max 10).

        Returns:
            List of {'title', 'url', 'snippet', 'provider', 'is_mock'} dicts.
        """
        num_results = min(num_results, 10)
        provider = resolve_search_provider()

        # Backend 1: Bocha (国内推荐)
        if provider == SearchProvider.BOCHA and self.bocha_api_key:
            results = await self._try_backend(self._bocha_search, query, num_results, "Bocha")
            if results:
                self.actual_provider = "bocha"
                return results

        # Backend 2: Brave Search API
        if provider in (SearchProvider.BRAVE, SearchProvider.AUTO) and self.brave_api_key:
            results = await self._try_backend(self._brave_search, query, num_results, "Brave")
            if results:
                self.actual_provider = "brave"
                return results

        # Backend 3: DDGS (国内可用)
        if provider in (SearchProvider.DUCKDUCKGO, SearchProvider.AUTO):
            results = await self._try_backend(self._ddgs_search, query, num_results, "DDGS")
            if results:
                self.actual_provider = "duckduckgo"
                return results

        # Backend 4: Wikipedia API
        results = await self._try_backend(self._wikipedia_search, query, num_results, "Wikipedia")
        if results:
            self.actual_provider = "wikipedia"
            return results

        # Backend 5: Mock (always available)
        self.actual_provider = "mock"
        return self._mock_search(query, num_results)

    async def _try_backend(self, fn, query, n, name) -> list[dict[str, str]] | None:
        try:
            results = await asyncio.wait_for(fn(query, n), timeout=8.0)
            if results and any(r.get("title") for r in results):
                return results
        except Exception:
            pass
        return None

    # ── Backend implementations ──────────────────────────────────────────

    async def _bocha_search(
        self, query: str, n: int
    ) -> list[dict[str, str]]:
        """Bocha API 搜索 (国内推荐)。需要 BOCHA_API_KEY。"""
        import httpx
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                "https://api.bochaai.com/v1/web/search",
                json={"query": query, "count": n},
                headers={"Authorization": f"Bearer {self.bocha_api_key}"},
            )
            data = response.json()
            results = []
            for r in data.get("data", {}).get("webPages", [])[:n]:
                results.append({
                    "title": r.get("name", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", ""),
                    "provider": "bocha",
                    "is_mock": False,
                })
            return results

    async def _brave_search(
        self, query: str, n: int
    ) -> list[dict[str, str]]:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": n},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.brave_api_key,
                },
            )
            data = response.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("description", ""),
                    "provider": "brave",
                    "is_mock": False,
                }
                for r in data.get("web", {}).get("results", [])[:n]
            ]

    async def _ddgs_search(
        self, query: str, n: int
    ) -> list[dict[str, str]]:
        from ddgs import DDGS

        loop = asyncio.get_running_loop()

        def _search():
            return list(DDGS().text(query, max_results=n))

        results = await loop.run_in_executor(None, _search)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "provider": "duckduckgo",
                "is_mock": False,
            }
            for r in results
        ]

    async def _wikipedia_search(
        self, query: str, n: int
    ) -> list[dict[str, str]]:
        import httpx

        async with httpx.AsyncClient(timeout=8.0) as client:
            # Search for pages
            r = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": n,
                },
                headers={"User-Agent": "HorizonRL-Agent/0.1"},
            )
            data = r.json()
            pages = data.get("query", {}).get("search", [])

            results = []
            for p in pages[:n]:
                # Get a short extract for each result
                try:
                    r2 = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "query",
                            "prop": "extracts",
                            "exintro": True,
                            "explaintext": True,
                            "pageids": p["pageid"],
                            "format": "json",
                        },
                        headers={"User-Agent": "HorizonRL-Agent/0.1"},
                    )
                    pages_data = r2.json().get("query", {}).get("pages", {})
                    page = pages_data.get(str(p["pageid"]), {})
                    snippet = page.get("extract", p.get("snippet", ""))[:300]
                except Exception:
                    snippet = p.get("snippet", "")

                # Clean HTML tags from snippet
                import re
                snippet = re.sub(r'<[^>]+>', '', snippet)

                results.append({
                    "title": p["title"],
                    "url": f"https://en.wikipedia.org/wiki/{p['title'].replace(' ', '_')}",
                    "snippet": snippet,
                    "provider": "wikipedia",
                    "is_mock": False,
                })
            return results

    def _mock_search(
        self, query: str, n: int
    ) -> list[dict[str, str]]:
        """Mock search with helpful status messages."""
        return [
            {
                "title": f"[Mock] 搜索结果 {i+1}: {query[:40]}",
                "url": f"https://mock-search.local/result-{i+1}",
                "snippet": (
                    f"这是关于 '{query[:60]}' 的模拟搜索结果 #{i+1}。"
                    f"内容涵盖相关概念、方法、应用场景与最新进展。"
                ),
                "provider": "mock",
                "is_mock": True,
            }
            for i in range(n)
        ]

    def __call__(self, query: str) -> list[dict[str, str]]:
        """Synchronous interface."""
        return asyncio.run(self.search(query))
