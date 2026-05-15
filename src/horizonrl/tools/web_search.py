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


def resolve_search_provider(configured: str = "auto") -> SearchProvider:
    """根据配置 + 环境变量 + 可用 Key 解析搜索提供商。

    优先级：
        1. 环境变量 HORIZON_SEARCH_PROVIDER (最高)
        2. 构造函数参数 / YAML 配置
        3. auto 模式：Bocha → Brave → DuckDuckGo → Wikipedia(→Mock)

    Args:
        configured: 来自 YAML 配置或构造函数的 provider 偏好。

    Returns:
        实际使用的 SearchProvider。
    """
    # 环境变量覆盖一切
    configured = os.getenv("HORIZON_SEARCH_PROVIDER", configured).lower()

    if configured == "mock":
        return SearchProvider.MOCK
    if configured == "bocha":
        if os.getenv("BOCHA_API_KEY"):
            return SearchProvider.BOCHA
        return SearchProvider.DUCKDUCKGO  # 无 Key 降级
    if configured == "brave":
        if os.getenv("BRAVE_API_KEY"):
            return SearchProvider.BRAVE
        return SearchProvider.DUCKDUCKGO  # 无 Key 降级
    if configured == "duckduckgo":
        return SearchProvider.DUCKDUCKGO

    # auto 模式: 按 Bocha → Brave → DDGS 优先级
    if os.getenv("BOCHA_API_KEY"):
        return SearchProvider.BOCHA
    if os.getenv("BRAVE_API_KEY"):
        return SearchProvider.BRAVE
    return SearchProvider.DUCKDUCKGO


class WebSearchTool:
    """Multi-backend web search with automatic fallback + provider tracking.

    支持通过配置文件或环境变量控制搜索后端。优先级：
      1. 构造函数参数 provider
      2. 环境变量 HORIZON_SEARCH_PROVIDER
      3. auto 模式：Bocha → Brave → DuckDuckGo → Wikipedia → Mock

    Attributes:
        actual_provider: 实际处理本次搜索的 provider (搜索后设置)。
    """

    name = "web_search"
    description = "Search the web for information on a given query."

    def __init__(
        self,
        brave_api_key: str | None = None,
        bocha_api_key: str | None = None,
        provider: str = "auto",
    ):
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY", "")
        self.bocha_api_key = bocha_api_key or os.getenv("BOCHA_API_KEY", "")
        self._configured_provider = provider  # 来自配置文件的 provider 偏好
        self.actual_provider: str = ""

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        """Execute web search with concurrent backend race in AUTO mode.

        AUTO mode: all available backends fire simultaneously, first success wins.
        Explicit provider: sequential fallback (Bocha→DDGS, Brave→DDGS, etc.)
        """
        num_results = min(num_results, 10)
        provider = resolve_search_provider(self._configured_provider)

        # ── AUTO 模式：并发竞速所有可用后端 ──
        if provider == SearchProvider.AUTO:
            result = await self._race_auto_backends(query, num_results)
            if result:
                return result
            # AUTO race failed → Wikipedia → Mock
            wiki = await self._try_backend(self._wikipedia_search, query, num_results, "Wikipedia")
            if wiki:
                self.actual_provider = "wikipedia"
                return wiki
            self.actual_provider = "mock"
            return self._mock_search(query, num_results)

        # ── 显式 provider 模式：顺序回退 ──
        if provider == SearchProvider.BOCHA and self.bocha_api_key:
            results = await self._try_backend(self._bocha_search, query, num_results, "Bocha")
            if results:
                self.actual_provider = "bocha"
                return results

        if provider == SearchProvider.BRAVE and self.brave_api_key:
            results = await self._try_backend(self._brave_search, query, num_results, "Brave")
            if results:
                self.actual_provider = "brave"
                return results

        if provider == SearchProvider.DUCKDUCKGO:
            results = await self._try_backend(self._ddgs_search, query, num_results, "DDGS")
            if results:
                self.actual_provider = "duckduckgo"
                return results

        results = await self._try_backend(self._wikipedia_search, query, num_results, "Wikipedia")
        if results:
            self.actual_provider = "wikipedia"
            return results

        self.actual_provider = "mock"
        return self._mock_search(query, num_results)

    async def _race_auto_backends(
        self, query: str, n: int
    ) -> list[dict[str, str]] | None:
        """Race all available backends concurrently. First valid response wins."""
        async def _try(name: str, fn):
            try:
                results = await asyncio.wait_for(fn(query, n), timeout=5.0)
                if results and any(r.get("title") for r in results):
                    return (name, results)
            except Exception:
                pass
            return None

        cors = []
        if self.bocha_api_key:
            cors.append(_try("bocha", self._bocha_search))
        if self.brave_api_key:
            cors.append(_try("brave", self._brave_search))
        cors.append(_try("duckduckgo", self._ddgs_search))

        tasks = [asyncio.create_task(c) for c in cors]
        done, pending = await asyncio.wait(
            tasks, timeout=6.0, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()

        for t in done:
            try:
                pair = t.result()
                if pair is not None:
                    name, results = pair
                    self.actual_provider = name
                    return results
            except Exception:
                pass
        return None

    async def _try_backend(self, fn, query, n, name) -> list[dict[str, str]] | None:
        try:
            results = await asyncio.wait_for(fn(query, n), timeout=6.0)
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
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.post(
                "https://api.bocha.cn/v1/web-search",
                json={"query": query, "count": n},
                headers={"Authorization": f"Bearer {self.bocha_api_key}"},
            )
            data = response.json()
            results = []
            web_pages = data.get("data", {}).get("webPages", {})
            items = web_pages.get("value", []) if isinstance(web_pages, dict) else []
            for r in items[:n]:
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
        """Synchronous interface — safe from any context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.search(query))
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, self.search(query)).result()
