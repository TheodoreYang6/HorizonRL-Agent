"""
Web search tool — multi-backend with graceful fallback.

Backends (tried in order):
    1. Brave Search API (if BRAVE_API_KEY set)
    2. DDGS (new duckduckgo_search, works in China)
    3. Wikipedia API (encyclopedic knowledge, works everywhere)
    4. Mock fallback (always available, offline-safe)

Each backend has a 5s timeout. On failure, automatically tries the next.
"""

from __future__ import annotations

import asyncio
import os


class WebSearchTool:
    """Multi-backend web search with automatic fallback.

    Examples:
        >>> tool = WebSearchTool()
        >>> results = await tool.search("Python asyncio")
        >>> results[0]["title"]
    """

    name = "web_search"
    description = "Search the web for information on a given query."

    def __init__(self, brave_api_key: str | None = None):
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY", "")

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        """Execute web search with automatic backend fallback.

        Args:
            query: Search query string.
            num_results: Number of results to return (max 10).

        Returns:
            List of {'title', 'url', 'snippet'} dicts.
        """
        num_results = min(num_results, 10)

        # Backend 1: Brave Search API
        if self.brave_api_key:
            results = await self._try_backend(
                self._brave_search, query, num_results, "Brave API"
            )
            if results:
                return results

        # Backend 2: DDGS (works in China, new package)
        results = await self._try_backend(
            self._ddgs_search, query, num_results, "DDGS"
        )
        if results:
            return results

        # Backend 3: Wikipedia API
        results = await self._try_backend(
            self._wikipedia_search, query, num_results, "Wikipedia"
        )
        if results:
            return results

        # Backend 4: Mock (always available)
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
                })
            return results

    async def _mock_search(
        self, query: str, n: int
    ) -> list[dict[str, str]]:
        """Mock search with helpful status messages."""
        import time
        return [
            {
                "title": f"[Mock] 搜索结果 {i+1}: {query[:40]}",
                "url": f"https://mock-search.local/result-{i+1}",
                "snippet": (
                    f"这是关于 '{query[:60]}' 的模拟搜索结果 #{i+1}。"
                    f"内容涵盖相关概念、方法、应用场景与最新进展。"
                ),
            }
            for i in range(n)
        ]

    def __call__(self, query: str) -> list[dict[str, str]]:
        """Synchronous interface."""
        return asyncio.run(self.search(query))
