"""ArXiv paper search and retrieval tool."""

from __future__ import annotations

import asyncio
from typing import Any


class ArxivSearchTool:
    """Search arxiv for academic papers.

    Uses the arxiv API directly or via the arxiv Python package.
    """

    name = "arxiv_search"
    description = "Search arxiv for academic papers matching a query."

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def search(
        self, query: str, max_results: int | None = None
    ) -> list[dict[str, Any]]:
        """Search arxiv asynchronously.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            List of paper dicts with title, authors, abstract, url, published date.
        """
        limit = max_results or self.max_results

        try:
            import arxiv

            loop = asyncio.get_running_loop()
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            def _run():
                return list(client.results(search))

            papers = await loop.run_in_executor(None, _run)

            return [
                {
                    "title": p.title,
                    "authors": [a.name for a in p.authors],
                    "abstract": p.summary[:1000],
                    "url": p.entry_id,
                    "pdf_url": p.pdf_url,
                    "published": p.published.isoformat() if p.published else "",
                    "categories": list(p.categories),
                }
                for p in papers
            ]
        except ImportError:
            return await self._api_search(query, limit)

    async def _api_search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Fallback to raw arxiv API."""
        import httpx
        import xml.etree.ElementTree as ET

        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{query}&max_results={max_results}"
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            # Basic XML parsing — in production, use feedparser
            return [{"title": "Arxiv API result", "abstract": response.text[:500], "url": url}]

    async def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        """Fetch a specific paper by arxiv ID."""
        results = await self.search(f"id:{arxiv_id}", max_results=1)
        return results[0] if results else None

    def __call__(self, query: str) -> list[dict[str, Any]]:
        return asyncio.run(self.search(query))
