"""ArXiv paper search — multi-endpoint concurrent race with mock fallback.

Design:
  1. Three Arxiv sources raced concurrently (first success wins, max 8s total)
  2. Primary: arxiv Python package (rich metadata)
  3. Backup: export.arxiv.org API (Atom XML, parsed with stdlib)
  4. Backup: arxiv.org API (same, different DNS)
  5. All fail → generate meaningful mock paper data (never empty, never hangs)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from xml.etree import ElementTree

# Arxiv API Atom XML namespace
_ATOM_NS = "http://www.w3.org/2005/Atom"

# API endpoints — 并发竞速 (FIRST_COMPLETED)
_ARXIV_ENDPOINTS = [
    "https://export.arxiv.org/api/query",   # 官方导出端点 (测试可用)
    "https://arxiv.org/api/query",          # 官方主端点 (自动重定向到 HTTPS)
]

# Total timeout for all endpoint attempts combined (generous for slow connections)
_RACE_TIMEOUT = 8.0


class ArxivSearchTool:
    """Search arxiv for academic papers with multi-endpoint redundancy.

    Two API endpoints + arxiv Python package tried concurrently:
      - export.arxiv.org / arxiv.org (官方端点)
      - arxiv Python package (richest metadata)

    First valid response wins. Total worst-case latency: _RACE_TIMEOUT seconds.
    If ALL fail, meaningful mock paper data is generated so the pipeline never blocks.
    """

    name = "arxiv_search"
    description = "Search arxiv for academic papers matching a query."

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    # ── Public API ───────────────────────────────────────────────────────

    async def search(
        self, query: str, max_results: int | None = None
    ) -> list[dict[str, Any]]:
        """Search arxiv with concurrent endpoint race.

        Three endpoints fire simultaneously. The first to return valid results
        wins. All pending requests are cancelled immediately.

        Total worst-case latency: _RACE_TIMEOUT seconds.
        """
        limit = max_results or self.max_results
        results = await self._race_endpoints(query, limit)
        if results:
            return results
        return self._generate_mock(query, limit)

    async def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        """Fetch a specific paper by arxiv ID."""
        results = await self.search(f"id:{arxiv_id}", max_results=1)
        return results[0] if results else None

    def __call__(self, query: str) -> list[dict[str, Any]]:
        """Synchronous interface — safe from any context."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.search(query))
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, self.search(query)).result()

    # ── Concurrent endpoint race ──────────────────────────────────────────

    async def _race_endpoints(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Fire all endpoints concurrently. First non-empty result wins.

        Uses asyncio.wait with FIRST_COMPLETED to return as soon as any
        endpoint responds with data. Remaining tasks are cancelled.
        """
        tasks = [
            asyncio.create_task(self._try_arxiv_pkg(query, limit)),
            *[asyncio.create_task(self._try_api(url, query, limit))
              for url in _ARXIV_ENDPOINTS],
        ]

        remaining = _RACE_TIMEOUT
        start = time.monotonic()

        while tasks and remaining > 0:
            done, pending = await asyncio.wait(
                tasks,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Check completed tasks — first non-empty wins
            for t in done:
                try:
                    result = t.result()
                    if result:
                        for p in pending:
                            p.cancel()
                        return result
                except Exception:
                    pass

            tasks = list(pending)
            remaining = _RACE_TIMEOUT - (time.monotonic() - start)

        # Timeout or all failed
        for t in tasks:
            t.cancel()
        return []

    # ── Endpoint implementations ──────────────────────────────────────────

    async def _try_arxiv_pkg(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Try arxiv Python package (richest metadata)."""
        try:
            import arxiv
        except ImportError:
            return []

        try:
            loop = asyncio.get_running_loop()
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            def _run():
                return list(client.results(search))

            papers = await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=_RACE_TIMEOUT,
            )
            return [
                {
                    "title": p.title,
                    "authors": [a.name for a in p.authors],
                    "abstract": p.summary[:1000],
                    "url": p.entry_id,
                    "pdf_url": p.pdf_url,
                    "published": p.published.isoformat() if p.published else "",
                    "categories": list(p.categories),
                    "is_mock": False,
                }
                for p in papers
            ]
        except Exception:
            return []

    async def _try_api(
        self, base_url: str, query: str, limit: int
    ) -> list[dict[str, Any]]:
        """Try a raw Arxiv API endpoint with XML parsing."""
        import httpx

        url = (
            f"{base_url}?"
            f"search_query=all:{query}&max_results={limit}"
            f"&sortBy=relevance&sortOrder=descending"
        )
        try:
            async with httpx.AsyncClient(timeout=_RACE_TIMEOUT) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": "HorizonRL-Agent/0.2"},
                )
                response.raise_for_status()
                return self._parse_atom(response.text)
        except Exception:
            return []

    # ── XML parsing ───────────────────────────────────────────────────────

    def _parse_atom(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse Arxiv Atom XML into paper dicts using stdlib ElementTree."""
        papers: list[dict[str, Any]] = []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return papers

        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            title_el = entry.find(f"{{{_ATOM_NS}}}title")
            title = (
                title_el.text.strip()
                if title_el is not None and title_el.text
                else ""
            )

            summary_el = entry.find(f"{{{_ATOM_NS}}}summary")
            abstract = (
                summary_el.text.strip()[:1000]
                if summary_el is not None and summary_el.text
                else ""
            )

            authors = []
            for author_el in entry.findall(f"{{{_ATOM_NS}}}author"):
                name_el = author_el.find(f"{{{_ATOM_NS}}}name")
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            abs_url = ""
            pdf_url = ""
            for link_el in entry.findall(f"{{{_ATOM_NS}}}link"):
                href = link_el.get("href", "")
                rel = link_el.get("rel", "")
                title_attr = link_el.get("title", "")
                if rel == "alternate":
                    abs_url = href
                elif "pdf" in title_attr.lower() or "pdf" in rel.lower():
                    pdf_url = href

            id_el = entry.find(f"{{{_ATOM_NS}}}id")
            entry_id = (
                id_el.text.strip()
                if id_el is not None and id_el.text
                else ""
            )
            if not abs_url and entry_id:
                abs_url = entry_id
            if not pdf_url and entry_id:
                pdf_url = entry_id.replace("/abs/", "/pdf/")

            published = ""
            published_el = entry.find(f"{{{_ATOM_NS}}}published")
            if published_el is not None and published_el.text:
                published = published_el.text.strip()

            categories = [
                cat.get("term", "")
                for cat in entry.findall(f"{{{_ATOM_NS}}}category")
                if cat.get("term")
            ]

            papers.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "url": abs_url,
                "pdf_url": pdf_url,
                "published": published,
                "categories": categories,
                "is_mock": False,
            })

        return papers

    # ── Mock fallback ─────────────────────────────────────────────────────

    def _generate_mock(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Generate meaningful mock paper data when all endpoints are unreachable.

        Never returns empty — ensures downstream tasks always have input to work with.
        Results are clearly marked with is_mock=True.
        """
        now = time.strftime("%Y-%m-%d")
        templates = [
            {
                "title": f"A Comprehensive Survey on {query[:50]}",
                "authors": ["[Mock] Zhang, W.", "[Mock] Li, X.", "[Mock] Wang, H."],
                "abstract": (
                    f"本文全面综述了 {query[:80]} 领域的研究现状与最新进展。"
                    f"系统性地梳理了主流方法、关键技术和应用场景，"
                    f"并在多个公开基准上进行了对比分析。"
                    f"（模拟数据 — Arxiv API 当前不可达）"
                ),
                "categories": ["cs.AI", "cs.CL"],
            },
            {
                "title": f"Advances in {query[:50]}: A Systematic Review",
                "authors": ["[Mock] Chen, Y.", "[Mock] Liu, J.", "[Mock] Brown, A."],
                "abstract": (
                    f"提出了一种针对 {query[:80]} 的创新性方法，"
                    f"结合深度学习与符号推理范式，"
                    f"在多个标准评测中显著超越现有基准。"
                    f"（模拟数据 — Arxiv API 当前不可达）"
                ),
                "categories": ["cs.LG", "stat.ML"],
            },
            {
                "title": f"Rethinking {query[:50]}: Challenges and Opportunities",
                "authors": ["[Mock] Kumar, R.", "[Mock] Park, S."],
                "abstract": (
                    f"批判性地审视了 {query[:80]} 领域当前面临的挑战，"
                    f"包括可扩展性、鲁棒性和可解释性问题，"
                    f"并提出了未来可能的研究方向。"
                    f"（模拟数据 — Arxiv API 当前不可达）"
                ),
                "categories": ["cs.AI"],
            },
            {
                "title": f"Scalable Approaches to {query[:50]}",
                "authors": ["[Mock] Müller, T.", "[Mock] Zhao, Q.", "[Mock] Garcia, M."],
                "abstract": (
                    f"探索了 {query[:80]} 的大规模实现方案，"
                    f"聚焦于分布式训练、模型压缩和推理优化，"
                    f"在工业级数据集上验证了方法的有效性。"
                    f"（模拟数据 — Arxiv API 当前不可达）"
                ),
                "categories": ["cs.DC", "cs.AI"],
            },
            {
                "title": f"Empirical Analysis of {query[:50]} Methods",
                "authors": ["[Mock] Smith, J.", "[Mock] Kim, D."],
                "abstract": (
                    f"通过大规模实验对比了 {query[:80]} 领域的主流方法，"
                    f"揭示了不同技术路线在准确率、效率和泛化性上的权衡。"
                    f"（模拟数据 — Arxiv API 当前不可达）"
                ),
                "categories": ["cs.AI", "stat.ML"],
            },
        ]

        papers = []
        for i in range(min(limit, len(templates))):
            t = templates[i]
            idx = str(i + 1).zfill(2)
            papers.append({
                "title": t["title"],
                "authors": t["authors"],
                "abstract": t["abstract"],
                "url": f"https://arxiv.org/abs/25{idx}.{idx.zfill(5)}",
                "pdf_url": f"https://arxiv.org/pdf/25{idx}.{idx.zfill(5)}",
                "published": now,
                "categories": t["categories"],
                "is_mock": True,
            })
        return papers
