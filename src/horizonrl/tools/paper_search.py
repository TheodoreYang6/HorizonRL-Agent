"""
学术论文搜索 — 多后端并发竞速 (FIRST_COMPLETED)。

面向国内用户的论文搜索, VPN 开/关自适应:
  - OpenAlex (api.openalex.org) — 国内直接访问, 2.5亿+论文
  - Semantic Scholar (api.semanticscholar.org) — 元数据丰富, 2亿+论文
  - Arxiv 官方端点 (export.arxiv.org / arxiv.org) — 经典论文库
  - Arxiv 国内镜像 (cn.arxiv.org) — 国内加速
  - Arxiv Python 包 — 最丰富元数据
  - Mock 兜底 — 离线/CI 自动降级

所有后端并发竞速, 最先返回有效结果的胜出。
每个后端独立超时 (6s), 总最坏延迟 ~6s (而非之前的 10s×2=20s 串行)。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from xml.etree import ElementTree

_ATOM_NS = "http://www.w3.org/2005/Atom"

# 各后端独立超时 (短超时快失败)
_BACKEND_TIMEOUT = 6.0

# Arxiv API 端点列表 (含国内镜像)
_ARXIV_ENDPOINTS = [
    ("https://export.arxiv.org/api/query", "arxiv_official"),
    ("https://arxiv.org/api/query", "arxiv_org"),
]


class PaperSearchTool:
    """学术论文搜索 — 多后端并发竞速。

    后端优先级 (并发竞速, 非顺序回退):
      - OpenAlex — 完全开放, 国内可访问, 2.5亿+论文
      - Semantic Scholar — 英文学术论文元数据最丰富
      - Arxiv Python 包 — 完整元数据 (authors/categories/pdf)
      - Arxiv API (export.arxiv.org / arxiv.org) — Atom XML
      - Mock — 离线兜底

    所有后端同时发起, 最先返回有效结果的胜出, 其余取消。
    """

    name = "paper_search"
    description = "搜索学术论文（OpenAlex + Semantic Scholar + Arxiv 并发竞速）。"

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    # ── Public API ───────────────────────────────────────────────────────

    async def search(
        self, query: str, max_results: int | None = None
    ) -> list[dict[str, Any]]:
        """搜索论文 — 多后端并发竞速。

        OpenAlex ‖ Semantic Scholar ‖ Arxiv Pkg ‖ Arxiv API×2
        最先返回有效结果的后端胜出。
        """
        limit = max_results or self.max_results
        results = await self._race_all(query, limit)
        if results:
            return results
        return self._generate_mock(query, limit)

    async def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        results = await self.search(paper_id, max_results=1)
        return results[0] if results else None

    def __call__(self, query: str) -> list[dict[str, Any]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.search(query))
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, self.search(query)).result()

    # ── Concurrent Race ──────────────────────────────────────────────────

    async def _race_all(self, query: str, limit: int) -> list[dict[str, Any]]:
        """并发启动所有后端, 最先返回非空结果的胜出。"""
        tasks = [
            asyncio.create_task(self._try_openalex(query, limit)),
            asyncio.create_task(self._try_semantic_scholar(query, limit)),
            asyncio.create_task(self._try_arxiv_pkg(query, limit)),
            *[asyncio.create_task(self._try_arxiv_api(url, label, query, limit))
              for url, label in _ARXIV_ENDPOINTS],
        ]

        remaining = _BACKEND_TIMEOUT + 2.0
        start = time.monotonic()

        while tasks and remaining > 0:
            done, pending = await asyncio.wait(
                tasks,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
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
            remaining = (_BACKEND_TIMEOUT + 2.0) - (time.monotonic() - start)

        for t in tasks:
            t.cancel()
        return []

    # ── OpenAlex ──────────────────────────────────────────────────────────

    async def _try_openalex(self, query: str, limit: int) -> list[dict[str, Any]]:
        """OpenAlex API — 国内可访问, 无速率限制。"""
        import httpx
        url = "https://api.openalex.org/works"
        params = {"search": query, "per_page": min(limit, 200)}
        try:
            async with httpx.AsyncClient(timeout=_BACKEND_TIMEOUT) as client:
                resp = await client.get(
                    url, params=params,
                    headers={"User-Agent": "Horizon-Agent/1.0"},
                )
                resp.raise_for_status()
                return self._parse_openalex(resp.json().get("results", []))
        except Exception:
            return []

    def _parse_openalex(self, works: list[dict]) -> list[dict[str, Any]]:
        results = []
        for w in works:
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in w.get("authorships", [])
            ]
            authors = [n for n in authors if n]
            abstract = ""
            inverted = w.get("abstract_inverted_index")
            if isinstance(inverted, dict) and inverted:
                try:
                    words = []
                    for word, positions in inverted.items():
                        for pos in positions:
                            words.append((pos, word))
                    words.sort(key=lambda x: x[0])
                    abstract = " ".join(w[1] for w in words)[:1000]
                except Exception:
                    pass
            doi = w.get("doi", "")
            url = f"https://doi.org/{doi}" if doi and not doi.startswith("http") else (doi or w.get("id", ""))
            pdf_url = w.get("primary_location", {}).get("pdf_url", "") or ""
            published = w.get("publication_date", "") or str(w.get("publication_year", ""))
            results.append({
                "title": w.get("title", ""),
                "authors": authors,
                "abstract": abstract,
                "url": url,
                "pdf_url": pdf_url,
                "published": published,
                "categories": [],
                "is_mock": False,
                "provider": "openalex",
            })
        return results

    # ── Semantic Scholar ──────────────────────────────────────────────────

    async def _try_semantic_scholar(self, query: str, limit: int) -> list[dict[str, Any]]:
        import httpx
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query, "limit": min(limit, 100),
            "fields": "title,abstract,authors,year,url,externalIds,publicationDate,openAccessPdf",
        }
        try:
            async with httpx.AsyncClient(timeout=_BACKEND_TIMEOUT) as client:
                resp = await client.get(
                    url, params=params,
                    headers={"User-Agent": "Horizon-Agent/1.0"},
                )
                if resp.status_code == 429:
                    return []
                resp.raise_for_status()
                return self._parse_s2(resp.json().get("data", []))
        except Exception:
            return []

    def _parse_s2(self, papers: list[dict]) -> list[dict[str, Any]]:
        results = []
        for p in papers:
            authors = [a.get("name", "") for a in p.get("authors", [])]
            pid = p.get("paperId", "")
            url = p.get("url", "") or f"https://www.semanticscholar.org/paper/{pid}"
            pdf_url = ""
            oa = p.get("openAccessPdf")
            if isinstance(oa, dict) and oa.get("url"):
                pdf_url = oa["url"]
            ext = p.get("externalIds")
            if isinstance(ext, dict) and ext.get("ArXiv"):
                pdf_url = f"https://arxiv.org/pdf/{ext['ArXiv']}"
            published = p.get("publicationDate", "") or str(p.get("year", ""))
            results.append({
                "title": p.get("title", ""),
                "authors": authors,
                "abstract": (p.get("abstract") or "")[:1000],
                "url": url,
                "pdf_url": pdf_url,
                "published": published,
                "categories": [],
                "is_mock": False,
                "provider": "semantic_scholar",
            })
        return results

    # ── Arxiv Python Package ──────────────────────────────────────────────

    async def _try_arxiv_pkg(self, query: str, limit: int) -> list[dict[str, Any]]:
        try:
            import arxiv
        except ImportError:
            return []
        try:
            loop = asyncio.get_running_loop()
            client = arxiv.Client()
            search = arxiv.Search(
                query=query, max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            def _run():
                return list(client.results(search))
            papers = await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=_BACKEND_TIMEOUT,
            )
            return [
                {
                    "title": p.title,
                    "authors": [a.name for a in p.authors],
                    "abstract": (p.summary or "")[:1000],
                    "url": p.entry_id or "",
                    "pdf_url": p.pdf_url or "",
                    "published": p.published.isoformat() if p.published else "",
                    "categories": list(p.categories) if p.categories else [],
                    "is_mock": False,
                    "provider": "arxiv_pkg",
                }
                for p in papers
            ]
        except Exception:
            return []

    # ── Arxiv API (Atom XML) ──────────────────────────────────────────────

    async def _try_arxiv_api(
        self, base_url: str, label: str, query: str, limit: int
    ) -> list[dict[str, Any]]:
        import httpx
        url = (
            f"{base_url}?"
            f"search_query=all:{query}&max_results={limit}"
            f"&sortBy=relevance&sortOrder=descending"
        )
        try:
            async with httpx.AsyncClient(timeout=_BACKEND_TIMEOUT) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Horizon-Agent/1.0"},
                )
                resp.raise_for_status()
                return self._parse_atom(resp.text, label)
        except Exception:
            return []

    def _parse_atom(self, xml_text: str, label: str) -> list[dict[str, Any]]:
        papers: list[dict[str, Any]] = []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return papers

        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            title_el = entry.find(f"{{{_ATOM_NS}}}title")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""

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
            entry_id = id_el.text.strip() if id_el is not None and id_el.text else ""
            if not abs_url and entry_id:
                abs_url = entry_id
            if not pdf_url and entry_id:
                pdf_url = entry_id.replace("/abs/", "/pdf/")

            published_el = entry.find(f"{{{_ATOM_NS}}}published")
            published = published_el.text.strip() if published_el is not None and published_el.text else ""

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
                "provider": label,
            })
        return papers

    # ── Mock 兜底 ─────────────────────────────────────────────────────────

    def _generate_mock(self, query: str, limit: int) -> list[dict[str, Any]]:
        now = time.strftime("%Y-%m-%d")
        templates = [
            {
                "title": f"A Comprehensive Survey on {query[:50]}",
                "authors": ["Zhang, W.", "Li, X.", "Wang, H."],
                "abstract": f"本文全面综述了 {query[:80]} 领域的研究现状与最新进展。系统性地梳理了主流方法、关键技术和应用场景。（离线 Mock 数据）",
                "categories": ["cs.AI", "cs.CL"],
            },
            {
                "title": f"Advances in {query[:50]}: A Systematic Review",
                "authors": ["Chen, Y.", "Liu, J.", "Brown, A."],
                "abstract": f"提出了一种针对 {query[:80]} 的创新性方法，结合深度学习与符号推理范式，在多个标准评测中显著超越现有基准。（离线 Mock 数据）",
                "categories": ["cs.LG", "stat.ML"],
            },
            {
                "title": f"Rethinking {query[:50]}: Challenges and Opportunities",
                "authors": ["Kumar, R.", "Park, S."],
                "abstract": f"批判性地审视了 {query[:80]} 领域当前面临的挑战，包括可扩展性、鲁棒性和可解释性问题，并提出了未来可能的研究方向。（离线 Mock 数据）",
                "categories": ["cs.AI"],
            },
            {
                "title": f"Scalable Approaches to {query[:50]}",
                "authors": ["Müller, T.", "Zhao, Q.", "Garcia, M."],
                "abstract": f"探索了 {query[:80]} 的大规模实现方案，聚焦于分布式训练、模型压缩和推理优化。（离线 Mock 数据）",
                "categories": ["cs.DC", "cs.AI"],
            },
            {
                "title": f"Empirical Analysis of {query[:50]} Methods",
                "authors": ["Smith, J.", "Kim, D."],
                "abstract": f"通过大规模实验对比了 {query[:80]} 领域的主流方法，揭示了不同技术路线在准确率、效率和泛化性上的权衡。（离线 Mock 数据）",
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
                "provider": "mock",
            })
        return papers
