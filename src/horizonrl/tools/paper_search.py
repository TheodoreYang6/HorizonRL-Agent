"""
学术论文搜索 — OpenAlex (主) + Semantic Scholar (备) + Mock 兜底。

面向国内用户的论文搜索:
  - api.openalex.org (主): 完全开放、无速率限制、国内可访问、2.5亿+论文
  - api.semanticscholar.org (备): 元数据丰富、2亿+论文、1 req/s限速
  - Mock 兜底: 离线/CI自动降级

接口兼容原 ArxivSearchTool: search(query, max_results) -> list[dict]
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

# 全局超时
_REQUEST_TIMEOUT = 10.0

# 全局速率限制 — 防止连续快速请求触发熔断器
_rate_limiter = asyncio.Semaphore(1)
_rate_last_call = 0.0
_MIN_CALL_INTERVAL = 1.5  # 秒


class PaperSearchTool:
    """学术论文搜索 — 国内可用，多后端冗余。

    后端优先级 (顺序回退):
      1. OpenAlex — 完全开放，国内可访问
      2. Semantic Scholar — 英文学术论文元数据最丰富
      3. Mock — 离线兜底

    内置全局速率限制，防止工具熔断器误触发。
    """

    name = "paper_search"
    description = "搜索学术论文（OpenAlex + Semantic Scholar），覆盖全学科。"

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    # ── Public API ───────────────────────────────────────────────────────

    async def search(
        self, query: str, max_results: int | None = None
    ) -> list[dict[str, Any]]:
        """搜索论文 — 顺序回退。

        OpenAlex -> Semantic Scholar -> Mock。
        """
        limit = max_results or self.max_results

        # 全局速率限制 — 确保调用间隔 >= _MIN_CALL_INTERVAL 秒
        global _rate_last_call
        async with _rate_limiter:
            now = time.monotonic()
            wait = _MIN_CALL_INTERVAL - (now - _rate_last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            _rate_last_call = time.monotonic()

        # 1. OpenAlex (首选: 国内可访问，无速率限制)
        results = await self._try_openalex(query, limit)
        if results:
            return results

        # 2. Semantic Scholar (备选: 元数据丰富)
        results = await self._try_semantic_scholar(query, limit)
        if results:
            return results

        # 3. Mock 兜底
        return self._generate_mock(query, limit)

    async def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        results = await self.search(paper_id, max_results=1)
        return results[0] if results else None

    def __call__(self, query: str) -> list[dict[str, Any]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.search(query))
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, self.search(query)).result()

    # ── OpenAlex ──────────────────────────────────────────────────────────

    async def _try_openalex(self, query: str, limit: int) -> list[dict[str, Any]]:
        """OpenAlex API — 完全开放的学术文献数据库。

        GET https://api.openalex.org/works?search={query}&per_page={limit}
        国内可访问，无速率限制，覆盖 2.5 亿+ 论文。
        """
        import httpx

        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per_page": min(limit, 200),
        }

        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.get(
                    url, params=params,
                    headers={"User-Agent": "HorizonRL-Agent/0.3"},
                )
                response.raise_for_status()
                body = response.json()
                return self._parse_openalex(body.get("results", []))
        except Exception:
            return []

    def _parse_openalex(self, works: list[dict]) -> list[dict[str, Any]]:
        """将 OpenAlex 响应转为统一格式。"""
        results = []
        for w in works:
            authors = []
            for auth in w.get("authorships", []):
                a = auth.get("author", {})
                name = a.get("display_name", "")
                if name:
                    authors.append(name)

            # 摘要: OpenAlex 用 inverted_index 格式，重建为纯文本
            abstract = ""
            inverted = w.get("abstract_inverted_index")
            if isinstance(inverted, dict) and inverted:
                try:
                    words: list[tuple[int, str]] = []
                    for word, positions in inverted.items():
                        for pos in positions:
                            words.append((pos, word))
                    words.sort(key=lambda x: x[0])
                    abstract = " ".join(w[1] for w in words)[:1000]
                except Exception:
                    pass

            # DOI URL
            url = w.get("doi", "")
            if url and not url.startswith("http"):
                url = f"https://doi.org/{url}"
            if not url:
                url = w.get("id", "")

            pdf_url = w.get("primary_location", {}).get("pdf_url", "") or ""
            published = w.get("publication_date", "")
            year = w.get("publication_year", "")

            results.append({
                "title": w.get("title", ""),
                "authors": authors,
                "abstract": abstract,
                "url": url,
                "pdf_url": pdf_url,
                "published": str(published or year or ""),
                "categories": [],
                "is_mock": False,
                "provider": "openalex",
            })
        return results

    # ── Semantic Scholar ──────────────────────────────────────────────────

    async def _try_semantic_scholar(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Semantic Scholar API — 免费学术搜索引擎。"""
        import httpx

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,authors,year,url,externalIds,publicationDate,openAccessPdf",
        }

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                    response = await client.get(
                        url, params=params,
                        headers={"User-Agent": "HorizonRL-Agent/0.3"},
                    )
                    if response.status_code == 429 and attempt == 0:
                        await asyncio.sleep(3.0)
                        continue
                    response.raise_for_status()
                    return self._parse_semantic_scholar(response.json().get("data", []))
            except Exception:
                if attempt == 0:
                    await asyncio.sleep(1.5)
                    continue
                return []
        return []

    def _parse_semantic_scholar(self, papers: list[dict]) -> list[dict[str, Any]]:
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

    # ── Mock 兜底 ─────────────────────────────────────────────────────────

    def _generate_mock(self, query: str, limit: int) -> list[dict[str, Any]]:
        """所有后端不可达时生成 Mock 数据。"""
        now = time.strftime("%Y-%m-%d")
        templates = [
            {
                "title": f"A Comprehensive Survey on {query[:50]}",
                "authors": ["[Mock] Zhang, W.", "[Mock] Li, X.", "[Mock] Wang, H."],
                "abstract": (
                    f"本文全面综述了 {query[:80]} 领域的研究现状与最新进展。"
                    "系统性地梳理了主流方法、关键技术和应用场景。"
                    "（离线 Mock 数据）"
                ),
            },
            {
                "title": f"Advances in {query[:50]}: A Systematic Review",
                "authors": ["[Mock] Chen, Y.", "[Mock] Liu, J."],
                "abstract": (
                    f"提出了一种针对 {query[:80]} 的创新性方法，"
                    "在多个标准评测中显著超越现有基准。"
                    "（离线 Mock 数据）"
                ),
            },
            {
                "title": f"Rethinking {query[:50]}: Challenges and Opportunities",
                "authors": ["[Mock] Kumar, R.", "[Mock] Park, S."],
                "abstract": (
                    f"批判性地审视了 {query[:80]} 领域的挑战并提出未来方向。"
                    "（离线 Mock 数据）"
                ),
            },
        ]

        papers = []
        for i in range(min(limit, len(templates))):
            t = templates[i]
            papers.append({
                "title": t["title"],
                "authors": t["authors"],
                "abstract": t["abstract"],
                "url": f"https://arxiv.org/abs/mock-{str(i+1).zfill(2)}",
                "pdf_url": "",
                "published": now,
                "categories": ["cs.AI"],
                "is_mock": True,
                "provider": "mock",
            })
        return papers
