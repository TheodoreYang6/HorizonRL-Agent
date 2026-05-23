"""
PubMed Search Plugin — 搜索生物医学论文。

使用 NCBI Entrez E-utilities API，无需认证即可使用 (3 req/s)，
设置 NCBI_API_KEY 环境变量可提升至 10 req/s。

放入 plugins/ 目录后自动被发现和注册。
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, ClassVar
from xml.etree import ElementTree

from pydantic import Field

from horizonrl.plugins.base import (
    PluginConfig,
    PluginEvidence,
    PluginParams,
    ToolPlugin,
    clean_search_query,
)

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubmedPluginConfig(PluginConfig):
    """PubMed 搜索配置。"""

    max_results: int = Field(default=10, ge=1, le=50, description="最大返回论文数")


class PubmedPluginParams(PluginParams):
    """PubMed 搜索参数。"""

    query: str = Field(..., description="搜索关键词 (支持 PubMed 查询语法)")
    max_results: int = Field(default=10, ge=1, le=50)


class PubmedPlugin(ToolPlugin):
    """PubMed 生物医学论文搜索插件。

    API 流程: ESearch(获取 PMID 列表) → EFetch(获取详情 XML) → 解析。
    无需 API Key，设置 NCBI_API_KEY 可提升速率。
    """

    name: ClassVar[str] = "pubmed_search"
    description: ClassVar[str] = "搜索 PubMed 生物医学论文 (NCBI Entrez API)"
    version: ClassVar[str] = "1.0.0"
    author: ClassVar[str] = "HorizonRL Team"
    param_schema: ClassVar[type[PluginParams]] = PubmedPluginParams
    config_schema: ClassVar[type[PluginConfig]] = PubmedPluginConfig

    async def execute(
        self, query: str = "", max_results: int = 10, **kwargs: Any,
    ) -> str:
        if not query:
            return json.dumps({"error": "未提供搜索关键词"}, ensure_ascii=False)

        max_r = max_results or getattr(self.config, "max_results", 10)
        api_key = os.environ.get("NCBI_API_KEY", "")

        # Step 1: ESearch — 获取 PMID 列表
        pmids = await self._esearch(query, max_r, api_key)
        if not pmids:
            return json.dumps(self._fallback_results(query), ensure_ascii=False)

        # Step 2: EFetch — 获取论文详情
        articles = await self._efetch(pmids, api_key)

        results = []
        for art in articles:
            results.append({
                "title": art.get("title", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{art.get('pmid', '')}/",
                "abstract": art.get("abstract", ""),
                "authors": art.get("authors", []),
                "journal": art.get("journal", ""),
                "year": art.get("year", ""),
                "provider": "pubmed",
            })

        return json.dumps(
            results or self._fallback_results(query),
            ensure_ascii=False,
        )

    def build_params(self, task_description: str, task_context: str = "") -> dict[str, Any]:
        query = clean_search_query(task_description)
        return {"query": query, "max_results": 10}

    def extract_evidence(
        self, output: str, task_description: str = "",
    ) -> list[PluginEvidence]:
        try:
            items = json.loads(output)
            if isinstance(items, list):
                return [
                    PluginEvidence(
                        content=f"{item.get('title', '')}: {item.get('abstract', '')}",
                        source=item.get("url", ""),
                        source_type="paper",
                        is_mock=item.get("provider") == "pubmed_fallback",
                    )
                    for item in items
                ]
        except (json.JSONDecodeError, TypeError):
            pass
        return [PluginEvidence(content=output[:2000], source_type="paper")]

    @classmethod
    def get_provider_info(cls) -> dict[str, str]:
        return {
            "provider_id": "ncbi",
            "env_var": "NCBI_API_KEY",
            "label": "NCBI/PubMed (可选，提升速率)",
            "url": "https://www.ncbi.nlm.nih.gov/account/",
        }

    # ── E-utilities ──

    async def _esearch(self, query: str, max_results: int, api_key: str) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        if api_key:
            params["api_key"] = api_key

        url = f"{ENTREZ_BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
        data = await self._fetch_json(url)
        if isinstance(data, dict):
            return data.get("esearchresult", {}).get("idlist", [])
        return []

    async def _efetch(self, pmids: list[str], api_key: str) -> list[dict]:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if api_key:
            params["api_key"] = api_key

        url = f"{ENTREZ_BASE}/efetch.fcgi?{urllib.parse.urlencode(params)}"
        xml_text = await self._fetch_text(url)
        if not xml_text:
            return []
        return self._parse_efetch_xml(xml_text)

    def _parse_efetch_xml(self, xml_text: str) -> list[dict]:
        articles: list[dict] = []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return articles

        for art_elem in root.iter("PubmedArticle"):
            try:
                medline = art_elem.find("MedlineCitation")
                if medline is None:
                    continue

                art = medline.find("Article")
                if art is None:
                    continue

                pmid = (medline.findtext("PMID") or "").strip()
                title = (art.findtext("ArticleTitle") or "").strip()
                abstract = (art.findtext(".//Abstract/AbstractText") or "").strip()
                journal = (art.findtext(".//Journal/Title") or "").strip()
                year = (art.findtext(".//Journal/JournalIssue/PubDate/Year") or
                        art.findtext(".//ArticleDate/Year") or "").strip()

                authors = []
                for auth in art.iter("Author"):
                    last = (auth.findtext("LastName") or "").strip()
                    fore = (auth.findtext("ForeName") or "").strip()
                    if last:
                        authors.append(f"{last} {fore}".strip())

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract[:2000] if abstract else "",
                    "authors": authors[:10],
                    "journal": journal,
                    "year": year,
                })
            except Exception:
                continue

        return articles

    async def _fetch_json(self, url: str) -> Any:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HorizonRL-Agent/1.0"})
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=getattr(self.config, "timeout", 10)),
            )
            return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
            return {}

    async def _fetch_text(self, url: str) -> str | None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HorizonRL-Agent/1.0"})
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=getattr(self.config, "timeout", 15)),
            )
            return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return None

    @staticmethod
    def _fallback_results(query: str) -> list[dict]:
        return [{
            "title": f"PubMed 搜索: {query}",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(query)}",
            "abstract": "PubMed API 暂不可用，请通过此链接手动查看",
            "provider": "pubmed_fallback",
            "is_mock": True,
        }]
