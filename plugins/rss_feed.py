"""
RSS Feed Plugin — 拉取和解析 RSS/Atom 源。

无需任何 API Key，支持任意公开 RSS/Atom feed。
优先使用 feedparser 库，未安装时自动回退 stdlib xml.etree 解析。

放入 plugins/ 目录后自动被发现和注册。
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
from typing import Any, ClassVar
from xml.etree import ElementTree

from pydantic import Field

from horizonrl.plugins.base import (
    PluginConfig,
    PluginEvidence,
    PluginParams,
    ToolPlugin,
)

_URL_PATTERN = re.compile(r"https?://[^\s]+")


class RssPluginConfig(PluginConfig):
    """RSS 插件配置。"""

    max_entries: int = Field(default=20, ge=1, le=100, description="最大返回条目数")


class RssPluginParams(PluginParams):
    """RSS 插件参数。"""

    url: str = Field(..., description="RSS/Atom feed URL")
    query: str = Field(default="", description="可选的关键词过滤")


class RssPlugin(ToolPlugin):
    """RSS/Atom 源拉取插件。

    从指定 URL 获取 feed，解析条目，支持关键词过滤。
    无需 API Key，纯 HTTP 请求。
    """

    name: ClassVar[str] = "rss_feed"
    description: ClassVar[str] = "拉取 RSS/Atom 源，搜索和过滤条目"
    version: ClassVar[str] = "1.0.0"
    author: ClassVar[str] = "HorizonRL Team"
    param_schema: ClassVar[type[PluginParams]] = RssPluginParams
    config_schema: ClassVar[type[PluginConfig]] = RssPluginConfig

    async def execute(
        self, url: str = "", query: str = "", **kwargs: Any,
    ) -> str:
        if not url:
            return json.dumps({"error": "未提供 RSS feed URL"}, ensure_ascii=False)

        timeout = getattr(self.config, "timeout", 15.0)
        max_entries = getattr(self.config, "max_entries", 20)

        raw_xml = await self._fetch_url(url, timeout)
        if not raw_xml:
            return json.dumps({"error": f"无法获取 feed: {url}"}, ensure_ascii=False)

        entries = self._parse_feed(raw_xml)
        if not entries:
            return json.dumps({"error": "无法解析 feed 内容"}, ensure_ascii=False)

        if query:
            qlower = query.lower()
            entries = [
                e for e in entries
                if qlower in e.get("title", "").lower()
                or qlower in e.get("summary", "").lower()
            ]

        return json.dumps(entries[:max_entries], ensure_ascii=False)

    def build_params(self, task_description: str, task_context: str = "") -> dict[str, Any]:
        # 从描述中提取 URL，其余作为过滤关键词
        urls = _URL_PATTERN.findall(task_description)
        url = urls[0] if urls else ""
        query = _URL_PATTERN.sub("", task_description).strip()
        return {"url": url, "query": query}

    def extract_evidence(
        self, output: str, task_description: str = "",
    ) -> list[PluginEvidence]:
        try:
            items = json.loads(output)
            if isinstance(items, list):
                return [
                    PluginEvidence(
                        content=f"{item.get('title', '')}: {item.get('summary', '')}",
                        source=item.get("link", ""),
                        source_type="rss",
                    )
                    for item in items
                ]
        except (json.JSONDecodeError, TypeError):
            pass
        return [PluginEvidence(content=output[:2000], source_type="rss")]

    # ── 内部方法 ──

    async def _fetch_url(self, url: str, timeout: float) -> str | None:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "HorizonRL-Agent/1.0", "Accept": "application/xml"},
            )
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=timeout),
            )
            return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            return None

    def _parse_feed(self, raw_xml: str) -> list[dict]:
        # 优先 feedparser
        result = _try_feedparser(raw_xml)
        if result is not None:
            return result
        # 回退 stdlib
        return _parse_feed_stdlib(raw_xml)


def _try_feedparser(raw_xml: str) -> list[dict] | None:
    try:
        import feedparser
        feed = feedparser.parse(raw_xml)
        if feed.bozo and not feed.entries:
            return None
        entries = []
        for entry in feed.entries:
            entries.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": _strip_html(entry.get("summary", entry.get("description", ""))),
                "published": entry.get("published", entry.get("updated", "")),
                "author": entry.get("author", ""),
            })
        return entries if entries else None
    except ImportError:
        return None
    except Exception:
        return None


def _parse_feed_stdlib(raw_xml: str) -> list[dict]:
    try:
        root = ElementTree.fromstring(raw_xml)
    except ElementTree.ParseError:
        return []

    entries: list[dict] = []

    # RSS 2.0
    for item in root.iter("item"):
        entries.append({
            "title": _text(item, "title"),
            "link": _text(item, "link"),
            "summary": _text(item, "description"),
            "published": _text(item, "pubDate"),
            "author": _text(item, "author"),
        })

    # Atom
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title_el = entry.find("atom:title", ns)
        if title_el is None:
            title_el = entry.find("{http://www.w3.org/2005/Atom}title")
        link_el = entry.find("atom:link", ns)
        if link_el is None:
            link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        summary_el = entry.find("atom:summary", ns)
        if summary_el is None:
            summary_el = entry.find("{http://www.w3.org/2005/Atom}summary")
        entries.append({
            "title": title_el.text if title_el is not None and title_el.text else "",
            "link": link_el.get("href", "") if link_el is not None else "",
            "summary": (
                _strip_html(summary_el.text)
                if summary_el is not None and summary_el.text
                else ""
            ),
            "published": "",
            "author": "",
        })

    return entries


def _text(element: ElementTree.Element, tag: str) -> str:
    child = element.find(tag)
    return child.text.strip() if child is not None and child.text else ""


def _strip_html(text: str) -> str:
    if not text:
        return ""
    import re as _re
    clean = _re.sub(r"<[^>]+>", "", text)
    return clean.strip()[:1000]
