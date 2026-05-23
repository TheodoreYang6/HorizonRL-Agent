"""
GitHub Search Plugin — 搜索 GitHub 仓库、代码和 Issues。

使用 GitHub REST API v3，无需认证即可使用 (60 req/h)，
设置 GITHUB_TOKEN 环境变量可提升至 5000 req/h。

放入 plugins/ 目录后自动被发现和注册。
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, ClassVar

from pydantic import Field

from horizonrl.plugins.base import (
    PluginConfig,
    PluginEvidence,
    PluginParams,
    ToolPlugin,
    clean_search_query,
)

GITHUB_API = "https://api.github.com"


class GithubPluginConfig(PluginConfig):
    """GitHub 搜索配置。"""

    search_type: str = Field(
        default="repositories",
        description="搜索类型: repositories | code | issues",
    )
    per_page: int = Field(default=10, ge=1, le=100, description="每页结果数")


class GithubPluginParams(PluginParams):
    """GitHub 搜索参数。"""

    query: str = Field(..., description="搜索关键词")
    search_type: str = Field(default="repositories", description="搜索类型")
    num_results: int = Field(default=10, ge=1, le=30)


class GithubPlugin(ToolPlugin):
    """GitHub 搜索插件。

    无需 API Key 也可使用 (60 req/h)。
    设置 GITHUB_TOKEN 环境变量提升速率。
    """

    name: ClassVar[str] = "github_search"
    description: ClassVar[str] = "搜索 GitHub 仓库、代码和 Issues"
    version: ClassVar[str] = "1.0.0"
    author: ClassVar[str] = "HorizonRL Team"
    param_schema: ClassVar[type[PluginParams]] = GithubPluginParams
    config_schema: ClassVar[type[PluginConfig]] = GithubPluginConfig

    _SEARCH_ENDPOINTS: ClassVar[dict[str, str]] = {
        "repositories": "/search/repositories",
        "code": "/search/code",
        "issues": "/search/issues",
    }

    async def execute(
        self, query: str = "", search_type: str = "repositories",
        num_results: int = 10, **kwargs: Any,
    ) -> str:
        stype = search_type or getattr(self.config, "search_type", "repositories")
        endpoint = self._SEARCH_ENDPOINTS.get(stype, "/search/repositories")
        per_page = min(num_results, 30)

        url = f"{GITHUB_API}{endpoint}?q={urllib.parse.quote(query)}&per_page={per_page}"
        token = os.environ.get("GITHUB_TOKEN", "")

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "HorizonRL-Agent/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        data = await self._fetch_json(url, headers)
        items = data.get("items", []) if isinstance(data, dict) else []

        results = []
        for item in items[:num_results]:
            if stype == "repositories":
                results.append({
                    "title": item.get("full_name", ""),
                    "url": item.get("html_url", ""),
                    "description": item.get("description", ""),
                    "stars": item.get("stargazers_count", 0),
                    "language": item.get("language", ""),
                    "topics": item.get("topics", []),
                    "provider": "github",
                })
            elif stype == "code":
                repo = item.get("repository", {})
                results.append({
                    "title": f"{repo.get('full_name', '')}: {item.get('path', '')}",
                    "url": item.get("html_url", ""),
                    "description": f"代码匹配 (仓库: {repo.get('full_name', '')})",
                    "provider": "github_code",
                })
            elif stype == "issues":
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("html_url", ""),
                    "description": (item.get("body", "") or "")[:500],
                    "state": item.get("state", ""),
                    "provider": "github_issues",
                })

        return json.dumps(results or self._fallback_results(query), ensure_ascii=False)

    def build_params(self, task_description: str, task_context: str = "") -> dict[str, Any]:
        query = clean_search_query(task_description)
        return {"query": query, "num_results": 10}

    def extract_evidence(
        self, output: str, task_description: str = "",
    ) -> list[PluginEvidence]:
        try:
            items = json.loads(output)
            if isinstance(items, list):
                return [
                    PluginEvidence(
                        content=f"{item.get('title', '')}: {item.get('description', '')}",
                        source=item.get("url", ""),
                        source_type="github",
                    )
                    for item in items
                ]
        except (json.JSONDecodeError, TypeError):
            pass
        return [PluginEvidence(content=output[:2000], source_type="github")]

    @classmethod
    def get_provider_info(cls) -> dict[str, str]:
        return {
            "provider_id": "github",
            "env_var": "GITHUB_TOKEN",
            "label": "GitHub (可选，提升速率)",
            "url": "https://github.com/settings/tokens",
        }

    async def _fetch_json(self, url: str, headers: dict) -> Any:
        try:
            req = urllib.request.Request(url, headers=headers)
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=getattr(self.config, "timeout", 10)),
            )
            return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _fallback_results(query: str) -> list[dict]:
        return [{
            "title": f"GitHub 搜索: {query}",
            "url": f"https://github.com/search?q={urllib.parse.quote(query)}",
            "description": "GitHub API 暂不可用，请通过此链接手动查看",
            "provider": "github_fallback",
            "is_mock": True,
        }]
