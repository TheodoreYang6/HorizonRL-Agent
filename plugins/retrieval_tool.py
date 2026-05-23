"""
Retrieval Plugin — 检索本地文档。

Agent 研究时自动调用此工具，在用户上传的文档中语义搜索相关内容。
与 web_search / paper_search 等网络工具互补，实现 RAG + Agent 混合模式。

放入 plugins/ 目录后自动被发现和注册。
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import Field

from horizonrl.plugins.base import (
    PluginConfig,
    PluginEvidence,
    PluginParams,
    ToolPlugin,
    clean_search_query,
)


class RetrievalPluginConfig(PluginConfig):
    """检索工具配置。"""

    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数")


class RetrievalPluginParams(PluginParams):
    """检索工具参数。"""

    query: str = Field(..., description="检索查询")
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalPlugin(ToolPlugin):
    """本地文档检索工具。

    在用户上传的文档中做语义搜索，返回最相关的文本块。
    结合 web_search 等网络工具，实现混合搜索。
    """

    name: ClassVar[str] = "retrieval_tool"
    description: ClassVar[str] = "检索本地已上传文档中的相关内容 (语义搜索)"
    version: ClassVar[str] = "1.0.0"
    author: ClassVar[str] = "HorizonRL Team"
    param_schema: ClassVar[type[PluginParams]] = RetrievalPluginParams
    config_schema: ClassVar[type[PluginConfig]] = RetrievalPluginConfig

    async def execute(self, query: str = "", top_k: int = 5, **kwargs: Any) -> str:
        if not query:
            return json.dumps({"error": "未提供检索关键词"}, ensure_ascii=False)

        top = top_k or getattr(self.config, "top_k", 5)
        doc_store = _get_doc_store()
        if doc_store is None:
            return json.dumps(
                {"error": "文档存储不可用，请先上传文档"},
                ensure_ascii=False,
            )

        results = doc_store.search(query, top_k=top)
        return json.dumps(results, ensure_ascii=False)

    def build_params(self, task_description: str, task_context: str = "") -> dict[str, Any]:
        query = clean_search_query(task_description)
        return {"query": query, "top_k": 5}

    def extract_evidence(
        self, output: str, task_description: str = "",
    ) -> list[PluginEvidence]:
        try:
            items = json.loads(output)
            if isinstance(items, list):
                return [
                    PluginEvidence(
                        content=item.get("chunk_text", str(item)),
                        source=f"本地文档: {item.get('filename', '')}",
                        source_type="local_document",
                    )
                    for item in items
                ]
        except (json.JSONDecodeError, TypeError):
            pass
        return [PluginEvidence(content=output[:2000], source_type="local_document")]


def _get_doc_store():
    """获取全局 DocumentStore 实例。"""
    try:
        from horizonrl.rag.document_store import DocumentStore
        return DocumentStore()
    except Exception:
        return None
