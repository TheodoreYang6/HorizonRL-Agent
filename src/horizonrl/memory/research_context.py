"""
Research Context Engine — 研究上下文记忆引擎。

与 L1/L2/L3 (Agent 工作记忆) 平行独立:
  - L1/L2/L3: 子任务执行经验 (Planner/Worker/Verifier 消费)
  - Context Store: 跨会话研究摘要 (多轮追问时语义检索)

核心能力:
  1. 研究完成后自动生成结构化摘要 → 存入 ChromaDB
  2. 追问时语义检索相关历史 → 注入 prompt 上下文
  3. Token 预算自动裁切

使用:
  store = ResearchContextStore()
  store.add(session_id, query, summary, topics, evidence_quality)
  results = store.search("Transformer注意力", top_k=2)
"""

from __future__ import annotations

import hashlib
from typing import Any


class ResearchContextStore:
    """研究上下文记忆 — ChromaDB 持久化 + 语义检索。

    与 L3 EpisodicArchive 共享 ChromaDB 基础设施，但使用独立 collection。
    """

    COLLECTION = "research_context"

    def __init__(self, persist_dir: str = "data/chromadb"):
        self._persist_dir = persist_dir
        self._store = None

    def _init(self):
        if self._store is not None:
            return
        from horizonrl.memory.vector_store import ChromaVectorStore
        self._store = ChromaVectorStore(
            persist_dir=self._persist_dir,
            collection_name=self.COLLECTION,
        )

    def add(
        self,
        session_id: str,
        query: str,
        summary: str,
        topics: list[str] | None = None,
        evidence_quality: float = 0.0,
    ) -> None:
        """存入一条研究摘要。

        Args:
            session_id: 会话 ID
            query: 原始研究问题
            summary: LLM 生成的结构化摘要 (或 fallback 模板)
            topics: 主题标签
            evidence_quality: 真实证据占比 (0.0-1.0)
        """
        self._init()
        # n-gram 嵌入 (与 L3 一致，确保离线可用)
        vec = self._ngram_embed(summary)
        key = hashlib.md5(session_id.encode()).hexdigest()[:16]
        self._store.add(
            embeddings=[vec],
            keys=[key],
            metadata=[{
                "session_id": session_id,
                "query": query[:300],
                "summary": summary[:500],
                "topics": ",".join(topics or []),
                "evidence_quality": evidence_quality,
            }],
        )

    def search(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """语义检索相关历史研究。

        Args:
            query: 当前追问或关键词
            top_k: 返回条数

        Returns:
            [{session_id, query, summary, topics, evidence_quality, score}, ...]
        """
        self._init()
        if self._store.count() == 0:
            return []
        vec = self._ngram_embed(query)
        results = self._store.search(vec, top_k=top_k)
        return [
            {
                "session_id": r["metadata"].get("session_id", ""),
                "query": r["metadata"].get("query", ""),
                "summary": r["metadata"].get("summary", ""),
                "topics": r["metadata"].get("topics", ""),
                "evidence_quality": r["metadata"].get("evidence_quality", 0.0),
                "score": r.get("score", 0.0),
            }
            for r in results
        ]

    def count(self) -> int:
        self._init()
        return self._store.count()

    def clear(self) -> None:
        self._init()
        self._store.clear()

    # ── n-gram 嵌入 (离线可用) ─────────────────────────────────────────

    def _ngram_embed(self, text: str, dim: int = 1024) -> list[float]:
        """MD5 n-gram 哈希向量化 — 与 L3 一致的确定性嵌入。"""
        import math
        vec = [0.0] * dim
        if not text:
            return vec
        for n in (2, 3, 4):
            for i in range(len(text) - n + 1):
                ngram = text[i:i + n]
                h = int.from_bytes(
                    hashlib.md5(ngram.encode("utf-8")).digest()[:4], "big"
                )
                vec[h % dim] += 1.0
        total = math.sqrt(sum(v * v for v in vec))
        if total > 0:
            vec = [v / total for v in vec]
        return vec


# 全局单例
_store: ResearchContextStore | None = None


def get_context_store() -> ResearchContextStore:
    global _store
    if _store is None:
        _store = ResearchContextStore()
    return _store
