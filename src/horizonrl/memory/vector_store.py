"""
ChromaDB 向量存储 — 替代 FAISS 文件读写的生产级方案。

接口兼容 L3EpisodicArchive:
  - add(embeddings, keys, metadata)  — 批量写入向量
  - search(query_embedding, top_k)   — 向量相似度检索
  - save() / load()                  — 自动持久化 (ChromaDB 自动处理)
  - count()                          — 向量总数

优势:
  - 自动持久化，无需手动 save/load
  - 支持元数据过滤 (按时间、来源、任务类型)
  - 增量写入，无需全量重建索引
  - Python 原生，零部署依赖
"""

from __future__ import annotations

from pathlib import Path


class ChromaVectorStore:
    """ChromaDB 向量存储封装 — 兼容 L3EpisodicArchive 接口。"""

    def __init__(self, persist_dir: str = "data/chromadb", collection_name: str = "episodic_memory"):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self):
        try:
            import chromadb
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            raise ImportError(
                "chromadb 未安装。运行: pip install chromadb"
            )

    # ── Public API (兼容 L3EpisodicArchive) ───────────────────────────

    def add(
        self,
        embeddings: list[list[float]],
        keys: list[str],
        metadata: list[dict] | None = None,
    ):
        """批量写入向量。

        Args:
            embeddings: 向量列表 (N × D)
            keys: 唯一标识符列表 (N)
            metadata: 元数据列表 (N), 可选
        """
        if not keys:
            return

        ids = keys
        # ChromaDB 不接受空 dict 作为 metadata，无元数据时传 None
        metas = metadata if metadata else None

        # ChromaDB 要求 metadata 值为 str/int/float/bool
        clean_metas = None
        if metas:
            clean_metas = []
            for m in metas:
                clean = {}
                for k, v in m.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    else:
                        clean[k] = str(v)
                clean_metas.append(clean)

        add_kwargs = {"ids": ids, "embeddings": embeddings}
        if clean_metas is not None:
            add_kwargs["metadatas"] = clean_metas
        self._collection.add(**add_kwargs)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_meta: dict | None = None,
    ) -> list[dict]:
        """向量相似度检索。

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数
            filter_meta: 元数据过滤条件 (ChromaDB where 语法)

        Returns:
            [{id, key, score, metadata}, ...]
        """
        where = filter_meta or None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            where=where,
        )

        items = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, doc_id in enumerate(ids):
            score = 1.0 / (1.0 + distances[i]) if distances and i < len(distances) else 1.0
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            items.append({
                "id": doc_id,
                "key": doc_id,
                "score": score,
                "metadata": meta,
            })
        return items

    def save(self):
        """ChromaDB 自动持久化，此方法为空操作。"""
        pass

    def load(self) -> bool:
        """ChromaDB 自动从磁盘加载，此方法始终返回 True。"""
        return self._collection is not None

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        """清空所有向量。"""
        if self._collection.count() > 0:
            ids = self._collection.get()["ids"]
            if ids:
                self._collection.delete(ids=ids)

    def delete_by_keys(self, keys: list[str]):
        """按 key 删除指定向量。"""
        self._collection.delete(ids=keys)


def create_vector_store(
    backend: str = "chromadb",
    persist_dir: str = "data/chromadb",
) -> ChromaVectorStore:
    """创建向量存储实例。

    Args:
        backend: "chromadb" (未来可扩展 "faiss", "milvus")
        persist_dir: 持久化目录

    Returns:
        ChromaVectorStore 实例
    """
    if backend == "chromadb":
        return ChromaVectorStore(persist_dir=persist_dir)
    raise ValueError(f"不支持的向量存储后端: {backend}")
