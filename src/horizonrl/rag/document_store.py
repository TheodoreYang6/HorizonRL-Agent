"""
DocumentStore — ChromaDB 封装的文档向量存储。

管理文档的分块、嵌入、索引和检索。使用独立 ChromaDB 集合 "user_documents"，
与 L3 EpisodicMemory 和 ResearchContext 隔离。
"""

from __future__ import annotations

import hashlib
import time
import uuid


class DocumentStore:
    """文档向量存储 — 分块、嵌入、索引、检索。

    内部使用 ChromaVectorStore，集合名为 "user_documents"。
    若 ChromaDB 不可用则回退 FAISS/n-gram。
    """

    def __init__(self, persist_dir: str = "data/chromadb"):
        self._persist_dir = persist_dir
        self._store = None
        self._init_store()

    def _init_store(self):
        try:
            from horizonrl.memory.vector_store import ChromaVectorStore
            self._store = ChromaVectorStore(
                persist_dir=self._persist_dir,
                collection_name="user_documents",
            )
        except Exception:
            self._store = None

    @property
    def is_available(self) -> bool:
        return self._store is not None

    def add_document(
        self,
        filename: str,
        content: str,
        embedding_client=None,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> dict:
        """解析文档、分块、嵌入、索引。

        Args:
            filename: 原始文件名（元数据用）。
            content: 文档纯文本内容。
            embedding_client: LLMClient 实例（有 embed 方法），None 则用 n-gram。
            chunk_size: 每块字符数。
            overlap: 块间重叠字符数。

        Returns:
            {"doc_id": str, "chunk_count": int, "total_chars": int}
        """
        doc_id = uuid.uuid4().hex[:12]
        chunks = _chunk_text(content, chunk_size, overlap)

        if not chunks:
            return {"doc_id": doc_id, "chunk_count": 0, "total_chars": 0}

        embeddings: list[list[float]] = []
        keys: list[str] = []
        metadata: list[dict] = []

        for i, chunk in enumerate(chunks):
            vec = _embed_text(chunk, embedding_client)
            embeddings.append(vec)
            keys.append(f"{doc_id}_{i}")
            metadata.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk[:1000],
                "char_count": len(chunk),
                "timestamp": time.time(),
            })

        if self._store is not None:
            self._store.add(embeddings=embeddings, keys=keys, metadata=metadata)
        else:
            _add_to_faiss(embeddings, keys, metadata)

        return {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "total_chars": len(content),
        }

    def search(
        self,
        query: str,
        embedding_client=None,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[dict]:
        """语义检索相关文档块。

        Args:
            query: 查询文本。
            embedding_client: LLMClient 实例。
            top_k: 返回最大结果数。
            doc_id: 可选，限定在指定文档中检索。

        Returns:
            [{"chunk_text", "filename", "doc_id", "chunk_index", "score"}, ...]
        """
        if not self._store:
            return _search_faiss(query, embedding_client, top_k, doc_id)

        q_vec = _embed_text(query, embedding_client)
        filter_meta = {"doc_id": doc_id} if doc_id else None

        try:
            raw = self._store.search(q_vec, top_k=top_k, filter_meta=filter_meta)
        except Exception:
            return []

        results: list[dict] = []
        for item in raw:
            meta = item.get("metadata", {})
            results.append({
                "chunk_text": meta.get("text", ""),
                "filename": meta.get("filename", ""),
                "doc_id": meta.get("doc_id", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(item.get("score", 0.0), 4),
            })
        return results

    def delete_document(self, doc_id: str) -> bool:
        """删除指定文档的所有块。"""
        if self._store is not None and self._store._collection is not None:
            # 获取所有以 doc_id_ 开头的 key
            try:
                all_data = self._store._collection.get()
                ids_to_delete = [
                    id_ for id_ in (all_data.get("ids") or [])
                    if id_.startswith(doc_id + "_")
                ]
                if ids_to_delete:
                    self._store._collection.delete(ids=ids_to_delete)
            except Exception:
                pass
        _delete_from_faiss(doc_id)
        return True

    def list_documents(self) -> list[dict]:
        """列出已索引的文档（去重）。"""
        if self._store is None:
            return _list_faiss_docs()

        # ChromaDB 无直接 list 接口，用 get 取所有 metadata 去重
        try:
            all_data = self._store._collection.get()
            seen: dict[str, dict] = {}
            if all_data and "metadatas" in all_data:
                for meta in all_data["metadatas"]:
                    if meta and "doc_id" in meta:
                        did = meta["doc_id"]
                        if did not in seen:
                            seen[did] = {
                                "doc_id": did,
                                "filename": meta.get("filename", ""),
                                "chunk_count": 1,
                                "timestamp": meta.get("timestamp", 0),
                            }
                        else:
                            seen[did]["chunk_count"] += 1
            return sorted(seen.values(), key=lambda d: d.get("timestamp", 0), reverse=True)
        except Exception:
            return []

    def count(self) -> int:
        if self._store is not None:
            return self._store.count()
        return len(_FAISS_DOCS)


# ── 文本分块与嵌入 ──


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """简单定长分块，保持段落完整性。"""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # 尝试在段落边界处截断
        if end < len(text):
            last_break = max(
                chunk.rfind("\n\n"), chunk.rfind("\n"),
                chunk.rfind("。"), chunk.rfind("."),
            )
            if last_break > chunk_size // 2:
                end = start + last_break + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap if end < len(text) else len(text)
    return [c for c in chunks if c]


def _embed_text(text: str, embedding_client=None) -> list[float]:
    """对文本做嵌入，优先用 embedding_client，回退 n-gram。"""
    if embedding_client is not None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, embedding_client.embed(text))
                    result = future.result(timeout=30)
            else:
                result = loop.run_until_complete(embedding_client.embed(text))
            if result.is_success and result.embedding:
                return list(result.embedding)
        except Exception:
            pass
    return _ngram_embed(text)


def _ngram_embed(text: str, dim: int = 1024) -> list[float]:
    """n-gram MD5 哈希回退嵌入（确定性，零依赖）。"""
    vec = [0.0] * dim
    for n in (2, 3, 4):
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            h = hashlib.md5(ngram.encode()).hexdigest()
            idx = int(h, 16) % dim
            vec[idx] += 1.0
    # L2 归一化
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


# ── FAISS 回退存储（内存级别，ChromaDB 不可用时用） ──

_FAISS_DOCS: dict[str, dict] = {}  # doc_id → {keys, embeddings, metadatas}
_FAISS_INDEX = None  # faiss.IndexFlatL2


def _add_to_faiss(embeddings: list[list[float]], keys: list[str], metadatas: list[dict]):
    global _FAISS_INDEX
    try:
        import faiss
        import numpy as np
        dim = len(embeddings[0])
        if _FAISS_INDEX is None:
            _FAISS_INDEX = faiss.IndexFlatL2(dim)
        mat = np.array(embeddings, dtype=np.float32)
        _FAISS_INDEX.add(mat)
    except ImportError:
        pass
    for k, m in zip(keys, metadatas):
        _FAISS_DOCS[k] = m


def _search_faiss(
    query: str, embedding_client=None, top_k: int = 5, doc_id: str | None = None,
) -> list[dict]:
    global _FAISS_INDEX
    results: list[dict] = []
    if _FAISS_INDEX is None or not _FAISS_DOCS:
        return results
    try:
        import numpy as np
        q_vec = _embed_text(query, embedding_client)
        mat = np.array([q_vec], dtype=np.float32)
        distances, indices = _FAISS_INDEX.search(mat, top_k)
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            keys_list = list(_FAISS_DOCS.keys())
            if idx < len(keys_list):
                key = keys_list[idx]
                meta = _FAISS_DOCS.get(key, {})
                if doc_id and meta.get("doc_id") != doc_id:
                    continue
                results.append({
                    "chunk_text": meta.get("text", ""),
                    "filename": meta.get("filename", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": round(1.0 / (1.0 + float(dist)), 4),
                })
    except ImportError:
        pass
    return results


def _delete_from_faiss(doc_id: str):
    to_remove = [k for k, m in _FAISS_DOCS.items() if m.get("doc_id") == doc_id]
    for k in to_remove:
        _FAISS_DOCS.pop(k, None)


def _list_faiss_docs() -> list[dict]:
    seen: dict[str, dict] = {}
    for m in _FAISS_DOCS.values():
        did = m.get("doc_id", "")
        if did not in seen:
            seen[did] = {
                "doc_id": did, "filename": m.get("filename", ""),
                "chunk_count": 1, "timestamp": m.get("timestamp", 0),
            }
        else:
            seen[did]["chunk_count"] += 1
    return sorted(seen.values(), key=lambda d: d.get("timestamp", 0), reverse=True)
