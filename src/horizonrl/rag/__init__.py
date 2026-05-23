"""RAG (Retrieval-Augmented Generation) — 本地文档索引与检索。"""

from horizonrl.rag.document_store import DocumentStore
from horizonrl.rag.parser import parse_document

__all__ = ["DocumentStore", "parse_document"]
