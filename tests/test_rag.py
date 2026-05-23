"""RAG 测试 — parser / document_store / retrieval plugin / API。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════════════
# Parser 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestParser:
    def test_parse_txt(self):
        from horizonrl.rag.parser import parse_document

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        ) as f:
            f.write("Hello World\n测试文档内容")
            path = f.name

        try:
            result = parse_document(path)
            assert "Hello World" in result
            assert "测试文档内容" in result
        finally:
            os.unlink(path)

    def test_parse_md(self):
        from horizonrl.rag.parser import parse_document

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as f:
            f.write("# Title\n\nContent paragraph.")
            path = f.name

        try:
            result = parse_document(path)
            assert "# Title" in result
            assert "Content paragraph" in result
        finally:
            os.unlink(path)

    def test_parse_unsupported_format(self):
        from horizonrl.rag.parser import parse_document

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False, encoding="utf-8",
        ) as f:
            f.write("test")
            path = f.name

        try:
            import pytest
            with pytest.raises(ValueError, match="不支持"):
                parse_document(path)
        finally:
            os.unlink(path)

    def test_parse_nonexistent_file(self):
        from horizonrl.rag.parser import parse_document

        import pytest
        with pytest.raises(FileNotFoundError):
            parse_document("/nonexistent/file.txt")


# ═══════════════════════════════════════════════════════════════════════════
# DocumentStore 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestDocumentStore:
    def test_init(self):
        from horizonrl.rag.document_store import DocumentStore

        store = DocumentStore()
        # ChromaDB may or may not be available, but store should exist
        assert store is not None

    def test_chunk_text(self):
        from horizonrl.rag.document_store import _chunk_text

        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。" * 50
        chunks = _chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 0
        for c in chunks:
            assert len(c) > 0

    def test_chunk_text_short(self):
        from horizonrl.rag.document_store import _chunk_text

        chunks = _chunk_text("short text", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_chunk_text_empty(self):
        from horizonrl.rag.document_store import _chunk_text

        chunks = _chunk_text("")
        assert len(chunks) == 0

    def test_ngram_embed(self):
        from horizonrl.rag.document_store import _ngram_embed

        vec = _ngram_embed("测试文本", dim=128)
        assert len(vec) == 128
        assert all(isinstance(v, float) for v in vec)
        # Should be L2 normalized
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_add_and_search(self):
        from horizonrl.rag.document_store import DocumentStore, _ngram_embed

        store = DocumentStore()
        result = store.add_document(
            filename="test.txt",
            content="Transformer 是一种基于自注意力机制的神经网络架构。" * 20,
        )
        assert result["chunk_count"] > 0
        assert result["doc_id"]

        # Search without embedding client (uses n-gram fallback)
        results = store.search("自注意力机制", top_k=3)
        assert len(results) > 0
        assert results[0]["filename"] == "test.txt"
        assert "doc_id" in results[0]

        # Cleanup
        store.delete_document(result["doc_id"])

    def test_list_documents(self):
        from horizonrl.rag.document_store import DocumentStore

        store = DocumentStore()
        result = store.add_document(
            filename="list_test.txt",
            content="测试文档列表功能的内容。" * 30,
        )
        docs = store.list_documents()
        filenames = [d["filename"] for d in docs]
        assert "list_test.txt" in filenames

        store.delete_document(result["doc_id"])

    def test_delete_document(self):
        from horizonrl.rag.document_store import DocumentStore

        store = DocumentStore()
        result = store.add_document(
            filename="delete_test.txt",
            content="待删除的测试文档。" * 40,
        )
        doc_id = result["doc_id"]

        store.delete_document(doc_id)
        docs = store.list_documents()
        assert doc_id not in [d["doc_id"] for d in docs]

    def test_search_with_doc_id_filter(self):
        from horizonrl.rag.document_store import DocumentStore

        store = DocumentStore()
        r1 = store.add_document(filename="a.txt", content="关于机器学习的内容。" * 30)
        r2 = store.add_document(filename="b.txt", content="关于深度学习的内容。" * 30)

        # Search scoped to doc a only
        results = store.search("机器学习", doc_id=r1["doc_id"])
        for r in results:
            assert r["doc_id"] == r1["doc_id"]

        store.delete_document(r1["doc_id"])
        store.delete_document(r2["doc_id"])


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval Plugin 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrievalPlugin:
    def test_config_defaults(self):
        from plugins.retrieval_tool import RetrievalPluginConfig

        cfg = RetrievalPluginConfig()
        assert cfg.top_k == 5

    def test_build_params(self):
        from plugins.retrieval_tool import RetrievalPlugin

        plugin = RetrievalPlugin()
        params = plugin.build_params("搜索Transformer注意力机制", "")
        assert params["query"] == "Transformer注意力机制"
        assert params["top_k"] == 5

    def test_extract_evidence(self):
        from plugins.retrieval_tool import RetrievalPlugin

        plugin = RetrievalPlugin()
        output = json.dumps([
            {"chunk_text": "重要发现: ...", "filename": "paper.pdf",
             "doc_id": "abc", "chunk_index": 0, "score": 0.95},
        ])
        evs = plugin.extract_evidence(output)
        assert len(evs) == 1
        assert evs[0].source_type == "local_document"
        assert "paper.pdf" in evs[0].source

    def test_extract_evidence_fallback(self):
        from plugins.retrieval_tool import RetrievalPlugin

        plugin = RetrievalPlugin()
        evs = plugin.extract_evidence("not json")
        assert len(evs) == 1
        assert evs[0].source_type == "local_document"

    def test_execute_no_query(self):
        import asyncio
        import json as _json
        from plugins.retrieval_tool import RetrievalPlugin

        plugin = RetrievalPlugin()
        result = asyncio.get_event_loop().run_until_complete(
            plugin.execute(query="")
        )
        data = _json.loads(result)
        assert "error" in data

    def test_plugin_discovered(self):
        from horizonrl.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        result = registry.discover("plugins")
        assert "retrieval_tool" in result


# ═══════════════════════════════════════════════════════════════════════════
# 集成: retrieval_tool 与 AgentWorker
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrievalWorkerIntegration:
    def test_worker_builds_params_for_retrieval(self):
        from horizonrl.agent.worker import AgentWorker
        from horizonrl.tools.manager import ToolManager
        from horizonrl.schemas.task import TaskSpec
        from plugins.retrieval_tool import RetrievalPlugin

        mgr = ToolManager()
        mgr.register_plugin("retrieval_tool", RetrievalPlugin())
        worker = AgentWorker(worker_id="w1", tool_manager=mgr)
        task = TaskSpec(
            id="t1", name="test", description="搜索本地文档中的相关内容",
            tool_names=["retrieval_tool"], depends_on=[], context="",
        )
        params = worker._build_params("retrieval_tool", task)
        assert params["query"] == "本地文档中的相关内容"
        assert params["top_k"] == 5
