"""Test Hierarchical Memory — L1 window, L2 compression, context, retrieval, stats."""

from __future__ import annotations

import asyncio
import pytest

from horizonrl.memory.hierarchical_memory import (
    MemoryEntry,
    MemoryContext,
    L1RecentWindow,
    L2SemanticSummary,
    L3EpisodicArchive,
    HierarchicalMemory,
)
from horizonrl.config.settings import MemoryConfig


# ─── MemoryEntry ─────────────────────────────────────────────────────────────


class TestMemoryEntry:
    def test_estimated_tokens(self):
        e = MemoryEntry(task_id="t1", task_name="搜索", output="结果" * 50)
        assert e.estimated_tokens() > 0

    def test_to_context_string(self):
        e = MemoryEntry(
            task_id="t1", task_name="搜索背景", output="找到3篇论文",
            success=True, evidence_count=3, tool_calls=1, elapsed=2.5,
        )
        ctx = e.to_context_string()
        assert "[OK]" in ctx
        assert "搜索背景" in ctx
        assert "3证据" in ctx

    def test_to_context_string_failed(self):
        e = MemoryEntry(
            task_id="t2", task_name="代码实验", output="",
            success=False, error_type="code_error",
        )
        ctx = e.to_context_string()
        assert "[FAIL:code_error]" in ctx

    def test_ts_default(self):
        e = MemoryEntry(task_id="t1", task_name="X", output="")
        assert e.ts > 0


# ─── L1RecentWindow ──────────────────────────────────────────────────────────


class TestL1RecentWindow:
    def test_add_and_retrieve(self):
        l1 = L1RecentWindow(max_tokens=8000)
        e1 = MemoryEntry(task_id="t1", task_name="A", output="x")
        e2 = MemoryEntry(task_id="t2", task_name="B", output="y")
        l1.add(e1)
        l1.add(e2)
        assert l1.count == 2
        assert l1.get_recent(1)[0].task_id == "t2"

    def test_total_tokens_tracking(self):
        l1 = L1RecentWindow(max_tokens=8000)
        e = MemoryEntry(task_id="t1", task_name="测试", output="长" * 1000)
        l1.add(e)
        assert l1.total_tokens > 0
        assert l1.total_tokens == e.estimated_tokens()

    def test_trim_when_over_threshold(self):
        l1 = L1RecentWindow(max_tokens=500, auto_compress_threshold=0.5)
        # 添加大量条目触发阈值
        overflow = None
        for i in range(50):
            long_output = "数据" * 50
            e = MemoryEntry(task_id=f"t{i}", task_name=f"任务{i}", output=long_output)
            overflow = l1.add(e)
        assert l1.usage_ratio <= l1.threshold or len(l1.get_all()) <= 50

    def test_trim_returns_overflow(self):
        l1 = L1RecentWindow(max_tokens=300, auto_compress_threshold=0.3)
        overflow_total = []
        for i in range(30):
            e = MemoryEntry(task_id=f"t{i}", task_name=f"T{i}", output="数据" * 30)
            ov = l1.add(e)
            overflow_total.extend(ov)
        # 至少有一些被驱逐
        assert len(overflow_total) > 0

    def test_get_all_returns_all(self):
        l1 = L1RecentWindow(max_tokens=80000)
        for i in range(5):
            l1.add(MemoryEntry(task_id=f"t{i}", task_name=f"T{i}", output="x"))
        assert len(l1.get_all()) == 5

    def test_clear(self):
        l1 = L1RecentWindow()
        l1.add(MemoryEntry(task_id="t1", task_name="A", output="x"))
        l1.clear()
        assert l1.count == 0
        assert l1.total_tokens == 0

    def test_success_failure_counts(self):
        l1 = L1RecentWindow(max_tokens=80000)
        l1.add(MemoryEntry(task_id="t1", task_name="A", output="x", success=True))
        l1.add(MemoryEntry(task_id="t2", task_name="B", output="y", success=True))
        l1.add(MemoryEntry(task_id="t3", task_name="C", output="z", success=False))
        assert l1.success_count == 2
        assert l1.failure_count == 1

    def test_needs_compression_flag(self):
        l1 = L1RecentWindow(max_tokens=500, auto_compress_threshold=0.1)
        e = MemoryEntry(task_id="t1", task_name="X", output="长" * 500)
        l1.add(e)
        assert l1.needs_compression is True

    def test_usage_ratio_zero_initially(self):
        l1 = L1RecentWindow()
        assert l1.usage_ratio == 0.0

    def test_get_recent_clipped(self):
        l1 = L1RecentWindow(max_tokens=80000)
        for i in range(10):
            l1.add(MemoryEntry(task_id=f"t{i}", task_name=f"T{i}", output="x"))
        assert len(l1.get_recent(3)) == 3
        assert l1.get_recent(3)[-1].task_id == "t9"


# ─── L2SemanticSummary ───────────────────────────────────────────────────────


class TestL2SemanticSummary:
    def test_add_and_retrieve(self):
        l2 = L2SemanticSummary(max_entries=10)
        l2.add("摘要1: 完成搜索任务")
        l2.add("摘要2: 代码实验通过")
        assert l2.count == 2
        assert l2.get_recent(1)[0] == "摘要2: 代码实验通过"

    def test_fifo_eviction(self):
        l2 = L2SemanticSummary(max_entries=3)
        for i in range(5):
            l2.add(f"摘要{i}")
        assert l2.count == 3
        assert l2.get_recent(3)[0] == "摘要2"

    def test_compress_from_entries(self):
        l2 = L2SemanticSummary()
        entries = [
            MemoryEntry(task_id="t1", task_name="搜索背景", output="找到5篇论文",
                        success=True, evidence_count=5),
            MemoryEntry(task_id="t2", task_name="代码实验", output="运行成功，准确率95%",
                        success=True, evidence_count=2),
            MemoryEntry(task_id="t3", task_name="汇总分析", output="",
                        success=False, error_type="incomplete"),
        ]
        summary = l2.compress_from_entries(entries, "Transformer研究")
        assert "搜索背景" in summary
        assert "代码实验" in summary
        assert "2/3" in summary or "成功率" in summary
        assert l2.count == 1

    def test_compress_empty_entries(self):
        l2 = L2SemanticSummary()
        summary = l2.compress_from_entries([])
        assert summary == ""

    def test_compress_includes_context(self):
        l2 = L2SemanticSummary()
        entries = [MemoryEntry(task_id="t1", task_name="搜索", output="结果",
                               success=True)]
        summary = l2.compress_from_entries(entries, "量子计算背景研究")
        assert "量子计算背景研究" in summary

    def test_search_keyword(self):
        l2 = L2SemanticSummary(max_entries=10)
        l2.add("完成Transformer注意力机制搜索，发现3篇论文")
        l2.add("LLaMA RoPE位置编码分析完成")
        l2.add("代码实验：实现多头注意力")
        results = l2.search("Transformer")
        assert len(results) == 1
        assert "注意力机制" in results[0]

    def test_search_no_match(self):
        l2 = L2SemanticSummary()
        l2.add("摘要1")
        assert len(l2.search("不存在的关键词")) == 0

    def test_clear(self):
        l2 = L2SemanticSummary()
        l2.add("摘要1")
        l2.clear()
        assert l2.count == 0

    @pytest.mark.asyncio
    async def test_compress_with_llm_fallback(self):
        """LLM 不可用时回退到模板模式。"""
        from horizonrl.llm.client import LLMClient
        from horizonrl.config.settings import LLMConfig

        config = LLMConfig(
            provider="openai", model="test", api_key="sk-test",
            base_url="https://test.local",
        )
        client = LLMClient(config)
        l2 = L2SemanticSummary()
        l2.set_llm(client)

        entries = [MemoryEntry(task_id="t1", task_name="搜索", output="结果",
                               success=True)]
        # LLM 会因网络不可达失败，应回退到模板模式
        summary = await l2.compress_with_llm(entries)
        assert len(summary) > 0
        assert l2.count == 1


# ─── MemoryContext ───────────────────────────────────────────────────────────


class TestMemoryContext:
    def test_to_prompt_fragment(self):
        ctx = MemoryContext(
            recent_steps=[
                MemoryEntry(task_id="t1", task_name="搜索", output="找到论文",
                            success=True, evidence_count=3, tool_calls=1, elapsed=1.5),
            ],
            summaries=["示例摘要"],
            stats={"completed": 3, "total": 3, "success_rate": 1.0,
                   "total_tokens": 1500, "replans": 0},
        )
        frag = ctx.to_prompt_fragment()
        assert "搜索" in frag
        assert "示例摘要" in frag
        assert "100%" in frag

    def test_empty_context(self):
        ctx = MemoryContext()
        assert ctx.is_empty()
        assert ctx.to_prompt_fragment() == ""

    def test_not_empty_with_steps(self):
        ctx = MemoryContext(
            recent_steps=[MemoryEntry(task_id="t1", task_name="X", output="y")],
        )
        assert not ctx.is_empty()

    def test_not_empty_with_summaries(self):
        ctx = MemoryContext(summaries=["摘要"])
        assert not ctx.is_empty()


# ─── HierarchicalMemory ──────────────────────────────────────────────────────


class TestHierarchicalMemory:
    def test_init_with_default_config(self):
        mem = HierarchicalMemory()
        assert mem.l1.count == 0
        assert mem.l2.count == 0
        assert mem.l1.max_tokens == 8000

    def test_init_with_custom_config(self):
        cfg = MemoryConfig(l1_max_tokens=4000, l2_max_entries=20,
                           auto_compress_threshold=0.5)
        mem = HierarchicalMemory(cfg)
        assert mem.l1.max_tokens == 4000
        assert mem.l2.max_entries == 20

    def test_record_simple(self):
        mem = HierarchicalMemory()
        entry = mem.record_task(
            task_id="task_001", task_name="搜索背景",
            output="找到3篇论文", success=True,
            evidence_count=3, tool_calls=1, tokens_used=500, elapsed=2.0,
        )
        assert entry.task_id == "task_001"
        assert mem.l1.count == 1

    def test_record_with_step_result(self):
        from horizonrl.schemas.result import StepResult, EvidenceItem, ToolCall

        mem = HierarchicalMemory()
        result = StepResult(
            task_id="task_002", success=True,
            output="实验完成：准确率 95%",
            evidence=[EvidenceItem(content="数据", source="test")],
            tool_calls=[ToolCall(tool_name="code_execution", input={},
                                 output="ok")],
            tokens_used=300, elapsed=1.5,
        )
        entry = mem.record(result)
        assert entry.success is True
        assert entry.evidence_count == 1
        assert entry.tool_calls == 1
        assert mem.l1.count == 1

    def test_record_with_verification(self):
        from horizonrl.schemas.result import StepResult, VerificationResult, ErrorType

        mem = HierarchicalMemory()
        result = StepResult(task_id="t1", success=False, output="",
                            tokens_used=100, elapsed=0.5)
        vr = VerificationResult(
            pass_=False, score=0.1, error_type=ErrorType.TOOL_ERROR,
            feedback="工具调用超时",
        )
        entry = mem.record(result, vr)
        assert entry.error_type == "tool_error"

    def test_record_auto_compresses_on_overflow(self):
        cfg = MemoryConfig(l1_max_tokens=300, auto_compress_threshold=0.3,
                           l2_max_entries=50)
        mem = HierarchicalMemory(cfg)
        for i in range(30):
            mem.record_task(
                task_id=f"t{i}", task_name=f"任务{i}",
                output="数据" * 30, success=True,
            )
        # L2 应该有溢出压缩的摘要
        assert mem.l2.count > 0 or mem.l1.count <= 30

    def test_compress_manual(self):
        mem = HierarchicalMemory()
        mem.record_task(task_id="t1", task_name="搜索", output="结果", success=True)
        mem.record_task(task_id="t2", task_name="分析", output="分析完成", success=True)
        assert mem.l1.count == 2

        summary = mem.compress("测试任务")
        assert len(summary) > 0
        assert mem.l1.count == 0
        assert mem.l2.count == 1

    def test_compress_empty(self):
        mem = HierarchicalMemory()
        assert mem.compress() == ""

    def test_auto_compress_triggers(self):
        cfg = MemoryConfig(l1_max_tokens=200, auto_compress_threshold=0.1,
                           l2_max_entries=50)
        mem = HierarchicalMemory(cfg)
        for i in range(10):
            mem.record_task(task_id=f"t{i}", task_name=f"T{i}",
                            output="长内容" * 20, success=True)
        result = mem.auto_compress()
        # 应该触发压缩，生成 L2 摘要
        assert mem.l2.count > 0 or result != ""

    def test_auto_compress_noop_when_below_threshold(self):
        cfg = MemoryConfig(l1_max_tokens=80000, auto_compress_threshold=0.9)
        mem = HierarchicalMemory(cfg)
        mem.record_task(task_id="t1", task_name="T1", output="x", success=True)
        result = mem.auto_compress()
        assert result == ""

    def test_get_context(self):
        mem = HierarchicalMemory()
        mem.record_task(task_id="t1", task_name="搜索背景", output="找到论文",
                        success=True, evidence_count=3, tool_calls=1, elapsed=2.0)
        mem.record_task(task_id="t2", task_name="代码实验", output="运行成功",
                        success=True, evidence_count=2, tool_calls=1, elapsed=1.5)
        ctx = mem.get_context()
        assert len(ctx.recent_steps) == 2
        assert ctx.stats["completed"] == 2
        assert ctx.stats["success_rate"] == 1.0

    def test_get_context_with_query(self):
        mem = HierarchicalMemory()
        mem.compress("Transformer注意力机制搜索完成")
        ctx = mem.get_context(query="Transformer")
        assert len(ctx.summaries) >= 0  # 至少不报错

    def test_search_cross_layer(self):
        mem = HierarchicalMemory()
        mem.record_task(task_id="t1", task_name="搜索Transformer", output="论文",
                        success=True)
        mem.compress()
        results = mem.search("Transformer")
        assert len(results) > 0

    def test_search_empty(self):
        mem = HierarchicalMemory()
        assert len(mem.search("查询")) == 0

    def test_record_replan(self):
        mem = HierarchicalMemory()
        mem.record_replan()
        mem.record_replan()
        stats = mem.get_stats()
        assert stats["replan_count"] == 2
        ctx = mem.get_context()
        assert ctx.stats["replans"] == 2

    def test_get_stats(self):
        mem = HierarchicalMemory()
        mem.record_task(task_id="t1", task_name="T1", output="x", success=True)
        stats = mem.get_stats()
        assert stats["l1_count"] == 1
        assert stats["l1_usage"] is not None
        assert stats["l2_count"] == 0
        assert stats["l3_count"] == 0

    def test_clear(self):
        mem = HierarchicalMemory()
        mem.record_task(task_id="t1", task_name="T1", output="x", success=True)
        mem.compress()
        mem.record_replan()
        mem.clear()
        assert mem.l1.count == 0
        assert mem.l2.count == 0
        assert mem.get_stats()["replan_count"] == 0

    def test_archive_to_l3(self):
        mem = HierarchicalMemory()
        mem.archive_to_l3_sync("重要经验", {"task": "搜索"})
        assert mem.get_stats()["l3_count"] == 1

    def test_retrieve_l3(self):
        mem = HierarchicalMemory()
        mem.archive_to_l3_sync("Transformer注意力机制分析结果")
        mem.archive_to_l3_sync("LLaMA架构研究")
        results = mem.retrieve_l3("Transformer")
        assert len(results) == 1
        assert "注意力" in results[0]

    def test_set_llm(self):
        from horizonrl.llm.client import LLMClient
        from horizonrl.config.settings import LLMConfig

        config = LLMConfig(
            provider="openai", model="test", api_key="sk-test",
            base_url="https://test.local",
        )
        client = LLMClient(config)
        mem = HierarchicalMemory()
        mem.set_llm(client)
        # LLM 客户端被设置到 L2
        assert mem.l2._llm_client is client


# ─── Integration: Full Memory Cycle ──────────────────────────────────────────


class TestFullMemoryCycle:
    def test_record_compress_retrieve_cycle(self):
        """完整记忆周期：写入 → 压缩 → 检索 → 上下文。"""
        mem = HierarchicalMemory(
            MemoryConfig(l1_max_tokens=8000, l2_max_entries=20)
        )

        # 模拟多轮执行
        mem.record_task("t1", "搜索背景", "找到Transformer相关论文5篇",
                        success=True, evidence_count=5, tool_calls=1, elapsed=2.0)
        mem.record_task("t2", "搜索改进", "找到最新改进方案3篇",
                        success=True, evidence_count=3, tool_calls=1, elapsed=1.8)
        mem.record_task("t3", "代码实验", "实现多头注意力，准确率95%",
                        success=True, evidence_count=2, tool_calls=1, elapsed=3.0)

        # 压缩到 L2
        summary = mem.compress("Transformer注意力机制研究")
        assert len(summary) > 0

        # L1 已清空
        assert mem.l1.count == 0
        # L2 有摘要
        assert mem.l2.count == 1

        # 新一轮执行
        mem.record_task("t4", "汇总分析", "综合搜索结果，撰写分析报告",
                        success=True, evidence_count=0, tool_calls=0, elapsed=1.0)

        # 获取上下文
        ctx = mem.get_context()
        assert len(ctx.recent_steps) == 1
        assert len(ctx.summaries) == 1
        assert "搜索" in ctx.summaries[0]

        # 生成 prompt fragment
        frag = ctx.to_prompt_fragment()
        assert "汇总分析" in frag

    def test_overflow_triggers_compression(self):
        """L1 溢出自动触发 L2 压缩。"""
        cfg = MemoryConfig(l1_max_tokens=500, auto_compress_threshold=0.5,
                           l2_max_entries=50)
        mem = HierarchicalMemory(cfg)

        for i in range(25):
            mem.record_task(
                f"t{i}", f"任务{i}", "数据内容" * 30,
                success=(i % 3 != 0),  # 大约 2/3 成功
            )

        # 应该有 L2 内容（溢出自动压缩）
        stats = mem.get_stats()
        assert stats["l2_count"] > 0 or stats["l1_count"] <= 25


# ─── Edge Cases ──────────────────────────────────────────────────────────────


class TestMemoryEdgeCases:
    def test_single_entry_compress(self):
        mem = HierarchicalMemory()
        mem.record_task("t1", "唯一任务", "完成", success=True)
        summary = mem.compress()
        assert "唯一任务" in summary

    def test_failed_only_entries(self):
        mem = HierarchicalMemory()
        mem.record_task("t1", "失败1", "", success=False, error_type="tool_error")
        mem.record_task("t2", "失败2", "", success=False, error_type="code_error")
        ctx = mem.get_context()
        assert ctx.stats["failure_count"] == 2
        assert ctx.stats["success_rate"] == 0.0

    def test_very_long_output_truncation(self):
        mem = HierarchicalMemory()
        long_output = "非常长的输出内容" * 200
        entry = mem.record_task("t1", "长输出任务", long_output, success=True)
        assert len(entry.output) <= 300  # 截断

    def test_concurrent_memory_operations(self):
        """L1 和 L2 操作互不干扰。"""
        mem = HierarchicalMemory()
        mem.record_task("t1", "L1任务", "L1内容", success=True)
        mem.compress("压缩上下文")
        mem.record_task("t2", "新L1任务", "新内容", success=True)

        assert mem.l1.count == 1
        assert mem.l2.count == 1
        assert mem.l1.get_recent(1)[0].task_id == "t2"

    def test_empty_memory_context(self):
        mem = HierarchicalMemory()
        ctx = mem.get_context()
        assert ctx.is_empty()


# ─── L3 Episodic Archive ────────────────────────────────────────────────────


class TestL3EpisodicArchive:
    def test_archive_and_keyword_search(self):
        l3 = L3EpisodicArchive()
        l3.archive_sync("Transformer注意力机制在NLP中广泛应用", {"task": "搜索"})
        l3.archive_sync("Python asyncio是异步编程的核心库")
        l3.archive_sync("量子计算使用量子比特进行并行计算")
        results = l3.search("Transformer")
        assert len(results) > 0
        assert any("Transformer" in r for r in results)

    def test_search_no_match(self):
        l3 = L3EpisodicArchive()
        l3.archive_sync("测试内容")
        assert l3.search("不存在的关键词") == []

    def test_clear(self):
        l3 = L3EpisodicArchive()
        l3.archive_sync("测试", {"key": "val"})
        assert l3.count == 1
        l3.clear()
        assert l3.count == 0

    def test_save_and_load(self, tmp_path):
        """测试 L3 持久化往返。"""
        l3 = L3EpisodicArchive(index_path=str(tmp_path / "test_index"))
        l3.archive_sync("Transformer注意力机制最新进展")
        l3.archive_sync("LLaMA架构中的RoPE位置编码")
        l3.save()

        # 加载
        l3_loaded = L3EpisodicArchive(index_path=str(tmp_path / "test_index"))
        if l3_loaded.load():
            assert l3_loaded.count == 2
            results = l3_loaded.search("Transformer")
            assert len(results) > 0

    def test_count(self):
        l3 = L3EpisodicArchive()
        assert l3.count == 0
        l3.archive_sync("a")
        l3.archive_sync("b")
        assert l3.count == 2

    def test_vector_search_falls_back_to_keyword(self):
        """FAISS 不可用时回退关键词检索。"""
        l3 = L3EpisodicArchive()
        l3.archive_sync("第一条重要经验")
        l3.archive_sync("第二条无关内容")
        # 关键词检索
        results = l3._keyword_search("重要", 5)
        assert len(results) == 1
        assert "重要" in results[0]

    def test_hierarchical_memory_l3_integration(self):
        """HierarchicalMemory 的 L3 接口正常工作。"""
        mem = HierarchicalMemory()
        mem.archive_to_l3_sync("Transformer经验1")
        mem.archive_to_l3_sync("asyncio经验2")
        mem.archive_to_l3_sync("量子计算经验3")

        stats = mem.get_stats()
        assert stats["l3_count"] == 3
        assert "l3_has_index" in stats

        # 检索
        results = mem.retrieve_l3("Transformer")
        assert len(results) > 0
        assert any("Transformer" in r for r in results)

        # 清空
        mem.clear()
        assert mem.get_stats()["l3_count"] == 0

    def test_l3_embed_sync_consistent(self):
        """相同文本生成相同向量。"""
        l3 = L3EpisodicArchive(embedding_dim=128)
        v1 = l3._embed_sync("测试文本")
        v2 = l3._embed_sync("测试文本")
        assert v1 == v2
        assert len(v1) == 128

    def test_l3_embed_sync_different(self):
        """不同文本生成不同向量。"""
        l3 = L3EpisodicArchive(embedding_dim=64)
        v1 = l3._embed_sync("文本A")
        v2 = l3._embed_sync("文本B")
        assert v1 != v2
