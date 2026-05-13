"""Test Writer — template synthesis, evidence collection, LLM fallback."""

from __future__ import annotations

import pytest

from horizonrl.agent.writer import Writer
from horizonrl.schemas.task import TaskSpec, PlanGraph, PlanNode, TaskStatus
from horizonrl.schemas.result import StepResult, VerificationResult, EvidenceItem, ErrorType


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def sample_evidence():
    return [
        EvidenceItem(content="Transformer 注意力机制通过 QKV 投影实现多头并行计算",
                     source="web", source_type="web"),
        EvidenceItem(content="FlashAttention 将内存复杂度从 O(N²) 降到 O(N)",
                     source="arxiv", source_type="arxiv"),
        EvidenceItem(content="代码运行成功，准确率 95%",
                     source="code", source_type="code_output"),
    ]


@pytest.fixture
def sample_results(sample_evidence):
    rs = {}
    for i in range(4):
        rs[f"task_{i}"] = StepResult(
            task_id=f"task_{i}", success=True,
            output=f"任务{i}完成",
            evidence=sample_evidence[: i + 1] if i < 3 else sample_evidence,
            tokens_used=100, elapsed=1.0,
        )
    return rs


@pytest.fixture
def sample_plan():
    g = PlanGraph()
    for i in range(4):
        spec = TaskSpec(
            id=f"task_{i}", name=f"子任务{i}",
            description=f"描述{i}", tool_names=["web_search"],
        )
        g.nodes[f"task_{i}"] = PlanNode(spec=spec, status=TaskStatus.SUCCESS)
    return g


@pytest.fixture
def sample_verifications():
    return {
        f"task_{i}": VerificationResult(
            pass_=True, score=0.85, error_type=ErrorType.NONE,
            feedback="通过",
        )
        for i in range(4)
    }


# ─── Tests ──────────────────────────────────────────────────────────────────


class TestWriterTemplate:
    def test_synthesize_returns_report(self, sample_plan, sample_results,
                                        sample_verifications):
        writer = Writer(mode="template")
        report = writer.synthesize(
            "Transformer 注意力机制",
            sample_plan, sample_results, sample_verifications,
        )
        assert len(report) > 100
        assert "Transformer" in report
        assert "## " in report  # Markdown 标题

    def test_report_contains_evidence_content(self, sample_plan, sample_results,
                                               sample_verifications):
        writer = Writer()
        report = writer.synthesize(
            "测试", sample_plan, sample_results, sample_verifications,
        )
        assert "FlashAttention" in report
        assert "QKV" in report

    def test_report_has_structure_sections(self, sample_plan, sample_results,
                                            sample_verifications):
        writer = Writer()
        report = writer.synthesize(
            "测试研究", sample_plan, sample_results, sample_verifications,
        )
        # 应有标题和至少一个章节
        assert report.startswith("# ")
        assert "## " in report

    def test_empty_results(self):
        writer = Writer()
        report = writer.synthesize("测试", PlanGraph(), {}, {})
        assert len(report) > 0
        assert "测试" in report

    def test_collect_evidence_dedup(self):
        writer = Writer()
        results = {
            "t1": StepResult(task_id="t1", success=True,
                             evidence=[EvidenceItem(content="相同内容", source_type="web"),
                                       EvidenceItem(content="相同内容", source_type="web")]),
        }
        evidence = writer._collect_evidence(results)
        assert len(evidence) == 1  # 去重

    def test_collect_tasks(self, sample_plan, sample_results, sample_verifications):
        writer = Writer()
        tasks = writer._collect_tasks(sample_plan, sample_results, sample_verifications)
        assert len(tasks) == 4
        assert all("name" in t for t in tasks)
        assert all("score" in t for t in tasks)

    def test_synthesize_with_failures(self):
        """包含失败任务的报告。"""
        g = PlanGraph()
        spec = TaskSpec(id="t1", name="失败任务", description="会失败",
                        tool_names=["web_search"])
        g.nodes["t1"] = PlanNode(spec=spec, status=TaskStatus.FAILED)

        results = {"t1": StepResult(task_id="t1", success=False, output="")}
        verifications = {
            "t1": VerificationResult(
                pass_=False, score=0.2, error_type=ErrorType.TOOL_ERROR,
                feedback="工具调用超时",
            ),
        }

        writer = Writer()
        report = writer.synthesize("测试", g, results, verifications)
        assert "失败任务" in report
        assert "未通过" in report

    def test_synthesize_with_memory_context(self, sample_plan, sample_results,
                                              sample_verifications):
        from horizonrl.memory.hierarchical_memory import MemoryContext, MemoryEntry

        ctx = MemoryContext(
            summaries=["研究总结: Transformer 注意力机制是当前主流方案"],
            recent_steps=[
                MemoryEntry(task_id="t1", task_name="搜索", output="找到论文",
                            success=True),
            ],
            stats={"completed": 4, "total": 4, "success_rate": 1.0, "replans": 0},
        )
        writer = Writer()
        report = writer.synthesize(
            "测试", sample_plan, sample_results, sample_verifications,
            memory_ctx=ctx,
        )
        assert "主流方案" in report

    def test_synthesize_with_no_plan(self):
        """没有 PlanGraph 时应不崩溃。"""
        writer = Writer()
        report = writer.synthesize("测试")
        assert len(report) > 0


class TestWriterLLM:
    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self, sample_plan, sample_results,
                                          sample_verifications):
        """LLM 不可用时回退到模板模式。"""
        from horizonrl.llm.client import LLMClient
        from horizonrl.config.settings import LLMConfig

        config = LLMConfig(
            provider="openai", model="test", api_key="sk-test",
            base_url="https://test.local",
        )
        client = LLMClient(config)
        writer = Writer(mode="llm", llm_client=client)

        report = await writer.synthesize_async(
            "测试", sample_plan, sample_results, sample_verifications,
        )
        # 应回退到模板生成
        assert len(report) > 100
        assert "测试" in report

    def test_template_mode_sync(self, sample_plan, sample_results,
                                 sample_verifications):
        """sync 模式直接生成。"""
        writer = Writer(mode="template")
        report = writer.synthesize(
            "测试研究", sample_plan, sample_results, sample_verifications,
        )
        assert "## " in report


class TestEvidenceGrouping:
    def test_groups_by_type(self):
        writer = Writer()
        results = {
            "t1": StepResult(task_id="t1", success=True, evidence=[
                EvidenceItem(content="web内容1", source_type="web"),
                EvidenceItem(content="arxiv内容1", source_type="arxiv"),
                EvidenceItem(content="web内容2", source_type="web"),
                EvidenceItem(content="code内容1", source_type="code_output"),
                EvidenceItem(content="unknown内容1", source_type="unknown_type"),
            ]),
        }
        evidence = writer._collect_evidence(results)
        types = {e["type"] for e in evidence}
        assert "web" in types
        assert "arxiv" in types
        assert "code_output" in types
