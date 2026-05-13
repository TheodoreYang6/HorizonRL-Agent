"""Test Writer v2 — UserAnswerWriter, DebugReportRenderer, WriterConfig, metadata."""

from __future__ import annotations

import pytest

from horizonrl.agent.writer import (
    Writer,
    WriterConfig,
    UserAnswerWriter,
    DebugReportRenderer,
    _mock_warning,
    _evidence_ref_text,
)
from horizonrl.schemas.task import TaskSpec, PlanGraph, PlanNode, TaskStatus
from horizonrl.schemas.result import StepResult, VerificationResult, EvidenceItem, ErrorType, SearchProvenance
from horizonrl.schemas.report import ReportMetadata


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def sample_evidence():
    return [
        EvidenceItem(content="Transformer QKV多头并行计算", source="web", source_type="web"),
        EvidenceItem(content="FlashAttention O(N) 内存", source="arxiv", source_type="arxiv"),
        EvidenceItem(content="代码运行成功 准确率95%", source="code", source_type="code_output"),
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
        g.nodes[f"task_{i}"] = PlanNode(
            spec=TaskSpec(id=f"task_{i}", name=f"子任务{i}",
                          description=f"描述{i}", tool_names=["web_search"]),
            status=TaskStatus.SUCCESS,
        )
    return g


@pytest.fixture
def sample_verifications():
    return {
        f"task_{i}": VerificationResult(pass_=True, score=0.85, error_type=ErrorType.NONE, feedback="通过")
        for i in range(4)
    }


# ─── WriterConfig ───────────────────────────────────────────────────────────


class TestWriterConfig:
    def test_defaults(self):
        cfg = WriterConfig()
        assert cfg.enable_llm_writer is True
        assert cfg.default_author == "HorizonRL-Agent"
        assert cfg.export_dir == "summaries"
        assert cfg.max_evidence_items == 10
        assert cfg.include_debug_stats is False

    def test_custom(self):
        cfg = WriterConfig(enable_llm_writer=False, default_author="Test", export_dir="out")
        assert cfg.enable_llm_writer is False
        assert cfg.default_author == "Test"
        assert cfg.export_dir == "out"


# ─── _mock_warning ──────────────────────────────────────────────────────────


class TestMockWarning:
    def test_majority_mock(self):
        evidence = [{"is_mock": True}, {"is_mock": True}, {"is_mock": False}]
        result = _mock_warning(evidence)
        assert "Mock Demo" in result

    def test_no_mock(self):
        evidence = [{"is_mock": False}, {"is_mock": False}]
        result = _mock_warning(evidence)
        assert result == ""

    def test_empty(self):
        assert _mock_warning([]) == ""


# ─── _evidence_ref_text ─────────────────────────────────────────────────────


class TestEvidenceRefText:
    def test_normal_evidence(self):
        ev = {"provider": "web_search", "content": "测试内容", "source": "http://example.com"}
        text = _evidence_ref_text(ev, 1)
        assert "[证据 1" in text
        assert "web_search" in text
        assert "测试内容" in text

    def test_mock_evidence(self):
        ev = {"provider": "web_search", "is_mock": True, "content": "mock内容"}
        text = _evidence_ref_text(ev, 2)
        assert "Mock" in text


# ─── DebugReportRenderer ────────────────────────────────────────────────────


class TestDebugReportRenderer:
    def test_render_contains_dag_info(self, sample_plan, sample_results, sample_verifications):
        r = DebugReportRenderer()
        report = r.render("测试", sample_plan, sample_results, sample_verifications)
        assert "DEBUG" in report or "debug" in report.lower() or "执行报告" in report
        assert "task_id" in report or "task_0" in report

    def test_render_includes_stats(self, sample_plan, sample_results, sample_verifications):
        r = DebugReportRenderer()
        report = r.render("测试", sample_plan, sample_results, sample_verifications,
                          stats={"total_count": 4, "rounds": 2})
        assert "4" in report
        assert "2" in report

    def test_render_empty_plan(self):
        r = DebugReportRenderer()
        report = r.render("测试", None, {}, {})
        assert len(report) > 0

    def test_render_with_metadata(self, sample_plan, sample_results, sample_verifications):
        r = DebugReportRenderer()
        meta = ReportMetadata(session_id="s1", author="Test", mode="debug")
        report = r.render("测试", sample_plan, sample_results, sample_verifications, metadata=meta)
        assert "s1" in report

    def test_collect_tasks(self, sample_plan, sample_results, sample_verifications):
        r = DebugReportRenderer()
        tasks = r._collect_tasks(sample_plan, sample_results, sample_verifications)
        assert len(tasks) == 4
        assert all("name" in t for t in tasks)
        assert all("task_id" in t for t in tasks)

    def test_collect_evidence_dedup(self):
        r = DebugReportRenderer()
        results = {"t1": StepResult(task_id="t1", success=True, evidence=[
            EvidenceItem(content="相同内容", source_type="web"),
            EvidenceItem(content="相同内容", source_type="web"),
        ])}
        evidence = r._collect_evidence(results)
        assert len(evidence) == 1


# ─── UserAnswerWriter ───────────────────────────────────────────────────────


class TestUserAnswerWriter:
    def test_template_write_includes_provenance(self, sample_plan, sample_results):
        w = UserAnswerWriter()
        evidence = w._collect_evidence(sample_results)
        report = w._template_write("测试", evidence)
        assert "核心结论" in report
        assert "参考证据" in report

    def test_final_answer_excludes_debug_info(self, sample_results):
        """final_answer 不得包含 task_id、Token、耗时等调试信息。"""
        w = UserAnswerWriter()
        evidence = w._collect_evidence(sample_results)
        report = w._template_write("测试", evidence)
        assert "task_id" not in report
        assert "Token" not in report

    def test_mock_data_shows_warning(self):
        """mock 数据占比高时显示 Mock Demo 提示。"""
        w = UserAnswerWriter()
        mock_ev = [
            {"type": "web", "content": "mock内容1", "is_mock": True},
            {"type": "arxiv", "content": "mock内容2", "is_mock": True},
            {"type": "web", "content": "真实内容", "is_mock": False},
        ]
        report = w._template_write("测试", mock_ev)
        assert "Mock Demo" in report

    def test_empty_evidence(self):
        w = UserAnswerWriter()
        report = w._template_write("测试", [])
        assert "未找到" in report

    def test_collect_evidence_dedup(self):
        w = UserAnswerWriter()
        results = {"t1": StepResult(task_id="t1", success=True, evidence=[
            EvidenceItem(content="相同", source_type="web"),
            EvidenceItem(content="相同", source_type="web"),
        ])}
        evidence = w._collect_evidence(results)
        assert len(evidence) == 1

    def test_collect_evidence_groups_by_type(self):
        w = UserAnswerWriter()
        results = {"t1": StepResult(task_id="t1", success=True, evidence=[
            EvidenceItem(content="w1", source_type="web"),
            EvidenceItem(content="a1", source_type="arxiv"),
            EvidenceItem(content="c1", source_type="code_output"),
        ])}
        evidence = w._collect_evidence(results)
        types = {e["type"] for e in evidence}
        assert "web" in types
        assert "arxiv" in types
        assert "code_output" in types

    @pytest.mark.asyncio
    async def test_llm_fallback(self, sample_plan, sample_results):
        """LLM 不可用时回退模板。"""
        from horizonrl.llm.client import LLMClient
        from horizonrl.config.settings import LLMConfig
        config = LLMConfig(provider="openai", model="test", api_key="sk-test", base_url="https://test.local")
        w = UserAnswerWriter(llm_client=LLMClient(config))
        report = await w.write("测试", sample_plan, sample_results)
        assert len(report) > 50
        assert "测试" in report


# ─── Writer 主编排 ──────────────────────────────────────────────────────────


class TestWriterOrchestrator:
    def test_synthesize_returns_report(self, sample_plan, sample_results, sample_verifications):
        writer = Writer(mode="template")
        report = writer.synthesize("Transformer", sample_plan, sample_results, sample_verifications)
        assert len(report) > 50
        assert "Transformer" in report

    def test_synthesize_empty_results(self):
        writer = Writer()
        report = writer.synthesize("测试", PlanGraph(), {}, {})
        assert len(report) > 0

    def test_synthesize_no_plan(self):
        writer = Writer()
        report = writer.synthesize("测试")
        assert len(report) > 0

    @pytest.mark.asyncio
    async def test_write_reports_dual_output(self, sample_plan, sample_results,
                                              sample_verifications, tmp_path):
        writer = Writer(mode="template",
                        config=WriterConfig(export_dir=str(tmp_path)))
        final_path, debug_path = await writer.write_reports(
            "测试", "s1", sample_plan, sample_results, sample_verifications,
        )
        assert final_path.endswith("final_answer.md")
        assert debug_path.endswith("debug_report.md")
        # 验证文件存在且有内容
        import pathlib
        assert pathlib.Path(final_path).exists()
        assert pathlib.Path(debug_path).exists()
        final_text = pathlib.Path(final_path).read_text(encoding="utf-8")
        debug_text = pathlib.Path(debug_path).read_text(encoding="utf-8")
        assert len(final_text) > 50
        assert len(debug_text) > 50

    def test_render_debug_report(self, sample_plan, sample_results, sample_verifications):
        writer = Writer()
        report = writer.render_debug_report("测试", sample_plan, sample_results, sample_verifications)
        assert len(report) > 50

    @pytest.mark.asyncio
    async def test_synthesize_async_template(self, sample_plan, sample_results, sample_verifications):
        writer = Writer(mode="template")
        report = await writer.synthesize_async("测试", sample_plan, sample_results, sample_verifications)
        assert len(report) > 50

    @pytest.mark.asyncio
    async def test_synthesize_async_llm_fallback(self, sample_plan, sample_results, sample_verifications):
        from horizonrl.llm.client import LLMClient
        from horizonrl.config.settings import LLMConfig
        config = LLMConfig(provider="openai", model="test", api_key="sk-test", base_url="https://test.local")
        writer = Writer(mode="llm", llm_client=LLMClient(config))
        report = await writer.synthesize_async("测试", sample_plan, sample_results, sample_verifications)
        assert len(report) > 50


# ─── Metadata ───────────────────────────────────────────────────────────────


class TestMetadata:
    def test_no_placeholder_name(self):
        """metadata 不得包含 [您的姓名/代号] 或固定 2023 日期。"""
        w = UserAnswerWriter()
        mock_ev = [
            {"type": "web", "content": "测试", "is_mock": True},
        ]
        report = w._template_write("测试", mock_ev)
        assert "您的姓名" not in report
        assert "代号" not in report
        assert "2023年10月27日" not in report


# ─── Search Provenance ──────────────────────────────────────────────────────


class TestSearchProvenance:
    def test_evidence_with_provenance(self):
        p = SearchProvenance(provider="brave", query="test query", is_mock=False,
                             url="https://example.com", timestamp=1715000000.0)
        ev = EvidenceItem(content="结果", provenance=p)
        assert ev.provenance.provider == "brave"
        assert ev.provenance_text() is not None

    def test_evidence_is_mock_flag(self):
        ev = EvidenceItem(content="mock result", is_mock=True)
        assert ev.is_mock is True
        assert "Mock" in ev.provenance_text()
