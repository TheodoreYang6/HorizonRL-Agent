"""Test Verifier — rule engine, LLM mode, hybrid mode, error classification."""

from __future__ import annotations

import pytest

from horizonrl.agent.verifier import RuleEngine, Verifier
from horizonrl.schemas.result import (
    ErrorType,
    EvidenceItem,
    StepResult,
    ToolCall,
    VerificationResult,
)
from horizonrl.schemas.task import TaskSpec

# ─── Fixtures ───────────────────────────────────────────────────────────


def _make_good_result() -> StepResult:
    return StepResult(
        task_id="t1",
        success=True,
        output="Transformer 多头注意力机制通过并行计算多个注意力头...",
        evidence=[
            EvidenceItem(content="论文A: 提出Multi-Head Attention机制", source="arxiv://1234", source_type="arxiv", retrieved_at=1000),
            EvidenceItem(content="论文B: 分析了头数对性能的影响", source="arxiv://5678", source_type="arxiv", retrieved_at=1001),
        ],
        tool_calls=[
            ToolCall(tool_name="arxiv_search", input={"q": "attention"}, output='[{"title":"A"}]', elapsed=0.5),
            ToolCall(tool_name="web_search", input={"q": "transformer"}, output='[{"title":"B"}]', elapsed=0.3),
        ],
        elapsed=1.0,
        worker_id="w1",
    )


def _make_empty_result() -> StepResult:
    return StepResult(
        task_id="t2",
        success=True,
        output="",
        evidence=[],
        tool_calls=[ToolCall(tool_name="web_search", input={"q": "x"}, output="", error="timeout")],
        worker_id="w2",
    )


def _make_code_error_result() -> StepResult:
    return StepResult(
        task_id="t3",
        success=False,
        output="Traceback (most recent call last):\n  File '<stdin>', line 1\nSyntaxError: invalid syntax",
        evidence=[],
        tool_calls=[ToolCall(tool_name="code_execution", input={"code": "x"}, output="SyntaxError", error="exec failed")],
        worker_id="w3",
    )


def _make_tool_error_result() -> StepResult:
    return StepResult(
        task_id="t4",
        success=False,
        output="[timeout] 超时 (20s)，已重试 2 次",
        evidence=[],
        tool_calls=[ToolCall(tool_name="web_search", input={"q": "x"}, output="", error="timeout")],
        error="工具调用失败",
        worker_id="w4",
    )


# ─── Rule Engine ────────────────────────────────────────────────────────


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_good_result_passes(self):
        check = self.engine.check(_make_good_result(), "研究注意力机制")
        assert check["pass"] is True
        assert check["score"] >= 0.7
        assert check["error_type"] == ErrorType.NONE

    def test_empty_result_fails(self):
        check = self.engine.check(_make_empty_result(), "搜索最新进展")
        assert check["pass"] is False
        assert check["error_type"] == ErrorType.EMPTY_RESULT
        assert len(check["suggested_actions"]) > 0

    def test_code_error_detected(self):
        check = self.engine.check(_make_code_error_result(), "运行测试代码")
        assert check["pass"] is False
        assert check["error_type"] == ErrorType.CODE_ERROR
        assert "代码执行" in check["feedback"]

    def test_tool_error_detected(self):
        check = self.engine.check(_make_tool_error_result(), "搜索信息")
        assert check["pass"] is False
        assert check["error_type"] == ErrorType.TOOL_ERROR
        assert "timeout" in check["feedback"].lower()

    def test_no_evidence_incomplete(self):
        result = StepResult(
            task_id="t5", success=True, output="完成", evidence=[], tool_calls=[], worker_id="w5",
        )
        check = self.engine.check(result, "分析数据")
        assert check["pass"] is False
        assert check["error_type"] == ErrorType.INCOMPLETE

    def test_all_tools_fail(self):
        # 输出非空，确保走"全部工具失败"分支而非空输出分支
        result = StepResult(
            task_id="t6", success=False, output="尝试调用工具但全部失败",
            evidence=[],
            tool_calls=[
                ToolCall(tool_name="web_search", input={}, output="", error="网络错误"),
                ToolCall(tool_name="arxiv_search", input={}, output="", error="超时"),
            ],
            worker_id="w6",
        )
        check = self.engine.check(result, "搜索")
        assert check["pass"] is False
        assert check["error_type"] == ErrorType.TOOL_ERROR
        assert check["score"] < 0.5

    def test_duckduckgo_unavailable_detected(self):
        result = StepResult(
            task_id="t7", success=True,
            output='[{"title":"DuckDuckGo unavailable","url":"","snippet":"Install duckduckgo-search package"}]',
            evidence=[],
            tool_calls=[ToolCall(tool_name="web_search", input={}, output="DuckDuckGo unavailable", error="")],
            worker_id="w7",
        )
        check = self.engine.check(result, "搜索")
        assert check["pass"] is False
        assert check["error_type"] == ErrorType.EMPTY_RESULT

    def test_circuit_open_detected(self):
        result = StepResult(
            task_id="t8", success=True,
            output="[circuit_open] 熔断器开启，拒绝调用",
            evidence=[],
            tool_calls=[ToolCall(tool_name="web_search", input={}, output="", error="熔断")],
            worker_id="w8",
        )
        check = self.engine.check(result, "搜索")
        assert check["error_type"] == ErrorType.TOOL_ERROR

    def test_score_based_on_evidence_count(self):
        r1 = _make_good_result()  # 2 evidence
        # 需要足够长的输出避免触发"输出过短"规则
        r2 = StepResult(task_id="x", success=True,
                        output="这是一个足够长的输出内容用来通过输出长度检查规则确保不会被误判为短输出",
                        evidence=[EvidenceItem(content="x", source="s", source_type="web")] * 5,
                        tool_calls=[ToolCall(tool_name="w", input={}, output="ok")], worker_id="w")
        c1 = self.engine.check(r1, "test")
        c2 = self.engine.check(r2, "test")
        assert c2["score"] > c1["score"]  # more evidence = higher score

    def test_short_output_with_evidence_passes(self):
        result = StepResult(
            task_id="t9", success=True, output="OK",
            evidence=[EvidenceItem(content="valid data", source="s", source_type="web")],
            tool_calls=[ToolCall(tool_name="w", input={}, output="ok")],
            worker_id="w9",
        )
        check = self.engine.check(result, "test")
        assert check["pass"] is True


# ─── Full Verifier ──────────────────────────────────────────────────────


class TestVerifierRuleMode:
    def setup_method(self):
        self.verifier = Verifier(mode="rule")

    @pytest.mark.asyncio
    async def test_verify_good_result(self):
        result = await self.verifier.verify(_make_good_result(), "研究注意力机制")
        assert isinstance(result, VerificationResult)
        assert result.pass_ is True
        assert result.score >= 0.7
        assert result.error_type == ErrorType.NONE

    @pytest.mark.asyncio
    async def test_verify_empty_result(self):
        result = await self.verifier.verify(_make_empty_result(), "搜索")
        assert result.pass_ is False
        assert result.error_type == ErrorType.EMPTY_RESULT
        assert len(result.suggested_actions) > 0

    @pytest.mark.asyncio
    async def test_verify_code_error(self):
        result = await self.verifier.verify(_make_code_error_result(), "运行代码")
        assert result.pass_ is False
        assert result.error_type == ErrorType.CODE_ERROR

    @pytest.mark.asyncio
    async def test_verify_with_task_spec(self):
        task = TaskSpec(id="t1", name="研究", description="研究 Transformer 注意力机制",
                        tool_names=["arxiv_search"])
        result = await self.verifier.verify(_make_good_result(), task)
        assert result.pass_ is True

    @pytest.mark.asyncio
    async def test_verify_with_string_task(self):
        result = await self.verifier.verify(_make_good_result(), "研究注意力机制")
        assert result.pass_ is True

    def test_verify_sync(self):
        result = self.verifier.verify_sync(_make_good_result(), "研究注意力")
        assert result.pass_ is True


class TestVerifierHybridMode:
    """Hybrid 模式下 RuleEngine 先跑，LLM 复核中间地带。"""

    @pytest.mark.asyncio
    async def test_hybrid_mode_accepts_good(self):
        # Without LLM client, hybrid falls back to rule
        verifier = Verifier(mode="hybrid", llm_client=None)
        result = await verifier.verify(_make_good_result(), "研究")
        assert result.pass_ is True

    @pytest.mark.asyncio
    async def test_hybrid_mode_rejects_bad(self):
        verifier = Verifier(mode="hybrid", llm_client=None)
        result = await verifier.verify(_make_empty_result(), "研究")
        assert result.pass_ is False

    @pytest.mark.asyncio
    async def test_hybrid_requires_llm_for_llm_mode(self):
        with pytest.raises(ValueError):
            Verifier(mode="llm", llm_client=None)


# ─── ErrorType Mapping to Replanner Actions ─────────────────────────────


class TestErrorTypeReplanMapping:
    """确保每种 ErrorType 都给出合理的 suggested_actions 供 Replanner 使用。"""

    def test_empty_result_suggests_rewrite(self):
        engine = RuleEngine()
        check = engine.check(_make_empty_result(), "搜索最新进展")
        actions = " ".join(check["suggested_actions"])
        assert any(w in actions for w in ["改写", "搜索", "备用", "重试"])

    def test_code_error_suggests_fix(self):
        engine = RuleEngine()
        check = engine.check(_make_code_error_result(), "运行测试")
        actions = " ".join(check["suggested_actions"])
        assert any(w in actions for w in ["修正", "代码", "检查", "语法"])

    def test_tool_error_suggests_retry_or_switch(self):
        engine = RuleEngine()
        check = engine.check(_make_tool_error_result(), "搜索")
        actions = " ".join(check["suggested_actions"])
        assert any(w in actions for w in ["重试", "备用", "切换"])

    def test_incomplete_suggests_retry(self):
        result = StepResult(task_id="t", success=True, output="完成", evidence=[], tool_calls=[], worker_id="w")
        engine = RuleEngine()
        check = engine.check(result, "分析数据")
        assert check["error_type"] == ErrorType.INCOMPLETE
        assert len(check["evidence_gaps"]) > 0

    def test_verification_result_is_serializable(self):
        """确保 VerificationResult 可以序列化（供 checkpoint/日志使用）。"""
        import dataclasses
        import json
        vr = VerificationResult(pass_=True, score=0.8, error_type=ErrorType.NONE,
                                 feedback="OK", evidence_gaps=[], suggested_actions=[])
        d = dataclasses.asdict(vr)
        json_str = json.dumps(d, default=str)
        assert "OK" in json_str
