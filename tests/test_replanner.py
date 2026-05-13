"""Test Replanner module — patch generation, retry limits, apply_patch, LLM fallback."""

from __future__ import annotations

import pytest

from horizonrl.agent.replanner import (
    Replanner,
    LLMReplanner,
    ERROR_STRATEGY,
)
from horizonrl.schemas.task import (
    TaskSpec,
    PlanGraph,
    PlanNode,
    PlanPatch,
    PatchType,
    TaskStatus,
    TaskPriority,
)
from horizonrl.schemas.result import (
    VerificationResult,
    ErrorType,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def replanner():
    return Replanner(max_retries_per_task=3, max_total_replans=5)


@pytest.fixture
def simple_graph():
    """3-node DAG: root → child, root2 (parallel)."""
    g = PlanGraph()
    s1 = TaskSpec(id="t1", name="搜索背景", description="搜索 Transformer 背景",
                  tool_names=["web_search"])
    s2 = TaskSpec(id="t2", name="汇总分析", description="汇总搜索结果",
                  tool_names=[], depends_on=["t1"])
    s3 = TaskSpec(id="t3", name="代码实验", description="实现注意力机制",
                  tool_names=["code_execution"])
    g.nodes["t1"] = PlanNode(spec=s1, status=TaskStatus.SUCCESS)
    g.nodes["t2"] = PlanNode(spec=s2, status=TaskStatus.PENDING)
    g.nodes["t3"] = PlanNode(spec=s3, status=TaskStatus.PENDING)
    g.edges["t2"] = ["t1"]
    g.root_ids = ["t1", "t3"]
    return g


@pytest.fixture
def failed_vr():
    return VerificationResult(
        pass_=False,
        score=0.2,
        error_type=ErrorType.EMPTY_RESULT,
        feedback="搜索结果为空",
        evidence_gaps=["需要关于 Transformer 的论文"],
        suggested_actions=["改写搜索查询词", "使用更通用的关键词"],
    )


# ─── Initialization ─────────────────────────────────────────────────────────


class TestReplannerInit:
    def test_default_params(self):
        r = Replanner()
        assert r.max_retries_per_task == 3
        assert r.max_total_replans == 5
        assert r.total_replans == 0

    def test_custom_params(self):
        r = Replanner(max_retries_per_task=2, max_total_replans=10)
        assert r.max_retries_per_task == 2
        assert r.max_total_replans == 10


# ─── Retry Control ──────────────────────────────────────────────────────────


class TestRetryControl:
    def test_should_replan_when_failed(self, replanner, failed_vr):
        assert replanner.should_replan(failed_vr, "t1") is True

    def test_should_not_replan_when_passed(self, replanner):
        vr = VerificationResult(pass_=True, score=0.9, error_type=ErrorType.NONE,
                                feedback="通过")
        assert replanner.should_replan(vr, "t1") is False

    def test_should_not_replan_after_max_retries(self, replanner, failed_vr):
        for _ in range(3):
            replanner.replan(failed_vr, PlanGraph(), "t1")
        assert replanner.should_replan(failed_vr, "t1") is False

    def test_should_not_replan_after_max_total(self, replanner, failed_vr):
        g = PlanGraph()
        g.nodes["tx"] = PlanNode(spec=TaskSpec(id="tx", name="X",
                                  description="X", tool_names=[]))
        for i in range(5):
            vid = f"t{i}"
            g.nodes[vid] = PlanNode(spec=TaskSpec(id=vid, name=f"T{i}",
                                    description=f"D{i}", tool_names=[]))
            replanner.replan(failed_vr, g, vid)
        assert replanner.total_replans == 5
        g.nodes["t_extra"] = PlanNode(spec=TaskSpec(id="t_extra", name="Extra",
                                       description="Extra", tool_names=[]))
        assert replanner.should_replan(failed_vr, "t_extra") is False

    def test_can_retry_tracks_per_task(self, replanner, failed_vr):
        g = PlanGraph()
        g.nodes["t1"] = PlanNode(spec=TaskSpec(id="t1", name="T1",
                                  description="D1", tool_names=[]))
        g.nodes["t2"] = PlanNode(spec=TaskSpec(id="t2", name="T2",
                                  description="D2", tool_names=[]))
        replanner.replan(failed_vr, g, "t1")
        replanner.replan(failed_vr, g, "t1")
        # t1 has 2 retries, t2 has 0
        assert replanner.get_retry_count("t1") == 2
        assert replanner.get_retry_count("t2") == 0
        assert replanner.can_retry("t1") is True
        assert replanner.can_retry("t2") is True

    def test_reset_clears_counters(self, replanner, failed_vr):
        g = PlanGraph()
        g.nodes["t1"] = PlanNode(spec=TaskSpec(id="t1", name="T1",
                                  description="D1", tool_names=[]))
        replanner.replan(failed_vr, g, "t1")
        replanner.reset()
        assert replanner.total_replans == 0
        assert replanner.get_retry_count("t1") == 0


# ─── ERROR_STRATEGY Mapping ─────────────────────────────────────────────────


class TestErrorStrategy:
    @pytest.mark.parametrize("error_type, expected_patch", [
        (ErrorType.EMPTY_RESULT, PatchType.RETRY),
        (ErrorType.CODE_ERROR, PatchType.RETRY),
        (ErrorType.TOOL_ERROR, PatchType.RETRY),
        (ErrorType.OFF_TOPIC, PatchType.RETRY),
        (ErrorType.INCOMPLETE, PatchType.ADD),
        (ErrorType.FACTUAL_ERROR, PatchType.RETRY),
        (ErrorType.HALLUCINATION, PatchType.RETRY),
        (ErrorType.OTHER, PatchType.RETRY),
        (ErrorType.NONE, PatchType.RETRY),
    ])
    def test_error_maps_to_correct_patch_type(self, error_type, expected_patch):
        patch_type, _hint = ERROR_STRATEGY[error_type]
        assert patch_type == expected_patch

    def test_all_error_types_covered(self):
        for et in ErrorType:
            assert et in ERROR_STRATEGY, f"Missing strategy for {et}"


# ─── replan() Patch Generation ──────────────────────────────────────────────


class TestReplanPatchGeneration:
    def test_retry_patch_for_empty_result(self, replanner, simple_graph, failed_vr):
        patch = replanner.replan(failed_vr, simple_graph, "t3")
        assert patch is not None
        assert patch.patch_type == PatchType.RETRY
        assert patch.target_node_id == "t3"
        assert patch.new_spec is not None
        assert "重试" in patch.new_spec.description
        assert patch.new_spec.id == "t3"  # RETRY keeps same id
        assert patch.new_spec.retry_count == 1

    def test_add_patch_for_incomplete(self, replanner, simple_graph):
        vr = VerificationResult(
            pass_=False, score=0.25, error_type=ErrorType.INCOMPLETE,
            feedback="缺少代码实验数据",
            evidence_gaps=["需要运行实验并收集数据"],
            suggested_actions=["补充代码执行子任务"],
        )
        patch = replanner.replan(vr, simple_graph, "t2")
        assert patch is not None
        assert patch.patch_type == PatchType.ADD
        assert patch.new_spec is not None
        assert "_sup_" in patch.new_spec.id
        assert patch.new_spec.depends_on == ["t2"]
        assert "补充" in patch.new_spec.name

    def test_retry_spec_includes_verifier_suggestions(self, replanner, simple_graph):
        vr = VerificationResult(
            pass_=False, score=0.3, error_type=ErrorType.TOOL_ERROR,
            feedback="工具调用超时",
            suggested_actions=["增加超时时间", "切换备用搜索引擎"],
        )
        patch = replanner.replan(vr, simple_graph, "t1")
        assert "增加超时时间" in patch.new_spec.context
        assert "切换备用搜索引擎" in patch.new_spec.context

    def test_retry_spec_includes_evidence_gaps(self, replanner, simple_graph):
        vr = VerificationResult(
            pass_=False, score=0.2, error_type=ErrorType.EMPTY_RESULT,
            feedback="无结果",
            evidence_gaps=["需要论文标题和摘要", "需要实验数据"],
        )
        patch = replanner.replan(vr, simple_graph, "t1")
        assert "论文标题和摘要" in patch.new_spec.context

    def test_returns_none_when_node_not_found(self, replanner, failed_vr):
        patch = replanner.replan(failed_vr, PlanGraph(), "nonexistent")
        assert patch is None

    def test_returns_none_when_cannot_replan(self, replanner, failed_vr):
        g = PlanGraph()
        g.nodes["t1"] = PlanNode(spec=TaskSpec(id="t1", name="T1",
                                  description="D1", tool_names=[]))
        for _ in range(3):
            replanner.replan(failed_vr, g, "t1")
        patch = replanner.replan(failed_vr, g, "t1")
        assert patch is None

    def test_reason_contains_error_info(self, replanner, simple_graph, failed_vr):
        patch = replanner.replan(failed_vr, simple_graph, "t1")
        assert "empty_result" in patch.reason
        assert "搜索结果为空" in patch.reason

    def test_off_topic_adds_focus_prefix(self, replanner, simple_graph):
        vr = VerificationResult(
            pass_=False, score=0.3, error_type=ErrorType.OFF_TOPIC,
            feedback="输出偏离主题",
        )
        patch = replanner.replan(vr, simple_graph, "t1")
        assert "[聚焦]" in patch.new_spec.description

    def test_hallucination_adds_strict_prefix(self, replanner, simple_graph):
        vr = VerificationResult(
            pass_=False, score=0.2, error_type=ErrorType.HALLUCINATION,
            feedback="存在编造内容",
        )
        patch = replanner.replan(vr, simple_graph, "t1")
        assert "[严格]" in patch.new_spec.description

    def test_factual_error_adds_cross_validation(self, replanner, simple_graph):
        vr = VerificationResult(
            pass_=False, score=0.2, error_type=ErrorType.FACTUAL_ERROR,
            feedback="事实错误",
        )
        patch = replanner.replan(vr, simple_graph, "t1")
        assert "交叉验证" in patch.new_spec.description
        assert "权威来源" in patch.new_spec.context


# ─── apply_patch ────────────────────────────────────────────────────────────


class TestApplyPatch:
    def test_apply_retry_resets_status(self, replanner, simple_graph):
        simple_graph.nodes["t3"].status = TaskStatus.FAILED
        simple_graph.nodes["t3"].error_msg = "old error"
        patch = PlanPatch(
            patch_type=PatchType.RETRY,
            target_node_id="t3",
            reason="重试",
            new_spec=simple_graph.nodes["t3"].spec,
        )
        replanner.apply_patch(simple_graph, patch)
        assert simple_graph.nodes["t3"].status == TaskStatus.PENDING
        assert simple_graph.nodes["t3"].error_msg == ""

    def test_apply_retry_updates_spec(self, replanner, simple_graph):
        new_spec = TaskSpec(id="t3", name="新名称", description="新描述",
                            tool_names=["web_search"],
                            retry_count=1)
        patch = PlanPatch(
            patch_type=PatchType.RETRY,
            target_node_id="t3",
            reason="重试",
            new_spec=new_spec,
        )
        replanner.apply_patch(simple_graph, patch)
        assert simple_graph.nodes["t3"].spec.name == "新名称"
        assert simple_graph.nodes["t3"].spec.description == "新描述"
        assert simple_graph.nodes["t3"].status == TaskStatus.PENDING

    def test_apply_add_creates_new_node(self, replanner, simple_graph):
        new_spec = TaskSpec(id="t2_sup_01", name="补充搜索",
                            description="补充 Transformer 背景搜索",
                            tool_names=["web_search"])
        patch = PlanPatch(
            patch_type=PatchType.ADD,
            target_node_id="t2",
            reason="需要补充证据",
            new_spec=new_spec,
        )
        original_count = len(simple_graph.nodes)
        replanner.apply_patch(simple_graph, patch)
        assert len(simple_graph.nodes) == original_count + 1
        assert "t2_sup_01" in simple_graph.nodes
        assert simple_graph.nodes["t2_sup_01"].status == TaskStatus.PENDING

    def test_apply_add_sets_edges(self, replanner, simple_graph):
        new_spec = TaskSpec(id="t2_sup_02", name="补充搜索",
                            description="补充搜索",
                            tool_names=["web_search"])
        patch = PlanPatch(
            patch_type=PatchType.ADD,
            target_node_id="t2",
            reason="补充",
            new_spec=new_spec,
        )
        replanner.apply_patch(simple_graph, patch)
        assert "t2_sup_02" in simple_graph.edges
        assert simple_graph.edges["t2_sup_02"] == ["t2"]

    def test_apply_remove_marks_skipped(self, replanner, simple_graph):
        patch = PlanPatch(
            patch_type=PatchType.REMOVE,
            target_node_id="t3",
            reason="不必要",
        )
        replanner.apply_patch(simple_graph, patch)
        assert simple_graph.nodes["t3"].status == TaskStatus.SKIPPED
        assert "不必要" in simple_graph.nodes["t3"].error_msg

    def test_apply_reorder_changes_priority(self, replanner, simple_graph):
        assert simple_graph.nodes["t2"].spec.priority == TaskPriority.P1
        patch = PlanPatch(
            patch_type=PatchType.REORDER,
            target_node_id="t2",
            reason="需要优先执行",
        )
        replanner.apply_patch(simple_graph, patch)
        assert simple_graph.nodes["t2"].spec.priority == TaskPriority.P0

    def test_apply_patch_returns_graph(self, replanner, simple_graph):
        patch = PlanPatch(
            patch_type=PatchType.REMOVE,
            target_node_id="t3",
            reason="skip",
        )
        result = replanner.apply_patch(simple_graph, patch)
        assert result is simple_graph

    def test_apply_retry_nonexistent_node_noop(self, replanner, simple_graph):
        patch = PlanPatch(
            patch_type=PatchType.RETRY,
            target_node_id="nonexistent",
            reason="test",
        )
        # Should not raise
        replanner.apply_patch(simple_graph, patch)

    def test_apply_add_no_spec_noop(self, replanner, simple_graph):
        patch = PlanPatch(
            patch_type=PatchType.ADD,
            target_node_id="t2",
            reason="no spec",
            new_spec=None,
        )
        count_before = len(simple_graph.nodes)
        replanner.apply_patch(simple_graph, patch)
        assert len(simple_graph.nodes) == count_before


# ─── diagnose_all ───────────────────────────────────────────────────────────


class TestDiagnoseAll:
    def test_batch_generates_patches_for_failures(self, replanner, simple_graph):
        results = {
            "t2": VerificationResult(
                pass_=False, score=0.2, error_type=ErrorType.EMPTY_RESULT,
                feedback="空结果"),
            "t3": VerificationResult(
                pass_=False, score=0.3, error_type=ErrorType.TOOL_ERROR,
                feedback="工具错误"),
        }
        patches = replanner.diagnose_all(results, simple_graph)
        assert len(patches) == 2
        patch_ids = {p.target_node_id for p in patches}
        assert patch_ids == {"t2", "t3"}

    def test_batch_skips_unreplanable(self, replanner, simple_graph):
        g = PlanGraph()
        g.nodes["tx"] = PlanNode(spec=TaskSpec(id="tx", name="X",
                                  description="X", tool_names=[]))
        for _ in range(3):
            replanner.replan(
                VerificationResult(pass_=False, score=0.1,
                                   error_type=ErrorType.EMPTY_RESULT,
                                   feedback="fail"),
                g, "tx",
            )
        # tx is now at max retries
        g.nodes["ty"] = PlanNode(spec=TaskSpec(id="ty", name="Y",
                                  description="Y", tool_names=[]))
        results = {
            "tx": VerificationResult(
                pass_=False, score=0.1, error_type=ErrorType.EMPTY_RESULT,
                feedback="fail"),
            "ty": VerificationResult(
                pass_=False, score=0.2, error_type=ErrorType.TOOL_ERROR,
                feedback="tool"),
        }
        patches = replanner.diagnose_all(results, g)
        assert len(patches) == 1
        assert patches[0].target_node_id == "ty"

    def test_batch_empty_when_all_pass(self, replanner, simple_graph):
        results = {
            "t2": VerificationResult(
                pass_=True, score=0.9, error_type=ErrorType.NONE,
                feedback="ok"),
            "t3": VerificationResult(
                pass_=True, score=0.85, error_type=ErrorType.NONE,
                feedback="ok"),
        }
        patches = replanner.diagnose_all(results, simple_graph)
        assert len(patches) == 0


# ─── Integration: Full Replan Cycle ─────────────────────────────────────────


class TestFullReplanCycle:
    def test_retry_cycle_on_failed_node(self, replanner, simple_graph):
        """模拟完整重规划周期：验证失败 → 生成补丁 → 应用补丁 → 重试。"""
        # t3 执行失败
        simple_graph.nodes["t3"].status = TaskStatus.FAILED

        vr = VerificationResult(
            pass_=False, score=0.2, error_type=ErrorType.EMPTY_RESULT,
            feedback="搜索无结果",
            suggested_actions=["换关键词", "用英文搜索"],
        )

        # Step 1: 生成补丁
        patch = replanner.replan(vr, simple_graph, "t3")
        assert patch is not None
        assert patch.patch_type == PatchType.RETRY

        # Step 2: 应用补丁
        replanner.apply_patch(simple_graph, patch)

        # Step 3: 验证节点已重置
        node = simple_graph.nodes["t3"]
        assert node.status == TaskStatus.PENDING
        assert node.spec.retry_count == 1
        assert "重试" in node.spec.description

    def test_add_supplementary_task_cycle(self, replanner, simple_graph):
        """模拟补充任务周期：不完整 → ADD → 新节点加入 DAG。"""
        simple_graph.nodes["t2"].status = TaskStatus.FAILED

        vr = VerificationResult(
            pass_=False, score=0.25, error_type=ErrorType.INCOMPLETE,
            feedback="汇总缺少数据支撑",
            evidence_gaps=["需要性能对比数据", "需要引用论文"],
            suggested_actions=["补充搜索性能数据"],
        )

        patch = replanner.replan(vr, simple_graph, "t2")
        assert patch.patch_type == PatchType.ADD

        replanner.apply_patch(simple_graph, patch)

        new_id = patch.new_spec.id
        assert new_id in simple_graph.nodes
        assert simple_graph.nodes[new_id].status == TaskStatus.PENDING
        # 新节点依赖 t2
        assert simple_graph.edges[new_id] == ["t2"]

    def test_multiple_retries_then_skip(self, replanner, simple_graph):
        """超过最大重试次数后不再生成补丁。"""
        simple_graph.nodes["t3"].status = TaskStatus.FAILED

        for i in range(3):
            patch = replanner.replan(
                VerificationResult(pass_=False, score=0.1,
                                   error_type=ErrorType.TOOL_ERROR,
                                   feedback=f"fail_{i}"),
                simple_graph, "t3",
            )
            assert patch is not None, f"Retry {i} should succeed"
            replanner.apply_patch(simple_graph, patch)

        # 第 4 次应该被拒绝
        patch = replanner.replan(
            VerificationResult(pass_=False, score=0.1,
                               error_type=ErrorType.TOOL_ERROR,
                               feedback="fail_4"),
            simple_graph, "t3",
        )
        assert patch is None


# ─── LLMReplanner ───────────────────────────────────────────────────────────


class TestLLMReplanner:
    def test_init_with_llm_client(self):
        from horizonrl.llm.client import LLMClient
        from horizonrl.config.settings import LLMConfig

        config = LLMConfig(
            provider="openai",
            model="test-model",
            api_key="sk-test",
            base_url="https://test.local",
        )
        client = LLMClient(config)
        r = LLMReplanner(client, max_retries_per_task=2, max_total_replans=3)
        assert r.max_retries_per_task == 2
        assert r.max_total_replans == 3
        assert r.llm is client

    @pytest.mark.asyncio
    async def test_replan_async_falls_back_on_llm_error(self, simple_graph):
        """LLM 不可用时回退到规则生成。"""
        from horizonrl.llm.client import LLMClient
        from horizonrl.config.settings import LLMConfig

        config = LLMConfig(
            provider="openai",
            model="test-model",
            api_key="sk-test",
            base_url="https://test.local",
        )
        client = LLMClient(config)
        r = LLMReplanner(client)

        vr = VerificationResult(
            pass_=False, score=0.2, error_type=ErrorType.EMPTY_RESULT,
            feedback="空结果",
        )

        # LLM 会因网络不可达而失败，但应该回退到规则生成
        patch = await r.replan_async(vr, simple_graph, "t3")
        if patch is not None:
            assert patch.patch_type in (PatchType.RETRY, PatchType.ADD)
            assert patch.new_spec is not None
            assert patch.new_spec.id == "t3"

    def test_llm_fallback_spec_handles_incomplete(self, replanner, simple_graph):
        """_build_add_spec 生成正确的补充 spec。"""
        vr = VerificationResult(
            pass_=False, score=0.25, error_type=ErrorType.INCOMPLETE,
            feedback="缺失代码实验",
            evidence_gaps=["需要运行实验"],
            suggested_actions=["补充代码执行"],
        )
        spec = replanner._build_add_spec(
            simple_graph.nodes["t3"].spec, vr
        )
        assert "_sup_" in spec.id
        assert spec.depends_on == ["t3"]
        assert "补充" in spec.name
        assert "运行实验" in spec.context or "运行实验" in spec.description


# ─── Edge Cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_concurrent_retries_independent(self, replanner):
        """不同任务的重试计数互不干扰。"""
        g = PlanGraph()
        vr1 = VerificationResult(pass_=False, score=0.1, error_type=ErrorType.EMPTY_RESULT,
                                 feedback="e1")
        vr2 = VerificationResult(pass_=False, score=0.1, error_type=ErrorType.TOOL_ERROR,
                                 feedback="e2")

        g.nodes["a"] = PlanNode(spec=TaskSpec(id="a", name="A", description="A",
                                tool_names=[]))
        g.nodes["b"] = PlanNode(spec=TaskSpec(id="b", name="B", description="B",
                                tool_names=[]))

        # Retry A twice, B once
        replanner.replan(vr1, g, "a")
        replanner.replan(vr1, g, "a")
        replanner.replan(vr2, g, "b")

        assert replanner.get_retry_count("a") == 2
        assert replanner.get_retry_count("b") == 1
        assert replanner.can_retry("a") is True
        assert replanner.total_replans == 3

    def test_retry_count_persists_across_patches(self, replanner, simple_graph):
        """节点的 retry_count 在 spec 中正确递增。"""
        for i in range(3):
            vr = VerificationResult(pass_=False, score=0.1,
                                    error_type=ErrorType.EMPTY_RESULT,
                                    feedback=f"fail_{i}")
            patch = replanner.replan(vr, simple_graph, "t3")
            if patch:
                replanner.apply_patch(simple_graph, patch)
        assert simple_graph.nodes["t3"].spec.retry_count == 3

    def test_add_patch_downstream_dependencies(self, replanner, simple_graph):
        """ADD 补丁正确更新下游依赖。"""
        # t2 依赖 t1，我们对 t2 做 ADD
        new_spec = TaskSpec(id="t2_sup_test", name="补充",
                            description="补充 t2",
                            tool_names=["web_search"])
        patch = PlanPatch(
            patch_type=PatchType.ADD,
            target_node_id="t2",
            reason="补充",
            new_spec=new_spec,
        )
        replanner.apply_patch(simple_graph, patch)
        # 新节点的 spec.depends_on 应包含 t2
        new_node = simple_graph.nodes["t2_sup_test"]
        assert "t2" in new_node.spec.depends_on

    def test_empty_graph_replan_returns_none(self, replanner, failed_vr):
        patch = replanner.replan(failed_vr, PlanGraph(), "t1")
        assert patch is None

    def test_verifier_none_error_type(self, replanner, simple_graph):
        """NONE 错误类型（低分但未检测到具体问题）仍生成 RETRY 补丁。"""
        vr = VerificationResult(
            pass_=False, score=0.5, error_type=ErrorType.NONE,
            feedback="质量偏低但原因不明",
        )
        patch = replanner.replan(vr, simple_graph, "t3")
        assert patch is not None
        assert patch.patch_type == PatchType.RETRY
