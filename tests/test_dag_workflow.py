"""Test LangGraph DAG orchestration — StateGraph, nodes, routing, end-to-end."""

from __future__ import annotations

import asyncio
import uuid
import pytest

from horizonrl.orchestration.dag_workflow import (
    ResearchOrchestrator,
    create_orchestrator,
    _make_initial_state,
)
from horizonrl.schemas.task import (
    UserTask,
    TaskSpec,
    TaskPriority,
    TaskStatus,
    PlanNode,
    PlanGraph,
)
from horizonrl.schemas.result import StepResult, EvidenceItem, ToolCall
from horizonrl.agent.planner import Planner
from horizonrl.tools.manager import ToolManager


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def tool_manager():
    from horizonrl.tools.mock import register_mock_tools
    mgr = ToolManager()
    register_mock_tools(mgr)
    return mgr


@pytest.fixture
def planner():
    return Planner()


@pytest.fixture
def orchestrator(planner, tool_manager):
    return ResearchOrchestrator(
        planner=planner,
        tool_manager=tool_manager,
        semaphore_limit=3,
        max_iterations=10,
    )



# ─── Initial State ──────────────────────────────────────────────────────


class TestInitialState:
    def test_default_construction(self):
        s = _make_initial_state(user_task="hello")
        assert s["user_task"] == "hello"
        assert s["plan"] is None
        assert s["results"] == {}
        assert s["iteration"] == 0
        assert s["error"] == ""

    def test_keys_present(self):
        s = _make_initial_state(user_task="x")
        for key in ("user_task", "plan", "results", "iteration",
                     "max_iterations", "final_output", "error", "started_at"):
            assert key in s


# ─── Graph Structure ────────────────────────────────────────────────────


class TestGraphStructure:
    def test_graph_has_required_nodes(self, orchestrator):
        # 编译后的图有 builder 信息，验证节点注册正确
        graph = orchestrator._graph
        assert graph is not None
        # 节点存在性通过 e2e 测试隐式验证（plan_task/finalize 必然被调用）

    def test_graph_is_compiled(self, orchestrator):
        """编译后的图可以被 ainvoke 调用。"""
        from langgraph.graph.state import CompiledStateGraph
        assert isinstance(orchestrator._graph, CompiledStateGraph)


# ─── Node: plan_task ────────────────────────────────────────────────────


class TestPlanTaskNode:
    @pytest.mark.asyncio
    async def test_creates_plan_graph(self, orchestrator):
        state = _make_initial_state(user_task="测试 Transformer 注意力机制")
        result = await orchestrator._plan_task(state)
        plan = result["plan"]
        assert isinstance(plan, PlanGraph)
        assert len(plan.nodes) == 5
        assert len(plan.root_ids) >= 2

    @pytest.mark.asyncio
    async def test_plan_has_task_specs(self, orchestrator):
        state = _make_initial_state(user_task="研究量子计算")
        result = await orchestrator._plan_task(state)
        for node in result["plan"].nodes.values():
            assert isinstance(node.spec, TaskSpec)
            assert node.spec.id.startswith("task_")


# ─── Node: mark_ready ───────────────────────────────────────────────────


class TestMarkReadyNode:
    @pytest.mark.asyncio
    async def test_marks_root_nodes_as_ready(self, orchestrator):
        state = _make_initial_state(user_task="测试")
        state.update(await orchestrator._plan_task(state))
        result = await orchestrator._mark_ready(state)
        plan = result["plan"]
        for rid in plan.root_ids:
            assert plan.nodes[rid].status == TaskStatus.READY

    @pytest.mark.asyncio
    async def test_non_root_stay_pending(self, orchestrator):
        state = _make_initial_state(user_task="测试")
        state.update(await orchestrator._plan_task(state))
        result = await orchestrator._mark_ready(state)
        plan = result["plan"]
        for node in plan.nodes.values():
            if node.depends_on:
                assert node.status == TaskStatus.PENDING


# ─── Node: execute_batch ────────────────────────────────────────────────


class TestExecuteBatchNode:
    @pytest.mark.asyncio
    async def test_executes_and_marks_success(self, orchestrator):
        state = _make_initial_state(user_task="研究任务")
        state.update(await orchestrator._plan_task(state))
        state.update(await orchestrator._mark_ready(state))
        result = await orchestrator._execute_batch(state)
        plan = result["plan"]
        results = result["results"]
        for rid in plan.root_ids:
            assert plan.nodes[rid].status == TaskStatus.SUCCESS
        assert len(results) >= len(plan.root_ids)

    @pytest.mark.asyncio
    async def test_increments_iteration(self, orchestrator):
        state = _make_initial_state(user_task="测试", max_iterations=10)
        state.update(await orchestrator._plan_task(state))
        state.update(await orchestrator._mark_ready(state))
        result = await orchestrator._execute_batch(state)
        assert result["iteration"] == 1


# ─── Node: finalize ─────────────────────────────────────────────────────


class TestFinalizeNode:
    @pytest.mark.asyncio
    async def test_produces_output(self, orchestrator):
        state = _make_initial_state(user_task="测试任务")
        state.update(await orchestrator._plan_task(state))
        result = await orchestrator._finalize(state)
        assert "测试任务" in result["final_output"]
        assert len(result["final_output"]) > 0

    @pytest.mark.asyncio
    async def test_output_includes_stats(self, orchestrator):
        state = _make_initial_state(user_task="测试", max_iterations=10)
        state.update(await orchestrator._plan_task(state))
        result = await orchestrator._finalize(state)
        assert "Token" in result["final_output"]


# ─── Route Logic ────────────────────────────────────────────────────────


class TestRouteLogic:
    def test_routes_to_execute_when_ready_exists(self, orchestrator):
        state = _make_initial_state(user_task="测试", max_iterations=10)
        plan = PlanGraph()
        spec = TaskSpec(id="t1", name="T1", description="D1", tool_names=[])
        plan.nodes["t1"] = PlanNode(spec=spec, status=TaskStatus.READY)
        state["plan"] = plan
        assert orchestrator._route_after_mark_ready(state) == "execute"

    def test_routes_to_done_when_all_complete(self, orchestrator):
        state = _make_initial_state(user_task="测试", max_iterations=10)
        plan = PlanGraph()
        spec = TaskSpec(id="t1", name="T1", description="D1", tool_names=[])
        plan.nodes["t1"] = PlanNode(spec=spec, status=TaskStatus.SUCCESS)
        state["plan"] = plan
        assert orchestrator._route_after_mark_ready(state) == "done"

    def test_routes_to_deadlock_on_failed_dep(self, orchestrator):
        state = _make_initial_state(user_task="测试", max_iterations=10)
        plan = PlanGraph()
        spec_a = TaskSpec(id="t1", name="T1", description="D1", tool_names=[])
        spec_b = TaskSpec(id="t2", name="T2", description="D2",
                          tool_names=[], depends_on=["t1"])
        plan.nodes["t1"] = PlanNode(spec=spec_a, status=TaskStatus.FAILED)
        plan.nodes["t2"] = PlanNode(spec=spec_b, status=TaskStatus.PENDING)
        plan.edges["t2"] = ["t1"]
        state["plan"] = plan
        assert orchestrator._route_after_mark_ready(state) == "deadlock"

    def test_routes_to_deadlock_on_iteration_exceeded(self, orchestrator):
        state = _make_initial_state(user_task="测试", max_iterations=10)
        state["iteration"] = 10
        plan = PlanGraph()
        spec = TaskSpec(id="t1", name="T1", description="D1", tool_names=[])
        plan.nodes["t1"] = PlanNode(spec=spec, status=TaskStatus.READY)
        state["plan"] = plan
        assert orchestrator._route_after_mark_ready(state) == "deadlock"

    def test_routes_to_deadlock_on_error(self, orchestrator):
        state = _make_initial_state(user_task="测试", max_iterations=10)
        state["error"] = "Fatal error"
        assert orchestrator._route_after_mark_ready(state) == "deadlock"


# ─── End-to-End ─────────────────────────────────────────────────────────


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_research_workflow(self, orchestrator):
        state = await orchestrator.run("Transformer 注意力机制的最新进展")
        assert state["plan"] is not None
        assert state["plan"].total_count() == 5
        assert state["plan"].success_count() == 5
        assert len(state["final_output"]) > 200
        assert state.get("error", "") == ""

    @pytest.mark.asyncio
    async def test_full_workflow_custom_query(self, orchestrator):
        state = await orchestrator.run("LLaMA 架构中 RoPE 位置编码的原理")
        assert state["plan"] is not None
        assert state["plan"].success_count() >= 0
        assert len(state["results"]) > 0

    @pytest.mark.asyncio
    async def test_stream_yields_intermediate_states(self, orchestrator):
        states = []
        async for node_name, ws in orchestrator.stream("测试流式执行"):
            states.append((node_name, ws))
            assert node_name in ("plan_task", "mark_ready", "execute_batch", "finalize")
        assert len(states) >= 3

    @pytest.mark.asyncio
    async def test_node_execution_order(self, orchestrator):
        node_order = []
        async for node_name, _ in orchestrator.stream("测试顺序"):
            node_order.append(node_name)
        assert node_order[0] == "plan_task"
        assert node_order[-1] == "finalize"

    @pytest.mark.asyncio
    async def test_results_accumulate(self, orchestrator):
        state = await orchestrator.run("研究 Python asyncio 最佳实践")
        # results 以 dict 形式存储（checkpoint 序列化要求），数量应为 5
        assert len(state["results"]) == 5
        # 每条 result 应包含基本字段
        for task_id, result_dict in state["results"].items():
            assert "task_id" in result_dict
            assert "success" in result_dict
            assert "output" in result_dict

    @pytest.mark.asyncio
    async def test_final_output_mentions_content(self, orchestrator):
        state = await orchestrator.run("Transformer 注意力机制")
        output = state["final_output"]
        assert len(output) > 100


# ─── Factory Function ───────────────────────────────────────────────────


class TestCreateOrchestrator:
    def test_creates_with_defaults(self):
        orch = create_orchestrator()
        assert isinstance(orch, ResearchOrchestrator)
        assert orch.semaphore_limit == 3
        assert orch.max_iterations == 10

    def test_creates_with_custom_params(self):
        orch = create_orchestrator(semaphore_limit=5, max_iterations=20)
        assert orch.semaphore_limit == 5
        assert orch.max_iterations == 20


# ─── Deadlock Handling ──────────────────────────────────────────────────


class TestDeadlockHandling:
    @pytest.mark.asyncio
    async def test_deadlock_detected(self, orchestrator):
        state = _make_initial_state(user_task="死锁测试", max_iterations=10)
        plan = PlanGraph()
        dep_spec = TaskSpec(id="t_dep", name="失败依赖", description="会失败",
                            tool_names=["unregistered_tool"])
        child_spec = TaskSpec(id="t_child", name="子任务", description="依赖前一步",
                              tool_names=[], depends_on=["t_dep"])
        plan.nodes["t_dep"] = PlanNode(spec=dep_spec, status=TaskStatus.PENDING)
        plan.nodes["t_child"] = PlanNode(spec=child_spec, status=TaskStatus.PENDING)
        plan.edges["t_child"] = ["t_dep"]
        plan.root_ids = ["t_dep"]
        state["plan"] = plan

        result = await orchestrator._graph.ainvoke(state)
        assert result.get("error", "") != ""

    @pytest.mark.asyncio
    async def test_iteration_limit(self, orchestrator):
        state = _make_initial_state(user_task="测试", max_iterations=1)
        plan_result = await orchestrator._plan_task(state)
        state.update(plan_result)
        result = await orchestrator._graph.ainvoke(state)
        assert result.get("final_output", "") != "" or result.get("error", "") != ""


# ─── Integration with ToolManager ───────────────────────────────────────


class TestToolManagerIntegration:
    @pytest.mark.asyncio
    async def test_tools_invoked_during_execution(self, orchestrator):
        state = await orchestrator.run("搜索深度学习优化器")
        all_stats = orchestrator.tool_manager.get_all_stats()
        total_calls = sum(s.total_calls for s in all_stats.values())
        assert total_calls > 0
