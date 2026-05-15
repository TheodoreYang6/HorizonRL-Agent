"""Test Planner module with new schema types."""

from __future__ import annotations

from horizonrl.agent.planner import Planner
from horizonrl.schemas.task import UserTask, TaskSpec, PlanGraph, TaskPriority


class TestPlanner:
    def test_plan_research_task(self):
        planner = Planner()
        task = UserTask(description="Transformer 注意力机制的最新进展")
        plan = planner.plan(task)

        assert isinstance(plan, PlanGraph)
        assert len(plan.nodes) == 5  # research template has 5 steps
        assert len(plan.root_ids) >= 2  # first two tasks are independent

    def test_plan_code_task(self):
        planner = Planner()
        task = UserTask(description="修复 login 函数的 bug")
        plan = planner.plan(task)

        assert isinstance(plan, PlanGraph)
        assert len(plan.nodes) == 5

    def test_plan_code_task_with_tool(self):
        planner = Planner()
        task = UserTask(
            description="修复数据处理问题",
            required_tools=["code_execution"],
        )
        plan = planner.plan(task)
        assert len(plan.nodes) == 5

    def test_all_nodes_have_unique_ids(self):
        planner = Planner()
        task = UserTask(description="测试任务")
        plan = planner.plan(task)

        ids = list(plan.nodes.keys())
        assert len(ids) == len(set(ids))  # all unique

    def test_root_ids_have_no_dependencies(self):
        planner = Planner()
        task = UserTask(description="任意研究任务")
        plan = planner.plan(task)

        for rid in plan.root_ids:
            node = plan.nodes[rid]
            assert node.depends_on == []

    def test_plan_has_dependencies(self):
        planner = Planner()
        task = UserTask(description="研究任务")
        plan = planner.plan(task)

        # At least some nodes should have dependencies
        nodes_with_deps = [n for n in plan.nodes.values() if n.depends_on]
        assert len(nodes_with_deps) > 0

    def test_get_ready_nodes_returns_root_initially(self):
        planner = Planner()
        task = UserTask(description="测试")
        plan = planner.plan(task)

        # Before any nodes marked success, only root nodes with no deps could be ready
        # But get_ready_nodes requires READY status, so none initially
        ready = plan.get_ready_nodes()
        assert len(ready) == 0  # all are PENDING, not READY

    def test_has_pending_work_initially_true(self):
        planner = Planner()
        task = UserTask(description="测试")
        plan = planner.plan(task)
        assert plan.has_pending_work() is True
        assert plan.success_count() == 0
        assert plan.total_count() == 5

    def test_task_spec_has_required_fields(self):
        planner = Planner()
        task = UserTask(description="用 web_search 搜索信息")
        plan = planner.plan(task)

        first_node = list(plan.nodes.values())[0]
        spec = first_node.spec
        assert spec.id.startswith("task_")
        assert len(spec.name) > 0
        assert len(spec.description) > 0
        assert isinstance(spec.priority, TaskPriority)
        assert spec.retry_count == 0
        assert spec.max_retries == 3

    def test_classify_comparison_task(self):
        planner = Planner()
        task = UserTask(description="对比 PyTorch 和 TensorFlow 的优劣")
        plan = planner.plan(task)
        assert len(plan.nodes) == 5  # comparison template 5 steps
        # 逐维度对比应依赖前三个任务
        nodes = list(plan.nodes.values())
        compare_node = nodes[3]
        assert len(compare_node.depends_on) == 3

    def test_classify_summary_task(self):
        planner = Planner()
        task = UserTask(description="综述一下大语言模型的训练方法")
        plan = planner.plan(task)
        assert len(plan.nodes) == 3  # summary template 3 steps
        nodes = list(plan.nodes.values())
        # 信息去重 → 依赖多源收集
        assert len(nodes[1].depends_on) == 1
        # 结构化汇总 → 依赖信息去重
        assert len(nodes[2].depends_on) == 1

    def test_classify_factual_qa_task(self):
        planner = Planner()
        task = UserTask(description="什么是 Transformer 中的自注意力机制")
        plan = planner.plan(task)
        assert len(plan.nodes) == 3  # factual_qa template 3 steps
        nodes = list(plan.nodes.values())
        # 交叉验证 → 依赖权威来源检索
        assert len(nodes[1].depends_on) == 1
        # 生成答案 → 依赖交叉验证
        assert len(nodes[2].depends_on) == 1

    def test_classify_code_with_keywords(self):
        planner = Planner()
        for kw in ("修复 login bug", "代码重构", "debug 内存泄漏",
                    "编译报错处理", "函数测试"):
            task = UserTask(description=kw)
            plan = planner.plan(task)
            assert len(plan.nodes) == 5  # code template

    def test_classify_defaults_to_research(self):
        planner = Planner()
        task = UserTask(description="人工智能的未来发展趋势")
        plan = planner.plan(task)
        assert len(plan.nodes) == 5  # falls back to research template
