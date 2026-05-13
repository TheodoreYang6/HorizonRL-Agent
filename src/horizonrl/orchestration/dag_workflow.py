"""
LangGraph DAG 编排 —— Agent 工作流的主状态机。

用 LangGraph StateGraph 管理从用户任务到最终报告的完整生命周期。
每个节点是纯函数（state -> partial_state），条件边根据 PlanGraph 进度路由。

── 图结构 ──

    START
      │
      ▼
  ┌──────────┐
  │ plan_task │  Planner 将 UserTask 分解为 PlanGraph
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │mark_ready│  将依赖已满足的 PENDING 节点标记为 READY
  └────┬─────┘
       │
       ▼
  ┌──────────────┐
  │ route_after  │  条件路由
  │  _mark_ready │
  └──┬───┬───┬───┘
     │   │   │
     │   │   └── "deadlock" ──► END (error)
     │   │
     │   └────── "done" ──────► finalize ──► END
     │
     └────────── "execute" ───► execute_batch ──┐
                                                 │
                    ◄────────────────────────────┘
                    (loop back to mark_ready)

── 扩展点 (Phase 2+) ──

    execute_batch ──► verify_step ──► route_verify
                         │              │
                         │   pass ──────► mark_ready
                         │   fail ──────► replan ──► mark_ready
                         │   fatal ─────► END

── 使用方式 ──

    orchestrator = ResearchOrchestrator(planner, tool_manager)
    result = await orchestrator.run("Transformer 注意力机制的最新进展")
    print(result["final_output"])
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from horizonrl.schemas.task import (
    TaskSpec,
    TaskStatus,
    PlanNode,
    PlanGraph,
    UserTask,
)
from horizonrl.schemas.result import StepResult, EvidenceItem, ToolCall
from horizonrl.agent.planner import Planner
from horizonrl.agent.worker import AgentWorker

# ─── LangGraph 工作流状态 ────────────────────────────────────────────────
# 使用 Annotated TypedDict 定义 LangGraph 状态 schema。
# 每个节点返回部分更新，LangGraph 按 reducer 语义合并。
#
# 字段说明：
#     user_task:     用户的原始问题描述
#     plan:          Planner 分解后的 PlanGraph（含 DAG 节点和边）
#     results:       task_id -> StepResult 累积映射（操作符合并）
#     iteration:     当前已执行轮数
#     max_iterations: 防止死循环的硬上限
#     final_output:  最终报告文本
#     error:         错误信息（死锁或异常时设置）
#     started_at:    工作流启动时间戳

from typing import TypedDict, Annotated
import operator


def _make_initial_state(
    user_task: str = "",
    max_iterations: int = 10,
) -> dict:
    """创建初始工作流状态（纯 dict，兼容 LangGraph checkpoint 序列化）。"""
    return {
        "user_task": user_task,
        "plan": None,
        "results": {},
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_output": "",
        "error": "",
        "started_at": time.time(),
    }


def _step_result_to_dict(r: StepResult) -> dict:
    """将 StepResult 转为 JSON-可序列化的纯 dict。

    LangGraph 的 InMemorySaver 用 JsonPlusSerializer 做 checkpoint 持久化，
    dataclass 嵌套类型可能无法正确序列化/反序列化。转为纯 dict 保证跨节点传递可靠。
    """
    return {
        "task_id": r.task_id,
        "success": r.success,
        "output": r.output,
        "evidence": [
            {
                "content": e.content,
                "source": e.source,
                "source_type": e.source_type,
                "relevance_score": e.relevance_score,
                "retrieved_at": e.retrieved_at,
            }
            for e in r.evidence
        ],
        "tool_calls": [
            {
                "tool_name": tc.tool_name,
                "input": tc.input,
                "output": tc.output,
                "elapsed": tc.elapsed,
                "error": tc.error,
                "tokens_used": tc.tokens_used,
            }
            for tc in r.tool_calls
        ],
        "tokens_used": r.tokens_used,
        "elapsed": r.elapsed,
        "error": r.error,
        "worker_id": r.worker_id,
    }


def _dict_to_step_result(d: dict) -> StepResult:
    """从纯 dict 恢复 StepResult。"""
    return StepResult(
        task_id=d.get("task_id", ""),
        success=d.get("success", False),
        output=d.get("output", ""),
        evidence=[
            EvidenceItem(
                content=e.get("content", ""),
                source=e.get("source", ""),
                source_type=e.get("source_type", ""),
                relevance_score=e.get("relevance_score", 0.0),
                retrieved_at=e.get("retrieved_at", 0.0),
            )
            for e in d.get("evidence", [])
        ],
        tool_calls=[
            ToolCall(
                tool_name=tc.get("tool_name", ""),
                input=tc.get("input", {}),
                output=tc.get("output", ""),
                elapsed=tc.get("elapsed", 0.0),
                error=tc.get("error", ""),
                tokens_used=tc.get("tokens_used", 0),
            )
            for tc in d.get("tool_calls", [])
        ],
        tokens_used=d.get("tokens_used", 0),
        elapsed=d.get("elapsed", 0.0),
        error=d.get("error", ""),
        worker_id=d.get("worker_id", ""),
    )


# ─── 路由决策类型 ────────────────────────────────────────────────────────

RouteDecision = Literal["execute", "done", "deadlock"]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ResearchOrchestrator ── 主编排器                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ResearchOrchestrator:
    """基于 LangGraph StateGraph 的多 Agent 研究编排器。

    管理完整生命周期：规划 → 执行（循环）→ 汇总。
    支持 MemorySaver checkpoint，可以暂停/恢复/重放。

    Attributes:
        planner: 任务分解器
        tool_manager: 统一工具管理器（共享给所有 Worker）
        semaphore_limit: 每轮最大并发 Worker 数
        max_iterations: 最多执行轮数（防止死循环）

    Examples:
        >>> orchestrator = ResearchOrchestrator(planner, tool_manager)
        >>> state = await orchestrator.run("调研 Transformer 注意力机制")
        >>> state.final_output[:200]
        '## task_xxx...'
    """

    def __init__(
        self,
        planner: Planner,
        tool_manager,
        semaphore_limit: int = 3,
        max_iterations: int = 10,
    ):
        self.planner = planner
        self.tool_manager = tool_manager
        self.semaphore_limit = semaphore_limit
        self.max_iterations = max_iterations

        # 构建并编译 LangGraph 状态图
        self._graph = self._build_graph()

    # ── 公共 API ────────────────────────────────────────────────────────

    async def run(self, user_task: str) -> dict:
        """执行完整的研究工作流。

        Args:
            user_task: 用户的研究问题（自然语言）。

        Returns:
            包含 plan、results、final_output 的完整状态 dict。
        """
        initial_state = _make_initial_state(
            user_task=user_task,
            max_iterations=self.max_iterations,
        )
        return await self._graph.ainvoke(initial_state)

    async def stream(self, user_task: str):
        """流式执行工作流，每步返回中间状态。

        Yields:
            (node_name, state_dict) 每完成一个节点就产出。
        """
        initial_state = _make_initial_state(
            user_task=user_task,
            max_iterations=self.max_iterations,
        )
        async for event in self._graph.astream(initial_state):
            for node_name, node_output in event.items():
                yield node_name, node_output

    # ── 图构建 ──────────────────────────────────────────────────────────

    def _build_graph(self):
        """构建 LangGraph StateGraph。

        Returns:
            编译后的 CompiledStateGraph，可直接 ainvoke(initial_state, config)。
        """
        # 使用纯 dict 作为状态类型，兼容 checkpoint 序列化
        builder = StateGraph(dict)

        # 节点注册
        builder.add_node("plan_task", self._plan_task)
        builder.add_node("mark_ready", self._mark_ready)
        builder.add_node("execute_batch", self._execute_batch)
        builder.add_node("finalize", self._finalize)

        # 边：START → plan_task → mark_ready
        builder.set_entry_point("plan_task")
        builder.add_edge("plan_task", "mark_ready")

        # 条件边：mark_ready → route → execute / done / deadlock
        builder.add_conditional_edges(
            "mark_ready",
            self._route_after_mark_ready,
            {
                "execute": "execute_batch",
                "done": "finalize",
                "deadlock": END,
            },
        )

        # 循环：execute_batch → mark_ready
        builder.add_edge("execute_batch", "mark_ready")

        # 终态：finalize → END
        builder.add_edge("finalize", END)

        # 编译（Phase 2+ 接入 InMemorySaver 做持久化 checkpoint）
        # 当前不启用以避免嵌套 dataclass 序列化兼容问题，
        # 状态在内存中以纯 dict 形式在节点间传递，已通过 _step_result_to_dict 保证序列化安全。
        return builder.compile()

    # ── 节点实现 ─────────────────────────────────────────────────────────

    async def _plan_task(self, state: dict) -> dict:
        """plan_task 节点：用户任务 → PlanGraph。

        仅在 plan 为 None 时执行分解；如果 state 中已有 plan（如从 checkpoint 恢复），
        则直接透传，避免覆盖外部注入的测试 PlanGraph。
        """
        if state.get("plan") is not None:
            return {}  # 已有 plan，不重新分解
        user_task = UserTask(
            description=state.get("user_task", ""),
            max_steps=30,
            max_tokens=50_000,
        )
        plan = self.planner.plan(user_task)
        return {"plan": plan}

    async def _mark_ready(self, state: dict) -> dict:
        """mark_ready 节点：将依赖满足的 PENDING 节点标记为 READY。"""
        plan = state.get("plan")
        if plan is None:
            return {"error": "plan is None"}

        for node in plan.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            deps_satisfied = all(
                plan.nodes[dep_id].status == TaskStatus.SUCCESS
                for dep_id in node.depends_on
            )
            if deps_satisfied:
                node.status = TaskStatus.READY

        # 兜底：回传已有 results，避免 LangGraph dict state merge 时丢失
        return {"plan": plan, "results": state.get("results", {})}

    async def _execute_batch(self, state: dict) -> dict:
        """execute_batch 节点：并发执行本轮所有 READY 任务。"""
        plan = state.get("plan")
        if plan is None:
            return {"error": "plan is None"}

        ready_nodes = plan.get_ready_nodes()
        if not ready_nodes:
            return {"iteration": state.get("iteration", 0) + 1}

        sem = asyncio.Semaphore(self.semaphore_limit)

        async def _run_one(node: PlanNode) -> StepResult:
            node.status = TaskStatus.RUNNING
            async with sem:
                worker = AgentWorker(
                    worker_id=f"wrk_{node.id}",
                    tool_manager=self.tool_manager,
                )
                result = await worker.execute(node.spec)
                node.finished_at = time.time()
                if result.success:
                    node.status = TaskStatus.SUCCESS
                else:
                    node.status = TaskStatus.FAILED
                    node.error_msg = result.error
                return result

        batch_results = await asyncio.gather(*[_run_one(n) for n in ready_nodes])

        # 累积结果 — 以 JSON 兼容 dict 存储（checkpoint 序列化要求）
        results = dict(state.get("results", {}))
        for r in batch_results:
            results[r.task_id] = _step_result_to_dict(r)

        return {
            "plan": plan,
            "results": results,
            "iteration": state.get("iteration", 0) + 1,
        }

    async def _finalize(self, state: dict) -> dict:
        """finalize 节点：汇总所有结果生成最终输出。"""
        plan = state.get("plan")
        raw_results = state.get("results", {})
        parts: list[str] = []

        if plan is not None:
            parts.append("# 研究任务执行报告\n")
            parts.append(f"任务描述: {state.get('user_task', 'N/A')}\n")
            parts.append(f"完成情况: {plan.success_count()}/{plan.total_count()} 子任务成功\n")
            parts.append(f"执行轮数: {state.get('iteration', 0)}\n\n")

            for node in plan.nodes.values():
                result_dict = raw_results.get(node.spec.id)
                if result_dict is None:
                    parts.append(f"## {node.spec.name}\n状态: {node.status.value}\n\n")
                    continue
                success = result_dict.get("success", False)
                elapsed = result_dict.get("elapsed", 0.0)
                evidence_count = len(result_dict.get("evidence", []))
                output_text = result_dict.get("output", "")
                icon = "+" if success else "-"
                parts.append(f"## {node.spec.name}\n")
                parts.append(f"状态: {icon} | 耗时: {elapsed:.1f}s | "
                             f"证据: {evidence_count}条\n\n")
                if output_text:
                    parts.append(f"{output_text[:500]}\n\n")

            parts.append("---\n## 统计\n")
            total_tokens = sum(r.get("tokens_used", 0) for r in raw_results.values())
            total_time_s = sum(r.get("elapsed", 0.0) for r in raw_results.values())
            total_evidence = sum(len(r.get("evidence", [])) for r in raw_results.values())
            parts.append(f"- Token: {total_tokens}\n")
            parts.append(f"- 总耗时: {total_time_s:.1f}s\n")
            parts.append(f"- 收集证据: {total_evidence}条\n")
        else:
            parts.append("No plan generated.")

        return {"final_output": "\n".join(parts), "plan": plan, "results": raw_results}

    # ── 路由逻辑 ─────────────────────────────────────────────────────────

    def _route_after_mark_ready(self, state: dict) -> RouteDecision:
        """mark_ready 之后的调度决策。

        决策树：
            1. error 已设置 → deadlock
            2. 迭代超限 → deadlock（防死循环）
            3. 有 READY 节点 → execute（继续执行）
            4. 无 READY 但有 PENDING → deadlock（依赖无法满足）
            5. 全部终态 → done
        """
        plan = state.get("plan")
        error = state.get("error", "")

        if error:
            return "deadlock"

        if state.get("iteration", 0) >= state.get("max_iterations", 10):
            return "deadlock"

        if plan is None:
            return "done"

        ready = plan.get_ready_nodes()
        if ready:
            return "execute"

        if not plan.has_pending_work():
            return "done"

        pending = [
            n for n in plan.nodes.values()
            if n.status in (TaskStatus.PENDING, TaskStatus.READY)
        ]
        if pending:
            unmet_info = []
            for n in pending:
                unmet = [
                    d for d in n.depends_on
                    if plan.nodes[d].status != TaskStatus.SUCCESS
                ]
                unmet_info.append(f"{n.id} waiting on {unmet}")
            state["error"] = "Deadlock: " + "; ".join(unmet_info)
            return "deadlock"

        return "done"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  工厂函数 ── 快速构建编排器                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def create_orchestrator(
    planner: Planner | None = None,
    tool_manager=None,
    semaphore_limit: int = 3,
    max_iterations: int = 10,
) -> ResearchOrchestrator:
    """工厂函数：一键创建可用的 ResearchOrchestrator。

    如果没有提供 planner 或 tool_manager，使用默认实例。

    Args:
        planner: Planner 实例，None 则创建默认。
        tool_manager: ToolManager 实例，None 则创建空的。
        semaphore_limit: 每轮最大并发数。
        max_iterations: 最大执行轮数。

    Returns:
        配置好的 ResearchOrchestrator。

    Examples:
        >>> orch = create_orchestrator()
        >>> state = await orch.run("调研 LLaMA 架构")
    """
    if planner is None:
        planner = Planner()
    if tool_manager is None:
        from horizonrl.tools.manager import ToolManager
        tool_manager = ToolManager()
    return ResearchOrchestrator(
        planner=planner,
        tool_manager=tool_manager,
        semaphore_limit=semaphore_limit,
        max_iterations=max_iterations,
    )
