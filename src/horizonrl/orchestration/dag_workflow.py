"""
LangGraph DAG 编排 —— Agent 工作流的主状态机（v2: Verifier+Replanner+Memory+Writer 全集成）。

── 图结构 ──

    START
      │
      ▼
  ┌──────────┐
  │plan_task │  Planner 将 UserTask 分解为 PlanGraph + 生成 session_id
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │mark_ready│  将依赖满足的 PENDING 节点标记为 READY
  └────┬─────┘
       │
       ▼
  ┌──────────────┐          ┌──────────┐
  │ route_after  │ "done" → │ finalize │ → END
  │  _mark_ready │──────────│(Writer双 │
  └──────┬───────┘          │ 输出)    │
         │                  └──────────┘
         │ "execute"            ▲
         ▼                     │ "deadlock"
  ┌──────────────┐             │ (也到finalize)
  │execute_batch │             │
  └──────┬───────┘             │
         │                     │
         ▼                     │
  ┌──────────────┐   ┌─────────┴──────┐
  │verify_batch  │   │ route_after    │
  │(Verifier+L1) │──▶│   _verify      │
  └──────────────┘   └──┬───┬───┬─────┘
                        │   │   │
    "continue"  ◄───────┘   │   └── "deadlock" → finalize
    (to mark_ready)         │
                    "done" ─┘
                    (to finalize)
                        │
                "replan" │
                        ▼
              ┌──────────────┐
              │   replan     │  Replanner 生成 PlanPatch → 回写 PlanGraph
              │(记录重规划)  │  L1 记录
              └──────┬───────┘
                     │
                     ▼
                 mark_ready (loop)

── 使用方式 ──

    orchestrator = ResearchOrchestrator(planner, tool_manager)
    result = await orchestrator.run("Transformer 注意力机制的最新进展")
    print(result["final_output"])
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import time
import types as _types
import typing
import uuid
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── LangGraph 工作流状态 ────────────────────────────────────────────────
# 每个节点返回部分更新，LangGraph 按 reducer 语义合并。
#
# 字段说明：
#     user_task:     用户的原始问题描述
#     session_id:    会话唯一标识（plan_task 生成）
#     plan:          Planner 分解后的 PlanGraph（含 DAG 节点和边）
#     results:       task_id -> StepResult dict 累积映射
#     verifications: node_id -> VerificationResult dict 累积映射
#     iteration:     当前已执行轮数
#     replan_count:  重规划触发次数
#     max_iterations: 防止死循环的硬上限
#     final_output:  最终报告文本（Writer v2 生成）
#     error:         错误信息（死锁或异常时设置）
from typing import Annotated, Literal, TypedDict, Union

from langgraph.graph import END, StateGraph

from horizonrl.agent.planner import Planner
from horizonrl.agent.replanner import Replanner
from horizonrl.agent.verifier import Verifier
from horizonrl.agent.worker import AgentWorker
from horizonrl.agent.writer import Writer
from horizonrl.memory.hierarchical_memory import HierarchicalMemory
from horizonrl.schemas.result import (
    StepResult,
    VerificationResult,
)
from horizonrl.schemas.task import (
    PatchType,
    PlanGraph,
    PlanNode,
    TaskStatus,
    UserTask,
)


def _dict_merge(a: dict, b: dict) -> dict:
    """Reducer：合并两个 dict（LangGraph Annotated 累加器）。"""
    return {**a, **b}


class WorkflowState(TypedDict):
    """LangGraph 工作流状态 TypedDict（v2: Phase 2 类型安全升级）。

    Annotated 字段使用 _dict_merge reducer 支持节点间增量累加；
    普通字段遵循默认 replace 语义（节点返回新值即替换）。
    """

    user_task: str
    session_id: str
    plan: PlanGraph | None
    results: Annotated[dict[str, dict], _dict_merge]  # 累加：结果永不删除
    verifications: dict[str, dict]  # 替换：重规划时需要删除旧验证记录
    iteration: int
    replan_count: int
    max_iterations: int
    final_output: str
    error: str
    started_at: float


def _make_initial_state(
    user_task: str = "",
    max_iterations: int = 10,
    session_id: str = "",
) -> WorkflowState:
    """创建初始工作流状态。"""
    return WorkflowState(
        user_task=user_task,
        session_id=session_id,
        plan=None,
        results={},
        verifications={},
        iteration=0,
        replan_count=0,
        max_iterations=max_iterations,
        final_output="",
        error="",
        started_at=time.time(),
    )


def _to_dict(obj):
    """递归序列化 dataclass/enum/list/dict 为 JSON 兼容的纯 Python 结构。

    基于 dataclasses.fields() 自动遍历所有字段，新增字段无需手动同步。
    """
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict(cls: type, data: dict):
    """从 dict 反序列化为 dataclass 实例。

    通过 typing.get_type_hints() 解析字段类型（含嵌套 dataclass 和枚举），
    递归重建完整对象图。
    """
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, object] = {}
    for f in dataclasses.fields(cls):
        key = f.name
        if key not in data:
            continue
        kwargs[key] = _convert_field(data[key], hints.get(key))
    return cls(**kwargs)


def _convert_field(value, field_type):
    """按目标类型递归转换字段值，处理 Union/Enum/dataclass/list[dataclass]。"""
    if value is None or field_type is None:
        return value
    origin = getattr(field_type, "__origin__", None)
    args = getattr(field_type, "__args__", ())
    # Optional[T] = Union[T, None]
    if origin in (_types.UnionType, Union):
        for arg in args:
            if arg is not type(None):  # noqa: E721
                return _convert_field(value, arg)
        return value
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return field_type(value)
    if dataclasses.is_dataclass(field_type):
        return _from_dict(field_type, value)
    if origin is list and args and dataclasses.is_dataclass(args[0]):
        return [_from_dict(args[0], item) for item in value]
    return value


# ─── 路由决策类型 ────────────────────────────────────────────────────────

RouteDecision = Literal["execute", "done", "deadlock"]
VerifyRoute = Literal["continue", "replan", "done", "deadlock"]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ResearchOrchestrator ── 主编排器                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ResearchOrchestrator:
    """基于 LangGraph StateGraph 的多 Agent 研究编排器（v2 全集成）。

    管理完整生命周期：规划 → 执行 → 验证 → 重规划 → 汇总。
    集成 Verifier/Replanner/HierarchicalMemory/Writer v2。

    Attributes:
        planner: 任务分解器
        tool_manager: 统一工具管理器（共享给所有 Worker）
        verifier: 结构化验证器（默认 rule 模式）
        replanner: 局部重规划器（默认 max_retries=3）
        memory: 分层记忆 L1/L2/L3（默认 HierarchicalMemory）
        writer: 报告合成器（默认 template 模式）
        semaphore_limit: 每轮最大并发 Worker 数
        max_iterations: 最多执行轮数（防止死循环）
    """

    def __init__(
        self,
        planner: Planner,
        tool_manager,
        semaphore_limit: int = 3,
        max_iterations: int = 10,
        verifier: Verifier | None = None,
        replanner: Replanner | None = None,
        memory: HierarchicalMemory | None = None,
        writer: Writer | None = None,
        embedding_client=None,  # LLMClient for embedding (default: None = n-gram fallback)
    ):
        self.planner = planner
        self.tool_manager = tool_manager
        self.semaphore_limit = semaphore_limit
        self.max_iterations = max_iterations
        self.verifier = verifier or Verifier(mode="rule")
        self.replanner = replanner or Replanner(
            max_retries_per_task=3, max_total_replans=5
        )
        self.memory = memory or HierarchicalMemory()
        self.writer = writer or Writer(mode="template")

        # 注入 embedding client 到 L3 经验归档（启用真实向量检索）
        if embedding_client is not None:
            self.memory.set_embedding_client(embedding_client)

        # 构建并编译 LangGraph 状态图
        self._graph = self._build_graph()

    # ── 公共 API ────────────────────────────────────────────────────────

    async def run(self, user_task: str, session_id: str = "") -> WorkflowState:
        """执行完整的研究工作流。

        Args:
            user_task: 用户的研究问题（自然语言）。
            session_id: 可选，指定会话 ID。空字符串则自动生成。

        Returns:
            包含 plan、results、final_output 的完整状态 dict。
        """
        # 每次 run() 重置 Replanner 和 Memory 状态，确保会话隔离
        self.replanner.reset()
        self.memory.clear()

        initial_state = _make_initial_state(
            user_task=user_task,
            max_iterations=self.max_iterations,
            session_id=session_id,
        )
        return await self._graph.ainvoke(initial_state)

    async def stream(self, user_task: str, session_id: str = ""):
        """流式执行工作流，每步返回中间状态。

        Yields:
            (node_name, state_dict) 每完成一个节点就产出。
        """
        initial_state = _make_initial_state(
            user_task=user_task,
            max_iterations=self.max_iterations,
            session_id=session_id,
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
        builder = StateGraph(WorkflowState)

        # 节点注册
        builder.add_node("plan_task", self._plan_task)
        builder.add_node("mark_ready", self._mark_ready)
        builder.add_node("execute_batch", self._execute_batch)
        builder.add_node("verify_batch", self._verify_batch)
        builder.add_node("replan", self._replan)
        builder.add_node("finalize", self._finalize)

        # 边：START → plan_task → mark_ready
        builder.set_entry_point("plan_task")
        builder.add_edge("plan_task", "mark_ready")

        # 条件边：mark_ready → route → execute / done / deadlock
        # 注意：deadlock 也路由到 finalize，确保用户能拿到部分报告
        builder.add_conditional_edges(
            "mark_ready",
            self._route_after_mark_ready,
            {
                "execute": "execute_batch",
                "done": "finalize",
                "deadlock": "finalize",
            },
        )

        # execute_batch → verify_batch（执行完后验证）
        builder.add_edge("execute_batch", "verify_batch")

        # 条件边：verify_batch → route_verify → continue / replan / done / deadlock
        # 注意：deadlock 也路由到 finalize，确保用户能拿到部分报告
        builder.add_conditional_edges(
            "verify_batch",
            self._route_after_verify,
            {
                "continue": "mark_ready",
                "replan": "replan",
                "done": "finalize",
                "deadlock": "finalize",
            },
        )

        # replan → mark_ready（重规划后重新调度）
        builder.add_edge("replan", "mark_ready")

        # 终态：finalize → END
        builder.add_edge("finalize", END)

        return builder.compile()

    # ── 节点实现 ─────────────────────────────────────────────────────────

    async def _plan_task(self, state: WorkflowState) -> WorkflowState:
        """plan_task 节点：用户任务 → PlanGraph。"""
        if state["plan"] is not None:
            # 已有 plan（如从 checkpoint 恢复），确保 session_id 存在
            sid = state["session_id"] or f"session_{uuid.uuid4().hex[:12]}"
            return {"session_id": sid}

        session_id = state["session_id"] or f"session_{uuid.uuid4().hex[:12]}"
        user_task = UserTask(
            description=state["user_task"],
            max_steps=30,
            max_tokens=50_000,
        )
        if inspect.iscoroutinefunction(self.planner.plan):
            plan = await self.planner.plan(user_task)
        else:
            plan = self.planner.plan(user_task)
        return {"plan": plan, "session_id": session_id}

    async def _mark_ready(self, state: WorkflowState) -> WorkflowState:
        """mark_ready 节点：将依赖满足的 PENDING 节点标记为 READY。

        同时检测死锁：若无 READY 节点但有 PENDING，说明依赖图存在死锁环
        或所有 PENDING 节点的依赖均无法满足。
        """
        plan = state["plan"]
        if plan is None:
            return {"error": "plan is None"}

        for node in plan.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            deps_satisfied = all(
                plan.nodes[dep_id].status in (
                    TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.SKIPPED,
                    TaskStatus.CANCELLED,
                )
                for dep_id in node.depends_on
            )
            if deps_satisfied:
                node.status = TaskStatus.READY

        # 死锁检测
        error = state["error"]
        if state["iteration"] >= state["max_iterations"]:
            error = f'Deadlock: iteration {state["iteration"]} >= max {state["max_iterations"]}'
        elif not plan.get_ready_nodes() and plan.has_pending_work():
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
                error = "Deadlock: " + "; ".join(unmet_info)

        return {"plan": plan, "error": error}

    async def _execute_batch(self, state: WorkflowState) -> WorkflowState:
        """execute_batch 节点：并发执行本轮所有 READY 任务。

        带有整体超时保护：若批次执行超过 120 秒则取消剩余任务，
        避免因个别工具挂起而导致整个编排管道永久阻塞。
        使用 asyncio.wait 而非 gather+wait_for 以确保取消传播完成后再收集结果。
        """
        plan = state["plan"]
        if plan is None:
            return {"error": "plan is None"}

        ready_nodes = plan.get_ready_nodes()
        if not ready_nodes:
            return {"plan": plan}

        sem = asyncio.Semaphore(self.semaphore_limit)

        async def _run_one(node: PlanNode) -> StepResult:
            node.status = TaskStatus.RUNNING
            try:
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
            except asyncio.CancelledError:
                node.finished_at = time.time()
                node.status = TaskStatus.FAILED
                node.error_msg = "批次执行超时"
                return StepResult(
                    task_id=node.spec.id,
                    success=False,
                    output="",
                    evidence=[],
                    tool_calls=[],
                    tokens_used=0,
                    elapsed=0.0,
                    error="批次执行超时，任务被取消",
                    worker_id=f"wrk_{node.id}",
                )

        # 整体批次超时：最多 120 秒（足够 2 工具 × 20s/次 × 1 重试 + 余量）
        batch_timeout = 120.0
        tasks = [asyncio.create_task(_run_one(n)) for n in ready_nodes]
        done, pending = await asyncio.wait(tasks, timeout=batch_timeout)

        # 取消超时任务并等待取消传播完成
        if pending:
            for t in pending:
                t.cancel()
            await asyncio.wait(pending)

        # 所有任务已完成（成功/失败/取消），收集结果
        results: dict[str, dict] = {}
        for t in tasks:
            try:
                r = t.result()
                results[r.task_id] = _to_dict(r)
            except Exception:
                pass

        return {
            "plan": plan,
            "results": results,
            "iteration": state["iteration"] + 1,
        }

    # ── Verifier + Replanner 节点 ────────────────────────────────────────

    async def _verify_batch(self, state: WorkflowState) -> WorkflowState:
        """verify_batch 节点：对本轮执行结果进行并行验证。

        遍历 results 中尚未验证的任务，调用 Verifier 做质量检查。
        asyncio.gather 并行验证所有待验证任务。
        """
        plan = state["plan"]
        if plan is None:
            return {"error": "plan is None"}

        raw_results = state["results"]

        # 收集待验证的任务（跳过已验证的）
        verify_tasks: list[tuple[str, PlanNode, StepResult]] = []
        for node_id, node in plan.nodes.items():
            result_dict = raw_results.get(node.spec.id)
            if result_dict is None:
                continue
            if node_id in state["verifications"]:
                continue
            if node.status not in (TaskStatus.SUCCESS, TaskStatus.FAILED):
                continue
            result = _from_dict(StepResult, result_dict)
            verify_tasks.append((node_id, node, result))

        if not verify_tasks:
            return {"plan": plan}

        # 并行验证
        async def _verify_one(node_id: str, node: PlanNode, result: StepResult):
            task_desc = node.spec.description or node.spec.name
            vr = await self.verifier.verify(result, task_desc)
            return node_id, node, result, vr

        # 从已有验证记录开始（replace 语义要求返回完整 dict）
        verifications = dict(state["verifications"])
        batch = await asyncio.gather(*[
            _verify_one(nid, node, result)
            for nid, node, result in verify_tasks
        ])
        for node_id, node, result, vr in batch:
            verifications[node_id] = _to_dict(vr)
            if vr.pass_:
                node.status = TaskStatus.SUCCESS
            else:
                node.status = TaskStatus.FAILED
                node.error_msg = vr.feedback
            self.memory.record_task(
                task_id=result.task_id,
                task_name=node.spec.name,
                output=result.output,
                success=result.success,
                error_type=vr.error_type.value if vr else "",
                evidence_count=len(result.evidence),
                tool_calls=len(result.tool_calls),
                tokens_used=result.tokens_used,
                elapsed=result.elapsed,
            )

        # L1 自动压缩（超阈值时 L1→L2）
        self.memory.auto_compress()

        return {"plan": plan, "verifications": verifications}

    def _route_after_verify(self, state: WorkflowState) -> VerifyRoute:
        """verify_batch 之后的调度决策。

        决策树：
            1. 验证全部通过且无更多 pending → done
            2. 有可重试的失败任务 → replan
            3. 全部通过但有更多 pending → continue
            4. 有失败但不可重试 → deadlock
        """
        plan = state["plan"]
        verifications = state["verifications"]

        if plan is None:
            return "done"

        # 收集本轮验证失败的节点
        failed_nodes: list[str] = []
        for node_id, vdict in verifications.items():
            vr = _from_dict(VerificationResult,vdict)
            if not vr.pass_:
                node = plan.nodes.get(node_id)
                if node and node.status == TaskStatus.FAILED:
                    failed_nodes.append(node_id)

        # 无失败 → 检查是否还有工作
        if not failed_nodes:
            if plan.has_pending_work():
                return "continue"
            return "done"

        # 有失败 → 检查是否可重规划
        can_replan_any = False
        for nid in failed_nodes:
            if self.replanner.should_replan(
                _from_dict(VerificationResult,verifications[nid]), nid
            ):
                can_replan_any = True
                break

        if can_replan_any:
            return "replan"

        # 失败且无法重试 → 检查是否有其他可继续的工作
        if plan.has_pending_work():
            return "continue"

        return "deadlock"

    async def _replan(self, state: WorkflowState) -> WorkflowState:
        """replan 节点：对验证失败的任务生成 PlanPatch 并应用。"""
        plan = state["plan"]
        if plan is None:
            return {"error": "replan: plan is None"}

        verifications = dict(state["verifications"])
        replan_count = state["replan_count"]

        for node_id, vdict in list(verifications.items()):
            vr = _from_dict(VerificationResult,vdict)
            if vr.pass_:
                continue
            if not self.replanner.should_replan(vr, node_id):
                continue

            if inspect.iscoroutinefunction(self.replanner.replan):
                patch = await self.replanner.replan(vr, plan, node_id)
            else:
                patch = self.replanner.replan(vr, plan, node_id)

            if patch is not None:
                self.replanner.apply_patch(plan, patch)
                replan_count += 1
                self.memory.record_replan()
                if patch.patch_type == PatchType.RETRY:
                    verifications.pop(node_id, None)

        return {
            "plan": plan,
            "replan_count": replan_count,
            "verifications": verifications,
        }

    # ── 终态节点 ──────────────────────────────────────────────────────────

    async def _finalize(self, state: WorkflowState) -> WorkflowState:
        """finalize 节点：用 Writer v2 生成 final_answer.md + debug_report.md。

        Writer 不可用时回退到原始 Markdown 拼接。
        """
        plan = state["plan"]
        raw_results = state["results"]
        verifications = state["verifications"]
        replan_count = state["replan_count"]
        query = state["user_task"]
        session_id = state["session_id"]

        if plan is None:
            return {
                "final_output": "No plan generated.",
                "error": state["error"],
            }

        # 转换 dict → 对象，供 Writer 使用
        results = {k: _from_dict(StepResult, v) for k, v in raw_results.items()}
        vr_objects = {k: _from_dict(VerificationResult,v) for k, v in verifications.items()}
        mem_ctx = self.memory.get_context()

        # 统计工具调用和耗时
        total_tool_calls = 0
        total_elapsed = 0.0
        for r in results.values():
            total_tool_calls += len(r.tool_calls)
            total_elapsed += r.elapsed
        started_at = state.get("started_at", time.time())
        wall_time = time.time() - started_at

        stats = {
            "total_count": plan.total_count(),
            "success_count": plan.success_count(),
            "rounds": state["iteration"],
            "total_tool_calls": total_tool_calls,
            "total_replans": replan_count,
            "total_elapsed": f"{wall_time:.1f}s",
        }

        try:
            final_path, debug_path = await self.writer.write_reports(
                query=query,
                session_id=session_id,
                plan=plan,
                results=results,
                verifications=vr_objects,
                memory_ctx=mem_ctx,
                stats=stats,
            )
            final_text = Path(final_path).read_text(encoding="utf-8")
            return {
                "final_output": final_text,
                "plan": plan,
                "replan_count": replan_count,
                "error": state["error"],
            }
        except Exception:
            logger.exception("Writer 生成报告失败，回退到原始 Markdown 拼接")
            final_text = self._build_raw_final(state)
            return {
                "final_output": final_text,
                "plan": plan,
                "replan_count": replan_count,
                "error": state["error"],
            }

    def _build_raw_final(self, state: WorkflowState) -> str:
        """原始 Markdown 拼接（Writer 不可用时的 fallback）。"""
        plan = state["plan"]
        raw_results = state["results"]
        verifications = state["verifications"]
        replan_count = state["replan_count"]
        parts: list[str] = []

        error = state["error"]

        if plan is None:
            return f"No plan generated.{chr(10)}Error: {error}" if error else "No plan generated."

        parts.append("# 研究任务执行报告\n")
        if error:
            parts.append(f"> [ERROR] 工作流异常: {error}\n")
        parts.append(f'任务描述: {state["user_task"]}\n')
        parts.append(f"完成情况: {plan.success_count()}/{plan.total_count()} 子任务成功\n")
        parts.append(f'执行轮数: {state["iteration"]}\n\n')

        for node in plan.nodes.values():
            result_dict = raw_results.get(node.spec.id)
            vdict = verifications.get(node.id, {})
            if result_dict is None:
                parts.append(f"## {node.spec.name}\n状态: {node.status.value}\n\n")
                continue
            success = result_dict.get("success", False)
            elapsed = result_dict.get("elapsed", 0.0)
            evidence_count = len(result_dict.get("evidence", []))
            output_text = result_dict.get("output", "")
            v_score = vdict.get("score", 0)
            v_pass = vdict.get("pass_", success)
            icon = "+" if (success and v_pass) else "-"
            parts.append(f"## {node.spec.name}\n")
            parts.append(f"状态: {icon} | 耗时: {elapsed:.1f}s | "
                         f"证据: {evidence_count}条 | 验证: {v_score:.1f}\n\n")
            if output_text:
                parts.append(f"{output_text[:500]}\n\n")

        parts.append("---\n## 统计\n")
        total_tokens = sum(r.get("tokens_used", 0) for r in raw_results.values())
        total_time_s = sum(r.get("elapsed", 0.0) for r in raw_results.values())
        total_evidence = sum(len(r.get("evidence", [])) for r in raw_results.values())
        parts.append(f"- Token: {total_tokens}\n")
        parts.append(f"- 总耗时: {total_time_s:.1f}s\n")
        parts.append(f"- 收集证据: {total_evidence}条\n")
        parts.append(f"- 重规划: {replan_count}次\n")

        mem_stats = self.memory.get_stats()
        if mem_stats["l1_count"] > 0:
            parts.append("\n## 记忆系统\n")
            parts.append(f"- L1条目: {mem_stats['l1_count']}, "
                         f"Token: {mem_stats['l1_tokens']}/{mem_stats['l1_max_tokens']}\n")
            parts.append(f"- L2摘要: {mem_stats['l2_count']}/{mem_stats['l2_max']}\n")
            parts.append(f"- L3归档: {mem_stats['l3_count']}条\n")

        return "\n".join(parts)

    # ── 路由逻辑 ─────────────────────────────────────────────────────────

    def _route_after_mark_ready(self, state: WorkflowState) -> RouteDecision:
        """mark_ready 之后的调度决策。

        决策树：
            1. error 已设置 → deadlock
            2. 迭代超限 → deadlock（防死循环）
            3. 有 READY 节点 → execute（继续执行）
            4. 无 READY 但有 PENDING → deadlock（由 _mark_ready 预设 error）
            5. 全部终态 → done
        """
        if state["error"]:
            return "deadlock"

        if state["iteration"] >= state["max_iterations"]:
            return "deadlock"

        plan = state["plan"]
        if plan is None:
            return "done"

        if plan.get_ready_nodes():
            return "execute"

        if not plan.has_pending_work():
            return "done"

        # 无 READY 但有 PENDING → deadlock（_mark_ready 已设置 error）
        return "deadlock"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  工厂函数 ── 快速构建编排器                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def create_orchestrator(
    planner: Planner | None = None,
    tool_manager=None,
    semaphore_limit: int = 3,
    max_iterations: int = 10,
    embedding_client=None,
) -> ResearchOrchestrator:
    """工厂函数：一键创建可用的 ResearchOrchestrator。

    如果没有提供 planner 或 tool_manager，使用默认实例。

    Args:
        planner: Planner 实例，None 则创建默认。
        tool_manager: ToolManager 实例，None 则创建空的。
        semaphore_limit: 每轮最大并发数。
        max_iterations: 最大执行轮数。
        embedding_client: LLMClient，用于 L3 向量嵌入；None 则用 n-gram 哈希 fallback。

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
        from horizonrl.tools.web_search import WebSearchTool
        tool_manager = ToolManager()
        # 自动注册 WebSearchTool — 优先 Bocha(需Key)，fallback DDGS→Wikipedia→Mock
        tool_manager.register("web_search", WebSearchTool(provider="auto"))
    return ResearchOrchestrator(
        planner=planner,
        tool_manager=tool_manager,
        semaphore_limit=semaphore_limit,
        max_iterations=max_iterations,
        embedding_client=embedding_client,
    )
