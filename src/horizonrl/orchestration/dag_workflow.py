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
import inspect
import time
import uuid
from pathlib import Path

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
from typing import Literal

from langgraph.graph import END, StateGraph

from horizonrl.agent.planner import Planner
from horizonrl.agent.replanner import Replanner
from horizonrl.agent.verifier import Verifier
from horizonrl.agent.worker import AgentWorker
from horizonrl.agent.writer import Writer
from horizonrl.memory.hierarchical_memory import HierarchicalMemory
from horizonrl.schemas.result import (
    ErrorType,
    EvidenceItem,
    StepResult,
    ToolCall,
    VerificationResult,
)
from horizonrl.schemas.task import (
    PatchType,
    PlanNode,
    TaskStatus,
    UserTask,
)


def _make_initial_state(
    user_task: str = "",
    max_iterations: int = 10,
) -> dict:
    """创建初始工作流状态（纯 dict，兼容 LangGraph checkpoint 序列化）。"""
    return {
        "user_task": user_task,
        "session_id": "",
        "plan": None,
        "results": {},
        "verifications": {},
        "iteration": 0,
        "replan_count": 0,
        "max_iterations": max_iterations,
        "final_output": "",
        "error": "",
        "started_at": time.time(),
    }


def _step_result_to_dict(r: StepResult) -> dict:
    """将 StepResult 转为 JSON-可序列化的纯 dict。"""
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
                "provider": e.provider,
                "search_query": e.search_query,
                "is_mock": e.is_mock,
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
                provider=e.get("provider", ""),
                search_query=e.get("search_query", ""),
                is_mock=e.get("is_mock", False),
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


def _verification_to_dict(vr: VerificationResult) -> dict:
    """将 VerificationResult 序列化为 dict。"""
    return {
        "pass_": vr.pass_,
        "score": vr.score,
        "error_type": vr.error_type.value,
        "feedback": vr.feedback,
        "evidence_gaps": vr.evidence_gaps,
        "suggested_actions": vr.suggested_actions,
        "tokens_used": vr.tokens_used,
        "elapsed": vr.elapsed,
    }


def _dict_to_verification(d: dict) -> VerificationResult:
    """从 dict 恢复 VerificationResult。"""
    error_str = d.get("error_type", "none")
    try:
        error_type = ErrorType(error_str)
    except ValueError:
        error_type = ErrorType.NONE
    return VerificationResult(
        pass_=d.get("pass_", False),
        score=d.get("score", 0.0),
        error_type=error_type,
        feedback=d.get("feedback", ""),
        evidence_gaps=d.get("evidence_gaps", []),
        suggested_actions=d.get("suggested_actions", []),
        tokens_used=d.get("tokens_used", 0),
        elapsed=d.get("elapsed", 0.0),
    )


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
        self.writer = Writer(mode="template")

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
        builder = StateGraph(dict)

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

    async def _plan_task(self, state: dict) -> dict:
        """plan_task 节点：用户任务 → PlanGraph。"""
        if state.get("plan") is not None:
            # 已有 plan（如从 checkpoint 恢复），确保 session_id 存在
            sid = state.get("session_id") or f"session_{uuid.uuid4().hex[:12]}"
            return {"session_id": sid}

        session_id = state.get("session_id") or f"session_{uuid.uuid4().hex[:12]}"
        user_task = UserTask(
            description=state.get("user_task", ""),
            max_steps=30,
            max_tokens=50_000,
        )
        plan = self.planner.plan(user_task)
        return {"plan": plan, "session_id": session_id}

    async def _mark_ready(self, state: dict) -> dict:
        """mark_ready 节点：将依赖满足的 PENDING 节点标记为 READY。"""
        plan = state.get("plan")
        if plan is None:
            return {
                "error": "plan is None",
                "results": state.get("results", {}),
                "verifications": state.get("verifications", {}),
                "replan_count": state.get("replan_count", 0),
            }

        for node in plan.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            deps_satisfied = all(
                plan.nodes[dep_id].status == TaskStatus.SUCCESS
                for dep_id in node.depends_on
            )
            if deps_satisfied:
                node.status = TaskStatus.READY

        # 兜底：回传已有状态，避免 LangGraph dict state merge 时丢失
        return {
            "plan": plan,
            "results": state.get("results", {}),
            "verifications": state.get("verifications", {}),
            "replan_count": state.get("replan_count", 0),
        }

    async def _execute_batch(self, state: dict) -> dict:
        """execute_batch 节点：并发执行本轮所有 READY 任务。"""
        plan = state.get("plan")
        if plan is None:
            return {
                "error": "plan is None",
                "results": state.get("results", {}),
                "verifications": state.get("verifications", {}),
                "replan_count": state.get("replan_count", 0),
            }

        ready_nodes = plan.get_ready_nodes()
        if not ready_nodes:
            # 防御：路由不应在此条件下到达，但保留安全返回
            return {
                "plan": plan,
                "results": state.get("results", {}),
                "verifications": state.get("verifications", {}),
                "replan_count": state.get("replan_count", 0),
            }

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
            "verifications": state.get("verifications", {}),
            "replan_count": state.get("replan_count", 0),
        }

    # ── Verifier + Replanner 节点 ────────────────────────────────────────

    async def _verify_batch(self, state: dict) -> dict:
        """verify_batch 节点：对本轮执行结果进行并行验证。

        遍历 results 中尚未验证的任务，调用 Verifier 做质量检查。
        使用 asyncio.gather 并行验证所有待验证任务。
        """
        plan = state.get("plan")
        raw_results = state.get("results", {})
        verifications = dict(state.get("verifications", {}))

        if plan is None:
            return {
                "verifications": verifications,
                "results": state.get("results", {}),
                "replan_count": state.get("replan_count", 0),
            }

        # 收集待验证的任务
        verify_tasks: list[tuple[str, PlanNode, StepResult]] = []
        for node_id, node in plan.nodes.items():
            result_dict = raw_results.get(node.spec.id)
            if result_dict is None:
                continue
            if node_id in verifications:
                continue
            if node.status not in (TaskStatus.SUCCESS, TaskStatus.FAILED):
                continue
            result = _dict_to_step_result(result_dict)
            verify_tasks.append((node_id, node, result))

        # 并行验证
        async def _verify_one(node_id: str, node: PlanNode, result: StepResult):
            task_desc = node.spec.description or node.spec.name
            vr = await self.verifier.verify(result, task_desc)
            return node_id, node, result, vr

        if verify_tasks:
            batch = await asyncio.gather(*[
                _verify_one(nid, node, result)
                for nid, node, result in verify_tasks
            ])
            for node_id, node, result, vr in batch:
                verifications[node_id] = _verification_to_dict(vr)
                if vr.pass_:
                    node.status = TaskStatus.SUCCESS
                else:
                    node.status = TaskStatus.FAILED
                    node.error_msg = vr.feedback
                self.memory.record(result, vr)

            # L1 自动压缩（超阈值时 L1→L2）
            self.memory.auto_compress()

        return {
            "plan": plan,
            "verifications": verifications,
            "results": raw_results,
            "replan_count": state.get("replan_count", 0),
        }

    def _route_after_verify(self, state: dict) -> VerifyRoute:
        """verify_batch 之后的调度决策。

        决策树：
            1. 验证全部通过且无更多 pending → done
            2. 有可重试的失败任务 → replan
            3. 全部通过但有更多 pending → continue
            4. 有失败但不可重试 → deadlock
        """
        plan = state.get("plan")
        verifications = state.get("verifications", {})

        if plan is None:
            return "done"

        # 收集本轮验证失败的节点
        failed_nodes: list[str] = []
        for node_id, vdict in verifications.items():
            vr = _dict_to_verification(vdict)
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
                _dict_to_verification(verifications[nid]), nid
            ):
                can_replan_any = True
                break

        if can_replan_any:
            return "replan"

        # 失败且无法重试 → 检查是否有其他可继续的工作
        if plan.has_pending_work():
            return "continue"

        return "deadlock"

    async def _replan(self, state: dict) -> dict:
        """replan 节点：对验证失败的任务生成 PlanPatch 并应用。"""
        plan = state.get("plan")
        verifications = dict(state.get("verifications", {}))
        replan_count = state.get("replan_count", 0)

        if plan is None:
            return {
                "error": "replan: plan is None",
                "results": state.get("results", {}),
                "verifications": state.get("verifications", {}),
                "replan_count": state.get("replan_count", 0),
            }

        for node_id, vdict in list(verifications.items()):
            vr = _dict_to_verification(vdict)
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
                # RETRY 后清除旧验证记录，让重试结果被重新验证
                if patch.patch_type == PatchType.RETRY:
                    verifications.pop(node_id, None)

        return {
            "plan": plan,
            "replan_count": replan_count,
            "verifications": verifications,
            "results": state.get("results", {}),
        }

    # ── 终态节点 ──────────────────────────────────────────────────────────

    async def _finalize(self, state: dict) -> dict:
        """finalize 节点：用 Writer v2 生成 final_answer.md + debug_report.md。

        Writer 不可用时回退到原始 Markdown 拼接。
        """
        plan = state.get("plan")
        raw_results = state.get("results", {})
        verifications = state.get("verifications", {})
        replan_count = state.get("replan_count", 0)
        query = state.get("user_task", "")
        session_id = state.get("session_id", "")

        if plan is None:
            return {
                "final_output": "No plan generated.",
                "error": state.get("error", ""),
                "replan_count": state.get("replan_count", 0),
                "results": state.get("results", {}),
                "verifications": state.get("verifications", {}),
            }

        # 转换 dict → 对象，供 Writer 使用
        results = {k: _dict_to_step_result(v) for k, v in raw_results.items()}
        vr_objects = {k: _dict_to_verification(v) for k, v in verifications.items()}
        mem_ctx = self.memory.get_context()
        stats = {
            "total_count": plan.total_count(),
            "rounds": state.get("iteration", 0),
            "total_replans": replan_count,
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
                "results": raw_results,
                "verifications": verifications,
                "replan_count": replan_count,
                "error": state.get("error", ""),
            }
        except Exception:
            # Writer 失败或 session_id 为空时回退到原始 Markdown 拼接
            final_text = self._build_raw_final(state)
            return {
                "final_output": final_text,
                "plan": plan,
                "results": raw_results,
                "verifications": verifications,
                "replan_count": replan_count,
                "error": state.get("error", ""),
            }

    def _build_raw_final(self, state: dict) -> str:
        """原始 Markdown 拼接（Writer 不可用时的 fallback）。"""
        plan = state.get("plan")
        raw_results = state.get("results", {})
        verifications = state.get("verifications", {})
        replan_count = state.get("replan_count", 0)
        parts: list[str] = []

        error = state.get("error", "")

        if plan is None:
            return f"No plan generated.{chr(10)}Error: {error}" if error else "No plan generated."

        parts.append("# 研究任务执行报告\n")
        if error:
            parts.append(f"> [ERROR] 工作流异常: {error}\n")
        parts.append(f"任务描述: {state.get('user_task', 'N/A')}\n")
        parts.append(f"完成情况: {plan.success_count()}/{plan.total_count()} 子任务成功\n")
        parts.append(f"执行轮数: {state.get('iteration', 0)}\n\n")

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
