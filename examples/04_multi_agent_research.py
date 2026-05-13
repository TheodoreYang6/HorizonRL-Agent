"""
=======================================================================
04_multi_agent_research.py — HorizonRL-Agent v1 完整集成 Demo
=======================================================================

这是 HorizonRL-Agent 的旗舰 Demo，将过去 12 个 Step 构建的所有模块串联成
一条完整的研究管道：

    UserTask → Planner → PlanGraph → Worker×N → Verifier → Replanner
                ↑ Memory (L1→L2) ↑  TrajectoryLogger (JSONL)  ↑
                                  ↓
                            FinalReport (Markdown)

Pipeline 阶段:
    Stage 0: 基础设施加载 (Config, Tools, Memory, Logger)
    Stage 1: 任务规划 (LLMPlanner / Planner → PlanGraph)
    Stage 2: DAG 并发执行 + 验证 + 重规划循环
    Stage 3: 记忆总结 (L1 → L2 语义压缩)
    Stage 4: 最终报告生成 (结构化 Markdown)
    Stage 5: 统计与输出

运行方式:
    # 离线模式（无需 API Key，使用模板规划 + 模拟工具）
    python examples/04_multi_agent_research.py

    # LLM 驱动模式（需要配置 .env 和 API Key）
    python examples/04_multi_agent_research.py --llm

    # 自定义研究问题
    python examples/04_multi_agent_research.py "量子计算对密码学的影响"

    # 指定并发数
    python examples/04_multi_agent_research.py --workers 5

输出文件:
    trajectories/session_<id>.jsonl  — 完整轨迹日志
    reports/session_<id>.md         — 最终研究报告
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Windows UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from horizonrl.config.settings import load_config, RootConfig
from horizonrl.schemas.task import (
    UserTask, TaskSpec, TaskStatus,
    PlanGraph, PlanNode,
)
from horizonrl.schemas.result import (
    StepResult, VerificationResult, ErrorType,
)
from horizonrl.schemas.event import EventType, TrajectoryEvent
from horizonrl.agent.planner import Planner, LLMPlanner
from horizonrl.agent.worker import AgentWorker
from horizonrl.agent.verifier import Verifier
from horizonrl.agent.replanner import Replanner
from horizonrl.agent.writer import Writer, WriterConfig
from horizonrl.tools.manager import ToolManager
from horizonrl.tools.mock import register_mock_tools
from horizonrl.memory.hierarchical_memory import HierarchicalMemory
from horizonrl.logging.trajectory_logger import TrajectoryLogger


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Stage 0: 基础设施                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def setup_infrastructure(use_llm: bool = False):
    """加载配置、注册工具、创建 LLM 客户端、初始化 Memory 和 Logger。

    Returns:
        (cfg, llm_client, tool_manager, memory, logger)
    """
    print_header("Stage 0: 基础设施加载")

    # ── 配置 ──
    dev_path = Path("configs/dev.yaml")
    try:
        cfg = load_config(dev_path if dev_path.exists() else None)
        print_status("config", f"Provider={cfg.llm.provider}, Model={cfg.llm.model}")
    except Exception:
        cfg = RootConfig()
        print_status("config", "使用默认配置", "warn")

    # ── 工具注册（真实优先，不可用时回退模拟）──
    mgr = ToolManager()

    # Web Search
    try:
        from horizonrl.tools.web_search import WebSearchTool
        mgr.register("web_search", WebSearchTool())
        print_status("tools", "WebSearch (DuckDuckGo)", "ok")
    except Exception:
        from horizonrl.tools.mock import MockWebSearch
        mgr.register("web_search", MockWebSearch())
        print_status("tools", "WebSearch → 模拟 (DuckDuckGo 不可用)", "warn")

    # Arxiv Search
    try:
        from horizonrl.tools.arxiv_search import ArxivSearchTool
        mgr.register("arxiv_search", ArxivSearchTool(max_results=5))
        print_status("tools", "ArxivSearch API", "ok")
    except Exception:
        from horizonrl.tools.mock import MockArxivSearch
        mgr.register("arxiv_search", MockArxivSearch())
        print_status("tools", "ArxivSearch → 模拟", "warn")

    # Code Execution (始终可用)
    try:
        from horizonrl.tools.code_execution import CodeExecutionTool
        mgr.register("code_execution", CodeExecutionTool(timeout=10.0))
        print_status("tools", "CodeExecution", "ok")
    except Exception:
        from horizonrl.tools.mock import MockCodeExecution
        mgr.register("code_execution", MockCodeExecution())
        print_status("tools", "CodeExecution → 模拟", "warn")

    # ── LLM 客户端 ──
    llm_client = None
    if use_llm and cfg.llm.api_key:
        from horizonrl.llm.client import LLMClient
        llm_client = LLMClient(cfg.llm)
        print_status("llm", f"LLM 模式: {cfg.llm.model}")
    else:
        print_status("llm", "离线模式 (模板规划 + 模拟工具)", "warn")

    # ── Memory ──
    memory = HierarchicalMemory(cfg.memory)
    print_status("memory", f"L1 容量={memory.l1.max_tokens} tokens, "
                 f"L2 容量={memory.l2.max_entries} 条")

    # ── Logger ──
    output_dir = cfg.agent.trajectory_dir if hasattr(cfg.agent, 'trajectory_dir') else "trajectories"
    logger = TrajectoryLogger(output_dir=output_dir)
    print_status("logger", f"输出目录: {logger.output_dir}")

    return cfg, llm_client, mgr, memory, logger


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Stage 1: 任务规划                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def stage1_planning(user_query: str, llm_client, logger: TrajectoryLogger):
    """将用户查询分解为 PlanGraph (DAG)。

    Returns:
        (PlanGraph, plan_tokens, plan_elapsed)
    """
    print_header("Stage 1: 任务规划")

    user_task = UserTask(description=user_query, max_steps=20)

    t0 = time.time()
    await logger.log(TrajectoryEvent(
        module="planner", event_type=EventType.PLAN_START,
        payload={"user_task": user_query},
    ))

    if llm_client is not None:
        planner = LLMPlanner(llm_client)
        plan = await planner.plan(user_task)
        mode = "LLM 智能拆解"
    else:
        planner = Planner()
        plan = planner.plan(user_task)
        mode = "模板拆解"

    elapsed = time.time() - t0

    await logger.log(TrajectoryEvent(
        module="planner", event_type=EventType.PLAN_COMPLETE,
        payload={
            "num_subtasks": plan.total_count(),
            "root_ids": plan.root_ids,
            "plan_json": _plan_to_summary(plan),
        },
        cost=0, latency=elapsed,
    ))

    print_status("plan", f"{mode}: {plan.total_count()} 个子任务, "
                 f"{len(plan.root_ids)} 个根节点 (可并行)")

    # 打印 DAG 结构
    for node in plan.nodes.values():
        deps = ", ".join(node.depends_on) if node.depends_on else "无"
        tools = ", ".join(node.spec.tool_names) or "无"
        print(f"    [{node.id}] {node.spec.name}")
        print(f"        工具: {tools}  |  依赖: {deps}")

    return plan, elapsed


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Stage 2: DAG 并发执行 + 验证 + 重规划                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def stage2_execution_loop(
    plan: PlanGraph,
    tool_manager: ToolManager,
    memory: HierarchicalMemory,
    logger: TrajectoryLogger,
    semaphore_limit: int = 3,
):
    """DAG 执行主循环：标记就绪 → 并发执行 → 验证 → 重规划 → 记录。

    Returns:
        {task_id: StepResult}, stats_dict
    """
    print_header("Stage 2: DAG 并发执行 + 验证 + 重规划")

    verifier = Verifier(mode="rule")
    replanner = Replanner(max_retries_per_task=3, max_total_replans=5)
    sem = asyncio.Semaphore(semaphore_limit)
    results: dict[str, StepResult] = {}
    verification_results: dict[str, VerificationResult] = {}

    round_num = 0
    total_tool_calls = 0
    total_replans = 0
    start_time = time.time()

    while plan.has_pending_work():
        round_num += 1

        # ── 标记就绪 ──
        for node in plan.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            deps_ok = all(
                plan.nodes[d].status == TaskStatus.SUCCESS
                for d in node.depends_on
            )
            if deps_ok:
                node.status = TaskStatus.READY

        ready = plan.get_ready_nodes()
        if not ready:
            pending = [n for n in plan.nodes.values()
                       if n.status in (TaskStatus.PENDING, TaskStatus.READY)]
            if pending:
                print(f"\n  ⚠ 死锁: {len(pending)} 个任务阻塞")
                for n in pending:
                    unmet = [d for d in n.depends_on
                             if plan.nodes[d].status != TaskStatus.SUCCESS]
                    print(f"     - {n.id}: 等待 {unmet}")
            break

        print(f"\n── Round {round_num}: {len(ready)} 个任务并发执行 "
              f"(Semaphore={semaphore_limit}) ──")

        # ── 并发执行 ──
        async def execute_one(node: PlanNode):
            node.status = TaskStatus.RUNNING
            node.started_at = time.time()

            await logger.log(TrajectoryEvent(
                module="worker", event_type=EventType.WORKER_START,
                payload={"task_id": node.id, "task_name": node.spec.name},
            ))

            async with sem:
                worker = AgentWorker(
                    worker_id=f"wrk_{node.id}",
                    tool_manager=tool_manager,
                )
                result = await worker.execute(node.spec)

            node.finished_at = time.time()

            await logger.log(TrajectoryEvent(
                module="worker",
                event_type=EventType.WORKER_COMPLETE if result.success else EventType.WORKER_ERROR,
                payload={"task_id": node.id, "success": result.success,
                         "evidence_count": len(result.evidence)},
                cost=result.tokens_used, latency=result.elapsed,
            ))

            return node, result

        batch = await asyncio.gather(*[execute_one(n) for n in ready])

        # ── 验证 + 重规划 ──
        for node, result in batch:
            results[result.task_id] = result

            # 验证
            t0 = time.time()
            vr = await verifier.verify(result, node.spec)
            verification_results[node.id] = vr

            await logger.log(TrajectoryEvent(
                module="verifier",
                event_type=EventType.VERIFY_COMPLETE if vr.pass_ else EventType.VERIFY_FAIL,
                payload={"task_id": node.id, "pass": vr.pass_, "score": vr.score,
                         "error_type": vr.error_type.value},
                latency=time.time() - t0,
            ))

            if vr.pass_:
                node.status = TaskStatus.SUCCESS
                memory.record(result, vr)
            else:
                # 尝试重规划
                patch = replanner.replan(vr, plan, node.id)
                if patch is not None:
                    replanner.apply_patch(plan, patch)
                    total_replans += 1
                    memory.record_replan()

                    await logger.log(TrajectoryEvent(
                        module="replanner", event_type=EventType.REPLAN_PATCH,
                        payload={"target_node": node.id,
                                 "patch_type": patch.patch_type.value,
                                 "reason": patch.reason[:200]},
                    ))

                    print(f"    🔄 {node.id}: 重规划 → {patch.patch_type.value} "
                          f"(重试 {replanner.get_retry_count(node.id)}/{replanner.max_retries_per_task})")
                else:
                    node.status = TaskStatus.FAILED
                    node.error_msg = vr.feedback
                    memory.record(result, vr)

                    await logger.log(TrajectoryEvent(
                        module="replanner", event_type=EventType.REPLAN_SKIP,
                        payload={"target_node": node.id, "reason": "超过最大重试"},
                    ))

                    print(f"    ❌ {node.id}: 失败且无法重试 — {vr.feedback[:80]}")

            # 统计工具调用
            total_tool_calls += len(result.tool_calls)

            # 打印结果
            icon = "✅" if vr.pass_ else "🔄" if patch else "❌"
            print(f"    {icon} {node.id} [{node.spec.name}] "
                  f"score={vr.score:.1f}, {len(result.evidence)}证据, "
                  f"{len(result.tool_calls)}工具, {result.elapsed:.2f}s")

        # 检查是否需要触发 L1→L2 压缩
        memory.auto_compress()

    elapsed = time.time() - start_time

    stats = {
        "rounds": round_num,
        "total_elapsed": elapsed,
        "total_tool_calls": total_tool_calls,
        "total_replans": total_replans,
        "success_count": plan.success_count(),
        "total_count": plan.total_count(),
    }

    return results, verification_results, stats


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Stage 3: 记忆总结                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def stage3_memory_summary(memory: HierarchicalMemory, user_query: str,
                          logger: TrajectoryLogger):
    """L1 → L2 语义压缩。"""
    print_header("Stage 3: 记忆总结")

    stats = memory.get_stats()
    print_status("memory", f"L1: {stats['l1_count']} 条, "
                 f"L2: {stats['l2_count']} 条摘要, "
                 f"重规划: {stats['replan_count']} 次")

    # 压缩 L1 → L2
    if memory.l1.count > 0:
        summary = memory.compress(user_query)
        # 同步记录 compress 事件
        try:
            logger.log_nowait(TrajectoryEvent(
                module="memory", event_type=EventType.MEMORY_COMPRESS,
                payload={"layer": "L1→L2", "num_items": memory.l2.count},
            ))
        except Exception:
            pass
        print_status("compress", f"L1 → L2: {memory.l2.count} 条摘要")
    else:
        print_status("compress", "L1 已空，无需压缩")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Stage 4: 最终报告                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def stage4_generate_report(
    user_query: str,
    plan: PlanGraph,
    results: dict[str, StepResult],
    verification_results: dict[str, VerificationResult],
    memory: HierarchicalMemory,
    stats: dict,
    plan_elapsed: float,
    session_id: str,
    llm_client=None,
):
    """用 Writer 合成自然语言研究报告。"""
    print_header("Stage 4: 最终报告")

    ctx = memory.get_context()
    writer_mode = "llm" if llm_client is not None else "template"
    writer = Writer(mode=writer_mode, llm_client=llm_client,
                    config=WriterConfig(export_dir="summaries"))

    # 生成两份报告: final_answer.md (用户) + debug_report.md (开发者)
    final_path, debug_path = await writer.write_reports(
        query=user_query, session_id=session_id,
        plan=plan, results=results, verifications=verification_results,
        memory_ctx=ctx,
        stats=stats,
    )

    report_text = Path(final_path).read_text(encoding="utf-8")
    debug_text = Path(debug_path).read_text(encoding="utf-8")

    print_status("report", f"final_answer: {final_path} ({len(report_text)} 字符)")
    print_status("report", f"debug_report: {debug_path} ({len(debug_text)} 字符)")
    print(f"\n{report_text[:1500]}")

    return report_text


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Stage 5: 统计输出                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def stage5_statistics(
    plan: PlanGraph,
    results: dict[str, StepResult],
    verification_results: dict[str, VerificationResult],
    tool_manager: ToolManager,
    memory: HierarchicalMemory,
    logger: TrajectoryLogger,
    stats: dict,
    plan_elapsed: float,
    session_id: str,
):
    """打印最终统计面板。"""
    print_header("Stage 5: 统计面板")

    # 任务统计
    total = stats['total_count']
    success = stats['success_count']
    failed = total - success
    rate = success / max(total, 1) * 100

    print(f"  📋 任务完成: {success}/{total} ({rate:.0f}%)")
    if failed > 0:
        print(f"     ❌ {failed} 个失败")
    print(f"  🔄 执行轮次: {stats['rounds']}")
    print(f"  🔧 工具调用: {stats['total_tool_calls']}")
    print(f"  ♻️  重规划: {stats['total_replans']}")
    print(f"  ⏱️  总耗时: {stats['total_elapsed']:.1f}s (规划: {plan_elapsed:.1f}s)")

    # Token 消耗
    total_tokens = sum(r.tokens_used for r in results.values())
    print(f"  💰 Token 消耗: {total_tokens}")

    # 验证分数分布
    if verification_results:
        scores = [vr.score for vr in verification_results.values()]
        avg_score = sum(scores) / len(scores)
        pass_count = sum(1 for vr in verification_results.values() if vr.pass_)
        print(f"  ✅ 验证通过: {pass_count}/{len(verification_results)} "
              f"(平均分: {avg_score:.2f})")

    # 证据汇总
    total_evidence = sum(len(r.evidence) for r in results.values())
    print(f"  📄 收集证据: {total_evidence} 条")

    # 工具调用统计
    print(f"\n  ── 工具调用明细 ──")
    for tool_name, tool_stats in tool_manager.get_all_stats().items():
        if tool_stats.total_calls > 0:
            print(f"    {tool_name}: {tool_stats.total_calls}次, "
                  f"成功={tool_stats.success_calls}, "
                  f"失败={tool_stats.failure_calls}, "
                  f"超时={tool_stats.timeout_calls}")

    # 记忆统计
    mem_stats = memory.get_stats()
    print(f"\n  ── 记忆系统 ──")
    print(f"    L1: {mem_stats['l1_count']}条 ({mem_stats['l1_usage']}用量)")
    print(f"    L2: {mem_stats['l2_count']}条摘要")
    print(f"    L3: {mem_stats['l3_count']}条归档")

    # 轨迹日志
    if logger.session:
        print(f"\n  ── 轨迹日志 ──")
        print(f"    Session: {session_id}")
        print(f"    事件数: {len(logger.session.events)}")
        print(f"    文件: {logger.output_dir / f'{session_id}.jsonl'}")

    # 总结
    print(f"\n  ═══════════════════════════════════════════════")
    if rate >= 80:
        print(f"  🎉 研究任务成功完成！")
    elif rate >= 40:
        print(f"  ⚠️  部分任务失败，但管道未中断。")
    else:
        print(f"  ❌ 多数任务失败，建议调整工具配置或重试策略。")
    print(f"  ═══════════════════════════════════════════════")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  主入口                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def run_pipeline(
    user_query: str,
    use_llm: bool = False,
    semaphore_limit: int = 3,
):
    """运行完整研究管道。

    Args:
        user_query: 研究问题。
        use_llm: 是否使用 LLM 驱动规划。
        semaphore_limit: 最大并发 Worker 数。
    """
    # ── Stage 0 ──
    cfg, llm_client, tool_manager, memory, logger = setup_infrastructure(use_llm)

    # 启动轨迹日志会话（使用 logger 生成的 session_id）
    session_id = await logger.start_session(user_query)

    print(f"\n{'='*70}")
    print(f"  HorizonRL-Agent v1 — 多 Agent 研究管道")
    print(f"{'='*70}")
    print(f"  Session: {session_id}")
    print(f"  问题: {user_query}")
    print(f"  模式: {'LLM 驱动' if use_llm else '离线 (模板+模拟工具)'}")
    print(f"  并发: {semaphore_limit} workers")
    print(f"{'='*70}")

    try:
        # ── Stage 1: 规划 ──
        plan, plan_elapsed = await stage1_planning(user_query, llm_client, logger)

        # ── Stage 2: 执行循环 ──
        results, vr_results, stats = await stage2_execution_loop(
            plan, tool_manager, memory, logger, semaphore_limit,
        )

        # ── Stage 3: 记忆总结 ──
        stage3_memory_summary(memory, user_query, logger)

        # ── Stage 4: 报告 ──
        await stage4_generate_report(
            user_query, plan, results, vr_results,
            memory, stats, plan_elapsed, session_id,
            llm_client=llm_client,
        )

        # ── Stage 5: 统计 ──
        stage5_statistics(
            plan, results, vr_results, tool_manager,
            memory, logger, stats, plan_elapsed, session_id,
        )

        # 结束日志会话
        success = stats['success_count'] == stats['total_count']
        await logger.end_session(success=success)

    except Exception as e:
        print(f"\n  ❌ Pipeline 异常: {e}")
        import traceback
        traceback.print_exc()
        await logger.end_session(success=False)
        raise


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  工具函数                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def print_header(title: str):
    """打印阶段标题。"""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def print_status(label: str, message: str, level: str = "info"):
    """打印状态行。"""
    icons = {"info": "  📌", "warn": "  ⚠️", "error": "  ❌", "ok": "  ✅"}
    icon = icons.get(level, icons["info"])
    print(f"{icon} [{label}] {message}")


def _plan_to_summary(plan: PlanGraph) -> str:
    """PlanGraph 的简要文本表示。"""
    lines = []
    for node in plan.nodes.values():
        deps = ",".join(node.depends_on) if node.depends_on else "root"
        lines.append(f"{node.id}({deps})")
    return "; ".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CLI 入口                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="HorizonRL-Agent v1 — 多 Agent 研究管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python examples/04_multi_agent_research.py
  python examples/04_multi_agent_research.py --llm
  python examples/04_multi_agent_research.py "量子计算对密码学的影响"
  python examples/04_multi_agent_research.py --workers 5 --llm
        """,
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="研究问题（默认: Transformer 多头注意力机制的最新进展）",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="启用 LLM 驱动模式（需配置 API Key）",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="最大并发 Worker 数（默认: 3）",
    )

    args = parser.parse_args()

    user_query = args.query or "Transformer 模型中多头注意力机制的工作原理与最新改进"
    asyncio.run(run_pipeline(
        user_query=user_query,
        use_llm=args.llm,
        semaphore_limit=args.workers,
    ))


if __name__ == "__main__":
    main()
