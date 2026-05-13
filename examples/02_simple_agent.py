"""
═══════════════════════════════════════════════════════════════════════════════
02_simple_agent.py — HorizonRL-Agent 最简完整链路 Demo
═══════════════════════════════════════════════════════════════════════════════

跑通一条完整的研究任务流水线：
    UserTask → Planner → PlanGraph → Worker×N → ToolManager → Tools → StepResult

── 运行方式 ──
    cd HorizonRL-Agent
    python examples/02_simple_agent.py

── 预期输出 ──
    1. 配置加载
    2. 工具注册（Web Search / Arxiv / Code Execution）
    3. Planner 将任务拆解为 5 个 TaskSpec
    4. 打印 PlanGraph 的 DAG 结构
    5. 按依赖顺序并发执行每个子任务
    6. 每个任务完成后打印 StepResult
    7. 最终汇总：成功/失败统计、工具调用统计、Token 消耗

── 不需要任何 API Key ──
    如果没有安装外部工具库（duckduckgo-search / arxiv），
    会使用内置模拟数据跑通流程，展示完整的管道能力。

── 对应架构位置 ──
    本文件是 DEVELOP_PLAN.md Step 7 的交付物，
    也是 GitHub Public Beta 的第一个可运行 Demo。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Windows: 强制 UTF-8 输出，避免 emoji 编码报错
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 确保 src/ 在 sys.path 中（不需要 pip install 也能跑）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from horizonrl.schemas.task import UserTask, TaskSpec, TaskStatus
from horizonrl.schemas.result import StepResult
from horizonrl.agent.planner import Planner
from horizonrl.agent.worker import AgentWorker, execute_workers
from horizonrl.tools.manager import ToolManager, ToolCallRequest


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Step 0: 准备工具                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 注册三个工具到 ToolManager。如果外部库不可用，注册模拟工具
# 确保流程能从头跑到尾，不依赖外部 API。


def setup_tools() -> ToolManager:
    """创建 ToolManager 并注册所有可用工具。

    注册策略：
      - duckduckgo-search 可用 → 使用真实 WebSearchTool
      - 不可用 → 使用内置 MockWebSearch
      - arxiv 包可用 → 使用真实 ArxivSearchTool
      - 不可用 → 使用内置 MockArxivSearch
      - CodeExecutionTool 始终可用（纯 Python）

    Returns:
        配置好的 ToolManager 实例。
    """
    mgr = ToolManager()

    # ── Web Search ────────────────────────────────────────────────────
    try:
        from horizonrl.tools.web_search import WebSearchTool
        web_tool = WebSearchTool()
        print("  ✅ WebSearchTool (DuckDuckGo)")
    except Exception:
        web_tool = _MockWebSearch()
        print("  ⚠️  WebSearchTool → 模拟模式 (pip install duckduckgo-search)")

    mgr.register("web_search", web_tool)

    # ── Arxiv Search ──────────────────────────────────────────────────
    try:
        from horizonrl.tools.arxiv_search import ArxivSearchTool
        arxiv_tool = ArxivSearchTool(max_results=5)
        print("  ✅ ArxivSearchTool")
    except Exception:
        arxiv_tool = _MockArxivSearch()
        print("  ⚠️  ArxivSearchTool → 模拟模式 (pip install arxiv)")

    mgr.register("arxiv_search", arxiv_tool)

    # ── Code Execution ────────────────────────────────────────────────
    try:
        from horizonrl.tools.code_execution import CodeExecutionTool
        code_tool = CodeExecutionTool(timeout=10.0)
        print("  ✅ CodeExecutionTool")
    except Exception:
        code_tool = _MockCodeExec()
        print("  ⚠️  CodeExecutionTool → 模拟模式")

    mgr.register("code_execution", code_tool)

    return mgr


# ─── 模拟工具（统一使用 tools/mock.py）──────────────────────────────────
from horizonrl.tools.mock import MockWebSearch as _MockWebSearch
from horizonrl.tools.mock import MockArxivSearch as _MockArxivSearch
from horizonrl.tools.mock import MockCodeExecution as _MockCodeExec


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Step 1: 加载配置                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def load_config():
    """加载配置，优先开发环境覆盖。"""
    from horizonrl.config.settings import load_config as _load

    dev_path = Path("configs/dev.yaml")
    if dev_path.exists():
        cfg = _load(dev_path)
        print(f"  📋 配置: dev.yaml (debug={cfg.debug})")
    else:
        cfg = _load()
        print(f"  📋 配置: 默认")
    return cfg


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Step 2: 任务规划                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def plan_task(user_query: str) -> tuple:
    """使用 Planner 将用户任务拆解为 PlanGraph。

    Args:
        user_query: 用户的研究问题。

    Returns:
        (Planner, UserTask, PlanGraph) 三元组。
    """
    planner = Planner()
    task = UserTask(
        description=user_query,
        max_steps=20,
        max_tokens=30_000,
        output_format="markdown",
    )
    plan = planner.plan(task)
    return planner, task, plan


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Step 3: 执行任务                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def execute_plan(
    plan,
    tool_manager: ToolManager,
    semaphore_limit: int = 3,
) -> dict:
    """按 DAG 依赖顺序执行 PlanGraph 中的所有任务。

    执行策略：
      1. 找出所有依赖已满足的 READY 节点
      2. 并发执行（受 Semaphore 限制）
      3. 完成后标记 SUCCESS/FAILED
      4. 重复直到所有节点处于终态

    Args:
        plan: PlanGraph 实例。
        tool_manager: 统一工具管理器。
        semaphore_limit: 最多同时执行的 Worker 数。

    Returns:
        {task_id: StepResult} 映射。
    """
    results: dict[str, StepResult] = {}
    sem = asyncio.Semaphore(semaphore_limit)

    round_num = 0
    while plan.has_pending_work():
        round_num += 1

        # 将依赖已满足的 PENDING 节点标记为 READY
        for node in plan.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            deps_satisfied = all(
                plan.nodes[dep_id].status == TaskStatus.SUCCESS
                for dep_id in node.depends_on
            )
            if deps_satisfied:
                node.status = TaskStatus.READY

        ready_nodes = plan.get_ready_nodes()
        if not ready_nodes:
            # 检查是否有死锁（剩余节点无法推进）
            pending = [
                n for n in plan.nodes.values()
                if n.status in (TaskStatus.PENDING, TaskStatus.READY)
            ]
            if pending:
                print(f"\n  ⚠️  死锁: {len(pending)} 个节点无法推进")
                for n in pending:
                    unmet = [
                        d for d in n.depends_on
                        if plan.nodes[d].status != TaskStatus.SUCCESS
                    ]
                    print(f"     - {n.id}: 等待 {unmet}")
            break

        print(f"\n── 第 {round_num} 轮: 并发执行 {len(ready_nodes)} 个任务 ──")

        async def run_one(node):
            node.status = TaskStatus.RUNNING
            async with sem:
                worker = AgentWorker(
                    worker_id=f"worker_{node.id}",
                    tool_manager=tool_manager,
                )
                result = await worker.execute(node.spec)
                node.finished_at = time.time()
                if result.success:
                    node.status = TaskStatus.SUCCESS
                else:
                    node.status = TaskStatus.FAILED
                    node.error_msg = result.error
                return result

        batch_results = await asyncio.gather(*[run_one(n) for n in ready_nodes])

        for r in batch_results:
            results[r.task_id] = r

        # 打印本轮结果
        for r in batch_results:
            status_icon = "✅" if r.success else "❌"
            evidence_count = len(r.evidence)
            tool_count = len(r.tool_calls)
            print(f"  {status_icon} {r.task_id}: "
                  f"成功={r.success}, 证据={evidence_count}条, "
                  f"工具调用={tool_count}次, 耗时={r.elapsed:.2f}s")

    return results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Step 4: 汇总报告                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def print_summary(plan, results: dict, tool_manager: ToolManager) -> None:
    """打印最终执行摘要。"""
    print("\n" + "=" * 65)
    print("📊 最终执行摘要")
    print("=" * 65)

    # 任务统计
    total = plan.total_count()
    success_count = sum(
        1 for tid, r in results.items() if r.success
    )
    failed_count = total - success_count

    print(f"\n  任务完成: {success_count}/{total} 成功"
          f"{' ✅' if failed_count == 0 else f' (❌ {failed_count} 失败)'}")

    # Token 消耗
    total_tokens = sum(r.tokens_used for r in results.values())
    print(f"  Token 消耗: {total_tokens}")

    # 总耗时
    total_time = sum(r.elapsed for r in results.values())
    print(f"  总耗时: {total_time:.2f}s")

    # 工具调用统计
    print(f"\n  工具调用统计:")
    for tool_name, stats in tool_manager.get_all_stats().items():
        if stats.total_calls > 0:
            print(f"    {tool_name}: {stats.total_calls} 次调用, "
                  f"{stats.success_calls} 成功, "
                  f"{stats.failure_calls} 失败, "
                  f"{stats.timeout_calls} 超时, "
                  f"平均 {stats.total_latency / max(stats.total_calls, 1):.2f}s")

    # 证据汇总
    total_evidence = sum(len(r.evidence) for r in results.values())
    print(f"\n  收集证据: {total_evidence} 条")

    # 各任务详情
    print(f"\n  各任务输出摘要:")
    for node in plan.nodes.values():
        result = results.get(node.spec.id)
        if result is None:
            continue
        icon = "✅" if result.success else "❌"
        preview = result.output[:120].replace("\n", " ")
        print(f"    {icon} [{node.spec.name}] {preview}...")

    print("\n" + "=" * 65)
    if failed_count == 0:
        print("🎉 所有任务执行成功！链路完整跑通。")
    else:
        print(f"⚠️  {failed_count} 个任务失败，详见上方错误信息。")
    print("=" * 65)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  main() — Demo 入口                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def main(user_query: str | None = None):
    """Demo 主函数：跑通 Planner → Worker → Tool 完整链路。

    Args:
        user_query: 研究问题。None 时使用默认示例。
    """
    if user_query is None:
        user_query = "Transformer 模型中多头注意力机制的原理与最新改进"

    print("=" * 65)
    print("🚀 HorizonRL-Agent 最简完整链路 Demo")
    print("=" * 65)
    print(f"\n📝 研究问题: {user_query}")

    # ── Step 0: 工具准备 ──
    print("\n── Step 0: 注册工具 ──")
    tool_manager = setup_tools()

    # ── Step 1: 配置加载 ──
    print("\n── Step 1: 加载配置 ──")
    try:
        cfg = load_config()
    except Exception as e:
        print(f"  ⚠️  配置加载失败 ({e})，使用默认值")
        from horizonrl.config.settings import RootConfig
        cfg = RootConfig()

    # ── Step 2: 任务规划 ──
    print("\n── Step 2: Planner 任务分解 ──")
    planner, user_task, plan = plan_task(user_query)

    print(f"  任务类型: {planner._classify_task(user_task)}")
    print(f"  拆解为 {plan.total_count()} 个 TaskSpec:")
    print(f"  DAG 结构:")
    for node in plan.nodes.values():
        deps = node.depends_on if node.depends_on else ["(无)"]
        dep_names = []
        for dep_id in node.depends_on:
            dep_node = plan.nodes.get(dep_id)
            dep_names.append(dep_node.spec.name if dep_node else dep_id)
        print(f"    [{node.id}] {node.spec.name}")
        print(f"        工具: {node.spec.tool_names or '(无需工具)'}")
        print(f"        依赖: {dep_names}")
        print(f"        优先级: {node.spec.priority.value}")

    print(f"\n  根节点 (可并行): {len(plan.root_ids)} 个")

    # ── Step 3: 并发执行 ──
    print("\n── Step 3: Worker 并发执行 ──")
    start_time = time.time()
    results = await execute_plan(plan, tool_manager, semaphore_limit=3)
    elapsed = time.time() - start_time

    # ── Step 4: 汇总 ──
    print_summary(plan, results, tool_manager)
    print(f"\n⏱️  端到端耗时: {elapsed:.2f}s")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  入口                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # 支持命令行传入自定义研究问题
    query = None
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"💡 自定义问题: {query}\n")

    asyncio.run(main(query))
