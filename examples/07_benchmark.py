"""
=======================================================================
07_benchmark.py — HorizonRL-Agent 标准任务评测框架 (v2: 全链路)
=======================================================================

v2 变更 (Day 6):
    - 任务从 JSONL 文件加载 (benchmarks/tasks.jsonl)
    - 走共享 research_service → ResearchOrchestrator 全链路
    - 结构化 Evaluator: mock_ratio / citation / trajectory 指标
    - 结果保存到 benchmark_results/run_YYYYmmdd_HHMMSS/

运行方式:
    python examples/07_benchmark.py                  # 全部20题, mock模式
    python examples/07_benchmark.py --category 概念原理  # 单类别
    python examples/07_benchmark.py --rounds 5       # 每问题5次
    python examples/07_benchmark.py --llm            # LLM模式
    python examples/07_benchmark.py --tasks benchmarks/tasks.jsonl  # 自定义任务集

输出:
    benchmark_results/run_*/report.md   — 评测报告
    benchmark_results/run_*/summary.json — 聚合指标
    benchmark_results/run_*/per_task_results.jsonl — 逐任务结果
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# 添加项目根目录以导入 benchmarks/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from horizonrl.config.settings import load_config, RootConfig
from horizonrl.services.research_service import run_research_session

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
DEFAULT_TASKS = BENCHMARK_DIR / "tasks.jsonl"


# ─── 主流程 ───────────────────────────────────────────────────────────────────


async def run_benchmark(
    tasks: list[dict],
    rounds: int = 3,
    llm_client=None,
    config: RootConfig | None = None,
    category_filter: str | None = None,
) -> tuple[list, object, str]:
    """运行完整 benchmark。

    每道题跑 rounds 次，通过共享 service 走全链路。

    Args:
        tasks: 任务列表 (从 JSONL 加载)。
        rounds: 每道题重复次数。
        llm_client: LLM 客户端，None = 模板模式。
        config: 全局配置。
        category_filter: 只跑指定类别。

    Returns:
        (results: list[RunResult], summary: BenchmarkSummary, report: str)
    """
    from benchmarks.evaluator import (
        evaluate_run,
        aggregate_results,
        generate_report,
        RunResult,
    )

    # 过滤类别
    if category_filter:
        tasks = [t for t in tasks if t.get("category") == category_filter]
        if not tasks:
            raise ValueError(f"未找到类别: {category_filter}")

    all_results: list[RunResult] = []
    total_runs = len(tasks) * rounds

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  HorizonRL-Agent Benchmark v2 (全链路)                       ║
║  题目: {len(tasks)} 题 × {rounds} 轮 = {total_runs} 次运行              ║
║  模式: {'LLM' if llm_client else 'Mock (模板)'}                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    for ti, task in enumerate(tasks):
        tid = task["id"]
        query = task["query"]
        cat = task.get("category", "")

        print(f"\n{'='*60}")
        print(f"  [{ti+1}/{len(tasks)}] {cat}/{tid}")
        print(f"  {query[:70]}")
        print(f"{'='*60}")

        for r in range(rounds):
            t0 = time.monotonic()
            label = f"  轮次 {r+1}/{rounds}"

            # ── 调用共享 Service (全链路) ──
            artifacts = await run_research_session(
                query=query,
                mode="deep",
                llm_client=llm_client,
                config=config,
                export_dir="reports",
                search_provider="mock",
                offline=True,
                semaphore_limit=3,
                max_iterations=10,
            )

            # ── 评测 ──
            result = evaluate_run(task, artifacts)
            all_results.append(result)

            icon = "✅" if result.passed else "❌"
            print(
                f"{label}: {icon} | "
                f"子任务 {result.success_tasks}/{result.total_tasks} | "
                f"工具 {result.tool_calls} | "
                f"重规划 {result.replans} | "
                f"Mock {result.mock_ratio:.0%} | "
                f"{result.runtime_ms/1000:.1f}s"
            )

    # ── 聚合 ──
    summary = aggregate_results(all_results)
    mode_label = "LLM" if llm_client else "Mock (模板)"
    report = generate_report(summary, all_results, len(tasks), rounds, mode_label)

    return all_results, summary, report


# ─── 主入口 ───────────────────────────────────────────────────────────────────


async def main(
    category: str | None = None,
    rounds: int = 3,
    use_llm: bool = False,
    tasks_path: str | None = None,
):
    t_start = time.time()

    # ── 加载配置 ──
    try:
        dev_path = Path("configs/dev.yaml")
        cfg = load_config(dev_path if dev_path.exists() else None)
    except Exception:
        cfg = RootConfig()

    # ── LLM ──
    llm_client = None
    if use_llm and cfg.llm.api_key:
        from horizonrl.llm.client import LLMClient
        llm_client = LLMClient(cfg.llm)

    # ── 加载任务 ──
    path = Path(tasks_path) if tasks_path else DEFAULT_TASKS
    if not path.exists():
        print(f"错误: 任务文件不存在: {path}")
        print("请确保 benchmarks/tasks.jsonl 存在")
        return

    from benchmarks.evaluator import load_tasks
    tasks = load_tasks(path)

    # ── 运行 ──
    results, summary, report = await run_benchmark(
        tasks=tasks,
        rounds=rounds,
        llm_client=llm_client,
        config=cfg,
        category_filter=category,
    )

    # ── 保存结果 ──
    from benchmarks.evaluator import save_results
    paths = save_results(results, summary, report)

    elapsed = time.time() - t_start

    print(f"""
{'='*60}
  Benchmark 完成! 总耗时: {elapsed:.1f}s
  通过率: {summary.overall_pass_rate:.1%}
  子任务成功率: {summary.avg_task_success_rate:.1%}
  Mock占比: {summary.avg_mock_ratio:.1%}
  报告: {paths['report']}
  摘要: {paths['summary']}
  结果: {paths['results']}
{'='*60}
""")

    # 预览
    print(report[:1500])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HorizonRL-Agent Benchmark v2 (全链路)"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="测试类别: 事实知识/概念原理/技术对比/代码实践/综述前沿",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="每问题运行次数 (默认: 3)",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="启用 LLM 模式",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help=f"自定义任务 JSONL 文件 (默认: {DEFAULT_TASKS})",
    )

    args = parser.parse_args()
    asyncio.run(main(args.category, args.rounds, args.llm, args.tasks))
