"""
=======================================================================
07_benchmark.py — HorizonRL-Agent 标准任务评测框架
=======================================================================

20 个标准问题，5 个类别，每问题跑 3 次取平均。

运行方式:
    python examples/07_benchmark.py                  # 全部 (mock模式)
    python examples/07_benchmark.py --category 概念原理  # 单类别
    python examples/07_benchmark.py --rounds 5       # 每问题5次
    python examples/07_benchmark.py --llm            # LLM模式

输出:
    summaries/benchmark_report.md  — 详细评测报告
    summaries/benchmark_scores.json — 原始分数数据
"""

from __future__ import annotations

import asyncio, sys, time, json
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from horizonrl.schemas.task import UserTask, TaskStatus
from horizonrl.agent.planner import Planner, LLMPlanner
from horizonrl.agent.worker import AgentWorker
from horizonrl.agent.verifier import Verifier
from horizonrl.agent.replanner import Replanner
from horizonrl.tools.manager import ToolManager
from horizonrl.tools.mock import register_mock_tools
from horizonrl.memory.hierarchical_memory import HierarchicalMemory
from horizonrl.config.settings import load_config, RootConfig


# ─── 标准测试问题集 (20题, 5类) ──────────────────────────────────────────────

BENCHMARK = {
    "事实知识": [
        "什么是Python的GIL(全局解释器锁)",
        "Transformer模型中的位置编码有什么作用",
        "量子计算中量子比特和经典比特的区别",
        "什么是RESTful API及其设计原则",
    ],
    "概念原理": [
        "Transformer多头注意力机制的工作原理",
        "Python asyncio事件循环的运行机制",
        "卷积神经网络中卷积操作的数学原理",
        "数据库索引的B+树结构及其查询优化原理",
    ],
    "技术对比": [
        "对比RNN、LSTM和GRU在处理长序列时的优劣",
        "PyTorch和TensorFlow在易用性和性能上的对比",
        "关系型数据库和非关系型数据库的适用场景对比",
        "HTTP/1.1和HTTP/2的主要区别和性能提升",
    ],
    "代码实践": [
        "Python中多线程和多进程的性能差异及适用场景",
        "如何使用Python的contextmanager实现资源管理",
        "Docker容器化部署相比传统部署的优势",
        "Git中rebase和merge的区别及最佳实践",
    ],
    "综述前沿": [
        "2024-2025年大语言模型的最新架构进展",
        "视觉Transformer(ViT)在计算机视觉中的发展现状",
        "强化学习在LLM对齐中的最新应用(RLHF/DPO)",
        "边缘计算和联邦学习的最新发展趋势",
    ],
}


# ─── 评测指标 ────────────────────────────────────────────────────────────────

@dataclass
class CategoryScores:
    """一个类别下的累计评测分数。"""
    category: str = ""
    questions: int = 0
    total_runs: int = 0
    passed_runs: int = 0
    total_score: float = 0.0
    total_tool_calls: int = 0
    total_replans: int = 0
    total_evidence: int = 0
    total_elapsed: float = 0.0
    total_tasks: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed_runs / max(self.total_runs, 1)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.total_runs, 1)

    @property
    def avg_tool_calls(self) -> float:
        return self.total_tool_calls / max(self.total_runs, 1)

    @property
    def avg_elapsed(self) -> float:
        return self.total_elapsed / max(self.total_runs, 1)

    @property
    def task_success_rate(self) -> float:
        """单个子任务的成功率 (不是整题 pass rate)。"""
        passed = self.total_tasks - (self.total_runs - self.passed_runs) * 2  # 估计
        return self.passed_runs / max(self.total_runs, 1)


# ─── 评测运行器 ──────────────────────────────────────────────────────────────

async def run_one_query(query: str, llm_client=None) -> dict:
    """对单个问题运行一次管道，返回原始分数。"""
    t0 = time.time()
    mgr = ToolManager()
    register_mock_tools(mgr)
    memory = HierarchicalMemory()
    verifier = Verifier(mode="rule")
    replanner = Replanner(max_retries_per_task=3, max_total_replans=5)
    sem = asyncio.Semaphore(3)

    # ── 规划 ──
    if llm_client:
        planner = LLMPlanner(llm_client)
        plan = await planner.plan(UserTask(description=query, max_steps=20))
    else:
        plan = Planner().plan(UserTask(description=query, max_steps=20))

    # ── 执行 ──
    results, scores, tool_calls_total, replans_total, evidence_total = {}, [], 0, 0, 0

    while plan.has_pending_work():
        for node in plan.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            if all(plan.nodes[d].status == TaskStatus.SUCCESS for d in node.depends_on):
                node.status = TaskStatus.READY

        ready = plan.get_ready_nodes()
        if not ready:
            break

        async def exec_one(node):
            node.status = TaskStatus.RUNNING
            async with sem:
                worker = AgentWorker(worker_id=f"wrk_{node.id}", tool_manager=mgr)
                return node, await worker.execute(node.spec)

        batch = await asyncio.gather(*[exec_one(n) for n in ready])

        for node, result in batch:
            results[result.task_id] = result
            vr = await verifier.verify(result, node.spec)
            scores.append(vr.score)
            tool_calls_total += len(result.tool_calls)
            evidence_total += len(result.evidence)

            if vr.pass_:
                node.status = TaskStatus.SUCCESS
                memory.record(result, vr)
            else:
                patch = replanner.replan(vr, plan, node.id)
                if patch is not None:
                    replanner.apply_patch(plan, patch)
                    replans_total += 1
                    memory.record_replan()
                else:
                    node.status = TaskStatus.FAILED
                    memory.record(result, vr)

        memory.auto_compress()

    elapsed = time.time() - t0
    all_passed = plan.success_count() == plan.total_count()

    return {
        "passed": all_passed,
        "success_count": plan.success_count(),
        "total_count": plan.total_count(),
        "avg_score": sum(scores) / max(len(scores), 1),
        "tool_calls": tool_calls_total,
        "replans": replans_total,
        "evidence": evidence_total,
        "elapsed": elapsed,
    }


async def run_benchmark(
    questions: dict[str, list[str]],
    rounds: int = 3,
    llm_client=None,
    category_filter: str | None = None,
) -> dict[str, CategoryScores]:
    """运行完整 benchmark。

    Returns:
        {category: CategoryScores}
    """
    results: dict[str, CategoryScores] = {}

    categories = {category_filter: questions[category_filter]} if category_filter else questions

    for cat, queries in categories.items():
        scores = CategoryScores(category=cat, questions=len(queries))

        print(f"\n{'='*60}")
        print(f"  {cat} ({len(queries)} 题 × {rounds} 轮)")
        print(f"{'='*60}")

        for qi, query in enumerate(queries):
            cat_passes, cat_scores = [], []

            for r in range(rounds):
                s = await run_one_query(query, llm_client)
                scores.total_runs += 1
                scores.total_score += s["avg_score"]
                scores.total_tool_calls += s["tool_calls"]
                scores.total_replans += s["replans"]
                scores.total_evidence += s["evidence"]
                scores.total_elapsed += s["elapsed"]
                scores.total_tasks += s["total_count"]
                if s["passed"]:
                    scores.passed_runs += 1
                cat_passes.append(s["passed"])
                cat_scores.append(s["avg_score"])

            avg_pass = sum(cat_passes) / len(cat_passes)
            avg_s = sum(cat_scores) / len(cat_scores)
            print(f"  [{qi+1}/{len(queries)}] {query[:50]:50s} | "
                  f"pass={avg_pass:.0%} | score={avg_s:.2f}")

        results[cat] = scores

    return results


# ─── 报告生成 ────────────────────────────────────────────────────────────────


def generate_benchmark_report(
    results: dict[str, CategoryScores],
    rounds: int,
    output_dir: str = "summaries",
) -> str:
    """生成 Benchmark 评测报告 (Markdown)。"""
    lines = [
        "# HorizonRL-Agent Benchmark 评测报告",
        "",
        f"**测试日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**问题总数**: {sum(s.questions for s in results.values())}",
        f"**每问题轮次**: {rounds}",
        f"**总运行次数**: {sum(s.total_runs for s in results.values())}",
        "",
        "---",
        "",
        "## 一、综合评分",
        "",
        "| 类别 | 题目 | Pass Rate | 平均评分 | 工具调用 | 重规划 | 证据 | 耗时 |",
        "|------|------|-----------|----------|----------|--------|------|------|",
    ]

    total_all = CategoryScores()
    for cat, s in results.items():
        lines.append(
            f"| {cat} | {s.questions} | {s.pass_rate:.0%} | {s.avg_score:.2f} | "
            f"{s.avg_tool_calls:.1f} | {s.total_replans} | {s.total_evidence} | "
            f"{s.avg_elapsed:.1f}s |"
        )
        total_all.questions += s.questions
        total_all.total_runs += s.total_runs
        total_all.passed_runs += s.passed_runs
        total_all.total_score += s.total_score
        total_all.total_tool_calls += s.total_tool_calls
        total_all.total_replans += s.total_replans
        total_all.total_evidence += s.total_evidence
        total_all.total_elapsed += s.total_elapsed

    # 总计行
    lines.append(
        f"| **总计** | {total_all.questions} | {total_all.pass_rate:.0%} | "
        f"{total_all.avg_score:.2f} | {total_all.avg_tool_calls:.1f} | "
        f"{total_all.total_replans} | {total_all.total_evidence} | "
        f"{total_all.total_elapsed:.1f}s |"
    )

    lines += [
        "",
        "---",
        "",
        "## 二、各类别详细分析",
        "",
    ]

    for cat, s in results.items():
        lines.append(f"### {cat}")
        lines.append("")
        lines.append(f"- **通过率**: {s.pass_rate:.0%}")
        lines.append(f"- **平均评分**: {s.avg_score:.2f}")
        lines.append(f"- **工具效率**: 平均 {s.avg_tool_calls:.1f} 次调用/题")
        lines.append(f"- **重规划**: {s.total_replans} 次 (共 {s.total_runs} 次运行)")
        lines.append(f"- **总证据**: {s.total_evidence} 条")
        lines.append(f"- **平均耗时**: {s.avg_elapsed:.1f}s")
        lines.append("")

    lines += [
        "---",
        "",
        "## 三、测试问题清单",
        "",
    ]

    for cat, queries in BENCHMARK.items():
        if cat not in results:
            continue
        lines.append(f"### {cat}")
        for i, q in enumerate(queries):
            lines.append(f"{i+1}. {q}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 四、评测说明",
        "",
        "- **Pass Rate**: 所有子任务通过 Verifier 验证的比例",
        "- **平均评分**: Verifier 对每个子任务的质量评分 (0.0-1.0)",
        "- **工具效率**: 每次运行的工具调用次数 (越低越高效)",
        "- **重规划**: Replanner 介入修复的次数",
        "- **Mock 模式**: 本测试使用模拟工具，确保可重复性",
        "",
        "---",
        "*本报告由 HorizonRL-Agent Benchmark 引擎自动生成*",
    ]

    report = "\n".join(lines)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/benchmark_report.md").write_text(report, encoding="utf-8")

    # 保存原始分数
    raw = {
        cat: {
            "pass_rate": s.pass_rate,
            "avg_score": s.avg_score,
            "avg_tool_calls": s.avg_tool_calls,
            "total_replans": s.total_replans,
            "total_evidence": s.total_evidence,
            "avg_elapsed": s.avg_elapsed,
        }
        for cat, s in results.items()
    }
    Path(f"{output_dir}/benchmark_scores.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return report


# ─── 主入口 ──────────────────────────────────────────────────────────────────


async def main(category: str | None = None, rounds: int = 3, use_llm: bool = False):
    t0 = time.time()

    llm_client = None
    if use_llm:
        try:
            cfg = load_config(Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None)
            if cfg.llm.api_key:
                from horizonrl.llm.client import LLMClient
                llm_client = LLMClient(cfg.llm)
        except Exception as e:
            print(f"LLM 不可用: {e}")

    total_questions = sum(len(v) for k, v in BENCHMARK.items()
                         if not category or k == category)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  HorizonRL-Agent Benchmark 评测                              ║
║  题目: {total_questions} 题 × {rounds} 轮 = {total_questions * rounds} 次运行           ║
║  模式: {'LLM' if llm_client else 'Mock (模板)'}                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    results = await run_benchmark(BENCHMARK, rounds, llm_client, category)
    report = generate_benchmark_report(results, rounds)

    print(f"\n{'='*60}")
    print(f"  Benchmark 完成! 总耗时: {time.time() - t0:.1f}s")
    print(f"  报告: summaries/benchmark_report.md")
    print(f"  分数: summaries/benchmark_scores.json")
    print(f"{'='*60}")
    print(f"\n{report[:1200]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HorizonRL-Agent Benchmark")
    parser.add_argument("--category", type=str, default=None, help="测试类别 (事实知识/概念原理/...)")
    parser.add_argument("--rounds", type=int, default=3, help="每问题运行次数")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 模式")
    args = parser.parse_args()
    asyncio.run(main(args.category, args.rounds, args.llm))
