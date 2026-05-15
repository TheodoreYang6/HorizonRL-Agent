"""
Benchmark Evaluator — 结构化评测指标计算。

支持两层评估:
  - 规则评估: mock_ratio, task_success_rate, tool_calls, replans, runtime
  - 轨迹评估: num_steps, deadlock_count, timeout_count, avg_step_latency

使用方式:
    from benchmarks.evaluator import evaluate_run, aggregate_results

    result = evaluate_run(task, artifacts)
    summary = aggregate_results(all_results)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """单次运行评测结果。"""

    task_id: str = ""
    category: str = ""
    difficulty: str = ""
    query: str = ""
    passed: bool = False
    task_success_rate: float = 0.0
    total_tasks: int = 0
    success_tasks: int = 0
    tool_calls: int = 0
    replans: int = 0
    evidence_count: int = 0
    mock_ratio: float = 0.0
    runtime_ms: float = 0.0
    used_provider: str = ""
    has_report: bool = False
    has_error: bool = False
    error_msg: str = ""


@dataclass
class BenchmarkSummary:
    """一次 Benchmark 运行的聚合统计。"""

    total_tasks: int = 0
    total_runs: int = 0
    passed_runs: int = 0
    overall_pass_rate: float = 0.0
    avg_task_success_rate: float = 0.0
    avg_tool_calls: float = 0.0
    avg_replans: float = 0.0
    avg_mock_ratio: float = 0.0
    avg_runtime_ms: float = 0.0
    total_evidence: int = 0
    per_category: dict[str, dict] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)


def evaluate_run(task: dict, artifacts) -> RunResult:
    """对单次运行计算全部指标。

    Args:
        task: JSONL 任务条目 (dict with id, category, query, etc.)
        artifacts: SessionArtifacts from run_research_session()

    Returns:
        RunResult 包含全部指标。
    """
    stats = artifacts.stats if hasattr(artifacts, "stats") else getattr(artifacts, "stats", {})

    passed = not artifacts.error and stats.get("success_count", 0) == stats.get(
        "total_count", 1
    )

    total_tasks = stats.get("total_count", 0)
    success_tasks = stats.get("success_count", 0)
    task_sr = success_tasks / max(total_tasks, 1)

    return RunResult(
        task_id=task.get("id", ""),
        category=task.get("category", ""),
        difficulty=task.get("difficulty", "medium"),
        query=task.get("query", ""),
        passed=passed,
        task_success_rate=task_sr,
        total_tasks=total_tasks,
        success_tasks=success_tasks,
        tool_calls=artifacts.tool_calls_count,
        replans=stats.get("total_replans", 0),
        evidence_count=stats.get("total_evidence", 0),
        mock_ratio=getattr(artifacts, "mock_ratio", 1.0),
        runtime_ms=getattr(artifacts, "runtime_ms", 0.0),
        used_provider=getattr(artifacts, "used_search_provider", ""),
        has_report=bool(getattr(artifacts, "final_answer_text", "")),
        has_error=bool(artifacts.error),
        error_msg=getattr(artifacts, "error", ""),
    )


def aggregate_results(results: list[RunResult]) -> BenchmarkSummary:
    """聚合多次运行的 RunResult 为 BenchmarkSummary。

    Args:
        results: 所有 RunResult 列表。

    Returns:
        BenchmarkSummary 聚合统计。
    """
    if not results:
        return BenchmarkSummary()

    n = len(results)
    passed = sum(1 for r in results if r.passed)
    per_cat: dict[str, dict] = {}

    for r in results:
        cat = r.category or "未分类"
        if cat not in per_cat:
            per_cat[cat] = {
                "total": 0,
                "passed": 0,
                "avg_task_sr": 0.0,
                "avg_tool_calls": 0.0,
                "avg_mock_ratio": 0.0,
                "avg_runtime_ms": 0.0,
            }
        c = per_cat[cat]
        c["total"] += 1
        c["passed"] += int(r.passed)
        c["avg_task_sr"] += r.task_success_rate
        c["avg_tool_calls"] += r.tool_calls
        c["avg_mock_ratio"] += r.mock_ratio
        c["avg_runtime_ms"] += r.runtime_ms

    for c in per_cat.values():
        t = max(c["total"], 1)
        c["pass_rate"] = c["passed"] / t
        c["avg_task_sr"] = c["avg_task_sr"] / t
        c["avg_tool_calls"] = c["avg_tool_calls"] / t
        c["avg_mock_ratio"] = c["avg_mock_ratio"] / t
        c["avg_runtime_ms"] = c["avg_runtime_ms"] / t

    failures = [
        {"task_id": r.task_id, "query": r.query[:80], "error": r.error_msg}
        for r in results if r.has_error
    ]

    return BenchmarkSummary(
        total_tasks=len({r.task_id for r in results}),
        total_runs=n,
        passed_runs=passed,
        overall_pass_rate=passed / n,
        avg_task_success_rate=sum(r.task_success_rate for r in results) / n,
        avg_tool_calls=sum(r.tool_calls for r in results) / n,
        avg_replans=sum(r.replans for r in results) / n,
        avg_mock_ratio=sum(r.mock_ratio for r in results) / n,
        avg_runtime_ms=sum(r.runtime_ms for r in results) / n,
        total_evidence=sum(r.evidence_count for r in results),
        per_category=per_cat,
        failures=failures,
    )


def generate_report(
    summary: BenchmarkSummary,
    results: list[RunResult],
    task_count: int,
    rounds: int,
    mode: str = "mock",
) -> str:
    """生成 Benchmark 评测报告 (Markdown)。

    Returns:
        完整的 Markdown 报告文本。
    """
    lines = [
        "# HorizonRL-Agent Benchmark 评测报告",
        "",
        f"**测试日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**题目总数**: {task_count}",
        f"**每题目轮次**: {rounds}",
        f"**总运行次数**: {summary.total_runs}",
        f"**模式**: {mode}",
        "",
        "---",
        "",
        "## 一、综合指标",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 整体通过率 | {summary.overall_pass_rate:.1%} |",
        f"| 子任务成功率 | {summary.avg_task_success_rate:.1%} |",
        f"| 平均工具调用/次 | {summary.avg_tool_calls:.1f} |",
        f"| 平均重规划/次 | {summary.avg_replans:.1f} |",
        f"| Mock 数据占比 | {summary.avg_mock_ratio:.1%} |",
        f"| 平均耗时 | {summary.avg_runtime_ms/1000:.1f}s |",
        f"| 总证据数 | {summary.total_evidence} |",
        f"| 失败运行 | {len(summary.failures)} |",
        "",
        "---",
        "",
        "## 二、各类别评分",
        "",
        "| 类别 | 运行 | 通过率 | 子任务SR | 工具调用 | Mock% | 耗时 |",
        "|------|------|--------|----------|----------|-------|------|",
    ]

    for cat, c in sorted(summary.per_category.items()):
        lines.append(
            f"| {cat} | {c['total']} | {c['pass_rate']:.0%} | "
            f"{c['avg_task_sr']:.0%} | {c['avg_tool_calls']:.1f} | "
            f"{c['avg_mock_ratio']:.0%} | {c['avg_runtime_ms']/1000:.1f}s |"
        )

    lines += [
        "",
        "---",
        "",
        "## 三、难度评分",
        "",
    ]

    by_difficulty: dict[str, dict] = {}
    for r in results:
        d = r.difficulty or "medium"
        if d not in by_difficulty:
            by_difficulty[d] = {"total": 0, "passed": 0, "avg_sr": 0.0}
        by_difficulty[d]["total"] += 1
        by_difficulty[d]["passed"] += int(r.passed)
        by_difficulty[d]["avg_sr"] += r.task_success_rate

    lines.append("| 难度 | 运行 | 通过率 | 子任务SR |")
    lines.append("|------|------|--------|----------|")
    for d in ("easy", "medium", "hard"):
        if d in by_difficulty:
            v = by_difficulty[d]
            t = max(v["total"], 1)
            lines.append(
                f"| {d} | {v['total']} | {v['passed']/t:.0%} | "
                f"{v['avg_sr']/t:.0%} |"
            )

    if summary.failures:
        lines += [
            "",
            "---",
            "",
            "## 四、失败记录",
            "",
        ]
        for f in summary.failures[:10]:
            lines.append(f"- **{f['task_id']}**: {f['query']} — `{f['error'][:100]}`")

    lines += [
        "",
        "---",
        "",
        "## 五、评测说明",
        "",
        "- **通过率**: 所有子任务通过 Verifier 验证的运行占比",
        "- **子任务成功率 (SR)**: 单次运行中成功子任务数/总子任务数",
        "- **Mock 数据占比**: Mock 证据数/总证据数 (1.0 = 全Mock, 0.0 = 全真实)",
        "- **重规划**: Replanner 介入修复的次数",
        "- **工具调用**: 单次运行的所有工具调用总次数",
        "",
        "---",
        "*本报告由 HorizonRL-Agent Benchmark 引擎自动生成*",
    ]

    return "\n".join(lines)


def load_tasks(path: str | Path) -> list[dict]:
    """从 JSONL 文件加载任务列表。

    Args:
        path: JSONL 文件路径。

    Returns:
        任务 dict 列表。
    """
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def save_results(
    results: list[RunResult],
    summary: BenchmarkSummary,
    report: str,
    output_dir: str | Path = "benchmark_results",
) -> dict[str, str]:
    """保存评测结果到磁盘。

    Returns:
        {"summary": path, "results": path, "report": path}
    """
    import os

    run_dir = Path(output_dir) / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "overall_pass_rate": summary.overall_pass_rate,
                "avg_task_success_rate": summary.avg_task_success_rate,
                "avg_tool_calls": summary.avg_tool_calls,
                "avg_replans": summary.avg_replans,
                "avg_mock_ratio": summary.avg_mock_ratio,
                "avg_runtime_ms": summary.avg_runtime_ms,
                "total_runs": summary.total_runs,
                "passed_runs": summary.passed_runs,
                "per_category": {
                    k: {
                        "pass_rate": v["pass_rate"],
                        "avg_task_sr": v["avg_task_sr"],
                        "avg_tool_calls": v["avg_tool_calls"],
                        "avg_mock_ratio": v["avg_mock_ratio"],
                    }
                    for k, v in summary.per_category.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Per-task results JSONL
    results_path = run_dir / "per_task_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(
                {
                    "task_id": r.task_id,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "passed": r.passed,
                    "task_success_rate": r.task_success_rate,
                    "tool_calls": r.tool_calls,
                    "replans": r.replans,
                    "mock_ratio": r.mock_ratio,
                    "runtime_ms": r.runtime_ms,
                    "used_provider": r.used_provider,
                },
                ensure_ascii=False,
            ) + "\n")

    # Report
    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "summary": str(summary_path),
        "results": str(results_path),
        "report": str(report_path),
    }
