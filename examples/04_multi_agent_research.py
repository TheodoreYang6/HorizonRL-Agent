"""
=======================================================================
04_multi_agent_research.py — HorizonRL-Agent v1 完整集成 Demo
=======================================================================

这是 HorizonRL-Agent 的旗舰 Demo，将全部模块串联成一条完整的研究管道：

    UserTask → Planner → PlanGraph → ResearchOrchestrator (LangGraph DAG)
        → Verifier → Replanner → HierarchicalMemory → Writer v2
        → final_answer.md + debug_report.md

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
    reports/{session_id}/final_answer.md  — 用户研究报告
    reports/{session_id}/debug_report.md  — 开发者调试报告
    trajectories/{session_id}.jsonl         — 完整轨迹日志

v2 变更 (Day 6):
    - 改为调用共享 research_service，CLI/Web/Benchmark 统一入口
    - 删除手动编排循环，走 ResearchOrchestrator 全链路
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from horizonrl.config.settings import load_config, RootConfig
from horizonrl.services.research_service import (
    run_research_session,
    SessionArtifacts,
    resolve_mode,
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CLI 展示层                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def print_header(title: str):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def print_status(label: str, message: str, level: str = "info"):
    icons = {"info": "  📌", "warn": "  ⚠️", "error": "  ❌", "ok": "  ✅"}
    icon = icons.get(level, icons["info"])
    print(f"{icon} [{label}] {message}")


def print_session_banner(session_id: str, query: str, mode: str, workers: int):
    print(f"\n{'='*70}")
    print(f"  HorizonRL-Agent v1 — 多 Agent 研究管道")
    print(f"{'='*70}")
    print(f"  Session: {session_id}")
    print(f"  问题: {query}")
    print(f"  模式: {mode}")
    print(f"  并发: {workers} workers")
    print(f"{'='*70}")


def print_results_panel(artifacts: SessionArtifacts):
    """打印 CLI 统计面板。"""
    print_header("执行结果")
    s = artifacts.stats
    total = s.get("total_count", 0)
    success = s.get("success_count", 0)
    rate = success / max(total, 1) * 100

    print(f"  📋 任务完成: {success}/{total} ({rate:.0f}%)")
    print(f"  🔄 执行轮次: {s.get('rounds', '?')}")
    print(f"  🔧 工具调用: {artifacts.tool_calls_count}")
    print(f"  ♻️  重规划: {s.get('total_replans', 0)}")
    print(f"  ⏱️  总耗时: {artifacts.runtime_ms / 1000:.1f}s")
    print(f"  📄 Mock 占比: {artifacts.mock_ratio:.0%}")
    print(f"  🔍 搜索提供商: {artifacts.used_search_provider}")

    if artifacts.error:
        print(f"  ⚠️  错误: {artifacts.error}")

    # 报告路径
    print(f"\n  ── 输出文件 ──")
    print(f"  📝 用户报告: {artifacts.final_answer_path}")
    print(f"  🔧 调试报告: {artifacts.debug_report_path}")
    print(f"  📊 轨迹日志: {artifacts.trajectory_path}")

    # 报告预览
    if artifacts.final_answer_text:
        preview = artifacts.final_answer_text[:1200]
        print(f"\n  ── 报告预览 ──")
        print(preview)

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
    search_provider: str = "auto",
    offline: bool = False,
):
    """运行完整研究管道 —— 通过共享 research_service 走全链路。

    Args:
        user_query: 研究问题。
        use_llm: 是否使用 LLM 驱动。
        semaphore_limit: 最大并发 Worker 数。
        search_provider: 搜索提供商 (auto/bocha/brave/duckduckgo/mock)。
        offline: 强制离线/Mock 模式。
    """
    t_start = time.time()

    # ── 加载配置 ──
    print_header("Stage 0: 基础设施加载")
    try:
        cfg = load_config(Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None)
    except Exception:
        cfg = RootConfig()

    # ── LLM 客户端 ──
    llm_client = None
    if use_llm and cfg.llm.api_key:
        from horizonrl.llm.client import LLMClient
        llm_client = LLMClient(cfg.llm)
        print_status("llm", f"LLM 模式: {cfg.llm.model}")
    else:
        print_status("llm", "离线模式 (模板规划 + 模拟工具)", "warn")

    # ── 搜索提供商 ──
    provider_label = search_provider
    if offline:
        provider_label = "mock (离线)"
        print_status("search", f"Provider: {provider_label}", "warn")
    else:
        import os as _os
        if _os.getenv("BOCHA_API_KEY"):
            provider_label = "Bocha(国内)"
        elif _os.getenv("BRAVE_API_KEY"):
            provider_label = "Brave"
        else:
            provider_label = "DuckDuckGo → Wikipedia → Mock"
        print_status("search", f"Provider: {provider_label}")

    mode_label = "LLM 驱动" if use_llm else "离线 (模板+模拟工具)"

    # ── 调用共享 Service (核心: 唯一执行入口) ──
    print_header("Stage 1: 任务规划")
    print_status("plan", "正在将问题拆解为子任务...")

    artifacts = await run_research_session(
        query=user_query,
        mode="deep",
        llm_client=llm_client,
        config=cfg,
        search_provider=search_provider,
        offline=offline,
        semaphore_limit=semaphore_limit,
        export_dir="reports",
    )

    # ── 逐阶段展示详细结果 ──
    s = artifacts.stats

    # Stage 1: 规划结果
    print_header("Stage 1: 任务规划 — 结果")
    root_count = sum(1 for td in artifacts.task_details if td.status == "success" or True)
    for td in artifacts.task_details:
        print(f"    📋 [{td.task_id}] {td.name}")
    print_status("plan", f"共 {s.get('total_count', 0)} 个子任务, "
                 f"按 DAG 依赖并发执行")

    # Stage 2: 逐任务执行结果
    print_header("Stage 2: DAG 并发执行 + 验证 + 重规划")
    for td in artifacts.task_details:
        icon = "✅" if td.passed else "❌"
        err_info = f" — {td.error_type}" if td.error_type and td.error_type not in ("none", "") else ""
        elapsed_str = f"{td.elapsed:.1f}s" if td.elapsed > 0 else "—"
        print(f"    {icon} [{td.task_id}] {td.name}")
        print(f"       score={td.score:.1f} | {td.evidence_count} 证据 | "
              f"{td.tool_calls} 工具 | 耗时 {elapsed_str}{err_info}")
        if td.feedback and not td.passed:
            print(f"       诊断: {td.feedback[:100]}")
    print_status("exec", f"完成: {s.get('success_count', 0)}/{s.get('total_count', 0)} 子任务 | "
                 f"{s.get('rounds', '?')} 轮 | {artifacts.tool_calls_count} 工具调用 | "
                 f"重规划 {s.get('total_replans', 0)} 次")
    if artifacts.error:
        print_status("warn", f"工作流异常: {artifacts.error}", "warn")

    # Stage 3: 报告输出
    print_header("Stage 3+4: 记忆总结 + 最终报告")
    print_status("memory", f"L1/L2 记忆已更新")
    print_status("report", f"final_answer: {artifacts.final_answer_path}")
    print_status("report", f"debug_report: {artifacts.debug_report_path}")
    print_status("trajectory", f"轨迹日志: {artifacts.trajectory_path}")

    # Stage 5: 统计面板
    print_header("Stage 5: 统计面板")
    print_session_banner(artifacts.session_id, user_query, mode_label, semaphore_limit)
    print_results_panel(artifacts)

    total_elapsed = time.time() - t_start
    print(f"\n⏱️  端到端耗时: {total_elapsed:.1f}s")
    return artifacts


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
  python examples/04_multi_agent_research.py --provider brave
  python examples/04_multi_agent_research.py --offline
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
    parser.add_argument(
        "--provider", type=str, default="auto",
        help="搜索提供商: auto/bocha/brave/duckduckgo/mock",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="强制离线/Mock 模式",
    )

    args = parser.parse_args()

    user_query = args.query or "Transformer 模型中多头注意力机制的工作原理与最新改进"
    asyncio.run(run_pipeline(
        user_query=user_query,
        use_llm=args.llm,
        semaphore_limit=args.workers,
        search_provider=args.provider,
        offline=args.offline,
    ))


if __name__ == "__main__":
    main()
