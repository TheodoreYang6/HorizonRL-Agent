"""
=======================================================================
03_llm_demo.py -- LLM API 连接测试 + 真实 LLM 驱动的任务规划 Demo
=======================================================================

这是第一个真正调用 LLM API 的 Demo。跑通它意味着你的 API 配置正确，
LLM 客户端可以正常工作，HorizonRL-Agent 具备了用大模型做任务分解的能力。

运行方式:
    python examples/03_llm_demo.py

前提条件:
    .env 文件已配置 OPENAI_API_KEY

Demo 流程:
    Step 0: 加载配置 + 检查 API Key
    Step 1: LLM 连接测试 (echo)
    Step 2: LLM 任务分解 (Planner)
    Step 3: 完整 Pipeline (LLM 规划 + Worker 执行 + 报告)
    Step 4: Token 用量统计
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from horizonrl.config.settings import load_config, RootConfig
from horizonrl.llm.client import LLMClient, LLMCallResult
from horizonrl.agent.planner import LLMPlanner
from horizonrl.schemas.task import UserTask
from horizonrl.orchestration import create_orchestrator
from horizonrl.tools.manager import ToolManager


# ======================================================================
# Step 0: 加载配置
# ======================================================================

def step0_load_config():
    """加载配置并检查 API Key 是否已设置。"""
    print("=" * 60)
    print("Step 0: 加载配置")
    print("=" * 60)

    dev_path = Path("configs/dev.yaml")
    path = dev_path if dev_path.exists() else None
    cfg = load_config(path)

    print(f"  Provider: {cfg.llm.provider}")
    print(f"  Model: {cfg.llm.model}")
    print(f"  Base URL: {cfg.llm.base_url or '(默认)'}")
    print(f"  Temperature: {cfg.llm.temperature}")
    print(f"  API Key: {'***' + cfg.llm.api_key[-4:] if cfg.llm.api_key else '未设置!'}")

    if not cfg.llm.api_key:
        print("\n  [错误] API Key 未设置!")
        print("  请检查:")
        print("    1. .env 文件是否在项目根目录")
        print("    2. .env 中是否有 OPENAI_API_KEY=sk-...")
        sys.exit(1)

    return cfg


# ======================================================================
# Step 1: LLM 连接测试
# ======================================================================

async def step1_connection_test(client: LLMClient):
    """最简单的 LLM 调用，验证 API 连通性。"""
    print("\n" + "=" * 60)
    print("Step 1: LLM 连接测试")
    print("=" * 60)

    result = await client.chat(
        prompt="请用一句话介绍你自己（用中文）",
        max_tokens=50,
    )

    if result.is_success:
        print(f"  [OK] 连接成功!")
        print(f"  模型: {result.model}")
        print(f"  回复: {result.content[:100]}")
        print(f"  Token: {result.tokens_prompt}+{result.tokens_completion}"
              f"={result.tokens_total}")
        print(f"  耗时: {result.elapsed:.2f}s")
    else:
        print(f"  [FAIL] 连接失败: {result.error}")
        print("\n  常见原因:")
        print("    1. API Key 无效或已过期")
        print("    2. 网络无法访问 API 端点")
        print("    3. 模型名称不正确 (当前: {})".format(client.config.model))
        print("\n  尝试修复:")
        print("    - 如果用 DeepSeek: 设置环境变量 HORIZON_LLM__BASE_URL=https://api.deepseek.com")
        print("    - 如果用 OpenAI: 确认 Key 格式为 sk-...")
        print("    - 如果用其他兼容 API: 设置对应的 base_url")
        sys.exit(1)

    return result


# ======================================================================
# Step 2: LLM 任务分解
# ======================================================================

async def step2_llm_planning(client: LLMClient, user_query: str):
    """使用 LLMPlanner 做真实的任务分解。"""
    print("\n" + "=" * 60)
    print("Step 2: LLM 任务分解")
    print("=" * 60)
    print(f"  输入: {user_query}")

    planner = LLMPlanner(client)
    task = UserTask(description=user_query, max_steps=20)
    plan = await planner.plan(task)

    print(f"  拆解结果: {plan.total_count()} 个子任务")
    print(f"  根节点 (可并行): {len(plan.root_ids)} 个")
    print(f"  DAG 结构:")

    for node in plan.nodes.values():
        deps = node.depends_on
        dep_names = []
        for did in deps:
            dn = plan.nodes.get(did)
            dep_names.append(dn.spec.name if dn else did)
        dep_str = ", ".join(dep_names) if dep_names else "(无)"
        tools_str = ", ".join(node.spec.tool_names) or "(无)"
        print(f"    [{node.id}] {node.spec.name}")
        print(f"        工具: {tools_str}  |  依赖: {dep_str}  |  优先级: {node.spec.priority.value}")

    return plan


# ======================================================================
# Step 3: 完整 Pipeline
# ======================================================================

async def step3_full_pipeline(plan, user_query: str):
    """用 LLM 规划结果跑完整执行 + 报告 Pipeline。"""
    print("\n" + "=" * 60)
    print("Step 3: 完整 Pipeline (执行 + 报告)")
    print("=" * 60)

    mgr = _setup_mock_tools()

    # 手动跑 DAG 执行循环（比 LangGraph ainvoke 更可控）
    import asyncio, time
    from horizonrl.schemas.task import TaskStatus
    from horizonrl.agent.worker import AgentWorker

    sem = asyncio.Semaphore(3)
    results = {}
    round_num = 0

    while plan.has_pending_work():
        # Mark ready
        for node in plan.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            deps_ok = all(
                plan.nodes[d].status == TaskStatus.SUCCESS
                for d in node.depends_on
            )
            if deps_ok:
                node.status = TaskStatus.READY

        ready = [n for n in plan.nodes.values() if n.status == TaskStatus.READY]
        if not ready:
            pending = [n for n in plan.nodes.values()
                       if n.status in (TaskStatus.PENDING, TaskStatus.READY)]
            if pending:
                print(f"  [死锁] {len(pending)} 个任务阻塞")
            break

        round_num += 1
        print(f"  Round {round_num}: 并发执行 {len(ready)} 个任务")

        async def run_one(node):
            node.status = TaskStatus.RUNNING
            async with sem:
                worker = AgentWorker(
                    worker_id=f"wrk_{node.id}",
                    tool_manager=mgr,
                )
                result = await worker.execute(node.spec)
                node.finished_at = time.time()
                if result.success:
                    node.status = TaskStatus.SUCCESS
                else:
                    node.status = TaskStatus.FAILED
                    node.error_msg = result.error
                return result

        batch = await asyncio.gather(*[run_one(n) for n in ready])
        for r in batch:
            results[r.task_id] = r
            icon = "+" if r.success else "-"
            print(f"    {icon} {r.task_id}: {len(r.evidence)}条证据, {r.elapsed:.2f}s")

    # 汇总
    print(f"\n  完成: {plan.success_count()}/{plan.total_count()} 成功")
    total_evidence = sum(len(r.evidence) for r in results.values())
    print(f"  证据: {total_evidence} 条")

    return {"plan": plan, "results": results}


def _setup_mock_tools():
    """注册模拟工具，确保离线可跑。"""
    from horizonrl.tools.mock import register_mock_tools
    mgr = ToolManager()
    register_mock_tools(mgr)
    return mgr


# ======================================================================
# Step 4: 统计汇总
# ======================================================================

def step4_summary(connect_result, pipeline_result):
    """打印最终统计。"""
    print("\n" + "=" * 60)
    print("Step 4: 最终统计")
    print("=" * 60)

    plan = pipeline_result.get("plan")
    results = pipeline_result.get("results", {})
    if plan:
        print(f"  任务完成: {plan.success_count()}/{plan.total_count()} 成功")

    print(f"\n  LLM Token 用量:")
    print(f"    连接测试: {connect_result.tokens_total} tokens ({connect_result.elapsed:.2f}s)")
    print(f"    Pipeline 工具执行: {len(results)} 个子任务完成")


# ======================================================================
# main
# ======================================================================

async def main():
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
                 "Transformer 模型中多头注意力机制的工作原理与最新改进"

    print("\n" + "=" * 60)
    print("  HorizonRL-Agent: LLM API 连接测试 + 规划 Demo")
    print("=" * 60)
    print(f"  任务: {user_query}")

    # Step 0
    cfg = step0_load_config()

    # 创建 LLM 客户端
    client = LLMClient(cfg.llm)

    # Step 1: 连接测试
    connect_result = await step1_connection_test(client)

    # Step 2: LLM 任务分解
    plan = await step2_llm_planning(client, user_query)

    # Step 3: 完整 Pipeline
    pipeline_result = await step3_full_pipeline(plan, user_query)

    # Step 4: 统计
    step4_summary(connect_result, pipeline_result)

    print("\n" + "=" * 60)
    print("  Demo 完成! 你的 API 配置正确，LLM 已接入 HorizonRL-Agent。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
