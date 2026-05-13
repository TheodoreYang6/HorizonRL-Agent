"""
HorizonRL-Agent asyncio 完整教程
=================================
本文件覆盖项目中所有需要的 asyncio 知识点，从基础到进阶，
每个知识点都标注了在项目中的实际用途。

学习顺序（建议）：按数字编号从 1->10 逐步运行和修改。

运行方式：
    python examples/01_async_demo.py              # 运行所有示例
    python examples/01_async_demo.py --example 3  # 只运行第3个示例

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
函数索引 (按学习顺序)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  example_1()  → async/await 基础：协程创建、await 等待、asyncio.run() 入口
  example_2()  → sleep 对比：time.sleep 阻塞 vs await asyncio.sleep 让出
  example_3()  → asyncio.gather：并发执行多个协程，结果保持提交顺序
  example_4()  → asyncio.Semaphore：控制并发数（防 Rate Limit/显存溢出）
  example_5()  → asyncio.create_task：启动后台任务，稍后取结果
  example_6()  → asyncio.as_completed：不等最慢的，先完成先处理
  example_7()  → asyncio.wait_for：超时控制，防止卡死
  example_8()  → return_exceptions=True：单任务失败不影响其他 Worker
  example_9()  → loop.run_in_executor：同步阻塞函数 → 异步非阻塞
  example_10() → 综合示例：Planner + 3 Workers + Semaphore 并发执行

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
项目中的应用映射
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  example_4  Semaphore   → Worker 并发调 OpenAI API 限流
  example_7  wait_for    → LLM 调用 30s 超时保护
  example_8  return_exceptions → 5 个 Worker 中 1 个失败不影响其余
  example_9  run_in_executor  → 包装同步 langchain_openai.invoke()
  example_10 综合示例    → src/horizonrl/orchestration/dag_workflow.py 的底层模型
"""

from __future__ import annotations

import asyncio
import time
import sys
from typing import Any


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. async/await 基础 —— 理解协程是什么                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 关键概念：
#   - def       -> 普通函数，调用就执行，执行完返回
#   - async def -> 协程函数，调用返回一个 coroutine 对象，不立即执行
#   - await     -> 等待一个协程执行完成，同时让出 CPU 给其他协程
#   - asyncio.run() -> 事件循环的入口，把协程跑起来
#
# 项目中哪里用到：
#   - AgentWorker.execute() 是 async def，因为它需要等待 LLM 响应
#   - 所有工具调用（搜索、代码执行）都是 async def
#   - Planner.plan_tasks() 调用 LLM 时是 async 的


def example_1():
    """最基础的 async/await 示例：理解协程的创建和执行。"""

    print("\n" + "=" * 70)
    print("1. async/await 基础")
    print("=" * 70)

    # --- 普通函数：调用即执行，一行跑完才跑下一行 ---
    def sync_greet(name: str) -> str:
        """普通函数：调用就立即执行，执行期间 CPU 不能干别的事。"""
        return f"你好, {name}!"

    # --- 协程函数：调用只创建协程对象，不执行 ---
    async def async_greet(name: str) -> str:
        """协程函数：前面有 async，调用不会立即执行，返回一个 coroutine。

        项目中对应：
            AgentWorker.execute() —— 调用时不会阻塞，可以用 gather 并行多个。
        """
        return f"你好, {name}!"

    # --- 协程函数里用 await 等待另一个协程 ---
    async def async_with_await(name: str) -> str:
        """在协程内部用 await 等待另一个协程的结果。

        注意：await 会让出执行权，事件循环可以在这期间切换去执行其他协程。
        这就是"并发"的来源 —— 不是多线程，是协作式多任务。

        项目中对应：
            worker.execute() 内部 await self._call_llm(prompt)
            等待 LLM 返回时，其他 worker 可以继续执行。
        """
        # await 后面只能跟 "可等待对象"：coroutine / Task / Future
        result = await async_greet(name)
        return f"协程内调用: {result}"

    # --- 演示 ---
    print("--- 普通函数 ---")
    result1 = sync_greet("Alice")       # 立即执行，result1 是字符串
    print(f"返回值: {result1}")
    print(f"返回值类型: {type(result1).__name__}")

    print("\n--- 协程函数（不 await） ---")
    coro = async_greet("Bob")           # 不执行！只创建协程对象
    print(f"返回值: {coro}")
    print(f"返回值类型: {type(coro).__name__}")
    print("[!] 注意：协程函数被调用时并不会执行，只是创建了一个 coroutine 对象")
    print("   必须用 await 或 asyncio.run() 才能真正执行它")

    print("\n--- 协程函数（用 asyncio.run 执行） ---")
    result2 = asyncio.run(async_greet("Bob"))   # asyncio.run() 启动事件循环
    print(f"返回值: {result2}")
    print(f"返回值类型: {type(result2).__name__}")

    print("\n--- 协程内部 await 另一个协程 ---")
    result3 = asyncio.run(async_with_await("Charlie"))
    print(f"返回值: {result3}")

    print("\n[OK] 核心理解：")
    print("   async def -> 定义一个协程（可以暂停和恢复的函数）")
    print("   await     -> 暂停当前协程，等待另一个协程完成")
    print("   asyncio.run() -> 启动事件循环，运行顶层协程")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. asyncio.sleep  vs  time.sleep —— 理解"让出"的意义                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 关键概念：
#   - time.sleep(1)     -> 阻塞整个线程 1 秒，期间啥也不能干
#   - await asyncio.sleep(1) -> 暂停当前协程 1 秒，但其他协程可以继续跑
#
# 项目中哪里用到：
#   - 重试等待（await asyncio.sleep(backoff)）
#   - 轮询 vLLM 推理结果
#   - 限速等待


def example_2():
    """理解同步阻塞 vs 异步非阻塞的区别。"""

    print("\n" + "=" * 70)
    print("2. sleep 对比：阻塞 vs 让出")
    print("=" * 70)

    # --- 同步版本：顺序执行，总时间 = 3秒 + 2秒 + 1秒 = 6秒 ---
    def sync_version():
        """同步版本：一个任务完成才开始下一个，总时间累加。

        项目中对应：
            如果不加 asyncio，20 个 Worker 顺序执行，每个等 LLM 5 秒，
            总共 100 秒 —— 太慢了！
        """
        print("同步版 开始")
        t0 = time.perf_counter()
        time.sleep(1.0)  # 阻塞整个线程 1 秒
        print(f"  任务A完成 (+{time.perf_counter() - t0:.1f}s)")
        time.sleep(1.0)
        print(f"  任务B完成 (+{time.perf_counter() - t0:.1f}s)")
        time.sleep(1.0)
        print(f"  任务C完成 (+{time.perf_counter() - t0:.1f}s)")
        return time.perf_counter() - t0

    # --- 异步版本：并发执行，总时间 ≈ 1秒（最慢的那个任务的时间） ---
    async def async_task(name: str, seconds: float) -> str:
        """模拟一个异步任务，比如调用 LLM API 等待响应。

        await asyncio.sleep() 期间，事件循环可以切去执行其他协程。
        """
        await asyncio.sleep(seconds)  # 让出 CPU，不阻塞
        return f"任务{name}完成"

    async def async_version() -> float:
        """异步版本：三个任务并发执行。

        项目中对应：
            await asyncio.gather(
                worker_a.execute(task_a),  # 耗时 5s
                worker_b.execute(task_b),  # 耗时 3s
                worker_c.execute(task_c),  # 耗时 4s
            )
            # 总耗时 ≈ 5s（最长的那个），而不是 12s
        """
        print("异步版 开始")
        t0 = time.perf_counter()
        # gather：并发运行所有协程，等全部完成后返回结果列表
        results = await asyncio.gather(
            async_task("A", 1.0),
            async_task("B", 1.0),
            async_task("C", 1.0),
        )
        elapsed = time.perf_counter() - t0
        for r in results:
            print(f"  {r} (+{elapsed:.1f}s)")
        return elapsed

    # --- 演示 ---
    elapsed_sync = sync_version()
    print(f"同步版总耗时: {elapsed_sync:.1f}s")

    print()
    elapsed_async = asyncio.run(async_version())
    print(f"异步版总耗时: {elapsed_async:.1f}s")
    print(f"加速比: {elapsed_sync / elapsed_async:.1f}x")

    print("\n[OK] 核心理解：")
    print("   time.sleep() -> 阻塞线程，其他任务干等着")
    print("   await asyncio.sleep() -> 让出 CPU，其他协程继续跑")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. asyncio.gather —— 并发执行多个协程（最常用的并发模式）                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 关键概念：
#   - gather(*coros) -> 并发执行所有协程，返回所有结果的列表
#   - 执行顺序：提交顺序 ≠ 完成顺序（谁先完成谁先返回），但结果列表保持提交顺序
#   - 如果某个协程抛异常：默认会传播异常（见 example_7 的错误处理）
#
# 项目中哪里用到：
#   - Planner 拆解出 5 个子任务 -> gather 5 个 Worker 并发执行
#   - RolloutManager 并发跑 N 个环境实例收集轨迹
#   - 并发调用多个搜索 API


def example_3():
    """gather 是最常用的并发原语：提交一组任务，等全部完成，拿到全部结果。"""

    print("\n" + "=" * 70)
    print("3. asyncio.gather —— 并发执行")
    print("=" * 70)

    async def search_tool(query: str, delay: float) -> dict[str, Any]:
        """模拟搜索工具调用，不同 query 耗时不同。

        项目中对应：
            tools/web_search.py 的 web_search() 函数
            tools/arxiv_search.py 的 arxiv_search() 函数
        """
        await asyncio.sleep(delay)  # 模拟网络延迟
        return {
            "query": query,
            "results": [f"{query} 相关结果{i}" for i in range(3)],
            "elapsed": delay,
        }

    async def research_worker_demo() -> None:
        """模拟 Planner 把一个大任务拆成 3 个子任务，并发搜索。

        项目中对应：
            Planner.plan_tasks() -> [Task("搜索A"), Task("搜索B"), Task("搜索C")]
            然后 asyncio.gather(*[worker.execute(t) for t in tasks])
        """
        print("Worker 开始并行搜索...")
        t0 = time.perf_counter()

        # gather: 传入多个协程，并发执行，返回列表（顺序 = 传入顺序）
        results = await asyncio.gather(
            search_tool("Transformer 注意力机制", delay=1.5),
            search_tool("长上下文记忆管理", delay=1.0),
            search_tool("强化学习 GRPO 算法", delay=0.8),
        )

        elapsed = time.perf_counter() - t0
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['query']} -> {len(r['results'])} 条结果 (耗时 {r['elapsed']}s)")

        # 关键：三个任务耗时分别是 1.5s / 1.0s / 0.8s
        # 并发执行总耗时 ≈ 1.5s（最长的那个），而不是 3.3s
        print(f"  总耗时: {elapsed:.1f}s（顺序执行需要 3.3s）")

    asyncio.run(research_worker_demo())

    print("\n[OK] 核心理解：")
    print("   await gather(task1, task2, task3) -> 三个任务同时跑")
    print("   总耗时 = 最慢任务的时间（不是累加）")
    print("   结果列表顺序 = 传入参数顺序（不管谁先完成）")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. asyncio.Semaphore —— 控制并发数（最重要的工程技巧）                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 关键概念：
#   - Semaphore(n) -> 最多允许 n 个协程同时进入保护区
#   - async with sem -> 进入（计数 -1），退出（计数 +1）
#   - 当计数为 0 时，后续协程会等待
#
# 项目中哪里用到（几乎无处不在）：
#   - 限制同时调用 OpenAI API 的数量（避免 rate limit）
#   - 限制同时访问数据库的连接数
#   - 限制同时运行的 Worker 数量（避免显存溢出）
#   - 限制同时进行的 HTTP 请求数量


def example_4():
    """Semaphore：控制并发上限，防止打爆 API 或撑爆内存。"""

    print("\n" + "=" * 70)
    print("4. asyncio.Semaphore —— 控制并发数")
    print("=" * 70)

    # --- 没有 Semaphore：20 个请求同时发出，可能触发 Rate Limit ---
    async def call_llm_api_no_limit(task_id: int) -> str:
        """没有并发限制的 LLM API 调用。

        项目中对应（危险示例）：
            20 个 Worker 同时调 OpenAI API -> 触发 429 Rate Limit Error
        """
        await asyncio.sleep(0.5)  # 模拟 API 响应时间
        return f"Task-{task_id} 完成"

    async def without_semaphore() -> float:
        """不加限制：20 个请求同时发出。"""
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            call_llm_api_no_limit(i) for i in range(20)
        ])
        elapsed = time.perf_counter() - t0
        print(f"无限制: 20 个请求全部同时发出，{elapsed:.1f}s 内完成")
        print(f"  [WARN]  OpenAI 会返回 429 Too Many Requests!")
        return elapsed

    # --- 有 Semaphore：最多同时 3 个请求 ---
    # 创建一个信号量，最大并发数为 3
    SEM_LIMIT = 3

    async def call_llm_api_with_limit(
        task_id: int,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """带并发限制的 LLM API 调用。

        async with semaphore 确保同一时刻最多只有 3 个协程在执行内部代码。

        项目中对应：
            class AgentWorker:
                def __init__(self, llm_config, semaphore):
                    self.semaphore = semaphore  # 所有 Worker 共享同一个 Semaphore

                async def _call_llm(self, prompt):
                    async with self.semaphore:       # ← 获取许可
                        return await openai_client.chat(prompt)  # ← 真正调用
        """
        # async with semaphore: 进入时 semaphore 计数 -1
        # 如果计数已经为 0（已有 3 个在执行），这里会阻塞等待
        async with semaphore:
            await asyncio.sleep(0.5)  # 模拟 API 响应时间
        # 退出 with 块后 semaphore 计数 +1，释放一个槽位给等待者
        return f"Task-{task_id} 完成"

    async def with_semaphore() -> float:
        """加了限制：同时最多 3 个请求，其余排队。"""
        sem = asyncio.Semaphore(SEM_LIMIT)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            call_llm_api_with_limit(i, sem) for i in range(20)
        ])
        elapsed = time.perf_counter() - t0
        # 20 个请求，每批 3 个，每批 0.5s -> 共 7 批 -> 约 3.5s
        batches = (20 + SEM_LIMIT - 1) // SEM_LIMIT
        print(f"有限制(最多{SEM_LIMIT}并发): 20 个请求分 {batches} 批，{elapsed:.1f}s")
        print(f"  [OK] 不会触发 Rate Limit，显存/连接数可控")
        return elapsed

    asyncio.run(without_semaphore())
    print()
    asyncio.run(with_semaphore())

    print("\n[OK] 核心理解：")
    print("   Semaphore(N) -> 最多 N 个协程同时进入关键区")
    print("   async with sem -> 获取许可 / 释放许可（自动）")
    print("   项目中几乎所有外部调用都需要 Semaphore 保护")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. asyncio.create_task —— 创建后台任务                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 关键概念：
#   - create_task(coro) -> 把协程包装成 Task，立即开始执行（不等待）
#   - Task 在后台运行，可以稍后再 await 获取结果
#   - 和 gather 的区别：gather 等全部完成；Task 可以先执行别的再回来取结果
#
# 项目中哪里用到：
#   - 启动一个 Worker 后不立刻等结果，先去做别的事
#   - 后台记录日志、写入文件（不阻塞主流程）
#   - 同时启动多个长时间任务，按需获取结果


def example_5():
    """create_task：启动后台任务，稍后再取结果。"""

    print("\n" + "=" * 70)
    print("5. asyncio.create_task —— 后台任务")
    print("=" * 70)

    async def long_running_search(query: str, delay: float) -> str:
        """模拟耗时较长的搜索任务。

        项目中对应：
            worker.execute(task) —— 执行一个子任务，可能需要 30 秒
        """
        print(f"    [后台] 开始搜索: {query}")
        await asyncio.sleep(delay)
        print(f"    [后台] 搜索完成: {query}")
        return f"{query} -> {delay}s"

    async def planner_with_background_tasks() -> None:
        """Planner 启动多个 Worker 作为后台任务，自己先做别的事。

        项目中对应：
            Planner 拆解任务后，用 create_task 启动 Worker，
            然后在等待期间做其他准备工作（如更新 Memory）。
        """
        print("Planner: 开始拆解任务...")

        # create_task 立即开始执行协程（不等待），返回 Task 对象
        task_a = asyncio.create_task(
            long_running_search("Transformer架构", delay=2.0),
            name="search-transformer",  # 给任务命名，方便调试
        )
        task_b = asyncio.create_task(
            long_running_search("GRPO算法", delay=3.0),
            name="search-grpo",
        )
        task_c = asyncio.create_task(
            long_running_search("分层记忆", delay=1.0),
            name="search-memory",
        )

        # Worker 已经在后台跑了，Planner 可以做其他事情
        print("Planner: Worker 已启动，我先做其他准备工作...")
        await asyncio.sleep(0.5)  # 模拟 Planner 做其他事
        print("Planner: 准备工作完成，等待 Worker 结果...")

        # 现在才真正等待结果
        # gather 可以接受 Task 对象
        results = await asyncio.gather(task_a, task_b, task_c)

        for r in results:
            print(f"  [OK] {r}")

    asyncio.run(planner_with_background_tasks())

    print("\n[OK] 核心理解：")
    print("   create_task(coro) -> 立即开始执行，不等结果")
    print("   稍后 await task -> 拿到结果")
    print("   和 gather 的区别：gather 直接等；create_task 可以先做别的")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6. asyncio.as_completed —— 谁先完成先处理谁                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 关键概念：
#   - gather：等全部完成，返回按提交顺序排列的结果
#   - as_completed：不等全部完成，谁先完成就 yield 谁
#
# 项目中哪里用到：
#   - 多个 Worker 执行不同复杂度的子任务，先完成的可以先展示给用户
#   - 搜索结果先返回的先加入 Memory，不用等最慢的那个


def example_6():
    """as_completed：不等最慢的，先完成先处理。"""

    print("\n" + "=" * 70)
    print("6. asyncio.as_completed —— 先完成先处理")
    print("=" * 70)

    async def search_with_varying_delay(query: str, delay: float) -> dict:
        """不同搜索耗时不同。"""
        await asyncio.sleep(delay)
        return {"query": query, "delay": delay, "results_count": int(delay * 10)}

    async def process_results_as_they_arrive() -> None:
        """搜索结果先返回的先展示，提升用户体验。

        项目中对应：
            3 个 Worker 并行搜索，结果先回来的先给用户展示，
            不需要等最慢的搜索完成再一起显示。
        """
        tasks = [
            search_with_varying_delay("快速搜索", delay=0.5),
            search_with_varying_delay("中等搜索", delay=1.5),
            search_with_varying_delay("慢速搜索", delay=3.0),
        ]

        print("开始搜索（不等全部完成，先到先得）...")
        t0 = time.perf_counter()

        # as_completed: 迭代器，每次 yield 一个完成了的 Future
        for coro in asyncio.as_completed(tasks):
            # await 拿到这个已完成的协程的结果
            result = await coro
            elapsed = time.perf_counter() - t0
            print(f"  [RECV] [{elapsed:.1f}s] {result['query']} 完成！({result['results_count']}条结果)")

        print(f"  全部完成，总耗时: {time.perf_counter() - t0:.1f}s")

    asyncio.run(process_results_as_they_arrive())

    print("\n[OK] 核心理解：")
    print("   gather(ts)    -> 等全部完成，结果顺序 = 传入顺序")
    print("   as_completed  -> 不等最慢的，谁快先拿谁")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7. 超时控制 —— asyncio.wait_for                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 关键概念：
#   - wait_for(coro, timeout) -> 给协程设超时，超时抛 TimeoutError
#   - 防止某个 Worker 卡死拖垮整个流程
#
# 项目中哪里用到：
#   - 单个 LLM 调用超时（30s 没返回就放弃）
#   - 单个搜索工具超时
#   - 整个 Agent 任务超时（30 步还没完成就终止）


def example_7():
    """超时控制：不能让一个卡住的任务拖死全流程。"""

    print("\n" + "=" * 70)
    print("7. asyncio.wait_for —— 超时控制")
    print("=" * 70)

    async def stubborn_search(query: str, delay: float) -> str:
        """模拟一个可能卡住的搜索。

        项目中对应：
            web_search() 请求某个网站，30 秒没响应就该放弃了。
        """
        await asyncio.sleep(delay)
        return f"{query} 搜索结果"

    async def worker_with_timeout() -> None:
        """Worker 执行子任务，每个子任务最多等 2 秒。"""

        # --- 正常完成 ---
        try:
            # wait_for: 如果协程在 2 秒内完成，返回结果
            result = await asyncio.wait_for(
                stubborn_search("快速查询", delay=0.5),
                timeout=2.0,
            )
            print(f"  [OK] 正常完成: {result}")
        except asyncio.TimeoutError:
            print("  [FAIL] 超时了！（这行不应该出现）")

        # --- 超时 ---
        try:
            # 这个协程需要 5 秒，但我们只给 2 秒 -> 会触发 TimeoutError
            result = await asyncio.wait_for(
                stubborn_search("超慢查询", delay=5.0),
                timeout=2.0,
            )
            print(f"  [OK] 完成: {result}")
        except asyncio.TimeoutError:
            print("  [TIMEOUT] 查询超时（5s任务，2s限制）-> 取消任务，记录失败")

        # --- 项目中实际的用法 ---
        print("\n  项目中 Worker 的超时保护:")
        print("    try:")
        print("        result = await asyncio.wait_for(")
        print("            self._call_llm(prompt),")
        print("            timeout=30.0,  # LLM最多等30秒")
        print("        )")
        print("    except asyncio.TimeoutError:")
        print("        return ActionResult(success=False, error='LLM调用超时')")

    asyncio.run(worker_with_timeout())

    print("\n[OK] 核心理解：")
    print("   wait_for(coro, timeout=N) -> N秒后抛 TimeoutError")
    print("   每个外部调用都应该设超时，防止卡死")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8. 错误处理 —— 一个 Worker 崩了不能影响其他 Worker                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 关键概念：
#   - gather 默认行为：一个协程抛异常，gather 会立即抛出，其他协程被取消
#   - gather(return_exceptions=True)：异常不传播，当作正常结果返回
#   - 项目中必须用 return_exceptions=True，因为一个 Worker 失败不应影响其他
#
# 项目中哪里用到：
#   - 5 个 Worker 并发执行，其中一个搜不到结果（抛异常），其他 4 个应该继续


def example_8():
    """错误处理：一个任务失败不影响其他任务。"""

    print("\n" + "=" * 70)
    print("8. 错误处理 —— 容错并发")
    print("=" * 70)

    async def reliable_search(query: str) -> dict:
        """正常搜索。"""
        await asyncio.sleep(0.3)
        return {"query": query, "status": "ok", "results": ["结果1", "结果2"]}

    async def failing_search(query: str) -> dict:
        """这个搜索会失败。"""
        await asyncio.sleep(0.2)
        raise ValueError(f"搜索 '{query}' 失败: API Key 无效")

    # --- 默认行为：一个失败，全部遭殃 ---
    async def without_error_handling():
        """gather 默认：一个异常 -> 抛异常，其余协程被取消。

        项目中（错误用法）：
            # 如果 task_b 的搜索 API 挂了，task_a 和 task_c 也被取消
            results = await asyncio.gather(task_a, task_b, task_c)
        """
        try:
            results = await asyncio.gather(
                reliable_search("查询A"),
                failing_search("查询B"),         # ← 这个会抛异常
                reliable_search("查询C"),         # ← 这个被连累取消
            )
        except ValueError as e:
            print(f"  [FAIL] 一个失败全盘结束: {e}")
            print(f"     查询C 虽然正常，但也被取消了！")

    # --- 正确做法：return_exceptions=True ---
    async def with_error_handling():
        """return_exceptions=True：异常变成返回值，不影响其他任务。

        项目中（正确用法）：
            results = await asyncio.gather(
                worker_a.execute(task_a),
                worker_b.execute(task_b),
                worker_c.execute(task_c),
                return_exceptions=True,  # ← 关键！
            )
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    logger.error(f"Worker {i} 失败: {r}")
                else:
                    process(r)  # 正常结果
        """
        results = await asyncio.gather(
            reliable_search("查询A"),
            failing_search("查询B"),         # ← 这个会抛异常
            reliable_search("查询C"),         # ← 这个不受影响！
            return_exceptions=True,           # ← 关键参数！
        )

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"  [WARN]  Worker[{i}] 失败: {r}")
            else:
                print(f"  [OK] Worker[{i}] 成功: {r['query']} -> {len(r['results'])} 条结果")

    print("--- 没有错误处理 ---")
    asyncio.run(without_error_handling())

    print("\n--- 有错误处理 (return_exceptions=True) ---")
    asyncio.run(with_error_handling())

    print("\n[OK] 核心理解：")
    print("   gather(return_exceptions=True) -> 单个失败不影响整体")
    print("   遍历结果时用 isinstance(r, Exception) 判断成功/失败")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  9. run_in_executor —— 在异步代码中运行同步阻塞函数                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 关键概念：
#   - 有些库没有 async 版本（如早期的 OpenAI SDK、某些数据库驱动）
#   - 在 async 函数里直接调用同步阻塞函数会卡住整个事件循环（非常危险！）
#   - run_in_executor 把同步函数放到线程池执行，不阻塞事件循环
#
# 项目中哪里用到：
#   - _sync_call_llm() 是同步的（调 OpenAI SDK），用 run_in_executor 包装
#   - 文件读写
#   - 调用没有 async 版本的第三方库


def example_9():
    """run_in_executor：在异步世界安全地运行同步代码。"""

    print("\n" + "=" * 70)
    print("9. run_in_executor —— 同步代码异步化")
    print("=" * 70)

    import time as time_mod

    # --- 模拟一个同步阻塞函数（比如老版本的 openai.ChatCompletion.create） ---
    def sync_llm_call(prompt: str) -> str:
        """同步的 LLM 调用（阻塞线程）。

        项目中对应：
            worker._sync_call_llm(prompt) —— 用 langchain_openai 的同步 invoke
        """
        time_mod.sleep(1.0)  # 模拟同步等待 LLM 响应
        return f"LLM 对 '{prompt[:20]}...' 的回复"

    # --- [FAIL] 错误做法：在 async 函数里直接调同步函数 ---
    async def bad_approach() -> None:
        """错误示范：async 函数里直接调 time.sleep() 会阻塞整个事件循环。

        如果 3 个 Worker 都这么干，它们会顺序执行（每个等 1s，总共 3s），
        因为事件循环被 time.sleep() 卡住了，无法切换。
        """
        print("  [FAIL] 错误做法：直接调同步函数")
        t0 = time.perf_counter()
        # 这三个是"顺序"执行的！因为 sync_llm_call 阻塞了线程
        result1 = sync_llm_call("任务1")
        result2 = sync_llm_call("任务2")
        result3 = sync_llm_call("任务3")
        print(f"     总耗时: {time.perf_counter() - t0:.1f}s —— 累加的！")

    # --- [OK] 正确做法：用 run_in_executor 包装 ---
    async def good_approach() -> None:
        """正确做法：用 run_in_executor 把同步函数放到线程池。

        项目中对应（worker.py 第88-91行）：
            async def _call_llm(self, prompt: str) -> str:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,                    # None = 默认线程池
                    self._sync_call_llm,     # 要执行的同步函数
                    prompt,                  # 传给函数的参数
                )
        """
        print("  [OK] 正确做法：用 run_in_executor")
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        # 三个同步函数在线程池中并发执行
        results = await asyncio.gather(
            loop.run_in_executor(None, sync_llm_call, "任务1"),
            loop.run_in_executor(None, sync_llm_call, "任务2"),
            loop.run_in_executor(None, sync_llm_call, "任务3"),
        )
        print(f"     总耗时: {time.perf_counter() - t0:.1f}s —— 并发的！")

    asyncio.run(bad_approach())
    print()
    asyncio.run(good_approach())

    print("\n[OK] 核心理解：")
    print("   永远不要在 async 函数里直接调用阻塞函数")
    print("   用 loop.run_in_executor() 把阻塞函数放到线程池")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  10. 综合示例 —— 模拟 HorizonRL-Agent 的 Worker 并发执行                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 这个示例整合了上面所有知识点，模拟项目中 Planner -> Workers 的完整流程：
#   - Planner 拆解 5 个子任务
#   - 3 个 Worker（受 Semaphore 限制）并发执行
#   - 每个 Worker 同步调用 LLM（用 run_in_executor）+ 搜索工具
#   - 超时控制 + 错误处理
#   - 结果汇总


def example_10():
    """综合示例：模拟完整的 Planner -> Workers 流程。"""

    print("\n" + "=" * 70)
    print("10. 综合示例 —— 模拟 Worker 并发执行")
    print("=" * 70)

    import random

    # --- 模拟同步 LLM 调用（用 run_in_executor 包装） ---
    def sync_generate(prompt: str) -> str:
        """模拟 langchain_openai.ChatOpenAI.invoke() 同步调用。"""
        time.sleep(random.uniform(0.3, 1.0))  # 模拟 LLM 推理时间
        return f"[LLM 回复] 关于 '{prompt[:30]}...' 的分析结果"

    # --- Worker 类（对应 src/horizonrl/agent/worker.py） ---
    class SimulatedWorker:
        """模拟的 AgentWorker。"""

        def __init__(self, worker_id: int, semaphore: asyncio.Semaphore):
            self.worker_id = worker_id
            self.semaphore = semaphore

        async def execute(self, task: dict) -> dict:
            """执行单个子任务。

            流程：
            1. 获取 Semaphore 许可（控制并发）
            2. 设置超时
            3. 调用 LLM（同步 -> run_in_executor）
            4. 调用搜索工具（异步）
            5. 返回结果
            """
            task_id = task["id"]

            # Step 1: 获取并发许可
            async with self.semaphore:
                print(f"    Worker-{self.worker_id} 开始执行 Task-{task_id}: {task['name']}")

                try:
                    # Step 2: 设置超时（整个子任务最多 3 秒）
                    result = await asyncio.wait_for(
                        self._do_work(task),
                        timeout=3.0,
                    )
                    return result

                except asyncio.TimeoutError:
                    return {
                        "task_id": task_id,
                        "worker_id": self.worker_id,
                        "success": False,
                        "error": "子任务超时（>3s）",
                    }
                except Exception as e:
                    return {
                        "task_id": task_id,
                        "worker_id": self.worker_id,
                        "success": False,
                        "error": str(e),
                    }

        async def _do_work(self, task: dict) -> dict:
            """实际执行工作：LLM推理 + 工具调用。"""
            loop = asyncio.get_running_loop()

            # Step 3: 调用 LLM（同步函数 -> 线程池）
            llm_response = await loop.run_in_executor(
                None,
                sync_generate,
                task["description"],
            )

            # Step 4: 模拟搜索工具调用（异步，不需要 run_in_executor）
            await asyncio.sleep(random.uniform(0.1, 0.5))  # 模拟网络IO

            return {
                "task_id": task["id"],
                "worker_id": self.worker_id,
                "success": True,
                "output": llm_response,
                "tool_calls": [{"tool": "search", "results": 3}],
            }

    # --- Planner 类（简化版） ---
    async def run_research_pipeline() -> None:
        """完整的 Planner -> Workers 流程。"""

        # Planner 把用户问题拆解为 5 个子任务
        tasks = [
            {"id": "T1", "name": "搜索Transformer架构最新论文",
             "description": "在arxiv搜索Transformer相关的最近论文"},
            {"id": "T2", "name": "搜索长上下文记忆方案",
             "description": "搜索长上下文记忆管理的现有方案"},
            {"id": "T3", "name": "搜索GRPO强化学习算法",
             "description": "搜索GRPO算法的原理和实现"},
            {"id": "T4", "name": "搜索Agent稳定性方法",
             "description": "搜索提高LLM Agent长链任务稳定性的方法"},
            {"id": "T5", "name": "汇总分析结果",
             "description": "将前四个搜索的结果汇总成研究报告"},
        ]

        # 最多 3 个 Worker 同时执行（Semaphore 控制）
        semaphore = asyncio.Semaphore(3)
        workers = [
            SimulatedWorker(worker_id=i, semaphore=semaphore)
            for i in range(3)
        ]

        print(f"Planner: 将用户任务拆解为 {len(tasks)} 个子任务")
        print(f"Planner: 启动 {len(workers)} 个 Worker 并发执行（最多3并发）")
        print()

        t0 = time.perf_counter()

        # 分配任务：轮询分配给 3 个 Worker
        # Worker-0 执行 T1, Worker-1 执行 T2, Worker-2 执行 T3
        # 谁先完成谁接着执行 T4, T5
        worker_tasks = []
        for i, task in enumerate(tasks):
            worker = workers[i % len(workers)]
            worker_tasks.append(worker.execute(task))

        # 并发执行所有任务，return_exceptions 保证单个失败不影响整体
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        elapsed = time.perf_counter() - t0

        # 统计结果
        success_count = 0
        fail_count = 0
        print(f"\n{'='*50}")
        print(f"执行结果汇总 (总耗时 {elapsed:.1f}s):")
        for r in results:
            if isinstance(r, Exception):
                print(f"  [FAIL] Worker异常: {r}")
                fail_count += 1
            elif r["success"]:
                print(f"  [OK] Task-{r['task_id']}: {r['output'][:50]}...")
                success_count += 1
            else:
                print(f"  [WARN]  Task-{r['task_id']}: {r['error']}")
                fail_count += 1

        print(f"\n成功率: {success_count}/{len(tasks)}")
        print(f"顺序执行预估耗时: ~{len(tasks) * 1.0:.1f}s")
        print(f"实际并发耗时: {elapsed:.1f}s")

    asyncio.run(run_research_pipeline())

    print("\n" + "=" * 70)
    print("教程结束！")
    print("=" * 70)
    print("""
回顾你学到的 10 个知识点：
  1. async/await  -> 定义协程，等待结果
  2. async sleep  -> 让出 CPU，不阻塞
  3. gather       -> 并发执行多个协程
  4. Semaphore    -> 控制并发数量（防止打爆 API）
  5. create_task  -> 启动后台任务
  6. as_completed -> 先完成先处理
  7. wait_for     -> 超时控制
  8. 错误处理     -> return_exceptions=True
  9. run_in_executor -> 同步代码异步化
  10. 综合应用    -> Planner + Worker 并发执行

下一步：examples/02_simple_agent.py —— 用 LangGraph 搭建第一个 Agent
""")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  main —— 运行入口                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

EXAMPLES = {
    "1": ("async/await 基础", example_1),
    "2": ("同步阻塞 vs 异步非阻塞", example_2),
    "3": ("asyncio.gather 并发执行", example_3),
    "4": ("asyncio.Semaphore 并发控制", example_4),
    "5": ("asyncio.create_task 后台任务", example_5),
    "6": ("asyncio.as_completed 先到先得", example_6),
    "7": ("asyncio.wait_for 超时控制", example_7),
    "8": ("错误处理 return_exceptions", example_8),
    "9": ("run_in_executor 同步异步化", example_9),
    "10": ("综合示例 Planner+Workers", example_10),
}


def main():
    """主函数：运行所有 asyncio 教学示例。"""
    import argparse

    parser = argparse.ArgumentParser(description="HorizonRL-Agent asyncio 完整教程")
    parser.add_argument(
        "--example", "-e",
        type=str,
        choices=list(EXAMPLES.keys()) + ["all"],
        default="all",
        help="运行哪个示例 (1-10, 默认全部)",
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HorizonRL-Agent asyncio 完整教程                            ║")
    print("║  覆盖项目所需的全部 asyncio 知识点                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if args.example == "all":
        for key in sorted(EXAMPLES.keys(), key=int):
            name, func = EXAMPLES[key]
            func()
            print()  # 示例间空行
    else:
        name, func = EXAMPLES[args.example]
        func()


if __name__ == "__main__":
    main()
