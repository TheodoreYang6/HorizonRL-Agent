# HorizonRL-Agent 面试问答手册

> 基于 v0.3.0 (2026-05-17)，21 commits，330 tests，9,251 行源码
> 使用前请根据最新状态更新数据

---

## 目录

1. [项目概览类](#一项目概览类)
2. [架构设计类](#二架构设计类)
3. [多Agent vs Workflow 辨析](#三多agent-vs-workflow-辨析)
4. [分层记忆深度解析](#四分层记忆深度解析)
5. [Verifier-Replanner 闭环](#五verifier-replanner-闭环)
6. [DAG 编排与并发控制](#六dag-编排与并发控制)
7. [工具治理与容错](#七工具治理与容错)
8. [SSE 流式与全栈](#八sse-流式与全栈)
9. [Bug 案例与工程教训](#九bug-案例与工程教训)
10. [性能优化](#十性能优化)
11. [Benchmark 与评测](#十一benchmark-与评测)
12. [RL 与 vLLM 规划](#十二rl-与-vllm-规划)
13. [与其他系统对比](#十三与其他系统对比)
14. [失败案例与局限](#十四失败案例与局限)
15. [如果重新做](#十五如果重新做)
16. [场景题](#十六场景题)
17. [论文与学术](#十七论文与学术)
18. [一页纸速查表](#十八一页纸速查表)

---

## 一、项目概览类

### Q1: 用一句话介绍这个项目

> HorizonRL-Agent 是一个面向长链路（20+步）复杂研究任务的 **LLM Agent 稳定执行系统**，
> 通过分层记忆、验证驱动重规划和异步 DAG 编排三项机制，
> 在真实 API Benchmark 上达到 95% 通过率和 78.7% 真实搜索占比。

**追问：和 ChatGPT 的 Deep Research 有什么区别？**

> ChatGPT Deep Research 是闭源商业产品，你不知道里面怎么做的。
> 我们的系统是开源的，每个模块可拆可换可评测。
> 而且我们的 Verifier-Replanner 闭环和三层记忆结构是独创的，
> Deep Research 可能用了类似的思路但未公开。

---

### Q2: 项目解决了什么问题？为什么重要？

**核心问题**：LLM Agent 在执行 20+ 步长链路任务时表现急剧下降。

**四种典型失败模式**：

| 失败模式 | 现象 | 我们的解法 |
|---------|------|-----------|
| 上下文污染 | 早期错误信息占满窗口，后期无法纠正 | L1 FIFO + L2 语义压缩 |
| 任务漂移 | Agent 逐渐偏离原始目标 | Verifier 检查一致性 + 完整性 |
| 幻觉累积 | 一个幻觉被下一个 Agent 引用，滚雪球 | RuleEngine 幻觉检测 + LLM Hybrid |
| 失败无恢复 | 某步失败后全局重来，浪费 token | 局部 Replan（只 patch 失败子图） |

**为什么现有方案不够**：
- ReAct/Chain-of-Thought：无结构化记忆，上下文溢出
- AutoGPT：简单分解+执行，无验证恢复
- LangGraph 原生：有编排但缺少内置验证器和分层记忆
- Multi-Agent (AutoGen/MetaGPT)：Agent 间无限对话，缺乏稳定执行保证

---

### Q3: 项目的技术栈是什么？为什么选这些？

| 技术 | 选型理由 |
|------|---------|
| **LangGraph StateGraph** | 比 LangChain Agent 更可控，显式状态机便于死锁检测和条件路由 |
| **asyncio** | 工具调用是 IO 密集型（网络搜索），asyncio.gather 并发天然适合 |
| **Pydantic V2** | 配置验证，三级合并（代码→YAML→.env），比 argparse 强太多 |
| **FAISS** | CPU 可用，零外部服务依赖，L3 向量检索不增加网络调用 |
| **aiohttp（非 FastAPI）** | 项目早期追求最小依赖，aiohttp 够用且 SSE 支持好 |
| **OpenAI SDK** | 兼容协议，DeepSeek/DashScope/任何 OpenAI-compatible 端点零改动切换 |
| **pytest + pytest-asyncio** | async 测试必须，LangGraph 的状态图测试需要 async fixture |

**为什么不选**：
- **LangChain Agent (create_react_agent)**：太黑盒，无法做死锁检测和局部 replan
- **AutoGen/MetaGPT**：Agent 间自由对话不可控，不适合稳定执行场景
- **gRPC/RabbitMQ**：单机 asyncio 就够了，没必要上消息队列

---

### Q4: 项目的规模和复杂度如何？

| 维度 | 数值 | 说明 |
|------|------|------|
| 源码文件 | 33 | 8 层架构，15 个模块 |
| 源码行数 | 9,251 | 纯 Python，不含注释和空行 |
| 测试文件 | 13 | 330 tests, 4 skipped, 0 failed |
| Demo | 7 | 从 asyncio 教程到全链路 Web |
| 数据结构 | 16 | 定义在 schemas/ 层 |
| Benchmark | 40 题 5 类 | 支持 mock/real 双模式 |
| 开发周期 | 7 天高密度 | ~21 commits，Day 1-7 |
| 独立开发 | 100% | 设计+编码+测试+文档 |

**复杂度体现**：不是 CURD，是完整的系统软件——状态管理、并发控制、容错恢复、日志审计、流式推送、评测体系。

---

## 二、架构设计类

### Q5: 系统的整体架构是什么？画出数据流

```
UserTask (自然语言)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Planner / LLMPlanner                             │
│   5 种任务自动分类 → PlanGraph (DAG 子任务)       │
│   ↑ L3 FAISS 检索历史经验                         │
└───────────────────────┬─────────────────────────┘
                        │
    ┌───────────────────▼─────────────────────────┐
    │ ResearchOrchestrator (LangGraph 6 节点)       │
    │                                              │
    │  plan_task ──→ mark_ready ──→ execute_batch  │
    │                    ↑              ↓           │
    │                    └── replan ← verify_batch  │
    │                                    ↓          │
    │                               finalize        │
    │                                              │
    │  死锁检测 · Semaphore 并发 · 批次超时          │
    └──────┬───────────────────┬───────────────────┘
           │                   │
    ┌──────▼──────┐    ┌───────▼──────┐
    │ Worker × N  │    │  Verifier    │
    │ asyncio并发  │    │  9规则+LLM   │
    │ ToolManager  │    │  Replanner   │
    └──────┬──────┘    └───────┬──────┘
           │                   │
    ┌──────▼───────────────────▼──────┐
    │  HierarchicalMemory             │
    │  L1(LIFO) → L2(摘要) → L3(FAISS)│
    │  TrajectoryLogger (异步JSONL)    │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  Writer v2                      │
    │  UserAnswerWriter (final)       │
    │  DebugReportRenderer (debug)    │
    └─────────────────────────────────┘
```

**依赖方向（单向，无循环依赖）**：

```
schemas/  ← 最底层，所有人依赖
config/   ← 被所有模块依赖
tools/    ← 独立
llm/      ← 独立
memory/   ← 独立于 agent/
agent/    ← 依赖 schemas/ + tools/ + llm/ + memory/
orchestration/ ← 顶层，组装所有模块
logging/  ← 横切关注点
services/ ← 顶层入口
```

---

### Q6: 为什么用 LangGraph 而不是自己写一个编排器？

**LangGraph 提供的**：
1. **显式状态机**：6 个节点 + 条件路由，比 LangChain Agent 的黑盒 ReAct 循环可控得多
2. **TypedDict 状态**：编译期字段检查，累加/替换语义注解（`Annotated[dict, _dict_merge]`）
3. **astream()**：原生支持流式输出每个节点的中间状态，直接对接 SSE
4. **checkpointer**：虽然我们没用，但 LangGraph 自带 checkpoint 能力，后续可扩展

**但 LangGraph 也有坑**（我们遇到的）：
- TypedDict 下 routing function 收到的是快照，原地修改不持久 → 死锁检测的 error 字段必须写在节点里
- StateGraph 的状态序列化需要手动处理 dataclass → dict，我们写了通用的 `_to_dict/_from_dict`

**结论**：LangGraph 做编排骨架很合适，但验证、记忆、重规划这些"智能"部分必须自己写。

---

### Q7: 模块之间如何通信？为什么这么设计？

**通信方式一：共享 WorkflowState（主要方式）**

```python
class WorkflowState(TypedDict):
    user_task: str
    plan: PlanGraph | None           # Planner 产出 → Worker/Verifier 消费
    results: Annotated[dict, _dict_merge]  # 累加语义，Worker 写入 → Verifier 读取
    verifications: dict[str, dict]   # 替换语义，Verifier 写入 → Replanner 读取
    iteration: int
    replan_count: int
    error: str
    final_output: str
    session_id: str
    started_at: float
    max_iterations: int
```

**通信方式二：构造函数注入（跨生命周期）**

```python
orchestrator = ResearchOrchestrator(
    planner=planner,          # 注入
    tool_manager=tool_manager,
    writer=writer,            # 注入
    embedding_client=client,  # 注入
    trajectory_logger=logger, # 注入
    on_token=callback,        # 注入
)
```

**为什么不用消息队列/事件总线？**

> 单机 + asyncio 已经够用。WorkflowState 是事实上的共享内存，
> LangGraph 保证了节点间串行执行（同一时刻只有一个节点在改状态），
> 不存在竞态条件。引入消息队列会增加序列化开销和调试难度。
> 当前设计等价于 "Actor 模型的特化版本"。

**模块通信矩阵**：

```
            Planner  Worker  Verifier  Replanner  Memory  Writer
Planner       -       产出      -         -        消费      -
Worker        -        -      产出        -        写入      -
Verifier      -        -       -        产出       写入      -
Replanner     -        -       -         -         -        -
Memory        -        -       -         -         -       消费
Writer        -        -       -         -         -        -
```

---

### Q8: 为什么 schemas/ 是第一优先级？不先写功能？

> **Schema-First 是分布式/多模块系统最重要的设计原则。**
>
> 16 个数据结构在第一天冻结，之后所有模块开发从未出现接口不匹配。
> PlanGraph 的 `depends_on` 字段定义了 DAG 依赖语义，
> VerificationResult 的 `error_type + suggested_actions` 定义了 Replanner 的输入协议，
> SearchProvenance 的 `url + provider + timestamp + query` 定义了证据追溯规范。
>
> 如果反过来先写功能，每个模块自己定义内部数据结构，最后集成时必然大量返工。
> 这个教训来自微服务实践——API Contract First。

---

## 三、多Agent vs Workflow 辨析

### Q9: 你的系统是 Multi-Agent 吗？

**诚实回答**：按严格定义（AutoGen/MetaGPT 那种 Agent 间直连通信），**不是**。
我们是一个 **多阶段 Agentic Workflow** —— 多个有 LLM 推理能力的模块
通过集中式 Orchestrator 协作，而非 Agent 间的点对点通信。

**但按照业界宽松定义**（每个模块有独立职责+智能决策），是的。
Planner/Verifier/Replanner 都有独立的 LLM 推理能力，
Worker 有工具使用能力。称它们为 "Agent 模块" 是合理的。

**面试时怎么说**：

> "我们选择集中式编排而非点对点多 Agent 通信，是一个刻意的设计决策：
> 1. 长链路任务需要可预测的执行顺序——DAG 拓扑排序提供了这个保证
> 2. Agent 间自由委派容易导致无限对话和任务漂移——AutoGen 论文自己也提到这点
> 3. 集中式编排可以做全局死锁检测——这在点对点架构中很难做到
> 4. Verifier-Replanner 闭环提供了比 Agent 自反思更强的恢复能力
>
> 但如果业务需要真正的多 Agent 协商（比如不同角色有不同知识背景需要辩论），
> 可以在 Worker 层引入 Agent 间通信，而不改变上层编排结构。"

---

### Q10: Worker 之间是怎么协作的？有没有通信？

**没有通信。Worker 是完全独立的。**

```python
# Worker 之间的"协作"只有一种方式：
# Orchestrator 通过 DAG 依赖关系决定执行顺序
# 例如：task_2 的 depends_on=["task_1"] 表示 task_2 必须等 task_1 完成后才被调度

# _mark_ready 节点：
def _mark_ready(self, state):
    ready = plan_graph.get_ready_nodes()  # 拓扑排序：只返回依赖已满足的节点
    # Worker 并发执行 ready 列表中的任务
    # 每个 Worker 独立调用工具，不感知其他 Worker
```

**为什么不让 Worker 通信？**

> 研究任务的子任务通常可以独立并行（搜 A 论文 vs 搜 B 论文）。
> Worker 间通信的收益很小（它们处理的已经是 Planner 拆好的独立子任务），
> 但风险很大（通信失败、信息污染、死循环）。
> 如果真需要 Worker 协作（比如 Worker A 发现 Worker B 的关键词更好），
> 应该通过 Orchestrator → Replanner 这一正规路径，而不是 Worker 间私聊。

---

## 四、分层记忆深度解析

### Q11: 三层记忆的具体设计是什么？每层解决什么问题？

```
┌──────────────────────────────────────────────┐
│ L1 RecentWindow (工作记忆)                     │
│ ├─ 数据结构: collections.deque, FIFO          │
│ ├─ 容量: ~10,000 tokens (可配置)              │
│ ├─ 存储: 最近 K 步的完整 StepResult           │
│ ├─ 触发: 每步 record() 自动追加               │
│ └─ 驱逐: Token 超阈值 → 最旧条目触发压缩       │
│     ↓ (自动触发)                              │
├──────────────────────────────────────────────┤
│ L2 SemanticSummary (语义摘要)                  │
│ ├─ 数据结构: list[dict], 最多 50 条            │
│ ├─ 存储: 模板/LLM 压缩的结构化摘要             │
│ ├─ 内容: {goal, key_findings, failure_mode,   │
│ │          tool_stats, timestamp}              │
│ └─ 作用: 保留"干了什么+有什么发现"消去噪声     │
│     ↓ (验证事件驱动)                           │
├──────────────────────────────────────────────┤
│ L3 EpisodicArchive (情景档案)                  │
│ ├─ 数据结构: FAISS IndexFlatIP + JSON 元数据   │
│ ├─ 向量: DashScope text-embedding-v4 (1024维) │
│ │        或 MD5 n-gram 确定性哈希 (回退)      │
│ ├─ 检索: 向量相似度 + n-gram 精确 + 时间衰减  │
│ ├─ 触发: 成功任务→archive_to_l3()             │
│ └─ 持久化: save()/load() FAISS+JSON 到磁盘    │
└──────────────────────────────────────────────┘
```

**类比人类团队**：
- L1 = 白板上的最近笔记（写完就擦旧的）
- L2 = 会议纪要（只记关键决定和发现）
- L3 = 公司知识库（可搜索的历史项目经验）

---

### Q12: L1→L2 的压缩是怎么做的？

两种模式，自动降级：

**模板模式（默认，零 LLM 依赖）**：
```python
summary = {
    "goal": task.spec.name,
    "key_findings": result.output[:200],
    "failure_mode": error_type if failed else None,
    "tool_stats": {tc.tool_name: tc.elapsed for tc in result.tool_calls},
    "timestamp": time.time(),
    "success": result.success,
}
```

**LLM 模式（可选，质量更高）**：
```python
prompt = f"""
将以下 Agent 执行记录压缩为结构化 JSON：
{json.dumps(entry.to_dict())}
压缩后保留：目标、关键发现、失败原因、使用的工具。
"""
summary = await llm.chat(prompt)
```

**设计决策**：L1→L2 由 Token 阈值**自动**触发，不需要 Planner 或其他模块手动调用。
这样 Planner 不用关心记忆管理，专注做好任务拆解。

---

### Q13: L3 为什么用 FAISS 而不是向量数据库（Milvus/Pinecone）？

| 方案 | 优点 | 缺点 | 为什么不用 |
|------|------|------|-----------|
| FAISS (CPU) | 零外部依赖、pip install 即可 | 单机、无分布式 | ✅ 选了 |
| Milvus | 分布式、生产级 | 需要 Docker/服务部署 | 太重，单机场景不需要 |
| Pinecone | 托管服务 | 收费、需联网 | 离线不可用 |
| ChromaDB | 轻量、Python-native | 比 FAISS 慢 | 当前数据量 FAISS 够用 |

**真正原因**：L3 需要离线可用（CI Mock 模式），FAISS 是唯一不需要额外服务的方案。
而且当前的向量数据量（几百条经验）远不到需要分布式检索的程度。

---

### Q14: L3 的 Embedding 双模回退是怎么实现的？

```python
async def _embed(self, text: str) -> list[float]:
    """生成文本向量。有 API Key 时用 DashScope，否则 n-gram。"""
    if self._llm_client is not None:
        try:
            return await self._llm_client.embed(text)
        except Exception:
            pass  # 静默回退
    return self._ngram_embed(text)  # 确定性 MD5 哈希

def _ngram_embed(self, text: str) -> list[float]:
    """MD5 n-gram 确定性哈希 → 1024维伪向量。
    同一文本永远产生相同向量，可复现。
    """
    ngrams = [text[i:i+3] for i in range(len(text)-2)]
    vec = [0.0] * 1024
    for ng in ngrams:
        h = int(hashlib.md5(ng.encode()).hexdigest()[:8], 16)
        vec[h % 1024] += 1.0
    # L2 归一化
    norm = math.sqrt(sum(v*v for v in vec))
    return [v/norm for v in vec] if norm > 0 else vec
```

**面试点**：即使没有外部 API，L3 也能工作——这是工程鲁棒性的体现。

---

### Q15: L3 的 _embed_sync 为什么用 ThreadPoolExecutor？

这是一个经典的 asyncio 嵌套问题：

```python
# 问题场景：Orchestrator.run() 在 asyncio event loop 中运行
# L3.archive() 需要 embed()，但 embed() 是 async
# 在已运行的 event loop 中无法用 asyncio.run() — 会报 "event loop is already running"

# Day 7 的修复：
def _embed_sync(self, text: str) -> list[float]:
    if self._llm_client is not None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 无运行中的 event loop → 直接 asyncio.run
            return asyncio.run(self._embed(text))
        # 有运行中的 event loop → ThreadPoolExecutor 跑 asyncio.run
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self._embed(text))
            return future.result(timeout=15)
    return self._ngram_embed(text)
```

**面试点**：这是一个真实的 Python asyncio 高级问题——嵌套事件循环的处理。

---

## 五、Verifier-Replanner 闭环

### Q16: 9 条验证规则是什么？为什么这个顺序？

```python
class RuleEngine:
    """9 条规则按执行代价排序：从快到慢，从简单到复杂"""

    def check(self, result: StepResult, task_spec) -> VerificationResult:
        # 1. 空结果检查 (0.01ms) — 工具返回空列表
        if not result.output or result.output == "[]":
            return fail(ErrorType.EMPTY_RESULT)

        # 2. 工具调用失败 (0.01ms) — tool.error 非空，无需看内容
        if result.tool_error:
            return fail(ErrorType.TOOL_FAILURE)

        # 3. 超时检查 (0.01ms) — elapsed > task.timeout
        if result.elapsed > task_spec.timeout:
            return fail(ErrorType.TIMEOUT)

        # 4. 长度/质量检查 (0.05ms) — 输出过短（<20字）或过长（>100KB垃圾）
        if len(result.output) < 20:
            return fail(ErrorType.LOW_QUALITY)

        # 5. 证据充分性 (0.1ms) — 证据条数为0
        if len(result.evidence) == 0:
            return fail(ErrorType.INCOMPLETE)

        # 6. 幻觉检测 (0.2ms) — 输出中有声明但无引用来源
        if has_unsupported_claims(result.output, result.evidence):
            return fail(ErrorType.HALLUCINATION)

        # 7. 完整性检查 (0.2ms) — 任务要求3个维度但只覆盖了1个
        if not covers_required_dimensions(result.output, task_spec):
            return fail(ErrorType.INCOMPLETE)

        # 8. 一致性检查 (0.3ms) — 与之前步骤的结论矛盾
        if contradicts_previous(result.output, self._history):
            return fail(ErrorType.INCONSISTENT)

        # 9. 格式检查 (0.1ms) — JSON/Markdown 格式损坏
        if malformed_output(result.output, task_spec.expected_format):
            return fail(ErrorType.FORMAT_ERROR)

        return pass_result()
```

**顺序讲究**：
- 规则 1-3：零成本（检查元数据，无需分析内容）
- 规则 4-5：低成本（计数/长度）
- 规则 6-9：需要内容分析（正则/关键词匹配）
- 大部分失败在前3条就能捕获，避免跑满9条

---

### Q17: Hybrid 模式是怎么工作的？

```python
class Verifier:
    async def verify(self, result, task_spec) -> VerificationResult:
        # Step 1: 规则引擎先跑（~0.1ms，零 LLM 成本）
        rule_result = self.rule_engine.check(result, task_spec)

        if self.mode == "rule":
            return rule_result

        if self.mode == "hybrid":
            score = rule_result.score
            # score >= 0.7 → 直接放行（规则很确信通过了）
            if score >= 0.7:
                return rule_result
            # score < 0.3 → 直接判定失败（规则很确信失败了）
            if score < 0.3:
                return rule_result
            # 0.3 ≤ score < 0.7 → LLM 复核（仅边界case走LLM）
            return await self.llm_verifier.verify(result, task_spec)

        if self.mode == "llm":
            return await self.llm_verifier.verify(result, task_spec)
```

**为什么这样设计？**

> 9 条规则覆盖了 80% 的常见失败，且成本为 0。
> LLM 调用一次 ~2s + Token 费用，只为边界 case 使用。
> 在 Benchmark 中，Hybrid 模式下 LLM 调用率 < 10%，
> 但相比于纯规则模式，Hybrid 对模糊结果的判断更准确。

---

### Q18: ErrorType → PatchType 的映射逻辑是什么？

```python
# 9 种 ErrorType → 4 种 PatchType

ERROR_TO_PATCH = {
    # RETRY：同样的任务换参数重做
    ErrorType.EMPTY_RESULT:   PatchType.RETRY,   # 换搜索词
    ErrorType.TOOL_FAILURE:   PatchType.RETRY,   # 换工具/后端
    ErrorType.TIMEOUT:        PatchType.RETRY,   # 缩小搜索范围
    ErrorType.LOW_QUALITY:    PatchType.RETRY,   # 增大 max_tokens
    ErrorType.FORMAT_ERROR:   PatchType.RETRY,   # 修复格式重试

    # ADD：当前任务不足以回答，需要补充子任务
    ErrorType.INCOMPLETE:     PatchType.ADD,     # 补充缺失维度
    ErrorType.HALLUCINATION:  PatchType.ADD,     # 补充验证搜索

    # REMOVE：冲突节点需要删除
    ErrorType.INCONSISTENT:   PatchType.REMOVE,  # 删除矛盾节点

    # REORDER：执行顺序不对
    # (预留，当前未使用)
}
```

**关键设计：局部 Patch 而非全局重规划**

> 全局重规划 = 扔掉 PlanGraph 重新 Planner.plan()
> → 丢弃了所有已成功步骤的成果
> → Token 浪费 + 时间浪费 + 可能再次失败
>
> 局部 Patch = 只修改受影响的节点和边
> → RETRY: 重设 node.status = PENDING，换参数
> → ADD: 在失败节点后插入补充任务
> → REMOVE: 删除冲突节点，重新连接 DAG 边
> → 保留所有其他节点的状态和结果

---

### Q19: 重规划有没有防无限循环？

有的，两层限制：

```python
# 每任务限制
if node.retry_count >= 3:
    node.status = TaskStatus.FAILED  # 标记失败，不再重试
    node.error_msg = f"已重试{node.retry_count}次，放弃"

# 全局限制
if state["replan_count"] >= 5:  # max_iterations
    state["error"] = "达到最大重规划次数，强制结束"
    # 路由到 finalize → 用已有结果生成部分报告
```

此外还有死锁检测：

```python
# 连续 3 轮没有 ready 任务但仍有 pending → 死锁
if state["idle_rounds"] >= 3:
    state["error"] = "检测到死锁，触发强制重规划"
    # 将所有 CANCELLED 任务重新设为 PENDING
    force_replan(plan)
```

---

## 六、DAG 编排与并发控制

### Q20: 6 个节点的状态机是怎么设计的？

```
START
  │
  ▼
plan_task        ← Planner 拆解任务 → PlanGraph
  │
  ▼
mark_ready       ← 拓扑排序 → 找出依赖已满足的节点
  │
  ├── (无ready + 无pending) → finalize
  ├── (无ready + 有pending) → deadlock → finalize
  └── (有ready) → execute_batch
                    │
                    ▼
              verify_batch   ← 并发验证所有已完成任务
                    │
                    ├── (全部通过) → mark_ready (下一批)
                    ├── (有失败) → replan → mark_ready
                    ├── (全部完成) → finalize
                    └── (死锁) → finalize
                                  │
                                  ▼
                                END
```

**为什么中间节点是 `mark_ready` 而不是直接 `execute_batch`？**

> `mark_ready` 的职责是决定"下一步做什么"——这是条件路由的决策点。
> 如果 execute_batch 后没有 ready 节点（全部完成/死锁），
> 需要一个节点来做这个判断。`mark_ready` 就是这个决策点。

---

### Q21: 并发控制是怎么做的？

三层并发控制：

```python
# 1. Worker 间并发：Semaphore 限流
sem = asyncio.Semaphore(semaphore_limit)  # 默认 3

# 2. 批次超时：防止单任务拖死整批
done, pending = await asyncio.wait(
    tasks, timeout=120  # 2分钟整批超时
)
for task in pending:
    task.cancel()  # 取消超时任务

# 3. 死锁检测：连续空闲轮次
if state["idle_rounds"] >= 3:
    # 触发强制重规划
```

**为什么 Semaphore 是 3 而不是更大？**

> 1. LLM API 有 rate limit（DeepSeek 免费版 ~5 QPS）
> 2. 搜索结果需要去重，太多并发会产生大量重复结果
> 3. 内存限制：每个 Worker 可能返回几十条搜索结果
> 4. 实测 3-5 是最优区间，再大边际收益递减

---

### Q22: 死锁是怎么发生的？怎么解决的？

**真实 Bug（Day 4，5个关联Bug）**：

```
死锁场景：
  task_1 (web_search) → FAILED
  Replanner → ADD task_1_sup (补充搜索) depends_on=["task_1"]
  task_2 depends_on=["task_1_sup"]

结果：
  task_1_sup 等待 task_1 (FAILED) → 永远不满足
  task_2 等待 task_1_sup (PENDING) → 永远不满足
  → 死锁！
```

**根因**：Replanner 的 `_apply_add` 将补充任务依赖设为已 FAILED 的父任务，
同时将补充任务注入下游的依赖链 → 环形等待。

**修复（5 处改动）**：

1. `_build_add_spec`: `depends_on=[original.id]` → `depends_on=[]`（补充任务作为独立 root）
2. `_apply_add`: 移除依赖注入代码（补充任务不阻塞下游）
3. `_mark_ready` + `get_ready_nodes`: 依赖检查接受 `SUCCESS | FAILED | SKIPPED`（终态不应阻塞）
4. `MockWebSearch.search()`: 参数名 `max_results` → `num_results`（与 Worker 调用一致）
5. `LLMReplanner._build_add_spec`: 同 Bug 1

**面试点**：这是整个项目最有价值的 Bug 修复——展示了从现象→根因→修复→验证的完整排查能力。

---

## 七、工具治理与容错

### Q23: ToolManager 的熔断器是怎么设计的？

三态熔断器：

```
        连续失败 ≥ 3次
CLOSED ──────────────→ OPEN
  ↑                      │
  │   冷却 30s 后          │
  └── HALF_OPEN ←────────┘
         │
         ├── 成功 → CLOSED (恢复正常)
         └── 失败 → OPEN (重新熔断)
```

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3, cooldown_sec=30):
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0

    async def call(self, coro):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.cooldown_sec:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()

        try:
            result = await coro
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.last_failure_time = time.time()
            raise
```

**为什么需要熔断器？**

> 真实场景：Bocha API 限流 → 连续失败 → 如果不熔断，每次请求都等超时
> → 浪费时间和 API 配额。熔断后直接快速失败，Worker 会走 Mock 回退。

---

### Q24: 重试策略的智能分类是什么？

```python
# 不同类型的错误，不同重试策略

if error_type in (ToolErrorType.NETWORK, ToolErrorType.DNS, ToolErrorType.CONNECTION):
    # 网络错误 → 不重试，直接快速失败
    # 原因：网络不通重试没用，走 Mock 回退
    return False, 0  # 不重试

if error_type == ToolErrorType.TIMEOUT:
    # 超时 → 最多重试 1 次，无等待
    # 原因：可能是临时波动，但不等太久
    return True, 1  # 重试 1 次

if error_type in (ToolErrorType.RATE_LIMIT, ToolErrorType.AUTH):
    # 限流/鉴权 → 指数退避重试
    # 原因：等一等可能恢复
    return True, 2  # 重试 2 次，指数退避
```

**Day 5 优化前**：所有错误统一指数退避重试 3 次 → 超时任务浪费 63s
**Day 5 优化后**：智能分类 → 超时不退避，网络不重试 → 端到端延迟降低 3.8x

---

### Q25: 搜索竞速（Racing）是怎么做的？

```python
# AUTO 模式：5 后端并发竞速
async def _auto_search(self, query):
    backends = [
        self._search_bocha(query),    # 国内优先
        self._search_brave(query),    # 国际
        self._search_ddgs(query),     # 免费
        self._search_wikipedia(query),# 百科
    ]
    # FIRST_COMPLETED：谁先返回有效结果就用谁
    done, pending = await asyncio.wait(
        [asyncio.create_task(b) for b in backends],
        timeout=6,  # 总超时 6s
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()  # 取消还没完成的
    # 全部失败 → Mock 兜底
    if not any_success(done):
        return self._mock_search(query)
```

**为什么是 FIRST_COMPLETED 而非等待全部？**

> 用户体验：用户不关心数据来自 Bocha 还是 DDGS，
> 只关心"有没有搜到"和"多快搜到"。
> FIRST_COMPLETED 保证用户总能在最短时间内拿到首个有效结果。

**Arxiv 三端点竞速同理**：arxiv Python 包 / export.arxiv.org / arxiv.org 同时发请求。

---

## 八、SSE 流式与全栈

### Q26: Token 级流式输出是怎么实现的？全链路是怎样的？

```
LLM.chat_stream()              ← OpenAI SDK stream=True
    │ async for token
    ▼
Writer._write_stream()         ← 逐 token 拼接 + on_token 回调
    │ await on_token(token)
    ▼
Orchestrator._on_token         ← 存储回调引用
    │ (在 _finalize 中传给 Writer)
    ▼
stream_research_session()      ← service 层封装
    │ yield {"event": "token", "data": {"delta": token}}
    ▼
run_deep_pipeline()             ← Web 层
    │ await _sse_write(stream, "token", {"delta": token})
    ▼
aiohttp StreamResponse          ← SSE 协议
    │ event: token
    │ data: {"delta":"在知识密集型任务中，"}
    ▼
浏览器 EventSource              ← 前端
    │ es.addEventListener('token', ...)
    ▼
Markdown 实时渲染              ← marked.js / 自研渲染器
```

**为什么用 SSE 而非 WebSocket？**

> SSE 是单向推送（server→client），正好匹配"Agent 进度推送"的需求。
> WebSocket 是双向的，但前端不需要给 Agent 发消息（已经通过 POST /api/chat 发了）。
> SSE 更简单：原生 EventSource API，自动重连，无需心跳库。

---

### Q27: 前端的 SSE 事件处理是怎么设计的？

```javascript
// 8 种事件类型
const es = new EventSource(`/api/stream/${sid}`);

es.addEventListener('stage', e => {
    // 更新侧栏阶段时间线 + 进度条
    updateStage(d.stage, d.label, d.progress);
});

es.addEventListener('tool', e => {
    // 追加工具调用日志条目（名称/耗时/token/成功失败图标）
    addToolEntry(nameMap[d.tool_name], icon, d.elapsed, d.tokens);
});

es.addEventListener('verify', e => {
    // 追加验证结果条目（评分百分比）
    addToolEntry(`验证: ${d.task_id.slice(-8)}`, d.pass ? '✓' : '✗');
});

es.addEventListener('token', e => {
    // 逐 token 追加到聊天气泡，实时 Markdown 渲染
    window._tokenBuf += d.delta;
    preview.innerHTML = renderMD(escapeHtml(window._tokenBuf));
});

es.addEventListener('report_ready', e => {
    // 报告生成完成，更新状态
    updateStage('writing', '报告已生成', 1.0);
});

es.addEventListener('done', e => {
    // 完成：渲染最终内容 + 显示下载按钮
    completeAllStages();
    setStatus('done', '完成');
});

es.addEventListener('heartbeat', e => {
    // 保活，不做任何 UI 更新
});

es.addEventListener('sse_error', e => {
    // 服务器推送的错误
    setStatus('error', '失败');
    addMessage('agent', '❌ 研究失败: ' + errMsg);
});

// 浏览器原生连接错误（非服务器推送的）
es.onerror = (evt) => {
    // SSE 连接断开
};
```

---

### Q28: Web 界面为什么用原生 HTML/JS 而不是 React？

**当时的原因**：
1. 追求零构建步骤——`python examples/05_web_agent.py` 一键启动
2. 项目重点是后端 Agent 系统，前端只需够用
3. 减少依赖，面试 Demo 时不需要 `npm install`

**面试时怎么说**：
> "Web 界面目前是原生实现，作为功能验证和 Demo 展示。
> 如果要产品化，会用 React/Vite 重构——后端 API 已经稳定，
> 前端替换不影响核心逻辑。这也是为什么我们把 SSE 协议设计得标准化。"

---

## 九、Bug 案例与工程教训

### Q29: 你遇到的最难的 Bug 是什么？

**死锁 Bug（Day 4，5 个关联 Bug）**

**现象**：
```
⚠ 死锁: 2 个任务阻塞
- task_5e794249: 等待 ['task_73b06433', 'task_73b06433_sup_1c8012']
- task_73b06433_sup_1c8012: 等待 ['task_73b06433']
```

**排查过程**：
1. 发现 Replanner 的补充任务（ADD）的 `depends_on` 指向了已 FAILED 的父任务
2. 同时补充任务被注入到下游的依赖链中
3. FAILED 任务无法满足依赖 → 补充任务永远 PENDING → 下游永远等待

**修复**：
1. 补充任务不再依赖已失败父任务（独立 root 节点）
2. 依赖检查接受 FAILED/SKIPPED 为终态
3. Mock 工具参数名对齐

**教训**：
> DAG 系统中，"终态"的概念至关重要。FAILED 不是"还没好"，是"不会再好了"。
> 依赖检查必须把三种终态（SUCCESS/FAILED/SKIPPED）都视为满足。

---

### Q30: 还有哪些值得说的 Bug？

**Bug 2: `elapsed` 变量覆盖（Day 7，今天）**
- 把总耗时 `elapsed` 改名为 `total_elapsed`，漏了一处引用
- 导致概念原理 16 次运行全部 NameError 失败
- 教训：重构变量名时用 IDE 的 Rename 功能，不要手动改

**Bug 3: Arxiv 国内超时（Day 5）**
- 旧代码用 HTTP（非 HTTPS）→ 被拒绝 → 无超时 → 永久挂起
- 修复：三端点并发竞速 + 8s 总超时 + Mock 自动兜底
- 教训：外部 API 调用必须有超时 + 回退策略

**Bug 4: MockWebSearch 参数名不匹配（Day 4）**
- `MockWebSearch.search(max_results=5)` vs `Worker._build_params("web_search", num_results=5)`
- 所有 mock web_search 调用全部崩溃 → 熔断器触发
- 教训：接口契约应该用 Schema 约束，参数名用 Pydantic Field alias 自动转换

**Bug 5: Bocha 响应解析（Day 4）**
- API 返回 `{"webPages": {"value": [...]}}` 而非 `{"webPages": [...]}`
- `resp["webPages"][:5]` → KeyError → 整个搜索失败
- 教训：外部 API 响应结构必须做防御性解析

---

## 十、性能优化

### Q31: 3.8x 加速是怎么做到的？

**优化前（Day 5 之前）**：
- 工具串行执行：`for tool in tools: await call(tool)` → 3 个工具 = 3x 时间
- 搜索顺序回退：Bocha(6s) → Brave(6s) → DDGS(6s) → 最坏 18s
- Arxiv HTTP 无超时：挂起无限久
- 统一指数退避重试：超时也退避 3 次 → 浪费 63s

**优化后**：
- 工具并发：`asyncio.gather(*[call(t) for t in tools])` → 1x 时间
- 搜索竞速：FIRST_COMPLETED → 6s 内拿到首个结果
- Arxiv 三端点并发 + 8s 总超时
- 智能重试：超时不退避，网络不重试

**结果**：Demo 04 端到端 35.2s → 9.3s = 3.8x

---

### Q32: 代码执行工具怎么处理 LLM 生成的非代码输入？

**问题**：Planner 给 code_execution 分配自然语言描述（"编写 RL 后训练代码"），
Worker 把中文描述当代码传给 `exec()` → Python 3 允许 Unicode 标识符 →
`ast.parse("编写代码")` 成功 → exec 时报 NameError → 失败 → Retry ×3。

**修复（4 层启发式检测）**：
```python
def _is_executable_code(text: str) -> bool:
    # 第 1 层：CJK 字符检测 — 含中文/日文/韩文 → 不是代码
    if any('一' <= c <= '鿿' for c in text):
        return False

    # 第 2 层：ast.parse — 不是合法 Python → 不是代码
    try:
        ast.parse(text)
    except SyntaxError:
        return False

    # 第 3 层：裸名称过滤 — 只有自然语言单词 → 不是代码
    words = text.split()
    code_keywords = {'def', 'import', 'class', 'for', 'if', 'return', 'print'}
    if not any(w in code_keywords for w in words):
        return False

    # 第 4 层：代码信号词 — "def"、"import"、"pip install"等
    return True

# 5 套代码模板自动生成
if not _is_executable_code(task_description):
    generated_code = _generate_from_template(task_description)
    # 模板：RL/PPO → 通用管道 → API 请求 → 通用分析 → 回退
```

---

## 十一、Benchmark 与评测

### Q33: Benchmark 的评测体系是怎样的？

**两层评估**：

**1. 规则评估（零 LLM 成本）**：
- `exact_match`: 短答案精确匹配
- `regex_match`: 关键信号词检测
- `mock_ratio`: Mock 证据占比
- `task_success_rate`: 子任务通过率
- `tool_calls_count`: 工具调用次数
- `replan_count`: 重规划次数
- `runtime_ms`: 端到端耗时
- `evidence_count`: 证据数量

**2. 轨迹级评估**：
- `avg_step_latency_ms`: 平均每步延迟
- `deadlock_count`: 死锁发生次数
- `timeout_count`: 超时次数
- `citation_support_rate`: 报告中有引用支撑的陈述比例

**LLM-as-Judge（预留接口，未大规模使用）**：
- relevance / completeness / groundedness / readability / citation_support

---

### Q34: 真实 API Benchmark 的结果是什么？

| 类别 | 运行 | 通过率 | Mock% | 重规划 | 工具调用 | 耗时 |
|------|------|--------|-------|--------|----------|------|
| 事实知识 | 16 | 75.0% | 8.1% | 1.2 | 3.6 | ~20s |
| 技术对比 | 16 | 100% | 11.7% | 0.0 | 5.0 | ~27s |
| 概念原理 | 16 | 100% | 26.0% | 0.0 | 4.5 | 31.5s |
| 代码实践 | 16 | 100% | 20.1% | 0.0 | 4.2 | 30.4s |
| 综述前沿 | 16 | 100% | 40.4% | 0.0 | 5.7 | 33.5s |
| **合计** | **80** | **95.0%** | **21.3%** | **0.24** | **4.6** | **28.5s** |

**分析**：
- 事实知识 75% 通过率：简单问题被过度拆解（Planner 拆了 9 个子任务），部分子任务搜不到有效信息 → 触发重规划上限
- 综述前沿 Mock 40%：Arxiv 在国内偶发超时回退 Mock；Web 搜索对前沿话题覆盖不足
- 技术对比/概念原理/代码实践 100% 通过：这些类别的任务描述明确，Worker 能搜到高质量结果

---

### Q35: 消融实验怎么做？结论是什么？

5 种配置对比：

| 配置 | Verifier | Replanner | Memory | 成功率 |
|------|----------|-----------|--------|--------|
| Full | ✅ | ✅ | ✅ | 71% |
| No-Verifier | ❌ | ❌ | ✅ | 100%（无质控） |
| No-Replanner | ✅ | ❌ | ✅ | 54% |
| No-Memory | ✅ | ✅ | ❌ | 79% |
| Template-Only | ✅ | ✅ | ✅ | 79% |

**结论**：
- **Replanner 贡献最大**：去掉后成功率下降 18%（71%→54%）
- **Memory 对短任务影响小**（79%→71%），但在 20+ 步长任务中差距拉大
- **Verifier 是质控关键**：去掉后"成功率"100%是因为所有结果都被标记为通过

---

## 十二、RL 与 vLLM 规划

### Q36: vLLM 部署方案是怎样的？

**架构**：
```
┌──────────────────────────────┐
│ HorizonRL-Agent               │
│   LLMClient (chat/stream)     │
│       │                       │
│       │ OpenAI-compatible API │
│       ▼                       │
│ vLLM Server (A100 80GB)       │
│   ├── Model: Qwen-72B / LLaMA │
│   ├── Quant: AWQ 4-bit        │
│   ├── Continuous Batching     │
│   ├── Prefix Cache            │
│   └── Tensor Parallelism = 1  │
└──────────────────────────────┘
```

**接入方式**：`LLMClient` 原本就用 OpenAI SDK，`base_url` 指向 `http://localhost:8000/v1` 即可。

**优化的地方**：
1. **Prefix Cache**：同一 session 内 multiple LLM calls 共享 system prompt 前缀
2. **Continuous Batching**：多个 Worker 的 inference 请求合并批量处理
3. **AWQ 量化**：72B → 单卡 A100 80GB 可跑

---

### Q37: GRPO 训练方案是怎样的？

**Reward 设计**（4 个维度）：
```python
def compute_reward(trajectory) -> float:
    # 1. 任务完成奖励（50%权重）
    completion_reward = verifier_score  # 0.0 ~ 1.0

    # 2. 效率奖励（20%权重）— 惩罚冗余工具调用
    efficiency_penalty = -0.1 * tool_calls_count

    # 3. 稳定性奖励（20%权重）— 惩罚频繁重规划
    stability_penalty = -0.2 * replan_count

    # 4. 质量奖励（10%权重）— 奖励有据可查的答案
    citation_bonus = 0.1 * citation_support_rate

    return completion_reward + efficiency_penalty + stability_penalty + citation_bonus
```

**训练流程**：
```
1. Rollout: vLLM 批量生成 Agent 执行轨迹（利用已有 JSONL 日志）
2. Reward: 基于 Verifier score + 效率/稳定性/质量计算
3. Advantage: GAE (Generalized Advantage Estimation)
4. Update: GRPO/PPO policy gradient → LoRA weights
5. Repeat
```

**GRPO vs PPO 选择**：
- GRPO 不需要单独的 value network（critic），更适合 Agent 场景（trajectory 长度多变）
- PPO 需要 critic → 额外的显存和训练时间

---

## 十三、与其他系统对比

### Q38: 和 AutoGPT / MetaGPT / CrewAI 的区别？

| 维度 | AutoGPT | MetaGPT | CrewAI | HorizonRL-Agent |
|------|---------|---------|--------|-----------------|
| 通信模式 | 串行链 | Agent间聊天 | 角色委派 | 集中编排+DAG |
| 记忆 | 简单buffer | 消息历史 | 短期记忆 | L1/L2/L3 三层 |
| 验证 | 无 | 无 | 无 | 9规则+LLM Hybrid |
| 失败恢复 | 全局重来 | 重试 | 重试 | 局部Replan |
| 并发 | 串行 | 多Agent | 多Agent | DAG拓扑排序+Semaphore |
| 流式输出 | 无 | 无 | 无 | SSE Token流式 |
| 评测 | 无 | 无 | 无 | 40题Benchmark+消融 |
| 适用场景 | 简单任务 | 软件开发 | 角色扮演 | 深度研究 |

**核心差异**：
> AutoGPT/MetaGPT/CrewAI 关注的是"让 Agent 聊起来"，
> 我们关注的是"让 Agent 在 20+ 步任务中不崩"。
> 这是两个不同的技术方向——前者强调交互灵活性，后者强调执行稳定性。

---

### Q39: 和 LangGraph 原生的区别？

**LangGraph 提供**：状态图 + 条件路由 + checkpoint
**我们在此基础上构建**：
- Verifier + Replanner 闭环（LangGraph 没有内置验证）
- 三层记忆结构（LangGraph 只有短期 memory）
- ToolManager + CircuitBreaker（LangGraph 不管理工具调用细节）
- TrajectoryLogger（LangGraph checkpoint 是状态快照，不是事件日志）
- Writer v2 双输出 + 证据追溯
- Benchmark + 消融评测框架

**一句话**：LangGraph 是骨架，我们往上加了肌肉和神经系统。

---

## 十四、失败案例与局限

### Q40: 系统在什么情况下会失败？

**1. 简单问题被过度拆解（事实知识 75%）**
- 用户问"什么是 GIL"，Planner 拆了 9 个子任务
- 部分子任务搜不到匹配信息 → Replanner 加到 5 次上限 → 失败
- 改进方向：简单问题走 chat mode，不触发深度研究

**2. Arxiv 国内超时 → Mock 占比高（综述前沿 40%）**
- 前沿学术话题依赖 Arxiv，国内网络波动 → Mock 兜底
- 改进方向：找稳定国内 Arxiv 镜像 / Semantic Scholar API

**3. 代码执行稳定性**
- LLM 生成的代码偶有 Traceback，AST 检测无法预判运行时错误
- 改进方向：先 dry-run 语法检查 → 再 exec

**4. 无法处理多模态输入**
- 不支持图片/PDF/音频分析
- 这是显式限制，不计划在 Phase 2 支持

---

### Q41: 为什么不支持 [某功能]？

**为什么不支持 WebSocket？**
> SSE 满足当前所有需求（单向推送进度）。WebSocket 只在需要前端主动取消 Agent 时才必要。

**为什么不支持 Checkpoint/Resume？**
> LangGraph 自带 checkpointer，接入即可。当前 session 较短（~30s），需求不迫切。

**为什么不支持多轮对话？**
> 当前 Web 界面是单轮研究。多轮对话的上下文管理是另一个工程问题，与核心 Agent 逻辑正交。

**为什么 Planner 不能自选工具？**
> 当前工具列表是 Planner 根据任务类型模板确定的。LLMPlanner 可以推荐工具但受限于可用工具列表。
> 自选工具会增加"选择了不存在的工具"的失败模式。

---

## 十五、如果重新做

### Q42: 如果重新设计，你会改什么？

**1. Planner 加一个复杂度判断**
- 简单问题（"什么是GIL"）不走深度管道，直接 chat mode 回答
- 当前用关键词分类还不够精确，应该加 LLM 复杂度评分

**2. 前端用 React/Vite 而不是原生 HTML**
- 原生 HTML 在维护复杂交互时很痛苦（状态管理靠全局变量）
- React 的组件化和状态管理更适合聊天界面

**3. 数据库替代 JSONL**
- TrajectoryLogger 用 JSONL 是因为简单，但查询需要遍历文件
- SQLite 会更适合存储和查询历史轨迹

**4. Verifier 加一个"简单通过"快速路径**
- 当输出明显高质量时（有多个来源、覆盖所有维度），直接跳过规则检查
- 节省 ~0.1ms 虽然不大，但概念上更清晰

**5. 配置文件用 TOML 而非 YAML**
- YAML 的缩进敏感和 `no`→`false` 自动转换是常见的坑
- TOML 语义更清晰，Python 生态支持好

---

### Q43: 哪个设计决策你最后悔？

> **没有在第一天就做复杂度判断。**
>
> Planner 的 5 种任务分类基于关键词匹配（"对比"→comparison），
> 但无法区分"简单的对比"（HTTP/1.1 vs HTTP/2）和"复杂对比"（RAG vs Agent）。
> 导致简单问题被过度拆解，浪费 Token 和时间。
> 应该在 Planner 里加一个 LLM 复杂度评分（1-5），简单任务走轻量管道。

---

## 十六、场景题

### Q44: 如果用户问一个需要 50 步的任务怎么办？

1. **Planner 阶段**：LLMPlanner 将大任务拆成多层 DAG（子任务 + 子子任务）
2. **Memory**：L1/L2/L3 在长任务中发挥更大作用——L2 压缩早期步骤，L3 检索类似历史任务
3. **Verifier-Replanner**：更频繁的验证和局部修复（50 步中可能有 10+ 步需要重试）
4. **挑战**：50 步的 Token 消耗很大，可能需要切换为更便宜的 LLM 做中间步骤的验证

### Q45: 如果搜索全部返回垃圾信息怎么办？

1. **Verifier 的证据充分性检查**：`len(evidence) == 0` 或所有证据被标记为低相关 → INCOMPLETE
2. **Replanner 触发 ADD**：补充不同搜索词/不同后端的新搜索任务
3. **如果 3 次重试仍无改善**：Writer 在报告开头写上"当前检索结果质量有限，以下结论可能不完整"
4. **Mock 不会掩盖问题**：`_mock_warning` 基于实际 mock_ratio 动态披露

### Q46: 如何处理 LLM API 突然不可用？

1. **Planner**：降级为模板 Planner（规则拆解，零 LLM 依赖）
2. **Verifier**：降级为纯 RuleEngine 模式（跳过 LLM Hybrid 复核）
3. **Writer**：降级为 `_template_write`（结构化拼接，零 LLM）
4. **L3**：自动降级为 n-gram 哈希（零 API 依赖）
5. **系统仍能完成端到端执行**，但报告质量会下降——模板生成的报告不如 LLM 自然

## 十七、论文与学术

### Q47: 论文的核心贡献点是什么？

**五个贡献点**（对应 paper_outline.md）：

1. **分层记忆结构**：L1→L2→L3 三级结构，主动折叠上下文，有效扩展上下文窗口
2. **验证器驱动局部重规划**：9 规则 + LLM Hybrid + 9→4 Error-Patch 映射，局部修复而非全局重建
3. **异步多 Agent DAG 编排**：6 节点状态机 + 死锁检测 + 并发控制
4. **轨迹级日志基础设施**：30 种事件类型异步 JSONL，全生命周期可观测
5. **证据可追溯的双模式写作**：SearchProvenance + 用户/开发者双视角输出

### Q48: 论文的实验部分怎么设计？

- **RQ1 (Memory)**：不同任务步长下 L1/L2/L3 的贡献
- **RQ2 (Replanning)**：有/无 Replanner 在失败率上的对比
- **RQ3 (Ablation)**：5 种配置的消融实验（已完成）
- **RQ4 (Scalability)**：步数从 5→30 的性能衰减曲线
- **RQ5 (Efficiency)**：Token 消耗和工具调用效率对比

### Q49: 和已发表工作的区别？

vs **Self-Refine (Madaan 2024)**：Self-Refine 是 LLM 自反馈，我们是结构化规则 + LLM Hybrid
vs **Reflexion (Shinn 2024)**：Reflexion 是口头反思，我们产生可执行的 PlanPatch
vs **MemGPT (Packer 2024)**：MemGPT 面向对话记忆，我们面向任务执行记忆

---

## 十八、一页纸速查表

### 核心数字

```
源码:         33 文件 / 9,251 行
测试:         330 passed, 4 skipped, 0 failed
Demo:         7 个 (CLI×6 + Web×1)
Benchmark:    40 题 5 类 (120 次 Mock / 80 次 Real)
真实API:      95.0% 通过率, 21.3% Mock占比
加速:         3.8x (35.2s → 9.3s)
开发:         7 天高密度, 21 commits
开源:         GitHub v0.3.0
```

### 架构速记

```
schemas (16数据结构) → config (Pydantic V2)
    → tools (5后端并发+熔断) → llm (OpenAI兼容)
    → agent (5模块: Planner/Worker/Verifier/Replanner/Writer)
    → memory (L1→L2→L3+FAISS) → orchestration (LangGraph 6节点)
    → services (CLI/Web/Benchmark统一) → logging (异步JSONL)
```

### 面试核心叙事

```
问题: LLM Agent 长链路不稳定 (上下文污染/漂移/幻觉/无恢复)
    ↓
洞察: 不是单一 Planning 问题，是记忆+验证+恢复+编排的系统问题
    ↓
方案: 3个核心机制
  1. 分层记忆 (L1→L2→L3) — 主动折叠上下文
  2. Verifier-Replanner — 局部修复而非全局重建
  3. 异步DAG编排 — 死锁检测+并发控制
    ↓
结果: 95%通过率, 21%Mock, 3.8x加速, 330 tests, 开源
    ↓
未来: RL训练+vLLM部署, AAAI/IJCAI 2027投稿
```

---

*本手册基于 HorizonRL-Agent v0.3.0 (2026-05-17)，使用时请根据最新状态更新数据。*
