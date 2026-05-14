# HorizonRL-Agent 系统开发手册 v3.0

> 最后更新: 2026-05-14 (Day 3)
> 测试: 323 passed, 4 skipped, 0 failed
> 代码量: ~16,000 行
> Phase 1+2 全部完成, P0/P1/P2 集成完毕

---

## 目录

1. [系统概述](#一系统概述)
2. [架构全景](#二架构全景)
3. [数据层 — schemas/](#三数据层--schemas)
4. [配置层 — config/](#四配置层--config)
5. [工具层 — tools/](#五工具层--tools)
6. [LLM 调用层 — llm/](#六llm-调用层--llm)
7. [Agent 业务层 — agent/](#七agent-业务层--agent)
8. [编排层 — orchestration/](#八编排层--orchestration)
9. [记忆层 — memory/](#九记忆层--memory)
10. [日志层 — logging/](#十日志层--logging)
11. [可运行 Demo](#十一可运行-demo)
12. [用户面向功能](#十二用户面向功能)
13. [测试体系](#十三测试体系)
14. [配置指南](#十四配置指南)
15. [完整数据流追踪](#十五完整数据流追踪)
16. [已知限制与后续改进空间](#十六已知限制与后续改进空间)

---

## 一、系统概述

### 1.1 一句话定位

给 LLM 配一支"研究团队"——输入问题，系统自动**分解任务 → 并行搜索 → 交叉验证 → 失败重试 → 合成自然语言报告**。

### 1.2 核心创新（v2 全部实现）

| 创新 | 实现位置 | 状态 |
|------|---------|------|
| Verifier 驱动重规划 | `agent/verifier.py` + `replanner.py` → 接入 `dag_workflow.py` | ✅ v2 全集成 |
| 分层记忆 L1→L2→L3 | `memory/hierarchical_memory.py` → 接入主循环 | ✅ v2 全集成 |
| L3 n-gram 向量检索 | `L3EpisodicArchive._ngram_embed()` + 混合检索 | ✅ MD5 确定性 |
| 异步多 Agent DAG | `orchestration/dag_workflow.py` LangGraph StateGraph | ✅ 6 节点 |
| 轨迹日志一等公民 | `logging/trajectory_logger.py` 30 种事件 JSONL | ✅ |
| 证据可追溯 | `schemas/result.py` SearchProvenance | ✅ |
| Writer 双输出 | `agent/writer.py` final_answer.md + debug_report.md | ✅ 接入 _finalize |

### 1.3 核心数据流（v2 最终）

```
UserTask (自然语言)
    │
    ▼
Planner / LLMPlanner ──→ PlanGraph (TaskSpec[] + DAG 依赖)
    │
    ▼
ResearchOrchestrator (LangGraph StateGraph, 6 节点)
    │
    │   ┌── plan_task ──→ mark_ready ──→ execute_batch ──→ verify_batch
    │   │    (规划)         (调度)         (并发执行)        (并行验证)
    │   │                                                     │
    │   │                                    ┌────────────────┼────────────┐
    │   │                                    │ continue       │ replan     │ done/deadlock
    │   │                                    ▼                ▼            │
    │   │                              mark_ready          replan          │
    │   │                              (下一轮)         (局部修复)         │
    │   │                                    ▲                │            │
    │   │                                    └────────────────┘            │
    │   │                                                               ▼
    │   └──────────────────────────────────────────────────────→ finalize → END
    │                                                        (Writer v2 双输出)
    │
    ├──→ Verifier (rule/hybrid/llm) → VerificationResult
    │       │
    │       └──→ Replanner → PlanPatch (RETRY/ADD/REMOVE/REORDER)
    │
    ├──→ HierarchicalMemory
    │       L1 (FIFO 窗口) → L2 (语义摘要) → L3 (FAISS n-gram 向量检索)
    │
    ├──→ TrajectoryLogger (异步 JSONL, 30 种事件)
    │
    └──→ Writer
            ├── UserAnswerWriter → final_answer.md (用户, 无调试信息)
            └── DebugReportRenderer → debug_report.md (开发者)
```

---

## 二、架构全景

### 2.1 模块结构

```
src/horizonrl/
├── schemas/           数据协议层 (4 文件, 1061 行) — 全项目通信协议
│   ├── task.py        TaskSpec, PlanGraph, PlanNode, PlanPatch, UserTask
│   ├── result.py      StepResult, VerificationResult, EvidenceItem, ToolCall
│   ├── event.py       TrajectoryEvent, TrajectorySession, EventType (30 种)
│   └── report.py      FinalReport, ReportSection, CitationMap, ReportMetadata
│
├── config/            配置管理层 (1 文件, 686 行)
│   └── settings.py    Pydantic V2: LLMConfig, MemoryConfig, AgentRuntimeConfig
│
├── tools/             工具层 (5 文件, 834 行)
│   ├── manager.py     ToolManager: 超时/重试/熔断/统计, CircuitBreaker
│   ├── web_search.py  Brave/DDGS/Wikipedia 4 级自动回退
│   ├── arxiv_search.py Arxiv API
│   ├── code_execution.py subprocess 沙箱
│   └── mock.py        Mock 工具 (CI/离线可用)
│
├── llm/               LLM 调用层 (1 文件, 185 行)
│   └── client.py      LLMClient: chat() + embed(), OpenAI-compatible
│
├── agent/             Agent 业务层 (5 文件, ~1950 行)
│   ├── planner.py     Planner (模板 2 类) + LLMPlanner (LLM 驱动 DAG 拆解)
│   ├── worker.py      AgentWorker (异步执行 + 证据提取) + execute_workers()
│   ├── verifier.py    Verifier (rule/llm/hybrid), RuleEngine (9 道检查)
│   ├── replanner.py   Replanner + LLMReplanner, 9 种 ErrorType→PatchType 策略
│   └── writer.py      Writer v2: UserAnswerWriter + DebugReportRenderer
│
├── orchestration/     编排层 (1 文件, 835 行)
│   └── dag_workflow.py ResearchOrchestrator: LangGraph 6 节点状态机
│
├── memory/            记忆层 (1 文件, ~750 行)
│   └── hierarchical_memory.py L1RecentWindow + L2SemanticSummary + L3EpisodicArchive
│
├── logging/           日志层 (1 文件, 411 行)
│   └── trajectory_logger.py TrajectoryLogger (异步 JSONL) + 5 分析工具
│
├── eval/              评测指标 (Phase 4 占位)
└── rl/                RL 训练 (Phase 3+ 占位)
```

### 2.2 依赖方向

```
schemas/  ← 最底层，全项目数据协议（零依赖）
    ↑
config/   ← 被所有模块依赖
    ↑
tools/  llm/  memory/  ← 独立模块
    ↑      ↑      ↑
    └──────┼──────┘
           ↑
        agent/  ← Planner, Worker, Verifier, Replanner, Writer
           ↑
    orchestration/  ← 顶层，组装所有模块
           ↑
        logging/  ← 横切关注点，所有模块写入
```

---

## 三、数据层 — schemas/

全项目 16 个数据结构，定义在 4 个文件中。所有模块通过 Schema 通信，不直接传递裸 dict。

### 3.1 schemas/task.py — 任务相关 (320 行)

| 类 | 用途 | 关键字段 |
|----|------|---------|
| `UserTask` | 用户输入的自然语言问题 + 约束 | description, max_steps, max_tokens, required_tools |
| `TaskSpec` | Planner 拆解出的单个子任务 | id, name, description, tool_names, depends_on, priority, retry_count |
| `TaskStatus` | 生命周期状态 | PENDING→READY→RUNNING→SUCCESS/FAILED/SKIPPED/CANCELLED |
| `TaskPriority` | 优先级 | P0 (关键路径) / P1 (正常) / P2 (后置) |
| `PlanNode` | PlanGraph 中的一个节点 | 包装 TaskSpec + 运行时状态 (status, error_msg, started_at) |
| `PlanGraph` | 完整任务有向无环图 | nodes{}, edges{}, root_ids[], get_ready_nodes(), has_pending_work() |
| `PlanPatch` | Replanner 的局部修改 | patch_type, target_node_id, new_spec, reason |
| `PatchType` | 修改类型 | ADD / REMOVE / REORDER / RETRY |

**实际使用**:
```python
from horizonrl.schemas.task import UserTask, TaskSpec, PlanGraph
task = UserTask(description="Transformer注意力机制", max_steps=20)
# Planner.plan(task) → PlanGraph (含 5 个 TaskSpec + DAG 依赖)
```

### 3.2 schemas/result.py — 执行结果 (256 行)

| 类 | 用途 | 关键字段 |
|----|------|---------|
| `ToolCall` | 单次工具调用记录 | tool_name, input, output, elapsed, error, tokens_used |
| `EvidenceItem` | 单条证据 (可追溯) | content, source, source_type, provider, search_query, is_mock, provenance |
| `SearchProvenance` | 搜索来源追溯 | provider, query, timestamp, raw_snippet, score, url, is_mock |
| `StepResult` | Worker 执行结果 | task_id, success, output, evidence[], tool_calls[], elapsed |
| `VerificationResult` | Verifier 验证结论 | pass_, score, error_type, feedback, evidence_gaps, suggested_actions |
| `ErrorType` | 9 种错误类型 | NONE/EMPTY_RESULT/CODE_ERROR/TOOL_ERROR/OFF_TOPIC/FACTUAL_ERROR/INCOMPLETE/HALLUCINATION/OTHER |

**provenance 追溯链**: 每条 EvidenceItem 记录 provider (搜索平台)、search_query (实际查询词)、is_mock (是否模拟数据)。Writer v2 在生成 final_answer 时利用这些信息生成来源引用。

### 3.3 schemas/event.py — 轨迹事件 (248 行)

| 类 | 用途 |
|----|------|
| `EventType` | 30 种事件类型枚举: plan.start→worker.complete→tool.call→verify.fail→replan.patch→session.end |
| `TrajectoryEvent` | 单条事件: ts, module, event_type, payload, cost, latency, session_id, step_id |
| `TrajectorySession` | 完整会话: events[], to_summary(), filter_by_module(), filter_by_type() |

### 3.4 schemas/report.py — 报告 (168 行)

| 类 | 用途 |
|----|------|
| `CitationMap` | 声明↔证据引用映射 |
| `ReportSection` | 报告章节: title, content, citations |
| `FinalReport` | 完整报告: title, sections[], summary, metadata |
| `ReportMetadata` | 生成元数据: session_id, author, mode, used_mock_data, llm_writer_used |

---

## 四、配置层 — config/

### 4.1 settings.py (686 行)

**三级配置合并**: `代码默认值 → YAML 文件 → .env 环境变量`

```python
from horizonrl.config.settings import load_config
cfg = load_config(Path("configs/dev.yaml"))
# 配置优先级: dev.yaml > default.yaml > 代码默认值
# 环境变量 HORIZON_XXX 可覆盖任何字段
```

| 配置类 | 控制内容 | 默认值 |
|--------|---------|--------|
| `LLMConfig` | provider, model, api_key, base_url, temperature, max_tokens | gpt-4o, temp=0.3 |
| `MemoryConfig` | l1_max_tokens (8000), l2_max_entries (50), auto_compress_threshold (0.8) | — |
| `AgentRuntimeConfig` | max_steps (30), semaphore_limit (3), task_timeout (120s), max_retries (3) | — |
| `ToolsConfig` | web_search, arxiv_search, code_execution 子配置 | — |
| `TrainingConfig` | RL 训练参数 (Phase 3+ 占位) | — |

**配置文件**:
- `configs/default.yaml` — 生产环境 (gpt-4o)
- `configs/dev.yaml` — 开发环境 (DeepSeek/deepseek-chat)
- `configs/eval.yaml` — 评测环境 (temperature=0)

**环境变量覆盖规则**:
```bash
HORIZON_LLM__MODEL=gpt-4o              # 双下划线 = 嵌套边界
HORIZON_AGENT__MAX_STEPS=20
HORIZON_MEMORY__L1_MAX_TOKENS=6000
OPENAI_API_KEY=sk-xxx                   # 自动注入到 llm.api_key
```

---

## 五、工具层 — tools/

### 5.1 manager.py — 工具管理器 (426 行)

**核心**: 统一工具调用入口。所有 Worker 必须通过 ToolManager 调用工具，不允许直接调用。

```python
mgr = ToolManager()
mgr.register("web_search", WebSearchTool())
result = await mgr.call(ToolCallRequest(tool_name="web_search", params={"query": "..."}))
```

| 组件 | 功能 | 关键参数 |
|------|------|---------|
| `CircuitBreaker` | 三态熔断: CLOSED→(5 次失败)→OPEN→(60s 冷却)→HALF_OPEN | failure_threshold=5, cooldown=60s |
| `ToolManager._invoke()` | 自动适配 3 种接口: async execute() / async search() / __call__() | — |
| `ToolManager.call()` | 完整流程: 熔断检查 → 超时控制 → 指数退避重试 → 统计 | timeout=20s, max_retries=2 |
| `ToolManager.normalize_search_results()` | 搜索结果统一规范化为带 provenance 的 dict 列表 | — |
| `ToolStats` | 每工具独立统计: total/success/failure/timeout/latency | — |
| `ToolErrorType` | 8 种错误分类: TIMEOUT/CIRCUIT_OPEN/AUTH/RATE_LIMIT/NETWORK/UNREGISTERED/INTERNAL/UNKNOWN | — |

### 5.2 web_search.py — 网页搜索

**4 级后端自动回退**: `Brave API (有 Key) → DDGS (国内可用) → Wikipedia API → Mock`

每个后端 8 秒超时，失败自动降级。国内 DDGS 偶尔超时，建议注册 Bocha Key。

### 5.3 arxiv_search.py — 学术论文搜索

直接调用 Arxiv API，返回真实论文标题、摘要、PDF 链接。

### 5.4 code_execution.py — 代码执行

subprocess 沙箱执行 Python 代码，10 秒超时。

### 5.5 mock.py — 模拟工具 (107 行)

离线/CI 环境使用的模拟工具。`register_mock_tools(mgr)` 一键注册全部 mock 工具。输出格式与真实工具一致，带 `is_mock=True` 标记。

---

## 六、LLM 调用层 — llm/

### 6.1 client.py — LLM 客户端 (185 行)

OpenAI 兼容的异步客户端。已打通 DeepSeek API。**v2 新增 embed() 方法**。

```python
from horizonrl.llm.client import LLMClient, LLMCallResult, EmbedResult

client = LLMClient(cfg.llm)

# Chat Completion
result = await client.chat("你好", system_prompt="你是一个助手", max_tokens=500)
print(result.content, result.tokens_total, result.elapsed)

# Embedding (v2 新增)
emb = await client.embed("需要向量化的文本")
print(emb.embedding[:10], emb.tokens_used)
```

| 类/方法 | 用途 | 返回 |
|---------|------|------|
| `LLMClient(config)` | 延迟初始化 AsyncOpenAI 客户端 | — |
| `chat(prompt, system_prompt, temperature, max_tokens)` | Chat Completion | `LLMCallResult` |
| `chat_sync(prompt, system_prompt)` | 同步 Chat (asyncio.run 包装) | `LLMCallResult` |
| `embed(text)` | Embedding API (v2 新增) | `EmbedResult` |
| `LLMCallResult` | content, model, tokens_prompt/completion/total, elapsed, error | — |
| `EmbedResult` | embedding (list[float]), tokens_used, elapsed, error | — |

**支持所有 OpenAI 兼容 API**: DeepSeek, OpenAI, vLLM, 任何 `/v1/chat/completions` 和 `/v1/embeddings` 端点。

---

## 七、Agent 业务层 — agent/

### 7.1 planner.py — 任务分解 (340 行)

**Planner (模板模式)**: 无 LLM 依赖，2 种任务模板：
- `research` 模板: 检索背景 → 最新进展 → 分析对比 → 局限性 → 综合汇总 (5 子任务)
- `code` 模板: 理解代码 → 定位问题 → 运行 → 修复 → 验证 (5 子任务)

**LLMPlanner (LLM 模式)**: 调用 LLM 智能拆解任意类型任务。LLM 失败时自动回退到模板 Planner。

```python
# 模板模式 (离线可用)
planner = Planner()
plan = planner.plan(UserTask(description="Transformer注意力机制"))

# LLM 模式 (需要 API)
planner = LLMPlanner(llm_client)
plan = await planner.plan(UserTask(description="任意复杂问题"))
# → PlanGraph (4-7 个子任务, DAG 依赖, 工具分配)
```

**v2 优化**: `_build_plan_graph(specs)` 模块级函数，Planner 和 LLMPlanner 共用，消除 28 行重复代码。

### 7.2 worker.py — 任务执行 (300 行)

**AgentWorker**: 执行单个 TaskSpec，调用工具，提取证据。

```python
worker = AgentWorker(worker_id="wrk_1", tool_manager=mgr)
result = await worker.execute(task_spec)
# → StepResult (含 output, evidence[], tool_calls[])
```

**execute_workers()**: 批量并发执行，受 asyncio.Semaphore 控制。

```python
results = await execute_workers(tasks, tool_manager, semaphore=asyncio.Semaphore(3))
```

**v2 修复**: `_extract_evidence` 的 `query_text` 现在使用 `task.description` 而非 `task_id`，确保证据的 `search_query` 字段记录真实查询词而非内部 ID。

### 7.3 verifier.py — 质量验证 (448 行)

**RuleEngine**: 9 道规则检查，<0.1ms，覆盖 90% 常见失败。

```
检查顺序 (按优先级):
1. Worker 自身失败     → OTHER
2. 输出为空/DuckDuckGo不可用 → EMPTY_RESULT
3. 代码错误 (Traceback)→ CODE_ERROR
4. 全部工具失败        → TOOL_ERROR
5. 工具错误信号词      → TOOL_ERROR
6. 无证据              → INCOMPLETE
7. 输出过短            → 低分通过
8. 全部通过            → 按证据数计分
```

**Verifier**: 三种模式
- `rule` — 纯规则，零延迟 (默认)
- `llm` — LLM 深度诊断 (偏题/幻觉/事实错误)
- `hybrid` — 规则快速筛查 + LLM 复核边界情况 (score 0.3-0.7)

```python
verifier = Verifier(mode="rule")
vr = await verifier.verify(step_result, task_spec)
# → VerificationResult {pass_, score, error_type, feedback, evidence_gaps, suggested_actions}
```

### 7.4 replanner.py — 局部重规划 (508 行)

**核心创新**: 失败后不重建整个计划，只修复失败的那个任务。

**ErrorType → PatchType 策略表**:
```
EMPTY_RESULT  → RETRY (改写查询词)
CODE_ERROR    → RETRY (修正代码)
TOOL_ERROR    → RETRY (切换工具)
OFF_TOPIC     → RETRY (重写描述)
INCOMPLETE    → ADD   (补充子任务)
HALLUCINATION → RETRY (严格指令)
FACTUAL_ERROR → RETRY (交叉验证)
OTHER/NONE    → RETRY (通用重试)
```

**防无限循环**:
- 单个任务最多重试 `max_retries_per_task` 次 (默认 3)
- 单次运行最多触发 `max_total_replans` 次重规划 (默认 5)

```python
replanner = Replanner(max_retries_per_task=3, max_total_replans=5)
patch = replanner.replan(verification_result, plan_graph, "task_003")
if patch:
    replanner.apply_patch(plan_graph, patch)
# → PlanGraph 原地修改 (RETRY 重置状态 / ADD 插入新节点 / REMOVE 标记跳过)
```

**LLMReplanner**: LLM 增强的重规划，调用 LLM 优化查询改写和任务描述，LLM 失败时自动回退到规则逻辑。

### 7.5 writer.py — 报告合成 (515 行)

**v2 双模式**:

| 类 | 输出 | 给谁看 | 禁止内容 |
|----|------|--------|---------|
| `UserAnswerWriter` | `final_answer.md` | 最终用户 | task_id, Token, 耗时, tool_calls JSON, StepResult dump |
| `DebugReportRenderer` | `debug_report.md` | 开发者 | — (包含全部调试信息) |

```python
writer = Writer(mode="template")  # 或 mode="llm" 需要 LLMClient

# 生成两份报告
final_path, debug_path = await writer.write_reports(
    query="Transformer注意力机制",
    session_id="session_xxx",
    plan=plan,
    results=results,
    verifications=verifications,
    memory_ctx=memory.get_context(),
    stats={"total_count": 5, "rounds": 3, "total_replans": 0},
)
# → summaries/session_xxx/final_answer.md
# → summaries/session_xxx/debug_report.md
```

**final_answer.md 结构**: 核心结论 → 详细解释 (按证据类型分组) → 关键要点 → 参考证据 (带 provider/query/URL provenance) → 局限说明

**Mock 数据隔离**: mock 占比 >50% 时顶部显示 "当前为 Mock Demo 模式" 警告。

**v2 集成**: `_finalize` 节点自动调用 Writer 生成双输出，Writer 不可用时回退到原始 Markdown 拼接。

---

## 八、编排层 — orchestration/

### 8.1 dag_workflow.py — DAG 调度 (835 行, v2 全集成)

**ResearchOrchestrator**: 基于 LangGraph StateGraph 的完整编排器。**v2 已将 Verifier/Replanner/Memory/Writer 全部接入主循环**。

#### 8.1.1 状态机 (6 节点)

```
START
  │
  ▼
plan_task ──→ mark_ready ──→ [route_mark] ──→ execute_batch ──→ verify_batch
  (规划)       (调度)           │     │          (并发执行)       (并行验证)
                                │     │                              │
                     done/deadlock  execute               [route_verify]
                          │          │                     │    │    │
                          │          │            continue│replan│done/deadlock
                          │          │                │    │       │
                          ▼          │                ▼    ▼       │
                       finalize ◄────┘           mark_ready replan │
                       (Writer v2)                    ▲       │    │
                          │                           └───────┘    │
                          ▼                                       │
                         END ◄────────────────────────────────────┘
```

#### 8.1.2 节点详解

| 节点 | 方法 | 功能 | 关键逻辑 |
|------|------|------|---------|
| `plan_task` | `_plan_task()` | UserTask → PlanGraph + session_id | plan 已存在时仅补 session_id |
| `mark_ready` | `_mark_ready()` | PENDING → READY (依赖满足时) | 遍历所有节点，检查 depends_on |
| `execute_batch` | `_execute_batch()` | 并发执行所有 READY 任务 | asyncio.gather + Semaphore，结果序列化存储 |
| `verify_batch` | `_verify_batch()` | **并行**验证执行结果 | asyncio.gather 并行调用 Verifier，写入 Memory L1，auto_compress |
| `replan` | `_replan()` | 对失败任务生成 PlanPatch | RETRY→重置 PENDING+清验证记录，ADD→插入新节点 |
| `finalize` | `_finalize()` | Writer v2 双输出 | 优先 Writer，不可用时回退 raw Markdown |

#### 8.1.3 路由逻辑

| 路由 | 方法 | 决策树 |
|------|------|--------|
| `route_after_mark_ready` | `_route_after_mark_ready()` | error?→deadlock \| iteration>=max?→deadlock \| ready?→execute \| no_pending?→done \| pending_blocked?→deadlock |
| `route_after_verify` | `_route_after_verify()` | no_fails+nopending?→done \| can_replan?→replan \| has_pending?→continue \| else→deadlock |

**关键设计**: deadlock 路由到 `finalize` 而非 `END`，确保即使任务失败用户也能获得部分报告。

#### 8.1.4 状态字段

| 字段 | 类型 | 来源 |
|------|------|------|
| `user_task` | str | 初始输入 |
| `session_id` | str | `_plan_task` 生成 |
| `plan` | PlanGraph | `_plan_task` 生成，`_replan` 修改 |
| `results` | dict[str, dict] | `_execute_batch` 累积 |
| `verifications` | dict[str, dict] | `_verify_batch` 累积 |
| `iteration` | int | `_execute_batch` 递增 |
| `replan_count` | int | `_replan` 递增 |
| `max_iterations` | int | 初始配置 |
| `final_output` | str | `_finalize` 生成 |
| `error` | str | 死锁检测时设置 |

#### 8.1.5 使用方式

```python
from horizonrl.orchestration.dag_workflow import create_orchestrator

# 最简单用法
orch = create_orchestrator()
state = await orch.run("Transformer 注意力机制的最新进展")

# 自定义注入
from horizonrl.agent.verifier import Verifier
orch = ResearchOrchestrator(
    planner=Planner(),
    tool_manager=mgr,
    verifier=Verifier(mode="hybrid", llm_client=client),
    memory=HierarchicalMemory(config),
)
```

---

## 九、记忆层 — memory/

### 9.1 hierarchical_memory.py — 分层记忆 (~750 行)

三层架构，**v2 已全部接入 dag_workflow.py 主循环**。

| 层 | 类 | 容量 | 行为 | 集成点 |
|----|-----|------|------|--------|
| L1 | `L1RecentWindow` | 8000 tokens | FIFO 队列，80% 满时旧条目驱逐→L2 | `_verify_batch` 每轮写入 |
| L2 | `L2SemanticSummary` | 50 条摘要 | 模板/LLM 压缩，FIFO 淘汰 | L1 溢出自动触发 |
| L3 | `L3EpisodicArchive` | 无上限 | FAISS n-gram 向量检索 + 混合检索 | `archive_to_l3()` 手动调用 |

```python
mem = HierarchicalMemory(MemoryConfig())
mem.record(step_result, verification_result)  # 写入 L1
mem.auto_compress()                            # L1 超阈值 → L2
mem.archive_to_l3("重要发现: ...")            # 手动归档 L3
ctx = mem.get_context()                        # 获取 MemoryContext
prompt_fragment = ctx.to_prompt_fragment()     # 注入 LLM prompt
```

### 9.2 L3EpisodicArchive — 经验归档 (v2 核心升级)

**v2 n-gram 特征哈希**: 替代旧的 SHA256 假向量，实现真正的语义近似检索。

```
嵌入方法 (优先级降序):
1. LLMClient.embed()     — OpenAI Embedding API (需 API)
2. _ngram_embed()         — MD5 确定性 n-gram 特征哈希 (零依赖，默认)
```

**n-gram 特征哈希算法**:
1. 提取文本 2-gram、3-gram、4-gram
2. 每个 n-gram 用 MD5 哈希到固定维度 (默认 1536)
3. 形成稀疏频率向量 → L2 归一化
4. MD5 保证跨进程确定性（同一文本永远产生相同向量）

**混合检索** (`_hybrid_search`):
1. FAISS n-gram 向量检索召回 top-k×2 候选 (高召回)
2. 关键词重叠后过滤 (保证精度)
3. L2 距离阈值 1.35 预过滤无关结果
4. FAISS 不可用时回退纯关键词匹配

**持久化**: `save()` → `{index_path}.faiss` + `{index_path}.json`，`load()` 恢复。

```python
l3 = L3EpisodicArchive(embedding_dim=256)
l3.archive("Transformer注意力机制详解", metadata={"source": "web"})
results = l3.search("注意力机制")  # → ["[L3] Transformer注意力机制详解"]
l3.save()
# ...进程重启...
l3.load()
results = l3.search("注意力机制")  # → 相同结果 (MD5 确定性)
```

---

## 十、日志层 — logging/

### 10.1 trajectory_logger.py — 轨迹日志 (411 行)

**TrajectoryLogger**: 异步非阻塞 JSONL 写入。asyncio.Queue 缓冲，后台 writer task 写入磁盘。

```python
logger = TrajectoryLogger(output_dir="trajectories")
await logger.start_session("研究问题")
await logger.log(TrajectoryEvent(
    module="planner", event_type=EventType.PLAN_COMPLETE,
    payload={"num_subtasks": 5}, cost=1200, latency=3.5,
))
session = await logger.end_session(success=True)
# → trajectories/session_xxx.jsonl
```

**30 种事件类型**: plan.*, worker.*, tool.*, verify.*, replan.*, memory.*, writer.*, session.*

**分析工具**:
```python
read_session("trajectories/session_xxx.jsonl")   # 读取会话
list_sessions("trajectories")                      # 列出所有会话
aggregate_stats("trajectories")                    # 聚合统计
event_type_distribution(filepath)                  # 事件类型分布
filter_events(filepath, module="worker")            # 按模块过滤
```

---

## 十一、可运行 Demo

| # | 文件 | 功能 | API | 命令 |
|---|------|------|-----|------|
| 01 | `01_async_demo.py` | asyncio 教程 (10 示例) | 否 | `python examples/01_async_demo.py` |
| 02 | `02_simple_agent.py` | 最简端到端管道 | 否 | `python examples/02_simple_agent.py` |
| 03 | `03_llm_demo.py` | LLM 连接测试 + 智能规划 | 是 | `python examples/03_llm_demo.py --llm` |
| 04 | `04_multi_agent_research.py` | v1 旗舰 (6-Stage Pipeline, 双报告) | 可选 | `python examples/04_multi_agent_research.py --llm` |
| 05 | `05_web_agent.py` | Web 对话界面 (双路由 + SSE + 下载) | 自动 | `python examples/05_web_agent.py` |
| 06 | `06_ablation_study.py` | 消融实验 (5 配置 + 压力注入) | 否 | `python examples/06_ablation_study.py` |
| 07 | `07_benchmark.py` | Benchmark 评测 (20 题 5 类) | 否 | `python examples/07_benchmark.py` |

---

## 十二、用户面向功能

### 12.1 命令行研究助手

```bash
# 离线模式 (无需配置, 秒出结果)
python examples/04_multi_agent_research.py "Transformer注意力机制"

# LLM 模式 (需要 API Key, 报告质量更高)
python examples/04_multi_agent_research.py --llm "最新 LLaMA 架构进展"
```

**输出**: `summaries/{session_id}/final_answer.md` + `debug_report.md` + `trajectories/{session_id}.jsonl`

### 12.2 Web 对话界面 (v2: 双路由)

```bash
python examples/05_web_agent.py
# 浏览器打开 http://localhost:8080
```

**三模式切换**: `auto` (自动判断) / `chat` (直接 LLM) / `deep` (强制 Agent 管道)

**API 端点**: `POST /api/chat` | `GET /api/report/{sid}` | `GET /api/download/{sid}/{kind}`

### 12.3 研究报告合成

自动将搜索证据合成为结构化报告: 核心结论 → 详细解释 (按证据类型分组) → 关键要点 → 参考证据 (带来源追溯) → 局限说明

### 12.4 轨迹分析

```python
from horizonrl.logging.trajectory_logger import read_session, aggregate_stats
session = read_session("trajectories/session_xxx.jsonl")
print(session.to_summary())
stats = aggregate_stats("trajectories")
```

---

## 十三、测试体系

### 13.1 测试覆盖

```
tests/
├── test_imports.py              26 模块导入 + 核心依赖检查 (50 tests)
├── test_dag_workflow.py         StateGraph 结构/节点/路由/端到端/死锁 (28 tests)
├── test_planner.py              模板分解/DAG 结构 (9 tests)
├── test_worker.py               执行/证据提取/并发 (8 tests)
├── test_verifier.py             9 规则/Hybrid/错误映射 (24 tests)
├── test_replanner.py            策略映射/补丁/重试限制 (51 tests)
├── test_memory.py               L1/L2/L3/压缩/检索/持久化/上下文 (63 tests)
├── test_tools_manager.py        熔断/超时/重试/统计 (19 tests)
├── test_trajectory_logger.py    写入/会话/分析/过滤 (41 tests)
└── test_writer.py               双模式/证据收集/元数据/provenance (31 tests)
                                ─────────
                                323 passed, 4 skipped, 0 failed
```

### 13.2 运行测试

```bash
python -m pytest tests/ -v          # 全部 (323 tests)
python -m pytest tests/test_dag_workflow.py -v  # 编排层 (28 tests)
python -m pytest tests/test_memory.py -v        # 记忆层 (63 tests)
```

---

## 十四、配置指南

### 14.1 最小配置 (离线可用)

无需任何配置。系统自动使用模板规划 + 模拟工具。零 API 依赖。

### 14.2 API 配置 (LLM 模式)

```bash
cp .env.example .env
# 编辑 .env:
OPENAI_API_KEY=sk-your-deepseek-key
```

编辑 `configs/dev.yaml`:
```yaml
llm:
  provider: openai
  model: deepseek-chat
  base_url: https://api.deepseek.com
```

### 14.3 搜索提供商

```bash
# .env 中设置
BOCHA_API_KEY=           # 国内推荐 (待注册)
BRAVE_API_KEY=            # 国际可选
HORIZON_SEARCH_PROVIDER=auto  # auto = Bocha→Brave→DDGS→Mock 自动回退
```

### 14.4 运行模式

```bash
HORIZON_OFFLINE_MODE=true    # 离线模式 (强制 mock)
ENABLE_LLM_WRITER=false      # 关闭 LLM 写作
HORIZON_SEARCH_PROVIDER=mock # 强制 mock 搜索
```

---

## 十五、完整数据流追踪

以下追踪一个典型研究问题的完整生命周期：

### Step 1: 用户输入

```python
user_task = "Transformer 注意力机制的最新进展"
```

### Step 2: Planner 分解 (`_plan_task`)

```
Planner.classify() → "research" 类型
→ _RESEARCH_TEMPLATE 展开 5 个 TaskSpec
→ _build_plan_graph() 构建 PlanGraph:
    nodes: {task_001..task_005}
    edges: task_003→[task_001, task_002], task_005→[task_003, task_004]
    root_ids: [task_001, task_002, task_004]
→ 生成 session_id
```

### Step 3: 调度 (`_mark_ready`)

```
task_001 (PENDING, deps=[]) → READY
task_002 (PENDING, deps=[]) → READY
task_004 (PENDING, deps=[]) → READY
task_003 (PENDING, deps=[task_001, task_002]) → 保持 PENDING
task_005 (PENDING, deps=[task_003, task_004]) → 保持 PENDING
→ route: execute
```

### Step 4: 并发执行 (`_execute_batch`)

```
asyncio.gather(
    AgentWorker.execute(task_001),  # web_search
    AgentWorker.execute(task_002),  # web_search + arxiv_search
    AgentWorker.execute(task_004),  # web_search
)
→ 每个 Worker: ToolManager.call() → 熔断检查 → 超时控制 → 工具调用
→ 提取 EvidenceItem[] (带 provider/search_query/provenance)
→ 结果序列化为 dict 存入 state.results
→ iteration++
```

### Step 5: 并行验证 (`_verify_batch`)

```
asyncio.gather(
    Verifier.verify(result_001, task_001),
    Verifier.verify(result_002, task_002),
    Verifier.verify(result_004, task_004),
)
→ RuleEngine 9 道检查 (全部通过: score=0.75-0.90)
→ 更新 node.status = SUCCESS
→ 写入 Memory L1 (3 条 MemoryEntry)
→ auto_compress() (L1 使用率 13% < 80%, 不触发)
```

### Step 6: 路由决策 (`_route_after_verify`)

```
无失败 → plan.has_pending_work()? task_003 (PENDING), task_005 (PENDING) → True
→ route: continue
```

### Step 7: 下一轮

```
_mark_ready: task_003 deps(task_001, task_002) 都是 SUCCESS → READY
_execute_batch: task_003 执行 → SUCCESS
_verify_batch: task_003 验证 → PASS, 写入 L1
_route_after_verify: task_005 仍在 PENDING (dep task_004 已 SUCCESS, dep task_003 SUCCESS)
→ route: continue

_mark_ready: task_005 deps(task_003, task_004) 都是 SUCCESS → READY
_execute_batch: task_005 执行 → SUCCESS
_verify_batch: task_005 验证 → PASS, 写入 L1
_route_after_verify: 全部通过, 无 pending → route: done
```

### Step 8: 报告生成 (`_finalize`)

```
Writer.write_reports():
    UserAnswerWriter → final_answer.md:
        # Transformer 注意力机制的最新进展
        ## 核心结论
        基于 5 条检索结果...
        ## 详细解释
        ## 关键要点
        ## 参考证据 (带 provider/query/URL)
    DebugReportRenderer → debug_report.md:
        任务 DAG + 验证详情 + 证据列表 + 工具调用 + Memory 统计
```

### 失败重试路径 (假设 task_002 失败)

```
verify_batch: task_002 验证 → FAIL (TOOL_ERROR, score=0.1)
_route_after_verify: task_002 failed, should_replan? yes → route: replan
_replan: RETRY patch → task_002 重置 PENDING, pop verification
self.memory.record_replan()
→ mark_ready → execute → verify → (重试后可能通过或再次失败)
→ 最多重试 3 次 (max_retries_per_task)
→ 耗尽后: route → continue (其他任务继续) 或 deadlock (无其他工作)
→ deadlock → finalize (用户拿到部分报告 + 错误信息)
```

---

## 十六、已知限制与后续改进空间

### 16.1 已知限制

| 问题 | 影响 | 改进方向 |
|------|------|---------|
| DDGS 国内偶尔超时 | Web 搜索慢 | 接入 Bocha API (国内最优) |
| n-gram 嵌入精度有限 | L3 检索有少量假阳性 | 升级为真实 Embedding API (OpenAI text-embedding-3-small) |
| 流式仅阶段级 | Web 界面需等待整阶段完成 | Phase 3: token 级流式输出 |
| Code Execution 易失败 | 代码任务需多次重试 | prompt 优化 + Docker 沙箱 |
| LLMClient.embed() 用 chat model | 嵌入 API 路径实际不工作 | 需要独立的 embedding model 配置 |
| Replanner 状态跨 run() 不重置 | 多次调用 run() 共享 retry_counts | 新增 `reset()` 调用或每次创建新实例 |
| Memory 跨 run() 累积 | L1/L2 不自动清理 | 按需提供 session 级隔离 |
| 无 GPU | RL 训练/本地模型不可用 | 等服务器 |
| Writer mode 不可通过 Orchestrator 参数设置 | 需后置赋值 `orch.writer = Writer(...)` | 新增 writer 参数 |

### 16.2 Phase 3 计划 (需 GPU)

```
□ vLLM 批量推理部署
□ GRPO/PPO RL 训练管线 (TRL + veRL)
□ token 级流式输出
□ 真实 Embedding API 集成
```

### 16.3 Phase 4 计划 (论文准备)

```
□ 论文大纲 + 写作 (AAAI/IJCAI/ACL Findings 2027)
□ 对比基线: AutoGPT / LangGraph 原生 / 无 Verifier / 无 Memory 消融
□ 更多 Benchmark: GAIA, WebArena
□ 开源发布优化: 文档/CI/CD/容器化
```

### 16.4 短期优化清单 (无需 GPU)

```
□ Bocha API 搜索接入 (国内 Web 搜索最优方案)
□ Replanner 和 Memory 跨 run() 状态重置
□ Orchestrator 支持 writer 参数注入
□ _finalize 的 except Exception 加上日志记录
□ worker.py _build_params 支持 retrieval 工具
□ Planner._classify_task 支持更多任务类型
□ L3 _ngram_embed 的 hashlib/math import 提升到文件顶部
```

---

*本手册 v3.0，随 HorizonRL-Agent v0.2.0 发布。*
*核心原则: 先能跑，再好看 → 先可测，再扩展 → 先日志化，再 RL → 先 GitHub，再论文*
