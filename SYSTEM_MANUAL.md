# HorizonRL-Agent 系统开发手册 v1.0

> 最后更新: 2026-05-13
> 测试: 314 passed, 4 skipped
> 代码量: ~14,000 行

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
15. [已知限制与后续计划](#十五已知限制与后续计划)

---

## 一、系统概述

### 1.1 一句话定位

给 LLM 配一支"研究团队"——输入问题，系统自动**分解任务 → 并行搜索 → 交叉验证 → 失败重试 → 合成自然语言报告**。

### 1.2 核心创新

| 创新 | 实现 | 说明 |
|------|------|------|
| 分层记忆 L1→L2→L3 | `memory/hierarchical_memory.py` | L1工作窗口→L2语义摘要→L3预留 |
| Verifier驱动重规划 | `agent/verifier.py` + `replanner.py` | 9规则检查 + 9种恢复策略 |
| 异步多Agent DAG | `orchestration/dag_workflow.py` | LangGraph + asyncio并发 + 死锁检测 |
| 轨迹日志一等公民 | `logging/trajectory_logger.py` | 30种事件JSONL异步写入 |
| 证据可追溯 | `schemas/result.py` | 每条证据带provider/query/URL |

### 1.3 核心数据流

```
你的问题
    │
    ▼
Planner / LLMPlanner         ← 拆成 5-6 个子任务, 画 DAG 依赖图
    │
    ▼
AgentWorker × N (异步并发)     ← 执行子任务, 调用搜索/论文/代码工具
    │
    ▼
StepResult + EvidenceItem[]   ← 每个任务的结果 + 收集到的证据
    │
    ├──→ Verifier              ← 检查结果质量 (9规则 + LLM深度诊断)
    │      │
    │      └──→ Replanner      ← 失败的自动修复, 最多重试3次
    │
    ├──→ HierarchicalMemory    ← 记录每一步, token满自动压缩为摘要
    ├──→ TrajectoryLogger      ← 全程JSONL日志, 不阻塞主流程
    │
    └──→ Writer                ← 证据→自然语言报告
           ├── final_answer.md (用户)
           └── debug_report.md (开发者)
```

---

## 二、架构全景

```
src/horizonrl/
├── schemas/           数据协议层 (4文件, 1061行) — 全项目通信协议
├── config/            配置管理层 (1文件, 686行)  — Pydantic V2三级配置
├── tools/             工具层 (5文件, 834行)      — 搜索/论文/代码 + 熔断重试
├── llm/               LLM调用层 (1文件, 135行)   — OpenAI兼容客户端
├── agent/             Agent业务层 (5文件, 1935行) — 规划/执行/验证/修复/写作
├── orchestration/     编排层 (1文件, 502行)      — LangGraph DAG调度
├── memory/            记忆层 (1文件, 550行)      — L1窗口 + L2摘要
└── logging/           日志层 (1文件, 411行)      — 异步JSONL轨迹日志
```

**依赖方向**: schemas → config → tools → agent → orchestration (上层不依赖下层内部实现)

---

## 三、数据层 — schemas/

全项目 16 个数据结构，定义在 4 个文件中。所有模块通过 Schema 通信。

### 3.1 schemas/task.py — 任务相关 (320行)

| 类 | 用途 |
|----|------|
| `UserTask` | 用户输入的自然语言问题 + 约束 (max_steps, max_tokens) |
| `TaskSpec` | Planner拆解出的单个子任务: id, name, description, tool_names, depends_on, priority, retry_count |
| `TaskStatus` | 生命周期: PENDING → READY → RUNNING → SUCCESS / FAILED / SKIPPED |
| `TaskPriority` | P0(关键路径) / P1(正常) / P2(后置) |
| `PlanNode` | PlanGraph中的一个节点: 包装 TaskSpec + 运行时状态 (status, error_msg, started_at) |
| `PlanGraph` | 完整任务图: nodes[], edges{}, root_ids[], 提供 get_ready_nodes(), has_pending_work() |
| `PlanPatch` | Replanner的局部修改: patch_type, target_node_id, new_spec, reason |
| `PatchType` | ADD / REMOVE / REORDER / RETRY |

**实际使用**:
```python
from horizonrl.schemas.task import UserTask, TaskSpec, PlanGraph
task = UserTask(description="Transformer注意力机制", max_steps=20)
# Planner.plan(task) → PlanGraph (含5个TaskSpec + DAG依赖)
```

### 3.2 schemas/result.py — 执行结果 (256行)

| 类 | 用途 |
|----|------|
| `ToolCall` | 单次工具调用记录: tool_name, input, output, elapsed, error, tokens_used |
| `EvidenceItem` | 单条证据: content, source, source_type, provider, search_query, is_mock, provenance |
| `SearchProvenance` | 搜索来源追溯: provider, query, timestamp, raw_snippet, score, url, is_mock |
| `StepResult` | Worker执行结果: task_id, success, output, evidence[], tool_calls[], elapsed |
| `VerificationResult` | Verifier结论: pass_, score, error_type, feedback, evidence_gaps, suggested_actions |
| `ErrorType` | 9种错误: NONE / EMPTY_RESULT / CODE_ERROR / TOOL_ERROR / OFF_TOPIC / FACTUAL_ERROR / INCOMPLETE / HALLUCINATION / OTHER |

**provenance 追溯**: 每条 EvidenceItem 记录来源平台(provider)、搜索query、抓取时间、是否为 mock 数据。

### 3.3 schemas/event.py — 轨迹事件 (248行)

| 类 | 用途 |
|----|------|
| `EventType` | 30种事件类型枚举: plan.start→worker.complete→tool.call→verify.fail→replan.patch→session.end |
| `TrajectoryEvent` | 单条事件: ts, module, event_type, payload, cost, latency, session_id, step_id |
| `TrajectorySession` | 完整会话: events[], to_summary(), filter_by_module(), filter_by_type() |

### 3.4 schemas/report.py — 报告 (168行)

| 类 | 用途 |
|----|------|
| `CitationMap` | 声明↔证据引用映射 |
| `ReportSection` | 报告章节: title, content, citations |
| `FinalReport` | 完整报告: title, sections[], summary, metadata |
| `ReportMetadata` | 生成元数据: session_id, author, mode, used_mock_data, llm_writer_used |

---

## 四、配置层 — config/

### 4.1 settings.py (686行)

**三级配置合并**: `代码默认值 → YAML文件 → .env环境变量`

```python
from horizonrl.config.settings import load_config
cfg = load_config(Path("configs/dev.yaml"))
# 配置优先级: dev.yaml > default.yaml > 代码默认值
# 环境变量 HORIZON_XXX 可覆盖任何字段
```

| 配置类 | 控制内容 |
|--------|---------|
| `LLMConfig` | provider, model, api_key, base_url, temperature, max_tokens |
| `MemoryConfig` | l1_max_tokens(8000), l2_max_entries(50), auto_compress_threshold(0.8) |
| `AgentRuntimeConfig` | max_steps, worker_semaphore_limit, task_timeout, max_retries |
| `ToolsConfig` | search_provider, allow_mock_fallback, enable_llm_writer |

**配置文件**:
- `configs/default.yaml` — 生产环境 (gpt-4o)
- `configs/dev.yaml` — 开发环境 (DeepSeek/deepseek-chat)
- `configs/eval.yaml` — 评测环境 (temperature=0)

---

## 五、工具层 — tools/

### 5.1 manager.py — 工具管理器 (426行)

**核心**: 统一工具调用入口，所有 Worker 必须通过 ToolManager 调用工具。

```python
mgr = ToolManager()
mgr.register("web_search", WebSearchTool())
result = await mgr.call(ToolCallRequest(tool_name="web_search", params={"query": "..."}))
```

| 组件 | 功能 |
|------|------|
| `CircuitBreaker` | 三态熔断: CLOSED→(5次失败)→OPEN→(60s冷却)→HALF_OPEN→(探测成功)→CLOSED |
| `ToolManager._invoke()` | 自动适配3种接口: async execute() / async search() / __call__() |
| `ToolManager.call()` | 完整流程: 熔断检查 → 超时控制(asyncio.wait_for) → 指数退避重试 → 统计更新 |
| `ToolStats` | 每工具独立统计: total_calls, success_calls, failure_calls, timeout_calls, total_latency |
| `ToolErrorType` | 8种错误分类: TIMEOUT, CIRCUIT_OPEN, AUTH, RATE_LIMIT, NETWORK, UNREGISTERED, INTERNAL, UNKNOWN |

### 5.2 web_search.py — 网页搜索 (可运行 ✅)

**4级后端自动回退**:
```
Brave API (有Key) → DDGS (国内可用) → Wikipedia API → Mock
```

每个后端 8 秒超时，失败自动降级。当前 DDGS 新包在国内可以正常搜索。

```python
tool = WebSearchTool()
results = await tool.search("Python asyncio", num_results=5)
# → [{"title": "...", "url": "https://...", "snippet": "..."}]
```

### 5.3 arxiv_search.py — 学术论文搜索 (可运行 ✅)

直接调用 Arxiv API，返回真实论文标题、摘要、PDF链接。

```python
tool = ArxivSearchTool(max_results=5)
results = await tool.search("transformer attention mechanism")
```

### 5.4 code_execution.py — 代码执行 (可运行 ✅)

subprocess 沙箱执行 Python 代码，10 秒超时。

```python
tool = CodeExecutionTool(timeout=10.0)
result = await tool.execute("print('hello world')")
```

### 5.5 mock.py — 模拟工具 (107行)

离线/CI 环境使用的模拟工具，输出格式与真实工具一致。`register_mock_tools(mgr)` 一键注册全部。

---

## 六、LLM 调用层 — llm/

### 6.1 client.py — LLM客户端 (135行)

OpenAI 兼容的异步客户端。已打通 DeepSeek API。

```python
from horizonrl.llm.client import LLMClient
client = LLMClient(cfg.llm)
result = await client.chat("你好", system_prompt="你是一个助手", max_tokens=500)
# result.content → LLM回复
# result.tokens_total → token消耗
# result.elapsed → 耗时
```

**支持所有 OpenAI 兼容 API**: DeepSeek, OpenAI, vLLM, 任何 `/v1/chat/completions` 端点。

---

## 七、Agent 业务层 — agent/

### 7.1 planner.py — 任务分解 (368行)

**Planner (模板模式)**: 无 LLM 依赖，2 种任务模板：
- `research` 模板: 检索背景 → 最新进展 → 分析对比 → 局限性 → 综合汇总
- `code` 模板: 理解代码 → 定位问题 → 运行 → 修复 → 验证

**LLMPlanner (LLM 模式)**: 调用 LLM 智能拆解任意类型任务，生成更合理的 DAG 依赖。

```python
# 模板模式 (离线可用)
planner = Planner()
plan = planner.plan(UserTask(description="Transformer注意力机制"))

# LLM模式 (需要API)
planner = LLMPlanner(llm_client)
plan = await planner.plan(UserTask(description="任意复杂问题"))
# → PlanGraph (4-7个子任务, DAG依赖, 工具分配)
```

### 7.2 worker.py — 任务执行 (270行)

**AgentWorker**: 执行单个 TaskSpec，调用工具，提取证据。
```python
worker = AgentWorker(worker_id="wrk_1", tool_manager=mgr)
result = await worker.execute(task_spec)
# → StepResult (含output, evidence[], tool_calls[])
```

**execute_workers()**: 批量并发执行，受 Semaphore 控制。
```python
results = await execute_workers(tasks, tool_manager, semaphore=asyncio.Semaphore(3))
```

**证据提取**: 自动解析工具返回的 JSON，标记 `is_mock`，填充 `provider` 和 `search_query`。

### 7.3 verifier.py — 质量验证 (448行)

**RuleEngine**: 9 道规则检查，<0.1ms，覆盖 90% 常见失败。
```
检查顺序 (按优先级):
1. Worker自身失败     → OTHER
2. 输出为空           → EMPTY_RESULT
3. 代码错误(Traceback)→ CODE_ERROR
4. 全部工具失败       → TOOL_ERROR
5. 工具错误信号词     → TOOL_ERROR
6. 无证据             → INCOMPLETE
7. 输出过短           → 低分通过
8. 全部通过           → 按证据数计分
```

**Verifier**: 三种模式
- `rule` — 纯规则，零延迟
- `llm` — LLM 深度诊断 (偏题/幻觉/事实错误)
- `hybrid` — 规则快速筛查 + LLM 复核边界情况 (默认)

```python
verifier = Verifier(mode="rule")
result = await verifier.verify(step_result, task_spec)
# → VerificationResult {pass, score, error_type, feedback, evidence_gaps, suggested_actions}
```

### 7.4 replanner.py — 局部重规划 (508行)

**核心**: 失败后不重建整个计划，只修复失败的那个任务。

**ErrorType → PatchType 策略表**:
```
EMPTY_RESULT  → RETRY (改写查询词)
CODE_ERROR    → RETRY (修正代码)
TOOL_ERROR    → RETRY (切换工具)
OFF_TOPIC     → RETRY (重写描述)
INCOMPLETE    → ADD   (补充子任务)
HALLUCINATION → RETRY (严格指令)
FACTUAL_ERROR → RETRY (交叉验证)
```

```python
replanner = Replanner(max_retries_per_task=3, max_total_replans=5)
patch = replanner.replan(verification_result, plan_graph, "task_003")
# → PlanPatch {patch_type, new_spec, reason}
replanner.apply_patch(plan_graph, patch)
# → PlanGraph原地修改 (RETRY重置状态 / ADD插入新节点 / REMOVE标记跳过 / REORDER提优先级)
```

### 7.5 writer.py — 报告合成 (可运行 ✅, 最新重构)

**v2 双模式**:

| 类 | 输出 | 给谁看 |
|----|------|--------|
| `UserAnswerWriter` | `final_answer.md` | 最终用户 |
| `DebugReportRenderer` | `debug_report.md` | 开发者 |

```python
writer = Writer(mode="llm", llm_client=client,
                config=WriterConfig(export_dir="summaries"))

# 生成两份报告
final_path, debug_path = await writer.write_reports(
    query="Transformer注意力机制",
    session_id="session_xxx",
    plan=plan, results=results, verifications=verifications,
)

# 或只生成用户答案
report = await writer.synthesize_async(query, plan, results)
```

**final_answer.md 严格禁止**: task_id, Token, 耗时, tool_calls JSON, StepResult dump, 原始 mock 数组

**final_answer.md 包含**: 核心结论, 详细解释, 关键要点, 参考证据(带 provider/query/URL provenance)

**Mock 数据隔离**: mock 占比 >50% 时顶部显示 "当前为Mock Demo模式" 警告

---

## 八、编排层 — orchestration/

### 8.1 dag_workflow.py — DAG调度 (502行)

**ResearchOrchestrator**: 基于 LangGraph StateGraph 的完整编排。

```python
orch = create_orchestrator()
state = await orch.run("Transformer注意力机制")
# 或流式
async for node_name, state in orch.stream("问题"):
    print(node_name, state["iteration"])
```

**状态机**:
```
plan_task → mark_ready → [有READY?] → execute_batch → (循环)
                          [全部完成?] → finalize → END
                          [死锁/超限?] → END
```

**死锁检测**: 依赖任务失败且无法重试 → 标记 deadlock，避免无限等待

---

## 九、记忆层 — memory/

### 9.1 hierarchical_memory.py — 分层记忆 (550行)

| 层 | 类 | 容量 | 行为 |
|----|-----|------|------|
| L1 | `L1RecentWindow` | 8000 tokens | FIFO队列，80%满时旧条目自动驱逐 |
| L2 | `L2SemanticSummary` | 50条摘要 | 模板/LLM压缩，FIFO淘汰 |
| L3 | (预留) | FAISS索引 | 向量检索，跨会话经验复用 |

```python
mem = HierarchicalMemory(MemoryConfig())
mem.record(step_result, verification_result)  # 写入L1
mem.compress("任务上下文")                    # L1→L2手动压缩
mem.auto_compress()                            # 超阈值自动压缩
ctx = mem.get_context()                        # 获取MemoryContext
# ctx.to_prompt_fragment() → 可注入LLM prompt的文本
```

---

## 十、日志层 — logging/

### 10.1 trajectory_logger.py — 轨迹日志 (411行, 可运行 ✅)

**TrajectoryLogger**: 异步非阻塞 JSONL 写入。asyncio.Queue 缓冲，后台 writer task 写入磁盘。

```python
logger = TrajectoryLogger(output_dir="trajectories")
await logger.start_session("研究问题")

await logger.log(TrajectoryEvent(
    module="planner", event_type=EventType.PLAN_COMPLETE,
    payload={"num_subtasks": 5}, cost=1200, latency=3.5,
))

session = await logger.end_session(success=True)
# → TrajectorySession (含所有事件 + 统计)
```

**分析工具**:
```python
read_session("trajectories/session_xxx.jsonl")  # 读取会话
list_sessions("trajectories")                    # 列出所有会话
aggregate_stats("trajectories")                  # 聚合统计
event_type_distribution(filepath)                # 事件类型分布
filter_events(filepath, module="worker")          # 按模块过滤
```

---

## 十一、可运行 Demo

| Demo | 文件 | 功能 | 需要API | 运行命令 |
|------|------|------|---------|---------|
| 01 | `01_async_demo.py` | asyncio 教程 (10个示例) | 否 | `python examples/01_async_demo.py` |
| 02 | `02_simple_agent.py` | 最简端到端管道 | 否 | `python examples/02_simple_agent.py` |
| 03 | `03_llm_demo.py` | LLM连接测试 + 智能规划 | 是 | `python examples/03_llm_demo.py --llm` |
| 04 | `04_multi_agent_research.py` | **v1 旗舰** (6-Stage Pipeline) | 可选 | `python examples/04_multi_agent_research.py --llm` |
| 05 | `05_web_agent.py` | Web 对话界面 | 自动检测 | `python examples/05_web_agent.py` |

**04 Demo 输出**:
```
summaries/{session_id}/
├── final_answer.md    ← 用户可读的自然语言答案
└── debug_report.md    ← 开发者视图 (DAG/验证/统计)
trajectories/{session_id}.jsonl  ← 完整轨迹日志
```

---

## 十二、用户面向功能

### 12.1 命令行研究助手

```bash
# 离线模式 (无需任何配置, 秒出结果)
python examples/04_multi_agent_research.py "Transformer注意力机制"

# LLM模式 (需要API Key, 报告质量更高)
python examples/04_multi_agent_research.py --llm "最新LLaMA架构进展"

# 自定义并发度
python examples/04_multi_agent_research.py --workers 5 "你的问题"
```

**输出**: 自然语言研究答案 + 调试报告 + 轨迹日志

### 12.2 Web 对话界面 (v2: 双路由)

```bash
python examples/05_web_agent.py
# 浏览器打开 http://localhost:8080
```

**三模式切换**:
| 模式 | 行为 | 适合 |
|------|------|------|
| `auto` | 自动判断: 简单→对话, 复杂→深度研究 | 默认 |
| `chat` | 直接 LLM 对话, 秒回 | 闲聊、定义、简单问题 |
| `deep` | 强制走 Agent 管道 | 研究、综述、对比分析 |

**API 端点**:
| 端点 | 用途 |
|------|------|
| `POST /api/chat` | 统一入口: `{"message":"...", "mode":"auto|chat|deep"}` |
| `GET /api/report/{sid}` | 轮询任务状态 (queued→running→completed/failed) |
| `GET /api/download/{sid}/{kind}` | 下载 final_answer.md 或 debug_report.md |

**自动下载**: 深度研究完成后自动触发 Markdown 文件下载。

**复杂度分类器**: `should_use_agent()` 按关键词 (综述/对比/最新进展/调研...) + 长度 自动判断是否触发 Agent 管道。

### 12.3 研究报告合成

系统自动将搜索证据合成为结构化报告:
- **核心结论** — 2-3句话总结
- **详细解释** — 按证据类型展开
- **关键要点** — 3-5个要点
- **参考证据** — 每条带来源追溯 (provider/query/URL)
- Mock 数据会明确标记，不假装是真实信息

### 12.4 轨迹分析

```python
from horizonrl.logging.trajectory_logger import read_session, aggregate_stats

# 读取一次会话
session = read_session("trajectories/session_xxx.jsonl")
print(session.to_summary())  # 统计摘要

# 按模块过滤
planner_events = session.filter_by_module("planner")

# 聚合所有会话
stats = aggregate_stats("trajectories")
# → {total_sessions, success_rate, avg_tokens_per_session, ...}
```

---

## 十三、测试体系

### 13.1 测试覆盖

```
tests/
├── test_imports.py             26模块导入 + 核心依赖检查 (84 tests)
├── test_tools_manager.py       熔断/超时/重试/统计 (19 tests)
├── test_planner.py             模板分解/DAG结构 (9 tests)
├── test_worker.py              执行/证据提取/并发 (8 tests)
├── test_verifier.py            9规则/Hybrid/错误映射 (24 tests)
├── test_replanner.py           策略映射/补丁/重试限制 (51 tests)
├── test_memory.py              L1/L2/压缩/检索/上下文 (54 tests)
├── test_trajectory_logger.py   写入/会话/分析/过滤 (41 tests)
├── test_writer.py              双模式/证据收集/元数据/provenance (31 tests)
└── test_dag_workflow.py        图结构/路由/端到端 (28 tests)
                                ─────────
                                314 passed, 4 skipped
```

### 13.2 运行测试

```bash
python -m pytest tests/ -v              # 全部
python -m pytest tests/test_writer.py -v  # 单模块
python -m pytest tests/ -q              # 安静模式
```

---

## 十四、配置指南

### 14.1 最小配置 (离线可用)

无需任何配置。系统自动使用模板规划 + 模拟工具。

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
BOCHA_API_KEY=           # 国内推荐
BRAVE_API_KEY=            # 国际可选
HORIZON_SEARCH_PROVIDER=auto  # auto = Bocha→Brave→DDGS→Mock 自动回退
```

### 14.4 运行模式

```bash
HORIZON_OFFLINE_MODE=true    # 离线模式 (强制mock)
ENABLE_LLM_WRITER=false      # 关闭LLM写作
HORIZON_SEARCH_PROVIDER=mock # 强制mock搜索
```

---

## 十五、已知限制与后续计划

### 15.1 已知限制

| 问题 | 影响 | 计划 |
|------|------|------|
| DDGS 国内偶尔超时 | Web搜索有时慢 | 等待 Bocha Key |
| LLMPlanner 生成 DAG 偏串行 | 简单问题也等 | 已优化 prompt，继续调 |
| 无流式输出 | Web干等30秒 | P1 |
| Code Execution 易失败 | 代码任务常需重试 | P2 |
| L3 FAISS 未实现 | 跨会话记忆不可用 | P2 |
| Bocha provider 未接入 | 国内搜索最优方案缺 | 需API Key |

### 15.2 后续计划

```
✅ 已完成 (本轮):
  ✅ Writer v2 双模式 (UserAnswerWriter + DebugReportRenderer)
  ✅ SearchProvenance + ReportMetadata
  ✅ Web 双路由 (/api/chat + /api/report + /api/download)
  ✅ CI 强制 mock 模式
  ✅ 代码质量优化 (ruff + imports + prompt)
  ✅ LLMPlanner 并行度优化
  ✅ 314 tests

P1 (本月):
  □ 流式输出 (SSE/WebSocket)
  □ Bocha API 接入 (需 Key)
  □ 消融实验框架

P2 (下月):
  □ L3 FAISS 向量检索
  □ Benchmark 评测
  □ 论文实验
```

---

*本手册随 HorizonRL-Agent v0.1.0 发布。项目 GitHub: https://github.com/YOUR_USERNAME/HorizonRL-Agent*
