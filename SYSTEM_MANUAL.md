# HorizonRL-Agent 系统开发手册 v8.0

> **更新**: 2026-05-15 (Day 6 最终) | **测试**: 330 passed | **Git**: 17 commits

---

## 目录

1. 系统概述
2. 架构全景
3. 数据层 — schemas/
4. 配置层 — config/
5. 工具层 — tools/
6. LLM 调用层 — llm/
7. Agent 业务层 — agent/
8. 编排层 — orchestration/
9. 共享服务层 — services/
10. 记忆层 — memory/
11. 日志层 — logging/
12. Demo 清单
13. Web 界面
14. Benchmark 评测
15. 测试体系
16. 配置指南
17. 已知限制

---

## 一、系统概述

### 1.1 一句话定位

给 LLM 配一支"研究团队"——输入问题，系统自动**分解任务 → 并行搜索 → 交叉验证 → 失败重试 → 合成自然语言报告**。

### 1.2 核心功能

| 创新 | 位置 | 说明 |
|------|------|------|
| Verifier 驱动重规划 | `agent/verifier.py` + `replanner.py` | 9 规则 → 9 ErrorType → 4 PatchType |
| 分层记忆 L1→L2→L3 | `memory/hierarchical_memory.py` | L3: DashScope Embedding / n-gram 回退 |
| 异步多 Agent DAG | `orchestration/dag_workflow.py` | LangGraph 6 节点, 全模块集成 |
| Token 流式输出 | `agent/writer.py` → SSE | LLM 写作逐字推送 Web 前端 |
| 轨迹日志 | `logging/trajectory_logger.py` | 30 事件类型, 25+ 事件/次 |
| 证据可追溯 | `schemas/result.py` | SearchProvenance 全链路追踪 |
| 5 种任务分类 | `agent/planner.py` | research/code/comparison/summary/factual_qa |

### 1.3 核心数据流

```
UserTask → Planner (5种) → PlanGraph (DAG)
    → ResearchOrchestrator (LangGraph 6节点)
        → AgentWorker×N (asyncio 并发)
            → ToolManager (熔断/超时/重试)
    → Verifier (9规则) → Replanner → 回写 DAG
    → HierarchicalMemory (L1→L2→L3, Embedding API)
    → Writer v2 (LLM + Token流式, 证据清洗)
    → final_answer.md + debug_report.md
```

---

## 二、架构全景

```
src/horizonrl/
├── schemas/           数据协议 (4 文件, 16 数据结构)
├── config/            配置 (Pydantic V2, 三级合并)
├── services/          共享服务层 (CLI/Web/Benchmark 统一)
├── tools/             工具层 (Web/Arxiv/Code/Mock)
├── llm/               LLM 客户端 (chat/stream/embed)
├── agent/             Agent 逻辑 (Planner/Worker/Verifier/Replanner/Writer)
├── orchestration/     编排层 (LangGraph 6 节点 DAG)
├── memory/            分层记忆 (L1/L2/L3 + FAISS)
└── logging/           轨迹日志 (异步 JSONL)
```

---

## 三、数据层 — schemas/

16 个数据结构，4 个文件。所有模块通过 Schema 通信。

| 文件 | 核心类 |
|------|--------|
| `task.py` | UserTask · TaskSpec · PlanNode · PlanGraph · PlanPatch · PatchType · TaskStatus · TaskPriority |
| `result.py` | StepResult · VerificationResult · EvidenceItem · ToolCall · ErrorType (9种) · SearchProvenance |
| `event.py` | TrajectoryEvent · TrajectorySession · EventType (30种) |
| `report.py` | FinalReport · ReportSection · CitationMap · ReportMetadata |

---

## 四、配置层 — config/

Pydantic V2 三级合并: 代码默认值 → YAML 文件 → `.env` 环境变量。

```python
from horizonrl.config.settings import load_config
cfg = load_config("configs/dev.yaml")
# 环境变量 HORIZON_LLM__MODEL=gpt-4o 可覆盖任何字段
```

| 配置类 | 控制 |
|--------|------|
| `LLMConfig` | provider/model/api_key/base_url/temperature/max_tokens |
| `MemoryConfig` | l1_max_tokens (10000) / l2_max_entries (50) / l3_index_path |
| `AgentRuntimeConfig` | max_steps/semaphore_limit/task_timeout/max_retries |
| `ToolsConfig` | web_search/arxiv_search/code_execution 子配置 |

---

## 五、工具层 — tools/

### 5.1 ToolManager
统一入口: 超时控制 → 重试策略 → 熔断保护 → 统计追踪 → 结果规范化。
三态熔断器: CLOSED → (连续失败) → OPEN → (冷却) → HALF_OPEN。

### 5.2 Web 搜索
5 后端 AUTO 并发竞速 (FIRST_COMPLETED): Bocha → Brave → DDGS → Wikipedia → Mock。
显式 provider 模式保持顺序回退。

### 5.3 Arxiv 搜索
双 API 端点 + arxiv Python 包并发竞速, 首个有效结果即返回。
全失败时自动生成 Mock 论文 (is_mock=True, 管道不中断)。

### 5.4 代码执行
AST 4 层启发式检测自然语言输入 → 5 套代码模板自动生成 → subprocess 安全沙箱 (15s 超时, safe_globals 含 import)。

---

## 六、LLM 调用层 — llm/

```python
client = LLMClient(cfg.llm)
result = await client.chat(prompt, system_prompt="...")      # 普通调用
async for token in client.chat_stream(prompt): ...            # 流式调用
emb = await client.embed(text)                                # Embedding
```

API Key 自动注入: 按 provider 匹配环境变量 (DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY)。

---

## 七、Agent 业务层 — agent/

### 7.1 Planner — 5 种任务类型

| 类型 | 触发关键词 | 模板步骤 |
|------|-----------|---------|
| research | (默认) | 检索背景 → 最新进展 → 分析对比 → 局限性 → 综合汇总 |
| code | 代码/修复/bug | 理解代码 → 定位问题 → 运行 → 修复 → 验证 |
| comparison | 对比/比较/vs | 对象A → 对象B → 维度 → 逐维对比 → 结论 |
| summary | 汇总/综述/概述 | 多源收集 → 去重排序 → 结构化汇总 |
| factual_qa | 是什么/定义 | 权威检索 → 交叉验证 → 生成答案 |

LLMPlanner: LLM 智能拆解 + 时间感知 (当前日期注入 system prompt) + 最大化并行度。

### 7.2 Worker
多工具 asyncio.gather 并发执行。`_clean_search_query()` 从冗长描述中提取干净搜索词。

### 7.3 Verifier
三模式: rule (0.1ms, 9 规则) / llm (深度语义) / hybrid (规则筛查 + LLM 复核边界)。

### 7.4 Replanner
9 种 ErrorType → 4 种 PatchType (RETRY/ADD/REMOVE/REORDER)。
防无限循环: 单任务最多 3 次, 全局最多 5 次。`reset()` 确保会话隔离。

### 7.5 Writer
双输出: UserAnswerWriter (LLM 流式 + 时间感知, 无调试泄露) + DebugReportRenderer (完整 DAG + 证据 + 工具追踪)。
`_collect_evidence()`: 清洗换行符 → 句子边界智能截断 → 去重。
`_build_llm_prompt()`: `_llm_write` + `_write_stream` 共用, 消除重复。

---

## 八、编排层 — orchestration/

### ResearchOrchestrator
LangGraph StateGraph 6 节点:
```
START → plan_task → mark_ready → execute_batch → verify_batch
                    ↑              ↓               ↓
                    └── replan ←── route ←────────┘
                                       ↓
                                   finalize → END
```

注入机制: `session_id` · `writer` · `trajectory_logger` · `embedding_client` · `on_token`。

关键修复:
- `_mark_ready`: 依赖检查含 CANCELLED 终态
- `_execute_batch`: asyncio.wait 批次超时 120s
- `_verify_batch`: L3 归档 + 持久化, L1 自动压缩
- `_plan_task`: L3 检索历史经验注入任务

---

## 九、共享服务层 — services/

CLI / Web / Benchmark 统一入口。

```python
# 同步执行 (CLI/Benchmark)
artifacts = await run_research_session(query="...", mode="deep", llm_client=client)

# 流式执行 (Web SSE)
async for event in stream_research_session(query="...", on_token=callback):
    # event: stage | tool | verify | token | report_ready | done | sse_error
```

`SessionArtifacts`: session_id · final_answer · report_paths · trajectory_path · task_details · stats · mock_ratio。

---

## 十、记忆层 — memory/

### L1: 近期工作窗口 (FIFO, 10000 tokens)
### L2: 语义摘要 (50 条, 模板/LLM 压缩)
### L3: 情景档案 (FAISS, DashScope Embedding / n-gram 回退)

L3 嵌入策略:
- 有 `DASHSCOPE_API_KEY` → text-embedding-v4 (1024 维稠密向量)
- 无 Key → MD5 n-gram 特征哈希 (确定性, 零依赖)
- `_embed_sync`: ThreadPoolExecutor 在事件循环中跑异步 API
- `save()` / `load()`: FAISS 索引 + JSON 元数据持久化

主流程集成: `_plan_task` 检索 L3 → `_verify_batch` 归档 L3 → `save()` 持久化。

---

## 十一、日志层 — logging/

TrajectoryLogger: 异步 JSONL (asyncio.Queue + 后台 Writer)。

事件类型: session.start/end · plan.start/complete · worker.start/complete · tool.result · verify.complete/fail · replan.patch。

Per-node 注入: orchestrator 每个节点通过 `self._log()` 写事件, `log_nowait()` 非阻塞。

分析工具: `read_session()` · `list_sessions()` · `aggregate_stats()` · `event_type_distribution()` · `filter_events()`。

---

## 十二、Demo 清单

| # | 文件 | 描述 | API |
|---|------|------|-----|
| 01 | `01_async_demo.py` | asyncio 教程 (10 示例) | 否 |
| 02 | `02_simple_agent.py` | 最小端到端管道 | 否 |
| 03 | `03_llm_demo.py` | LLM 连接测试 + 智能规划 | 是 |
| 04 | `04_multi_agent_research.py` | 旗舰全链路, 双报告 | 可选 |
| 05 | `05_web_agent.py` | Web (SSE+Token+下载) | 自动 |
| 06 | `06_ablation_study.py` | 消融实验 (5 配置) | 否 |
| 07 | `07_benchmark.py` | Benchmark (20 题) | 可选 |

---

## 十三、Web 界面

```bash
python examples/05_web_agent.py    # http://localhost:8080
```

路由:
- `POST /api/chat` — 对话入口 (auto/chat/deep)
- `GET /api/stream/{sid}` — SSE 实时进度 (stage/tool/verify/token/report_ready/done)
- `GET /api/report/{sid}` — 报告状态查询 (页面刷新恢复)
- `GET /api/download/{sid}/{kind}` — 下载 final/debug markdown

前端: 双栏布局 (侧栏时间线 + 主聊天区), 工具调用日志面板, Markdown 实时渲染 (表格/链接/代码块/列表), 下载按钮 (纯中文, 卡片式)。

---

## 十四、Benchmark 评测

20 题 5 类别, JSONL 任务文件, 全链路 Evaluator。

```bash
python examples/07_benchmark.py                   # Mock 模式
python examples/07_benchmark.py --llm             # LLM 模式
python examples/07_benchmark.py --category 技术对比  # 单类别
```

指标: 通过率 · 子任务成功率 · Mock 占比 · 工具调用 · 重规划 · 耗时 · 失败记录。
输出: `benchmark_results/run_YYYYmmdd_HHMMSS/` (summary.json + per_task_results.jsonl + report.md)。

LLM Benchmark 结果: 20/20 100% 通过, 0 重规划, Mock ~80% (web 部分真实, arxiv 全部 mock)。

---

## 十五、测试体系

```
tests/ (11 文件)
├── test_imports.py          50 tests
├── test_dag_workflow.py     28 tests
├── test_planner.py          14 tests (5 种任务类型)
├── test_worker.py            8 tests
├── test_verifier.py         32 tests
├── test_replanner.py        51 tests
├── test_memory.py           63 tests
├── test_tools_manager.py    19 tests
├── test_trajectory_logger.py 41 tests
├── test_writer.py           31 tests
└── (manual/)                 3 tests
                             ───
                    330 passed, 4 skipped
```

---

## 十六、配置指南

```bash
# .env
DEEPSEEK_API_KEY=sk-xxx        # LLM 推理
DASHSCOPE_API_KEY=sk-xxx       # L3 Embedding (可选, 国内推荐)
BOCHA_API_KEY=sk-xxx           # Web 搜索 (可选, 国内推荐)

# 环境变量覆盖 (双下划线 = 嵌套)
HORIZON_SEARCH_PROVIDER=auto   # auto | bocha | brave | mock
HORIZON_OFFLINE_MODE=true      # 强制离线/Mock
```

三级合并: `configs/default.yaml` (生产) → `configs/dev.yaml` (开发) → `.env` (最高优先级)。

---

## 十七、已知限制

| 问题 | 影响 | 计划 |
|------|------|------|
| Paper Search 速率限制 | 免费 API 有频率限制, 触发后 Mock 兜底 | 已加全局速率限制 + 退避重试 |
| 流式仅 Writer 阶段 | 工具调用无 token 流 | Phase 3 |
| 会话仅内存存储 | 服务重启丢失历史 | Phase 1: SQLite 持久化 |
| Benchmark 仅 20 题 | 统计意义有限 | 扩充 + GAIA/BrowseComp |

---

*本手册 v8.0 — HorizonRL-Agent v0.3.0 — 2026-05-15*
