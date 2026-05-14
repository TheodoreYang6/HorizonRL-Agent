# HorizonRL-Agent

<div align="center">

**Long-Horizon Agentic RL System — 让 LLM Agent 稳定完成 20+ 步复杂任务**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-323%20passed-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/Phase-P1%2FP2%20Complete-orange.svg)]()
[![Lines](https://img.shields.io/badge/Code-~16,000%20lines-blue.svg)]()
[![Docs](https://img.shields.io/badge/Manual-v3.0-ff69b4.svg)](SYSTEM_MANUAL.md)

</div>

---

## 一句话介绍

输入一个研究问题，AI Agent 自动**分解任务 → 并行搜索 → 质量验证 → 失败修复 → 生成报告**。全程异步并发，离线可用，接入 LLM 后效果更佳。

## 为什么需要这个项目？

现有 LLM Agent 在长链路任务中面临三大挑战：

| 挑战 | 问题表现 | 我们的方案 |
|------|---------|-----------|
| **上下文污染** | 第 10 步的噪音干扰第 20 步的决策 | L1→L2→L3 分层记忆，自动压缩驱逐 |
| **错误累积** | 第 5 步的小错误导致后面全盘崩溃 | Verifier 9 规则实时诊断 + Replanner 局部修复 |
| **调度混乱** | 串行太慢，并行又死锁 | LangGraph DAG + asyncio Semaphore 并发 + 死锁检测 |

## 核心创新（全部实现）

### 1. Verifier 驱动的局部重规划

不只判断 pass/fail。Verifier 输出**错误类型 + 证据缺口 + 恢复建议**；Replanner 只对失败节点做局部 patch，**不重建整个计划**。

```
9 种错误类型 × 4 种补丁策略 = 36 条恢复路径
EMPTY_RESULT → RETRY 改写查询 | INCOMPLETE → ADD 补充任务 | HALLUCINATION → RETRY 严格指令
```

### 2. 分层记忆 L1→L2→L3

```
L1 最近工作窗口 (FIFO, 8000 tokens)
  → 80% 满 → 自动压缩 → 
L2 语义摘要 (50 条, 模板/LLM)
  → 关键经验手动归档 →
L3 FAISS n-gram 向量检索 (MD5 确定性, 跨进程持久化)
```

### 3. 异步多 Agent DAG 编排

LangGraph 6 节点状态机：`plan → schedule → execute → verify → replan → finalize`。所有执行和验证都是 `asyncio.gather` 并行，Semaphore 控制并发上限。

### 4. 轨迹日志一等公民

30 种事件类型，异步 JSONL 写入，不阻塞主流程。为**消融实验和 RL 训练**提供数据基础。

## 架构（v2 最终状态）

```
UserTask (自然语言)
    │
    ▼
Planner / LLMPlanner ──→ PlanGraph (TaskSpec[] + DAG)
    │
    ▼
ResearchOrchestrator (LangGraph 6 节点状态机)
    │
    │  plan_task → mark_ready → execute_batch → verify_batch
    │    (规划)      (调度)       (并发执行)      (并行验证)
    │                                                  │
    │                              ┌───────────────────┼──────────┐
    │                         continue               replan  done/deadlock
    │                              │                    │          │
    │                         mark_ready            replan     finalize
    │                         (下一轮)            (局部修复)   (Writer v2)
    │                              ▲                    │          │
    │                              └────────────────────┘          │
    │                                                             ▼
    │                                                            END
    │
    ├──→ Verifier (rule/hybrid/llm) → VerificationResult
    │       └──→ Replanner → PlanPatch (RETRY/ADD/REMOVE/REORDER)
    │
    ├──→ HierarchicalMemory
    │       L1 (FIFO 窗口) → L2 (语义摘要) → L3 (FAISS n-gram)
    │
    ├──→ TrajectoryLogger (异步 JSONL, 30 种事件, 全程)
    │
    └──→ Writer v2
            ├── UserAnswerWriter → final_answer.md (用户)
            └── DebugReportRenderer → debug_report.md (开发者)
```

## 快速开始

### 环境要求

- Python 3.10+
- Windows / Linux / macOS
- 无需 GPU（RL 训练阶段才需要）

### 安装

```bash
git clone https://github.com/TheodoreYang6/HorizonRL-Agent.git
cd HorizonRL-Agent
pip install -r requirements.txt
```

### 3 秒跑通（离线模式，零 API 依赖）

```bash
python examples/02_simple_agent.py "Transformer 注意力机制"
```

输出：5 个子任务并发执行 → 质量验证 → 最终报告。

### LLM 驱动模式（推荐）

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env: OPENAI_API_KEY=sk-your-key

# 2. 默认使用 DeepSeek (configs/dev.yaml)，也可以改成 OpenAI

# 3. 运行
python examples/04_multi_agent_research.py --llm "最新 LLaMA 架构进展"
```

输出：`summaries/{session_id}/final_answer.md` + `debug_report.md` + 轨迹日志。

### Web 交互界面

```bash
python examples/05_web_agent.py
# 浏览器打开 http://localhost:8080
```

三种模式：Auto（自动判断）| Chat（秒回）| Deep（完整 Agent 管道）。Deep 模式完成后自动下载 Markdown 报告。

## 全部 Demo

| # | 文件 | 功能 | API |
|---|------|------|-----|
| 01 | `01_async_demo.py` | asyncio 完整教程 (10 示例) | 否 |
| 02 | `02_simple_agent.py` | **最简端到端** (规划→执行→验证→报告) | 否 |
| 03 | `03_llm_demo.py` | LLM 连接测试 + 智能任务分解 | 是 |
| 04 | `04_multi_agent_research.py` | **v1 旗舰** (6-Stage Pipeline, 双报告) | 可选 |
| 05 | `05_web_agent.py` | Web 对话界面 (双路由 + SSE + 下载) | 自动 |
| 06 | `06_ablation_study.py` | 消融实验框架 (5 配置 + 压力注入) | 否 |
| 07 | `07_benchmark.py` | Benchmark 评测 (20 题 5 类) | 否 |

## 运行测试

```bash
python -m pytest tests/ -v                    # 全部 323 tests
python -m pytest tests/test_dag_workflow.py -v  # 编排层 28 tests
python -m pytest tests/test_memory.py -v        # 记忆层 63 tests
```

```
tests/
├── test_imports.py              # 26 模块导入 (50 tests)
├── test_dag_workflow.py         # 图结构/节点/路由/端到端/死锁 (28 tests)
├── test_memory.py               # L1/L2/L3/压缩/检索/持久化 (63 tests)
├── test_replanner.py            # 策略映射/补丁/重试限制 (51 tests)
├── test_writer.py               # 双模式/证据/元数据/provenance (31 tests)
├── test_verifier.py             # 9 规则/Hybrid/错误映射 (24 tests)
├── test_trajectory_logger.py    # 写入/会话/分析/过滤 (41 tests)
├── test_tools_manager.py        # 熔断/超时/重试/统计 (19 tests)
├── test_worker.py               # 执行/证据提取/并发 (8 tests)
└── test_planner.py              # 模板分解/DAG 结构 (9 tests)
                                ─────────
                        323 passed, 4 skipped, 0 failed
```

## 项目结构

```
src/horizonrl/
├── schemas/           数据协议层 (4 文件, 16 数据结构, 1061 行)
│   ├── task.py        TaskSpec · PlanGraph · PlanNode · PlanPatch · UserTask
│   ├── result.py      StepResult · VerificationResult · EvidenceItem · ToolCall
│   ├── event.py       TrajectoryEvent · TrajectorySession · EventType (30 种)
│   └── report.py      FinalReport · ReportSection · CitationMap · ReportMetadata
│
├── config/            配置管理层 (1 文件, 686 行)
│   └── settings.py    Pydantic V2: LLMConfig · MemoryConfig · AgentRuntimeConfig
│
├── tools/             工具层 (5 文件, 834 行)
│   ├── manager.py     ToolManager: 超时/重试/熔断/统计 · CircuitBreaker
│   ├── web_search.py  Brave → DDGS → Wikipedia → Mock 4 级自动回退
│   ├── arxiv_search.py   Arxiv API
│   ├── code_execution.py  subprocess 沙箱
│   └── mock.py        Mock 工具 (CI/离线可用)
│
├── llm/               LLM 调用层 (1 文件, 185 行)
│   └── client.py      chat() + embed() · OpenAI-compatible · DeepSeek 已打通
│
├── agent/             Agent 业务层 (5 文件, ~1950 行)
│   ├── planner.py     Planner (模板 2 类) + LLMPlanner (LLM DAG 拆解)
│   ├── worker.py      AgentWorker (异步执行 + 证据提取) + execute_workers()
│   ├── verifier.py    RuleEngine (9 规则) + Verifier (rule/llm/hybrid)
│   ├── replanner.py   Replanner (9 策略) + LLMReplanner (LLM 增强)
│   └── writer.py      UserAnswerWriter + DebugReportRenderer 双模式
│
├── orchestration/     编排层 (1 文件, 835 行)
│   └── dag_workflow.py  ResearchOrchestrator: LangGraph 6 节点状态机
│
├── memory/            记忆层 (1 文件, ~750 行)
│   └── hierarchical_memory.py  L1RecentWindow · L2SemanticSummary · L3EpisodicArchive
│
└── logging/           日志层 (1 文件, 411 行)
    └── trajectory_logger.py  异步 JSONL · TrajectorySession · 5 分析工具
```

## 技术栈

| 层面 | 技术 | 说明 |
|------|------|------|
| **编排** | LangGraph StateGraph | 6 节点 DAG + 条件路由 + 循环 |
| **异步** | Python asyncio | gather · Semaphore · Queue · wait_for |
| **LLM** | OpenAI SDK | DeepSeek / OpenAI / vLLM / 任何兼容 API |
| **嵌入** | MD5 n-gram 特征哈希 | L3 向量检索, 零依赖, 确定性 (可升级为 Embedding API) |
| **向量检索** | FAISS IndexFlatL2 | L2 距离 + 阈值过滤 + 关键词后过滤 |
| **配置** | Pydantic V2 | 代码默认 → YAML → .env 三级合并 |
| **日志** | JSONL 异步写入 | asyncio.Queue + 后台 writer task |
| **测试** | pytest + pytest-asyncio | 323 tests, 10 测试文件 |

## 配置

```bash
# API Key
OPENAI_API_KEY=sk-your-key      # .env 文件

# 运行模式
HORIZON_OFFLINE_MODE=true       # 离线 mock 模式
ENABLE_LLM_WRITER=false         # 模板写作 (不调用 LLM)
HORIZON_SEARCH_PROVIDER=mock    # 强制 mock 搜索

# 配置覆盖 (双下划线 = 嵌套)
HORIZON_LLM__MODEL=deepseek-chat
HORIZON_AGENT__MAX_STEPS=20
HORIZON_MEMORY__L1_MAX_TOKENS=6000
```

详细配置见 [SYSTEM_MANUAL.md](SYSTEM_MANUAL.md) 第十四章。

## 已知限制

| 问题 | 影响 | 计划 |
|------|------|------|
| DDGS 国内偶尔超时 | Web 搜索慢 | 接入 Bocha API |
| n-gram 嵌入精度有限 | L3 检索有少量假阳性 | 升级 Embedding API |
| 流式仅阶段级 | Web 等整阶段完成 | Phase 3: token 级流式 |
| 无 GPU | RL 训练暂不可用 | 等服务器后 Phase 3 |
| Code Execution 易失败 | 代码任务需重试 | prompt 优化 + Docker |

详见 [SYSTEM_MANUAL.md](SYSTEM_MANUAL.md) 第十六章。

## 路线图

| 阶段 | 状态 | 内容 |
|------|------|------|
| Phase 1 (Steps 0-14) | ✅ 完成 | 核心基础设施 — schemas · tools · agent · orchestration |
| Phase 2 (Steps 15-18) | ✅ 完成 | 产品化 — 消融 · FAISS · Benchmark · SSE · v2 全集成 |
| Phase 3 | ⬜ 等 GPU | RL 训练 (GRPO/PPO) · vLLM · token 级流式 |
| Phase 4 | ⬜ 论文 | AAAI/IJCAI 投稿 · 对比基线 · GAIA/WebArena |

详见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)。

## 文档

- [SYSTEM_MANUAL.md](SYSTEM_MANUAL.md) — 系统开发手册 v3.0 (910 行, 16 章)
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) — 开发路线图 v7.0
- [CLAUDE.md](CLAUDE.md) — AI 助手指令

## 许可证

MIT License — 详见 [LICENSE](LICENSE)

## 作者

**杨启铎 (Theodore Yang)** — NWPU 硕士

研究方向：LLM Agent 长链路稳定执行 · 分层记忆 · Verifier-guided Replanning · Agentic RL

GitHub: [@TheodoreYang6](https://github.com/TheodoreYang6)

---

*HorizonRL-Agent v0.2.0 — Day 3 Build | Verifier + Replanner + Memory + Writer 全集成 | 323 tests*
