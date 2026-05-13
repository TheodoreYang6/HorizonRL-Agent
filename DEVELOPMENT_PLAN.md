# HorizonRL-Agent 最终开发路线图 v4.0

> **更新日期**: 2026-05-13 (Day 2 收尾)
> **状态**: Phase 1 全部完成 (Steps 0-13/14)，MVP 可运行
> **测试**: 284 passed, 4 skipped, 0 failed
> **代码量**: ~11,300 行 (src 5858 + tests 3067 + examples 2369)

---

# 一、项目定位与研究问题

## 一句话定位

面向长链路研究与工具使用任务，构建一个以**分层记忆、验证器诊断和局部重规划**为核心的多智能体 LLM Agent 稳定执行系统。

## 核心研究问题

**LLM Agent 在 20+ 步复杂任务中的稳定执行问题**。具体表现为：
- 上下文污染（Context Collapse）
- 任务漂移（Planning Drift）
- 无效操作（Tool Misuse）
- 幻觉累积（Hallucination Accumulation）
- 失败后不会恢复（No Recovery）
- 长任务中工具调用效率低下

## 核心创新点

1. **分层记忆结构**：L1 最近工作记忆 → L2 语义摘要 → L3 经验归档
2. **Verifier 驱动的重规划**：Verifier 不只输出 pass/fail，还给出错误类型、证据缺口和恢复建议；Replanner 只做局部 patch 而非全局重置
3. **轨迹级日志与评测**：轨迹日志作为一等基础设施，追踪计划漂移、工具误用、重试和幻觉累积
4. **上下文压缩与证据可追溯写作**：Memory 折叠为带 provenance 的摘要，Writer 可回溯证据来源
5. **后置 RL 增强闭环**：系统、日志和评测稳定后才引入 GRPO/PPO

---

# 二、实际开发进度

## 2.1 完成状态总览

```
Phase 1: 核心基础设施 (W1-2 目标，2 天完成)

Step 0  ✅  项目蓝图 + .claude/ 规则体系               (Day 1 上午)
Step 1  ✅  .gitignore + CLAUDE.local.md                (Day 1 上午)
Step 2  ✅  schemas/ (4 文件, 16 数据结构)              (Day 1 下午)
Step 3  ✅  configs/ (Pydantic V2 三级配置)             (Day 1 下午)
Step 4  ✅  examples/01_async_demo.py (10 示例)         (Day 1 下午)
Step 5  ✅  tools/manager.py + 3 工具                   (Day 2 上午)
Step 6  ✅  agent/planner.py + worker.py                (Day 2 上午)
Step 7  ✅  examples/02_simple_agent.py (端到端)        (Day 2 上午)
Step 8  ✅  orchestration/dag_workflow.py (LangGraph)   (Day 2 上午)
Step 9  ✅  agent/verifier.py (9规则 + LLM Hybrid)      (Day 2 下午)
Step 10 ✅  agent/replanner.py (局部重规划)              (Day 2 下午)
Step 11 ✅  memory/hierarchical_memory.py (L1/L2/L3)    (Day 2 下午)
Step 12 ✅  logging/trajectory_logger.py (JSONL)         (Day 2 下午)
Step 13 ✅  examples/04_multi_agent_research.py (v1)     (Day 2 下午)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 14 ⬜  GitHub Public Beta (README + 架构图 + 快速开始)
```

## 2.2 实际 vs 计划对比

| 维度 | 原计划 (W1-2) | 实际 (Day 1-2) | 倍率 |
|------|-------------|----------------|------|
| 进度 | Steps 0-7 | Steps 0-13 | 1.9× |
| 测试 | ~30 | 284 | 9.5× |
| 源码行数 | ~2000 | 5858 | 2.9× |
| 测试行数 | ~500 | 3067 | 6.1× |
| 端到端 Demo | 1 个 | 4 个 | 4× |
| LLM 集成 | 未规划 | 已打通 (DeepSeek) | — |
| Verifier | Phase 2 (W5-6) | 已完成 (规则 + LLM Hybrid) | — |
| Replanner | Phase 2 (W5-6) | 已完成 (规则 + LLM) | — |
| Memory L1/L2 | Phase 2 (W7-8) | 已完成 | — |
| Trajectory Logger | Phase 2 (W7-8) | 已完成 (异步 JSONL) | — |

**结论**: Claude Code 辅助开发将开发周期从 16 周压缩到 2 天。

---

# 三、仓库结构（实际）

```
horizonrl-agent/
├── CLAUDE.md                         ← 精简入口 + 开发状态
├── CLAUDE.local.md                   ← 个人偏好 (git-ignored)
├── DEVELOP_PLAN.md                   ← 本文件 (v4.0)
├── deep-research-report.md           ← 详细设计文档
├── .env.example                      ← API Key 配置模板
├── .gitignore
├── pyproject.toml
├── requirements.txt
│
├── configs/                          ← 配置文件
│   ├── default.yaml                  ← 生产 (gpt-4o)
│   ├── dev.yaml                      ← 开发 (DeepSeek)
│   └── eval.yaml                     ← 评测 (temperature=0)
│
├── src/horizonrl/                    ← ★ 源码 (~5858 行)
│   ├── __init__.py                   ← v0.1.0
│   │
│   ├── schemas/                      ← ★ 数据协议 (1061 行)
│   │   ├── __init__.py               (69 行)
│   │   ├── task.py                   (320 行) UserTask, TaskSpec, PlanGraph, PlanNode, PlanPatch
│   │   ├── result.py                 (256 行) StepResult, VerificationResult, EvidenceItem, ToolCall
│   │   ├── event.py                  (248 行) TrajectoryEvent, TrajectorySession, EventType(30种)
│   │   └── report.py                 (168 行) FinalReport, ReportSection, CitationMap
│   │
│   ├── config/
│   │   └── settings.py               (686 行) LLMConfig, MemoryConfig, AgentRuntimeConfig, RootConfig
│   │
│   ├── tools/                        ← ★ 工具层 (834 行)
│   │   ├── __init__.py               (33 行)
│   │   ├── manager.py                (426 行) ToolManager, CircuitBreaker(3态), 8错误分类
│   │   ├── web_search.py            (79 行) Brave API / DuckDuckGo
│   │   ├── arxiv_search.py          (86 行) Arxiv API
│   │   ├── code_execution.py        (103 行) subprocess 沙箱
│   │   └── mock.py                  (107 行) MockWebSearch, MockArxivSearch, MockCodeExecution
│   │
│   ├── llm/                          ← LLM 调用层 (140 行)
│   │   ├── client.py                 (135 行) LLMClient: OpenAI-compatible, async, 超时/错误处理
│   │   └── __init__.py               (5 行)
│   │
│   ├── agent/                        ← Agent 业务逻辑 (1611 行)
│   │   ├── __init__.py               (17 行)
│   │   ├── planner.py                (368 行) Planner(模板2类) + LLMPlanner(LLM驱动)
│   │   ├── worker.py                 (270 行) AgentWorker + execute_workers()
│   │   ├── verifier.py               (448 行) Verifier(rule/llm/hybrid) + RuleEngine(9道检查)
│   │   └── replanner.py              (508 行) Replanner + LLMReplanner + 9种ErrorType策略
│   │
│   ├── orchestration/                ← 顶层编排 (515 行)
│   │   ├── __init__.py               (13 行)
│   │   └── dag_workflow.py           (502 行) ResearchOrchestrator, LangGraph StateGraph
│   │
│   ├── memory/                       ← 分层记忆 (567 行)
│   │   ├── __init__.py               (17 行)
│   │   └── hierarchical_memory.py    (550 行) L1RecentWindow + L2SemanticSummary + MemoryContext
│   │
│   ├── logging/                      ← 轨迹日志 (432 行)
│   │   ├── __init__.py               (21 行)
│   │   └── trajectory_logger.py      (411 行) TrajectoryLogger + 5个分析工具函数
│   │
│   ├── eval/                         ← Phase 4 预留
│   └── rl/                           ← Phase 3+ 预留
│
├── examples/                         ← 4 个可运行 Demo (2369 行)
│   ├── 01_async_demo.py              (985 行) asyncio 完整教程 (10 示例)
│   ├── 02_simple_agent.py            (392 行) 端到端 Demo (无需API)
│   ├── 03_llm_demo.py                (282 行) LLM 连接测试 + 智能规划 Demo
│   └── 04_multi_agent_research.py    (710 行) v1 旗舰 Demo (6-Stage Pipeline)
│
├── tests/                            ← 8 个测试文件, 284 tests (3067 行)
│   ├── test_imports.py               (199 行)  26个内部模块 + 核心依赖
│   ├── test_tools_manager.py         (211 行)  19 tests: 熔断器/超时/重试/统计
│   ├── test_planner.py               (92 行)   9 tests: 模板分解/DAG结构
│   ├── test_worker.py                (150 行)  8 tests: 执行/证据/并发
│   ├── test_verifier.py              (279 行)  24 tests: 规则引擎/Hybrid/错误映射
│   ├── test_replanner.py             (627 行)  51 tests: 策略映射/补丁/重试限制
│   ├── test_memory.py                (535 行)  54 tests: L1/L2/压缩/检索/上下文
│   ├── test_trajectory_logger.py     (638 行)  41 tests: 写入/会话/分析/过滤
│   └── test_dag_workflow.py          (336 行)  28 tests: 图结构/路由/端到端
│
├── scripts/
│   └── update_resume.py
│
├── summaries/                        ← Demo 输出的 Markdown 报告
├── trajectories/                     ← Demo 输出的 JSONL 轨迹日志
│
└── .claude/                          ← Claude Code 配置
    ├── settings.json
    ├── settings.local.json
    ├── commands/                     ← /review, /eval, /train
    ├── rules/                        ← code-style, testing, architecture
    ├── skills/                       ← security-review
    └── agents/                       ← code-reviewer, test-generator
```

---

# 四、开发历程回顾

## 4.1 实际开发顺序

```
Day 1 (05/12)
  上午: Step 0-1 (骨架) → Step 2 (schemas/) → Step 3 (configs/)
  下午: Step 4 (01_async_demo) → Step 5 (tools/) → Step 6 (agent/planner+worker)
        Step 7 (02_simple_agent 端到端) → Step 8 (orchestration)
  ── 当天: 8 步完成, 138 tests ──

Day 2 (05/13)
  上午: API 配置 (LLMClient + LLMPlanner + DeepSeek 打通) → Step 9 (verifier)
        → 优化: tools/mock.py 标准化, CLAUDE.md 重写
  下午: Step 10 (replanner) → Step 11 (memory L1/L2)
        → Step 12 (trajectory_logger) → Step 13 (04_multi_agent v1 Demo)
  ── 当天: 5 步完成 + 1 优化, 284 tests ──
```

## 4.2 关键设计决策记录

| 决策 | 时间 | 选择 | 事后验证 |
|------|------|------|---------|
| Schema-first | Day 1 | 先冻结 4 层 16 数据结构，再写功能 | ✅ 零接口冲突 |
| LangGraph StateGraph | Day 1 | dict state 绕过序列化问题 | ⚠️ 技术债，Phase 2 升级 |
| Pydantic V2 三级配置 | Day 1 | 默认→YAML→.env 合并 | ✅ 灵活且安全 |
| Circuit Breaker 模式 | Day 2 | 三态熔断器 (CLOSED/OPEN/HALF_OPEN) | ✅ 防止级联故障 |
| Rule + Hybrid Verifier | Day 2 | 规则覆盖90% (0.1ms) + LLM复核边界 | ✅ 快速且准确 |
| ErrorType → PatchType 映射 | Day 2 | 九种错误各自有恢复策略 | ✅ Replanner 高效 |
| L1→L2 自动压缩 | Day 2 | Token 阈值触发，模板/LLM 双模式 | ✅ 零成本默认 |
| 异步 JSONL 日志 | Day 2 | asyncio.Queue + 后台 writer | ✅ 非阻塞，不丢事件 |
| 统一 Mock 工具 | Day 2 | tools/mock.py 避免重复定义 | ✅ 3 Demo 2 Test 复用 |

## 4.3 踩过的坑

1. **Windows GBK 编码** — 中文 emoji 输出报 `UnicodeEncodeError`，加 `sys.stdout.reconfigure`
2. **LangGraph 序列化** — JsonPlusSerializer 不能处理嵌套 dataclass，改用 dict + 转换函数
3. **API Key 优先级** — 系统环境变量覆盖 .env，需 `override=True` + `_inject_api_keys()`
4. **Verifier 规则顺序** — "全部工具失败"误判空结果，需调整 9 道检查优先级
5. **`nonlocal` 误用** — 同作用域变量不需要 nonlocal 声明
6. **LLM 回退双写** — `compress_with_llm` 回退时 `add()` 被调用两次

---

# 五、数据流（实际实现）

```
UserTask "Transformer 注意力机制..."
    │
    ▼
LLMPlanner / Planner ──→ PlanGraph (5 TaskSpec, DAG 依赖)
    │
    ▼
ResearchOrchestrator (LangGraph StateGraph)
    │  plan_task → mark_ready ⟲ execute_batch → finalize
    ▼
AgentWorker × N (asyncio + Semaphore 并发)
    │  ToolManager.call() → CircuitBreaker → Timeout → Retry → Tool
    ▼
StepResult + EvidenceItem[]                    ← 每次工具调用
    │
    ▼
Verifier (rule engine 9 checks / LLM deep)
    │  VerificationResult {pass, score, error_type, gaps, actions}
    ▼
Replanner (ErrorType → PatchType)
    │  PlanPatch {RETRY/ADD/REMOVE/REORDER}
    │  apply_patch() → 回写 PlanGraph
    ▼
HierarchicalMemory
    │  record(StepResult, VerificationResult) → L1
    │  auto_compress() → L1→L2 语义摘要
    ▼
TrajectoryLogger (异步 JSONL)
    │  全程记录: 30种 EventType, 每步 timestamp + cost + latency
    ▼
FinalReport (结构化 Markdown + 证据引用)
```

---

# 六、MVP 功能清单（最终版）

| 模块 | 文件 | 行数 | 测试 | 状态 |
|------|------|------|------|------|
| **Schemas** 数据协议 | 4 文件 | 1061 | 间接 | ✅ |
| **Config** 配置管理 | settings.py | 686 | 间接 | ✅ |
| **ToolManager** 工具层 | manager.py + 4工具 | 834 | 19 | ✅ |
| **Planner** 任务分解 | planner.py (Template + LLM) | 368 | 9 | ✅ |
| **Worker** 并行执行 | worker.py | 270 | 8 | ✅ |
| **Verifier** 质量验证 | verifier.py (Rule + LLM Hybrid) | 448 | 24 | ✅ |
| **Replanner** 局部重规划 | replanner.py (9种策略) | 508 | 51 | ✅ |
| **DAG Orchestrator** | dag_workflow.py (LangGraph) | 502 | 28 | ✅ |
| **Memory L1/L2** | hierarchical_memory.py | 550 | 54 | ✅ |
| **Trajectory Logger** | trajectory_logger.py (JSONL) | 411 | 41 | ✅ |
| **LLM Client** | client.py (OpenAI-compatible) | 135 | 间接 | ✅ |
| **v1 Demo** | 04_multi_agent_research.py | 710 | 端到端验证 | ✅ |
| **总计** | 30 文件 | 5858 | 284 | — |

---

# 七、下一步行动

```
Phase 1 收尾 (本周):
  ⬜ Step 14: GitHub Public Beta
      ├── README.md (项目介绍 + 架构图 + 快速开始)
      ├── .github/workflows/ (CI: pytest)
      ├── LICENSE (MIT)
      └── 推送到 GitHub

Phase 2 (下周):
  ⬜ LangGraph State TypedDict 升级
  ⬜ Planner Hybrid 融合 (LLM + Template fallback)
  ⬜ L3 FAISS 向量检索
  ⬜ examples/05_ablation_study.py (消融实验框架)

Phase 3+ (GPU 环境就绪后):
  ⬜ vLLM 部署 + 批量推理
  ⬜ GRPO/PPO RL 训练管线
  ⬜ Benchmark 评测
```

---

# 八、协作经验总结

经过 2 天 13 个 Step 的实际协作，验证了以下模式高效：

1. **一次一个模块** — 专注单文件，实现→测试→通过→下一步
2. **Schema 是合同** — 模块间通过 Schema 通信，不需要读对方的代码
3. **先跑通再优化** — 每个 Step 写完立刻跑测试，红了就修
4. **Demo 驱动开发** — 每个阶段都有可运行的 Demo 验证集成
5. **测试即文档** — 284 tests 比任何注释都更准确描述行为
6. **中文沟通 + 英文代码** — 注释解释 WHY，代码表达 WHAT

---

*本文件 v4.0，基于 Day 1-2 完整开发数据。*
*核心原则不变：先能跑，再好看 → 先可测，再扩展 → 先日志化，再 RL → 先 GitHub，再论文*
