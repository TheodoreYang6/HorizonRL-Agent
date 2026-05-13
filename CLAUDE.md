# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project: HorizonRL-Agent

A Long-Horizon Agentic RL System enabling LLM agents to stably complete 20+ step complex tasks.
Three core innovations: **Hierarchical Memory** (L1→L2→L3), **Verifier-guided Replanning**,
and **Async Multi-Agent DAG Orchestration**.

Target: AAAI/IJCAI/ACL Findings 2027 submission + open-source release.

## Quick Reference

| 想看什么 | 去哪儿 |
|---------|------|
| 开发路线图 | `DEVELOPMENT_PLAN.md` |
| 详细设计 | `deep-research-report.md` |
| 代码规范 | `.claude/rules/code-style.md` |
| 测试规范 | `.claude/rules/testing.md` |
| 架构约束 | `.claude/rules/architecture.md` |
| 个人偏好 | `CLAUDE.local.md`（git-ignored） |
| API 配置 | `.env.example` → 复制为 `.env` |

## Development Status (2026-05-13 最终)

```
Phase 1: 核心基础设施 — 2 天完成, MVP 可运行, GitHub 已发布
  ✅ Step 0-1:  项目骨架 (.claude/, .gitignore)
  ✅ Step 2:    schemas/ (4文件16数据结构)
  ✅ Step 3:    configs/ (Pydantic V2 三级配置)
  ✅ Step 4:    examples/01_async_demo.py (10示例)
  ✅ Step 5:    tools/manager.py + 3工具 + mock
  ✅ Step 6:    agent/planner.py + worker.py
  ✅ Step 7:    examples/02_simple_agent.py (端到端)
  ✅ Step 8:    orchestration/dag_workflow.py (LangGraph)
  ✅ Step 9:    agent/verifier.py (9规则 + LLM Hybrid)
  ✅ Step 10:   agent/replanner.py (局部重规划)
  ✅ Step 11:   memory/hierarchical_memory.py (L1/L2/L3)
  ✅ Step 12:   logging/trajectory_logger.py (JSONL)
  ✅ Step 13:   examples/04_multi_agent_research.py (v1 Demo)
  ⬜ Step 14:   GitHub Public Beta (README/架构图/快速开始)

  测试: 296 passed, 4 skipped, 0 failed
  代码: ~14,000 行 (源码 + 测试 + Demo + 文档)
  LLM: DeepSeek 已打通  |  Web Search: ddgs + Wikipedia 国内可用
  GitHub: 已发布  |  ruff: 代码质量已优化
```

## Tech Stack

- **Agent Framework**: LangGraph (StateGraph, conditional routing, InMemorySaver)
- **Async**: Python asyncio (gather, Semaphore, wait_for, run_in_executor)
- **LLM Client**: OpenAI SDK → DeepSeek/OpenAI/vLLM (OpenAI-compatible API)
- **Inference**: vLLM (Phase 3+, continuous batching)
- **RL Training**: TRL + veRL (Phase 3+, GRPOTrainer, PPOTrainer)
- **Memory**: FAISS (Phase 2+, vector retrieval for hierarchical memory)
- **Hardware**: 2×A800 80GB or 1×A100 80GB (server)

## Architecture (Actual Implementation)

```
src/horizonrl/
├── schemas/           ← 数据协议 (4文件, 16数据结构, 1061行)
│   ├── task.py        TaskSpec, PlanGraph, PlanNode, PlanPatch, UserTask
│   ├── result.py      StepResult, VerificationResult, EvidenceItem, ToolCall, ErrorType
│   ├── event.py       TrajectoryEvent, TrajectorySession, EventType (30种)
│   └── report.py      FinalReport, ReportSection, CitationMap
│
├── config/            ← 配置管理 (686行)
│   └── settings.py    Pydantic V2: LLMConfig, MemoryConfig, AgentRuntimeConfig, etc.
│
├── tools/             ← 工具层 (834行)
│   ├── manager.py     ToolManager: 超时/重试/熔断/统计, CircuitBreaker(三态)
│   ├── mock.py        MockWebSearch, MockArxivSearch, MockCodeExecution
│   ├── web_search.py  Brave API / DuckDuckGo
│   ├── arxiv_search.py Arxiv API
│   └── code_execution.py  subprocess 沙箱
│
├── llm/               ← LLM 调用层 (140行)
│   └── client.py      LLMClient: OpenAI-compatible, async chat(), token 统计
│
├── agent/             ← Agent 业务逻辑 (1611行)
│   ├── planner.py     Planner (模板2类) + LLMPlanner (LLM驱动 DAG拆解)
│   ├── worker.py      AgentWorker (异步执行, 证据提取) + execute_workers()
│   ├── verifier.py    Verifier (rule/llm/hybrid), RuleEngine (9道检查), LLMVerifier
│   └── replanner.py   Replanner + LLMReplanner, 9种 ErrorType→PatchType 策略
│
├── orchestration/     ← 顶层编排 (515行)
│   └── dag_workflow.py  ResearchOrchestrator: LangGraph StateGraph DAG调度
│
├── memory/            ← 分层记忆 (567行)
│   └── hierarchical_memory.py  L1RecentWindow + L2SemanticSummary + MemoryContext
│
├── logging/           ← 轨迹日志 (432行)
│   └── trajectory_logger.py  TrajectoryLogger (异步JSONL) + 5分析工具
│
├── eval/              ← 评测指标 [Phase 4]
└── rl/                ← RL 训练 [Phase 3+]
```

## Data Flow

```
UserTask (自然语言)
    │
    ▼
Planner / LLMPlanner ──→ PlanGraph (TaskSpec[] + DAG)
    │
    ▼
ResearchOrchestrator (LangGraph StateGraph)
    │  plan_task → mark_ready ⟲ execute_batch → finalize
    │  死锁检测 / 迭代上限 / 并发控制
    ▼
AgentWorker (asyncio 并发)
    │  ToolManager.call() → 熔断 → 超时 → 重试 → 工具
    ▼
StepResult + EvidenceItem[]
    │
    ├──→ Verifier (rule / llm / hybrid)
    │      VerificationResult {pass, score, error_type, feedback, evidence_gaps, suggested_actions}
    │        │
    │        ▼ (if failed)
    │      Replanner → PlanPatch {RETRY/ADD/REMOVE/REORDER} → (回写 PlanGraph)
    │
    ├──→ HierarchicalMemory
    │      L1 (record) → auto_compress → L2 (语义摘要)
    │
    └──→ TrajectoryLogger (异步 JSONL, 全程)
          30种 EventType, timestamp + cost + latency
    │
    ▼
FinalReport (结构化 Markdown + 证据引用)
```

## Key Classes

| 类 | 文件 | 职责 |
|----|------|------|
| `Planner` | agent/planner.py | 模板任务分解 (2种类型, 5子任务) |
| `LLMPlanner` | agent/planner.py | LLM 智能任务分解 (任意类型) |
| `AgentWorker` | agent/worker.py | 异步子任务执行 + 证据提取 |
| `Verifier` | agent/verifier.py | 结构化验证 (rule/hybrid/llm) |
| `RuleEngine` | agent/verifier.py | 9道规则检查 (0.1ms) |
| `Replanner` | agent/replanner.py | 局部重规划 (9种策略映射) |
| `LLMReplanner` | agent/replanner.py | LLM 增强重规划 (失败回退规则) |
| `ToolManager` | tools/manager.py | 统一工具入口 (超时/重试/熔断) |
| `CircuitBreaker` | tools/manager.py | 三态熔断器 (CLOSED→OPEN→HALF_OPEN) |
| `ResearchOrchestrator` | orchestration/dag_workflow.py | LangGraph DAG 编排 |
| `LLMClient` | llm/client.py | OpenAI-compatible LLM 客户端 |
| `HierarchicalMemory` | memory/hierarchical_memory.py | L1/L2/L3 分层记忆 |
| `L1RecentWindow` | memory/hierarchical_memory.py | L1 FIFO 窗口, Token 阈值压缩 |
| `L2SemanticSummary` | memory/hierarchical_memory.py | L2 语义摘要 (模板/LLM) |
| `MemoryContext` | memory/hierarchical_memory.py | Agent 消费的结构化上下文 |
| `TrajectoryLogger` | logging/trajectory_logger.py | 异步 JSONL 轨迹日志 |
| `TrajectorySession` | schemas/event.py | 会话管理, 自动统计聚合 |

## Running

```bash
# 测试 (284 tests)
python -m pytest tests/ -v                          # 全部
python -m pytest tests/test_replanner.py -v         # 单模块

# Demo (4 个)
python examples/01_async_demo.py                    # asyncio 教程
python examples/02_simple_agent.py                  # 端到端 (无需API)
python examples/02_simple_agent.py "你的问题"        # 自定义任务
python examples/03_llm_demo.py "你的问题"            # LLM 驱动 (需API)
python examples/04_multi_agent_research.py          # v1 旗舰 Demo (无需API)
python examples/04_multi_agent_research.py --llm    # LLM 模式

# 导入检查
python -m pytest tests/test_imports.py -v
```

## Design Principles

1. Not a demo — benchmark, evaluation, ablation, system analysis
2. Not just prompt engineering — rollout, RL, memory, async systems
3. Focus on **Long-Horizon Agent Stability** as the single deep research direction
4. Every module must be evaluable in isolation (ablation-ready)
5. Schema-first: 数据协议先冻结，再写功能代码

## Environment

- Python 3.10+ (Ubuntu 22.04 server) / Python 3.13 (Windows local)
- GPU: 2×A800 80GB or 1×A100 80GB
- API: DeepSeek (OpenAI-compatible endpoint: `https://api.deepseek.com`)
