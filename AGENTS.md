# AGENTS.md

This file provides guidance to Codex when working in this repository.

## Project: HorizonRL-Agent

面向开发者和知识工作者的 AI 深度研究助手。输入一个研究问题，自动搜索网络和学术论
文、交叉验证、撰写结构化报告。支持 CLI / Web / API 三种使用方式，可私有化部署。

三项核心功能: **Hierarchical Memory** (L1→L2→L3 with real Embedding API)、
**Verifier-guided Replanning** (9 rules → 9 strategies)、**Async Multi-Agent DAG Orchestration**。

目标: 打造国内可用的、生产级 AI 深度研究工具。

## Quick Reference

| 想看什么 | 去哪儿 |
|---------|------|
| 开发路线图 | `DEVELOPMENT_PLAN.md` |
| 详细设计 | `SYSTEM_MANUAL.md` |
| 代码规范 | `.Codex/rules/code-style.md` |
| 测试规范 | `.Codex/rules/testing.md` |
| 架构约束 | `.Codex/rules/architecture.md` |
| 个人偏好 | `Codex.local.md`（git-ignored） |
| API 配置 | `.env.example` → 复制为 `.env` |

## Development Status (2026-05-18 Day 8 工程化转型)

```
Phase 1+2: 全部完成 + Day 6-8 产品化增强

Day 8 核心交付:
  ✅ 文档工程化转型 — 清除论文痕迹，重定位为产品
  ✅ 论文搜索根治 — OpenAlex 主力 (国内可用)
  ✅ 工具熔断优化 — 阈值调整 + 全局速率限制
  ✅ FastAPI Web v4 — SSE 防重入 + 下载修复

测试: 355 passed, 4 skipped, 0 failed
```

## Architecture

```
src/horizonrl/
├── schemas/           数据协议 (4 文件, 16 数据结构)
│   ├── task.py        TaskSpec · PlanGraph · PlanNode · PlanPatch · UserTask
│   ├── result.py      StepResult · VerificationResult · EvidenceItem · SearchProvenance
│   ├── event.py       TrajectoryEvent · TrajectorySession · EventType (30 种)
│   └── report.py      FinalReport · ReportSection · CitationMap · ReportMetadata
│
├── config/            配置管理 (Pydantic V2 三级合并: 代码 → YAML → .env)
│   └── settings.py    LLMConfig · MemoryConfig · AgentRuntimeConfig · ToolsConfig
│
├── services/          共享服务层 — CLI/Web/Benchmark 统一入口
│   └── research_service.py  SessionArtifacts · run/stream · 模式判断 · L3注入
│
├── tools/             工具层 (5 文件)
│   ├── manager.py     ToolManager: 超时/重试/熔断/统计/结果规范化
│   ├── web_search.py  5 后端并发竞速 (Bocha/Brave/DDGS/Wikipedia/Mock)
│   ├── arxiv_search.py  双端点并发竞速 + Mock 回退
│   ├── code_execution.py  AST 检测 + 5 模板自动生成 + 安全沙箱
│   └── mock.py        离线/CI 用 Mock 工具
│
├── llm/               LLM 客户端
│   └── client.py      chat() + chat_stream() + embed() · OpenAI 兼容
│
├── agent/             Agent 逻辑 (5 文件)
│   ├── planner.py     Planner (5 种任务类型) + LLMPlanner (LLM DAG 拆解 + 时间感知)
│   ├── worker.py      AgentWorker (多工具并发 + 证据提取 + 搜索词清洗)
│   ├── verifier.py    RuleEngine (9 规则) + Verifier (rule/llm/hybrid)
│   ├── replanner.py   Replanner (9 策略) + LLMReplanner + 状态重置
│   └── writer.py      UserAnswerWriter + DebugReportRenderer (LLM + 流式 + 证据清洗)
│
├── orchestration/     编排层
│   └── dag_workflow.py  ResearchOrchestrator: LangGraph 6 节点 DAG
│                       session_id 注入 · writer 注入 · logger 注入 · L3 注入
│
├── memory/            分层记忆
│   └── hierarchical_memory.py  L1RecentWindow · L2SemanticSummary · L3EpisodicArchive
│                               (DashScope Embedding API / n-gram MD5 回退)
│
└── logging/           轨迹日志
    └── trajectory_logger.py  TrajectoryLogger (异步 JSONL) + 5 分析工具
```

## Data Flow

```
UserTask (自然语言)
    │
    ▼
Planner / LLMPlanner ──→ PlanGraph (5 种任务类型, DAG 依赖)
    │  ↑ L3 检索历史经验
    ▼
ResearchOrchestrator (LangGraph 6 节点, 全链路集成)
    │  plan_task → mark_ready → execute_batch → verify_batch → replan → finalize
    │  死锁检测 · 迭代上限 · Semaphore 并发 · 批次超时 · 会话隔离
    ▼
AgentWorker × N (asyncio 并发)
    │  ToolManager → CircuitBreaker → 超时/重试 → 工具 (Web/Arxiv/Code)
    ▼
StepResult + EvidenceItem[] (SearchProvenance)
    │
    ├──→ Verifier (rule/llm/hybrid) → VerificationResult
    │      └──→ Replanner → PlanPatch (RETRY/ADD/REMOVE/REORDER)
    │
    ├──→ HierarchicalMemory (L1 FIFO → L2 语义摘要 → L3 Embedding 检索)
    │      └── _verify_batch: archive_to_l3() → save()
    │
    ├──→ TrajectoryLogger (异步 JSONL, 25+ 事件/次, per-node 追踪)
    │
    └──→ Writer v2
            ├── UserAnswerWriter → final_answer.md (LLM + Token 流式)
            └── DebugReportRenderer → debug_report.md (证据清洗 + 智能截断)
```

## Key Classes

| 类 | 文件 | 职责 |
|----|------|------|
| `Planner` | agent/planner.py | 5 种任务类型模板分解 (research/code/comparison/summary/factual_qa) |
| `LLMPlanner` | agent/planner.py | LLM DAG 拆解 + 时间感知 + 最大化并行度 |
| `AgentWorker` | agent/worker.py | 多工具并发执行 + 证据提取 + 搜索词清洗 |
| `Verifier` | agent/verifier.py | 三模式验证 (rule/hybrid/llm) |
| `RuleEngine` | agent/verifier.py | 9 道规则检查 (~0.1ms) |
| `Replanner` | agent/replanner.py | 9 策略局部重规划 + 状态重置 |
| `UserAnswerWriter` | agent/writer.py | LLM 流式写作 + 时间感知 + 证据清洗 |
| `DebugReportRenderer` | agent/writer.py | 开发者报告 + 智能截断 + 工具追踪 |
| `ToolManager` | tools/manager.py | 超时/重试/熔断/统计/搜索规范化 |
| `WebSearchTool` | tools/web_search.py | 5 后端并发竞速 (AUTO: FIRST_COMPLETED) |
| `ArxivSearchTool` | tools/arxiv_search.py | 双端点 + arxiv pkg 并发竞速 + Mock 兜底 |
| `CodeExecutionTool` | tools/code_execution.py | AST 检测 + 5 模板自动代码生成 |
| `ResearchOrchestrator` | orchestration/ | LangGraph 6 节点 DAG, 全模块注入 |
| `LLMClient` | llm/client.py | chat/chat_stream/embed · OpenAI 兼容 |
| `HierarchicalMemory` | memory/ | L1/L2/L3, Embedding API + n-gram 回退 |
| `L3EpisodicArchive` | memory/ | FAISS 向量检索 + 持久化 |
| `TrajectoryLogger` | logging/ | 异步 JSONL + 25+ per-node 事件 |
| `SessionArtifacts` | services/ | 会话完整产出 (报告/轨迹/统计/指标) |
| `Evaluator` | benchmarks/ | 结构化评测 (mock_ratio/citation/trajectory) |

## Running

```bash
# 测试 (330 tests)
python -m pytest tests/ -v

# Demo (7 个)
python examples/01_async_demo.py           # asyncio 教程
python examples/02_simple_agent.py         # 最简端到端
python examples/03_llm_demo.py             # LLM 连接测试
python examples/04_multi_agent_research.py # 旗舰全链路 (--llm 可选)
python examples/05_web_agent.py            # Web 界面 (SSE + Token流式)
python examples/06_ablation_study.py       # 组件分析
python examples/07_benchmark.py            # Benchmark (--llm 可选)
```

## Environment

- Python 3.10+ (Ubuntu server) / Python 3.13 (Windows local)
- CPU 即可运行 (无需 GPU)
- LLM: DeepSeek (OpenAI-compatible), Embedding: DashScope
