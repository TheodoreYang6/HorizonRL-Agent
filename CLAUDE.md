# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project: Horizon-Agent · 溯证智搜

多 Agent 协同研究系统。输入一个问题，多个 Agent 并发搜索网络和学术论文、
交叉验证、撰写带证据溯源的结构化报告。支持 CLI / Web / API 三种使用方式，可私有化部署。

三项核心技术: **Hierarchical Memory** (L1→L2→L3 with real Embedding API)、
**Verifier-guided Replanning** (9 rules → 9 strategies)、**Async Multi-Agent DAG Orchestration**。

目标: 打造国内可用的、生产级 AI 深度研究工具。

## Quick Reference

| 想看什么 | 去哪儿 |
|---------|------|
| 开发路线图 | `DEVELOPMENT_PLAN.md` |
| 详细设计 | `SYSTEM_MANUAL.md` |
| 代码规范 | `.claude/rules/code-style.md` |
| 测试规范 | `.claude/rules/testing.md` |
| 架构约束 | `.claude/rules/architecture.md` |
| 个人偏好 | `CLAUDE.local.md`（git-ignored） |
| API 配置 | `.env.example` → 复制为 `.env` |

## Development Status (2026-05-19 Day 9 — Phase 1+2 完成)

```
Phase 1: ████████████ 100% (5/5) 产品化基础
Phase 2: ████████████ 100% (6/6) 体验优化

Day 9 交付:
  ✅ SQLite 会话持久化 — 双后端(内存+SQLite), 缓存机制, 自动迁移
  ✅ 会话历史 CRUD — 7 REST 端点, 分页列表, 删除确认
  ✅ React 18 SPA — htm 零构建, 184KB 本地化, 三栏响应式
  ✅ ChromaDB 向量数据库 — L3 双后端, 元数据过滤, 自动持久化 (默认)
  ✅ 100+ 环境变量覆盖 — 全部模块可通过 .env/HORIZON_ 配置
  ✅ GitHub Actions CI/CD — Test Matrix (3.10-3.13) + lint + benchmark smoke
  ✅ 论文搜索 5 后端竞速 — OpenAlex‖S2‖Arxiv Pkg‖Arxiv API×2‖Mock
  ✅ Research Context Engine — 研究上下文引擎, 语义检索替代文本拼接
  ✅ 多轮对话 — 上下文继承, conversation_history 自动追加
  ✅ 暗色/亮色主题 — CSS 变量切换 + localStorage 持久化
  ✅ 研究任务模板 — 5 种模板: 综述/对比/摘要/解释/代码
  ✅ API Key 管理页 — Web 配置, .env 读写
  ✅ ruff lint 零错误 — 30+ 文件自动修复

测试: 414 passed, 4 skipped, 0 failed (+59 vs Day 8)
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
├── tools/             工具层 (6 文件)
│   ├── manager.py     ToolManager: 超时/重试/熔断/统计/结果规范化
│   ├── web_search.py  5 后端并发竞速 (Bocha/Brave/DDGS/Wikipedia/Mock)
│   ├── paper_search.py  OpenAlex + Semantic Scholar, 国内可用 + Mock 兜底
│   ├── arxiv_search.py  原 Arxiv 搜索 (保留兼容)
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
├── web/               Web 界面 (React 18 SPA, v0.2.0)
│   ├── app.py         应用工厂 · lifespan · CORS · 静态挂载 · 懒加载 SQLite
│   ├── models.py      请求/响应 Pydantic 模型 (5 个)
│   ├── session_manager.py  SessionState · SessionManager (内存) · SqliteSessionManager (缓存+SQLite)
│   ├── routes/
│   │   ├── chat.py    POST /api/chat (chat/deep 双模式)
│   │   ├── stream.py  GET /api/stream/{sid} (SSE + asyncio.Queue 桥接 + 防重入)
│   │   ├── report.py  GET /api/report/{sid} + /api/download/{sid}/{kind}
│   │   └── sessions.py  GET/DELETE /api/sessions (历史会话 CRUD)
│   ├── templates/     Jinja2 模板 (仅 index.html 骨架)
│   └── static/
│       ├── css/style.css   v6.0 暗色专业风格 · 三栏响应式
│       ├── js/app.js       React 18 + htm SPA (useReducer 状态管理)
│       └── js/vendor/      React 10KB · ReactDOM 132KB · htm 1.5KB · marked 40KB
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
    │  ToolManager → CircuitBreaker → 超时/重试 → 工具 (Web/Paper/Code)
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
| `PaperSearchTool` | tools/paper_search.py | OpenAlex + Semantic Scholar 双后端, 国内可用 |
| `CodeExecutionTool` | tools/code_execution.py | AST 检测 + 5 模板自动代码生成 |
| `ResearchOrchestrator` | orchestration/ | LangGraph 6 节点 DAG, 全模块注入 |
| `LLMClient` | llm/client.py | chat/chat_stream/embed · OpenAI 兼容 |
| `HierarchicalMemory` | memory/ | L1/L2/L3, Embedding API + n-gram 回退 |
| `L3EpisodicArchive` | memory/ | FAISS 向量检索 + 持久化 |
| `TrajectoryLogger` | logging/ | 异步 JSONL + 25+ per-node 事件 |
| `SessionArtifacts` | services/ | 会话完整产出 (报告/轨迹/统计/指标) |
| `SessionManager` | web/ | 内存会话管理 (开发/测试用) |
| `SqliteSessionManager` | web/ | SQLite 持久化 + 缓存机制 (生产用) |
| `Evaluator` | benchmarks/ | 结构化评测 (mock_ratio/citation/trajectory) |

## Running

```bash
# 测试 (384 tests)
python -m pytest tests/ -v

# Demo (7 个)
python examples/01_async_demo.py           # asyncio 教程
python examples/02_simple_agent.py         # 最简端到端
python examples/03_llm_demo.py             # LLM 连接测试
python examples/04_multi_agent_research.py # 旗舰全链路 (--llm 可选)
python examples/05_web_agent.py            # Web 界面 (SSE + Token流式)
python examples/06_ablation_study.py       # 组件分析 (模块消融)
python examples/07_benchmark.py            # Benchmark (--llm 可选)
```

## Environment

- Python 3.10+ (Ubuntu server) / Python 3.13 (Windows local)
- CPU 即可运行 (无需 GPU)
- LLM: DeepSeek (OpenAI-compatible), Embedding: DashScope
