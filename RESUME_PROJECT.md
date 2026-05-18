# HorizonRL-Agent — 项目简历素材

> 生成日期: 2026-05-17 | 版本: v0.3.0 | 21 commits | 330 tests

---

## 一、项目一句话简介

**HorizonRL-Agent** 是一个面向长链路（20+步）复杂任务的 **多智能体 LLM Agent 稳定执行系统**，
通过 **分层记忆 (Hierarchical Memory)**、**验证器驱动重规划 (Verifier-Guided Replanning)** 和
**异步 DAG 编排 (Async Multi-Agent DAG)** 三项核心创新，解决 LLM Agent 在长任务中的
上下文污染、任务漂移、幻觉累积和失败无法恢复等关键问题。

---

## 二、项目定位

| 维度 | 说明 |
|------|------|
| **领域** | LLM Agent / 多智能体系统 / 自然语言处理 |
| **工程方向** | AI 深度研究助手、多智能体编排、分层记忆、验证驱动恢复 |
| **产品定位** | 面向开发者和知识工作者的私有化 AI 研究工具 |
| **技术栈** | Python 3.10+, LangGraph, asyncio, FAISS, OpenAI SDK, Pydantic V2 |
| **开源状态** | GitHub 已发布 (v0.3.0) |
| **代码规模** | 33 源文件 / 9,251 行源码 / 13 测试文件 / 7 Demo |

---

## 三、核心功能特性 (5 项)

### 1. 分层记忆结构 (Hierarchical Memory: L1 → L2 → L3)

```
L1 RecentWindow (FIFO, Token 阈值触发压缩)
  → 保留最近 K 步完整轨迹，自动驱逐旧数据

L2 SemanticSummary (模板/LLM 语义压缩)
  → 对 L1 溢出内容做结构化摘要
  → 保留：目标、关键发现、失败模式、工具调用统计

L3 EpisodicArchive (FAISS 向量检索 + DashScope Embedding API)
  → 持久化存储历史任务经验
  → 混合检索：向量相似度 + n-gram 精确匹配 + 时间衰减
  → 双模回退: DashScope text-embedding-v4 (1024维) / MD5 n-gram 确定性哈希
```

**设计精髓**: 三层记忆分别解决"最近上下文"、"语义压缩"、"长期经验检索"三个层次的需求，
L1→L2 自动触发，L3 由验证事件驱动归档。MemoryContext 对 Planner/Worker/Verifier
提供统一只读接口，避免上下文污染。

### 2. 验证器驱动的局部重规划 (Verifier-Guided Replanning)

```
RuleEngine (9 条规则, ~0.1ms)
  空结果检查 / 工具调用失败 / 超时 / 长度质量 / 证据充分性
  / 幻觉检测 / 完整性 / 一致性 / 格式检查

LLMVerifier (DeepSeek/OpenAI, ~2s)
  → 对规则引擎不确定的边界 case 做 LLM 深度语义诊断

Replanner (9 种 ErrorType → 4 种 PatchType)
  EMPTY_RESULT → RETRY        HALLUCINATION → ADD
  TOOL_FAILURE → RETRY         INCOMPLETE   → ADD
  TIMEOUT     → REDUCE_SCOPE   INCONSISTENT → REMOVE
  LOW_QUALITY → RETRY          IRRELEVANT   → REPLACE
  FORMAT_ERROR → RETRY
```

**关键设计**: 局部 patch 而非全局重规划 — 只修改 PlanGraph 中受影响子图，
保留已成功步骤的成果。单任务最多 3 次重试，全局最多 5 次，防止无限循环。

### 3. 异步多智能体 DAG 编排 (Async Multi-Agent DAG)

基于 LangGraph StateGraph 的 6 节点状态机:

```
START → plan_task → mark_ready → execute_batch → verify_batch
                    ↑              ↓               ↓
                    └── replan ←── route ←────────┘
                                       ↓
                                   finalize → END
```

- **并发控制**: asyncio.Semaphore 限制最大并发 Worker 数
- **死锁检测**: 连续 N 轮无 ready 任务但仍有 pending → 触发强制重规划
- **批次超时**: asyncio.wait(timeout=120s) 防止单任务拖死整批
- **TypedDict 状态管理**: 累加/替换语义分离，序列化重构 (删除 110 行手写代码)

### 4. Token 级流式输出与 SSE 实时推送

- **LLM Token 流式**: `chat_stream()` 异步生成器，逐 token yield
- **SSE 实时推送**: 8 种事件类型 (stage / tool / verify / token / report_ready / done / sse_error / heartbeat)
- **Web v3 界面**: 双栏布局 (侧栏时间线 + 主聊天区)，Markdown 实时渲染，一键下载
- **Writer 流式写作**: 证据清洗 (换行→空格，句子边界智能截断) + LLM 逐 token 回调

### 5. 轨迹级日志与证据可追溯 (Trajectory Logging & Provenance)

- **30 种事件类型**: 覆盖 session/plan/worker/tool/verify/replan 全生命周期
- **异步 JSONL**: asyncio.Queue + 后台 Writer，不阻塞主循环
- **SearchProvenance**: 每条证据携带来源 URL、检索时间、相关性分数、provider 标识
- **5 种分析工具**: read_session / list / aggregate / event_type_distribution / filter_events

---

## 四、系统架构

```
src/horizonrl/           (33 源文件, 9,251 行)
├── schemas/             数据协议层 (4 文件, 16 数据结构)
│   ├── task.py              UserTask · TaskSpec · PlanGraph · PlanNode · PlanPatch
│   ├── result.py            StepResult · VerificationResult · EvidenceItem · SearchProvenance
│   ├── event.py             TrajectoryEvent · TrajectorySession · EventType (30种)
│   └── report.py            FinalReport · ReportSection · CitationMap · ReportMetadata
│
├── config/              配置管理 (Pydantic V2 三级合并: 代码 → YAML → .env)
│   └── settings.py          9 个 BaseModel, 环境变量嵌套覆盖 (HORIZON_LLM__MODEL=xxx)
│
├── services/            共享服务层 (CLI/Web/Benchmark 统一入口)
│   └── research_service.py  run_research_session() + stream_research_session()
│                            SessionArtifacts: 会话完整产出 (报告/轨迹/统计)
│
├── tools/               工具层 (5 文件)
│   ├── manager.py           ToolManager: 超时/重试/熔断 (三态 CircuitBreaker)
│   ├── web_search.py        5 后端并发竞速 (Bocha/Brave/DDGS/Wikipedia/Mock)
│   ├── arxiv_search.py      双端点 + arxiv pkg 并发竞速 + Atom XML 解析
│   ├── code_execution.py    AST 4层检测 + 5套代码模板 + subprocess 安全沙箱
│   └── mock.py              离线/CI 用 Mock 工具
│
├── llm/                 LLM 调用层 (OpenAI 兼容)
│   └── client.py            chat() + chat_stream() + embed() · 超时/错误处理
│
├── agent/               Agent 业务逻辑 (5 文件)
│   ├── planner.py           Planner (5种模板) + LLMPlanner (LLM DAG拆解 + 时间感知)
│   ├── worker.py            AgentWorker (多工具 asyncio.gather 并发执行)
│   ├── verifier.py          RuleEngine (9规则) + LLMVerifier + Verifier (三模式)
│   ├── replanner.py         Replanner (9策略) + LLMReplanner + 状态重置
│   └── writer.py            UserAnswerWriter + DebugReportRenderer + Writer
│
├── orchestration/       编排层
│   └── dag_workflow.py      ResearchOrchestrator: LangGraph 6节点, 全模块注入
│
├── memory/              分层记忆
│   └── hierarchical_memory.py  L1/L2/L3 + FAISS + Embedding API
│
└── logging/             轨迹日志
    └── trajectory_logger.py  异步 JSONL + 5 分析工具
```

### 完整数据流

```
UserTask (自然语言)
    │
    ▼
Planner / LLMPlanner ──→ PlanGraph (5-8 子任务, DAG 依赖)
    │  ↑ L3 检索历史经验
    ▼
ResearchOrchestrator (LangGraph 6 节点)
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
    ├──→ TrajectoryLogger (异步 JSONL, 25+ 事件/次)
    └──→ Writer v2 → final_answer.md + debug_report.md
```

---

## 五、关键技术指标

### 5.1 代码与测试

| 指标 | 数值 |
|------|------|
| 源码文件 | 33 |
| 测试文件 | 13 |
| Demo 文件 | 7 |
| 源码行数 | 9,251 |
| 测试行数 | 3,786 |
| Demo 行数 | 3,802 |
| 总行数 | ~16,839 |
| 测试用例 | 330 passed, 4 skipped, 0 failed |
| Git commits | 21 |
| 模块数 | 15 |

### 5.2 核心性能

| 指标 | 数值 | 说明 |
|------|------|------|
| 端到端延迟 (Mock) | 2.3s | 5-6 子任务并发执行 |
| 端到端延迟 (LLM) | 12.9s | DeepSeek V3 + 真实搜索 |
| 并发加速比 | 3.8x | vs Day 5 之前的串行版本 |
| L3 Embedding 维度 | 1024 | DashScope text-embedding-v4 |
| 熔断恢复时间 | 30s | CircuitBreaker 冷却窗口 |
| 工具调用超时 | 12s | ToolManager 默认超时 |

### 5.3 Benchmark 数据 (真实 API)

| 类别 | 运行 | 通过率 | Mock% | 重规划 | 工具调用 |
|------|------|--------|-------|--------|----------|
| 事实知识 | 16 | 75.0% | 8.1% | 1.2 | 3.6 |
| 技术对比 | 16 | 100.0% | 11.7% | 0.0 | 5.0 |
| **合计** | **32** | **87.5%** | **9.9%** | **0.6** | **4.3** |

- 真实搜索证据占比 **90.1%** (mock_ratio 仅 9.9%)
- 对比类任务 100% 通过，0 重规划
- 40 题 × 3 轮 Mock 模式: **100% 通过**

---

## 六、工程能力体现

### 6.1 架构设计能力

- **Schema-First 设计**: 16 个数据结构在先，所有模块通过 Schema 通信，0 接口冲突
- **依赖方向单向**: schemas ← config/tools/llm ← agent/memory ← orchestration，无循环依赖
- **关注点分离**: 数据协议 / 配置 / 工具 / LLM / Agent 逻辑 / 编排 / 记忆 / 日志 8 层独立
- **注入机制**: orchestrator 通过构造函数注入 writer / logger / embedding_client / on_token

### 6.2 稳定性工程

- **多层回退**: LLM→规则→模板, API→并发竞速→Mock, Embedding API→n-gram 哈希
- **熔断保护**: 三态 CircuitBreaker (CLOSED→OPEN→HALF_OPEN)，连续 3 次失败触发
- **死锁检测**: DAG 依赖图循环检测 + 自动触发强制重规划
- **批次超时**: asyncio.wait(timeout=120s)，超时后取消传播
- **会话隔离**: 每次 run() 自动重置 Replanner 和 Memory

### 6.3 并发与性能

- **工具内并发**: asyncio.gather 并发执行多工具 (web_search + arxiv_search 同一 Worker)
- **Worker 间并发**: asyncio.Semaphore 控制，DAG 依赖自动拓扑排序
- **搜索竞速**: Bocha/Brave/DDGS/Wikipedia 4 后端 FIRST_COMPLETED 并发
- **Arxiv 三端点竞速**: arxiv pkg + export.arxiv.org + arxiv.org 并发
- **异步日志**: asyncio.Queue + 后台 Writer，不阻塞主循环

### 6.4 测试与质量

- **330 tests, 0 failures**，覆盖率覆盖所有 15 个模块
- **E2E 测试**: 完整 LangGraph 工作流测试 (28 tests)
- **消融实验框架**: 5 种配置 × 6 题 = 30 次运行，量化每个模块贡献
- **Benchmark 系统**: 40 题 5 类别，支持 mock/real 双模式

### 6.5 全栈能力

- **后端**: Python asyncio + LangGraph + aiohttp (11 路由)
- **前端**: 原生 HTML/JS (无需构建)，SSE EventSource + Markdown 实时渲染
- **LLM 集成**: OpenAI 兼容协议，支持 DeepSeek/DashScope/任意兼容端点
- **向量检索**: FAISS + DashScope Embedding API + n-gram 回退
- **CI/CD**: GitHub Actions，mock-only 必过路径

---

## 七、技术栈清单

| 层次 | 技术 | 用途 |
|------|------|------|
| **语言** | Python 3.10+ / 3.13 | 全项目 |
| **Agent 框架** | LangGraph (StateGraph) | DAG 编排 + 条件路由 |
| **异步** | asyncio (gather, Semaphore, Queue, wait) | 并发执行 + 流式输出 |
| **LLM** | OpenAI SDK → DeepSeek V3 | 推理 (Planner/Verifier/Replanner/Writer) |
| **Embedding** | DashScope text-embedding-v4 (1024维) | L3 向量检索 |
| **向量检索** | FAISS (CPU) | L3 持久化索引 |
| **配置** | Pydantic V2 + YAML + .env | 三级配置合并 |
| **Web 后端** | aiohttp | HTTP + SSE 服务 |
| **Web 前端** | 原生 HTML/CSS/JS | SSE EventSource + Markdown |
| **搜索** | Bocha API / Brave API / DDGS / Wikipedia | 多后端并发竞速 |
| **学术搜索** | Arxiv API + arxiv Python pkg | 论文检索 |
| **代码沙箱** | subprocess + AST 检测 | 安全代码执行 |
| **测试** | pytest + pytest-asyncio | 330 tests |
| **CI/CD** | GitHub Actions | 自动测试 (mock-only) |
| **版本控制** | Git + GitHub | 21 commits |

---

## 八、个人角色与贡献

### 架构设计
- 独立设计并实现了完整的 8 层模块化架构 (schemas → config → tools → llm → agent → memory → orchestration → logging)
- 设计了 16 个核心数据结构，定义了全项目通信协议
- 设计了分层记忆的三层结构 (L1/L2/L3) 及其与 Verifier/Replanner 的联动机制

### 核心算法
- 实现了 9 规则的验证引擎 (~0.1ms) + LLM Hybrid 双模验证
- 实现了 9 种 ErrorType → 4 种 PatchType 的局部重规划映射
- 实现了 AST 4 层启发式自然语言检测 + 5 套代码模板自动生成

### 系统工程
- 实现了 5 后端并发竞速搜索 + 3 端点 Arxiv 并发
- 实现了三态熔断器 + 智能重试 (超时不退避 / 网络错不重试 / 限流退避)
- 实现了 TypedDict 状态管理 + 通用序列化 (删除 110 行手写代码)
- 实现了 L3 Embedding: DashScope API + ThreadPoolExecutor + n-gram 回退

### 全栈开发
- 自包含 Web 应用: aiohttp 后端 11 路由 + SSE 实时推送 + 原生前端
- Token 级流式输出: LLM chat_stream → Writer _write_stream → SSE → 前端 EventSource
- Benchmark 全链路: JSONL 任务加载 → 服务层执行 → 结构化 Evaluator → 报告生成

### 质量保障
- 330 个测试用例，覆盖所有 15 个模块
- 消融实验框架：量化每个模块的独立贡献
- 21 次规范化 Git 提交，清晰的 Day-by-Day 开发记录

---

## 九、产品路线图

### Phase 1: 产品化基础
- SQLite 会话持久化 (替换内存 SessionManager)
- ChromaDB 向量数据库 (替换 FAISS 文件读写)
- 会话历史列表 + 多轮对话
- GitHub Actions CI/CD

### Phase 2: 体验优化
- 用户 API Key 管理页
- 报告导出 PDF
- Markdown 渲染增强 (代码高亮、表格)
- 研究任务模板 (论文综述 / 技术对比 / 新闻摘要)

### Phase 3: 能力扩展
- 工具插件机制 (用户可注册自定义工具)
- 更多数据源 (GitHub、RSS、文档)
- RAG + Agent 混合模式
- 多语言支持

### Phase 4: 部署与发布
- Docker + docker-compose 一键部署
- Nginx + HTTPS 反向代理
- 健康检查 + 监控仪表盘
- GitHub Release v1.0.0

---

## 十、简历要点提炼

### 适合放在简历中的一句话

> 独立设计并实现了一个面向长链路任务的 **多智能体 LLM Agent 系统**，
> 包含分层记忆、验证驱动重规划、异步 DAG 编排三项核心创新，
> 9,000+ 行 Python 代码，330 个测试用例，7 个可运行 Demo，
> 在真实 API Benchmark 上取得 87.5% 通过率和 9.9% Mock 占比。

### 关键技术关键词

`LLM Agent` `Multi-Agent System` `LangGraph` `asyncio` `Hierarchical Memory`
`FAISS` `Vector Embedding` `Verifier-Guided Replanning` `DAG Orchestration`
`SSE Streaming` `Token-level Output` `Tool Use` `Circuit Breaker`
`Search Provenance` `Trajectory Logging` `Pydantic V2` `pytest`

### 适合面试展开的点

1. **分层记忆为什么有效** — 不是简单存更多，而是通过 L2 摘要过滤噪声，
   L3 提供跨 session 的经验迁移
2. **为什么局部重规划优于全局重规划** — 保留已成功步骤的成果，
   只 patch 受影响的子图，避免重复计算和上下文浪费
3. **死锁检测与恢复** — 真实 Bug 案例：ADD 补充任务依赖 FAILED 父任务
   导致环形等待，如何通过依赖检查放宽 (接受 FAILED/SKIPPED 终态) 来解决
4. **搜索竞速设计** — Bocha/Brave/DDGS 并发 FIRST_COMPLETED，
   任何单点故障不阻塞管道，Mock 自动兜底
5. **L3 Embedding 双模回退** — DashScope API 可用时用 1024 维稠密向量，
   不可用时用确定性 MD5 n-gram 哈希，零外部依赖

---

*本文档基于 HorizonRL-Agent v0.3.0 (2026-05-17)，21 commits，330 tests，7 demos*
