# HorizonRL-Agent

<div align="center">

**面向 20+ 步复杂任务的稳定长时域 Agentic RL 系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-330%20passed-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/Phase-2%20Complete-orange.svg)]()

</div>

---

## 项目概述

HorizonRL-Agent 是一个多智能体 LLM 编排系统，能够让 AI Agent 可靠地完成 20+ 步的复杂长时域研究任务。
它将用户问题拆解为有向无环图 (DAG) 形式的子任务，通过并发的质量验证执行，
并在失败时通过定向重规划自动恢复——全部由三层层次化记忆系统支撑。

**核心创新**: 层次化记忆 (L1→L2→L3) · 验证器引导重规划 · 异步多 Agent DAG 编排

## 快速开始

```bash
git clone https://github.com/TheodoreYang6/HorizonRL-Agent.git
cd HorizonRL-Agent
pip install -e .
```

### 离线运行（零 API 依赖）

```bash
python examples/04_multi_agent_research.py "Transformer注意力机制"
```

### LLM 驱动（推荐）

```bash
cp .env.example .env        # 填入 DEEPSEEK_API_KEY=sk-xxx
python examples/04_multi_agent_research.py --llm "你的研究问题"
```

输出: `reports/{session_id}/final_answer.md` + `debug_report.md`

### Web 界面

```bash
python examples/05_web_agent.py    # http://localhost:8080
```

三种模式: 自动判断 (Auto) | 即时对话 (Chat) | 深度研究 (Deep, SSE 实时进度 + Token 流式)

### Benchmark 评测

```bash
python examples/07_benchmark.py                   # Mock 模式, 20题5类
python examples/07_benchmark.py --llm             # LLM 模式
python examples/07_benchmark.py --category 技术对比  # 单类别
```

## 系统架构

```
UserTask (自然语言)
    │
    ▼
Planner / LLMPlanner ──→ PlanGraph (5种任务类型自动分类, DAG依赖)
    │
    ▼
ResearchOrchestrator (LangGraph 6节点状态机)
    │  plan_task → mark_ready → execute_batch → verify_batch → replan/finalize
    │  死锁检测 · 迭代上限 · Semaphore并发 · 批次超时 · 会话隔离
    ▼
AgentWorker × N (asyncio 并发)
    │  ToolManager → 熔断 → 超时/重试 → 工具 (Web/Arxiv/Code)
    ▼
StepResult + EvidenceItem[] (SearchProvenance 可追溯)
    │
    ├──→ Verifier (规则/LLM/混合) → 9道规则 · 9种ErrorType
    │       └──→ Replanner → PlanPatch (RETRY/ADD/REMOVE/REORDER) → 回写DAG
    │
    ├──→ HierarchicalMemory (L1 FIFO → L2 语义摘要 → L3 FAISS向量检索)
    │
    ├──→ TrajectoryLogger (异步JSONL, 30种事件, per-node追踪)
    │
    └──→ Writer v2
            ├── UserAnswerWriter → final_answer.md (面向用户, Token流式)
            └── DebugReportRenderer → debug_report.md (面向开发者)
```

## 核心特性

### 1. 验证器引导的局部重规划

失败时不重建整个计划——仅修复失败节点:

```
ErrorType → PatchType:
  EMPTY_RESULT   → RETRY (改写查询)    CODE_ERROR    → RETRY (修正代码)
  TOOL_ERROR     → RETRY (切换工具)    OFF_TOPIC     → RETRY (重写描述)
  INCOMPLETE     → ADD   (补充任务)    HALLUCINATION → RETRY (严格指令)
  FACTUAL_ERROR  → RETRY (交叉验证)    OTHER         → RETRY
```

防无限循环: 单任务最多3次重试, 全局最多5次重规划。

### 2. 三层层次化记忆

| 层级 | 类型 | 容量 | 行为 |
|------|------|------|------|
| L1 | 近期工作窗口 | 8K tokens (FIFO) | 80% 满载时自动压缩至 L2 |
| L2 | 语义摘要 | 50 条 | 模板/LLM 压缩, FIFO 淘汰 |
| L3 | 情景档案 | 无限 | FAISS n-gram 向量检索 + 混合召回, 持久化 |

L3 使用确定性 MD5 n-gram 哈希（零依赖），可按需升级真实 Embedding API。

### 3. 异步多 Agent DAG 编排

- LangGraph StateGraph + 条件路由 + 死锁检测
- 全任务 `asyncio.gather` 并行执行和验证
- `asyncio.Semaphore` 并发控制 · `asyncio.wait` 批次超时
- 熔断器阻止对故障工具的重复调用
- 每次运行会话隔离 (Replanner/Memory 状态重置)

### 4. 多端点工具层

| 工具 | 策略 |
|------|------|
| Web 搜索 | 5 后端: Bocha → Brave → DDGS → Wikipedia → Mock。AUTO 模式并发竞速 |
| Arxiv 搜索 | 5 端点并发竞速: arxiv.loli.net → cn.arxiv.org → xxx.itp.ac.cn → 官方端点。全失败时生成 Mock 论文 |
| 代码执行 | AST 启发式检测自然语言输入, 5 套代码模板自动生成, 安全沙箱 |

### 5. Token 级流式输出

LLM 写作阶段逐字推送至 Web 前端, 配合 SSE 实时进度和阶段时间线。

### 6. 结构化 Benchmark

20 题 5 类别, JSONL 任务文件, 全链路 Evaluator (通过率/子任务SR/Mock占比/工具调用/重规划/耗时)。

## Demo 一览

| # | 文件 | 描述 | 需要 API |
|---|------|------|----------|
| 02 | `02_simple_agent.py` | 最小端到端流水线 | 否 |
| 03 | `03_llm_demo.py` | LLM 连接测试 + 智能规划 | 是 |
| 04 | `04_multi_agent_research.py` | **旗舰 Demo**: 全链路 DAG, 双报告输出 | 可选 |
| 05 | `05_web_agent.py` | Web 界面: SSE + Token流式 + 工具面板 + 报告下载 | 自动 |
| 06 | `06_ablation_study.py` | 消融实验框架 (5 配置 + 压力注入) | 否 |
| 07 | `07_benchmark.py` | Benchmark (20 题, 5 类别, JSONL + Evaluator) | 可选 |

## 项目结构

```
src/horizonrl/
├── schemas/           数据协议 (4 文件, 16 数据结构)
│   ├── task.py        TaskSpec · PlanGraph · PlanNode · PlanPatch · UserTask
│   ├── result.py      StepResult · VerificationResult · EvidenceItem · SearchProvenance
│   ├── event.py       TrajectoryEvent · TrajectorySession · EventType (30 种)
│   └── report.py      FinalReport · ReportSection · CitationMap · ReportMetadata
│
├── config/            配置管理
│   └── settings.py    Pydantic V2 三级合并 (代码 → YAML → 环境变量)
│
├── services/          共享服务层 (CLI/Web/Benchmark 统一入口)
│   └── research_service.py  SessionArtifacts · run/stream · 模式判断
│
├── tools/             工具层 (5 文件)
│   ├── manager.py     ToolManager: 超时/重试/熔断/统计/结果规范化
│   ├── web_search.py  5 后端并发竞速 (Bocha/Brave/DDGS/Wikipedia/Mock)
│   ├── arxiv_search.py  5 端点并发竞速 + Mock 回退
│   ├── code_execution.py  AST 检测 + 模板自动生成 + 安全沙箱
│   └── mock.py        离线/CI 用 Mock 工具
│
├── llm/               LLM 客户端
│   └── client.py      chat() + chat_stream() + embed() · OpenAI 兼容
│
├── agent/             Agent 逻辑 (5 文件)
│   ├── planner.py     Planner (5 种任务类型) + LLMPlanner (LLM DAG 拆解)
│   ├── worker.py      AgentWorker (多工具并发 + 证据提取 + 搜索词清洗)
│   ├── verifier.py    RuleEngine (9 规则) + Verifier (规则/LLM/混合)
│   ├── replanner.py   Replanner (9 策略) + LLMReplanner + 状态重置
│   └── writer.py      UserAnswerWriter + DebugReportRenderer (双模式 + Token流式)
│
├── orchestration/     编排层
│   └── dag_workflow.py  ResearchOrchestrator: LangGraph 6 节点, 全模块集成
│
├── memory/            记忆系统
│   └── hierarchical_memory.py  L1RecentWindow · L2SemanticSummary · L3EpisodicArchive
│
└── logging/           日志系统
    └── trajectory_logger.py  异步 JSONL · per-node 事件 · 5 种分析工具
```

## 运行测试

```bash
pytest tests/ -v                          # 全部 330 项测试
pytest tests/test_dag_workflow.py -v      # 编排层 (28 项)
pytest tests/test_memory.py -v            # 记忆层 (63 项)
pytest tests/test_replanner.py -v         # 重规划策略 (51 项)
pytest tests/test_planner.py -v           # 5种任务类型 (14 项)
```

## 配置说明

三级合并: 代码默认值 → YAML → 环境变量。

```bash
# .env 文件
DEEPSEEK_API_KEY=sk-your-key          # LLM 推理
BOCHA_API_KEY=sk-your-key             # Web 搜索 (国内推荐)
BRAVE_API_KEY=sk-your-key             # Web 搜索 (国际)
HORIZON_SEARCH_PROVIDER=auto          # auto / bocha / brave / mock

# 嵌套覆盖 (双下划线)
HORIZON_LLM__MODEL=deepseek-chat
HORIZON_AGENT__MAX_STEPS=20
HORIZON_MEMORY__L1_MAX_TOKENS=6000
```

配置文件: `configs/default.yaml` (生产) | `configs/dev.yaml` (开发) | `configs/eval.yaml` (评测)

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| 编排 | LangGraph StateGraph (6 节点 DAG, 条件路由) |
| 异步 | Python asyncio (gather, Semaphore, Queue, wait, wait_for) |
| LLM | OpenAI 兼容 SDK → DeepSeek / OpenAI / vLLM |
| 向量检索 | FAISS IndexFlatL2 + MD5 n-gram 特征哈希 (零依赖) |
| 配置 | Pydantic V2 三级合并 |
| 日志 | 异步 JSONL + 后台写入 + per-node 事件追踪 |
| Web | aiohttp + SSE + Token 流式 |
| 测试 | pytest + pytest-asyncio (330 项, 11 个测试模块) |

## 研究方向

本项目聚焦 **长时域 Agent 稳定性 (Long-Horizon Agent Stability)**:

1. **层次化记忆**: L1 (工作) → L2 (语义) → L3 (情景), 确定性 n-gram 检索
2. **验证器引导重规划**: 9 条诊断规则 → 定向局部补丁 (非全量重建)
3. **轨迹级日志**: 30 种事件类型, per-node 追踪, 支撑消融实验与 RL 训练
4. **证据溯源链**: SearchProvenance, 报告中的每个结论可追溯至搜索来源
5. **5 种任务自动分类**: research / code / comparison / summary / factual_qa

目标: AAAI / IJCAI / ACL Findings 2027

## 许可证

MIT License — 详见 [LICENSE](LICENSE)

## 引用

```bibtex
@misc{horizonrl-agent,
  author = {Qiduo Yang},
  title = {HorizonRL-Agent: Long-Horizon Agentic RL with Hierarchical Memory and Verifier-Guided Replanning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/TheodoreYang6/HorizonRL-Agent}
}
```

---

*HorizonRL-Agent v0.3.0 — 330 项测试 · 7 个 Demo · 共享 Service 层 · SSE + Token 流式 · 17 commits*
