# HorizonRL-Agent

<div align="center">

**面向 20+ 步复杂任务的稳定长时域 Agentic RL 系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-325%20passed-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/Phase-2%20Complete-orange.svg)]()

</div>

---

## 项目概述

HorizonRL-Agent 是一个多智能体 LLM 编排系统，能够让 AI Agent 可靠地完成 20+ 步的复杂长时域研究任务。
它将用户问题拆解为有向无环图 (DAG) 形式的子任务，通过并发的质量验证执行，
并在失败时通过定向重规划自动恢复——全部由三层层次化记忆系统支撑。

**与标准 LangGraph / AutoGPT 的关键区别：**

| 挑战 | 常规做法 | HorizonRL-Agent |
|------|---------|-----------------|
| 长流程中的上下文污染 | 单一扁平上下文窗口 | L1→L2→L3 层次化记忆，自动压缩 |
| 多步任务中的错误级联 | 失败即重建全部计划 | 9 规则验证器诊断错误，重规划仅修补失败节点 |
| 工具失败阻塞流水线 | 阻塞重试 + 固定退避 | 熔断器 + 快速失败 + Mock 兜底 |
| 工具串行执行效率低 | 一次一个工具 | asyncio.gather 多 Worker 并发执行 |

## 快速开始

### 环境要求

- Python 3.10+
- 无需 GPU（仅 RL 训练阶段需要）

### 安装

```bash
git clone https://github.com/TheodoreYang6/HorizonRL-Agent.git
cd HorizonRL-Agent
pip install -e .
```

### 离线运行（零 API 依赖）

```bash
# 端到端研究流水线（Mock 工具）
python examples/02_simple_agent.py "Transformer注意力机制"

# 完整多智能体 DAG 流水线
python examples/04_multi_agent_research.py "RL在LLM训练中的最新进展"
```

两条命令均会生成带有证据引用的结构化研究报告。

### LLM 驱动运行（推荐）

```bash
cp .env.example .env
# 填入你的 API Key: DEEPSEEK_API_KEY=sk-xxx

python examples/04_multi_agent_research.py --llm "你的研究问题"
```

输出路径：`reports/{session_id}/final_answer.md` + `debug_report.md`

### Web 界面

```bash
python examples/05_web_agent.py
# 浏览器打开 http://localhost:8080
```

三种模式：自动 (Auto) | 即时对话 (Chat) | 深度研究 (Deep，带 SSE 进度推送)

## 系统架构

```
UserTask (自然语言)
    │
    ▼
Planner / LLMPlanner ──→ PlanGraph (DAG 子任务图)
    │
    ▼
ResearchOrchestrator (LangGraph StateGraph, 6 节点状态机)
    │
    │   plan_task → mark_ready → execute_batch → verify_batch
    │                                        │
    │                     ┌──────────────────┼──────────┐
    │                continue             replan   done/deadlock
    │                     │                  │           │
    │                mark_ready          replan      finalize
    │                (下一轮)         (局部修复)     (Writer)
    │                     │                  │           │
    │                     └──────────────────┘           │
    │                                                    ▼
    │                                                   END
    │
    ├──→ Verifier (规则/LLM/混合) → VerificationResult
    │       └──→ Replanner → PlanPatch (重试/追加/移除/重排)
    │
    ├──→ HierarchicalMemory
    │       L1 (FIFO 窗口) → L2 (语义摘要) → L3 (FAISS 检索)
    │
    ├──→ TrajectoryLogger (异步 JSONL, 30 种事件类型)
    │
    └──→ Writer
            ├── UserAnswerWriter → final_answer.md (面向用户)
            └── DebugReportRenderer → debug_report.md (面向开发者)
```

## 核心特性

### 1. 验证器引导的局部重规划

失败时不重建整个计划，而是：
- 对每个任务结果执行 **9 道规则检查**（<0.1ms）
- 将错误分类为 **9 种类型**（空结果、代码错误、偏题、幻觉等）
- 施加**定向补丁**：RETRY（重写查询）、ADD（补充任务）、REMOVE（跳过）、REORDER（重排）

```
ErrorType → PatchType 映射:
  EMPTY_RESULT   → RETRY   |  CODE_ERROR    → RETRY
  TOOL_ERROR     → RETRY   |  OFF_TOPIC     → RETRY
  INCOMPLETE     → ADD     |  HALLUCINATION → RETRY
  FACTUAL_ERROR  → RETRY   |  OTHER         → RETRY
```

### 2. 三层层次化记忆

| 层级 | 类型 | 容量 | 行为 |
|------|------|------|------|
| L1 | 近期工作窗口 | 8K tokens (FIFO) | 80% 满载时自动压缩至 L2 |
| L2 | 语义摘要 | 50 条 | 模板/LLM 压缩, FIFO 淘汰 |
| L3 | 情景档案 | 无限 | FAISS n-gram 向量检索 + 关键词混合召回 |

L3 使用确定性 MD5 n-gram 哈希（零依赖，无需 Embedding API），可按需升级为真实 Embedding。

### 3. 异步多智能体 DAG 编排

- LangGraph StateGraph + 条件路由 + 死锁检测
- 所有任务执行和验证均采用 `asyncio.gather` 并行化
- `asyncio.Semaphore` 控制最大并发 Worker 数
- `asyncio.wait` + 超时（120s）防止批次级卡死
- 熔断器阻止对故障工具的重复调用

### 4. 多端点工具层

每个工具均配备多重回退路径与并发端点竞速：

| 工具 | 策略 |
|------|------|
| Web 搜索 | 5 后端自动回退：Bocha → Brave → DDGS → Wikipedia → Mock。AUTO 模式并发竞速 |
| Arxiv 搜索 | 3 端点并发竞速（首个成功即返回，8s 上限）。全部失败时生成 Mock 论文 |
| 代码执行 | 自动识别自然语言输入并生成代码示例。15s 超时 + 安全全局变量沙箱 |

### 5. 轨迹日志

全流水线 30 种事件类型，异步 JSONL 记录——从 `plan.start` 到 `session.end`。为消融实验和 RL 训练提供数据基础。

### 6. 双模式报告生成

- **UserAnswerWriter**：清爽 Markdown 报告，含来源引用，零调试信息
- **DebugReportRenderer**：完整任务 DAG、验证详情、工具调用追踪、记忆统计

## Demo 一览

| # | 文件 | 描述 | 需要 API |
|---|------|------|----------|
| 02 | `02_simple_agent.py` | 最小端到端流水线 | 否 |
| 03 | `03_llm_demo.py` | LLM 连接测试 + 智能规划 | 是 |
| 04 | `04_multi_agent_research.py` | **旗舰 Demo**：6 阶段流水线，双报告输出 | 可选 |
| 05 | `05_web_agent.py` | Web 对话界面（双路由 + SSE + 下载） | 自动检测 |
| 06 | `06_ablation_study.py` | 消融实验框架（5 配置 + 压力注入） | 否 |
| 07 | `07_benchmark.py` | Benchmark 评测（20 题，5 类别） | 否 |

## 项目结构

```
src/horizonrl/
├── schemas/           数据协议 (4 文件, 16 数据结构, 1061 行)
│   ├── task.py        TaskSpec · PlanGraph · PlanNode · PlanPatch · UserTask
│   ├── result.py      StepResult · VerificationResult · EvidenceItem · ToolCall
│   ├── event.py       TrajectoryEvent · TrajectorySession · EventType (30 种)
│   └── report.py      FinalReport · ReportSection · CitationMap · ReportMetadata
│
├── config/            配置管理 (1 文件, 686 行)
│   └── settings.py    Pydantic V2: LLMConfig · MemoryConfig · AgentRuntimeConfig
│
├── tools/             工具层 (5 文件, ~950 行)
│   ├── manager.py     ToolManager: 超时/重试/熔断/统计
│   ├── web_search.py  多后端并发竞速 (Bocha/Brave/DDGS/Wikipedia/Mock)
│   ├── arxiv_search.py  多端点并发竞速 + Mock 回退
│   ├── code_execution.py  沙箱执行 + 自然语言自动生成代码
│   └── mock.py        离线/CI 用 Mock 工具
│
├── llm/               LLM 客户端 (1 文件, 185 行)
│   └── client.py      chat() + embed() · OpenAI 兼容 · DeepSeek 已验证
│
├── agent/             Agent 逻辑 (5 文件, ~2000 行)
│   ├── planner.py     Planner (2 类模板) + LLMPlanner (LLM 驱动 DAG 拆解)
│   ├── worker.py      AgentWorker (异步执行 + 证据提取)
│   ├── verifier.py    RuleEngine (9 规则) + Verifier (规则/LLM/混合)
│   ├── replanner.py   Replanner (9 策略) + LLMReplanner
│   └── writer.py      UserAnswerWriter + DebugReportRenderer (双模式输出)
│
├── orchestration/     编排层 (1 文件, ~850 行)
│   └── dag_workflow.py  ResearchOrchestrator: LangGraph 6 节点状态机
│
├── memory/            记忆系统 (1 文件, ~750 行)
│   └── hierarchical_memory.py  L1RecentWindow · L2SemanticSummary · L3EpisodicArchive
│
└── logging/           日志系统 (1 文件, 411 行)
    └── trajectory_logger.py  异步 JSONL · TrajectorySession · 5 种分析工具
```

## 运行测试

```bash
pytest tests/ -v                          # 全部 325 项测试
pytest tests/test_dag_workflow.py -v      # 编排层 (28 项)
pytest tests/test_memory.py -v            # 记忆层 (63 项)
pytest tests/test_replanner.py -v         # 重规划策略 (51 项)
```

## 配置说明

配置采用**三级合并**机制：代码默认值 → YAML 文件 → 环境变量。

```bash
# .env 文件
DEEPSEEK_API_KEY=sk-your-key          # LLM 推理
DASHSCOPE_API_KEY=sk-your-key         # Embedding (可选)
BOCHA_API_KEY=sk-your-key             # Web 搜索 (可选, 国内友好)

# 环境变量覆盖 (双下划线 = 嵌套层级)
HORIZON_LLM__MODEL=deepseek-chat
HORIZON_AGENT__MAX_STEPS=20
HORIZON_MEMORY__L1_MAX_TOKENS=6000
HORIZON_SEARCH_PROVIDER=auto
```

配置文件：`configs/default.yaml` (生产) | `configs/dev.yaml` (开发) | `configs/eval.yaml` (评测)

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| 编排 | LangGraph StateGraph (6 节点 DAG, 条件路由) |
| 异步 | Python asyncio (gather, Semaphore, Queue, wait, wait_for) |
| LLM | OpenAI 兼容 SDK (DeepSeek, OpenAI, vLLM 等任意兼容 API) |
| 向量化 | MD5 n-gram 特征哈希 (零依赖, 确定性) — 可升级至 Embedding API |
| 向量检索 | FAISS IndexFlatL2 + L2 距离阈值 + 关键词重排序 |
| 配置 | Pydantic V2 三级合并 (代码 → YAML → 环境变量) |
| 日志 | 异步 JSONL + 后台写入任务 |
| 测试 | pytest + pytest-asyncio (325 项, 10 个测试模块) |

## 研究方向

本项目聚焦**长时域 Agent 稳定性 (Long-Horizon Agent Stability)** 问题。核心研究贡献：

1. **层次化记忆**：L1 (工作记忆) → L2 (语义记忆) → L3 (情景记忆)，确定性 n-gram 检索
2. **验证器引导重规划**：9 条诊断规则 → 定向局部补丁（非全量重建）
3. **轨迹级日志**：30 种事件类型，支撑消融实验与 RL 训练
4. **证据溯源链**：最终报告中的每个结论均可追溯至其搜索来源

目标期刊/会议：AAAI / IJCAI / ACL Findings 2027

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

*HorizonRL-Agent v0.2.0 — 325 项测试 · 7 个 Demo · 16K 行代码 · Phase 2 完成*
