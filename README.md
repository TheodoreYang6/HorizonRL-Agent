# HorizonRL-Agent

<div align="center">

**长链路 Agentic RL 系统 — 让 LLM Agent 稳定完成 20+ 步复杂任务**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-296%20passed-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/Status-Public%20Beta-orange.svg)]()

</div>

---

## 一句话介绍

输入一个研究问题，AI Agent 自动**分解任务 → 并行搜索 → 质量验证 → 失败修复 → 生成报告**。全程异步并发，离线可用，接入 LLM 后效果更佳。

## 核心创新

1. **分层记忆 (L1→L2→L3)**：L1 工作窗口 → L2 语义摘要 → L3 经验归档，解决上下文污染
2. **Verifier 驱动重规划**：不只判断 pass/fail，还给出错误类型、证据缺口和恢复建议；Replanner 只做局部 patch 而非全局重建
3. **异步多 Agent DAG 编排**：LangGraph StateGraph + asyncio.Semaphore，死锁检测，迭代上限
4. **轨迹日志一等公民**：30 种事件类型，JSONL 异步写入，为消融实验和 RL 训练提供数据基础

## 架构

```
UserTask (自然语言)
    │
    ▼
LLMPlanner / Planner ──→ PlanGraph (DAG 任务图)
    │
    ▼
AgentWorker × N (asyncio 并发 + Semaphore)
    │  ToolManager → 熔断 → 超时 → 重试 → 工具
    ▼
StepResult + Evidence[]
    │
    ├──→ Verifier (9 规则 / LLM 深度诊断)
    │      ├── 通过 → 记录到 Memory
    │      └── 失败 → Replanner → 局部修复 → 重试
    │
    ├──→ HierarchicalMemory (L1 → 自动压缩 → L2)
    ├──→ TrajectoryLogger (异步 JSONL, 全程)
    └──→ Writer → 自然语言研究报告
```

## 快速开始

### 环境要求

- Python 3.10+
- Windows / Linux / macOS

### 安装

```bash
git clone https://github.com/YOUR_USERNAME/HorizonRL-Agent.git
cd HorizonRL-Agent
pip install -r requirements.txt
```

### 3 秒跑通（离线模式，无需 API）

```bash
python examples/04_multi_agent_research.py "Transformer 注意力机制"
```

### LLM 驱动模式（推荐，报告质量更高）

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 API Key（支持 DeepSeek / OpenAI / 任何兼容 API）

# 2. 编辑 configs/dev.yaml 修改模型和端点（默认 DeepSeek）

# 3. 运行
python examples/04_multi_agent_research.py --llm "Transformer 注意力机制最新进展"
```

### Web 界面

```bash
python examples/05_web_agent.py
# 浏览器打开 http://localhost:8080
```

## 所有 Demo

| 文件 | 说明 | 需要 API |
|------|------|---------|
| `01_async_demo.py` | asyncio 完整教程 (10 示例) | 否 |
| `02_simple_agent.py` | 最简端到端管道 | 否 |
| `03_llm_demo.py` | LLM 连接测试 + 智能规划 | 是 |
| `04_multi_agent_research.py` | **v1 旗舰 Demo** (6-Stage Pipeline) | 可选 |
| `05_web_agent.py` | Web 交互界面 | 否 |

## 运行测试

```bash
python -m pytest tests/ -v        # 全部 (296 tests)
python -m pytest tests/test_verifier.py -v  # 单模块
```

## 项目结构

```
src/horizonrl/
├── schemas/        ← 数据协议 (4 文件, 16 数据结构)
├── config/         ← Pydantic V2 三级配置
├── tools/          ← 工具层 (Web/Arxiv/Code + 熔断/重试/超时)
├── llm/            ← LLM 客户端 (OpenAI 兼容)
├── agent/          ← Agent 业务逻辑
│   ├── planner.py      Planner + LLMPlanner
│   ├── worker.py       AgentWorker + 并发调度
│   ├── verifier.py     Verifier (9 规则 + LLM Hybrid)
│   ├── replanner.py    Replanner (9 种修复策略)
│   └── writer.py       自然语言报告合成
├── orchestration/  ← LangGraph DAG 编排
├── memory/         ← 分层记忆 (L1/L2/L3)
└── logging/        ← 异步 JSONL 轨迹日志
```

## 技术栈

- **Agent 框架**: LangGraph (StateGraph, conditional routing)
- **异步**: Python asyncio (gather, Semaphore, Queue)
- **LLM**: OpenAI SDK → DeepSeek / OpenAI / 任何兼容 API
- **配置**: Pydantic V2 三级合并 (默认 → YAML → .env)
- **测试**: pytest + pytest-asyncio (296 tests)

## 许可证

MIT License — 详见 [LICENSE](LICENSE)

## 作者

**杨启铎** — NWPU 硕士

研究方向：LLM Agent 长链路稳定执行、分层记忆、Agentic RL

---

*HorizonRL-Agent v0.1.0 — Public Beta*
