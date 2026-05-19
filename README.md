# Horizon-Agent · 溯证智搜

<div align="center">

**多 Agent 协同研究系统 — 提出问题 · 自动检索 · 交叉验证 · 证据溯源**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-414%20passed-brightgreen.svg)](tests/)

</div>

---

## 亮点速览

| 亮点 | 一句话 |
|------|--------|
| 🔬 **多 Agent DAG 并发** | 不是单线执行——多个 Agent 按 DAG 依赖图并发执行子任务，谁先完成谁先验证 |
| 🧠 **三层层次化记忆** | L1 工作窗口 → L2 语义摘要 → L3 向量检索，跨会话复用历史经验 |
| 🔍 **验证器驱动重规划** | 9 条诊断规则自动检测幻觉/事实错误/证据不足，失败时局部修复不重建 |
| 💬 **Research Context Engine** | 多轮追问用语义检索相关历史，不是原始文本拼接 |
| 📝 **全链路证据溯源** | 报告中每个结论可追溯到搜索来源，引用支持率可量化 |
| ⚡ **5 后端并发竞速** | Web 搜索 5 后端 + 论文搜索 5 后端同时并发，最先返回胜出 |
| 🎨 **React 18 SPA** | 零构建三栏布局，全部 184KB 本地化，国内网络零延迟 |
| 🔧 **可视化配置** | 模型/URL/参数/ApiKey Web 界面全可配，修改即保存 |

---

## 这是什么

你提出一个问题。多个 Agent 分工协作——有的搜网页、有的查论文、有的跑代码——结果出来后交叉验证，最后写一份带完整引用来源的结构化报告。

不是聊天。是研究。

**三种使用方式**: CLI 命令行 · Web 界面 · HTTP API · 私有化部署 · 无需 GPU

---

## 快速开始

```bash
git clone https://github.com/TheodoreYang6/HorizonRL-Agent.git
cd HorizonRL-Agent
pip install -e .
```

### 离线模式

零 API 依赖，内置 Mock 数据体验完整流程：

```bash
python examples/04_multi_agent_research.py "Transformer注意力机制的最新进展"
```

### LLM 驱动

```bash
cp .env.example .env    # 填入 DEEPSEEK_API_KEY=sk-xxx
python examples/04_multi_agent_research.py --llm "对比 PyTorch 和 TensorFlow"
```

### Web 界面

```bash
python examples/05_web_agent.py
# http://localhost:8000
```

三种模式：**自动判断** · **即时对话** · **深度研究**（SSE 实时进度 + Token 流式输出）

---

## 架构原理

```
提出问题
    │
    ▼
Plan ──→ LLM 智能拆解为子任务 DAG（5 种任务类型自动识别）
    │       ↑ L3 检索相关历史经验
    ▼
Execute ──→ 多 Agent 并发执行子任务
    │       Web 搜索 (5后端竞速) ‖ 学术论文 (5后端竞速) ‖ 代码执行 ‖ 知识检索
    │       熔断保护 · 自动重试 · 全局速率限制
    ▼
Verify ──→ 交叉验证（9 条诊断规则 ~0.1ms）
    │       空结果 · 工具故障 · 幻觉检测 · 事实错误 · 证据不足
    │       ↓
    ├──→ Replan ──→ 局部修复（仅重试失败节点，不重建计划）
    ▼
Write ──→ 撰写结构化报告（每个结论附带来源引用）
           UserAnswerWriter (用户友好) + DebugReportRenderer (开发者视图)
```

### 记忆系统双链路

```
链路 1: Agent 工作记忆 (L1/L2/L3)        链路 2: 研究上下文引擎
─────────────────────────────        ───────────────────────
消费者: Planner/Worker/Verifier      消费者: 多轮追问
粒度:   子任务执行片段               粒度:   整轮 Q&A 摘要
存储:   ChromaDB episodic_memory     存储:   ChromaDB research_context
作用:   "我刚才做了什么"             作用:   "历史上研究过什么相关的"
```

---

## 核心特性详解

### 1. 多 Agent DAG 并发编排

基于 LangGraph 的 6 节点状态机。不是线性 pipeline——子任务按照 DAG 依赖图自动调度，无依赖的任务通过 `asyncio.gather` 并发执行。

| 机制 | 说明 |
|------|------|
| 条件路由 | `mark_ready` 检测 DAG 依赖满足后放行 |
| 并发控制 | `asyncio.Semaphore` 限制同时执行的 Worker 数 |
| 死锁检测 | 所有未完成节点依赖 FAILED/SKIPPED 任务时自动终止 |
| 批次超时 | `asyncio.wait(tasks, timeout=120)` 防止无限挂起 |
| 会话隔离 | 每次 `run()` 自动 `reset()` Replanner + `clear()` Memory |

### 2. 三层层次化记忆

| 层级 | 容量 | 写入 | 检索 |
|------|------|------|------|
| L1 最近窗口 | 10K tokens (FIFO) | 每个子任务完成后 `mem.record()` | Agent 实时读取最近 5 条 |
| L2 语义摘要 | 50 条 | L1 80% 满载触发压缩 | 关键词匹配 |
| L3 经验归档 | 无限 | Verifier 验证后 `archive_to_l3()` | 向量检索 (ChromaDB/FAISS) |

L3 支持双后端：**ChromaDB**（默认，自动持久化 + 元数据过滤）或 **FAISS**（零依赖回退）。Embedding 优先用 DashScope API，不可用时回退 n-gram MD5 哈希。

### 3. 验证器驱动重规划

9 条诊断规则 (~0.1ms) 自动检测问题，9 种策略定向修复：

| 错误类型 | 策略 | 动作 |
|---------|------|------|
| 空结果 | RETRY | 改写搜索词重新查询 |
| 工具故障 | RETRY | 切换备用工具后端 |
| 证据不足 | ADD | 补充额外检索任务 |
| 幻觉检测 | RETRY | 严格指令约束重新生成 |
| 事实错误 | RETRY | 交叉验证多个来源 |

防无限循环：单任务最多 3 次重试，全局最多 5 次重规划。

### 4. 多端点工具层 — 并发竞速

| 工具 | 后端 | 策略 |
|------|------|------|
| Web 搜索 | Bocha / Brave / DDGS / Wikipedia / Mock | 5 后端并发，FIRST_COMPLETED 竞速 |
| 学术论文 | OpenAlex / Semantic Scholar / Arxiv Pkg / Arxiv API×2 / Mock | 5 后端并发，VPN 开/关自适应 |
| 代码执行 | subprocess 沙箱 | AST 检测 + 5 模板自动生成 |
| 知识检索 | L3 向量搜索 | Embedding API → n-gram 回退 |

每个后端独立 6s 超时，总最坏延迟 ≤8s（串行时 20s）。

### 5. Research Context Engine（研究上下文引擎）

多轮追问不是简单的文本拼接：

```
研究完成 → 提取结构化摘要 + 主题词 → 存入 ChromaDB (research_context)
用户追问 → 语义检索 top-2 相关历史 → 注入 prompt 上下文
```

相比原始对话拼接：上下文更精准、Token 消耗更可控、跨会话知识可复用。

### 6. 证据溯源链

`EvidenceItem → SearchProvenance → ToolCall → SearchResult` 完整溯源链路。Citation Map 自动关联声明与来源，支持计算幻觉率和引用支持率。

### 7. 可视化配置

Web 设置面板 4 个 Tab：推理模型 (provider/model/url/temp) · Agent 参数 (并发数/超时/重试) · 工具 (搜索引擎/L3后端) · API Key (6 提供商)。下拉菜单覆盖 DeepSeek/OpenAI/Anthropic/DashScope 主流模型。修改即保存，自动写入 `.env`。

### 8. React 18 SPA

三栏布局：侧栏（进度时间线 + 历史会话）· 主区（聊天 + Markdown 报告）· 详情（实时统计）。
- React 18 + htm（JSX 替代，1.5KB），零 npm/build
- 全部静态资源本地化（184KB），国内网络零延迟
- 暗色/亮色主题切换 + 本地持久化
- 多轮对话 + 新对话 + 5 种研究模板

### 9. 结构化评测

40 题 × 5 类别，JSONL 任务文件，全链路 Evaluator：通过率 / Mock占比 / 工具调用 / 重规划 / 耗时。

---

## Demo

| # | 文件 | 描述 | API |
|---|------|------|-----|
| 02 | `02_simple_agent.py` | 最小端到端 | 否 |
| 03 | `03_llm_demo.py` | LLM 连接测试 | 是 |
| 04 | `04_multi_agent_research.py` | **旗舰**: 全链路 DAG + 双报告 | 可选 |
| 05 | `05_web_agent.py` | Web 界面: SSE + Token 流式 + 会话历史 | 自动 |
| 06 | `06_ablation_study.py` | 组件消融（6 题 × 5 配置） | 否 |
| 07 | `07_benchmark.py` | 结构化评测（40 题，5 类别） | 可选 |

---

## 项目结构

```
src/horizonrl/
├── schemas/           数据协议 — 16 数据结构，所有模块通信基础
├── config/            配置管理 — Pydantic V2 三级合并 (代码→YAML→.env)
├── agent/             Agent 业务逻辑
│   ├── planner.py     Planner (5任务类型) + LLMPlanner (LLM DAG拆解)
│   ├── worker.py      AgentWorker (多工具并发 + LLM分析回退)
│   ├── verifier.py    RuleEngine (9规则) + Verifier (rule/llm/hybrid)
│   ├── replanner.py   Replanner (9策略局部修复)
│   └── writer.py      UserAnswerWriter + DebugReportRenderer
├── tools/             工具层 — 多后端并发竞速 + 熔断保护
│   ├── manager.py     ToolManager (超时/重试/熔断/统计)
│   ├── web_search.py  5 后端并发竞速
│   ├── paper_search.py  5 后端并发竞速 (国内可用)
│   └── code_execution.py  AST 检测 + 安全沙箱
├── orchestration/     LangGraph 6 节点 DAG 编排
├── memory/            记忆系统 (双链路)
│   ├── hierarchical_memory.py  L1·L2·L3 Agent 工作记忆
│   ├── research_context.py     研究上下文引擎 (语义检索)
│   └── vector_store.py         ChromaDB 封装
├── web/               Web 界面 — React 18 SPA + FastAPI
│   ├── app.py         FastAPI 工厂 (SQLite 懒加载 + 测试注入)
│   ├── session_manager.py  SessionState · 内存/SQLite 双后端
│   ├── routes/        8 个 API 端点
│   ├── templates/     Jinja2 骨架
│   └── static/        React 18 + htm · CSS v6 · vendor (184KB)
├── services/          共享服务层 (CLI/Web/Benchmark 统一入口)
├── llm/               LLM 客户端 (OpenAI 兼容)
└── logging/           轨迹日志 (异步 JSONL)
```

---

## API 参考

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/chat` | 统一入口（chat 对话 / deep 深度研究） |
| GET | `/api/stream/{sid}` | SSE 实时推送（11 事件类型） |
| GET | `/api/report/{sid}` | 会话状态查询 |
| GET | `/api/download/{sid}/{kind}` | 下载报告 (final / debug) |
| GET | `/api/sessions` | 历史会话列表（分页） |
| GET | `/api/sessions/{sid}` | 会话详情 |
| DELETE | `/api/sessions/{sid}` | 删除会话 + 报告 |
| GET | `/api/settings/config` | 读取配置 |
| POST | `/api/settings/config` | 保存配置 |
| GET | `/api/settings/keys` | API Key 状态 |
| POST | `/api/settings/keys` | 保存 API Key |
| DELETE | `/api/settings/keys/{p}` | 删除 API Key |

---

## 配置

```bash
# 最小 — 只需一个 Key
DEEPSEEK_API_KEY=sk-your-key

# 全面 — Web 设置面板可视化配置，修改即保存
# 或手动 .env: 100+ HORIZON_ 变量覆盖
HORIZON_LLM__MODEL=gpt-4o
HORIZON_MEMORY__L3_BACKEND=chromadb
HORIZON_AGENT__WORKER_SEMAPHORE_LIMIT=4
```

三套 YAML：`configs/default.yaml` · `configs/dev.yaml` · `configs/eval.yaml`

详见 [.env.example](.env.example)

---

## 测试

```bash
pytest tests/ -v                          # 全部 414 项
pytest tests/test_memory.py -v            # 记忆层 (76 项)
pytest tests/test_session_manager.py -v   # 会话管理 (31 项)
pytest tests/test_web_api.py -v           # Web API (22 项)
pytest tests/test_config.py -v            # 配置系统 (17 项)
pytest tests/test_dag_workflow.py -v      # DAG 编排 (28 项)
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 编排 | LangGraph StateGraph (6 节点 DAG + 条件路由) |
| 并发 | Python asyncio (gather · Semaphore · Queue · wait_for) |
| LLM | OpenAI 兼容 SDK → DeepSeek / OpenAI / Anthropic / vLLM |
| 向量检索 | ChromaDB (默认) + FAISS (回退) · 1024维 n-gram MD5 |
| 配置 | Pydantic V2 三级合并 (代码→YAML→.env) |
| Web 后端 | FastAPI + SSE + SQLite WAL (12 端点) |
| Web 前端 | React 18 + htm · CSS v6 (零构建, 184KB 本地化) |
| 测试 | pytest + pytest-asyncio (414 项, 15 模块) |
| CI/CD | GitHub Actions (3.10-3.13 + lint + benchmark) |

---

## 路线图

| Phase | 内容 | 进度 |
|-------|------|------|
| Phase 1 | 产品化基础 (SQLite · React · ChromaDB · 配置 · CI/CD) | ✅ 100% |
| Phase 2 | 体验优化 (多轮对话 · 主题 · 模板 · API Key管理) | ✅ 100% |
| Phase 3 | 能力扩展 (工具插件 · 多数据源 · RAG · i18n) | 待开始 |
| Phase 4 | 部署发布 (Docker · HTTPS · 监控 · Release v1.0.0) | 待开始 |

---

## 许可证

MIT License — [LICENSE](LICENSE)

---

<div align="center">

**Horizon-Agent** · 溯证智搜

多 Agent 协同 · 三层记忆 · 研究上下文引擎 · 证据溯源

Qiduo Yang · [GitHub](https://github.com/TheodoreYang6)

</div>
