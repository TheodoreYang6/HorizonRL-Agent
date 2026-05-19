# Horizon-Agent · 溯证智搜

<div align="center">

**多 Agent 协同研究系统 — 提出问题 · 自动检索 · 交叉验证 · 撰写报告**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-414%20passed-brightgreen.svg)](tests/)

</div>

---

## 这是什么

Horizon-Agent 是一个多 Agent 协同研究系统。你提出一个问题，多个 Agent 并发搜索网络和学术论文、交叉验证信息来源、撰写带有完整证据溯源的结构化报告。

与普通 AI 聊天不同——Horizon-Agent 不会凭空编造答案。它像一支严谨的研究团队：**多个 Agent 分工协作，先查资料，再交叉验证，最后写成报告**。报告中的每个结论都可以追溯到原始来源。

**三种使用方式**: CLI 命令行 · Web 界面 · HTTP API，均可私有化部署，无需 GPU。

---

## 快速开始

```bash
git clone https://github.com/TheodoreYang6/HorizonRL-Agent.git
cd HorizonRL-Agent
pip install -e .
```

### 离线模式（零 API 依赖）

无需任何 API Key，使用内置 Mock 数据体验完整流程：

```bash
python examples/04_multi_agent_research.py "Transformer注意力机制的最新进展"
```

输出：`reports/<session_id>/final_answer.md` + `debug_report.md`

### LLM 驱动（推荐）

配置 DeepSeek API Key 以获得最佳体验：

```bash
cp .env.example .env          # 编辑填入 DEEPSEEK_API_KEY=sk-xxx
python examples/04_multi_agent_research.py --llm "对比 PyTorch 和 TensorFlow"
```

### Web 界面

```bash
python examples/05_web_agent.py
# 打开 http://localhost:8000
```

三种模式：**自动判断** · **即时对话** · **深度研究**（SSE 实时进度 + Token 流式输出）

---

## 它是如何工作的

```
提出问题
    │
    ▼
Plan ──→ 智能拆解为子任务 DAG（5 种任务类型自动识别）
    │
    ▼
Execute ──→ 多 Agent 并发执行（Web 搜索 · 学术论文 · 代码执行 · 知识检索）
    │         工具层自动竞速选择最优后端，熔断/重试/超时保护
    ▼
Verify ──→ 交叉验证结果（9 条诊断规则，自动检测证据不足、幻觉、事实错误）
    │
    ▼
Replan ──→ 局部修复（仅重试失败节点，不重建整个计划）
    │
    ▼
Write ──→ 撰写结构化报告（每个结论附带来源引用，Markdown 格式）
```

---

## 核心特性

### 1. 验证器引导的局部重规划

失败时不重建整个计划——只修复出问题的节点：

| 错误类型 | 策略 | 动作 |
|---------|------|------|
| 空结果 | RETRY | 改写搜索词重新查询 |
| 工具故障 | RETRY | 切换到备用工具后端 |
| 证据不足 | ADD | 补充额外检索任务 |
| 幻觉检测 | RETRY | 严格指令约束重新生成 |
| 事实错误 | RETRY | 交叉验证多个来源 |
| 离线分析 | ADD | 补充缺失的分析步骤 |

防无限循环：单任务最多 3 次重试，全局最多 5 次重规划。

### 2. 三层层次化记忆

| 层级 | 类型 | 容量 | 行为 |
|------|------|------|------|
| L1 | 最近工作窗口 | 10,000 tokens | FIFO，80% 满载时自动压缩到 L2 |
| L2 | 语义摘要 | 50 条 | LLM 或模板压缩，FIFO 淘汰 |
| L3 | 情景档案 | 无限 | 向量检索，跨会话复用 |

L3 支持双后端：
- **FAISS**（默认）：n-gram 特征哈希，零额外依赖
- **ChromaDB**（生产推荐）：自动持久化，元数据过滤，余弦相似度检索

### 3. 异步多 Agent DAG 编排

- 基于 LangGraph 的 6 节点状态机 + 条件路由 + 死锁检测
- Python asyncio 并发执行和验证，Semaphore 控制并发度
- 熔断器保护：5 次失败触发，15 秒冷却，阻止重复调用故障工具
- 每次运行会话隔离，状态自动重置

### 4. 多端点工具层

| 工具 | 策略 |
|------|------|
| Web 搜索 | 5 后端并发竞速：Bocha → Brave → DDGS → Wikipedia → Mock。AUTO 模式并发取最快 |
| 学术论文 | 5 后端并发竞速：OpenAlex ‖ Semantic Scholar ‖ Arxiv Pkg ‖ Arxiv API×2 → Mock。VPN 开/关自适应 |
| 代码执行 | AST 启发式检测自然语言输入，5 套代码模板自动生成，安全沙箱执行 |
| 知识检索 | L3 向量搜索：Embedding API 优先，n-gram MD5 哈希回退 |

### 5. Research Context Engine（研究上下文引擎）

多轮追问不再是简单的文本拼接。每次研究完成后，系统自动生成结构化摘要存入 ChromaDB，追问时通过语义检索找到最相关的历史研究作为上下文注入。

```
研究完成 → LLM 提取摘要 + 主题词 → ChromaDB
用户追问 → 语义检索 top-2 相关历史 → 注入 prompt
```

相比原始文本拼接：上下文更精准、Token 消耗更可控、跨会话知识可复用。

### 6. 证据溯源链

报告中每个结论都通过 `SearchProvenance` 追溯到原始搜索来源。支持从 `EvidenceItem` → `SearchProvenance` → `ToolCall` → `SearchResult` 的完整溯源链路。

### 7. React 前端（零构建）

三栏布局：侧栏（进度时间线 + 历史会话）· 主区（聊天 + Markdown 报告）· 详情（实时统计）。

- React 18 + htm（JSX 替代，1.5KB），零 npm/build 步骤
- 全部静态资源本地化（React + ReactDOM + htm + marked = 184KB）
- SSE 实时推送：11 种事件类型，Token 级流式输出
- 会话历史管理：SQLite 持久化，分页查询，删除确认
- 诊断系统：6 步加载追踪，错误边界，骨架屏加载态

### 8. 结构化评测框架

40 题 × 5 类别（研究/代码/对比/摘要/问答），JSONL 任务文件，全链路 Evaluator：

| 指标 | 说明 |
|------|------|
| 通过率 | 子任务成功完成比例 |
| Mock 占比 | 真实搜索 vs 模拟数据的比例 |
| 工具调用 | 有效工具调用次数 |
| 重规划 | 失败后重试次数 |
| 总耗时 | 端到端执行时间 |

---

## 项目结构

```
src/horizonrl/
├── schemas/           数据协议层 — 16 个数据结构，所有模块通信基础
│   ├── task.py        TaskSpec · PlanGraph · PlanNode · PlanPatch
│   ├── result.py      StepResult · VerificationResult · EvidenceItem · SearchProvenance
│   ├── event.py       TrajectoryEvent · TrajectorySession（30 种事件类型）
│   └── report.py      FinalReport · ReportSection · CitationMap · ReportMetadata
│
├── config/            配置管理 — Pydantic V2 三级合并（代码 → YAML → .env）
│   └── settings.py    10 个配置模型，100+ HORIZON_ 环境变量覆盖
│
├── agent/             Agent 业务逻辑
│   ├── planner.py     Planner（5 种任务类型模板）+ LLMPlanner（LLM DAG 拆解）
│   ├── worker.py      AgentWorker（多工具并发 + 证据提取 + LLM 分析回退）
│   ├── verifier.py    RuleEngine（9 规则 ~0.1ms）+ Verifier（rule/llm/hybrid 三模式）
│   ├── replanner.py   Replanner（9 策略局部修复 + 状态重置 + 防无限循环）
│   └── writer.py      UserAnswerWriter + DebugReportRenderer（LLM 流式 + 证据清洗）
│
├── tools/             工具层 — 多后端并发竞速 + 熔断保护
│   ├── manager.py     ToolManager（超时/重试/熔断/统计/结果规范化）
│   ├── web_search.py  5 后端并发竞速（Bocha/Brave/DDGS/Wikipedia/Mock）
│   ├── paper_search.py  5 后端并发竞速（OpenAlex/S2/Arxiv Pkg/Arxiv API×2/Mock）
│   ├── code_execution.py  AST 检测 + 5 模板 + 安全沙箱
│   └── mock.py        离线/CI Mock 工具
│
├── orchestration/     编排层
│   └── dag_workflow.py  ResearchOrchestrator（LangGraph 6 节点 DAG，全模块注入）
│
├── memory/            记忆系统（双链路）
│   ├── hierarchical_memory.py  L1·L2·L3 Agent 工作记忆
│   ├── research_context.py     研究上下文引擎（语义检索）
│   └── vector_store.py         ChromaDB 封装
│
├── web/               Web 界面 — React 18 SPA + FastAPI
│   ├── app.py         FastAPI 工厂（lifespan · CORS · 懒加载 SQLite · 测试注入）
│   ├── session_manager.py  SessionState · 内存/SQLite 双后端（缓存机制）
│   ├── models.py      5 个 Pydantic 请求/响应模型
│   ├── routes/        7 个 API 端点（chat/stream/report/download/sessions）
│   ├── templates/     Jinja2 骨架（14 行）
│   └── static/        React 18 + htm · CSS v6.0 · vendor（184KB）
│
├── services/          共享服务层 — CLI/Web/Benchmark 统一入口
│   └── research_service.py  run()/stream()/模式判断/L3 注入
│
├── llm/               LLM 调用层
│   └── client.py      chat() + chat_stream() + embed() · OpenAI 兼容协议
│
└── logging/           轨迹日志
    └── trajectory_logger.py  异步 JSONL · 25+ per-node 事件 · 5 种分析工具
```

---

## API 参考

Web 界面提供 7 个 REST API 端点：

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/chat` | 统一入口（chat 对话 / deep 深度研究） |
| GET | `/api/stream/{sid}` | SSE 实时推送（stage/tool/verify/token/done） |
| GET | `/api/report/{sid}` | 会话状态查询（页面刷新恢复） |
| GET | `/api/download/{sid}/{kind}` | 下载报告（final_answer.md / debug_report.md） |
| GET | `/api/sessions` | 历史会话列表（分页查询） |
| GET | `/api/sessions/{sid}` | 单个会话详情 |
| DELETE | `/api/sessions/{sid}` | 删除会话及关联报告文件 |

访问 `http://localhost:8000/docs` 查看 Swagger UI 交互式文档。

---

## 配置

三级配置合并：代码默认值 → YAML 文件 → `.env` 环境变量。

```bash
# 最小配置 — 只需一个 API Key
DEEPSEEK_API_KEY=sk-your-key

# 全面控制 — 100+ HORIZON_ 变量可用
HORIZON_LLM__MODEL=gpt-4o
HORIZON_MEMORY__L3_BACKEND=chromadb
HORIZON_AGENT__MAX_STEPS=20
HORIZON_VERIFIER__STRICT_MODE=true
SESSION_BACKEND=sqlite
```

### 可用 API

| API | 环境变量 | 用途 |
|-----|---------|------|
| DeepSeek | `DEEPSEEK_API_KEY` | LLM 推理（推荐，国内可用） |
| OpenAI | `OPENAI_API_KEY` | LLM 推理 |
| Anthropic | `ANTHROPIC_API_KEY` | LLM 推理 |
| DashScope | `DASHSCOPE_API_KEY` | L3 向量嵌入（1024 维） |
| Bocha | `BOCHA_API_KEY` | Web 搜索（国内推荐） |
| Brave | `BRAVE_API_KEY` | Web 搜索（国际备选） |

三套 YAML 配置：`configs/default.yaml`（生产）· `configs/dev.yaml`（开发调试）· `configs/eval.yaml`（评测基准）

全部环境变量详见 [.env.example](.env.example)。

---

## Demo

| # | 文件 | 描述 | 需要 API |
|---|------|------|----------|
| 02 | `02_simple_agent.py` | 最小端到端流水线，了解核心流程 | 否 |
| 03 | `03_llm_demo.py` | LLM 连接测试 + 智能规划 | 是 |
| 04 | `04_multi_agent_research.py` | **旗舰 Demo**：全链路 DAG + 双报告输出 | 可选 |
| 05 | `05_web_agent.py` | Web 界面：SSE 实时推送 + Token 流式 + 会话历史 | 自动 |
| 06 | `06_ablation_study.py` | 组件消融分析（6 题 × 5 配置） | 否 |
| 07 | `07_benchmark.py` | 结构化评测（40 题，5 类别） | 可选 |

---

## 测试

```bash
# 全部测试（414 项，14 个测试模块）
pytest tests/ -v

# 按模块
pytest tests/test_memory.py -v            # 记忆层（76 项，含 FAISS + ChromaDB）
pytest tests/test_session_manager.py -v   # 会话管理（31 项，双后端）
pytest tests/test_web_api.py -v           # Web API（22 项，7 端点）
pytest tests/test_config.py -v            # 配置系统（17 项，三级合并）
pytest tests/test_dag_workflow.py -v      # DAG 编排（28 项，E2E + 死锁）
pytest tests/test_replanner.py -v         # 重规划策略（51 项）
pytest tests/test_verifier.py -v          # 验证器（32 项，9 规则）
pytest tests/test_writer.py -v            # 报告生成（31 项）
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 编排引擎 | LangGraph StateGraph（6 节点 DAG + 条件路由） |
| 并发模型 | Python asyncio（gather · Semaphore · Queue · wait_for） |
| LLM 接入 | OpenAI 兼容 SDK → DeepSeek / OpenAI / Anthropic / vLLM |
| 向量检索 | FAISS IndexFlatL2 + ChromaDB（自动持久化 + 元数据过滤） |
| Embedding | DashScope text-embedding-v4（1024 维）+ n-gram MD5 回退 |
| 配置系统 | Pydantic V2（三级合并：代码默认 → YAML → .env） |
| Web 后端 | FastAPI + SSE + SQLite（7 个 REST 端点，WAL 模式） |
| Web 前端 | React 18 + htm（零构建，184KB 全部本地化） |
| 日志系统 | 异步 JSONL + 后台 Writer（25+ per-node 事件类型） |
| 测试框架 | pytest + pytest-asyncio（414 项，14 个测试模块） |

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)

---

<div align="center">

**Horizon-Agent** · 溯证智搜

多 Agent 协同 · 三层记忆 · 研究上下文引擎 · 证据溯源 · 私有化部署

Qiduo Yang · [GitHub](https://github.com/TheodoreYang6)

</div>
