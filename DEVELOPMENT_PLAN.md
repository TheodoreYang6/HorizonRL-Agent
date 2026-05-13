# HorizonRL-Agent 最终开发路线图 v5.0

> **更新日期**: 2026-05-13 (Day 2 收尾)
> **状态**: Phase 1 全部完成 (Steps 0-14)，已推送到 GitHub
> **测试**: 296 passed, 4 skipped, 0 failed
> **代码量**: ~13,900 行 (源码 6100 + 测试 3700 + Demo 2800 + 配置/文档 1300)

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

## 核心创新点

1. **分层记忆结构**：L1 最近工作记忆 → L2 语义摘要 → L3 经验归档
2. **Verifier 驱动的重规划**：Verifier 给出错误类型+证据缺口+恢复建议；Replanner 局部 patch 而非全局重建
3. **轨迹级日志**：30 种事件类型，JSONL 异步写入，为消融实验和 RL 训练提供数据
4. **证据可追溯写作**：每条 EvidenceItem 带 source/source_type，Writer 合成自然语言报告
5. **后置 RL 增强**：系统、日志和评测稳定后才引入 GRPO/PPO

---

# 二、开发完成状态

## 2.1 Phase 1 全部完成

```
Step 0  ✅  项目蓝图 + .claude/ 规则体系
Step 1  ✅  .gitignore + CLAUDE.local.md
Step 2  ✅  schemas/ (4 文件, 16 数据结构)
Step 3  ✅  configs/ (Pydantic V2 三级配置)
Step 4  ✅  examples/01_async_demo.py (10 示例)
Step 5  ✅  tools/manager.py (熔断/重试/超时) + 3 工具 + mock
Step 6  ✅  agent/planner.py + worker.py
Step 7  ✅  examples/02_simple_agent.py (端到端)
Step 8  ✅  orchestration/dag_workflow.py (LangGraph StateGraph)
Step 9  ✅  agent/verifier.py (9 规则 + LLM Hybrid)
Step 10 ✅  agent/replanner.py (局部重规划, 9 种策略)
Step 11 ✅  memory/hierarchical_memory.py (L1/L2/L3)
Step 12 ✅  logging/trajectory_logger.py (异步 JSONL)
Step 13 ✅  examples/04_multi_agent_research.py (v1 Demo)
       ✅  agent/writer.py (自然语言报告合成)
       ✅  examples/05_web_agent.py (Web 交互界面)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 14 ✅  GitHub Public Beta (README + LICENSE + CI + 推送)
```

## 2.2 实际 vs 计划

| 维度 | 原计划 (16周) | 实际 (2天) |
|------|-------------|-----------|
| Steps 完成 | Phase 1: 7 steps | Phase 1: 14 steps |
| 模块数 | ~8 | 15 |
| 测试 | ~30 | 296 |
| 源码行数 | ~2000 | 6100 |
| Demo | 1 个 | 5 个 |
| LLM 集成 | Phase 2 | 已打通 DeepSeek |
| Verifier | Phase 2 (W5-6) | Phase 1 完成 |
| Replanner | Phase 2 (W5-6) | Phase 1 完成 |
| Memory L1/L2 | Phase 2 (W7-8) | Phase 1 完成 |
| Trajectory Logger | Phase 2 (W7-8) | Phase 1 完成 |
| Writer | 未规划 | Phase 1 完成 |
| GitHub 发布 | Phase 2 | Phase 1 完成 |

---

# 三、仓库结构（最终）

```
horizonrl-agent/
├── README.md                         ← 项目主页
├── LICENSE                           ← MIT
├── CONTRIBUTING.md                   ← 贡献指南
├── CLAUDE.md                         ← Claude Code 入口
├── DEVELOP_PLAN.md                   ← 本文件 (v5.0)
├── .env.example                      ← API 配置模板
├── .gitignore
├── pyproject.toml
├── requirements.txt
│
├── .github/workflows/
│   └── ci.yml                        ← pytest 自动运行
│
├── configs/
│   ├── default.yaml                  ← 生产 (gpt-4o)
│   ├── dev.yaml                      ← 开发 (DeepSeek)
│   └── eval.yaml                     ← 评测
│
├── src/horizonrl/                    ← 源码 (~6100 行, 30 文件)
│   ├── schemas/                      ← 数据协议 (4 文件, 1061 行)
│   │   ├── task.py       UserTask, TaskSpec, PlanGraph, PlanNode, PlanPatch
│   │   ├── result.py     StepResult, VerificationResult, EvidenceItem, ToolCall
│   │   ├── event.py      TrajectoryEvent, TrajectorySession, EventType(30种)
│   │   └── report.py     FinalReport, ReportSection, CitationMap
│   │
│   ├── config/settings.py            ← Pydantic V2 三级配置 (686 行)
│   │
│   ├── tools/                        ← 工具层 (5 文件, 834 行)
│   │   ├── manager.py    ToolManager, CircuitBreaker(三态), 8错误分类
│   │   ├── web_search.py DuckDuckGo / Brave API
│   │   ├── arxiv_search.py Arxiv API
│   │   ├── code_execution.py subprocess 沙箱
│   │   └── mock.py       MockWebSearch, MockArxivSearch, MockCodeExecution
│   │
│   ├── llm/client.py                 ← OpenAI 兼容客户端 (135 行)
│   │
│   ├── agent/                        ← Agent 业务逻辑 (5 文件, 1935 行)
│   │   ├── planner.py    Planner(模板2类) + LLMPlanner(LLM驱动)
│   │   ├── worker.py     AgentWorker + execute_workers()
│   │   ├── verifier.py   Verifier(rule/llm/hybrid) + RuleEngine(9道检查)
│   │   ├── replanner.py  Replanner + LLMReplanner, 9种 ErrorType→PatchType
│   │   └── writer.py     Writer(模板/LLM), 证据→自然语言报告
│   │
│   ├── orchestration/dag_workflow.py ← LangGraph DAG 编排 (502 行)
│   ├── memory/hierarchical_memory.py ← L1窗口 + L2摘要 (550 行)
│   └── logging/trajectory_logger.py  ← 异步 JSONL + 分析工具 (411 行)
│
├── examples/                         ← 5 个 Demo (~2800 行)
│   ├── 01_async_demo.py               asyncio 教程
│   ├── 02_simple_agent.py             端到端 (无需API)
│   ├── 03_llm_demo.py                 LLM 连接测试
│   ├── 04_multi_agent_research.py     v1 旗舰 (6-Stage)
│   └── 05_web_agent.py               Web 交互界面
│
├── tests/                            ← 9 文件, 296 tests (~3700 行)
│   ├── test_imports.py               26 模块导入检查
│   ├── test_tools_manager.py         19 tests: 熔断/超时/重试
│   ├── test_planner.py               9 tests: 模板分解
│   ├── test_worker.py                8 tests: 执行/证据
│   ├── test_verifier.py              24 tests: 规则引擎
│   ├── test_replanner.py             51 tests: 策略映射/补丁
│   ├── test_memory.py                54 tests: L1/L2/压缩
│   ├── test_trajectory_logger.py     41 tests: 写入/分析
│   ├── test_writer.py                12 tests: 合成
│   └── test_dag_workflow.py          28 tests: 图结构/端到端
│
└── .claude/                          ← Claude Code 配置
    ├── settings.json
    ├── rules/                         code-style, testing, architecture
    ├── commands/                      6 个 slash 命令
    ├── skills/                        security-review
    └── agents/                        code-reviewer, test-generator
```

---

# 四、开发历程回顾

## 4.1 实际开发顺序

```
Day 1 (05/12)
  上午: Step 0-1 (骨架) → Step 2 (schemas) → Step 3 (configs)
  下午: Step 4 (01_async_demo) → Step 5 (tools) → Step 6 (planner+worker)
        Step 7 (02_simple_agent) → Step 8 (dag_workflow)
  ── 8 Steps, 138 tests ──

Day 2 (05/13)
  上午: API 配置 (LLMClient + DeepSeek 打通)
        Step 9 (verifier) → 优化 (mock 标准化, CLAUDE.md)
  下午: Step 10 (replanner) → Step 11 (memory)
        Step 12 (trajectory_logger) → Step 13 (04_demo + writer + web)
        Step 14 (GitHub 发布)
  ── 6 Steps + 优化 + 发布, 296 tests ──
```

## 4.2 关键设计决策

| 决策 | 选择 | 验证 |
|------|------|------|
| Schema-first | 先冻结 16 种数据结构 | ✅ 零接口冲突 |
| LangGraph dict state | 纯 dict 绕过序列化 | ⚠️ Phase 2 升级 TypedDict |
| Pydantic V2 三级配置 | 默认→YAML→.env 合并 | ✅ 灵活安全 |
| Circuit Breaker | 三态 CLOSED/OPEN/HALF_OPEN | ✅ 防级联故障 |
| Rule + Hybrid Verifier | 规则 90% + LLM 边界 | ✅ 快且准 |
| ErrorType → PatchType | 9 种映射表 | ✅ Replanner 高效 |
| L1→L2 自动压缩 | Token 阈值触发 | ✅ 零成本 |
| 异步 JSONL | Queue + 后台 writer | ✅ 非阻塞 |
| Writer 双模式 | 模板 / LLM 合成 | ✅ 离线/在线均可 |

## 4.3 踩过的坑

| 问题 | 原因 | 解决 |
|------|------|------|
| Windows GBK 编码 | emoji 输出报错 | `sys.stdout.reconfigure(encoding='utf-8')` |
| LangGraph 序列化 | 嵌套 dataclass 丢失 | dict state + _to_dict() 转换 |
| API Key 优先级 | 系统环境变量覆盖 | `override=True` + `_inject_api_keys()` |
| Verifier 规则顺序 | 误判空结果为工具错误 | 调整 9 道检查优先级 |
| `nonlocal` 误用 | 同作用域变量 | 删除 nonlocal |
| LLM 回退双写 | compress fallback 重复 add | 条件分支修复 |
| DuckDuckGo 国内被墙 | Bing API 不可达 | 5s 超时快速回退 |

---

# 五、数据流（最终）

```
UserTask (自然语言问题)
    │
    ▼
LLMPlanner / Planner ──→ PlanGraph (DAG, 5-6 任务, 依赖边)
    │
    ▼
AgentWorker × N (asyncio + Semaphore 并发)
    │  ToolManager.call() → CircuitBreaker → Timeout → Retry → Tool
    │  真实工具: Arxiv ✅ / Web 部分可用 / Code ✅
    ▼
StepResult + EvidenceItem[]
    │
    ├──→ Verifier (9 规则 <0.1ms / LLM 深度 2-5s)
    │      VerificationResult {pass, score, error_type, gaps, actions}
    │        │
    │        ▼ (失败时)
    │      Replanner → PlanPatch {RETRY/ADD/REMOVE/REORDER}
    │        apply_patch() → 回写 PlanGraph (最多3次/任务)
    │
    ├──→ HierarchicalMemory
    │      L1: record(StepResult, VR) → token阈值触发
    │      L2: compress_from_entries() → 语义摘要
    │      L3: 预留 FAISS 接口
    │
    ├──→ TrajectoryLogger (异步, 非阻塞)
    │      asyncio.Queue → 后台 writer → JSONL
    │      30 种 EventType, 每步 timestamp + cost + latency
    │
    └──→ Writer (模板 / LLM)
           证据 → 结构化自然语言研究报告 (Markdown)
```

---

# 六、已知问题

## 6.1 Web Search 联网

| 问题 | 状态 | 计划 |
|------|------|------|
| DuckDuckGo 国内被墙 | ⚠️ 已加 5s 超时回退 | 后续方案见下方 |

**解决路径（按优先级）：**
1. 注册免费 Brave Search API Key（月 2000 次）→ 设置 `BRAVE_API_KEY` 环境变量
2. 使用代理/VPN
3. 集成国内可用的搜索 API
4. 用 DeepSeek LLM 的搜索能力做替代

## 6.2 其他

| 问题 | 优先级 | 说明 |
|------|--------|------|
| LangGraph dict state | P1 | Phase 2 升级 TypedDict + 自定义 serializer |
| Code Execution 易失败 | P2 | LLMPlanner 生成的代码可能不完整，触发 Traceback |
| L3 FAISS 未实现 | P2 | 当前仅占位，需要 embedding 模型 |
| DuckDuckGo 包名废弃 | P3 | `duckduckgo_search` → `ddgs`，需迁移 |

---

# 七、下一步：Phase 2 路线图

## 7.1 短期（本周）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| **Web Search 修复** | P0 | 配 Brave API 或找国内替代方案 |
| **LLMPlanner 质量提升** | P1 | 当前生成的 DAG 有时过于串行（1 个 root），优化 prompt |
| **Code Execution 增强** | P1 | LLM 生成的代码加 try/except 包装，或给出更具体的 prompt |

## 7.2 中期（下周）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| **LangGraph State TypedDict** | P1 | 解决 dict state 技术债 |
| **消融实验框架** | P1 | `examples/06_ablation_study.py`，验证每个模块独立贡献 |
| **L3 FAISS 检索** | P2 | 需要 embedding API 或本地模型 |
| **Writer LLM 模式测试** | P2 | Writer 的 LLM 合成已写但未充分测试 |

## 7.3 长期（GPU 环境就绪后）

| 任务 | 说明 |
|------|------|
| **vLLM 部署** | 批量推理，降低延迟 |
| **GRPO/PPO RL 训练** | 用轨迹日志做过程奖励信号 |
| **Benchmark 评测** | GAIA、WebArena 等标准基准 |

---

# 八、协作经验总结

1. **一次一个模块** — 专注单文件，实现→测试→通过→下一步
2. **Schema 是合同** — 模块间通过 Schema 通信，不需要读对方的代码
3. **Demo 驱动开发** — 每个阶段都有可运行的 Demo 验证集成
4. **测试即文档** — 296 tests 比任何注释都更准确描述行为
5. **先能跑再优化** — 不求一步完美，但求每步可跑可测

---

*本文件 v5.0，基于 Day 1-2 完整开发 + GitHub 发布。*
*核心原则：先能跑，再好看 → 先可测，再扩展 → 先日志化，再 RL → 先 GitHub，再论文*
