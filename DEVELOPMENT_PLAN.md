# HorizonRL-Agent 开发路线图 v6.0 (最终)

> **更新日期**: 2026-05-13 (Day 2 最终)
> **状态**: Phase 1 + Phase 2 全部完成, GitHub 已发布
> **测试**: 323 passed, 4 skipped, 0 failed
> **代码量**: ~15,000 行
> **Git**: 14 commits

---

# 一、项目定位与研究问题

## 一句话定位

面向长链路研究与工具使用任务，构建一个以**分层记忆、验证器诊断和局部重规划**为核心的多智能体 LLM Agent 稳定执行系统。

## 核心创新点

1. **分层记忆结构**：L1 最近工作记忆 → L2 语义摘要 → L3 FAISS 向量检索
2. **Verifier 驱动的重规划**：9 规则检查 + 9 种 ErrorType→PatchType 恢复策略
3. **轨迹级日志**：30 种事件类型 JSONL 异步写入
4. **证据可追溯写作**：SearchProvenance + UserAnswerWriter + DebugReportRenderer 双输出
5. **后置 RL 增强**：Phase 3+ 接入 GRPO/PPO

---

# 二、开发完成状态

## 2.1 Phase 1: 核心基础设施 (14 Steps)

```
Step 0  ✅ 项目蓝图 + .claude/ 规则体系
Step 1  ✅ .gitignore + CLAUDE.local.md
Step 2  ✅ schemas/ (4 文件, 16+ 数据结构)
Step 3  ✅ configs/ (Pydantic V2 三级配置)
Step 4  ✅ examples/01_async_demo.py
Step 5  ✅ tools/manager.py + 3 工具 + mock
Step 6  ✅ agent/planner.py + worker.py
Step 7  ✅ examples/02_simple_agent.py
Step 8  ✅ orchestration/dag_workflow.py (LangGraph)
Step 9  ✅ agent/verifier.py (9规则 + LLM Hybrid)
Step 10 ✅ agent/replanner.py (9种策略)
Step 11 ✅ memory/hierarchical_memory.py (L1/L2)
Step 12 ✅ logging/trajectory_logger.py (JSONL)
Step 13 ✅ examples/04_multi_agent_research.py (v1 Demo)
       ✅ agent/writer.py + examples/05_web_agent.py
Step 14 ✅ GitHub Public Beta
```

## 2.2 Phase 2: 产品化 + 评测 (4 Steps)

```
━━━ ChatGPT 建议优化 ━━━
      ✅ Writer v2: UserAnswerWriter + DebugReportRenderer 双模式
      ✅ SearchProvenance + ReportMetadata
      ✅ Web 双路由: /api/chat + /api/report + /api/download
      ✅ ToolManager 搜索结果规范化 + provider 路由
      ✅ CI 强制 mock 模式
      ✅ LLMPlanner prompt 优化并行度
      ✅ 代码质量 ruff 优化
      ✅ SYSTEM_MANUAL.md (15章开发手册)

━━━ Phase 2 主线 ━━━
Step 15 ✅ 消融实验框架     (examples/06_ablation_study.py + StressInjector)
Step 16 ✅ L3 FAISS         (L3EpisodicArchive + 向量检索 + 持久化)
Step 17 ✅ Benchmark        (examples/07_benchmark.py, 20题5类)
Step 18 ✅ SSE 流式输出     (Web Demo 实时进度推送)
```

## 2.3 Phase 3 & 4 (未开始)

```
Phase 3 (需 GPU):
  ⬜ RL 训练管线 (GRPO/PPO via TRL + veRL)
  ⬜ vLLM 批量推理部署

Phase 4 (论文准备):
  ⬜ 论文写作
  ⬜ 对比基线 (AutoGPT / LangGraph 原生)
  ⬜ 更多 Benchmark (GAIA, WebArena)
```

---

# 三、开发历程

## 3.1 时间线

```
Day 1 (05/12): Steps 0-8  — 骨架 + schemas + tools + agent + orchestration
Day 2 (05/13): Steps 9-18 — verifier + replanner + memory + logger + writer
                            + web + LLM集成 + 消融 + FAISS + benchmark + SSE
                            + ChatGPT建议产品化优化
```

## 3.2 关键数字

```
Day 1: 8 Steps, 138 tests, ~7,600 行
Day 2: 10 Steps + 优化, 323 tests, ~15,000 行
```

| 维度 | 数值 |
|------|------|
| 源码文件 | 30+ |
| 测试文件 | 9 |
| Demo | 7 个 (01-07) |
| Tests | 323 passed |
| Commits | 14 |
| GitHub Stars | 即将 |

---

# 四、数据流 (最终)

```
UserTask
    │
    ▼
LLMPlanner / Planner ──→ PlanGraph (DAG)
    │
    ▼
AgentWorker × N (asyncio + Semaphore)
    │  ToolManager → CircuitBreaker → 超时/重试 → 工具
    │  SearchProvider 路由: bocha→brave→ddgs→wikipedia→mock
    ▼
StepResult + EvidenceItem[] (带 SearchProvenance)
    │
    ├──→ Verifier (9规则 + LLM Hybrid)
    │      VerificationResult {pass, score, error_type, gaps, actions}
    │        │
    │        ▼ (失败)
    │      Replanner → PlanPatch → apply_patch → 回写 PlanGraph
    │
    ├──→ HierarchicalMemory
    │      L1 (FIFO窗口) → L2 (语义摘要) → L3 (FAISS向量检索)
    │
    ├──→ TrajectoryLogger (异步 JSONL, 30种事件)
    │
    └──→ Writer
           ├── UserAnswerWriter → final_answer.md (用户, 无调试信息)
           └── DebugReportRenderer → debug_report.md (开发者)
```

---

# 五、Demo 清单

| # | 文件 | 功能 | 需要 API |
|---|------|------|---------|
| 01 | `01_async_demo.py` | asyncio 教程 (10示例) | 否 |
| 02 | `02_simple_agent.py` | 最简端到端管道 | 否 |
| 03 | `03_llm_demo.py` | LLM连接测试 + 智能规划 | 是 |
| 04 | `04_multi_agent_research.py` | v1 旗舰 (6-Stage Pipeline, 双报告输出) | 可选 |
| 05 | `05_web_agent.py` | Web 对话界面 (双路由 + SSE流式 + 文件下载) | 自动检测 |
| 06 | `06_ablation_study.py` | 消融实验框架 (5配置 + 压力注入) | 否 |
| 07 | `07_benchmark.py` | Benchmark 评测 (20题5类) | 否 |

---

# 六、已知限制

| 问题 | 影响 | 计划 |
|------|------|------|
| DDGS 国内偶尔超时 | Web搜索偶慢 | Bocha Key |
| Bocha API 未接入 | 国内最优方案缺 | 需注册Key |
| 流式仅阶段级 | 不是逐字输出 | Phase 3 |
| Code Execution 易失败 | 代码任务需重试 | prompt优化 |
| 无 GPU | RL训练/本地模型 | 等服务器 |

---

# 七、下一步

```
明天:
  □ 整体验收: 跑所有 Demo + 看效果
  □ 论文大纲
  □ 录 demo 视频 (用于展示)

Phase 3 (等 GPU):
  □ vLLM 部署
  □ GRPO/PPO RL 训练
```

---

*本文件 v6.0，基于 Day 1-2 完整开发数据。*
*核心原则: 先能跑，再好看 → 先可测，再扩展 → 先日志化，再 RL → 先 GitHub，再论文*
