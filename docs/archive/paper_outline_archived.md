# HorizonRL-Agent 论文大纲 v2.0

> **工作标题**: HorizonRL-Agent: Improving Long-Horizon Agent Stability via
>   Hierarchical Memory and Verifier-Guided Replanning
> **目标会议**: AAAI 2027 / IJCAI 2027 / ACL Findings 2027
> **投稿窗口**: AAAI 2026.08 / IJCAI 2027.01 / ACL 2026.12
> **当前状态**: 系统 v0.3.0 (330 tests, 7 demos, 40 benchmark tasks, 18 commits)
> **更新**: 2026-05-17 Day 7

---

## 一、论文元信息

| 字段 | 内容 |
|------|------|
| 标题 | HorizonRL-Agent: Improving Long-Horizon Agent Stability via Hierarchical Memory and Verifier-Guided Replanning |
| 作者 | Qiduo Yang et al. |
| 类型 | Long Paper (8 pages + references) |
| 关键词 | LLM Agents, Long-Horizon Tasks, Hierarchical Memory, Verifier-Guided Replanning, Multi-Agent Orchestration |
| 贡献点 | 5 个 (见下) |

### 五个核心贡献点

1. **分层记忆结构 (Hierarchical Memory)**：L1 FIFO 工作窗口 → L2 语义压缩摘要 → L3 FAISS 情节归档，在长任务中主动折叠上下文，将有效上下文窗口扩展 3-5x
2. **验证器驱动的局部重规划 (Verifier-Guided Replanning)**：9 规则引擎 (0.1ms) + LLM Hybrid 验证器，9 种 ErrorType→PatchType 恢复策略，仅对未来子图做局部 patch 而非全局重置
3. **异步多智能体 DAG 编排 (Async Multi-Agent DAG)**：基于 LangGraph StateGraph 的 6 节点状态机，支持死锁检测、并发控制、迭代上限
4. **轨迹级日志基础设施 (Trajectory-Level Logging)**：30 种事件类型异步 JSONL 写入，支持全生命周期可观测、可恢复、可审计
5. **证据可追溯的双模式写作 (Evidence-Traceable Writing)**：SearchProvenance + UserAnswerWriter/DebugReportRenderer 双输出，引用→证据可回溯

---

## 二、各章节大纲

### Section 1: Introduction (约 1 页)

**段落结构**：

**Para 1 — 问题背景**
- LLM Agent 在复杂任务中展现出强大能力 (SWE-bench, WebArena, GAIA)
- 但当前主流 Agent 在 20+ 步长链路任务中表现急剧下降
- 典型失败模式：上下文污染、任务漂移、无效操作循环、幻觉累积、失败后无法恢复
- 给出一组具体数字 (引用外部基准中的成功率-步数衰减曲线)

**Para 2 — 现有方法的局限**
- ReAct / AutoGPT：单层上下文，无结构化记忆管理，步数增加后上下文窗口溢出
- LangGraph 原生：提供工作流编排，但缺少内置的验证-恢复闭环和分层记忆
- RL-based Agent：关注最终奖励，忽略过程级稳定性，学到的策略脆弱

**Para 3 — 本文核心洞察**
- 核心论点：**长链路稳定性不是单一的 planning 问题，而是记忆管理、错误诊断、局部恢复、轨迹级评测共同作用的系统问题**
- 三条设计原则：
  1. 上下文必须被主动折叠而非被动截断 (Hierarchical Memory)
  2. 失败恢复必须是局部的而非全局的 (Verifier-Guided Replanning)
  3. 并发执行需要 DAG 级死锁检测和状态管理 (Async DAG Orchestration)

**Para 4 — 贡献总结**
- 提出 HorizonRL-Agent，包含上述三项核心机制
- 在 XX 个长链路基准上，相比 AutoGPT/LangGraph 原生/ReAct 基线，成功率提升 XX%
- 开源发布完整代码、基准数据和复现脚本

---

### Section 2: Related Work (约 1 页)

**2.1 LLM Agent Architectures**
- ReAct (Yao et al., 2023): 推理-行动交替，缺少结构化记忆
- AutoGPT / BabyAGI: 简单的任务分解+执行，无验证-恢复
- LangGraph (LangChain, 2024): StateGraph 编排，但缺少内置验证器和分层记忆
- 本文定位：在 LangGraph 基础上增加记忆、验证、重规划三层基础设施

**2.2 Memory for LLM Agents**
- MemGPT (Packer et al., 2024): OS 风格内存管理，但面向对话而非任务执行
- HiMem (2024): 分层长期记忆，但缺少与重规划的联动
- Context-Folding (2024): 上下文折叠概念，本文将其系统化为 L1→L2→L3
- 本文区别：三层记忆与验证器、重规划形成闭环，记忆压缩由验证事件触发

**2.3 Verification and Self-Correction**
- Self-Refine (Madaan et al., 2024): LLM 自反馈，缺少结构化规则
- Reflexion (Shinn et al., 2024): 口头反思，但无局部 plan patch
- RLVMR (2024): 验证器驱动的重规划，本文将其从单一 LLM 验证扩展为 rule + LLM hybrid
- 本文区别：9 规则引擎 + LLM Hybrid 双层验证，9 种 ErrorType→PatchType 精确映射

**2.4 RL for Agent Training**
- GRPO (DeepSeek-R1, 2025): Group Relative Policy Optimization
- ARPO / LOOP: Agentic RL 最新进展
- 本文定位：RL 是后期增强而非前期依赖，先在冻结模型上建立系统基础设施

---

### Section 3: HorizonRL-Agent System Design (约 2.5 页)

**3.1 System Overview**
- 给出系统架构图 (已有 Mermaid 图，需要画成论文矢量图)
- 6 节点 LangGraph 状态机：`plan_task → mark_ready ⟲ execute_batch → verify → replan → finalize`
- 数据流：UserTask → PlanGraph → StepResult[] → VerificationResult[] → PlanPatch → FinalReport

**3.2 Hierarchical Memory (核心模块 1)**

```
L1 RecentWindow (FIFO, Token 阈值触发压缩)
  → 保留最近 K 步完整轨迹
  → 超过阈值 → 触发 compress_to_L2()

L2 SemanticSummary (模板 / LLM 压缩)
  → 对 L1 溢出内容做结构化摘要
  → 保留：目标、关键发现、失败模式、工具调用统计

L3 EpisodicArchive (FAISS 向量检索 + MD5 n-gram 哈希)
  → 持久化存储历史任务经验
  → 混合检索：向量相似度 + n-gram 精确匹配 + 时间衰减
  → MemoryContext 结构：Agent 每步消费的结构化上下文
```

- 关键设计决策：
  - L1→L2 压缩由 Token 阈值自动触发，而非手动配置
  - L3 使用 DashScope text-embedding-v4 API (1024维) + MD5 n-gram 哈希双模回退
  - MemoryContext 对 Planner/Worker/Verifier 提供统一接口
  - L3 归档由验证事件驱动：成功任务自动存入长期经验

**3.3 Verifier-Guided Replanning (核心模块 2)**

```
RuleEngine (9 条规则, 0.1ms)
  1. 空结果检查        6. 幻觉检测 (无引用声明)
  2. 工具调用失败检查   7. 完整性检查
  3. 超时检查          8. 一致性检查 (与之前步骤矛盾)
  4. 长度/质量检查     9. 格式检查
  5. 证据充分性检查

LLMVerifier (DeepSeek/OpenAI, ~2s)
  → 对 RuleEngine 不确定的 case 做 LLM 诊断
  → 输出: {pass, score, error_type, feedback, evidence_gaps, suggested_actions}

Replanner (9 种 ErrorType→PatchType 策略)
  ErrorType → PatchType 映射:
    EMPTY_RESULT → RETRY
    TOOL_FAILURE → RETRY (换工具)
    HALLUCINATION → ADD (补充搜索)
    INCOMPLETE → ADD (补充子任务)
    INCONSISTENT → REMOVE (删除冲突节点)
    IRRELEVANT → REPLACE
    LOW_QUALITY → RETRY
    TIMEOUT → REDUCE_SCOPE (缩小范围)
    FORMAT_ERROR → RETRY
```

- 关键设计：局部 patch 而非全局重规划 — 只修改 PlanGraph 中受影响的子图

**3.4 Async DAG Orchestration**
- 基于 LangGraph StateGraph 的 6 节点状态机
- `mark_ready` 节点：DAG 依赖解析 + Semaphore 并发控制
- 死锁检测：若连续 N 轮无 ready 任务但仍有 pending，触发强制重规划
- `execute_batch`：asyncio.gather 并发执行所有 ready 任务
- 迭代上限：max_iterations 防止无限循环

**3.5 Trajectory Logging**
- 30 种事件类型 (EventType enum)
- 异步 JSONL 写入 (不阻塞主循环)
- 5 种分析工具：timeline, cost_breakdown, error_distribution, tool_usage_stats, replay

**3.6 Evidence-Traceable Writing**
- SearchProvenance：每个搜索结果携带来源 URL、检索时间、相关性分数
- UserAnswerWriter：面向最终用户，无调试信息，引用标注
- DebugReportRenderer：面向开发者，完整轨迹 + 决策链 + 错误回溯

---

### Section 4: Experimental Setup (约 1 页)

**4.1 Research Questions**
- **RQ1 (Memory)**: 分层记忆 (L1+L2+L3) 相比单层上下文，在 20+ 步任务上的成功率提升？
- **RQ2 (Replanning)**: Verifier-guided replanning 相比无恢复 / 全局重规划，恢复率提升？
- **RQ3 (Ablation)**: 每个模块 (Memory / Verifier / Replanner / DAG) 的独立贡献？
- **RQ4 (Scalability)**: 任务步数从 5 到 30 步，各方法的性能衰减曲线？
- **RQ5 (Efficiency)**: Token 消耗、工具调用效率、平均完成时间的对比？

**4.2 Benchmarks**
| 基准 | 任务数 | 平均步数 | 类型 |
|------|--------|---------|------|
| HorizonRL-Bench (自建) | 20 | 15-30 | 混合 (研究/代码/数据分析) |
| GAIA | 165+ | 5-20 | 多模态 QA |
| WebArena | 812 | 3-10 | Web 导航 |

**4.3 Baselines**
- **ReAct**: 基础推理-行动循环，无记忆，无重规划
- **AutoGPT**: 简单任务分解 + 执行，单层上下文
- **LangGraph Native**: 原生 StateGraph 编排，无验证器，无分层记忆
- **HorizonRL w/o Memory**: 消融：仅 DAG + Verifier + Replanner
- **HorizonRL w/o Replanning**: 消融：仅 DAG + Memory
- **HorizonRL (Full)**: 完整系统

**4.4 Metrics**
- Success Rate (任务级 / 子任务级)
- Recovery Rate (失败后恢复成功的比例)
- Token Cost (总输入/输出 token)
- Tool Efficiency (有效工具调用 / 总调用)
- Trajectory Length (action-observation-verification 步数)
- Hallucination Rate (最终报告中无引用支持的陈述比例)

**4.5 Implementation Details**
- LLM: DeepSeek-V3 (via OpenAI-compatible API)
- Embedding: MD5 n-gram 哈希 (L3) + DeepSeek Embedding API
- 硬件: 2×A800 80GB / 1×A100 80GB
- 并发: asyncio + Semaphore(5)
- 超参数: max_iterations=30, l1_token_threshold=4096, semaphore_limit=5

---

### Section 5: Results and Analysis (约 1.5 页)

**5.1 Main Results (表格)**
- 给出所有方法在所有基准上的 Success Rate、Recovery Rate、Token Cost 对比表
- 预期：Full Model 在 20+ 步任务上比 AutoGPT 高 30-40% 成功率

**5.2 Ablation Study**
- 逐步移除 Memory / Verifier / Replanner，记录性能降幅
- 预期：Memory 对长任务贡献最大 (>15 步后差距拉大)
- 预期：Replanner 对困难任务 (高频工具调用) 贡献最大

**5.3 Scalability Analysis**
- 固定任务类型，变化步数 (5, 10, 15, 20, 25, 30)
- 绘制各方法的成功率-步数衰减曲线
- 预期：Full Model 衰减最慢，AutoGPT 在 10+ 步后急剧下降

**5.4 Error Analysis**
- 失败模式分类统计
- 各 ErrorType 的恢复成功率
- 案例分析：展示一个典型的长任务轨迹，对比 Full Model vs AutoGPT 的行为差异

**5.5 Cost Analysis**
- 各方法的平均 Token 消耗、工具调用次数、总耗时
- 分层记忆带来的 Token 节省 (L2 摘要替代完整 L1 历史)

---

### Section 6: Discussion (约 0.5 页)

- **为什么分层记忆有效**：不是简单存更多，而是通过 L2 摘要过滤掉噪声，让 Planner 看到的是"浓缩信息"而非"原始日志"
- **为什么局部重规划优于全局重规划**：全局重规划丢弃了已成功步骤的成果，局部 patch 保留了有效历史
- **局限性**：
  - 当前 L3 使用 n-gram 哈希，语义匹配能力弱于真实 Embedding (Phase 3 改进)
  - Verifier 的规则阈值需要针对不同任务类型手动调整
  - 流式输出仅为阶段级，非 token 级
- **未来工作**：RL 训练 (GRPO/PPO) 自动优化重规划策略、token 级流式、多模态 Agent

---

### Section 7: Conclusion (约 0.3 页)

- 本文提出 HorizonRL-Agent，通过分层记忆、验证器驱动的局部重规划和异步 DAG 编排三项机制，提升 LLM Agent 在 20+ 步长链路任务中的稳定性
- 实验表明，相比 AutoGPT 和 LangGraph 原生基线，成功率提升 XX%，Token 消耗降低 XX%
- 开源发布完整系统，为长链路 Agent 研究提供可复现的基础设施

---

## 三、待补充数据 (GPU 到位后优先做)

| 优先级 | 数据 | 依赖 |
|--------|------|------|
| P0 | HorizonRL-Bench 20 题全量跑通 (含 Full Model + 所有消融配置) | 无 (mock 模式可跑) |
| P0 | AutoGPT / ReAct 基线复现 | 无 |
| P1 | GAIA / WebArena 部分题评测 | API |
| P1 | 成功率-步数衰减曲线 | P0 数据 |
| P1 | Error 分类统计 + 恢复率 | P0 数据 |
| P2 | RL 训练 (GRPO) | A100 GPU |
| P2 | vLLM 批量推理 | A100 GPU |

---

## 四、投稿时间线

```
2026.05 — 论文大纲完成 (当前)
2026.05-06 — 跑实验数据 (消融 + 基线对比)
2026.06 — 论文初稿 (方法 + 实验部分)
2026.07 — 论文修改 + 补充实验
2026.08 — AAAI 2027 投稿 (8月初截止)
2026.09 — ICLR 2027 投稿 (备选)
2026.12 — ACL 2027 投稿 (备选)
2027.01 — IJCAI 2027 投稿 (备选)
```

---

## 五、写作注意事项

1. **Motivation 要具体**：不要泛泛地说 "LLM agents fail on long tasks"，要给具体的失败模式和频率数据
2. **贡献要可度量**：每个 contribution 后面必须跟实验数据支撑
3. **图表质量**：系统架构图、成功率衰减曲线、消融条形图 — 这三个图是 reviewer 第一眼看的
4. **Related Work 要公正**：不要只说别人不好，要说清楚"别人做了什么 + 我们补充了什么"
5. **代码开源**：匿名仓库链接放在 footnote 里 (review 阶段用 anonymous repo)
6. **附录**：更多 case study、完整 prompt 模板、超参数敏感性分析

---

*本大纲 v1.0，基于系统当前实现状态 (2026-05-14) 撰写。随着实验数据补充持续更新。*
