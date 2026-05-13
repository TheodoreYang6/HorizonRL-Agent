# Architecture — 架构约束

## 模块边界（不可违反）
```
schemas/      — 数据协议 (最底层，所有人依赖)
config/       — 配置管理 (被所有模块依赖)
tools/        — 外部工具接口 (独立模块)
llm/          — LLM 调用层 (独立模块，依赖 config/)
memory/       — 分层记忆 (独立于 agent/)
agent/        — Agent 业务逻辑 (Planner, Worker, Verifier, Replanner)
logging/      — 轨迹日志 (横切关注点)
orchestration/ — LangGraph DAG 编排 (顶层，组装所有模块)
rl/           — RL 训练管线 (Phase 3+)
eval/         — 评测指标 (Phase 4)
```

## 依赖方向
- `schemas/` 是最底层，定义全项目数据协议
- `config/` 被所有模块依赖，提供配置入口
- `tools/` 独立，通过 ToolManager 暴露
- `llm/` 独立，提供 OpenAI-compatible 调用能力
- `memory/` 独立于 `agent/`
- `agent/` 依赖 `schemas/`、`tools/`、`llm/`、`memory/`
- `orchestration/` 是顶层，组装 `agent/` + `tools/` + `memory/`
- `logging/` 是横切关注点，所有模块写入

## 数据流（实际实现）
```
UserTask
    │
    ▼
Planner / LLMPlanner ──→ PlanGraph (TaskSpec[] + DAG 依赖)
    │
    ▼
ResearchOrchestrator (LangGraph StateGraph)
    │  plan_task → mark_ready ⟲ execute_batch → finalize
    │  死锁检测 / 迭代上限 / Semaphore 并发
    ▼
AgentWorker (asyncio 并发)
    │  ToolManager.call() → 熔断 → 超时 → 重试 → 工具
    ▼
StepResult + EvidenceItem[]
    │
    ▼
Verifier (rule / llm / hybrid)
    │  VerificationResult {pass, score, error_type, feedback,
    │      evidence_gaps, suggested_actions}
    ▼
[Replanner] → PlanPatch → (回写 PlanGraph)   ← 待实现
    │
    ▼
FinalReport (结构化 Markdown)
```

## 状态管理
- Agent 状态由 LangGraph StateGraph 管理
- 当前使用纯 dict 状态（Phase 2 升级 TypedDict + 自定义序列化）
- 不在模块之间直接共享可变状态
- Memory 是唯一的状态持久化通道

## 关键约束
- 所有模块通过 schemas/ 中定义的数据结构通信
- 工具调用必须走 ToolManager，不允许直接调用
- Worker 不直接访问 LLM，通过 ToolManager 间接使用
- Verifier 的输出 (suggested_actions) 是 Replanner 的输入
