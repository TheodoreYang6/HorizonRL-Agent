"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
轨迹事件数据结构 —— 项目数据模型第三层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本文件定义所有"事件/日志"相关的数据结构。这是系统的"神经系统"——
每一个模块的执行动作都被记录为结构化的 TrajectoryEvent。

── 为什么轨迹日志是一等公民 ──

  没有结构化轨迹 → 无法做消融实验 → 无法分析失败原因 → RL 训练不可解释
  有了结构化轨迹 → 可以回答：
    - 哪个模块最耗时？
    - 哪种工具调用失败率最高？
    - 重规划是否真的改善了成功率？
    - 幻觉通常发生在第几步？

── 数据结构一览 ──

    EventType        — 事件类型枚举（规划/执行/验证/工具/记忆/重规划/输出）
    TrajectoryEvent  — 单条轨迹事件（含时间戳、模块、类型、负载、成本、延迟）
    TrajectorySession — 一次完整 Agent 运行的所有事件 + 统计摘要

── 数据流向 ──

    所有模块 → TrajectoryLogger.log(event: TrajectoryEvent)
                ↓ 异步写入（不阻塞主流程）
              JSONL / Parquet 文件
                ↓ Phase 3/4 读取
              TrajectoryBuffer → GRPOTrainer / metrics.py

── 被哪些模块依赖 ──
    logging/trajectory_logger.py — 写入
    rl/trajectory_buffer.py       — 读取
    eval/metrics.py               — 读取计算指标
    scripts/export_traces.py      — 导出
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ─── 事件类型 ──────────────────────────────────────────────────────────
# 覆盖 Agent 执行全生命周期。每个模块的动作对应一种事件类型。
# Logger 写入时根据 event_type 决定序列化策略和索引策略。


class EventType(str, Enum):
    """轨迹事件类型 —— 覆盖全生命周期。"""

    # ── Planner ──
    PLAN_START = "plan.start"            # Planner 开始拆解任务
    PLAN_COMPLETE = "plan.complete"      # Planner 生成 PlanGraph 完成
    PLAN_ERROR = "plan.error"            # Planner 拆解失败

    # ── Worker ──
    WORKER_START = "worker.start"        # Worker 开始执行子任务
    WORKER_STEP = "worker.step"          # Worker 执行中的中间步骤
    WORKER_COMPLETE = "worker.complete"  # Worker 完成子任务
    WORKER_ERROR = "worker.error"        # Worker 执行失败

    # ── Tool ──
    TOOL_CALL = "tool.call"              # 工具调用开始
    TOOL_RESULT = "tool.result"          # 工具调用返回
    TOOL_ERROR = "tool.error"            # 工具调用失败（超时/权限/解析）

    # ── Verifier ──
    VERIFY_START = "verify.start"        # Verifier 开始验证
    VERIFY_COMPLETE = "verify.complete"  # Verifier 验证完成
    VERIFY_FAIL = "verify.fail"          # 验证不通过（注意：这是正常事件，不是错误）

    # ── Replanner ──
    REPLAN_START = "replan.start"        # Replanner 开始分析失败
    REPLAN_PATCH = "replan.patch"        # Replanner 生成了一个 PlanPatch
    REPLAN_SKIP = "replan.skip"          # 超过最大重试，跳过

    # ── Memory ──
    MEMORY_UPDATE = "memory.update"      # Memory 写入新条目
    MEMORY_COMPRESS = "memory.compress"  # L1 → L2 压缩
    MEMORY_ARCHIVE = "memory.archive"    # L2 → L3 归档
    MEMORY_RETRIEVE = "memory.retrieve"  # Memory 检索查询

    # ── Writer ──
    WRITER_START = "writer.start"        # Writer 开始生成报告
    WRITER_COMPLETE = "writer.complete"  # Writer 完成最终报告

    # ── 系统级 ──
    SESSION_START = "session.start"      # Agent 会话开始
    SESSION_END = "session.end"          # Agent 会话结束（含最终统计）
    ERROR = "system.error"               # 未分类系统错误


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TrajectoryEvent — 单条轨迹事件                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 这是系统中最重要的日志数据结构。每个模块在关键动作后都应创建一个
# TrajectoryEvent 并写入 TrajectoryLogger。
#
# 设计原则：
#   1. 每个事件都有精确时间戳（ts），用于延迟分析
#   2. 每个事件都有 module 和 type，方便按模块/类型过滤
#   3. payload 是自由字典，不同事件类型填写不同字段（见下文约定）
#   4. cost 和 latency 单独字段，方便聚合统计
#
# Payload 约定（每个事件类型的 payload 应包含的字段）：
#   plan.*       → {"user_task": str, "num_subtasks": int, "plan_json": str}
#   worker.*     → {"task_id": str, "worker_id": str, "prompt_len": int}
#   tool.*       → {"tool_name": str, "input": dict, "output": str, "tool_elapsed": float}
#   verify.*     → {"task_id": str, "pass": bool, "score": float, "error_type": str}
#   replan.*     → {"target_node": str, "patch_type": str, "reason": str}
#   memory.*     → {"layer": str, "action": str, "num_items": int}
#   writer.*     → {"report_sections": int, "citation_count": int}
#   session.*    → {"session_id": str, "user_task": str}
#   system.error → {"exception": str, "traceback": str, "module": str}


@dataclass
class TrajectoryEvent:
    """单条轨迹事件。

    Attributes:
        ts: Unix 时间戳（秒），精确到微秒
        module: 产生事件的模块名（"planner", "worker", "verifier", ...）
        event_type: 事件类型枚举
        payload: 事件负载（自由字典，不同事件类型填不同字段）
        cost: 本次操作消耗的 token 数
        latency: 本次操作耗时（秒）
        session_id: 会话 ID（同一 Agent 运行的所有事件共享）
        step_id: 步序号（从 1 开始递增，用于追踪执行进度）
    """

    ts: float = field(default_factory=_time.time)
    module: str = ""
    event_type: EventType = EventType.ERROR
    payload: dict[str, Any] = field(default_factory=dict)
    cost: int = 0        # token 消耗
    latency: float = 0.0  # 操作耗时（秒）
    session_id: str = ""
    step_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于 JSONL 导出）。"""
        return {
            "ts": self.ts,
            "module": self.module,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "cost": self.cost,
            "latency": self.latency,
            "session_id": self.session_id,
            "step_id": self.step_id,
        }

    def __repr__(self) -> str:
        return (
            f"[{self.step_id:03d}] {self.event_type.value:20s} | "
            f"{self.module:10s} | cost={self.cost:5d} | "
            f"lat={self.latency:.3f}s"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TrajectorySession — 一次完整 Agent 运行                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 一次 Agent 运行（从 UserTask 到 FinalReport）的所有事件 + 统计摘要。
# 这是导出和分析的基本单位。


@dataclass
class TrajectorySession:
    """一次完整 Agent 运行的所有轨迹事件。

    Attributes:
        session_id: 会话唯一标识
        user_task: 原始用户任务描述
        events: 所有 TrajectoryEvent（按时间顺序）
        started_at: 会话开始时间戳
        finished_at: 会话结束时间戳
        success: 最终是否成功
        total_tokens: 总 token 消耗
        total_steps: 总步数
        total_tool_calls: 总工具调用次数
        replan_count: 重规划次数
    """

    session_id: str = ""
    user_task: str = ""
    events: list[TrajectoryEvent] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    success: bool = False
    total_tokens: int = 0
    total_steps: int = 0
    total_tool_calls: int = 0
    replan_count: int = 0

    @property
    def wall_time(self) -> float:
        """总耗时（秒）。"""
        return self.finished_at - self.started_at

    @property
    def avg_latency(self) -> float:
        """事件平均延迟。"""
        latencies = [e.latency for e in self.events if e.latency > 0]
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    def filter_by_module(self, module: str) -> list[TrajectoryEvent]:
        """按模块过滤事件。"""
        return [e for e in self.events if e.module == module]

    def filter_by_type(self, event_type: EventType) -> list[TrajectoryEvent]:
        """按事件类型过滤。"""
        return [e for e in self.events if e.event_type == event_type]

    def add_event(self, event: TrajectoryEvent) -> None:
        """添加事件并自动更新统计。"""
        event.session_id = self.session_id
        event.step_id = len(self.events) + 1
        self.events.append(event)
        # 自动累加统计
        self.total_tokens += event.cost
        if event.event_type == EventType.TOOL_RESULT:
            self.total_tool_calls += 1
        if event.event_type == EventType.REPLAN_PATCH:
            self.replan_count += 1

    def to_summary(self) -> dict[str, Any]:
        """生成统计摘要。"""
        return {
            "session_id": self.session_id,
            "user_task": self.user_task[:200],
            "success": self.success,
            "wall_time": self.wall_time,
            "total_tokens": self.total_tokens,
            "total_steps": len(self.events),
            "total_tool_calls": self.total_tool_calls,
            "replan_count": self.replan_count,
            "avg_latency": self.avg_latency,
        }
