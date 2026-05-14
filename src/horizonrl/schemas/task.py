"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
任务相关数据结构 —— 项目数据模型第一层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本文件定义所有"任务"相关的数据结构。任务是系统中流动的最基本单位。
在整个项目中，数据流是这样的：

    UserTask (用户输入)
        ↓ Planner.plan_tasks()
    PlanGraph (包含多个 TaskSpec)
        ↓ Worker.execute()
    StepResult (每个子任务的执行结果)
        ↓ Verifier.verify()
    VerificationResult (验证通过/失败)
        ↓ (如果失败) Replanner.replan()
    PlanPatch (局部修正)
        ↓ (如果成功) Memory.update()
    ...重复直到完成...
        ↓ Writer.generate()
    FinalReport (最终输出)

── 数据结构一览 ──

    UserTask    — 用户输入的原始任务（自然语言描述）
    TaskSpec    — Planner 拆解后的单个子任务规格（结构化、可执行）
    PlanNode    — PlanGraph 中的一个节点（有依赖关系）
    PlanGraph   — 整个任务的有向无环图（DAG）
    PlanPatch   — Replanner 对 PlanGraph 的局部修改（增/删/重排/重试）

── 被哪些模块依赖 ──
    agent/planner.py    — 输出 TaskSpec[], PlanGraph
    agent/worker.py     — 输入 TaskSpec, 输出 StepResult
    agent/replanner.py  — 输入 PlanGraph, 输出 PlanPatch
    orchestration/dag_workflow.py — 调度 PlanGraph 执行
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ─── 任务优先级 ───────────────────────────────────────────────────────────
# 优先级决定 Worker 调度时谁先执行。
# 项目中：Semaphore 释放后，高优先级 TaskSpec 先被取出。


class TaskPriority(str, Enum):
    """任务优先级。P0 最高，P2 最低。"""

    P0 = "p0"  # 关键路径任务，必须先执行（如"搜索背景"阻塞后续汇总）
    P1 = "p1"  # 正常任务
    P2 = "p2"  # 可后置任务（如"润色格式"）


# ─── 任务状态 ───────────────────────────────────────────────────────────
# PlanGraph 中每个节点都会经历这些状态。


class TaskStatus(str, Enum):
    """PlanNode 的生命周期状态。"""

    PENDING = "pending"        # 等待执行（依赖未满足）
    READY = "ready"            # 依赖已满足，等待 Worker 取走
    RUNNING = "running"        # Worker 正在执行
    SUCCESS = "success"        # Verifier 验证通过
    FAILED = "failed"          # Verifier 验证失败，等待 Replanner 处理
    SKIPPED = "skipped"        # 被 Replanner 标记为跳过（不阻塞整体）
    CANCELLED = "cancelled"    # 被用户或超时取消


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UserTask — 用户输入                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 这是一切开始的入口。用户只需要提供一个自然语言描述和一些可选约束。
# Planner 会将 UserTask 拆解为结构化 TaskSpec 列表。


@dataclass
class UserTask:
    """用户提交的原始任务。

    Attributes:
        description: 自然语言任务描述，如 "调研 Transformer 注意力机制的最新进展"
        max_steps: 最多允许的总执行步数（默认 30），超限则任务终止
        max_tokens: 总 token 预算上限（默认 50000），用于控制成本
        required_tools: 必须使用的工具列表，如 ["web_search", "arxiv_search"]
        output_format: 期望的输出格式提示，如 "带引用的结构化报告"
    """

    description: str
    max_steps: int = 30
    max_tokens: int = 50_000
    required_tools: list[str] = field(default_factory=list)
    output_format: str = "markdown"

    def __repr__(self) -> str:
        return (
            f"UserTask(description={self.description[:60]}..., "
            f"max_steps={self.max_steps})"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TaskSpec — Planner 拆解出的单个子任务                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 这是 Planner 的核心输出。一个 UserTask 会被拆成 3-6 个 TaskSpec。
# Worker 拿到 TaskSpec 后就知道"该干什么、用什么工具、依赖谁的结果"。
#
# 关键字段说明：
#   id          — 全局唯一标识，用 task_ + UUID 前 8 位
#   depends_on  — 这个任务依赖哪些其他 TaskSpec 的 id，形成 DAG 边
#   tool_names  — Worker 可以使用的工具列表（白名单）
#   context     — Planner 给这个子任务附带的额外上下文（如"参考前一步搜索结果"）


@dataclass
class TaskSpec:
    """Planner 拆解后的单个子任务。

    Examples:
        >>> task = TaskSpec(
        ...     id="task_001",
        ...     name="搜索 Transformer 注意力机制",
        ...     description="在 arxiv 搜索 Transformer attention 相关论文",
        ...     tool_names=["arxiv_search"],
        ...     depends_on=[],
        ... )
    """

    id: str
    name: str                          # 简短名称，用于日志和展示
    description: str                   # 详细描述，Worker 据此理解任务
    tool_names: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  # 依赖的 TaskSpec.id
    priority: TaskPriority = TaskPriority.P1
    context: str = ""                  # Planner 附带的额外上下文
    retry_count: int = 0               # 已重试次数
    max_retries: int = 3               # 最多重试次数

    def __repr__(self) -> str:
        deps = ",".join(self.depends_on) if self.depends_on else "无"
        return (
            f"TaskSpec(id={self.id}, name={self.name[:40]}..., "
            f"tools={self.tool_names}, depends_on=[{deps}], "
            f"priority={self.priority}, retries={self.retry_count}/{self.max_retries})"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PlanNode — PlanGraph 中的一个节点                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# PlanNode 是对 TaskSpec 的运行时包装。TaskSpec 定义"做什么"，
# PlanNode 记录"做到哪了、做了什么、结果如何"。
#
# 关系：TaskSpec : PlanNode ≈ 菜谱 : 正在做的菜


@dataclass
class PlanNode:
    """PlanGraph 中的单个节点，包含运行时状态。

    Attributes:
        spec: 对应的 TaskSpec（不可变）
        status: 当前执行状态
        assigned_worker: 被分配给的 Worker ID（用于并发调度追踪）
        started_at: 开始执行的时间戳
        finished_at: 完成时间戳
        error_msg: 失败时的错误信息（给 Replanner 诊断用）
    """

    spec: TaskSpec
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    error_msg: str = ""

    @property
    def id(self) -> str:
        """快捷访问：节点 ID = TaskSpec ID。"""
        return self.spec.id

    @property
    def depends_on(self) -> list[str]:
        """快捷访问：依赖列表。"""
        return self.spec.depends_on

    @property
    def is_terminal(self) -> bool:
        """是否处于终态（成功/失败/跳过/取消）。"""
        return self.status in (
            TaskStatus.SUCCESS,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
            TaskStatus.CANCELLED,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PlanGraph — 整个任务的有向无环图                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 这是 Planner 的最终输出。包含所有子任务节点和它们之间的依赖关系。
# LangGraph 的 StateGraph 会根据这个 DAG 来决定：
#   - 哪些节点可以并行执行（无依赖关系）
#   - 哪些节点必须串行执行（有依赖关系）
#   - 哪些节点可以跳过（上游失败且不可恢复）


@dataclass
class PlanGraph:
    """任务分解后的完整有向无环图。

    Attributes:
        nodes: 所有 PlanNode（key = node.spec.id）
        edges: 邻接表，表示依赖关系（key 依赖 value 列表）
               edges["task_002"] = ["task_001"] 表示 task_002 依赖 task_001
        root_ids: 入度为 0 的节点 ID，第一批可并行执行的任务
        total_tokens_spent: 整个计划已消耗的 token 数
        created_at: 创建时间戳
    """

    nodes: dict[str, PlanNode] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)  # node_id -> [dep_ids]
    root_ids: list[str] = field(default_factory=list)
    total_tokens_spent: int = 0
    created_at: float = 0.0

    def get_ready_nodes(self) -> list[PlanNode]:
        """获取所有 READY 状态的节点（依赖已满足，等待执行）。

        Worker 调度器调用此方法来取下一批要执行的任务。
        """
        ready = []
        for node in self.nodes.values():
            if node.status != TaskStatus.READY:
                continue
            # 确认所有依赖节点都处于终态
            deps_satisfied = all(
                self.nodes[dep_id].status in (
                    TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.SKIPPED
                )
                for dep_id in node.depends_on
            )
            if deps_satisfied:
                ready.append(node)
        return ready

    def has_pending_work(self) -> bool:
        """是否还有未完成的工作。"""
        return any(
            node.status not in (
                TaskStatus.SUCCESS,
                TaskStatus.FAILED,
                TaskStatus.SKIPPED,
                TaskStatus.CANCELLED,
            )
            for node in self.nodes.values()
        )

    def success_count(self) -> int:
        """成功完成的节点数。"""
        return sum(
            1 for n in self.nodes.values()
            if n.status == TaskStatus.SUCCESS
        )

    def total_count(self) -> int:
        """总节点数。"""
        return len(self.nodes)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PlanPatch — Replanner 的局部修正                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Replanner 不会重建整个 PlanGraph，而是生成一个 PlanPatch。
# PlanPatch 只修改图中需要变的部分，类似 git diff。
#
# 四种 Patch 类型：
#   ADD     — 插入新任务（如"需要额外搜索这个关键词"）
#   REMOVE  — 删除任务（如"这条路径已证明不可行"）
#   REORDER — 调整执行顺序
#   RETRY   — 重新执行失败任务（修改查询词或参数后重试）


class PatchType(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    REORDER = "reorder"
    RETRY = "retry"


@dataclass
class PlanPatch:
    """对 PlanGraph 的局部修改。

    Attributes:
        patch_type: 修改类型
        target_node_id: 要修改的目标节点 ID
        reason: 修改原因（来自 Verifier 的反馈）
        new_spec: 仅在 ADD/RETRY 时提供新的 TaskSpec
        reorder_after: 仅在 REORDER 时指定放在哪个节点之后
    """

    patch_type: PatchType
    target_node_id: str
    reason: str
    new_spec: TaskSpec | None = None
    reorder_after: str | None = None

    def __repr__(self) -> str:
        return (
            f"PlanPatch({self.patch_type.value} {self.target_node_id}: "
            f"{self.reason[:60]}...)"
        )
