"""
Horizon-Agent 数据模型层（Schema Layer）。

所有模块依赖的数据结构在此定义。在写任何功能代码之前，
先在此冻结数据结构，确保模块间协议统一。

使用方式：
    from horizonrl.schemas import TaskSpec, StepResult, TrajectoryEvent
    from horizonrl.schemas.task import PlanGraph, PlanPatch
    from horizonrl.schemas.result import VerificationResult, EvidenceItem
    from horizonrl.schemas.event import TrajectorySession, EventType
    from horizonrl.schemas.report import FinalReport, CitationMap
"""

from horizonrl.schemas.event import (
    EventType,
    TrajectoryEvent,
    TrajectorySession,
)
from horizonrl.schemas.report import (
    CitationMap,
    FinalReport,
    ReportMetadata,
    ReportSection,
)
from horizonrl.schemas.result import (
    ActionResult,
    ErrorType,
    EvidenceItem,
    SearchProvenance,
    StepResult,
    ToolCall,
    VerificationResult,
)
from horizonrl.schemas.task import (
    PatchType,
    PlanGraph,
    PlanNode,
    PlanPatch,
    TaskPriority,
    TaskSpec,
    TaskStatus,
    UserTask,
)

__all__ = [
    # task
    "TaskPriority",
    "TaskStatus",
    "UserTask",
    "TaskSpec",
    "PlanNode",
    "PlanGraph",
    "PatchType",
    "PlanPatch",
    # result
    "ErrorType",
    "ToolCall",
    "EvidenceItem",
    "SearchProvenance",
    "StepResult",
    "VerificationResult",
    "ActionResult",
    # event
    "EventType",
    "TrajectoryEvent",
    "TrajectorySession",
    # report
    "CitationMap",
    "ReportSection",
    "ReportMetadata",
    "FinalReport",
]
