"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
执行结果与证据数据结构 —— 项目数据模型第二层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本文件定义所有"执行结果"相关的数据结构。这些是系统中流动的第二层数据。

数据流位置：
    Worker.execute(TaskSpec) → StepResult
        ├── 包含 EvidenceItem[]（搜索到的论文、代码运行输出等）
        └── 包含 ToolCall[]（这次执行用了哪些工具、输入输出是什么）
    Verifier.verify(StepResult) → VerificationResult
        └── 包含错误类型、证据缺口、恢复建议

── 数据结构一览 ──

    ToolCall         — 单次工具调用的完整记录（输入/输出/耗时/错误）
    EvidenceItem     — Worker 产出的单条证据（论文片段/代码输出/数据）
    StepResult       — Worker 执行完一个子任务后的结果
    VerificationResult — Verifier 对 StepResult 的验证结论
    ActionResult     — (兼容旧接口) 简化的执行结果

── 被哪些模块依赖 ──
    agent/worker.py     — 输出 StepResult
    agent/verifier.py   — 输入 StepResult, 输出 VerificationResult
    agent/replanner.py  — 读取 VerificationResult.feedback
    agent/writer.py     — 读取 EvidenceItem[] 合成报告
    logging/trajectory_logger.py — 序列化 StepResult, VerificationResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ─── 验证错误类型 ───────────────────────────────────────────────────────
# Verifier 不只是一个 pass/fail 开关，它必须告诉系统"哪里错了"，
# 这样 Replanner 才能对症下药。


class ErrorType(str, Enum):
    """Verifier 诊断出的错误类型。

    Replanner 会根据 ErrorType 选择不同的恢复策略：
      - EMPTY_RESULT  → 改写搜索查询
      - CODE_ERROR     → 修正代码语法
      - OFF_TOPIC      → 重新理解任务，调整方向
      - FACTUAL_ERROR  → 交叉验证，查找更可靠的来源
      - TOOL_ERROR     → 换用备用工具或重试
    """

    NONE = "none"                # 无错误
    EMPTY_RESULT = "empty_result"      # 搜索结果为空 / 无有效输出
    CODE_ERROR = "code_error"          # 代码语法错误或运行异常
    OFF_TOPIC = "off_topic"            # 输出偏离任务目标
    FACTUAL_ERROR = "factual_error"    # 事实性错误（与已知知识矛盾）
    TOOL_ERROR = "tool_error"          # 工具调用失败（超时/权限/解析失败）
    INCOMPLETE = "incomplete"          # 任务未完成，缺少必要步骤
    HALLUCINATION = "hallucination"    # 明显幻觉（编造不存在的论文/数据）
    OTHER = "other"                    # 未分类错误


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ToolCall — 单次工具调用记录                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 每一个工具调用都会被记录，用于：
#   1. 轨迹日志（追踪工具使用效率）
#   2. Token 成本统计
#   3. 调试和回归分析


@dataclass
class ToolCall:
    """单次工具调用的完整记录。

    Attributes:
        tool_name: 工具名称（如 "web_search", "arxiv_search", "code_exec"）
        input: 传给工具的输入参数
        output: 工具返回的原始输出
        elapsed: 工具调用耗时（秒）
        error: 工具调用失败时的错误信息
        tokens_used: 这次工具调用消耗的 token 数
    """

    tool_name: str
    input: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    elapsed: float = 0.0
    error: str = ""
    tokens_used: int = 0

    @property
    def is_success(self) -> bool:
        """工具调用是否成功。"""
        return self.error == ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SearchProvenance — 搜索来源可追溯信息                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class SearchProvenance:
    """搜索结果的来源追溯信息。

    记录每次搜索的 provider、query、时间等元数据，
    让最终答案中的每一条证据都可追溯到具体来源。
    """

    provider: str = ""         # bocha / brave / duckduckgo / arxiv / mock
    query: str = ""            # 实际使用的搜索 query
    timestamp: float = 0.0     # 抓取时间戳
    raw_snippet: str = ""      # 原始返回片段
    score: float = 0.0         # provider 返回的相关度评分
    url: str = ""              # 结果 URL
    is_mock: bool = False      # 是否为模拟数据


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EvidenceItem — 单条证据                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 证据是报告中可追溯的最小单元。每一条证据都必须记录来源，
# 这样 Writer 才能生成带 citation 的报告，Verifier 才能做事实核查。


@dataclass
class EvidenceItem:
    """Worker 产出的单条证据。

    Attributes:
        content: 证据内容
        source: 来源 URL
        source_type: 来源类型（"arxiv", "web", "code_output", "api"）
        relevance_score: 相关度评分（0-1）
        retrieved_at: 获取时间戳
        provider: 搜索提供商
        search_query: 实际搜索 query
        is_mock: 是否为模拟数据
        provenance: 完整来源追溯（可选）
    """

    content: str
    source: str = ""
    source_type: str = ""
    relevance_score: float = 0.0
    retrieved_at: float = 0.0
    provider: str = ""
    search_query: str = ""
    is_mock: bool = False
    provenance: SearchProvenance | None = None

    def __repr__(self) -> str:
        mock_tag = " [MOCK]" if self.is_mock else ""
        return (
            f"EvidenceItem(source={self.source[:40]}..., "
            f"relevance={self.relevance_score:.2f}{mock_tag})"
        )

    def provenance_text(self) -> str:
        """生成可读的来源追溯文本。"""
        if self.provenance:
            p = self.provenance
            provider = p.provider or self.provider or "unknown"
            tag = "Mock" if (p.is_mock or self.is_mock) else provider
            ts_str = ""
            if p.timestamp:
                import datetime
                ts_str = datetime.datetime.fromtimestamp(p.timestamp).strftime("%Y-%m-%d %H:%M")
            return (
                f"[{'Mock' if (p.is_mock or self.is_mock) else provider}"
                f"{' | ' + ts_str if ts_str else ''}]"
            )
        if self.is_mock:
            return "[Mock]"
        return f"[{self.provider or self.source_type or 'unknown'}]"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  StepResult — Worker 执行完一个子任务的结果                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 这是 Worker.execute() 的返回值。包含 LLM 响应、工具调用记录、证据列表。
# Verifier 会对 StepResult 进行验证。


@dataclass
class StepResult:
    """Worker 执行单个 TaskSpec 的结果。

    Attributes:
        task_id: 对应的 TaskSpec.id
        success: Worker 自身判断是否成功（不是 Verifier 的最终判断）
        output: LLM 生成的文本响应
        evidence: 执行过程中收集的证据列表
        tool_calls: 所有工具调用记录
        tokens_used: 本次执行消耗的总 token 数
        elapsed: 执行耗时（秒）
        error: 执行失败时的错误信息
        worker_id: 执行者 Worker ID（用于调度追踪）
    """

    task_id: str
    success: bool = False
    output: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_used: int = 0
    elapsed: float = 0.0
    error: str = ""
    worker_id: str = ""

    @property
    def tool_success_count(self) -> int:
        """成功工具调用次数。"""
        return sum(1 for tc in self.tool_calls if tc.is_success)

    @property
    def tool_total_count(self) -> int:
        """总工具调用次数。"""
        return len(self.tool_calls)

    def __repr__(self) -> str:
        status = "[OK]" if self.success else "[FAIL]"
        return (
            f"StepResult(task_id={self.task_id}, {status} "
            f"evidence={len(self.evidence)}, tools={self.tool_total_count}, "
            f"tokens={self.tokens_used}, elapsed={self.elapsed:.1f}s)"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VerificationResult — Verifier 的验证结论                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Verifier 不只返回 pass/fail。它必须给出：
#   1. 是否通过 (pass)
#   2. 质量评分 (score: 0.0-1.0)，用于过程奖励信号
#   3. 错误类型 (error_type)，告诉 Replanner 问题在哪
#   4. 诊断反馈 (feedback)，供 Replanner 和日志使用
#   5. 证据缺口 (evidence_gaps)，供 Worker 补充搜索


@dataclass
class VerificationResult:
    """Verifier 对 StepResult 的验证结论。

    Attributes:
        pass_: 是否通过验证（用 pass_ 避免与 Python 关键字冲突）
        score: 质量评分 0.0-1.0（用于 RL 过程奖励）
        error_type: 失败原因分类
        feedback: 人类可读的诊断信息，如"搜索关键词太宽泛，建议加 'attention mechanism'"
        evidence_gaps: 缺失的证据类型列表，如 ["需要性能对比数据"]
        suggested_actions: 建议 Worker 执行的补救步骤
        tokens_used: 验证过程消耗的 token 数
        elapsed: 验证耗时
    """

    pass_: bool
    score: float = 0.0
    error_type: ErrorType = ErrorType.NONE
    feedback: str = ""
    evidence_gaps: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    tokens_used: int = 0
    elapsed: float = 0.0

    @property
    def is_pass(self) -> bool:
        """是否通过验证。"""
        return self.pass_

    def __repr__(self) -> str:
        status = "[PASS]" if self.pass_ else f"[FAIL:{self.error_type.value}]"
        return (
            f"VerificationResult({status} score={self.score:.2f}, "
            f"feedback={self.feedback[:50]}...)"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ActionResult — 兼容旧接口的简化结果                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 保留此类型以兼容之前 agent/worker.py 中的 ActionResult 定义。
# 新代码应优先使用 StepResult（包含更丰富的证据和工具调用信息）。


@dataclass
class ActionResult:
    """执行子任务的简化结果（兼容旧接口）。

    新代码建议使用 StepResult 替代本类型。
    """

    task_id: str
    success: bool
    output: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    error: str | None = None
