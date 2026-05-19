"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
报告输出数据结构 —— 项目数据模型第四层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本文件定义最终报告相关的数据结构。Writer 模块读取 Memory 中的证据和上下文，
组装为结构化最终报告。

── 数据结构一览 ──

    CitationMap    — 引用映射（报告中的声明 → 证据来源）
    ReportSection  — 报告的一个章节（标题 + 内容 + 引用列表）
    FinalReport    — 最终报告的完整结构

── 被哪些模块依赖 ──
    agent/writer.py — 输出 FinalReport
    eval/metrics.py — 读取 FinalReport 计算 hallucination rate, citation support rate
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ReportMetadata — 报告元数据                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class ReportMetadata:
    """报告的生成元数据。

    Attributes:
        session_id: 会话 ID
        generated_at: 生成时间戳
        author: 作者标识
        mode: "user" | "debug"
        used_mock_data: 是否使用了模拟数据
        mock_ratio: Mock 证据占比 (0.0 ~ 1.0)
        llm_writer_used: Writer 是否使用了 LLM
    """

    session_id: str = ""
    generated_at: float = field(default_factory=_time.time)
    author: str = "Horizon-Agent"
    mode: str = "user"
    used_mock_data: bool = False
    mock_ratio: float = 0.0
    llm_writer_used: bool = False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CitationMap — 引用映射                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 将报告中的每一条声明（claim）与证据来源（EvidenceItem）关联。
# 这是计算 citation support rate 的唯一依据。


@dataclass
class CitationMap:
    """报告声明到证据来源的映射。

    Attributes:
        claim: 报告中的一段声明文本
        source_url: 支持的证据 URL
        source_content: 证据内容摘要（用于验证）
        confidence: 置信度 0.0-1.0（Writer 自己评估）
    """

    claim: str
    source_url: str = ""
    source_content: str = ""
    confidence: float = 0.0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ReportSection — 报告章节                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 每个章节包含标题、正文内容、引用列表。
# Writer 将 PlanGraph 中不同 TaskSpec 的结果编排为多个 ReportSection。


@dataclass
class ReportSection:
    """报告的一个章节。

    Attributes:
        title: 章节标题（如 "2. 方法现状"）
        content: 章节正文（markdown）
        citations: 本章节引用的证据映射
        source_task_ids: 本章节数据来源的 TaskSpec ID（可追溯）
        word_count: 字数统计
    """

    title: str
    content: str = ""
    citations: list[CitationMap] = field(default_factory=list)
    source_task_ids: list[str] = field(default_factory=list)
    word_count: int = 0

    def __repr__(self) -> str:
        return (
            f"ReportSection(title={self.title[:40]}..., "
            f"words={self.word_count}, citations={len(self.citations)})"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FinalReport — 最终报告                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Writer 的最终输出。包含完整的结构化报告，可直接渲染为 Markdown 或 JSON。


@dataclass
class FinalReport:
    """Writer 生成的最终报告。

    Attributes:
        title: 报告标题
        user_task: 原始用户任务描述
        sections: 报告章节列表
        summary: 执行摘要
        total_citations: 总引用数
        supported_claims: 有证据支持的声明数
        total_claims: 总声明数
        generation_time: 报告生成耗时（秒）
        failed_sections: 生成失败的章节标题列表
    """

    title: str = ""
    user_task: str = ""
    sections: list[ReportSection] = field(default_factory=list)
    summary: str = ""
    total_citations: int = 0
    supported_claims: int = 0
    total_claims: int = 0
    generation_time: float = 0.0
    failed_sections: list[str] = field(default_factory=list)
    metadata: ReportMetadata | None = None

    @property
    def citation_support_rate(self) -> float:
        """有证据支持的声明比例（用于 hallucination 评估）。

        Returns:
            0.0-1.0，1.0 表示所有声明都有引用支持。
            如果 total_claims 为 0，返回 0.0。
        """
        if self.total_claims == 0:
            return 0.0
        return self.supported_claims / self.total_claims

    @property
    def word_count(self) -> int:
        """报告总字数。"""
        return sum(s.word_count for s in self.sections)

    def to_markdown(self) -> str:
        """渲染为 Markdown 文本。

        Returns:
            可直接写入 .md 文件的完整报告。
        """
        lines = [
            f"# {self.title}",
            "",
            f"> 研究问题: {self.user_task}",
            "",
            "## 摘要",
            "",
            self.summary,
            "",
            "---",
        ]
        for i, section in enumerate(self.sections, 1):
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
            if section.citations:
                lines.append(f"*引用来源 ({len(section.citations)} 条):*")
                for c in section.citations:
                    lines.append(f"- [{c.source_url}]({c.source_url})")
                lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FinalReport(title={self.title[:40]}..., "
            f"sections={len(self.sections)}, "
            f"citation_support={self.citation_support_rate:.2f}, "
            f"words={self.word_count})"
        )
