"""
Hierarchical Memory — 分层记忆系统。

HorizonRL-Agent 三大核心创新之一。三层结构：
    L1 — 最近工作窗口（FIFO, token 上限, 自动溢出到 L2）
    L2 — 语义摘要（压缩后的已完成片段摘要, FIFO 淘汰）
    L3 — 经验归档（向量检索, FAISS, Phase 2+ 启用）

使用方式：
    mem = HierarchicalMemory(MemoryConfig())
    mem.record(step_result, verification_result)   # L1 写入
    ctx = mem.get_context()                         # 获取上下文供 Agent 消费
    mem.compress()                                  # 手动触发 L1→L2
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from horizonrl.config.settings import MemoryConfig
    from horizonrl.llm.client import LLMClient
    from horizonrl.schemas.result import StepResult, VerificationResult


# ─── 记忆条目 ────────────────────────────────────────────────────────────────


@dataclass
class MemoryEntry:
    """L1 中的单条记忆记录 —— 一个已完成子任务的精简快照。

    Attributes:
        task_id: TaskSpec.id
        task_name: 子任务名称
        output: 输出摘要（截断到 300 字符）
        success: 是否成功
        error_type: Verifier 判定的错误类型
        evidence_count: 收集到的证据数
        tool_calls: 工具调用次数
        tokens_used: token 消耗
        elapsed: 执行耗时
        ts: 写入时间戳
    """

    task_id: str
    task_name: str = ""
    output: str = ""
    success: bool = False
    error_type: str = ""
    evidence_count: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    elapsed: float = 0.0
    ts: float = field(default_factory=time.time)

    def estimated_tokens(self) -> int:
        """估算此条目占用的 token 数（中文按 1.5 char/token）。"""
        text = f"{self.task_name} {self.output} {self.error_type}"
        chars = len(text)
        return max(1, int(chars / 1.5))

    def to_context_string(self) -> str:
        """格式化为 Agent 可消费的上下文文本。"""
        status = "[OK]" if self.success else f"[FAIL:{self.error_type}]"
        return (
            f"{status} {self.task_name}: {self.output[:200]} "
            f"({self.evidence_count}证据, {self.tool_calls}工具, {self.elapsed:.1f}s)"
        )


# ─── L1: 最近工作窗口 ───────────────────────────────────────────────────────


class L1RecentWindow:
    """L1 最近工作窗口 —— FIFO 队列，token 上限控制。

    自动追踪总 token 数，超限时触发压缩回调。
    """

    def __init__(self, max_tokens: int = 8000, auto_compress_threshold: float = 0.8):
        self.max_tokens = max_tokens
        self.threshold = auto_compress_threshold
        self._entries: list[MemoryEntry] = []
        self._total_tokens: int = 0

    def add(self, entry: MemoryEntry) -> list[MemoryEntry]:
        """添加一条记录，返回因溢出被驱逐的条目列表（可能为空）。"""
        self._entries.append(entry)
        self._total_tokens += entry.estimated_tokens()
        return self._trim()

    def _trim(self) -> list[MemoryEntry]:
        """如果超过阈值，从头部驱逐旧条目。返回被驱逐的条目。"""
        threshold_tokens = int(self.max_tokens * self.threshold)
        overflow: list[MemoryEntry] = []
        while self._total_tokens > threshold_tokens and len(self._entries) > 1:
            removed = self._entries.pop(0)
            self._total_tokens -= removed.estimated_tokens()
            overflow.append(removed)
        return overflow

    def get_recent(self, n: int | None = None) -> list[MemoryEntry]:
        """获取最近 n 条记录（默认全部）。"""
        if n is None:
            return list(self._entries)
        return self._entries[-n:]

    def get_all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def usage_ratio(self) -> float:
        return self._total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0

    @property
    def needs_compression(self) -> bool:
        return self.usage_ratio >= self.threshold

    @property
    def success_count(self) -> int:
        return sum(1 for e in self._entries if e.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for e in self._entries if not e.success)


# ─── L2: 语义摘要 ───────────────────────────────────────────────────────────


class L2SemanticSummary:
    """L2 语义摘要 —— 压缩后的已完成片段摘要，FIFO 淘汰。

    支持两种压缩模式：
      template — 确定性模板，零成本，始终可用（默认）
      llm      — 调用轻量 LLM 生成高质量摘要
    """

    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self._summaries: list[str] = []
        self._llm_client: LLMClient | None = None

    def set_llm(self, client: LLMClient) -> None:
        self._llm_client = client

    def add(self, summary: str) -> None:
        self._summaries.append(summary)
        if len(self._summaries) > self.max_entries:
            self._summaries = self._summaries[-self.max_entries:]

    def compress_from_entries(
        self, entries: list[MemoryEntry], task_context: str = ""
    ) -> str:
        """将一组 MemoryEntry 压缩为一个语义摘要（模板模式）。"""
        if not entries:
            return ""

        success = sum(1 for e in entries if e.success)
        total = len(entries)
        names = [e.task_name for e in entries]
        key_outputs = []
        for e in entries:
            if e.output and len(e.output) > 20:
                key_outputs.append(e.output[:150])

        summary = (
            f"[片段] {' → '.join(names[:8])}"
            f"{'...' if len(names) > 8 else ''} | "
            f"成功率: {success}/{total}"
        )
        if key_outputs:
            summary += f" | 关键发现: {'; '.join(key_outputs[:3])}"
        if task_context:
            summary += f" | 上下文: {task_context[:100]}"

        self.add(summary)
        return summary

    async def compress_with_llm(
        self, entries: list[MemoryEntry], task_context: str = ""
    ) -> str:
        """LLM 驱动的语义压缩（更高质量）。"""
        if self._llm_client is None:
            return self.compress_from_entries(entries, task_context)

        prompt = self._build_compress_prompt(entries, task_context)
        result = await self._llm_client.chat(
            prompt,
            system_prompt="你是一个信息压缩器。将多条执行记录压缩为一段简洁的摘要。只输出摘要文本。",
            max_tokens=200,
        )
        if result.is_success and result.content.strip():
            summary = result.content[:300]
            self.add(summary)
        else:
            # LLM 失败则回退到模板模式（compress_from_entries 内部已 add）
            summary = self.compress_from_entries(entries, task_context)
        return summary

    def _build_compress_prompt(
        self, entries: list[MemoryEntry], task_context: str
    ) -> str:
        lines = []
        for e in entries:
            status = "OK" if e.success else f"FAIL({e.error_type})"
            lines.append(
                f"- [{status}] {e.task_name}: {e.output[:150]} "
                f"({e.evidence_count}证据, {e.tokens_used}tok)"
            )
        context_line = f"任务背景: {task_context}\n" if task_context else ""
        return (
            f"{context_line}请将以下 {len(entries)} 条执行记录压缩为一段中文摘要 "
            f"（2-3 句话，包含完成了什么、成功/失败、关键发现）:\n"
            + "\n".join(lines)
        )

    def get_recent(self, n: int = 3) -> list[str]:
        return self._summaries[-n:]

    def get_all(self) -> list[str]:
        return list(self._summaries)

    def search(self, query: str) -> list[str]:
        """简单的关键词匹配搜索。"""
        results: list[str] = []
        terms = query.lower().split()
        for s in self._summaries:
            if any(t in s.lower() for t in terms):
                results.append(s)
        return results

    def clear(self) -> None:
        self._summaries.clear()

    @property
    def count(self) -> int:
        return len(self._summaries)


# ─── 记忆上下文（Agent 消费）─────────────────────────────────────────────────


@dataclass
class MemoryContext:
    """从 Memory 中提取的上下文，供 Agent 各模块消费。

    Attributes:
        recent_steps: 最近 N 条 L1 记录
        summaries: 相关的 L2 摘要
        stats: 聚合统计
    """

    recent_steps: list[MemoryEntry] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_prompt_fragment(self, max_entries: int = 5) -> str:
        """生成为可注入 LLM prompt 的文本片段。"""
        parts: list[str] = []

        if self.stats:
            parts.append(
                f"## 执行统计\n"
                f"完成: {self.stats.get('completed', 0)}/{self.stats.get('total', 0)}, "
                f"成功率: {self.stats.get('success_rate', 0):.0%}, "
                f"总 token: {self.stats.get('total_tokens', 0)}, "
                f"重规划: {self.stats.get('replans', 0)}次"
            )

        if self.summaries:
            parts.append(
                "## 历史摘要\n" + "\n".join(f"- {s}" for s in self.summaries[-3:])
            )

        if self.recent_steps:
            steps_text = "\n".join(
                s.to_context_string() for s in self.recent_steps[-max_entries:]
            )
            parts.append(f"## 最近步骤\n{steps_text}")

        return "\n\n".join(parts)

    def is_empty(self) -> bool:
        return not self.recent_steps and not self.summaries


# ─── 主编排类 ────────────────────────────────────────────────────────────────


class HierarchicalMemory:
    """三层分层记忆系统。

    Examples:
        >>> mem = HierarchicalMemory(MemoryConfig())
        >>> mem.record(step_result, verification)  # 写入 L1
        >>> ctx = mem.get_context()                 # 获取上下文
        >>> prompt = ctx.to_prompt_fragment()       # 注入 LLM prompt
    """

    def __init__(self, config: MemoryConfig | None = None):
        from horizonrl.config.settings import MemoryConfig as MC

        self.config = config or MC()

        self.l1 = L1RecentWindow(
            max_tokens=self.config.l1_max_tokens,
            auto_compress_threshold=self.config.auto_compress_threshold,
        )
        self.l2 = L2SemanticSummary(
            max_entries=self.config.l2_max_entries,
        )

        # L3 为 Phase 2+ 预留
        self._l3_entries: list[dict] = []
        self._replan_count: int = 0

    # ── 写入 ──────────────────────────────────────────────────────────────

    def record(
        self,
        result: StepResult,
        verification: VerificationResult | None = None,
    ) -> MemoryEntry:
        """记录一条执行结果到 L1。

        Args:
            result: Worker 产出的 StepResult。
            verification: Verifier 的验证结论（可选）。

        Returns:
            创建的 MemoryEntry。
        """

        entry = MemoryEntry(
            task_id=result.task_id,
            task_name="",  # 由调用方通过 result 推断
            output=result.output[:300] if result.output else "",
            success=result.success,
            error_type=verification.error_type.value if verification else "",
            evidence_count=len(result.evidence),
            tool_calls=len(result.tool_calls),
            tokens_used=result.tokens_used,
            elapsed=result.elapsed,
        )

        # 尝试从 task_id 推断名称
        entry.task_name = result.task_id

        overflow = self.l1.add(entry)

        # 溢出条目自动压缩到 L2
        if overflow:
            self.l2.compress_from_entries(overflow)

        return entry

    def record_task(
        self,
        task_id: str,
        task_name: str,
        output: str,
        success: bool,
        error_type: str = "",
        evidence_count: int = 0,
        tool_calls: int = 0,
        tokens_used: int = 0,
        elapsed: float = 0.0,
    ) -> MemoryEntry:
        """简化的记录接口 —— 不需要完整的 StepResult。

        适用于不需要完整 Schema 的场景。
        """
        entry = MemoryEntry(
            task_id=task_id,
            task_name=task_name,
            output=output[:300],
            success=success,
            error_type=error_type,
            evidence_count=evidence_count,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            elapsed=elapsed,
        )
        overflow = self.l1.add(entry)
        if overflow:
            self.l2.compress_from_entries(overflow)
        return entry

    def record_replan(self) -> None:
        """记录一次重规划事件。"""
        self._replan_count += 1

    # ── 压缩 ──────────────────────────────────────────────────────────────

    def compress(self, task_context: str = "") -> str:
        """手动触发 L1→L2 压缩。将当前 L1 全部内容压缩为一个摘要。

        Args:
            task_context: 可选的任务上下文，提高摘要质量。

        Returns:
            生成的 L2 摘要字符串。
        """
        entries = self.l1.get_all()
        if not entries:
            return ""
        summary = self.l2.compress_from_entries(entries, task_context)
        self.l1.clear()
        return summary

    async def compress_with_llm(self, task_context: str = "") -> str:
        """LLM 驱动的 L1→L2 压缩（更高质量）。"""
        entries = self.l1.get_all()
        if not entries:
            return ""
        summary = await self.l2.compress_with_llm(entries, task_context)
        self.l1.clear()
        return summary

    def auto_compress(self, task_context: str = "") -> str:
        """如果 L1 超过阈值则自动压缩。调用方应在每个 step 后调用。"""
        if self.l1.needs_compression:
            return self.compress(task_context)
        return ""

    # ── 检索 ──────────────────────────────────────────────────────────────

    def get_context(
        self,
        query: str | None = None,
        recent_n: int = 5,
        summary_n: int = 3,
    ) -> MemoryContext:
        """获取当前记忆上下文。

        Args:
            query: 可选检索查询（用于 L2 语义搜索）。
            recent_n: L1 返回最近几条。
            summary_n: L2 返回几条摘要。

        Returns:
            MemoryContext 供 Agent 消费。
        """
        recent = self.l1.get_recent(recent_n)

        if query:
            summaries = self.l2.search(query)[-summary_n:]
        else:
            summaries = self.l2.get_recent(summary_n)

        stats = {
            "completed": self.l1.count,
            "total": self.l1.count,
            "success_count": self.l1.success_count,
            "failure_count": self.l1.failure_count,
            "success_rate": (
                self.l1.success_count / max(self.l1.count, 1)
            ),
            "total_tokens": sum(e.tokens_used for e in self.l1.get_all()),
            "l2_count": self.l2.count,
            "replans": self._replan_count,
        }

        return MemoryContext(
            recent_steps=recent,
            summaries=summaries,
            stats=stats,
        )

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """跨层检索相关记忆。

        L1 最近记录总是包含，L2 做关键词匹配。
        """
        results: list[str] = []

        # L1: 最近 5 条
        for e in self.l1.get_recent(5):
            results.append(f"[L1] {e.to_context_string()}")

        # L2: 关键词匹配
        l2_matches = self.l2.search(query)[:top_k]
        for s in l2_matches:
            results.append(f"[L2] {s[:200]}")

        return results[:top_k]

    # ── L3 接口（Phase 2+）───────────────────────────────────────────────

    def archive_to_l3(self, text: str, metadata: dict | None = None) -> None:
        """归档到 L3 经验记忆（当前为占位实现）。"""
        self._l3_entries.append({
            "text": text,
            "metadata": metadata or {},
            "ts": time.time(),
        })

    def retrieve_l3(self, query: str, top_k: int = 5) -> list[str]:
        """从 L3 检索（当前为关键词匹配占位）。"""
        results: list[str] = []
        terms = query.lower().split()
        for entry in self._l3_entries:
            text = entry.get("text", "")
            if any(t in text.lower() for t in terms):
                results.append(f"[L3] {text[:200]}")
        return results[:top_k]

    # ── 统计与状态 ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """获取记忆系统统计信息。"""
        return {
            "l1_count": self.l1.count,
            "l1_tokens": self.l1.total_tokens,
            "l1_max_tokens": self.l1.max_tokens,
            "l1_usage": f"{self.l1.usage_ratio:.1%}",
            "l1_needs_compression": self.l1.needs_compression,
            "l2_count": self.l2.count,
            "l2_max": self.l2.max_entries,
            "l3_count": len(self._l3_entries),
            "replan_count": self._replan_count,
        }

    def clear(self) -> None:
        """清空所有记忆。"""
        self.l1.clear()
        self.l2.clear()
        self._l3_entries.clear()
        self._replan_count = 0

    def set_llm(self, client: LLMClient) -> None:
        """注入 LLM 客户端以启用 LLM 压缩模式。"""
        self.l2.set_llm(client)
