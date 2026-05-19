"""
Writer — 研究报告合成器 (v2: 用户模式 + 开发者模式分离)。

v2 核心改进:
    - UserAnswerWriter: 生成 final_answer.md（用户友好，无调试信息）
    - DebugReportRenderer: 生成 debug_report.md（开发者视图）
    - WriterConfig: 统一配置路由
    - 修复 metadata: 不再出现 [您的姓名/代号] 或 2023 固定日期
    - 证据引用带 provenance

使用方式:
    # 用户模式（默认）
    writer = Writer(mode="llm", llm_client=client)
    final_md = await writer.write_final_answer(query, plan, results, ctx)

    # 获取两份报告
    final_md, debug_md = await writer.write_reports(...)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from horizonrl.schemas.report import ReportMetadata
from horizonrl.schemas.result import StepResult, VerificationResult
from horizonrl.schemas.task import PlanGraph

if TYPE_CHECKING:
    from horizonrl.llm.client import LLMClient
    from horizonrl.memory.hierarchical_memory import MemoryContext


# ─── 配置 ────────────────────────────────────────────────────────────────────


@dataclass
class WriterConfig:
    """Writer 运行策略配置。"""

    enable_llm_writer: bool = True
    default_author: str = "Horizon-Agent"
    include_debug_stats: bool = False
    export_dir: str = "reports"
    max_evidence_items: int = 10


# ─── 帮助函数 ────────────────────────────────────────────────────────────────


def _mock_warning(evidence_items: list) -> str:
    """基于实际 mock_ratio 生成数据来源披露。兼容 EvidenceItem 对象和 dict。"""
    if not evidence_items:
        return ""
    total = len(evidence_items)
    mock_count = sum(
        1 for e in evidence_items
        if (getattr(e, "is_mock", False) or (isinstance(e, dict) and e.get("is_mock", False)))
    )
    if mock_count == 0:
        return ""
    if mock_count == total:
        return (
            "> ⚠️ 当前为离线 Mock 模式，所有搜索结果均为模拟数据。"
            "配置 API Key 后可使用真实搜索。\n\n"
        )
    real_count = total - mock_count
    return (
        f"> ⚠️ 数据来源说明：{mock_count}/{total} 条证据为模拟数据，"
        f"{real_count}/{total} 条来自真实搜索。"
        f"部分结论可能受模拟数据影响。\n\n"
    )


def _evidence_ref_text(ev, index: int) -> str:
    """生成单条证据的可读引用文本。兼容 EvidenceItem 和 dict。"""
    def _get(key, default=""):
        if isinstance(ev, dict):
            return ev.get(key, default)
        return getattr(ev, key, default)

    provider = _get("provider") or _get("type", "unknown")
    is_mock = _get("is_mock", False)
    tag = "Mock" if is_mock else provider
    query = _get("search_query") or _get("query", "")
    source = _get("source", "")
    content = _get("content", "")[:300]

    lines = [f"[证据 {index} | provider={tag}]"]
    if query:
        lines.append(f"  query: {query}")
    if source:
        lines.append(f"  URL: {source}")
    lines.append(f"  {content}")
    return "\n".join(lines)


def _collect_evidence(results: dict[str, StepResult]) -> list[dict]:
    """从 StepResult 中提取去重后的证据列表，供 Writer 各子模块共用。

    对内容做清洗：合并换行符为空格，在句子边界智能截断，避免 markdown 渲染混乱。
    """
    seen = set()
    items = []
    for r in results.values():
        for ev in r.evidence:
            # 清洗内容：换行/制表→空格，合并多余空格
            clean = re.sub(r'[\n\r\t]+', ' ', ev.content)
            clean = re.sub(r'\s{2,}', ' ', clean).strip()
            # 智能截断：在句子边界 (。！？.!?) 或空格处断开
            if len(clean) > 280:
                chunk = clean[:300]
                # 优先找句子结束标点
                m = re.search(r'[。！？](?=[^。！？]*$)', chunk)
                if not m:
                    m = re.search(r'[.!?](?=\s|$)(?=[^.!?]*$)', chunk)
                if m:
                    clean = chunk[:m.end()] + '...'
                else:
                    # 回退：空格处截断
                    last_space = chunk.rfind(' ')
                    clean = (chunk[:last_space] if last_space > 200 else chunk[:280]) + '...'
            key = clean[:120]
            if key not in seen:
                seen.add(key)
                items.append({
                    "type": ev.source_type or ev.provider or "unknown",
                    "content": clean,
                    "source": ev.source or "",
                    "provider": ev.provider or ev.source_type or "",
                    "search_query": ev.search_query or "",
                    "is_mock": ev.is_mock,
                    "score": ev.relevance_score,
                })
    return items


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DebugReportRenderer — 开发者调试报告                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class DebugReportRenderer:
    """生成 debug_report.md —— 保留完整执行过程供开发者分析。"""

    def render(
        self,
        query: str,
        plan: PlanGraph | None,
        results: dict[str, StepResult],
        verifications: dict[str, VerificationResult],
        memory_ctx: MemoryContext | None = None,
        stats: dict | None = None,
        metadata: ReportMetadata | None = None,
    ) -> str:
        tasks = self._collect_tasks(plan, results, verifications)
        evidence = _collect_evidence(results)
        lines = []

        lines.append(f"# [DEBUG] 执行报告: {query}")
        lines.append("")
        if metadata:
            lines.append(f"Session: `{metadata.session_id}`")
            lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata.generated_at))}")
            lines.append(f"模式: {metadata.mode}")
            mock_count = sum(1 for e in evidence if e.get("is_mock", False))
            real_count = len(evidence) - mock_count
            lines.append(f"证据: {len(evidence)} 条 (真实 {real_count} / Mock {mock_count})")
            lines.append("")

        # ── 执行概要 ──
        lines.append("## 执行概要")
        lines.append("")
        success_count = sum(1 for t in tasks if t["passed"])
        if stats:
            lines.append("| 指标 | 数值 |")
            lines.append("|------|------|")
            lines.append(f"| 子任务 | {stats.get('total_count', len(tasks))} |")
            lines.append(f"| 成功 | {success_count} |")
            lines.append(f"| 轮次 | {stats.get('rounds', '?')} |")
            lines.append(f"| 工具调用 | {stats.get('total_tool_calls', '?')} |")
            lines.append(f"| 重规划 | {stats.get('total_replans', '?')} |")
            lines.append(f"| 总耗时 | {stats.get('total_elapsed', '?')} |")
        lines.append("")

        # ── 任务 DAG ──
        lines.append("## 任务 DAG")
        lines.append("")
        for t in tasks:
            icon = "✅" if t["passed"] else "❌"
            lines.append(
                f"- {icon} **{t['name']}** (id=`{t['task_id']}`) — "
                f"工具:{t['tools']}, 评分:{t['score']:.1f}, "
                f"{t['evidence_count']}证据, {t['elapsed']}s"
            )
            if t.get("feedback"):
                lines.append(f"  - 诊断: {t['feedback']}")
        lines.append("")

        # ── 验证详情 ──
        lines.append("## 验证详情")
        lines.append("")
        for t in tasks:
            icon = "PASS" if t["passed"] else f"FAIL({t.get('error_type', 'unknown')})"
            lines.append(f"- [{icon}] {t['name']}: score={t['score']:.2f}")
        lines.append("")

        # ── 证据列表 ──
        lines.append(f"## 证据列表 ({len(evidence)} 条)")
        lines.append("")
        for i, ev in enumerate(evidence):
            tag = "Mock" if ev.get("is_mock") else ev.get("type", "unknown")
            lines.append(f"{i+1}. [{tag}] {ev.get('content', '')}")
        lines.append("")

        # ── 工具调用明细 ──
        lines.append("## 工具调用")
        lines.append("")
        tool_stats: dict[str, int] = {}
        for r in results.values():
            for tc in r.tool_calls:
                name = tc.tool_name
                tool_stats[name] = tool_stats.get(name, 0) + 1
        for name, count in sorted(tool_stats.items()):
            lines.append(f"- {name}: {count} 次")
        lines.append("")

        # ── 记忆摘要 ──
        if memory_ctx and memory_ctx.summaries:
            lines.append("## 记忆摘要")
            for s in memory_ctx.summaries:
                lines.append(f"- {s}")
            lines.append("")

        lines.append("---")
        lines.append("*Debug Report — Horizon-Agent v0.1.0*")
        return "\n".join(lines)

    def _collect_tasks(self, plan, results, verifications):
        tasks = []
        if plan is None:
            return tasks
        for node in plan.nodes.values():
            r = results.get(node.spec.id)
            vr = verifications.get(node.id)
            tasks.append({
                "task_id": node.spec.id,
                "name": node.spec.name,
                "tools": ", ".join(node.spec.tool_names) or "无",
                "status": node.status.value,
                "score": vr.score if vr else 0,
                "passed": vr.pass_ if vr else False,
                "evidence_count": len(r.evidence) if r else 0,
                "elapsed": f"{r.elapsed:.1f}" if r else "0",
                "feedback": vr.feedback if vr and not vr.pass_ else "",
                "error_type": vr.error_type.value if vr else "",
            })
        return tasks

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UserAnswerWriter — 用户友好答案                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class UserAnswerWriter:
    """生成 final_answer.md —— 面向最终用户的自然语言答案。

    严格禁止输出: task_id、工具 JSON dump、Token 数、耗时、StepResult dump
    """

    def __init__(self, config: WriterConfig | None = None, llm_client: LLMClient | None = None):
        self.config = config or WriterConfig()
        self.llm = llm_client

    async def write(
        self,
        query: str,
        plan: PlanGraph | None,
        results: dict[str, StepResult],
        verifications: dict[str, VerificationResult] | None = None,
        memory_ctx: MemoryContext | None = None,
        metadata: ReportMetadata | None = None,
        on_token=None,  # async callable(str) for token streaming
    ) -> str:
        """生成 final_answer，LLM 可用时走 LLM，否则模板 fallback。

        on_token: 可选异步回调, 每个 LLM token 调用一次, 用于流式输出。
        """
        verifications = verifications or {}
        evidence = _collect_evidence(results)

        # Token 流式路径
        if on_token and self.config.enable_llm_writer and self.llm is not None:
            return await self._write_stream(query, evidence, metadata, on_token)

        # LLM 路径
        if self.config.enable_llm_writer and self.llm is not None:
            try:
                return await self._llm_write(query, evidence, metadata)
            except Exception:
                pass

        # 模板 fallback
        return self._template_write(query, evidence, metadata)

    def _build_llm_prompt(self, query: str, evidence: list[dict]) -> str:
        """构建 LLM 写作 prompt (_llm_write 和 _write_stream 共用)。"""
        evidence_text = ""
        mock_count = 0
        for i, ev in enumerate(evidence[:self.config.max_evidence_items]):
            is_mock = ev.get("is_mock", False)
            if is_mock:
                mock_count += 1
            tag = "Mock" if is_mock else ev.get("type", "web")
            evidence_text += f"[{tag}] {ev.get('content', '')[:300]}\n\n"
        mock_note = _mock_warning(evidence)
        current_date = time.strftime('%Y年%m月%d日')

        # 部分 mock 时，额外提示 LLM 注意区分
        mock_guidance = ""
        if 0 < mock_count < len(evidence[:self.config.max_evidence_items]):
            mock_guidance = (
                f"注意：{mock_count}/"
                f"{len(evidence[:self.config.max_evidence_items])} 条证据标记为 Mock 模拟数据，"
                "请优先采信标记为非 Mock 的证据，"
                "对于 Mock 证据中的信息，如果与其他来源一致可以采用，否则请谨慎对待。\n"
            )

        return f"""你是一位科技研究分析师。请根据以下检索到的证据，用流畅的中文回答用户的问题。

注意: 当前日期是 {current_date}。在讨论"最新进展"、"近期研究"等内容时，请以证据的实际内容和发布日期为准，不要凭训练数据猜测时间。

## 用户问题
{query}

## 检索到的证据
{evidence_text if evidence_text else '(未找到相关证据)'}

请按以下结构撰写答案：
1. **核心结论** — 2-3句话直接回答
2. **详细解释** — 结合证据展开说明
3. **关键要点** — 3-5个要点总结
4. **局限与说明** — 如证据不足或信息不确定，诚实说明

要求：
- 使用自然、友好、专业的中文
- 不要使用 task_id、Token、耗时等内部调试信息
- 如果证据显示是模拟数据，不要假装是真实信息
- 如果有引用证据，在段落中用 [来源] 标记

{mock_guidance}{mock_note}"""

    async def _llm_write(self, query: str, evidence: list[dict], metadata=None) -> str:
        prompt = self._build_llm_prompt(query, evidence)
        result = await self.llm.chat(
            prompt,
            system_prompt="你是一个友好、专业的科技研究助手。用流畅的中文回答用户问题。",
            temperature=0.4, max_tokens=2000,
        )
        if result.is_success and len(result.content) > 50:
            return result.content
        return self._template_write(query, evidence, metadata)

    async def _write_stream(
        self, query: str, evidence: list[dict], metadata=None, on_token=None
    ) -> str:
        """LLM 流式写作 — 逐 token 回调, 最后返回完整文本。"""
        import logging
        _log = logging.getLogger(__name__)
        prompt = self._build_llm_prompt(query, evidence)
        full_text = ""
        try:
            async for token in self.llm.chat_stream(
                prompt, system_prompt="你是一个友好、专业的科技研究助手。用流畅的中文回答用户问题。",
                temperature=0.4, max_tokens=2000,
            ):
                full_text += token
                if on_token:
                    await on_token(token)
        except Exception as e:
            _log.warning(f"LLM 流式写作失败, 回退模板: {e}")

        if len(full_text) > 50:
            return full_text
        return self._template_write(query, evidence, metadata)

    def _template_write(self, query: str, evidence: list[dict], metadata=None) -> str:
        lines = [mock_warning_str := _mock_warning(evidence)]
        if mock_warning_str.strip():
            lines.append("")

        lines.append(f"# {query}")
        lines.append("")

        # ── 核心结论 ──
        web_evidence = [e for e in evidence if e.get("type") == "web"]
        arxiv_evidence = [e for e in evidence if e.get("type") == "arxiv"]
        code_evidence = [e for e in evidence if e.get("type") == "code_output"]

        total = len(evidence)
        real_count = sum(1 for e in evidence if not e.get("is_mock", False))

        lines.append("## 核心结论")
        lines.append("")
        if total == 0:
            lines.append("未找到相关证据，建议调整搜索词或扩大搜索范围后重试。")
        else:
            real_note = f"其中 {real_count} 条来自真实数据源" if real_count > 0 else ""
            lines.append(
                f"基于 {total} 条检索结果{real_note}，以下是对该问题的分析总结。"
            )
        lines.append("")

        # ── 详细解释 ──
        if web_evidence:
            lines.append("## 网络检索发现")
            lines.append("")
            for ev in web_evidence[:5]:
                content = ev.get("content", "").strip()
                if content:
                    lines.append(f"{content}")
                    lines.append("")
            if len(web_evidence) > 5:
                lines.append(f"*...以及其他 {len(web_evidence) - 5} 条网络结果*")
                lines.append("")

        if arxiv_evidence:
            lines.append("## 学术论文发现")
            lines.append("")
            for ev in arxiv_evidence[:3]:
                content = ev.get("content", "").strip()
                if content:
                    lines.append(f"{content}")
                    lines.append("")
            if len(arxiv_evidence) > 3:
                lines.append(f"*...以及其他 {len(arxiv_evidence) - 3} 篇论文*")
                lines.append("")

        if code_evidence:
            lines.append("## 代码实验发现")
            lines.append("")
            for ev in code_evidence[:3]:
                lines.append(f"- {ev.get('content', '')[:300]}")
            lines.append("")

        # ── 关键要点 ──
        lines.append("## 关键要点")
        lines.append("")
        for i, ev in enumerate(evidence[:5]):
            content = ev.get("content", "")[:150]
            if content:
                lines.append(f"{i+1}. {content}")
        lines.append("")

        # ── 参考证据 ──
        if evidence:
            lines.append("## 参考证据")
            lines.append("")
            for i, ev in enumerate(evidence[:self.config.max_evidence_items]):
                lines.append(_evidence_ref_text(ev, i + 1))
                lines.append("")

        if metadata:
            lines.append("---")
            lines.append("*本答案由 Horizon-Agent 自动生成*")
        return "\n".join(lines)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Writer 主编排                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class Writer:
    """研究报告合成器 —— 证据 → 自然语言报告。

    v2: UserAnswerWriter + DebugReportRenderer 双模式。
    """

    def __init__(self, mode: str = "template", llm_client: LLMClient | None = None,
                 config: WriterConfig | None = None):
        self.mode = mode
        self.llm = llm_client
        self.config = config or WriterConfig(enable_llm_writer=(mode == "llm"))
        self._debug = DebugReportRenderer()
        self._user = UserAnswerWriter(self.config, llm_client)

    # ── 主编排 ──────────────────────────────────────────────────────────

    def synthesize(
        self, query: str, plan: PlanGraph | None = None,
        results: dict[str, StepResult] | None = None,
        verifications: dict[str, VerificationResult] | None = None,
        memory_ctx: MemoryContext | None = None,
    ) -> str:
        """同步合成（模板模式）。"""
        return self._user._template_write(
            query,
            _collect_evidence(results or {}),
        )

    async def synthesize_async(
        self, query: str, plan: PlanGraph | None = None,
        results: dict[str, StepResult] | None = None,
        verifications: dict[str, VerificationResult] | None = None,
        memory_ctx: MemoryContext | None = None,
    ) -> str:
        """异步合成。LLM 模式时调用 LLM，否则模板。"""
        results = results or {}
        verifications = verifications or {}
        evidence = _collect_evidence(results)
        mock_count = sum(1 for e in evidence if e.get("is_mock"))

        meta = ReportMetadata(
            author=self.config.default_author,
            mode="user",
            used_mock_data=(mock_count > 0),
            mock_ratio=mock_count / len(evidence) if evidence else 0.0,
            llm_writer_used=(self.mode == "llm" and self.llm is not None),
        )

        if self.mode == "llm" and self.llm is not None:
            return await self._user.write(query, plan, results, verifications, memory_ctx, meta)
        return self._user._template_write(query, evidence, meta)

    async def write_reports(
        self, query: str, session_id: str,
        plan: PlanGraph | None = None,
        results: dict[str, StepResult] | None = None,
        verifications: dict[str, VerificationResult] | None = None,
        memory_ctx: MemoryContext | None = None,
        stats: dict | None = None,
        on_token=None,  # async callable(str) for token streaming
    ) -> tuple[str, str]:
        """生成并保存 final_answer.md 和 debug_report.md。

        Returns:
            (final_answer_path, debug_report_path)
        """
        results = results or {}
        verifications = verifications or {}
        evidence = _collect_evidence(results)
        mock_count = sum(1 for e in evidence if e.get("is_mock"))

        meta = ReportMetadata(
            session_id=session_id,
            author=self.config.default_author,
            mode="user",
            used_mock_data=(mock_count > 0),
            mock_ratio=mock_count / len(evidence) if evidence else 0.0,
            llm_writer_used=(self.mode == "llm" and self.llm is not None),
        )

        # 生成两份报告
        final_md = await self._user.write(query, plan, results, verifications, memory_ctx, meta, on_token=on_token)
        debug_md = self._debug.render(query, plan, results, verifications, memory_ctx, stats, meta)

        # 保存文件
        export_dir = Path(self.config.export_dir) / session_id
        export_dir.mkdir(parents=True, exist_ok=True)

        final_path = export_dir / "final_answer.md"
        debug_path = export_dir / "debug_report.md"

        final_path.write_text(final_md, encoding="utf-8")
        debug_path.write_text(debug_md, encoding="utf-8")

        return str(final_path), str(debug_path)

    def render_debug_report(
        self, query: str, plan: PlanGraph | None,
        results: dict[str, StepResult],
        verifications: dict[str, VerificationResult] | None = None,
        memory_ctx: MemoryContext | None = None,
        stats: dict | None = None,
    ) -> str:
        """同步生成 debug report。"""
        return self._debug.render(query, plan, results, verifications or {}, memory_ctx, stats)
