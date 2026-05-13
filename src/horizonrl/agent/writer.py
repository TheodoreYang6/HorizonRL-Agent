"""
Writer — 研究报告合成器。

将 Worker 产出的证据、Verifier 的验证、Memory 的摘要合成为
结构化的自然语言研究报告。这是 HorizonRL-Agent 的输出层。

两种模式：
    template — 确定性模板，按证据类型组织，无需 LLM（默认）
    llm      — LLM 深度合成，流畅通顺，更像人类写的报告

使用方式：
    writer = Writer(mode="template")
    report = writer.synthesize(query, plan, results, memory_context)

    # LLM 模式
    writer = Writer(mode="llm", llm_client=client)
    report = await writer.synthesize_async(query, plan, results, ctx)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from horizonrl.schemas.result import StepResult, VerificationResult
from horizonrl.schemas.task import PlanGraph

if TYPE_CHECKING:
    from horizonrl.llm.client import LLMClient
    from horizonrl.memory.hierarchical_memory import MemoryContext


class Writer:
    """研究报告合成器 —— 证据 → 自然语言报告。

    Examples:
        >>> writer = Writer()
        >>> report = writer.synthesize("Transformer 注意力机制", plan, results, ctx)
        >>> print(report[:200])
    """

    def __init__(self, mode: str = "template", llm_client: LLMClient | None = None):
        self.mode = mode
        self.llm = llm_client

    # ── 主编排 ──────────────────────────────────────────────────────────

    def synthesize(
        self,
        query: str,
        plan: PlanGraph | None = None,
        results: dict[str, StepResult] | None = None,
        verifications: dict[str, VerificationResult] | None = None,
        memory_ctx: MemoryContext | None = None,
    ) -> str:
        """合成研究报告（同步，模板模式）。

        Args:
            query: 用户的研究问题。
            plan: 执行过的 PlanGraph（可选）。
            results: {task_id: StepResult}（可选）。
            verifications: {task_id: VerificationResult}（可选）。
            memory_ctx: MemoryContext 摘要（可选）。

        Returns:
            完整的 Markdown 格式研究报告。
        """
        results = results or {}
        verifications = verifications or {}

        evidence = self._collect_evidence(results)
        tasks = self._collect_tasks(plan, results, verifications)

        return self._build_template_report(query, evidence, tasks, memory_ctx)

    async def synthesize_async(
        self,
        query: str,
        plan: PlanGraph | None = None,
        results: dict[str, StepResult] | None = None,
        verifications: dict[str, VerificationResult] | None = None,
        memory_ctx: MemoryContext | None = None,
    ) -> str:
        """合成研究报告（异步，LLM 模式）。

        若 LLM 不可用则自动回退到模板模式。
        """
        if self.mode == "llm" and self.llm is not None:
            try:
                return await self._llm_synthesize(
                    query, plan, results, verifications, memory_ctx
                )
            except Exception:
                pass  # 回退到模板

        return self.synthesize(query, plan, results, verifications, memory_ctx)

    # ── 证据收集 ────────────────────────────────────────────────────────

    def _collect_evidence(
        self, results: dict[str, StepResult]
    ) -> list[dict]:
        """从所有 StepResult 中收集并去重证据。"""
        seen: set[str] = set()
        items: list[dict] = []
        for r in results.values():
            for ev in r.evidence:
                key = ev.content[:100]
                if key not in seen:
                    seen.add(key)
                    items.append({
                        "type": ev.source_type or "unknown",
                        "source": ev.source or "",
                        "content": ev.content,
                    })
        return items

    def _collect_tasks(
        self,
        plan: PlanGraph | None,
        results: dict[str, StepResult],
        verifications: dict[str, VerificationResult],
    ) -> list[dict]:
        """收集任务执行信息。"""
        tasks: list[dict] = []
        if plan is None:
            return tasks
        for node in plan.nodes.values():
            r = results.get(node.spec.id)
            vr = verifications.get(node.id)
            tasks.append({
                "name": node.spec.name,
                "status": node.status.value,
                "tools": ", ".join(node.spec.tool_names) or "无",
                "output": r.output[:300] if r else "",
                "evidence_count": len(r.evidence) if r else 0,
                "score": vr.score if vr else 0,
                "passed": vr.pass_ if vr else False,
                "feedback": vr.feedback if vr and not vr.pass_ else "",
            })
        return tasks

    # ── 模板报告生成 ────────────────────────────────────────────────────

    def _build_template_report(
        self,
        query: str,
        evidence: list[dict],
        tasks: list[dict],
        memory_ctx: MemoryContext | None = None,
    ) -> str:
        """用确定性模板生成自然语言报告。"""
        lines: list[str] = []

        # ── 标题与概述 ──
        lines.append(f"# {query}")
        lines.append("")

        success_count = sum(1 for t in tasks if t["passed"])
        total = len(tasks)
        lines.append(
            f"本研究通过多 Agent 协作完成，共执行 {total} 个子任务，"
            f"其中 {success_count} 个通过验证，"
            f"收集到 {len(evidence)} 条证据。"
        )
        lines.append("")

        # ── 证据分组 ──
        web_evidence = [e for e in evidence if e["type"] == "web"]
        arxiv_evidence = [e for e in evidence if e["type"] == "arxiv"]
        code_evidence = [e for e in evidence if e["type"] == "code_output"]
        other_evidence = [e for e in evidence if e["type"] not in ("web", "arxiv", "code_output")]

        # ── 背景与概述 ──
        lines.append("## 一、研究概述")
        lines.append("")

        # 从任务名称推断研究阶段
        task_names = [t["name"] for t in tasks]
        phases = []
        if any("背景" in n for n in task_names):
            phases.append("背景调研")
        if any("进展" in n or "最新" in n for n in task_names):
            phases.append("前沿追踪")
        if any("对比" in n or "分析" in n for n in task_names):
            phases.append("方法分析")
        if any("局限" in n for n in task_names):
            phases.append("局限性评估")
        if any("汇总" in n or "综合" in n for n in task_names):
            phases.append("综合总结")

        if phases:
            lines.append(f"研究按以下阶段展开：{' → '.join(phases)}。")
        lines.append("")

        # ── 网络搜索发现 ──
        if web_evidence:
            lines.append("## 二、网络调研发现")
            lines.append("")
            for i, ev in enumerate(web_evidence[:8]):
                content = ev["content"].strip()
                # 让内容读起来更自然
                if content:
                    lines.append(f"{content}")
                    lines.append("")
            if len(web_evidence) > 8:
                lines.append(f"*...以及其他 {len(web_evidence) - 8} 条网络搜索结果*")
                lines.append("")

        # ── 学术论文发现 ──
        if arxiv_evidence:
            lines.append("## 三、学术论文调研")
            lines.append("")
            for i, ev in enumerate(arxiv_evidence[:5]):
                content = ev["content"].strip()
                if content:
                    lines.append(f"{content}")
                    lines.append("")
            if len(arxiv_evidence) > 5:
                lines.append(f"*...以及其他 {len(arxiv_evidence) - 5} 篇相关论文*")
                lines.append("")

        # ── 代码实验 ──
        if code_evidence:
            lines.append("## 四、代码实验结果")
            lines.append("")
            for i, ev in enumerate(code_evidence[:5]):
                content = ev["content"].strip()
                if content:
                    lines.append(f"- {content}")
            lines.append("")

        # ── 其他来源 ──
        if other_evidence:
            lines.append("## 五、补充发现")
            lines.append("")
            for ev in other_evidence[:5]:
                content = ev["content"].strip()
                if content:
                    lines.append(f"- {content}")
            lines.append("")

        # ── 执行质量评估 ──
        if tasks:
            lines.append("## 六、执行质量")
            lines.append("")
            avg_score = sum(t["score"] for t in tasks) / max(len(tasks), 1)
            lines.append(f"各子任务验证平均得分: {avg_score:.2f}。")
            lines.append("")

            failed_tasks = [t for t in tasks if not t["passed"]]
            if failed_tasks:
                lines.append("以下子任务未通过验证：")
                for t in failed_tasks:
                    lines.append(f"- **{t['name']}**: {t['feedback']}")
                lines.append("")

        # ── 记忆摘要 ──
        if memory_ctx and memory_ctx.summaries:
            lines.append("## 七、研究总结")
            lines.append("")
            for s in memory_ctx.summaries:
                lines.append(f"> {s}")
                lines.append("")

        # ── 证据来源统计 ──
        lines.append("## 八、证据来源")
        lines.append("")
        web_count = len(web_evidence)
        arxiv_count = len(arxiv_evidence)
        code_count = len(code_evidence)
        other_count = len(other_evidence)
        lines.append("| 来源 | 数量 |")
        lines.append("|------|------|")
        if web_count:
            lines.append(f"| 网络搜索 | {web_count} |")
        if arxiv_count:
            lines.append(f"| 学术论文 | {arxiv_count} |")
        if code_count:
            lines.append(f"| 代码执行 | {code_count} |")
        if other_count:
            lines.append(f"| 其他 | {other_count} |")
        lines.append(f"| **总计** | **{len(evidence)}** |")
        lines.append("")

        lines.append("---")
        lines.append("*本报告由 HorizonRL-Agent v0.1.0 自动生成*")

        return "\n".join(lines)

    # ── LLM 报告生成 ────────────────────────────────────────────────────

    async def _llm_synthesize(
        self,
        query: str,
        plan: PlanGraph | None,
        results: dict[str, StepResult] | None,
        verifications: dict[str, VerificationResult] | None,
        memory_ctx: MemoryContext | None = None,
    ) -> str:
        """LLM 驱动的深度研究报告合成。"""
        results = results or {}
        verifications = verifications or {}

        evidence = self._collect_evidence(results)
        evidence_text = ""
        for i, ev in enumerate(evidence[:12]):
            evidence_text += f"[{ev['type']}] {ev['content'][:300]}\n\n"

        tasks = self._collect_tasks(plan, results, verifications)
        tasks_text = ""
        for t in tasks:
            status = "通过" if t["passed"] else f"未通过({t['feedback']})"
            tasks_text += f"- {t['name']}: {status}, 评分{t['score']:.1f}\n"

        total_tasks = len(tasks)
        passed_tasks = sum(1 for t in tasks if t["passed"])

        prompt = f"""你是一位资深研究分析师。请根据以下执行过程和收集到的证据，撰写一份专业的研究报告。

## 研究问题
{query}

## 执行过程
{total_tasks} 个子任务并行执行，{passed_tasks} 个通过验证。

{tasks_text}

## 收集到的证据
{evidence_text if evidence_text else '(无证据)'}

请撰写一份结构清晰的中文研究报告（Markdown 格式），要求：
1. **研究摘要** — 2-3句话概括核心发现
2. **背景** — 问题的背景和意义
3. **核心发现** — 按主题组织，融入具体证据
4. **方法分析** — 如果有不同方法，做对比
5. **当前局限** — 目前方法的不足之处
6. **结论与展望** — 总结和未来方向
7. **来源说明** — 列出证据来源类型和数量

报告要读起来像人类写的，不要像数据转储。使用流畅的中文。"""

        result = await self.llm.chat(
            prompt,
            system_prompt="你是一位资深的科技研究分析师，擅长将碎片化证据合成为流畅的研究报告。",
            temperature=0.4,
            max_tokens=2000,
        )

        if result.is_success and len(result.content) > 50:
            return result.content
        else:
            # LLM 失败，回退模板
            return self._build_template_report(query, evidence, tasks, memory_ctx)
