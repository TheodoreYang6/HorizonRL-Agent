"""
Replanner — 局部重规划。

当 Verifier 判定某个子任务失败后，Replanner 生成 PlanPatch 对 PlanGraph 做局部修复，
而不是全局重建。这是 HorizonRL-Agent 的核心创新之一。

策略映射（ErrorType → PatchType）：
    EMPTY_RESULT  → RETRY（改写查询词，扩大搜索范围）
    CODE_ERROR    → RETRY（修正代码语法，简化输入）
    TOOL_ERROR    → RETRY（切换备用工具或检查配置）
    OFF_TOPIC     → RETRY（重写任务描述，聚焦核心目标）
    INCOMPLETE    → ADD（补充子任务填补证据缺口）
    HALLUCINATION → RETRY（严格指令重试）
    FACTUAL_ERROR → RETRY（交叉验证多个来源）

防无限循环：
    - 单个任务最多重试 max_retries_per_task 次（默认 3）
    - 单次运行最多触发 max_total_replans 次重规划（默认 5）

使用方式：
    replanner = Replanner()
    patch = replanner.replan(verification_result, plan_graph, "task_003")
    if patch:
        replanner.apply_patch(plan_graph, patch)
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from horizonrl.schemas.result import (
    ErrorType,
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
)

if TYPE_CHECKING:
    from horizonrl.llm.client import LLMClient


# ─── 策略映射 ────────────────────────────────────────────────────────────────
# ErrorType → (PatchType, 策略提示)

ERROR_STRATEGY: dict[ErrorType, tuple[PatchType, str]] = {
    ErrorType.EMPTY_RESULT: (PatchType.RETRY, "扩大搜索范围，使用更通用的关键词"),
    ErrorType.CODE_ERROR: (PatchType.RETRY, "修正代码语法错误，简化输入用例"),
    ErrorType.TOOL_ERROR: (PatchType.RETRY, "切换备用工具或检查工具配置后重试"),
    ErrorType.OFF_TOPIC: (PatchType.RETRY, "重写任务描述，聚焦核心目标"),
    ErrorType.INCOMPLETE: (PatchType.ADD, "补充子任务以填补证据缺口"),
    ErrorType.FACTUAL_ERROR: (PatchType.RETRY, "交叉验证多个来源，优先使用权威数据"),
    ErrorType.HALLUCINATION: (PatchType.RETRY, "使用更严格的指令和约束重新执行"),
    ErrorType.OTHER: (PatchType.RETRY, "通用重试"),
    ErrorType.NONE: (PatchType.RETRY, "验证通过但分数偏低，优化输出质量"),
}


class Replanner:
    """规则驱动的局部重规划器。

    根据 VerificationResult 的 error_type 自动选择修复策略，
    生成 PlanPatch 供调度器应用到 PlanGraph。

    Examples:
        >>> replanner = Replanner(max_retries_per_task=3, max_total_replans=5)
        >>> patch = replanner.replan(vr, plan_graph, "task_003")
        >>> if patch:
        ...     replanner.apply_patch(plan_graph, patch)
    """

    def __init__(
        self,
        max_retries_per_task: int = 3,
        max_total_replans: int = 5,
    ):
        self.max_retries_per_task = max_retries_per_task
        self.max_total_replans = max_total_replans
        self._retry_counts: dict[str, int] = {}
        self._total_replans: int = 0

    # ── 重试控制 ──────────────────────────────────────────────────────────

    def should_replan(self, vr: VerificationResult, task_id: str) -> bool:
        """判断是否应该触发重规划。

        Args:
            vr: 验证结果。
            task_id: 失败的任务 ID。

        Returns:
            True 如果可以重试，False 如果已达上限或验证通过。
        """
        if vr.pass_:
            return False
        if self._total_replans >= self.max_total_replans:
            return False
        if self._retry_counts.get(task_id, 0) >= self.max_retries_per_task:
            return False
        return True

    def can_retry(self, task_id: str) -> bool:
        """检查某个任务是否还能重试。"""
        return self._retry_counts.get(task_id, 0) < self.max_retries_per_task

    def get_retry_count(self, task_id: str) -> int:
        return self._retry_counts.get(task_id, 0)

    @property
    def total_replans(self) -> int:
        return self._total_replans

    def reset(self) -> None:
        """重置所有计数器（用于新的规划周期）。"""
        self._retry_counts.clear()
        self._total_replans = 0

    # ── 主入口 ─────────────────────────────────────────────────────────────

    def replan(
        self,
        vr: VerificationResult,
        plan_graph: PlanGraph,
        failed_node_id: str,
    ) -> PlanPatch | None:
        """根据验证结果生成修复补丁。

        Args:
            vr: Verifier 的验证结论。
            plan_graph: 当前 PlanGraph。
            failed_node_id: 失败节点 ID。

        Returns:
            PlanPatch 或 None（不应重规划时）。
        """
        if not self.should_replan(vr, failed_node_id):
            return None

        self._retry_counts[failed_node_id] = (
            self._retry_counts.get(failed_node_id, 0) + 1
        )
        self._total_replans += 1

        node = plan_graph.nodes.get(failed_node_id)
        if node is None:
            return None

        patch_type, strategy_hint = ERROR_STRATEGY.get(
            vr.error_type,
            (PatchType.RETRY, "通用重试"),
        )

        reason = self._build_reason(vr, strategy_hint)

        new_spec = None
        if patch_type == PatchType.RETRY:
            new_spec = self._build_retry_spec(node.spec, vr)
        elif patch_type == PatchType.ADD:
            new_spec = self._build_add_spec(node.spec, vr)

        return PlanPatch(
            patch_type=patch_type,
            target_node_id=failed_node_id,
            reason=reason,
            new_spec=new_spec,
        )

    def _build_reason(self, vr: VerificationResult, strategy_hint: str) -> str:
        """构建人类可读的修改原因。"""
        parts = [f"验证失败 [{vr.error_type.value}]: {vr.feedback}"]
        if vr.suggested_actions:
            parts.append(f"建议: {'; '.join(vr.suggested_actions[:3])}")
        parts.append(f"策略: {strategy_hint}")
        return " | ".join(parts)

    def _build_retry_spec(self, original: TaskSpec, vr: VerificationResult) -> TaskSpec:
        """为 RETRY 构建修正后的 TaskSpec（保持相同 id）。"""
        new_desc = original.description
        new_tools = list(original.tool_names)
        new_context = original.context

        if vr.error_type == ErrorType.EMPTY_RESULT:
            new_desc = f"[重试] {original.description} — 使用更广泛的关键词"
        elif vr.error_type == ErrorType.TOOL_ERROR:
            new_context = (
                f"{original.context}\n[重试] 前次工具失败({vr.feedback})，"
                f"尝试不同参数或备用工具"
            ).strip()
        elif vr.error_type == ErrorType.OFF_TOPIC:
            new_desc = f"[聚焦] {original.description}"
        elif vr.error_type == ErrorType.HALLUCINATION:
            new_desc = f"[严格] {original.description} — 仅使用已验证事实"
        elif vr.error_type == ErrorType.FACTUAL_ERROR:
            new_desc = f"[交叉验证] {original.description}"
            new_context = (
                f"{original.context}\n[重试] 优先使用权威来源，标注置信度"
            ).strip()

        if vr.suggested_actions:
            new_context = (
                f"{new_context}\n[Verifier建议] {'; '.join(vr.suggested_actions[:3])}"
            ).strip()

        if vr.evidence_gaps:
            new_context = (
                f"{new_context}\n[缺失证据] {'; '.join(vr.evidence_gaps[:3])}"
            ).strip()

        return TaskSpec(
            id=original.id,  # 保持相同 id，原地更新
            name=original.name,
            description=new_desc,
            tool_names=new_tools,
            depends_on=list(original.depends_on),
            priority=original.priority,
            context=new_context,
            retry_count=original.retry_count + 1,
            max_retries=original.max_retries,
        )

    def _build_add_spec(self, original: TaskSpec, vr: VerificationResult) -> TaskSpec:
        """为 ADD 构建新的补充 TaskSpec。"""
        new_id = f"{original.id}_sup_{uuid.uuid4().hex[:6]}"
        desc = f"[补充] 针对 '{original.name}' 的缺失证据: "
        desc += "; ".join(vr.evidence_gaps[:3]) if vr.evidence_gaps else "补充搜索"

        context = f"原任务: {original.description}\n"
        context += f"失败原因: {vr.feedback}\n"
        if vr.suggested_actions:
            context += f"建议动作: {'; '.join(vr.suggested_actions[:3])}"

        return TaskSpec(
            id=new_id,
            name=f"[补充] {original.name}",
            description=desc,
            tool_names=list(original.tool_names),
            depends_on=[],  # 补充任务独立执行，不依赖失败父任务
            priority=TaskPriority.P1,
            context=context,
            retry_count=0,
            max_retries=original.max_retries,
        )

    # ── Patch 应用 ─────────────────────────────────────────────────────────

    def apply_patch(self, graph: PlanGraph, patch: PlanPatch) -> PlanGraph:
        """将 PlanPatch 应用到 PlanGraph（原地修改）。

        Args:
            graph: 要修改的 PlanGraph。
            patch: 要应用的补丁。

        Returns:
            修改后的 PlanGraph（与输入同一对象）。
        """
        if patch.patch_type == PatchType.RETRY:
            self._apply_retry(graph, patch)
        elif patch.patch_type == PatchType.ADD:
            self._apply_add(graph, patch)
        elif patch.patch_type == PatchType.REMOVE:
            self._apply_remove(graph, patch)
        elif patch.patch_type == PatchType.REORDER:
            self._apply_reorder(graph, patch)
        return graph

    def _apply_retry(self, graph: PlanGraph, patch: PlanPatch) -> None:
        """RETRY: 更新节点 spec，重置为 PENDING。"""
        node = graph.nodes.get(patch.target_node_id)
        if node is None:
            return
        if patch.new_spec:
            node.spec = patch.new_spec
        node.status = TaskStatus.PENDING
        node.error_msg = ""

    def _apply_add(self, graph: PlanGraph, patch: PlanPatch) -> None:
        """ADD: 添加独立补充任务，不修改现有依赖。"""
        if patch.new_spec is None:
            return
        new_node = PlanNode(spec=patch.new_spec, status=TaskStatus.PENDING)
        graph.nodes[patch.new_spec.id] = new_node
        graph.edges[patch.new_spec.id] = list(patch.new_spec.depends_on)
        if not patch.new_spec.depends_on:
            graph.root_ids.append(patch.new_spec.id)

    def _apply_remove(self, graph: PlanGraph, patch: PlanPatch) -> None:
        """REMOVE: 将节点标记为 SKIPPED。"""
        node = graph.nodes.get(patch.target_node_id)
        if node is None:
            return
        node.status = TaskStatus.SKIPPED
        node.error_msg = patch.reason

    def _apply_reorder(self, graph: PlanGraph, patch: PlanPatch) -> None:
        """REORDER: 提升节点优先级为 P0。"""
        node = graph.nodes.get(patch.target_node_id)
        if node is None:
            return
        node.spec.priority = TaskPriority.P0

    # ── 批量操作 ───────────────────────────────────────────────────────────

    def diagnose_all(
        self,
        verification_results: dict[str, VerificationResult],
        graph: PlanGraph,
    ) -> list[PlanPatch]:
        """对所有失败结果批量生成补丁。

        Args:
            verification_results: {node_id: VerificationResult}。
            graph: 当前 PlanGraph。

        Returns:
            生成的 PlanPatch 列表（已排除不可重试的）。
        """
        patches: list[PlanPatch] = []
        for node_id, vr in verification_results.items():
            patch = self.replan(vr, graph, node_id)
            if patch is not None:
                patches.append(patch)
        return patches


# ─── LLM 增强重规划器 ────────────────────────────────────────────────────────


class LLMReplanner(Replanner):
    """LLM 增强的重规划器 —— 用 LLM 生成更智能的查询改写和任务描述。

    继承 Replanner 的所有规则逻辑，但在生成 new_spec 时调用 LLM
    来优化查询词和任务描述，处理规则难以覆盖的边缘情况。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_retries_per_task: int = 3,
        max_total_replans: int = 5,
    ):
        super().__init__(max_retries_per_task, max_total_replans)
        self.llm = llm_client

    async def replan_async(
        self,
        vr: VerificationResult,
        plan_graph: PlanGraph,
        failed_node_id: str,
    ) -> PlanPatch | None:
        """LLM 增强的异步重规划。"""
        if not self.should_replan(vr, failed_node_id):
            return None

        self._retry_counts[failed_node_id] = (
            self._retry_counts.get(failed_node_id, 0) + 1
        )
        self._total_replans += 1

        node = plan_graph.nodes.get(failed_node_id)
        if node is None:
            return None

        patch_type, strategy_hint = ERROR_STRATEGY.get(
            vr.error_type,
            (PatchType.RETRY, "通用重试"),
        )

        reason = self._build_reason(vr, strategy_hint)
        new_spec = None

        if patch_type in (PatchType.RETRY, PatchType.ADD):
            try:
                new_spec = await self._llm_build_spec(node.spec, vr, patch_type)
            except Exception:
                # LLM 失败则回退到规则生成
                if patch_type == PatchType.RETRY:
                    new_spec = self._build_retry_spec(node.spec, vr)
                else:
                    new_spec = self._build_add_spec(node.spec, vr)

        return PlanPatch(
            patch_type=patch_type,
            target_node_id=failed_node_id,
            reason=reason,
            new_spec=new_spec,
        )

    # 同步兼容接口
    async def replan(self, vr, plan_graph, failed_node_id):
        """同步兼容接口，内部调用异步。"""
        return await self.replan_async(vr, plan_graph, failed_node_id)

    async def _llm_build_spec(
        self,
        original: TaskSpec,
        vr: VerificationResult,
        patch_type: PatchType,
    ) -> TaskSpec:
        """调用 LLM 生成修正后的 TaskSpec。"""
        prompt = self._build_llm_prompt(original, vr, patch_type)
        result = await self.llm.chat(
            prompt,
            system_prompt=(
                "你是一个任务修复专家。根据失败原因改写子任务的描述、搜索查询和工具选择。"
                "只输出 JSON，不输出其他内容。"
            ),
            temperature=0.2,
            max_tokens=300,
        )

        if not result.is_success:
            raise RuntimeError(f"LLM 调用失败: {result.error}")

        return self._parse_llm_spec(original, result.content, patch_type)

    def _build_llm_prompt(
        self,
        original: TaskSpec,
        vr: VerificationResult,
        patch_type: PatchType,
    ) -> str:
        action = "重试" if patch_type == PatchType.RETRY else "补充"
        return f"""原任务:
  名称: {original.name}
  描述: {original.description}
  工具: {', '.join(original.tool_names) or '无'}
  上下文: {original.context or '无'}

验证失败:
  错误类型: {vr.error_type.value}
  诊断: {vr.feedback}
  证据缺口: {', '.join(vr.evidence_gaps) if vr.evidence_gaps else '无'}
  建议动作: {', '.join(vr.suggested_actions) if vr.suggested_actions else '无'}

请为这次失败生成{action}用的子任务。输出 JSON:
{{"name": "修正后的任务名", "description": "修正后的描述（含改进策略）", "tool_names": ["工具1"], "context": "给 Worker 的额外上下文提示"}}"""

    def _parse_llm_spec(
        self,
        original: TaskSpec,
        content: str,
        patch_type: PatchType,
    ) -> TaskSpec:
        import json
        import re

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return self._fallback_spec(original, patch_type)

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return self._fallback_spec(original, patch_type)

        new_id = (
            original.id
            if patch_type == PatchType.RETRY
            else f"{original.id}_sup_{uuid.uuid4().hex[:6]}"
        )
        depends_on = (
            list(original.depends_on)
            if patch_type == PatchType.RETRY
            else []  # 补充任务独立执行，不依赖失败父任务
        )

        return TaskSpec(
            id=new_id,
            name=data.get("name", original.name),
            description=data.get("description", original.description),
            tool_names=data.get("tool_names", list(original.tool_names)),
            depends_on=depends_on,
            priority=original.priority,
            context=data.get("context", original.context),
            retry_count=original.retry_count + 1,
            max_retries=original.max_retries,
        )

    def _fallback_spec(self, original: TaskSpec, patch_type: PatchType) -> TaskSpec:
        """LLM 解析失败时的回退。"""
        if patch_type == PatchType.RETRY:
            vr = VerificationResult(
                pass_=False, score=0.0, error_type=ErrorType.OTHER,
                feedback="LLM 响应不可解析",
            )
            return self._build_retry_spec(original, vr)
        else:
            vr = VerificationResult(
                pass_=False, score=0.0, error_type=ErrorType.INCOMPLETE,
                feedback="LLM 响应不可解析",
            )
            return self._build_add_spec(original, vr)
