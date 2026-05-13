"""
Verifier —— 结构化验证层。

对 Worker 产出的 StepResult 进行多维度验证，输出 VerificationResult。
Replanner 依赖 Verifier 的 error_type + evidence_gaps + suggested_actions 来做恢复决策。

模式：
  rule   — 纯规则引擎，零延迟，覆盖常见失败模式
  llm    — LLM 深度诊断（偏题/幻觉/事实性错误）
  hybrid — 规则引擎快速筛查 + LLM 复核可疑结果（默认）

使用方式：
    verifier = Verifier(mode="hybrid", llm_client=client)
    result = await verifier.verify(step_result, task_spec)
    if not result.pass_:
        print(f"失败: {result.error_type} → {result.suggested_actions}")
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from horizonrl.schemas.result import (
    ErrorType,
    StepResult,
    VerificationResult,
    EvidenceItem,
    ToolCall,
)
from horizonrl.schemas.task import TaskSpec

if TYPE_CHECKING:
    from horizonrl.llm.client import LLMClient


# ─── 规则引擎 ────────────────────────────────────────────────────────────


class RuleEngine:
    """快速规则验证引擎 —— 基于输出模式做确定性判断。

    覆盖 8 种 ErrorType 的规则检测，每条规则返回 (pass, score, error_type, detail)。
    无外部依赖，纯 Python 字符串/正则匹配，延迟 <1ms。
    """

    # 工具错误信号（输出中含这些关键词 → TOOL_ERROR）
    TOOL_ERROR_PATTERNS = [
        (r"工具未注册", "工具未在 ToolManager 中注册"),
        (r"ToolManager 未初始化", "ToolManager 未注入 Worker"),
        (r"熔断器开启", "工具熔断器触发，拒绝调用"),
        (r"\[timeout\]", "工具调用超时"),
        (r"\[circuit_open\]", "熔断器开启"),
        (r"\[auth\]", "工具鉴权失败"),
        (r"\[rate_limit\]", "工具触发限流"),
        (r"\[network\]", "工具网络错误"),
    ]

    # 空结果信号
    EMPTY_PATTERNS = [
        (r"^\s*$", "输出为空白"),
        (r"DuckDuckGo unavailable", "DuckDuckGo 搜索不可用"),
        (r"Install duckduckgo-search", "缺少 duckduckgo-search 依赖"),
    ]

    # 代码错误信号
    CODE_ERROR_PATTERNS = [
        (r"Traceback\s*\(most recent call last\)", "Python 代码执行异常"),
        (r"SyntaxError", "Python 语法错误"),
        (r"ModuleNotFoundError", "Python 模块未找到"),
        (r"ImportError", "Python 导入错误"),
        (r"Execution timed out", "代码执行超时"),
    ]

    # 不完整信号
    INCOMPLETE_PATTERNS = [
        (r"无需工具调用，等待后续 LLM 处理", "纯分析任务未接入 LLM"),
    ]

    def check(self, result: StepResult, task_desc: str) -> dict:
        """对 StepResult 执行全规则检查。

        Returns:
            {"pass": bool, "score": float, "error_type": ErrorType, "feedback": str,
             "evidence_gaps": list[str], "suggested_actions": list[str]}
        """
        output = result.output
        evidence = result.evidence
        tool_calls = result.tool_calls

        # ── 1. Worker 本身失败 ──
        if not result.success and not tool_calls:
            return self._fail(
                ErrorType.OTHER,
                f"Worker 执行失败: {result.error or '未知错误'}",
                score=0.0,
                actions=["重试该子任务", "检查 Worker 日志"],
            )

        # ── 3. 空/无效输出 ──
        for pattern, desc in self.EMPTY_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return self._fail(
                    ErrorType.EMPTY_RESULT,
                    f"输出为空或无效: {desc}",
                    score=0.0,
                    gaps=["需要有效的搜索结果"],
                    actions=["改写搜索查询词", "使用备用搜索引擎", "扩大搜索范围"],
                )

        # ── 4. 代码错误（在工具错误之前，Traceback 更具体） ──
        for pattern, desc in self.CODE_ERROR_PATTERNS:
            if re.search(pattern, output):
                return self._fail(
                    ErrorType.CODE_ERROR,
                    f"代码执行错误: {desc}",
                    score=0.2,
                    actions=["修正代码语法", "检查依赖导入", "简化输入用例"],
                )

        # ── 5. 全部工具失败（所有 tool call 都报错） ──
        if tool_calls and all(not tc.is_success for tc in tool_calls):
            error_details = [tc.error for tc in tool_calls if tc.error]
            return self._fail(
                ErrorType.TOOL_ERROR,
                f"所有工具调用失败: {'; '.join(error_details[:3])}",
                score=0.1,
                actions=["切换备用工具", "检查 API Key 和网络", "重试"],
            )

        # ── 6. 工具错误信号（输出中包含工具错误关键词） ──
        for pattern, desc in self.TOOL_ERROR_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return self._fail(
                    ErrorType.TOOL_ERROR,
                    f"工具调用失败: {desc}",
                    score=0.1,
                    actions=["重试该工具", "检查工具配置", "使用备用工具"],
                )

        # ── 8. 无证据 ──
        if not evidence and not any(self._has_code_output(tc) for tc in tool_calls):
            return self._fail(
                ErrorType.INCOMPLETE,
                "任务完成但未收集到任何证据",
                score=0.3,
                gaps=["需要至少 1 条有效证据"],
                actions=["重试并指定更具体的搜索参数", "检查工具返回格式"],
            )

        # ── 9. 输出过短（可能无意义） ──
        clean_output = output.strip()
        if len(clean_output) < 20 and evidence:
            return self._pass(
                score=0.6,
                feedback="输出较短但已收集证据，质量待 LLM 确认",
            )

        # ── 8. 全部通过 ──
        score = self._calc_score(result)
        return self._pass(
            score=score,
            feedback=f"规则检查通过: {len(evidence)}条证据, "
                     f"{result.tool_success_count}/{result.tool_total_count}工具成功",
        )

    def _calc_score(self, result: StepResult) -> float:
        """根据证据数量和质量计算基础分数。"""
        evidence_count = len(result.evidence)
        tool_total = result.tool_total_count
        tool_ok = result.tool_success_count

        if evidence_count == 0:
            return 0.4
        if evidence_count == 1:
            base = 0.6
        elif evidence_count <= 3:
            base = 0.75
        else:
            base = 0.85

        # 工具成功率修正
        if tool_total > 0:
            tool_rate = tool_ok / tool_total
            if tool_rate < 0.5:
                base -= 0.2
            elif tool_rate == 1.0:
                base += 0.05

        return min(1.0, max(0.0, base))

    def _has_code_output(self, tc: ToolCall) -> bool:
        """检查工具调用是否包含有效代码输出。"""
        if tc.tool_name != "code_execution":
            return False
        if not tc.is_success:
            return False
        import json
        try:
            data = json.loads(tc.output)
            return bool(data.get("stdout") or data.get("success"))
        except (json.JSONDecodeError, TypeError):
            return bool(tc.output.strip())

    def _pass(self, score: float, feedback: str) -> dict:
        return {
            "pass": True,
            "score": score,
            "error_type": ErrorType.NONE,
            "feedback": feedback,
            "evidence_gaps": [],
            "suggested_actions": [],
        }

    def _fail(
        self,
        error_type: ErrorType,
        feedback: str,
        score: float,
        gaps: list[str] | None = None,
        actions: list[str] | None = None,
    ) -> dict:
        return {
            "pass": False,
            "score": score,
            "error_type": error_type,
            "feedback": feedback,
            "evidence_gaps": gaps or [],
            "suggested_actions": actions or [],
        }


# ─── LLM 增强验证器 ──────────────────────────────────────────────────────


class LLMVerifier:
    """用 LLM 对 StepResult 做深度语义验证。

    覆盖规则引擎无法判断的复杂场景：
      - 输出是否偏离任务目标 (off-topic)
      - 输出是否包含明显事实错误 (factual_error)
      - 输出是否包含编造内容 (hallucination)
      - 证据质量是否支撑结论
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def verify(self, result: StepResult, task_desc: str) -> dict:
        """LLM 深度验证。

        Returns:
            同 RuleEngine.check() 格式的 dict。
        """
        prompt = self._build_prompt(result, task_desc)
        llm_result = await self.llm.chat(
            prompt,
            system_prompt=(
                "你是一个严格的质量验证器。你需要判断 Agent 的输出是否满足任务要求。"
                "你只输出 JSON，不输出其他内容。"
            ),
            temperature=0.0,
            max_tokens=256,
        )

        if not llm_result.is_success:
            # LLM 调用失败，回退到规则引擎
            engine = RuleEngine()
            return engine.check(result, task_desc)

        return self._parse_llm_response(llm_result.content)

    def _build_prompt(self, result: StepResult, task_desc: str) -> str:
        evidence_text = ""
        for i, e in enumerate(result.evidence[:5]):
            evidence_text += f"[{i+1}] {e.content[:200]}...\n"

        tool_text = ""
        for tc in result.tool_calls[:5]:
            status = "OK" if tc.is_success else f"FAIL: {tc.error}"
            tool_text += f"  {tc.tool_name}: {status} ({tc.elapsed:.1f}s)\n"

        return f"""请验证以下 Agent 输出。

任务描述: {task_desc}

Agent 输出:
{result.output[:800]}

证据 ({len(result.evidence)}条):
{evidence_text or '(无证据)'}

工具调用:
{tool_text or '(无工具调用)'}

请判断:
1. pass: 输出是否满足任务要求 (true/false)
2. score: 质量评分 0.0-1.0
3. error_type: 如果失败/低质量，属于哪类错误
   (empty_result|code_error|off_topic|factual_error|tool_error|incomplete|hallucination|none)
4. feedback: 一句话诊断
5. evidence_gaps: 缺少的证据类型 (数组)
6. suggested_actions: 建议的补救步骤 (数组)

输出 JSON:
{{"pass": true/false, "score": 0.0-1.0, "error_type": "...", "feedback": "...", "evidence_gaps": [...], "suggested_actions": [...]}}"""

    def _parse_llm_response(self, content: str) -> dict:
        import json
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {
                "pass": True, "score": 0.5, "error_type": ErrorType.NONE,
                "feedback": "LLM 验证响应不可解析", "evidence_gaps": [], "suggested_actions": [],
            }
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "pass": True, "score": 0.5, "error_type": ErrorType.NONE,
                "feedback": "LLM 响应 JSON 解析失败", "evidence_gaps": [], "suggested_actions": [],
            }

        # 映射 error_type 字符串到枚举
        error_str = data.get("error_type", "none")
        try:
            error_type = ErrorType(error_str)
        except ValueError:
            error_type = ErrorType.NONE

        return {
            "pass": data.get("pass", True),
            "score": float(data.get("score", 0.5)),
            "error_type": error_type,
            "feedback": str(data.get("feedback", ""))[:200],
            "evidence_gaps": data.get("evidence_gaps", [])[:5],
            "suggested_actions": data.get("suggested_actions", [])[:5],
        }


# ─── 主编排器 ────────────────────────────────────────────────────────────


class Verifier:
    """结构化验证器 —— 对 StepResult 做多维质量评估。

    支持三种模式：
      rule   — 纯规则引擎，零延迟，~0.1ms
      llm    — LLM 深度诊断，2-5s，覆盖偏题/幻觉
      hybrid — 规则快速筛查 + LLM 复核低分结果（默认推荐）

    Examples:
        >>> verifier = Verifier(mode="rule")
        >>> result = await verifier.verify(step_result, task_spec)
        >>> if not result.pass_:
        ...     print(result.error_type, result.suggested_actions)
    """

    def __init__(
        self,
        mode: str = "hybrid",
        llm_client: LLMClient | None = None,
    ):
        """
        Args:
            mode: "rule" | "llm" | "hybrid"
            llm_client: LLM 客户端（mode=llm/hybrid 时必需）
        """
        self.mode = mode
        self.rule_engine = RuleEngine()
        self.llm_verifier = LLMVerifier(llm_client) if llm_client else None

        if mode == "llm" and self.llm_verifier is None:
            raise ValueError("LLM 模式需要提供 llm_client")

    async def verify(
        self,
        result: StepResult,
        task: TaskSpec | str,
    ) -> VerificationResult:
        """验证单个 StepResult。

        Args:
            result: Worker 产出的步骤结果。
            task: TaskSpec 或任务描述字符串。

        Returns:
            VerificationResult（含 pass/score/error_type/feedback/gaps/actions）。
        """
        task_desc = task.description if isinstance(task, TaskSpec) else task
        start = time.monotonic()

        # ── Rule 模式：纯规则引擎 ──
        if self.mode == "rule":
            check = self.rule_engine.check(result, task_desc)
            return self._to_result(check, start)

        # ── LLM 模式：直接 LLM ──
        if self.mode == "llm":
            if self.llm_verifier is None:
                raise ValueError("LLM 模式需要提供 llm_client")
            check = await self.llm_verifier.verify(result, task_desc)
            tokens = 0  # 无法精确追踪，由 LLMClient 记录
            return self._to_result(check, start, tokens)

        # ── Hybrid 模式（默认）：规则引擎快速筛查 + LLM 复核 ──
        if self.mode == "hybrid":
            check = self.rule_engine.check(result, task_desc)

            # 规则引擎通过的 + 高分 → 直接放行
            if check["pass"] and check["score"] >= 0.7:
                return self._to_result(check, start)

            # 规则引擎明确失败（低分）→ 直接用规则结果
            if not check["pass"] and check["score"] < 0.3:
                return self._to_result(check, start)

            # 中间地带（0.3-0.7）或 通过但低分 → LLM 复核
            if self.llm_verifier is not None:
                try:
                    llm_check = await self.llm_verifier.verify(result, task_desc)
                    # 以 LLM 意见为准（更权威）
                    return self._to_result(llm_check, start, 0)
                except Exception:
                    pass  # LLM 失败则降级到规则结果

            return self._to_result(check, start)

    def verify_sync(self, result: StepResult, task: TaskSpec | str) -> VerificationResult:
        """同步验证接口（内部 asyncio.run）。"""
        import asyncio
        return asyncio.run(self.verify(result, task))

    def _to_result(
        self, check: dict, start: float, tokens: int = 0
    ) -> VerificationResult:
        elapsed = time.monotonic() - start
        return VerificationResult(
            pass_=check["pass"],
            score=check["score"],
            error_type=check["error_type"],
            feedback=check["feedback"],
            evidence_gaps=check["evidence_gaps"],
            suggested_actions=check["suggested_actions"],
            tokens_used=tokens,
            elapsed=elapsed,
        )
