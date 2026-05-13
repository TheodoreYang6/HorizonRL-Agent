"""
Agent Worker —— 异步子任务执行模块。

Worker 接收 TaskSpec，通过 ToolManager 调用工具，收集证据，
返回 StepResult。多个 Worker 可通过 asyncio 并发执行。

输入：TaskSpec + ToolManager
输出：StepResult（含 EvidenceItem[] + ToolCall[]）
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from horizonrl.schemas.result import StepResult, EvidenceItem, ToolCall
from horizonrl.schemas.task import TaskSpec

if TYPE_CHECKING:
    from horizonrl.config.settings import RootConfig
    from horizonrl.tools.manager import ToolManager, ToolCallRequest


class AgentWorker:
    """异步子任务执行器。

    每个 Worker 独立执行一个 TaskSpec，通过 ToolManager 调用工具。
    支持 LLM 增强执行（Phase 2+）和纯工具调用执行（MVP）。

    Examples:
        >>> worker = AgentWorker(tool_manager=mgr)
        >>> result = await worker.execute(task_spec)
        >>> print(result.success, len(result.evidence))
    """

    def __init__(
        self,
        worker_id: str = "",
        tool_manager: ToolManager | None = None,
        config: RootConfig | None = None,
    ):
        """
        Args:
            worker_id: Worker 标识符（用于日志追踪）。
            tool_manager: 统一工具管理器。
            config: 全局配置。
        """
        self.worker_id = worker_id or f"worker_{id(self):x}"
        self.tool_manager = tool_manager
        self.config = config

    async def execute(self, task: TaskSpec) -> StepResult:
        """执行单个子任务。

        执行流程：
          1. 根据 TaskSpec.tool_names 确定要调用的工具
          2. 依次调用每个工具（通过 ToolManager）
          3. 从工具返回结果中提取 EvidenceItem
          4. 汇总为 StepResult

        Args:
            task: 要执行的子任务规格。

        Returns:
            StepResult 包含执行结果、证据、工具调用记录。
        """
        start = time.monotonic()
        tool_calls: list[ToolCall] = []
        evidence_items: list[EvidenceItem] = []
        all_outputs: list[str] = []

        if not task.tool_names:
            # 无需工具的任务（如纯分析/汇总），返回占位结果
            elapsed = time.monotonic() - start
            return StepResult(
                task_id=task.id,
                success=True,
                output=f"[{task.name}] 无需工具调用，等待后续 LLM 处理",
                evidence=[],
                tool_calls=[],
                tokens_used=0,
                elapsed=elapsed,
                worker_id=self.worker_id,
            )

        # 逐个调用工具（MVP 串行，Phase 2+ 可按需并行）
        for tool_name in task.tool_names:
            tc = await self._call_tool(tool_name, task)
            tool_calls.append(tc)

            if tc.is_success:
                all_outputs.append(tc.output)
                # 从工具输出中提取证据
                evidence = self._extract_evidence(tool_name, tc.output, task.id)
                evidence_items.extend(evidence)
            else:
                all_outputs.append(f"[{tool_name}] 失败: {tc.error}")

        elapsed = time.monotonic() - start
        total_tokens = sum(tc.tokens_used for tc in tool_calls)
        success = all(tc.is_success for tc in tool_calls)

        return StepResult(
            task_id=task.id,
            success=success,
            output="\n\n".join(all_outputs),
            evidence=evidence_items,
            tool_calls=tool_calls,
            tokens_used=total_tokens,
            elapsed=elapsed,
            error="" if success else "部分工具调用失败",
            worker_id=self.worker_id,
        )

    async def _call_tool(self, tool_name: str, task: TaskSpec) -> ToolCall:
        """通过 ToolManager 调用单个工具。"""
        if self.tool_manager is None:
            return ToolCall(
                tool_name=tool_name,
                input={"query": task.description},
                output="",
                elapsed=0.0,
                error="ToolManager 未初始化",
            )

        from horizonrl.tools.manager import ToolCallRequest

        # 根据工具类型构建参数
        params = self._build_params(tool_name, task)

        request = ToolCallRequest(
            tool_name=tool_name,
            params=params,
            task_id=task.id,
        )
        return await self.tool_manager.call(request)

    def _build_params(self, tool_name: str, task: TaskSpec) -> dict:
        """根据工具类型构建合适的参数。"""
        if tool_name in ("web_search", "arxiv_search"):
            return {"query": task.description}
        elif tool_name == "code_execution":
            return {"code": task.description}
        elif tool_name == "retrieval":
            return {"query": task.description, "top_k": 5}
        return {"input": task.description}

    def _extract_evidence(
        self, tool_name: str, output: str, task_id: str
    ) -> list[EvidenceItem]:
        """从工具输出中提取 EvidenceItem 列表。

        Args:
            tool_name: 工具名称。
            output: 工具原始输出（可能是字符串化的 JSON）。
            task_id: 关联的任务 ID。

        Returns:
            EvidenceItem 列表。
        """
        items: list[EvidenceItem] = []
        now = time.time()

        if tool_name == "web_search":
            # 尝试解析 JSON 列表格式
            parsed = self._try_parse_json(output)
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict):
                        items.append(EvidenceItem(
                            content=entry.get("snippet", entry.get("body", str(entry))),
                            source=entry.get("url", entry.get("href", "")),
                            source_type="web",
                            retrieved_at=now,
                        ))
            else:
                items.append(EvidenceItem(
                    content=output[:2000],
                    source="web_search",
                    source_type="web",
                    retrieved_at=now,
                ))

        elif tool_name == "arxiv_search":
            parsed = self._try_parse_json(output)
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict):
                        items.append(EvidenceItem(
                            content=f"{entry.get('title', '')}: {entry.get('abstract', '')}",
                            source=entry.get("url", entry.get("pdf_url", "")),
                            source_type="arxiv",
                            retrieved_at=now,
                        ))
            else:
                items.append(EvidenceItem(
                    content=output[:2000],
                    source="arxiv_search",
                    source_type="arxiv",
                    retrieved_at=now,
                ))

        elif tool_name == "code_execution":
            parsed = self._try_parse_json(output)
            if isinstance(parsed, dict):
                items.append(EvidenceItem(
                    content=parsed.get("stdout", parsed.get("output", output)),
                    source="code_execution",
                    source_type="code_output",
                    retrieved_at=now,
                ))
            else:
                items.append(EvidenceItem(
                    content=output[:2000],
                    source="code_execution",
                    source_type="code_output",
                    retrieved_at=now,
                ))

        else:
            items.append(EvidenceItem(
                content=output[:2000],
                source=tool_name,
                source_type="api",
                retrieved_at=now,
            ))

        return items

    @staticmethod
    def _try_parse_json(output: str):
        """尝试将字符串解析为 JSON，失败时返回原始字符串。"""
        import json
        try:
            return json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return output


async def execute_workers(
    tasks: list[TaskSpec],
    tool_manager: ToolManager,
    semaphore: asyncio.Semaphore | None = None,
) -> list[StepResult]:
    """并发执行多个 TaskSpec。

    这是 Worker 调度的核心函数。所有独立任务并行执行，
    通过 Semaphore 控制并发数。

    Args:
        tasks: 要执行的任务列表。
        tool_manager: 共享的 ToolManager 实例。
        semaphore: 并发控制信号量。None 表示不限制。

    Returns:
        按输入顺序排列的 StepResult 列表。
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(len(tasks))

    async def _execute_one(task: TaskSpec) -> StepResult:
        async with semaphore:
            worker = AgentWorker(
                worker_id=f"worker_{task.id}",
                tool_manager=tool_manager,
            )
            return await worker.execute(task)

    return list(await asyncio.gather(*[_execute_one(t) for t in tasks]))
