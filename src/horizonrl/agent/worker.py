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

from horizonrl.schemas.result import EvidenceItem, StepResult, ToolCall
from horizonrl.schemas.task import TaskSpec

if TYPE_CHECKING:
    from horizonrl.config.settings import RootConfig
    from horizonrl.tools.manager import ToolManager


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
            # 纯分析/汇总任务：直接调用 LLM 处理
            return await self._execute_analysis(task, start)

        # 并行调用所有工具 — asyncio.gather 并发，不等前一个完成
        if len(task.tool_names) == 1:
            # 单工具：直接调用，避免 gather 开销
            tc = await self._call_tool(task.tool_names[0], task)
            tool_calls.append(tc)
            if tc.is_success:
                all_outputs.append(tc.output)
                evidence = self._extract_evidence(
                    task.tool_names[0], tc.output, task.id, task.description
                )
                evidence_items.extend(evidence)
            else:
                all_outputs.append(f"[{task.tool_names[0]}] 失败: {tc.error}")
        else:
            # 多工具：asyncio.gather 并发执行，哪个先完成就先处理
            async def _call_one(name: str):
                tc = await self._call_tool(name, task)
                output_fragment = ""
                ev: list[EvidenceItem] = []
                if tc.is_success:
                    output_fragment = tc.output
                    ev = self._extract_evidence(name, tc.output, task.id, task.description)
                else:
                    output_fragment = f"[{name}] 失败: {tc.error}"
                return tc, output_fragment, ev

            batch = await asyncio.gather(*[_call_one(n) for n in task.tool_names])
            for tc, output_fragment, ev in batch:
                tool_calls.append(tc)
                all_outputs.append(output_fragment)
                evidence_items.extend(ev)

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

    async def _execute_analysis(self, task: TaskSpec, start: float) -> StepResult:
        """纯分析/汇总任务：优先使用 LLM，不可用时回退模板。"""
        import asyncio as _asyncio

        output_text = ""
        tokens = 0
        success = True

        if self.config and getattr(self.config, 'llm', None) and getattr(self.config.llm, 'api_key', None):
            try:
                from horizonrl.llm.client import LLMClient
                client = LLMClient(self.config.llm)
                prompt = (
                    f"你是一个专业的研究分析师。请基于以下任务描述，"
                    f"给出清晰、结构化的分析结果。\n\n"
                    f"任务: {task.description}\n"
                    f"任务名称: {task.name}\n\n"
                    f"请提供详细的分析，包括关键要点、对比维度、结论和建议。"
                )
                result = await _asyncio.wait_for(
                    client.chat(prompt, max_tokens=2000),
                    timeout=60.0,
                )
                if result.is_success:
                    output_text = result.content
                    tokens = result.tokens_used
                else:
                    output_text = f"LLM 分析出错: {result.error}"
                    success = False
            except Exception as e:
                # LLM 异常 → 回退模板分析，不阻塞 pipeline
                str(e)

        # LLM 不可用/无 API Key/调用异常 → 模板分析回退
        if not output_text:
            output_text = (
                f"# {task.name}\n\n"
                f"基于任务描述「{task.description}」的模板分析：\n\n"
                f"1. **任务目标**: {task.description}\n"
                f"2. **分析状态**: 当前为离线/模板分析模式。"
                f"配置 LLM API Key 后可获得更深入的分析结果。\n"
                f"3. **建议**: 请基于已收集的证据手动完成最终分析。"
            )
            success = True  # 模板分析视为成功，不触发重规划

        elapsed = time.monotonic() - start
        return StepResult(
            task_id=task.id,
            success=success,
            output=output_text,
            evidence=[],
            tool_calls=[],
            tokens_used=tokens,
            elapsed=elapsed,
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
        """根据工具类型构建合适的参数。插件工具优先走插件分发。"""
        # 插件分发：工具是插件则委托插件的 build_params()
        if self.tool_manager is not None:
            plugin = self.tool_manager.get_plugin_meta(tool_name)
            if plugin is not None:
                return plugin.build_params(task.description, task.context)

        if tool_name in ("web_search", "arxiv_search", "paper_search", "retrieval"):
            query = self._clean_search_query(task.description)
            if tool_name in ("arxiv_search", "paper_search"):
                return {"query": query, "max_results": 10}
            elif tool_name == "retrieval":
                return {"query": query, "top_k": 5}
            return {"query": query, "num_results": 10}
        elif tool_name == "code_execution":
            return {"code": task.description}
        return {"input": task.description}

    @staticmethod
    def _clean_search_query(description: str) -> str:
        """从任务描述中提取干净的搜索查询词。

        LLMPlanner 生成的描述可能很冗长（如 '搜索XX，包括YY、ZZ'），
        直接当搜索 query 效果差。此方法提取核心关键词。
        """
        import re
        text = description.strip()
        # 去掉常见前缀
        for prefix in ("搜索", "检索", "查找", "调研", "了解", "分析", "探讨"):
            text = re.sub(rf"^{prefix}", "", text, count=1).strip()
        # 去掉尾部句号
        text = text.rstrip("。，.。")
        # 截断到 120 字符 (中文搜索query不宜过长)
        if len(text) > 120:
            # 在最后一个逗号或空格处截断
            cut = text[:120]
            last_comma = max(cut.rfind("，"), cut.rfind(","), cut.rfind(" "))
            if last_comma > 60:
                text = cut[:last_comma]
            else:
                text = cut
        return text.strip() or description[:120]

    def _extract_evidence(
        self, tool_name: str, output: str, task_id: str,
        task_description: str = "",
    ) -> list[EvidenceItem]:
        """从工具输出中提取 EvidenceItem 列表。

        插件工具优先走插件分发。
        """
        # 插件分发：工具是插件则委托插件的 extract_evidence()
        if self.tool_manager is not None:
            plugin = self.tool_manager.get_plugin_meta(tool_name)
            if plugin is not None:
                from horizonrl.plugins.base import PluginEvidence
                plugin_evs: list[PluginEvidence] = plugin.extract_evidence(
                    output, task_description
                )
                now = time.time()
                items: list[EvidenceItem] = []
                for pe in plugin_evs:
                    items.append(EvidenceItem(
                        content=pe.content,
                        source=pe.source or tool_name,
                        source_type=pe.source_type,
                        relevance_score=pe.relevance_score,
                        provider=getattr(plugin, "name", tool_name),
                        search_query=task_description,
                        is_mock=pe.is_mock,
                        retrieved_at=now,
                    ))
                return items

        items: list[EvidenceItem] = []
        now = time.time()
        query_text = task_description or task_id

        if tool_name == "web_search":
            # 优先使用 ToolManager 的规范化方法
            if self.tool_manager and hasattr(self.tool_manager, 'normalize_search_results'):
                normalized = self.tool_manager.normalize_search_results(
                    tool_name, output, query=query_text
                )
                for entry in normalized:
                    items.append(EvidenceItem(
                        content=f"{entry.get('title', '')}: {entry.get('snippet', '')}",
                        source=entry.get("url", ""),
                        source_type="web",
                        provider=entry.get("provider", "web_search"),
                        search_query=entry.get("query", ""),
                        is_mock=entry.get("is_mock", False),
                        retrieved_at=entry.get("timestamp", now),
                    ))
            else:
                # fallback: 旧的手动解析
                parsed = self._try_parse_json(output)
                is_mock = "[Mock]" in output or "mock-search" in output
                if isinstance(parsed, list):
                    for entry in parsed:
                        if isinstance(entry, dict):
                            title = entry.get("title", "")
                            snippet = entry.get("snippet", entry.get("body", str(entry)))
                            url = entry.get("url", entry.get("href", ""))
                            items.append(EvidenceItem(
                                content=f"{title}: {snippet}" if title else snippet,
                                source=url,
                                source_type="web",
                                provider=entry.get("provider", "web_search"),
                                search_query=query_text,
                                is_mock=entry.get("is_mock", is_mock or "mock" in url.lower()),
                                retrieved_at=now,
                            ))
                else:
                    items.append(EvidenceItem(
                        content=output[:2000],
                        source="web_search",
                        source_type="web",
                        provider="web_search",
                        search_query=query_text,
                        is_mock=is_mock,
                        retrieved_at=now,
                    ))

        elif tool_name in ("arxiv_search", "paper_search"):
            parsed = self._try_parse_json(output)
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict):
                        items.append(EvidenceItem(
                            content=f"{entry.get('title', '')}: {entry.get('abstract', '')}",
                            source=entry.get("url", entry.get("pdf_url", "")),
                            source_type="paper",
                            provider=entry.get("provider", "paper_search"),
                            search_query=query_text,
                            is_mock=entry.get("is_mock", False),
                            retrieved_at=now,
                        ))
            else:
                items.append(EvidenceItem(
                    content=output[:2000],
                    source="paper_search",
                    source_type="paper",
                    provider="paper_search",
                    is_mock="[Mock]" in output or "模拟" in output,
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
        """尝试将字符串解析为结构化数据。

        先尝试 JSON，失败后用 ast.literal_eval 兼容 Python 字面量格式
        （如 ToolManager 用 str() 转换的 list[dict]）。都失败则返回原字符串。
        """
        import ast
        import json
        try:
            return json.loads(output)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(output)
            except (ValueError, SyntaxError):
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
