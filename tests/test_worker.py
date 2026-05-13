"""Test AgentWorker module with new schema types."""

from __future__ import annotations

import asyncio
import pytest

from horizonrl.agent.worker import AgentWorker, execute_workers
from horizonrl.schemas.task import TaskSpec, TaskPriority
from horizonrl.schemas.result import StepResult, EvidenceItem, ToolCall
from horizonrl.tools.manager import ToolManager, ToolCallRequest


class MockTool:
    """A fake tool for Worker tests — uses shared mock internally."""

    name = "mock_tool"

    async def execute(self, query: str = "", **kwargs) -> str:
        from horizonrl.tools.mock import MockWebSearch
        mock = MockWebSearch()
        return await mock.search(query)


class TestAgentWorker:
    def test_execute_without_tools(self):
        worker = AgentWorker(worker_id="w1")
        task = TaskSpec(
            id="task_001",
            name="纯分析任务",
            description="分析已有信息并汇总",
            tool_names=[],
        )
        result = asyncio.run(worker.execute(task))

        assert isinstance(result, StepResult)
        assert result.task_id == "task_001"
        assert result.success is True
        assert len(result.tool_calls) == 0
        assert result.worker_id == "w1"

    def test_execute_with_mock_tool(self):
        mgr = ToolManager()
        mgr.register("mock_tool", MockTool())
        worker = AgentWorker(worker_id="w1", tool_manager=mgr)

        task = TaskSpec(
            id="task_002",
            name="信息检索",
            description="搜索 Transformer 注意力机制",
            tool_names=["mock_tool"],
        )
        result = asyncio.run(worker.execute(task))

        assert result.success is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].is_success
        assert len(result.evidence) > 0

    def test_execute_without_tool_manager(self):
        worker = AgentWorker(worker_id="w1", tool_manager=None)
        task = TaskSpec(
            id="task_003",
            name="搜索任务",
            description="搜索信息",
            tool_names=["web_search"],
        )
        result = asyncio.run(worker.execute(task))

        assert result.success is False
        assert len(result.tool_calls) == 1
        assert not result.tool_calls[0].is_success
        assert "ToolManager" in result.tool_calls[0].error

    def test_evidence_extraction(self):
        mgr = ToolManager()
        mgr.register("web_search", MockTool())
        worker = AgentWorker(worker_id="w1", tool_manager=mgr)

        task = TaskSpec(
            id="task_004",
            name="网页搜索",
            description="搜索最新进展",
            tool_names=["web_search"],
        )
        result = asyncio.run(worker.execute(task))

        assert len(result.evidence) > 0
        evidence = result.evidence[0]
        assert isinstance(evidence, EvidenceItem)
        assert evidence.source_type == "web"
        assert "mock-search" in evidence.source

    def test_elapsed_time_tracked(self):
        worker = AgentWorker(worker_id="w1")
        task = TaskSpec(
            id="task_005",
            name="空任务",
            description="什么都不做",
            tool_names=[],
        )
        result = asyncio.run(worker.execute(task))
        assert result.elapsed >= 0


class TestExecuteWorkers:
    def test_concurrent_execution(self):
        mgr = ToolManager()
        mgr.register("mock_tool", MockTool())

        tasks = [
            TaskSpec(id="a", name="A", description="任务A", tool_names=[]),
            TaskSpec(id="b", name="B", description="任务B", tool_names=[]),
            TaskSpec(id="c", name="C", description="任务C", tool_names=[]),
        ]
        results = asyncio.run(execute_workers(tasks, mgr))

        assert len(results) == 3
        assert all(r.task_id in ("a", "b", "c") for r in results)
        assert all(r.success for r in results)

    def test_execute_with_semaphore(self):
        mgr = ToolManager()
        mgr.register("mock_tool", MockTool())
        sem = asyncio.Semaphore(2)

        tasks = [
            TaskSpec(id=f"task_{i}", name=f"T{i}", description=f"任务 {i}", tool_names=[])
            for i in range(5)
        ]
        results = asyncio.run(execute_workers(tasks, mgr, semaphore=sem))

        assert len(results) == 5
        assert all(r.success for r in results)

    def test_mixed_success_and_failure(self):
        mgr = ToolManager()
        mgr.register("mock_tool", MockTool())

        tasks = [
            TaskSpec(id="ok", name="OK", description="正常", tool_names=[]),
            TaskSpec(id="fail", name="FAIL", description="失败", tool_names=["unregistered"]),
        ]
        results = asyncio.run(execute_workers(tasks, mgr))

        assert len(results) == 2
        ok_result = next(r for r in results if r.task_id == "ok")
        fail_result = next(r for r in results if r.task_id == "fail")
        assert ok_result.success is True
        assert fail_result.success is False
