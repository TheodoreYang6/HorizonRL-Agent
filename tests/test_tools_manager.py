"""Test ToolManager, CircuitBreaker, and tool call flow."""

from __future__ import annotations

import asyncio
import pytest

from horizonrl.tools.manager import (
    ToolCallRequest,
    ToolErrorType,
    ToolStats,
    CircuitBreaker,
    ToolManager,
)
from horizonrl.schemas.result import ToolCall


# ─── CircuitBreaker ──────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_allow_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=999)
        for _ in range(3):
            cb.on_failure()
        assert cb.state == "OPEN"
        assert cb.allow() is False

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=999)
        cb.on_failure()
        cb.on_failure()
        cb.on_success()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0)
        cb.on_failure()
        cb.on_failure()
        assert cb.state == "OPEN"
        # cooldown_seconds=0 表示立即冷却
        assert cb.allow() is True
        assert cb.state == "HALF_OPEN"

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0)
        cb.on_failure()
        cb.on_failure()
        cb.allow()  # enters HALF_OPEN
        cb.on_failure()  # fails the probe
        assert cb.state == "OPEN"


# ─── ToolCallRequest ─────────────────────────────────────────────────────


class TestToolCallRequest:
    def test_defaults(self):
        req = ToolCallRequest(tool_name="web_search")
        assert req.tool_name == "web_search"
        assert req.params == {}
        assert req.timeout == 0.0
        assert req.max_retries == 0

    def test_with_params(self):
        req = ToolCallRequest(
            tool_name="arxiv_search",
            params={"query": "attention mechanism", "max_results": 10},
            timeout=15.0,
            max_retries=2,
            task_id="task_001",
        )
        assert req.params["query"] == "attention mechanism"
        assert req.timeout == 15.0
        assert req.max_retries == 2
        assert req.task_id == "task_001"


# ─── ToolErrorType ───────────────────────────────────────────────────────


class TestToolErrorType:
    def test_all_values(self):
        values = {e.value for e in ToolErrorType}
        assert "timeout" in values
        assert "network" in values
        assert "circuit_open" in values


# ─── ToolManager ─────────────────────────────────────────────────────────


class MockSlowTool:
    """Simulates a tool that can be slow."""
    name = "slow_tool"

    def __init__(self, delay: float = 10.0, should_fail: bool = False):
        self.delay = delay
        self.should_fail = should_fail

    async def execute(self, query: str = "") -> str:
        await asyncio.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError("tool failure")
        return f"result for {query}"


class TestToolManager:
    def test_register_and_list(self):
        mgr = ToolManager()
        mgr.register("test_tool", MockSlowTool(delay=0))
        assert mgr.is_registered("test_tool") is True
        assert "test_tool" in mgr.list_tools()

    def test_unregistered_tool_returns_error(self):
        mgr = ToolManager()
        req = ToolCallRequest(tool_name="nonexistent")
        result = asyncio.run(mgr.call(req))
        assert not result.is_success
        assert "未注册" in result.error

    def test_successful_call(self):
        mgr = ToolManager()
        mgr.register("test_tool", MockSlowTool(delay=0.01))
        req = ToolCallRequest(tool_name="test_tool", params={"query": "hello"})
        result = asyncio.run(mgr.call(req))
        assert result.is_success
        assert "hello" in result.output
        assert result.elapsed > 0

    def test_timeout(self):
        mgr = ToolManager()
        mgr._default_timeout = 0.1
        mgr._default_max_retries = 0
        mgr.register("slow_tool", MockSlowTool(delay=1.0))
        req = ToolCallRequest(tool_name="slow_tool")
        result = asyncio.run(mgr.call(req))
        assert not result.is_success
        assert "timeout" in result.error.lower()

    def test_retry_on_failure(self):
        mgr = ToolManager()
        mgr._default_max_retries = 2
        mgr.register("failing_tool", MockSlowTool(delay=0.01, should_fail=True))
        req = ToolCallRequest(tool_name="failing_tool")
        result = asyncio.run(mgr.call(req))
        assert not result.is_success

    def test_stats_tracking(self):
        mgr = ToolManager()
        mgr.register("test_tool", MockSlowTool(delay=0.01))
        req = ToolCallRequest(tool_name="test_tool", params={"query": "test"})
        asyncio.run(mgr.call(req))
        asyncio.run(mgr.call(req))
        stats = mgr.get_stats("test_tool")
        assert stats is not None
        assert stats.total_calls == 2
        assert stats.success_calls == 2

    def test_circuit_open_after_repeated_timeouts(self):
        mgr = ToolManager()
        mgr._default_timeout = 0.1
        mgr._default_max_retries = 0
        mgr._circuit_failure_threshold = 2
        mgr.register("very_slow", MockSlowTool(delay=2.0))

        req = ToolCallRequest(tool_name="very_slow")
        asyncio.run(mgr.call(req))  # timeout #1
        asyncio.run(mgr.call(req))  # timeout #2 → opens circuit
        result = asyncio.run(mgr.call(req))  # rejected by circuit breaker

        assert "circuit_open" in result.error.lower() or "熔断" in result.error

    def test_get_circuit_state(self):
        mgr = ToolManager()
        mgr.register("test_tool", MockSlowTool(delay=0.01))
        assert mgr.get_circuit_state("test_tool") == "CLOSED"
        assert mgr.get_circuit_state("nonexistent") == "UNREGISTERED"

    def test_reset_circuit(self):
        mgr = ToolManager()
        mgr._default_timeout = 0.1
        mgr._default_max_retries = 0
        mgr._circuit_failure_threshold = 2
        mgr.register("slow", MockSlowTool(delay=2.0))

        req = ToolCallRequest(tool_name="slow")
        asyncio.run(mgr.call(req))
        asyncio.run(mgr.call(req))
        assert mgr.get_circuit_state("slow") == "OPEN"

        mgr.reset_circuit("slow")
        assert mgr.get_circuit_state("slow") == "CLOSED"

    def test_get_all_stats(self):
        mgr = ToolManager()
        mgr.register("a", MockSlowTool(delay=0.01))
        mgr.register("b", MockSlowTool(delay=0.01))
        all_stats = mgr.get_all_stats()
        assert "a" in all_stats
        assert "b" in all_stats
