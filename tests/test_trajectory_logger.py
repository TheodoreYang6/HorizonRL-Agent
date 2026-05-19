"""Test TrajectoryLogger — async JSONL write, session lifecycle, analysis utilities."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from horizonrl.logging.trajectory_logger import (
    TrajectoryLogger,
    aggregate_stats,
    create_logger,
    event_type_distribution,
    filter_events,
    list_sessions,
    read_session,
)
from horizonrl.schemas.event import (
    EventType,
    TrajectoryEvent,
    TrajectorySession,
)

# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
async def logger(tmp_dir):
    """创建一个已启动会话的 logger。"""
    log = TrajectoryLogger(output_dir=tmp_dir)
    yield log
    # cleanup
    if log.is_active:
        await log.end_session(success=False)
    await log.close()


@pytest.fixture
def sample_events():
    """一组模拟 Agent 执行事件。"""
    return [
        TrajectoryEvent(
            module="planner",
            event_type=EventType.PLAN_START,
            payload={"user_task": "测试任务"},
            cost=500, latency=0.5,
        ),
        TrajectoryEvent(
            module="planner",
            event_type=EventType.PLAN_COMPLETE,
            payload={"num_subtasks": 5},
            cost=1200, latency=3.5,
        ),
        TrajectoryEvent(
            module="worker",
            event_type=EventType.WORKER_START,
            payload={"task_id": "task_001", "worker_id": "wrk_1"},
            cost=100,
        ),
        TrajectoryEvent(
            module="worker",
            event_type=EventType.WORKER_COMPLETE,
            payload={"task_id": "task_001"},
            cost=800, latency=2.0,
        ),
        TrajectoryEvent(
            module="tool",
            event_type=EventType.TOOL_CALL,
            payload={"tool_name": "web_search"},
            cost=50, latency=0.8,
        ),
        TrajectoryEvent(
            module="tool",
            event_type=EventType.TOOL_RESULT,
            payload={"tool_name": "web_search"},
            cost=30, latency=0.3,
        ),
        TrajectoryEvent(
            module="verifier",
            event_type=EventType.VERIFY_COMPLETE,
            payload={"task_id": "task_001", "pass": True, "score": 0.85},
            cost=100, latency=0.1,
        ),
        TrajectoryEvent(
            module="replanner",
            event_type=EventType.REPLAN_PATCH,
            payload={"target_node": "task_002", "patch_type": "retry"},
            cost=300, latency=1.2,
        ),
    ]


# ─── Initialization ─────────────────────────────────────────────────────────


class TestInit:
    def test_creates_output_dir(self, tmp_dir):
        d = Path(tmp_dir) / "sub_logs"
        TrajectoryLogger(output_dir=str(d))
        assert d.exists()

    def test_default_dir(self):
        log = TrajectoryLogger()
        assert log.output_dir.name == "trajectories"

    def test_no_active_session_initially(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        assert log.session is None
        assert not log.is_active


# ─── Session Lifecycle ──────────────────────────────────────────────────────


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_start_session_returns_id(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        sid = await log.start_session("测试研究任务")
        assert sid.startswith("session_")
        assert log.is_active
        await log.end_session()

    @pytest.mark.asyncio
    async def test_start_session_creates_file(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务")
        files = list(Path(tmp_dir).glob("*.jsonl"))
        assert len(files) == 1
        await log.end_session()

    @pytest.mark.asyncio
    async def test_end_session_writes_summary(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务A")
        await log.log(TrajectoryEvent(
            module="worker", event_type=EventType.WORKER_COMPLETE,
            payload={"task_id": "t1"}, cost=500,
        ))
        session = await log.end_session(success=True)

        assert session.success is True
        assert session.total_tokens == 500 + 0  # session start 也有事件但 cost=0
        assert session.total_steps > 0
        assert not log.is_active

    @pytest.mark.asyncio
    async def test_new_session_replaces_old(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        sid1 = await log.start_session("任务1")
        await log.end_session()
        sid2 = await log.start_session("任务2")
        assert sid1 != sid2
        files = sorted(Path(tmp_dir).glob("*.jsonl"))
        assert len(files) == 2
        await log.end_session()

    @pytest.mark.asyncio
    async def test_end_without_start_raises(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        with pytest.raises(RuntimeError):
            await log.end_session()


# ─── Event Logging ──────────────────────────────────────────────────────────


class TestEventLogging:
    @pytest.mark.asyncio
    async def test_log_single_event(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务")
        await log.log(TrajectoryEvent(
            module="planner",
            event_type=EventType.PLAN_COMPLETE,
            payload={"num_subtasks": 5},
            cost=1000, latency=2.0,
        ))
        await log.end_session()

        # 从文件读取验证
        session = read_session(Path(tmp_dir) / f"{log.session.session_id}.jsonl")
        planner_events = session.filter_by_module("planner")
        assert len(planner_events) >= 1
        e = planner_events[-1]
        assert e.event_type == EventType.PLAN_COMPLETE
        assert e.payload["num_subtasks"] == 5

    @pytest.mark.asyncio
    async def test_log_multiple_events(self, tmp_dir, sample_events):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("复杂任务")

        for e in sample_events:
            await log.log(e)

        session = await log.end_session()
        assert session.total_steps >= len(sample_events)

    @pytest.mark.asyncio
    async def test_log_sets_session_id(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        sid = await log.start_session("任务")
        event = TrajectoryEvent(
            module="worker", event_type=EventType.WORKER_COMPLETE,
            payload={},
        )
        await log.log(event)
        assert event.session_id == sid
        await log.end_session()

    @pytest.mark.asyncio
    async def test_log_nowait_sync(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务")
        log.log_nowait(TrajectoryEvent(
            module="worker", event_type=EventType.WORKER_COMPLETE,
            payload={},
        ))
        await log.flush()
        assert log.event_count == 2  # session.start + this event
        await log.end_session()

    @pytest.mark.asyncio
    async def test_log_without_session_raises(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        with pytest.raises(RuntimeError):
            await log.log(TrajectoryEvent(
                module="test", event_type=EventType.ERROR, payload={},
            ))


# ─── Session Stats ──────────────────────────────────────────────────────────


class TestSessionStats:
    @pytest.mark.asyncio
    async def test_total_tokens_accumulated(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务")
        await log.log(TrajectoryEvent(
            module="worker", event_type=EventType.WORKER_COMPLETE,
            cost=500,
        ))
        await log.log(TrajectoryEvent(
            module="tool", event_type=EventType.TOOL_RESULT,
            cost=200,
        ))
        session = await log.end_session()
        assert session.total_tokens >= 700

    @pytest.mark.asyncio
    async def test_tool_calls_counted(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务")
        await log.log(TrajectoryEvent(
            module="tool", event_type=EventType.TOOL_CALL,
            payload={"tool_name": "web_search"},
        ))
        await log.log(TrajectoryEvent(
            module="tool", event_type=EventType.TOOL_RESULT,
            payload={"tool_name": "web_search"},
        ))
        await log.log(TrajectoryEvent(
            module="tool", event_type=EventType.TOOL_CALL,
            payload={"tool_name": "arxiv_search"},
        ))
        session = await log.end_session()
        assert session.total_tool_calls == 1  # 仅 TOOL_RESULT 计数，避免重复

    @pytest.mark.asyncio
    async def test_replan_counted(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务")
        await log.log(TrajectoryEvent(
            module="replanner", event_type=EventType.REPLAN_PATCH,
            payload={},
        ))
        await log.log(TrajectoryEvent(
            module="replanner", event_type=EventType.REPLAN_PATCH,
            payload={},
        ))
        session = await log.end_session()
        assert session.replan_count == 2

    @pytest.mark.asyncio
    async def test_wall_time_computed(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务")
        # 稍等一下
        await asyncio.sleep(0.05)
        session = await log.end_session()
        assert session.wall_time > 0

    @pytest.mark.asyncio
    async def test_avg_latency(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务")
        await log.log(TrajectoryEvent(
            module="worker", event_type=EventType.WORKER_COMPLETE,
            latency=1.5,
        ))
        await log.log(TrajectoryEvent(
            module="worker", event_type=EventType.WORKER_COMPLETE,
            latency=2.5,
        ))
        session = await log.end_session()
        assert session.avg_latency == pytest.approx(2.0, rel=0.1)


# ─── flush ──────────────────────────────────────────────────────────────────


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_writes_to_disk(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("任务")
        await log.log(TrajectoryEvent(
            module="worker", event_type=EventType.WORKER_COMPLETE,
            payload={"data": "test"},
        ))
        await log.flush()

        # 读取文件验证事件已写入
        files = list(Path(tmp_dir).glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "worker.complete" in content
        await log.end_session()

    @pytest.mark.asyncio
    async def test_flush_noop_without_session(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.flush()  # 不应崩溃


# ─── JSONL Format ───────────────────────────────────────────────────────────


class TestJSONLFormat:
    @pytest.mark.asyncio
    async def test_each_line_is_valid_json(self, tmp_dir, sample_events):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("格式测试")
        for e in sample_events:
            await log.log(e)
        session = await log.end_session()

        filepath = Path(tmp_dir) / f"{session.session_id}.jsonl"
        lines = filepath.read_text(encoding="utf-8").strip().split("\n")
        for i, line in enumerate(lines):
            if line.strip():
                data = json.loads(line)
                assert "ts" in data, f"Line {i}: missing ts"
                assert "module" in data, f"Line {i}: missing module"
                assert "event_type" in data, f"Line {i}: missing event_type"

    @pytest.mark.asyncio
    async def test_events_in_order(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("顺序测试")
        await log.log(TrajectoryEvent(module="a", event_type=EventType.PLAN_START))
        await log.log(TrajectoryEvent(module="b", event_type=EventType.WORKER_START))
        await log.log(TrajectoryEvent(module="c", event_type=EventType.WORKER_COMPLETE))
        session = await log.end_session()

        filepath = Path(tmp_dir) / f"{session.session_id}.jsonl"
        content = filepath.read_text(encoding="utf-8")
        a_pos = content.find('"module": "a"')
        b_pos = content.find('"module": "b"')
        c_pos = content.find('"module": "c"')
        assert a_pos < b_pos < c_pos


# ─── Read Session ───────────────────────────────────────────────────────────


class TestReadSession:
    @pytest.mark.asyncio
    async def test_roundtrip(self, tmp_dir, sample_events):
        """写入后读取，验证事件完整。"""
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("往返测试")
        for e in sample_events:
            await log.log(e)
        session = await log.end_session()

        filepath = Path(tmp_dir) / f"{session.session_id}.jsonl"
        restored = read_session(filepath)

        assert restored.session_id == session.session_id
        assert restored.user_task == "往返测试"
        assert len(restored.events) == len(session.events)

    def test_read_nonexistent_file(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            read_session(Path(tmp_dir) / "nonexistent.jsonl")

    def test_read_empty_dir(self, tmp_dir):
        """空目录读取应返回空 session。"""
        # 创建空文件
        empty_file = Path(tmp_dir) / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        session = read_session(empty_file)
        assert isinstance(session, TrajectorySession)

    def test_read_malformed_lines(self, tmp_dir):
        """畸形行应被跳过。"""
        bad_file = Path(tmp_dir) / "bad.jsonl"
        bad_file.write_text(
            '{"ts": 1, "module": "a", "event_type": "plan.start", '
            '"payload": {}, "cost": 0, "latency": 0, "session_id": "s1", "step_id": 1}\n'
            'not valid json\n'
            '{"ts": 2, "module": "b", "event_type": "worker.start", '
            '"payload": {}, "cost": 0, "latency": 0, "session_id": "s1", "step_id": 2}\n',
            encoding="utf-8",
        )
        session = read_session(bad_file)
        assert len(session.events) == 2  # 畸形行跳过


# ─── list_sessions ──────────────────────────────────────────────────────────


class TestListSessions:
    @pytest.mark.asyncio
    async def test_lists_created_sessions(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("A")
        await log.end_session()
        await log.start_session("B")
        await log.end_session()

        files = list_sessions(tmp_dir)
        assert len(files) == 2

    def test_returns_empty_for_nonexistent_dir(self):
        files = list_sessions("/tmp/nonexistent_dir_12345")
        assert files == []


# ─── aggregate_stats ────────────────────────────────────────────────────────


class TestAggregateStats:
    @pytest.mark.asyncio
    async def test_aggregates_multiple_sessions(self, tmp_dir):
        for i in range(3):
            log = TrajectoryLogger(output_dir=tmp_dir)
            await log.start_session(f"任务{i}")
            await log.log(TrajectoryEvent(
                module="worker", event_type=EventType.WORKER_COMPLETE,
                cost=100, latency=1.0,
            ))
            await log.end_session(success=(i < 2))

        stats = aggregate_stats(tmp_dir)
        assert stats["total_sessions"] == 3
        assert stats["success_count"] == 2
        assert stats["failure_count"] == 1
        assert stats["success_rate"] == pytest.approx(2 / 3)
        assert stats["total_events"] > 0

    def test_aggregates_empty_dir(self, tmp_dir):
        stats = aggregate_stats(tmp_dir)
        assert stats["total_sessions"] == 0


# ─── event_type_distribution ────────────────────────────────────────────────


class TestEventTypeDistribution:
    @pytest.mark.asyncio
    async def test_distribution(self, tmp_dir, sample_events):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("分布测试")
        for e in sample_events:
            await log.log(e)
        session = await log.end_session()

        filepath = Path(tmp_dir) / f"{session.session_id}.jsonl"
        dist = event_type_distribution(filepath)

        assert "plan.start" in dist
        assert "plan.complete" in dist
        assert "worker.complete" in dist
        assert "tool.call" in dist
        assert dist["plan.start"] == 1
        assert dist["plan.complete"] == 1


# ─── filter_events ──────────────────────────────────────────────────────────


class TestFilterEvents:
    @pytest.mark.asyncio
    async def test_filter_by_module(self, tmp_dir, sample_events):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("过滤测试")
        for e in sample_events:
            await log.log(e)
        session = await log.end_session()

        filepath = Path(tmp_dir) / f"{session.session_id}.jsonl"

        planner_events = filter_events(filepath, module="planner")
        assert len(planner_events) == 2
        assert all(e.module == "planner" for e in planner_events)

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self, tmp_dir, sample_events):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("过滤测试")
        for e in sample_events:
            await log.log(e)
        session = await log.end_session()

        filepath = Path(tmp_dir) / f"{session.session_id}.jsonl"

        tool_calls = filter_events(filepath, event_type=EventType.TOOL_CALL)
        assert len(tool_calls) == 1
        assert tool_calls[0].payload["tool_name"] == "web_search"

    @pytest.mark.asyncio
    async def test_filter_combined(self, tmp_dir, sample_events):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("组合过滤")
        for e in sample_events:
            await log.log(e)
        session = await log.end_session()

        filepath = Path(tmp_dir) / f"{session.session_id}.jsonl"

        results = filter_events(
            filepath,
            module="tool",
            event_type=EventType.TOOL_CALL,
        )
        assert len(results) == 1


# ─── create_logger Factory ──────────────────────────────────────────────────


class TestCreateLogger:
    def test_returns_logger(self, tmp_dir):
        log = create_logger(output_dir=tmp_dir)
        assert isinstance(log, TrajectoryLogger)


# ─── Edge Cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_close_without_start(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.close()  # 不应崩溃

    @pytest.mark.asyncio
    async def test_close_with_active_session(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("测试")
        await log.close()
        assert not log.is_active
        assert log._file_handle is None

    @pytest.mark.asyncio
    async def test_high_volume_events(self, tmp_dir):
        """大量事件不丢失、不阻塞。"""
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("高容量测试")

        tasks = []
        for i in range(100):
            tasks.append(log.log(TrajectoryEvent(
                module="worker",
                event_type=EventType.WORKER_STEP,
                payload={"step": i},
                cost=10, latency=0.01,
            )))
        await asyncio.gather(*tasks)

        session = await log.end_session()
        # 100 worker.step + session.start + session.end
        assert session.total_steps >= 100
        assert len(session.events) >= 100

    @pytest.mark.asyncio
    async def test_session_end_includes_start_event(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        await log.start_session("包含测试")
        session = await log.end_session()

        start_events = session.filter_by_type(EventType.SESSION_START)
        end_events = session.filter_by_type(EventType.SESSION_END)
        assert len(start_events) == 1
        assert len(end_events) == 1

    @pytest.mark.asyncio
    async def test_filename_matches_session_id(self, tmp_dir):
        log = TrajectoryLogger(output_dir=tmp_dir)
        sid = await log.start_session("文件名测试")
        await log.end_session()

        expected_file = Path(tmp_dir) / f"{sid}.jsonl"
        assert expected_file.exists()

    @pytest.mark.asyncio
    async def test_multiple_loggers_no_conflict(self, tmp_dir):
        """多个 logger 实例不会冲突。"""
        log1 = TrajectoryLogger(output_dir=Path(tmp_dir) / "log1")
        log2 = TrajectoryLogger(output_dir=Path(tmp_dir) / "log2")

        await log1.start_session("A")
        await log2.start_session("B")

        await log1.log(TrajectoryEvent(module="a", event_type=EventType.WORKER_COMPLETE))
        await log2.log(TrajectoryEvent(module="b", event_type=EventType.WORKER_COMPLETE))

        s1 = await log1.end_session()
        s2 = await log2.end_session()

        assert s1.user_task == "A"
        assert s2.user_task == "B"
        assert len(s1.events) >= 2  # start + worker (+ end)
        assert len(s2.events) >= 2  # start + worker (+ end)
