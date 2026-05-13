"""
Trajectory Logger — 异步轨迹日志系统。

将 Agent 执行全生命周期的结构化的 TrajectoryEvent 异步写入 JSONL 文件。
轨迹日志是 HorizonRL-Agent 的"一等基础设施"——它是后续消融实验、
成功率分析和 RL 训练的数据源。

特性：
    - 异步非阻塞写入（asyncio.Queue + 后台 writer task）
    - JSONL 格式，一行一个事件，方便 grep/jq/pandas 分析
    - 自动管理 TrajectorySession 生命周期
    - 会话结束时输出统计摘要
    - 提供读取/分析工具函数

使用方式：
    logger = TrajectoryLogger(output_dir="trajectories")
    await logger.start_session("Transformer 注意力机制研究")
    await logger.log(TrajectoryEvent(
        module="planner", event_type=EventType.PLAN_COMPLETE,
        payload={"num_subtasks": 5}, cost=1200, latency=3.5,
    ))
    session = await logger.end_session(success=True)
    print(session.to_summary())
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from horizonrl.schemas.event import (
    EventType,
    TrajectoryEvent,
    TrajectorySession,
)

# ─── 常量 ────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "trajectories"
QUEUE_SIZE = 10_000  # 缓冲队列上限


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TrajectoryLogger — 异步轨迹日志器                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class TrajectoryLogger:
    """异步轨迹日志器 —— JSONL 格式，非阻塞写入。

    Examples:
        >>> logger = TrajectoryLogger()
        >>> await logger.start_session("研究任务")
        >>> await logger.log(event)
        >>> session = await logger.end_session(success=True)
    """

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._session: TrajectorySession | None = None
        self._queue: asyncio.Queue[TrajectoryEvent | None] | None = None
        self._writer_task: asyncio.Task | None = None
        self._file_handle = None
        self._event_count: int = 0
        self._closed: bool = False

    # ── Session 管理 ──────────────────────────────────────────────────────

    async def start_session(self, user_task: str) -> str:
        """开始新的日志会话。

        Args:
            user_task: 用户的研究问题。

        Returns:
            session_id。
        """
        if self._session is not None:
            await self.end_session(success=False)

        session_id = f"session_{uuid.uuid4().hex[:12]}"
        self._session = TrajectorySession(
            session_id=session_id,
            user_task=user_task,
            started_at=time.time(),
        )
        self._event_count = 0
        self._closed = False

        # 打开 JSONL 文件
        filepath = self.output_dir / f"{session_id}.jsonl"
        self._file_handle = open(str(filepath), "w", encoding="utf-8")

        # 启动缓冲队列和后台 writer
        self._queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._writer_task = asyncio.create_task(self._writer_loop())

        # 记录 session start 事件
        await self.log(TrajectoryEvent(
            module="system",
            event_type=EventType.SESSION_START,
            payload={"session_id": session_id, "user_task": user_task},
            session_id=session_id,
        ))

        return session_id

    async def end_session(self, success: bool = False) -> TrajectorySession:
        """结束当前会话，刷新所有缓冲事件并输出统计。

        Args:
            success: 最终是否成功完成。

        Returns:
            完整的 TrajectorySession（含所有事件和统计）。
        """
        if self._session is None:
            raise RuntimeError("没有活跃的会话")

        self._session.success = success
        self._session.finished_at = time.time()
        self._session.total_steps = len(self._session.events)

        # 写入 session end 事件
        await self.log(TrajectoryEvent(
            module="system",
            event_type=EventType.SESSION_END,
            payload=self._session.to_summary(),
            session_id=self._session.session_id,
        ))

        # 发送结束信号并等待 writer 完成
        if self._queue is not None:
            await self._queue.put(None)  # 哨兵，通知 writer 退出

        if self._writer_task is not None:
            try:
                await asyncio.wait_for(self._writer_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._writer_task.cancel()

        # 关闭文件
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

        self._closed = True
        session = self._session
        return session

    # ── 事件记录 ──────────────────────────────────────────────────────────

    async def log(self, event: TrajectoryEvent) -> None:
        """异步记录一条轨迹事件（非阻塞）。

        将事件放入缓冲队列，由后台 writer 异步写入磁盘。
        同时更新 session 统计。
        """
        if self._queue is None:
            raise RuntimeError("未启动会话，请先调用 start_session()")

        if self._session is not None:
            self._session.add_event(event)

        await self._queue.put(event)
        self._event_count += 1

    def log_nowait(self, event: TrajectoryEvent) -> None:
        """同步记录事件（可能在事件循环未运行时调用）。

        非阻塞放入队列，同时更新 session 统计。队列满时丢弃事件。
        """
        if self._session is not None:
            self._session.add_event(event)
        if self._queue is not None:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
            self._event_count += 1

    # ── 刷新 ──────────────────────────────────────────────────────────────

    async def flush(self) -> None:
        """强制刷新：等待队列中所有事件写入磁盘。"""
        if self._queue is None:
            return
        # 放入一个刷新标记
        flush_event = TrajectoryEvent(
            module="system",
            event_type=EventType.ERROR,  # 临时借用
            payload={"_flush": True},
        )
        await self._queue.put(flush_event)
        # 等待队列清空
        while not self._queue.empty():
            await asyncio.sleep(0.01)

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def session(self) -> TrajectorySession | None:
        return self._session

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def is_active(self) -> bool:
        return self._session is not None and not self._closed

    # ── 内部：后台 Writer ─────────────────────────────────────────────────

    async def _writer_loop(self) -> None:
        """后台任务：从队列消费事件并写入 JSONL 文件。"""
        while True:
            event = await self._queue.get()

            if event is None:  # 哨兵：结束信号
                self._queue.task_done()
                break

            # 跳过内部刷新标记
            if event.payload.get("_flush"):
                self._queue.task_done()
                continue

            # 序列化并写入
            try:
                line = json.dumps(event.to_dict(), ensure_ascii=False)
                if self._file_handle is not None:
                    self._file_handle.write(line + "\n")
                    self._file_handle.flush()
            except Exception:
                pass  # 写入失败不阻塞主流程

            self._queue.task_done()

    async def close(self) -> None:
        """关闭日志器，清理资源。"""
        if self._session is not None and not self._closed:
            await self.end_session(success=False)
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  便捷工厂函数                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def create_logger(output_dir: str = DEFAULT_OUTPUT_DIR) -> TrajectoryLogger:
    """创建 TrajectoryLogger 实例的便捷工厂。"""
    return TrajectoryLogger(output_dir=output_dir)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  读取 & 分析工具                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def read_session(filepath: str | Path) -> TrajectorySession:
    """从 JSONL 文件读取一个会话。

    Args:
        filepath: JSONL 文件路径。

    Returns:
        重建的 TrajectorySession。
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"日志文件不存在: {filepath}")

    events: list[TrajectoryEvent] = []
    with open(str(filepath), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event = TrajectoryEvent(
                    ts=data.get("ts", 0),
                    module=data.get("module", ""),
                    event_type=EventType(data.get("event_type", "system.error")),
                    payload=data.get("payload", {}),
                    cost=data.get("cost", 0),
                    latency=data.get("latency", 0.0),
                    session_id=data.get("session_id", ""),
                    step_id=data.get("step_id", 0),
                )
                events.append(event)
            except (json.JSONDecodeError, ValueError):
                continue

    if not events:
        return TrajectorySession()

    # 从 session.start 事件恢复元数据
    first = events[0]
    session = TrajectorySession(
        session_id=first.session_id,
        user_task=first.payload.get("user_task", ""),
        started_at=first.ts,
    )

    # 从 session.end 事件恢复结束状态
    last = events[-1]
    if last.event_type == EventType.SESSION_END:
        session.finished_at = last.ts
        session.success = last.payload.get("success", False)

    for event in events:
        session.add_event(event)

    return session


def list_sessions(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """列出所有日志会话文件。"""
    d = Path(output_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*.jsonl"))


def aggregate_stats(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    """聚合所有会话的统计信息。

    Returns:
        {"total_sessions": int, "total_events": int, "avg_success_rate": float, ...}
    """
    sessions = []
    for f in list_sessions(output_dir):
        try:
            session = read_session(f)
            sessions.append(session)
        except Exception:
            continue

    if not sessions:
        return {"total_sessions": 0}

    total_events = sum(len(s.events) for s in sessions)
    success_count = sum(1 for s in sessions if s.success)
    total_tokens = sum(s.total_tokens for s in sessions)
    total_tool_calls = sum(s.total_tool_calls for s in sessions)
    total_replans = sum(s.replan_count for s in sessions)

    return {
        "total_sessions": len(sessions),
        "total_events": total_events,
        "success_count": success_count,
        "failure_count": len(sessions) - success_count,
        "success_rate": success_count / len(sessions) if sessions else 0.0,
        "total_tokens": total_tokens,
        "total_tool_calls": total_tool_calls,
        "total_replans": total_replans,
        "avg_events_per_session": total_events / len(sessions) if sessions else 0,
        "avg_tokens_per_session": total_tokens / len(sessions) if sessions else 0,
    }


def event_type_distribution(
    filepath: str | Path,
) -> dict[str, int]:
    """统计单个会话中各事件类型的分布。

    Returns:
        {"plan.complete": 1, "worker.complete": 5, ...}
    """
    session = read_session(filepath)
    dist: dict[str, int] = {}
    for event in session.events:
        key = event.event_type.value
        dist[key] = dist.get(key, 0) + 1
    return dist


def filter_events(
    filepath: str | Path,
    module: str | None = None,
    event_type: EventType | None = None,
) -> list[TrajectoryEvent]:
    """按模块和/或事件类型过滤轨迹事件。

    Args:
        filepath: JSONL 文件路径。
        module: 可选，模块名过滤。
        event_type: 可选，事件类型过滤。

    Returns:
        匹配的 TrajectoryEvent 列表。
    """
    session = read_session(filepath)
    events = session.events
    if module:
        events = [e for e in events if e.module == module]
    if event_type:
        events = [e for e in events if e.event_type == event_type]
    return events
