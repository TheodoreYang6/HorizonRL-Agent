"""会话状态管理 — 内存 / SQLite 双后端。

内存模式 (SessionManager): 开发/测试用，重启丢失
SQLite 模式 (SqliteSessionManager): 生产用，持久化存储

切换方式: 设置环境变量 SESSION_BACKEND=sqlite
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class SessionState:
    """单个研究会话的完整状态。"""
    session_id: str
    query: str
    status: str = "queued"           # queued | running | completed | failed
    phase: str = ""
    label: str = ""
    events: list[dict] = field(default_factory=list)
    final_answer: str = ""
    final_answer_path: str = ""
    debug_report_path: str = ""
    runtime_ms: float = 0.0
    error: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "status": self.status,
            "phase": self.phase,
            "label": self.label,
            "events": self.events,
            "final_answer": self.final_answer,
            "final_answer_path": self.final_answer_path,
            "debug_report_path": self.debug_report_path,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SessionState:
        events = d.get("events", [])
        if isinstance(events, str):
            try:
                events = json.loads(events)
            except (json.JSONDecodeError, TypeError):
                events = []
        return cls(
            session_id=d.get("session_id", ""),
            query=d.get("query", ""),
            status=d.get("status", "queued"),
            phase=d.get("phase", ""),
            label=d.get("label", ""),
            events=events,
            final_answer=d.get("final_answer", ""),
            final_answer_path=d.get("final_answer_path", ""),
            debug_report_path=d.get("debug_report_path", ""),
            runtime_ms=d.get("runtime_ms", 0.0),
            error=d.get("error", ""),
            created_at=d.get("created_at", time.time()),
        )


# ── Session Manager Protocol ────────────────────────────────────────────

class SessionManagerProtocol(Protocol):
    """会话管理器接口协议。"""
    def get(self, sid: str) -> SessionState | None: ...
    def create(self, sid: str, query: str) -> SessionState: ...
    def update(self, sid: str, **kwargs) -> SessionState | None: ...
    def delete(self, sid: str) -> bool: ...


# ── In-Memory Session Manager ────────────────────────────────────────────

class SessionManager:
    """管理所有活跃会话的线程安全容器（内存模式）。

    纯内存存储，提供 CRUD + TTL 自动清理。
    适用于开发和测试环境。
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._sessions: dict[str, SessionState] = {}
        self._ttl = ttl_seconds

    def get(self, sid: str) -> SessionState | None:
        return self._sessions.get(sid)

    def create(self, sid: str, query: str) -> SessionState:
        state = SessionState(session_id=sid, query=query)
        self._sessions[sid] = state
        return state

    def update(self, sid: str, **kwargs) -> SessionState | None:
        state = self._sessions.get(sid)
        if state is None:
            return None
        for k, v in kwargs.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state

    def delete(self, sid: str) -> bool:
        return self._sessions.pop(sid, None) is not None

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s.created_at > self._ttl]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def list_all(self, limit: int = 50, offset: int = 0) -> list[SessionState]:
        """列出所有会话，按创建时间倒序。"""
        sorted_sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.created_at,
            reverse=True,
        )
        return sorted_sessions[offset:offset + limit]

    def count(self) -> int:
        return len(self._sessions)


# ── SQLite Session Manager ───────────────────────────────────────────────

class SqliteSessionManager:
    """SQLite 持久化会话管理器。

    会话重启不丢失，支持分页列表查询。
    WAL 模式确保并发安全。
    """

    def __init__(self, db_path: str = "data/sessions.db", ttl_seconds: int = 86400):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT DEFAULT 'queued',
                    phase TEXT DEFAULT '',
                    label TEXT DEFAULT '',
                    events_json TEXT DEFAULT '[]',
                    final_answer TEXT DEFAULT '',
                    final_answer_path TEXT DEFAULT '',
                    debug_report_path TEXT DEFAULT '',
                    runtime_ms REAL DEFAULT 0.0,
                    error TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON sessions(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created
                ON sessions(created_at DESC)
            """)

    # ── CRUD ────────────────────────────────────────────────────────────

    def get(self, sid: str) -> SessionState | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["events"] = d.pop("events_json", "[]")
            return SessionState.from_dict(d)

    def create(self, sid: str, query: str) -> SessionState:
        now = time.time()
        state = SessionState(session_id=sid, query=query, created_at=now)
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO sessions (session_id, query, status, phase, label,
                   events_json, final_answer, final_answer_path, debug_report_path,
                   runtime_ms, error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sid, query, state.status, state.phase, state.label,
                 "[]", state.final_answer, state.final_answer_path,
                 state.debug_report_path, state.runtime_ms, state.error,
                 now, now),
            )
        return state

    def update(self, sid: str, **kwargs) -> SessionState | None:
        state = self.get(sid)
        if state is None:
            return None

        # 更新内存中的状态
        for k, v in kwargs.items():
            if hasattr(state, k):
                setattr(state, k, v)

        # 同步到 SQLite
        now = time.time()
        events_json = json.dumps(state.events, ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE sessions SET status=?, phase=?, label=?,
                   events_json=?, final_answer=?, final_answer_path=?,
                   debug_report_path=?, runtime_ms=?, error=?, updated_at=?
                   WHERE session_id=?""",
                (state.status, state.phase, state.label, events_json,
                 state.final_answer, state.final_answer_path,
                 state.debug_report_path, state.runtime_ms, state.error,
                 now, sid),
            )
        return state

    def delete(self, sid: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (sid,)
            )
            return cursor.rowcount > 0

    def cleanup_expired(self) -> int:
        cutoff = time.time() - self._ttl
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE created_at < ? AND status != 'running'",
                (cutoff,),
            )
            return cursor.rowcount

    # ── List ─────────────────────────────────────────────────────────────

    def list_all(self, limit: int = 50, offset: int = 0) -> list[SessionState]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["events"] = d.pop("events_json", "[]")
                results.append(SessionState.from_dict(d))
            return results

    def count(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            return row[0] if row else 0

    @property
    def active_count(self) -> int:
        return self.count()


# ── Factory ──────────────────────────────────────────────────────────────

def create_session_manager(backend: str = "memory", **kwargs) -> SessionManager | SqliteSessionManager:
    """创建会话管理器实例。

    Args:
        backend: "memory" 或 "sqlite"
        **kwargs: 传递给具体后端的参数

    Returns:
        SessionManager 或 SqliteSessionManager
    """
    if backend == "sqlite":
        db_path = kwargs.pop("db_path", "data/sessions.db")
        ttl = kwargs.pop("ttl_seconds", 86400)  # 默认 24 小时
        return SqliteSessionManager(db_path=db_path, ttl_seconds=ttl, **kwargs)
    else:
        ttl = kwargs.pop("ttl_seconds", 3600)  # 默认 1 小时
        return SessionManager(ttl_seconds=ttl, **kwargs)
