"""会话管理器单元测试 — 内存 + SQLite 双后端。"""

from __future__ import annotations

import time

import pytest

from horizonrl.web.session_manager import (
    SessionManager,
    SessionState,
    SqliteSessionManager,
    create_session_manager,
)


class TestSessionState:
    def test_create_default_state(self):
        state = SessionState(session_id="test", query="hello")
        assert state.session_id == "test"
        assert state.query == "hello"
        assert state.status == "queued"
        assert state.events == []
        assert state.created_at > 0

    def test_field_mutable(self):
        state = SessionState(session_id="s1", query="q")
        state.status = "running"
        state.phase = "executing"
        state.events.append({"type": "test"})
        assert state.status == "running"
        assert state.phase == "executing"
        assert len(state.events) == 1

    def test_to_dict_and_from_dict_roundtrip(self):
        state = SessionState(
            session_id="sid",
            query="测试问题",
            status="completed",
            phase="writing",
            events=[{"type": "stage", "data": {"stage": "plan"}}],
            final_answer="这是答案",
            final_answer_path="/tmp/report.md",
            runtime_ms=1234.5,
        )
        d = state.to_dict()
        restored = SessionState.from_dict(d)
        assert restored.session_id == state.session_id
        assert restored.query == state.query
        assert restored.status == state.status
        assert restored.events == state.events
        assert restored.final_answer == state.final_answer
        assert restored.runtime_ms == state.runtime_ms

    def test_from_dict_handles_json_string_events(self):
        """from_dict 会将 events 的 JSON 字符串反序列化为 list。"""
        d = {
            "session_id": "sid",
            "query": "q",
            "events": '[{"type":"test"}]',
        }
        state = SessionState.from_dict(d)
        assert state.events == [{"type": "test"}]

    def test_from_dict_defaults_missing_fields(self):
        state = SessionState.from_dict({"session_id": "s", "query": "q"})
        assert state.status == "queued"
        assert state.events == []
        assert state.runtime_ms == 0.0


class TestSessionManager:
    """内存模式测试。"""

    def test_create_and_get(self):
        sm = SessionManager()
        sm.create("sid1", "query1")
        state = sm.get("sid1")
        assert state is not None
        assert state.query == "query1"

    def test_get_nonexistent(self):
        sm = SessionManager()
        assert sm.get("noexist") is None

    def test_update_existing(self):
        sm = SessionManager()
        sm.create("sid1", "q")
        result = sm.update("sid1", status="running", phase="planning")
        assert result is not None
        assert result.status == "running"
        assert result.phase == "planning"

    def test_update_nonexistent(self):
        sm = SessionManager()
        assert sm.update("noexist", status="running") is None

    def test_delete(self):
        sm = SessionManager()
        sm.create("sid1", "q")
        assert sm.delete("sid1") is True
        assert sm.get("sid1") is None
        assert sm.delete("sid1") is False

    def test_active_count(self):
        sm = SessionManager()
        assert sm.active_count == 0
        sm.create("a", "1")
        sm.create("b", "2")
        assert sm.active_count == 2

    def test_cleanup_expired(self):
        sm = SessionManager(ttl_seconds=0)
        sm.create("old", "query")
        time.sleep(0.01)
        count = sm.cleanup_expired()
        assert count == 1
        assert sm.get("old") is None

    def test_cleanup_keeps_fresh(self):
        sm = SessionManager(ttl_seconds=3600)
        sm.create("fresh", "query")
        count = sm.cleanup_expired()
        assert count == 0
        assert sm.get("fresh") is not None

    def test_list_all_order(self):
        sm = SessionManager()
        sm.create("a", "first")
        time.sleep(0.01)
        sm.create("b", "second")
        sessions = sm.list_all()
        assert len(sessions) == 2
        assert sessions[0].session_id == "b"  # 最新在前

    def test_same_reference_for_events_mutation(self):
        """内存模式下 get() 返回同一对象，原地修改有效。"""
        sm = SessionManager()
        sm.create("sid1", "q")
        state = sm.get("sid1")
        state.events.append({"type": "test"})
        # 不调 update，直接 get 确认已修改
        state2 = sm.get("sid1")
        assert len(state2.events) == 1


class TestSqliteSessionManager:
    """SQLite 持久化模式测试。"""

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "test_sessions.db")

    def test_create_and_get(self, db_path):
        sm = SqliteSessionManager(db_path=db_path)
        sm.create("sid1", "query1")
        state = sm.get("sid1")
        assert state is not None
        assert state.query == "query1"

    def test_get_nonexistent(self, db_path):
        sm = SqliteSessionManager(db_path=db_path)
        assert sm.get("noexist") is None

    def test_update_and_persist(self, db_path):
        sm = SqliteSessionManager(db_path=db_path)
        sm.create("sid1", "q")
        sm.update("sid1", status="running", phase="planning")
        # 同一个 sm 实例内
        state = sm.get("sid1")
        assert state.status == "running"
        assert state.phase == "planning"
        # 新建 sm 实例（模拟重启）
        sm2 = SqliteSessionManager(db_path=db_path)
        state2 = sm2.get("sid1")
        assert state2.status == "running"
        assert state2.phase == "planning"

    def test_same_reference_for_events_mutation(self, db_path):
        """缓存保证 get() 返回同一对象引用，原地修改后 update 可持久化。"""
        sm = SqliteSessionManager(db_path=db_path)
        sm.create("sid1", "q")
        state = sm.get("sid1")
        state.events.append({"type": "stage", "data": {"stage": "plan"}})
        state.events.append({"type": "done", "data": {}})
        # 调 update 触发 DB 同步
        sm.update("sid1", status="running")
        # 新实例验证事件已持久化
        sm2 = SqliteSessionManager(db_path=db_path)
        state2 = sm2.get("sid1")
        assert len(state2.events) == 2
        assert state2.events[0]["type"] == "stage"
        assert state2.events[1]["type"] == "done"

    def test_flush_syncs_events(self, db_path):
        """flush() 无需修改其他字段即可同步 events。"""
        sm = SqliteSessionManager(db_path=db_path)
        sm.create("sid1", "q")
        state = sm.get("sid1")
        state.events.append({"type": "token", "data": {"delta": "hello"}})
        sm.flush("sid1")
        sm2 = SqliteSessionManager(db_path=db_path)
        assert len(sm2.get("sid1").events) == 1

    def test_delete(self, db_path):
        sm = SqliteSessionManager(db_path=db_path)
        sm.create("sid1", "q")
        assert sm.delete("sid1") is True
        assert sm.get("sid1") is None
        assert sm.delete("sid1") is False

    def test_delete_removes_from_cache(self, db_path):
        sm = SqliteSessionManager(db_path=db_path)
        sm.create("sid1", "q")
        sm.get("sid1")  # 确保缓存
        sm.delete("sid1")
        # 新建实例确认 DB 中也没有
        sm2 = SqliteSessionManager(db_path=db_path)
        assert sm2.get("sid1") is None

    def test_list_all_order(self, db_path):
        sm = SqliteSessionManager(db_path=db_path)
        sm.create("a", "first")
        time.sleep(0.01)
        sm.create("b", "second")
        sessions = sm.list_all()
        assert len(sessions) == 2
        assert sessions[0].session_id == "b"

    def test_list_all_returns_cached_objects(self, db_path):
        """list_all 返回缓存中的活跃对象（events 可能已更新）。"""
        sm = SqliteSessionManager(db_path=db_path)
        sm.create("sid1", "q")
        state = sm.get("sid1")
        state.events.append({"type": "new_event"})
        sessions = sm.list_all()
        cached = [s for s in sessions if s.session_id == "sid1"][0]
        assert len(cached.events) == 1

    def test_count(self, db_path):
        sm = SqliteSessionManager(db_path=db_path)
        assert sm.count() == 0
        sm.create("a", "1")
        sm.create("b", "2")
        assert sm.count() == 2

    def test_cleanup_expired(self, db_path):
        sm = SqliteSessionManager(db_path=db_path, ttl_seconds=0)
        sm.create("old", "query")
        time.sleep(0.01)
        count = sm.cleanup_expired()
        assert count == 1
        assert sm.get("old") is None

    def test_cleanup_keeps_fresh(self, db_path):
        sm = SqliteSessionManager(db_path=db_path, ttl_seconds=3600)
        sm.create("fresh", "query")
        count = sm.cleanup_expired()
        assert count == 0
        assert sm.get("fresh") is not None

    def test_cleanup_skips_running(self, db_path):
        sm = SqliteSessionManager(db_path=db_path, ttl_seconds=0)
        sm.create("running_session", "q")
        sm.update("running_session", status="running")
        time.sleep(0.01)
        count = sm.cleanup_expired()
        assert count == 0  # running 状态不删除

    def test_update_nonexistent(self, db_path):
        sm = SqliteSessionManager(db_path=db_path)
        assert sm.update("noexist", status="running") is None


class TestCreateSessionManager:
    """工厂函数测试。"""

    def test_memory_backend(self):
        sm = create_session_manager(backend="memory")
        assert isinstance(sm, SessionManager)

    def test_sqlite_backend(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        sm = create_session_manager(backend="sqlite", db_path=db_path)
        assert isinstance(sm, SqliteSessionManager)

    def test_default_is_memory(self):
        sm = create_session_manager()
        assert isinstance(sm, SessionManager)
