"""会话管理器单元测试。"""

from __future__ import annotations

import time
import pytest
from horizonrl.web.session_manager import SessionState, SessionManager


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


class TestSessionManager:
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
        # re-read
        state = sm.get("sid1")
        assert state.status == "running"

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
        sm = SessionManager(ttl_seconds=0)  # immediate expiry
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
