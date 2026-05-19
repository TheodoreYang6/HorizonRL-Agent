"""Web API 集成测试 — FastAPI TestClient。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from horizonrl.web.app import create_app
from horizonrl.web.session_manager import SessionManager


@pytest.fixture
def client():
    """使用内存 SessionManager 隔离测试，避免污染 SQLite DB。"""
    sm = SessionManager()
    app = create_app(session_mgr=sm)
    with TestClient(app) as c:
        yield c


class TestIndex:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_index_contains_app_name(self, client):
        resp = client.get("/")
        assert "Horizon-Agent" in resp.text


class TestChatEndpoint:
    def test_empty_message_returns_422(self, client):
        resp = client.post("/api/chat", json={"message": "", "mode": "auto"})
        # Pydantic 校验失败返回 422 (min_length=1)
        assert resp.status_code == 422

    def test_chat_mode_returns_answer(self, client):
        resp = client.post("/api/chat", json={"message": "你好", "mode": "chat"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "chat"
        assert "answer" in data

    def test_deep_mode_returns_session_id(self, client):
        resp = client.post("/api/chat",
                           json={"message": "研究Transformer注意力机制", "mode": "deep"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "agent"
        assert "session_id" in data
        assert data["session_id"].startswith("session_")

    def test_message_too_long_is_rejected(self, client):
        # FastAPI Pydantic validation returns 422 for validation errors
        resp = client.post("/api/chat",
                           json={"message": "x" * 501, "mode": "auto"})
        assert resp.status_code in (400, 422)


class TestReportEndpoint:
    def test_nonexistent_session_returns_404(self, client):
        resp = client.get("/api/report/nonexistent")
        assert resp.status_code == 404

    def test_existing_session_returns_status(self, client):
        # Create a session first
        resp = client.post("/api/chat",
                           json={"message": "研究Transformer注意力机制", "mode": "deep"})
        sid = resp.json()["session_id"]

        resp = client.get(f"/api/report/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"


class TestDownloadEndpoint:
    def test_nonexistent_session_returns_404(self, client):
        resp = client.get("/api/download/nonexistent/final")
        assert resp.status_code == 404

    def test_invalid_kind_returns_400(self, client):
        resp = client.get("/api/download/some_session/invalid")
        assert resp.status_code == 400


class TestStreamEndpoint:
    def test_nonexistent_session_returns_404(self, client):
        resp = client.get("/api/stream/nonexistent")
        assert resp.status_code == 404


class TestStaticFiles:
    def test_css_served(self, client):
        resp = client.get("/static/css/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]

    def test_js_served(self, client):
        resp = client.get("/static/js/app.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"].lower() or \
               "text/" in resp.headers["content-type"].lower()


class TestSessionsAPI:
    """GET/DELETE /api/sessions 端点测试。"""

    def test_list_empty(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_list_with_sessions(self, client):
        # 创建几个会话
        client.post("/api/chat", json={"message": "问题1", "mode": "deep"})
        client.post("/api/chat", json={"message": "问题2", "mode": "deep"})
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["sessions"]) == 2

    def test_list_respects_limit(self, client):
        for i in range(5):
            client.post("/api/chat", json={"message": f"问题{i}", "mode": "deep"})
        resp = client.get("/api/sessions?limit=2&offset=0")
        data = resp.json()
        assert len(data["sessions"]) == 2
        assert data["total"] == 5

    def test_get_session_detail(self, client):
        r = client.post("/api/chat", json={"message": "研究AI", "mode": "deep"})
        sid = r.json()["session_id"]
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["query"] == "研究AI"

    def test_get_session_404(self, client):
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        r = client.post("/api/chat", json={"message": "研究AI", "mode": "deep"})
        sid = r.json()["session_id"]
        resp = client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == sid
        # 确认已删除
        assert client.get(f"/api/sessions/{sid}").status_code == 404

    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/sessions/nonexistent")
        assert resp.status_code == 404


class TestAPIDocs:
    def test_swagger_ui_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Horizon-Agent"
        assert "/api/chat" in schema["paths"]
