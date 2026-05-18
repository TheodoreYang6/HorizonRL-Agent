"""Web API 集成测试 — FastAPI TestClient。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from horizonrl.web.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestIndex:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_index_contains_app_name(self, client):
        resp = client.get("/")
        assert "HorizonRL-Agent" in resp.text


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


class TestAPIDocs:
    def test_swagger_ui_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "HorizonRL-Agent"
        assert "/api/chat" in schema["paths"]
