"""请求/响应数据模型 — Pydantic V2。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/chat 请求体。"""
    message: str = Field(..., min_length=1, max_length=500, description="用户输入")
    mode: str = Field(default="auto", pattern="^(auto|chat|deep)$", description="模式")
    session_id: str | None = Field(default=None, description="多轮对话: 继续已有会话")


class ChatResponse(BaseModel):
    """对话模式同步响应。"""
    mode: str  # "chat"
    answer: str


class AgentResponse(BaseModel):
    """深度研究模式响应（返回 session_id，前端通过 SSE 订阅进度）。"""
    model_config = {"extra": "allow"}
    mode: str  # "agent"
    session_id: str
    status: str = "queued"


class ErrorResponse(BaseModel):
    """统一错误响应。"""
    error: str
    detail: str | None = None


class SessionStatusResponse(BaseModel):
    """GET /api/report/{sid} 响应。"""
    model_config = {"extra": "allow"}
    status: str
    phase: str = ""
    label: str = ""
    events: list[dict] = []
    final_answer: str | None = None
    download_url_final: str | None = None
    download_url_debug: str | None = None
    runtime_ms: float | None = None
    final_path: str | None = None
    debug_path: str | None = None
    error: str | None = None
