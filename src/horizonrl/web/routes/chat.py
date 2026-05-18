"""POST /api/chat — 统一对话入口（chat/deep 双模式）。"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from horizonrl.config.settings import load_config, RootConfig
from horizonrl.services.research_service import resolve_mode
from horizonrl.web.models import ChatRequest, ChatResponse, AgentResponse, ErrorResponse

router = APIRouter()


async def _run_chat(query: str) -> str:
    """轻量 LLM 对话。"""
    try:
        cfg = load_config(
            Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None
        )
    except Exception:
        cfg = RootConfig()

    if not cfg.llm.api_key:
        return (
            "你好！我是 HorizonRL-Agent。\n\n"
            "我可以帮你做深度研究：搜索资料、对比分析、汇总报告。\n"
            "试试输入一个研究问题，比如「Transformer注意力机制的最新进展」。"
        )

    from horizonrl.llm.client import LLMClient

    try:
        client = LLMClient(cfg.llm)
        result = await client.chat(
            query,
            system_prompt="你是一个友好、专业的AI助手。用简洁流畅的中文回答。",
            max_tokens=1000,
        )
        return result.content if result.is_success else f"LLM 调用失败: {result.error}"
    except Exception as e:
        return f"LLM 错误: {e}"


@router.post("/api/chat")
async def handle_chat(body: ChatRequest, request: Request):
    """处理对话/研究请求。

    - chat 模式：直接返回 LLM 回答
    - deep 模式：创建会话，返回 session_id 供前端 SSE 订阅
    """
    message = body.message.strip()
    if not message:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="无效问题", detail="消息不能为空").model_dump(),
        )

    resolved = resolve_mode(message, body.mode)

    if resolved == "chat":
        answer = await _run_chat(message)
        return ChatResponse(mode="chat", answer=answer).model_dump()

    # deep 模式：创建会话，前端通过 SSE 获取进度
    sid = f"session_{uuid.uuid4().hex[:12]}"
    sm = request.app.state.session_manager
    sm.create(sid, message)
    return AgentResponse(mode="agent", session_id=sid, status="queued").model_dump()
