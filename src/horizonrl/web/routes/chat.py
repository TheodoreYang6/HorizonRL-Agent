"""POST /api/chat — 统一对话入口（chat/deep 双模式）。"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from horizonrl.config.settings import RootConfig, load_config
from horizonrl.services.research_service import resolve_mode
from horizonrl.web.models import AgentResponse, ChatRequest, ChatResponse, ErrorResponse

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
    - 多轮对话：传入 session_id 则继承父会话上下文
    """
    message = body.message.strip()
    if not message:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="无效问题", detail="消息不能为空").model_dump(),
        )

    resolved = resolve_mode(message, body.mode)
    sm = request.app.state.session_manager

    # 多轮对话：继承已有会话上下文 + 语义检索相关历史
    parent_sid = body.session_id or ""
    conversation_history = []
    parent_query = ""
    parent_answer = ""
    related_contexts = []

    if parent_sid and sm.get(parent_sid):
        parent = sm.get(parent_sid)
        if parent and parent.status in ("completed", "failed"):
            conversation_history = list(parent.conversation_history)
            conversation_history.append({
                "role": "user",
                "content": parent.query[:300],
            })
            conversation_history.append({
                "role": "assistant",
                "content": (parent.final_answer or "")[:500],
            })
            parent_query = parent.query
            parent_answer = parent.final_answer or ""

        # 语义检索相关历史研究 (Research Context Engine)
        try:
            from horizonrl.memory.research_context import get_context_store
            store = get_context_store()
            if store.count() > 0:
                related = store.search(message, top_k=2)
                related_contexts = [
                    r for r in related
                    if r["session_id"] != parent_sid  # 排除当前追问链
                ]
        except Exception:
            pass

    if resolved == "chat":
        ctx_parts = []
        # 语义相关历史 (优先)
        for rc in related_contexts[:1]:
            ctx_parts.append(f"[相关研究] {rc['query'][:100]}: {rc['summary'][:200]}")
        # 线性对话历史 (补充)
        for h in conversation_history[-4:]:
            role = "用户" if h["role"] == "user" else "助手"
            ctx_parts.append(f"[{role}]: {h['content'][:200]}")
        if ctx_parts:
            answer = await _run_chat(
                "研究上下文:\n" + "\n".join(ctx_parts) + f"\n\n当前问题: {message}"
            )
        else:
            answer = await _run_chat(message)
        return ChatResponse(mode="chat", answer=answer).model_dump()

    # deep 模式: 将语义相关历史注入 conversation_history
    if related_contexts:
        for rc in related_contexts:
            conversation_history.insert(0, {
                "role": "assistant",
                "content": f"[相关历史] {rc['query'][:100]}: {rc['summary'][:200]}",
            })

    sid = f"session_{uuid.uuid4().hex[:12]}"
    sm.create(
        sid, message,
        parent_session_id=parent_sid,
        conversation_history=conversation_history,
    )
    return AgentResponse(
        mode="agent",
        session_id=sid,
        status="queued",
        # 传递追问上下文给前端
        **({"parent_query": parent_query, "parent_answer": parent_answer[:300]}
           if parent_sid else {}),
    ).model_dump()
