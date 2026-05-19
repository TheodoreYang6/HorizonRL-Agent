"""GET /api/stream/{sid} — SSE 实时进度推送（asyncio.Queue 桥接）。"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from horizonrl.config.settings import RootConfig, load_config
from horizonrl.llm.client import LLMClient
from horizonrl.services.research_service import stream_research_session

router = APIRouter()


def _format_sse(event: str, data: dict) -> str:
    """将事件格式化为 SSE 协议字符串。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _update_session(sm, sid: str, evt_type: str, data: dict):
    """根据事件类型更新会话状态。"""
    if evt_type == "stage":
        sm.update(sid, phase=data.get("stage", ""), label=data.get("label", ""))
    elif evt_type == "report_ready":
        sm.update(
            sid,
            final_answer_path=data.get("final_answer_path", ""),
            debug_report_path=data.get("debug_report_path", ""),
        )
    elif evt_type == "done":
        session = sm.get(sid)
        if session:
            # 追加到对话历史（多轮对话用）
            new_history = list(session.conversation_history)
            new_history.append({"role": "user", "content": session.query[:300]})
            new_history.append({"role": "assistant", "content": (data.get("final_answer_text", "") or "")[:500]})
            sm.update(
                sid,
                status="completed",
                final_answer=data.get("final_answer_text", ""),
                runtime_ms=data.get("runtime_ms", 0),
                conversation_history=new_history,
            )
        else:
            sm.update(
                sid,
                status="completed",
                final_answer=data.get("final_answer_text", ""),
                runtime_ms=data.get("runtime_ms", 0),
            )
    elif evt_type == "error":
        sm.update(sid, status="failed", error=data.get("error", ""))


@router.get("/api/stream/{session_id}")
async def handle_stream(session_id: str, request: Request):
    """SSE 端点：实时推送研究进度。

    事件类型: stage | tool | verify | token | report_ready | done | error | heartbeat

    防重入:
      - completed: 直接回放已存事件（不重新执行）
      - running: 返回 409 冲突（不允许重复连接）
      - queued: 正常启动研究管道
    """
    sm = request.app.state.session_manager
    session = sm.get(session_id)

    if session is None or not session.query:
        return JSONResponse(status_code=404, content={"error": "session not found"})

    # 已完成 → 回放已存事件，不重新执行
    if session.status == "completed":
        async def replay_events():
            for evt in session.events:
                yield _format_sse(evt["type"], evt["data"])
                await asyncio.sleep(0)  # 让出事件循环
            # 重新发送 report_ready 和 done（确保客户端能下载）
            if session.final_answer_path:
                yield _format_sse("report_ready", {
                    "session_id": session_id,
                    "final_answer_path": session.final_answer_path,
                    "debug_report_path": session.debug_report_path,
                })
            yield _format_sse("done", {
                "session_id": session_id,
                "mode_resolved": "deep",
                "final_answer_text": session.final_answer[:500],
                "runtime_ms": session.runtime_ms,
            })

        return StreamingResponse(
            replay_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 正在运行 → 拒绝重复连接
    if session.status == "running":
        return JSONResponse(
            status_code=409,
            content={"error": "会话正在运行中，请勿重复连接"},
        )

    # 失败/排队 → 允许重新执行
    query = session.query
    # 多轮对话：注入历史上下文
    if session.conversation_history:
        ctx_parts = []
        for h in session.conversation_history[-4:]:
            role_label = "用户" if h["role"] == "user" else "助手"
            ctx_parts.append(f"[{role_label}]: {h['content'][:300]}")
        if ctx_parts:
            query = "对话背景:\n" + "\n".join(ctx_parts) + f"\n\n当前问题: {query}"
    sm.update(session_id, status="running")
    # 重置事件列表（如果是重试）
    session.events.clear()

    async def event_generator():
        # 队列承载所有 SSE 事件（token 回调 + 主循环）
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _push_token(token: str):
            """Token 回调：将 LLM 流式 token 推入队列。"""
            await queue.put(_format_sse("token", {"delta": token}))

        async def _run_session():
            """执行研究管道，将所有事件推入队列。"""
            try:
                cfg = load_config(
                    Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None
                )
            except Exception:
                cfg = RootConfig()

            llm_client = None
            if cfg.llm.api_key:
                try:
                    llm_client = LLMClient(cfg.llm)
                except Exception:
                    pass

            try:
                async for event in stream_research_session(
                    query=query,
                    mode="deep",
                    session_id=session_id,  # 确保文件路径与 URL session_id 一致
                    llm_client=llm_client,
                    config=cfg,
                    export_dir="reports",
                    on_token=_push_token,
                ):
                    evt_type = event["event"]
                    data = event["data"]

                    # 同步更新会话状态（供 /api/report 查询和 /api/download 下载）
                    session.events.append({"type": evt_type, "data": data})
                    _update_session(sm, session_id, evt_type, data)

                    await queue.put(_format_sse(evt_type, data))

                # 确保最终状态
                final_session = sm.get(session_id)
                if final_session and final_session.status == "running":
                    final_text = ""
                    if final_session.final_answer_path and Path(final_session.final_answer_path).exists():
                        final_text = Path(final_session.final_answer_path).read_text(encoding="utf-8")[:500]
                    sm.update(session_id, status="completed", final_answer=final_text)

            except Exception as exc:
                sm.update(session_id, status="failed", error=str(exc))
                await queue.put(_format_sse("sse_error", {"error": str(exc)}))
            finally:
                await queue.put(None)  # 哨兵：流结束

        # 启动生产者任务
        task = asyncio.create_task(_run_session())

        # 消费者：从队列读取并 yield SSE 字符串
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                # 15s 无事件 → 发送心跳保活
                yield _format_sse("heartbeat", {"ts": time.monotonic()})
                continue

            if item is None:  # 哨兵
                break
            yield item

        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
