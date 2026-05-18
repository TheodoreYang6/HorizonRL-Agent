"""会话历史 API — 列表 / 详情 / 删除。"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/api/sessions")
async def list_sessions(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """获取历史会话列表，按创建时间倒序。"""
    sm = request.app.state.session_manager
    sessions = sm.list_all(limit=limit, offset=offset)
    total = sm.count()

    items = []
    for s in sessions:
        items.append({
            "session_id": s.session_id,
            "query": s.query[:100],
            "status": s.status,
            "runtime_ms": s.runtime_ms,
            "created_at": s.created_at,
        })

    return {"sessions": items, "total": total, "limit": limit, "offset": offset}


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    """获取单个会话详情。"""
    sm = request.app.state.session_manager
    session = sm.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "session not found"})
    return session.to_dict()


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """删除会话及其报告文件。"""
    import shutil
    from pathlib import Path

    sm = request.app.state.session_manager
    session = sm.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "session not found"})

    # 删除报告文件
    for p in (session.final_answer_path, session.debug_report_path):
        if p:
            report_dir = Path(p).parent
            if report_dir.exists():
                shutil.rmtree(report_dir, ignore_errors=True)

    sm.delete(session_id)
    return {"ok": True, "deleted": session_id}
