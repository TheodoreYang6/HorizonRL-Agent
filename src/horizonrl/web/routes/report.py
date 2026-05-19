"""GET /api/report/{sid} · GET /api/download/{sid}/{kind}。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from horizonrl.web.models import SessionStatusResponse

router = APIRouter()
_DEFAULT_EXPORT_DIR = "reports"


def _find_report_file(session_id: str, kind: str) -> Path | None:
    filename = "final_answer.md" if kind == "final" else "debug_report.md"
    candidate = Path(_DEFAULT_EXPORT_DIR) / session_id / filename
    return candidate if candidate.is_file() else None


@router.get("/api/report/{session_id}")
async def handle_report(session_id: str, request: Request):
    sm = request.app.state.session_manager
    session = sm.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "session not found"})

    kwargs: dict = {
        "status": session.status, "phase": session.phase,
        "label": session.label, "events": session.events,
    }
    if session.status == "completed":
        fp = session.final_answer_path
        dp = session.debug_report_path
        if not fp or not Path(fp).is_file():
            f = _find_report_file(session_id, "final")
            fp = str(f) if f else fp
        if not dp or not Path(dp).is_file():
            f = _find_report_file(session_id, "debug")
            dp = str(f) if f else dp
        kwargs.update({
            "final_answer": session.final_answer or "",
            "download_url_final": f"/api/download/{session_id}/final",
            "download_url_debug": f"/api/download/{session_id}/debug",
            "runtime_ms": session.runtime_ms,
            "final_path": fp, "debug_path": dp,
        })
    elif session.status == "failed":
        kwargs["error"] = session.error
    return SessionStatusResponse(**kwargs)


@router.get("/api/download/{session_id}/{kind}")
async def handle_download(session_id: str, kind: str, request: Request):
    if kind not in ("final", "debug"):
        return JSONResponse(status_code=400, content={"error": "无效的文件类型"})

    sm = request.app.state.session_manager
    session = sm.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "会话不存在"})

    path_attr = "final_answer_path" if kind == "final" else "debug_report_path"
    filepath = getattr(session, path_attr, "")
    if not filepath or not Path(filepath).is_file():
        found = _find_report_file(session_id, kind)
        if found:
            filepath = str(found)
        else:
            return JSONResponse(status_code=404, content={"error": "报告文件未找到"})

    filename = "final_answer.md" if kind == "final" else "debug_report.md"
    return FileResponse(
        path=filepath, media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
