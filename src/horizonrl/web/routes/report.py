"""GET /api/report/{sid} 和 GET /api/download/{sid}/{kind}。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, FileResponse

from horizonrl.web.models import SessionStatusResponse

router = APIRouter()

# Writer 默认输出目录，与 stream_research_session 一致
_DEFAULT_EXPORT_DIR = "reports"


def _find_report_file(session_id: str, kind: str) -> Path | None:
    """查找报告文件：先查会话记录路径，再查默认输出目录。"""
    filename = "final_answer.md" if kind == "final" else "debug_report.md"
    candidate = Path(_DEFAULT_EXPORT_DIR) / session_id / filename
    if candidate.is_file():
        return candidate
    return None


@router.get("/api/report/{session_id}")
async def handle_report(session_id: str, request: Request):
    """查询会话状态（页面刷新后恢复进度）。"""
    sm = request.app.state.session_manager
    session = sm.get(session_id)

    if session is None:
        return JSONResponse(status_code=404, content={"error": "session not found"})

    kwargs: dict = {
        "status": session.status,
        "phase": session.phase,
        "label": session.label,
        "events": session.events,
    }

    if session.status == "completed":
        # 优先使用会话记录的路径，回退到默认路径
        final_path = session.final_answer_path
        debug_path = session.debug_report_path
        if not final_path or not Path(final_path).is_file():
            found = _find_report_file(session_id, "final")
            final_path = str(found) if found else final_path
        if not debug_path or not Path(debug_path).is_file():
            found = _find_report_file(session_id, "debug")
            debug_path = str(found) if found else debug_path

        kwargs.update({
            "final_answer": session.final_answer or "",
            "download_url_final": f"/api/download/{session_id}/final",
            "download_url_debug": f"/api/download/{session_id}/debug",
            "runtime_ms": session.runtime_ms,
            "final_path": final_path,
            "debug_path": debug_path,
        })
    elif session.status == "failed":
        kwargs["error"] = session.error

    return SessionStatusResponse(**kwargs)


@router.get("/api/download/{session_id}/{kind}")
async def handle_download(session_id: str, kind: str, request: Request):
    """下载 Markdown 报告文件（final 或 debug）。

    优先从会话记录路径获取，回退到 reports/{sid}/ 目录查找。
    """
    if kind not in ("final", "debug"):
        return JSONResponse(status_code=400, content={"error": "无效的文件类型"})

    sm = request.app.state.session_manager
    session = sm.get(session_id)

    if session is None:
        return JSONResponse(status_code=404, content={"error": "会话不存在，请重新发起研究"})

    # 从会话记录获取路径
    path_attr = "final_answer_path" if kind == "final" else "debug_report_path"
    filepath = getattr(session, path_attr, "")

    # 若路径为空或文件不存在，回退到默认目录查找
    if not filepath or not Path(filepath).is_file():
        found = _find_report_file(session_id, kind)
        if found:
            filepath = str(found)
        else:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "报告文件未找到",
                    "detail": f"会话 {session_id} 的 {kind} 报告尚未生成或已被清理",
                },
            )

    filename = "final_answer.md" if kind == "final" else "debug_report.md"
    return FileResponse(
        path=filepath,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
