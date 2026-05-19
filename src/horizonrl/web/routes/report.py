"""GET /api/report/{sid} · GET /api/download/{sid}/{kind} · GET /api/download/{sid}/pdf。"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from horizonrl.web.models import SessionStatusResponse

router = APIRouter()

_DEFAULT_EXPORT_DIR = "reports"


def _find_report_file(session_id: str, kind: str) -> Path | None:
    filename = "final_answer.md" if kind == "final" else "debug_report.md"
    candidate = Path(_DEFAULT_EXPORT_DIR) / session_id / filename
    if candidate.is_file():
        return candidate
    return None


def _safe_filename(text: str, ext: str) -> str:
    """从文本生成安全的文件名。"""
    safe = re.sub(r'[\\/:*?"<>|]', '', text)
    safe = safe.strip().replace(' ', '_')[:50]
    return f"{safe}.{ext}" if safe else f"report.{ext}"


@router.get("/api/report/{session_id}")
async def handle_report(session_id: str, request: Request):
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
            "download_url_pdf": f"/api/download/{session_id}/pdf",
            "runtime_ms": session.runtime_ms,
            "final_path": final_path,
            "debug_path": debug_path,
        })
    elif session.status == "failed":
        kwargs["error"] = session.error

    return SessionStatusResponse(**kwargs)


# ── PDF 路由必须在泛型 /{kind} 路由之前 ──────────────────────────────────

@router.get("/api/download/{session_id}/pdf")
async def handle_download_pdf(session_id: str, request: Request):
    """下载 PDF 报告。需要 weasyprint + GTK 库 (Linux)。"""
    sm = request.app.state.session_manager
    session = sm.get(session_id)

    if session is None:
        return JSONResponse(status_code=404, content={"error": "会话不存在"})

    filepath = session.final_answer_path
    if not filepath or not Path(filepath).is_file():
        found = _find_report_file(session_id, "final")
        if found:
            filepath = str(found)
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "报告未找到，请先完成研究"},
            )

    try:
        pdf_bytes = _markdown_to_pdf(filepath, session.query or "研究报告")
    except ImportError:
        return JSONResponse(
            status_code=500,
            content={
                "error": "PDF 导出需要 weasyprint",
                "detail": "Ubuntu: sudo apt install libgtk-3-dev && pip install weasyprint",
            },
        )
    except OSError:
        return JSONResponse(
            status_code=500,
            content={
                "error": "PDF 引擎 GTK 库未安装",
                "detail": (
                    "Ubuntu/Debian: sudo apt install libpango-1.0-0 libgdk-pixbuf2.0-0 libcairo2\n"
                    "或下载 Markdown 后用 pandoc: pandoc report.md -o report.pdf"
                ),
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "PDF 生成失败", "detail": str(e)},
        )

    filename = _safe_filename(session.query or "report", "pdf")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── 泛型下载路由 (final / debug) ────────────────────────────────────────

@router.get("/api/download/{session_id}/{kind}")
async def handle_download(session_id: str, kind: str, request: Request):
    """下载 Markdown 报告 (final 或 debug)。"""
    if kind not in ("final", "debug"):
        return JSONResponse(status_code=400, content={"error": "无效的文件类型，可选: final, debug, pdf"})

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


# ── PDF 生成 ──────────────────────────────────────────────────────────────

def _markdown_to_pdf(md_path: str | Path, title: str = "研究报告") -> bytes:
    """Markdown → HTML → PDF (weasyprint 管道)。"""
    import markdown

    md_text = Path(md_path).read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["extra", "codehilite", "tables", "toc"],
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', sans-serif;
         max-width: 800px; margin: 40px auto; padding: 0 20px;
         font-size: 14px; line-height: 1.8; color: #333; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #58a6ff; padding-bottom: 8px; }}
  h2 {{ font-size: 18px; margin-top: 24px; color: #1a1a2e; }}
  h3 {{ font-size: 15px; color: #333; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 13px; }}
  pre {{ background: #1a1a2e; color: #e6edf3; padding: 14px; border-radius: 8px;
         overflow-x: auto; font-size: 12px; line-height: 1.6; }}
  blockquote {{ border-left: 3px solid #58a6ff; padding: 6px 14px;
                color: #555; margin: 12px 0; background: #f8f9fa; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #58a6ff; color: #fff; }}
  a {{ color: #58a6ff; }}
</style>
</head>
<body>{html_body}</body>
</html>"""

    from weasyprint import HTML
    doc = HTML(string=html)
    return doc.write_pdf()
