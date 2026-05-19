"""GET /api/report/{sid} 和 GET /api/download/{sid}/{kind}。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

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


@router.get("/api/download/{session_id}/pdf")
async def handle_download_pdf(session_id: str, request: Request):
    """下载 PDF 报告（final_answer.md → HTML → PDF）。

    需要安装 weasyprint: pip install weasyprint
    """
    sm = request.app.state.session_manager
    session = sm.get(session_id)

    if session is None:
        return JSONResponse(status_code=404, content={"error": "会话不存在"})

    # 查找 markdown 文件
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

    # 生成 PDF
    try:
        pdf_bytes = _markdown_to_pdf(filepath, session.query or "研究报告")
    except ImportError:
        return JSONResponse(
            status_code=500,
            content={
                "error": "PDF 导出需要 weasyprint + GTK 库",
                "detail": "Ubuntu: apt install libgtk-3-dev && pip install weasyprint",
            },
        )
    except OSError:
        return JSONResponse(
            status_code=500,
            content={
                "error": "PDF 引擎 GTK 库未安装",
                "detail": "Ubuntu/Debian: sudo apt install libpango-1.0-0 libgdk-pixbuf2.0-0 libcairo2\n"
                          "或下载 Markdown 格式后用 pandoc 转换: pandoc report.md -o report.pdf",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "PDF 生成失败", "detail": str(e)},
        )

    from fastapi.responses import Response

    filename = f"{session.query[:30] or 'report'}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _markdown_to_pdf(md_path: str | Path, title: str = "研究报告") -> bytes:
    """将 Markdown 文件转换为 PDF 字节流。

    使用 markdown → HTML → weasyprint 管道。
    """
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
