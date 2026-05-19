"""GET /api/report/{sid} · /api/download/{sid}/pdf · /api/download/{sid}/{kind}。"""

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
    return candidate if candidate.is_file() else None


def _safe_filename(text: str, ext: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]', '', text).strip().replace(' ', '_')[:50]
    return f"{safe}.{ext}" if safe else f"report.{ext}"


# ── Report Status ────────────────────────────────────────────────────────

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
            "download_url_pdf": f"/api/download/{session_id}/pdf",
            "runtime_ms": session.runtime_ms,
            "final_path": fp, "debug_path": dp,
        })
    elif session.status == "failed":
        kwargs["error"] = session.error
    return SessionStatusResponse(**kwargs)


# ── PDF Download (must be before /{kind} generic route) ──────────────────

@router.get("/api/download/{session_id}/pdf")
async def handle_download_pdf(session_id: str, request: Request):
    """下载 PDF 报告。优先 weasyprint (HTML渲染)，不可用时 fpdf2 (纯文本)。"""
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
            return JSONResponse(status_code=404, content={"error": "报告未找到，请先完成研究"})

    try:
        pdf_bytes = _markdown_to_pdf(filepath, session.query or "研究报告")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "PDF 生成失败", "detail": str(e)})

    filename = _safe_filename(session.query or "report", "pdf")
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Generic Download (final / debug) ─────────────────────────────────────

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


# ── PDF Generation ───────────────────────────────────────────────────────

def _markdown_to_pdf(md_path: str | Path, title: str = "研究报告") -> bytes:
    """Markdown → PDF。优先 weasyprint，不可用时 fpdf2。"""
    md_text = Path(md_path).read_text(encoding="utf-8")
    try:
        return _weasyprint_pdf(md_text, title)
    except (ImportError, OSError):
        return _fpdf2_pdf(md_text, title)


def _weasyprint_pdf(md_text: str, title: str) -> bytes:
    import markdown
    html_body = markdown.markdown(md_text, extensions=["extra", "codehilite", "tables", "toc"])
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;max-width:800px;margin:40px auto;padding:0 20px;font-size:14px;line-height:1.8;color:#333}}
h1{{font-size:22px;border-bottom:2px solid #58a6ff;padding-bottom:8px}}
h2{{font-size:18px;margin-top:24px;color:#1a1a2e}}
code{{background:#f0f0f0;padding:2px 6px;border-radius:3px}}
pre{{background:#1a1a2e;color:#e6edf3;padding:14px;border-radius:8px;overflow-x:auto}}
blockquote{{border-left:3px solid #58a6ff;padding:6px 14px;color:#555;background:#f8f9fa}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}th{{background:#58a6ff;color:#fff}}
</style></head><body>{html_body}</body></html>"""
    from weasyprint import HTML
    return HTML(string=html).write_pdf()


def _fpdf2_pdf(md_text: str, title: str) -> bytes:
    """fpdf2 纯文本 PDF 回退。"""
    import re as _re

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title[:80], new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)

    for line in md_text.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
        if line.startswith("---") or line.startswith("==="):
            continue
        if line.startswith("# ") and len(line) < 100:
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 8, line[2:][:100], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(2)
            continue
        if line.startswith("## ") and len(line) < 100:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 7, line[3:][:100], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(2)
            continue
        if line.startswith("- ") or line.startswith("* "):
            line = "  - " + line[2:]
        line = _re.sub(r'\*\*(.+?)\*\*', r'\1', line)
        line = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
        line = _re.sub(r'`([^`]+)`', r'\1', line)
        pdf.multi_cell(0, 5.5, line[:200])
        pdf.ln(1)

    return pdf.output()
