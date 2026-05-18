"""
=======================================================================
05_web_agent.py — HorizonRL-Agent Web 界面 (v4: FastAPI + SSE)
=======================================================================

生产级 FastAPI 架构，模块化前后端分离。

路由:
    GET  /                          — Web UI (Jinja2 模板)
    POST /api/chat                  — 对话入口 (chat/auto/deep 三模式)
    GET  /api/stream/{sid}          — SSE 实时进度推送
    GET  /api/report/{sid}          — 报告状态查询 (页面刷新恢复)
    GET  /api/download/{sid}/{kind} — 下载 final/debug markdown

v4 变更:
    - aiohttp → FastAPI + uvicorn (自动 OpenAPI docs)
    - 内联 HTML/CSS/JS → Jinja2 模板 + 独立静态文件
    - 全新 UI: 深邃星空 × 玻璃质感 × 微动画
    - 会话管理: SessionManager 类型化状态
    - SSE: asyncio.Queue 桥接 token 流式
    - 新增右侧详情面板 (实时统计 + 系统信息)
    - 移动端响应式

API 文档:
    http://localhost:8000/docs       — Swagger UI
    http://localhost:8000/redoc      — ReDoc

运行:
    python examples/05_web_agent.py
    http://localhost:8000

旧版 (aiohttp):
    python examples/05_web_agent_aiohttp.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    import uvicorn
    from horizonrl.web.app import create_app

    app = create_app()
    port = int(os.environ.get("PORT", 8000))

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  HorizonRL-Agent Web v4 (FastAPI + SSE)                      ║
║  http://localhost:{port}                                         ║
║  http://localhost:{port}/docs        — API 文档 (Swagger)        ║
║  Ctrl+C 停止                                                 ║
╚══════════════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
