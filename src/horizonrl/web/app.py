"""FastAPI 应用工厂 — lifespan、CORS、静态文件、路由注册。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from horizonrl.web.routes.chat import router as chat_router
from horizonrl.web.routes.stream import router as stream_router
from horizonrl.web.routes.report import router as report_router
from horizonrl.web.routes.sessions import router as sessions_router
import os
from horizonrl.web.session_manager import create_session_manager

# 通过环境变量切换后端: SESSION_BACKEND=sqlite (默认 memory)
_session_backend = os.environ.get("SESSION_BACKEND", "memory")
session_manager = create_session_manager(backend=_session_backend)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    app.state.session_manager = session_manager
    yield
    session_manager.cleanup_expired()


def create_app() -> FastAPI:
    """创建并返回已配置的 FastAPI 应用实例。"""
    app = FastAPI(
        title="HorizonRL-Agent",
        description="Long-Horizon Agentic RL System — Web Interface",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — 允许跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 静态文件挂载
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Jinja2 模板
    templates_dir = Path(__file__).resolve().parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates

    # 注册子路由
    app.include_router(chat_router)
    app.include_router(stream_router)
    app.include_router(report_router)
    app.include_router(sessions_router)

    # 根路由：返回 SPA 页面
    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"app_version": "0.1.0"},
        )

    return app
