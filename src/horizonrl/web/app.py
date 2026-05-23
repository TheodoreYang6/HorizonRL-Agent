"""FastAPI 应用工厂 — lifespan、CORS、静态文件、路由注册。"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from horizonrl.web.routes.chat import router as chat_router
from horizonrl.web.routes.documents import router as documents_router
from horizonrl.web.routes.report import router as report_router
from horizonrl.web.routes.sessions import router as sessions_router
from horizonrl.web.routes.settings import router as settings_router
from horizonrl.web.routes.stream import router as stream_router
from horizonrl.web.session_manager import create_session_manager

# 模块级单例（懒创建，避免 import 时产生 DB 文件）
_session_manager = None


def _get_default_session_manager():
    """懒创建默认 session_manager（SQLite 持久化，路径可通过环境变量配置）。"""
    global _session_manager
    if _session_manager is None:
        backend = os.environ.get("SESSION_BACKEND", "sqlite")
        db_path = os.environ.get("SESSION_DB_PATH", "data/sessions.db")
        _session_manager = create_session_manager(backend=backend, db_path=db_path)
    return _session_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    if not hasattr(app.state, "session_manager") or app.state.session_manager is None:
        app.state.session_manager = _get_default_session_manager()

    # 初始化 DocumentStore（文档上传/RAG）
    if not hasattr(app.state, "document_store") or app.state.document_store is None:
        try:
            from horizonrl.rag.document_store import DocumentStore
            app.state.document_store = DocumentStore()
        except Exception:
            app.state.document_store = None

    # 初始化 Embedding Client（文档索引用）
    if not hasattr(app.state, "embedding_client") or app.state.embedding_client is None:
        try:
            from horizonrl.config.settings import load_config
            from horizonrl.llm.client import LLMClient
            cfg = load_config()
            if cfg.embedding.api_key:
                app.state.embedding_client = LLMClient(cfg.embedding)
        except Exception:
            pass

    yield
    app.state.session_manager.cleanup_expired()


def create_app(session_mgr=None) -> FastAPI:
    """创建并返回已配置的 FastAPI 应用实例。

    Args:
        session_mgr: 可选的自定义 SessionManager，用于测试注入。
                     默认使用 SQLite 持久化单例（data/sessions.db）。
    """
    app = FastAPI(
        title="Horizon-Agent",
        description="Horizon-Agent — Multi-Agent Verified Research",
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

    # 会话管理器：优先使用注入，否则懒创建默认单例
    app.state.session_manager = (
        session_mgr if session_mgr is not None else _get_default_session_manager()
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
    app.include_router(settings_router)
    app.include_router(documents_router)

    # 根路由：返回 SPA 页面
    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"app_version": "0.2.0"},
        )

    return app
