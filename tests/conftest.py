"""pytest 全局配置 — 测试隔离清理。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _clean_shared_state():
    """清理共享的 ChromaDB/DB 状态，确保测试隔离。"""
    for d in ("data/chromadb", ".memory/test_*"):
        p = Path(d)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    yield
    # teardown: 测试后清理
    for d in ("data/chromadb",):
        p = Path(d)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
