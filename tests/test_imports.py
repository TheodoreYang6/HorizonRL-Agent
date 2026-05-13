"""Test that all project dependencies import correctly."""

import importlib
import sys
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _try_import(name: str, package: str | None = None) -> Any:
    """Import a module, returning the module or raising the original error."""
    return importlib.import_module(name, package=package)


def _skip_if_missing(name: str):
    """pytest skip marker — checks whether *name* is importable."""
    try:
        importlib.import_module(name)
    except ImportError:
        return pytest.mark.skip(reason=f"{name} not installed")
    return pytest.mark.none()


# ---------------------------------------------------------------------------
# Core dependencies (Phase 1) — must be installed
# ---------------------------------------------------------------------------

CORE_MODULES = [
    # Agent framework
    ("langgraph", "langgraph"),
    ("langchain", "langchain"),
    ("langchain_community", "langchain-community"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_anthropic", "langchain-anthropic"),
    # Async / network
    ("httpx", "httpx"),
    ("aiohttp", "aiohttp"),
    # LLM clients
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    # Data / config
    ("numpy", "numpy"),
    ("pydantic", "pydantic"),
    ("yaml", "pyyaml"),
    ("tiktoken", "tiktoken"),
    # Vector search
    ("faiss", "faiss-cpu"),
    # CLI / logging
    ("rich", "rich"),
    ("tqdm", "tqdm"),
]


@pytest.mark.parametrize("module_name, pip_name", CORE_MODULES)
def test_core_import(module_name: str, pip_name: str) -> None:
    """Every core package must be importable."""
    mod = _try_import(module_name)
    assert mod is not None, f"Failed to import {module_name} (pip: {pip_name})"


# ---------------------------------------------------------------------------
# Optional: RL training (Phase 3)
# ---------------------------------------------------------------------------

RL_MODULES = [
    ("trl", "trl"),
    ("torch", "torch"),
    ("accelerate", "accelerate"),
]


@pytest.mark.parametrize("module_name, pip_name", RL_MODULES)
def test_rl_import(module_name: str, pip_name: str) -> None:
    """RL packages — skip if not installed (Phase 3+)."""
    try:
        mod = _try_import(module_name)
        assert mod is not None
    except ImportError:
        pytest.skip(f"{pip_name} not installed (optional RL dep)")


# verl uses a different import scheme
def test_verl_import() -> None:
    """verl — optional RL training framework."""
    verl = pytest.importorskip("verl", reason="verl not installed (optional RL dep)")
    assert verl is not None


# ---------------------------------------------------------------------------
# Optional: Inference (Phase 1+)
# ---------------------------------------------------------------------------

def test_vllm_import() -> None:
    """vLLM — optional inference engine."""
    vllm = pytest.importorskip("vllm", reason="vllm not installed (optional inference dep)")
    assert vllm is not None


# ---------------------------------------------------------------------------
# Optional: Evaluation (Phase 4)
# ---------------------------------------------------------------------------

EVAL_MODULES = [
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]


@pytest.mark.parametrize("module_name, pip_name", EVAL_MODULES)
def test_eval_import(module_name: str, pip_name: str) -> None:
    """Evaluation packages — skip if not installed (Phase 4+)."""
    try:
        mod = _try_import(module_name)
        assert mod is not None
    except ImportError:
        pytest.skip(f"{pip_name} not installed (optional eval dep)")


# ---------------------------------------------------------------------------
# Internal horizonrl package
# ---------------------------------------------------------------------------

INTERNAL_MODULES = [
    "horizonrl",
    "horizonrl.agent",
    "horizonrl.agent.planner",
    "horizonrl.agent.worker",
    "horizonrl.agent.verifier",
    "horizonrl.agent.replanner",
    "horizonrl.config",
    "horizonrl.config.settings",
    "horizonrl.memory",
    "horizonrl.memory.hierarchical_memory",
    "horizonrl.orchestration",
    "horizonrl.orchestration.dag_workflow",
    "horizonrl.schemas",
    "horizonrl.schemas.task",
    "horizonrl.schemas.result",
    "horizonrl.schemas.event",
    "horizonrl.schemas.report",
    "horizonrl.tools",
    "horizonrl.tools.manager",
    "horizonrl.tools.web_search",
    "horizonrl.tools.arxiv_search",
    "horizonrl.tools.code_execution",
    "horizonrl.logging",
    "horizonrl.eval",
    "horizonrl.rl",
]


@pytest.mark.parametrize("module_name", INTERNAL_MODULES)
def test_internal_import(module_name: str) -> None:
    """Every internal horizonrl sub-module must be importable."""
    mod = _try_import(module_name)
    assert mod is not None, f"Failed to import {module_name}"


# ---------------------------------------------------------------------------
# Sub-module attribute sanity checks
# ---------------------------------------------------------------------------

def test_version_attribute() -> None:
    """horizonrl exports __version__."""
    import horizonrl

    assert horizonrl.__version__ == "0.1.0"


def test_settings_exports_agent_config() -> None:
    """settings module exports AgentConfig and load_config."""
    from horizonrl.config.settings import AgentConfig, load_config

    cfg = AgentConfig()
    assert cfg.debug is False
    assert cfg.llm.provider == "openai"


def test_hierarchical_memory_exports_class() -> None:
    """hierarchical_memory exports HierarchicalMemory with L1/L2/L3."""
    from horizonrl.memory.hierarchical_memory import (
        HierarchicalMemory,
        MemoryEntry,
        MemoryContext,
        L1RecentWindow,
        L2SemanticSummary,
    )

    mem = HierarchicalMemory()
    assert mem.l1.count == 0
    assert mem.l2.count == 0
    assert isinstance(mem.get_stats(), dict)
