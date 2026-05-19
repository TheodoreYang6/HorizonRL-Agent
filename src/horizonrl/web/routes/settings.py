"""应用配置管理 — GET/POST /api/settings/config + API Key 管理。"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

_PROVIDERS = {
    "deepseek": {"env": "DEEPSEEK_API_KEY", "label": "DeepSeek", "url": "https://api.deepseek.com"},
    "openai": {"env": "OPENAI_API_KEY", "label": "OpenAI", "url": "https://api.openai.com"},
    "anthropic": {"env": "ANTHROPIC_API_KEY", "label": "Anthropic", "url": "https://api.anthropic.com"},
    "dashscope": {"env": "DASHSCOPE_API_KEY", "label": "DashScope (Embedding)", "url": "https://dashscope.aliyuncs.com"},
    "bocha": {"env": "BOCHA_API_KEY", "label": "Bocha (Web搜索)", "url": "https://api.bocha.cn"},
    "brave": {"env": "BRAVE_API_KEY", "label": "Brave (Web搜索)", "url": "https://api.search.brave.com"},
}

# 主流模型列表
_MODELS = {
    "deepseek": [
        {"value": "deepseek-chat", "label": "DeepSeek-V3 (推荐)"},
        {"value": "deepseek-reasoner", "label": "DeepSeek-R1 (推理)"},
    ],
    "openai": [
        {"value": "gpt-4o", "label": "GPT-4o"},
        {"value": "gpt-4o-mini", "label": "GPT-4o Mini"},
        {"value": "gpt-4-turbo", "label": "GPT-4 Turbo"},
    ],
    "anthropic": [
        {"value": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6"},
        {"value": "claude-opus-4-7", "label": "Claude Opus 4.7"},
        {"value": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
    ],
    "dashscope": [
        {"value": "text-embedding-v4", "label": "text-embedding-v4 (1024维)"},
        {"value": "text-embedding-v3", "label": "text-embedding-v3"},
    ],
}

_EMBEDDING_MODELS = [
    {"value": "text-embedding-v4", "label": "DashScope text-embedding-v4 (1024维)"},
    {"value": "text-embedding-3-small", "label": "OpenAI text-embedding-3-small"},
    {"value": "text-embedding-3-large", "label": "OpenAI text-embedding-3-large"},
]

_SEARCH_ENGINES = [
    {"value": "auto", "label": "自动竞速 (Bocha→Brave→DDGS→Wikipedia)"},
    {"value": "bocha", "label": "Bocha (国内推荐, 需Key)"},
    {"value": "brave", "label": "Brave (国际)"},
    {"value": "duckduckgo", "label": "DuckDuckGo (免费)"},
    {"value": "mock", "label": "Mock (离线测试)"},
]


def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return key[:4] + "***" + key[-4:]
    return key[:6] + "***" + key[-4:]


def _read_env() -> dict[str, str]:
    env_path = Path(".env")
    if not env_path.exists():
        return {}
    result = {}
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _write_env(vars_to_update: dict[str, str]) -> None:
    env_path = Path(".env")
    if not env_path.exists():
        example = Path(".env.example")
        if example.exists():
            import shutil
            shutil.copy(example, env_path)

    lines = []
    updated_keys = set()
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    lines.append(line.rstrip("\n"))
                    continue
                key = stripped.partition("=")[0].strip()
                if key in vars_to_update:
                    lines.append(f"{key}={vars_to_update[key]}")
                    updated_keys.add(key)
                else:
                    lines.append(line.rstrip("\n"))

    for key, val in vars_to_update.items():
        if key not in updated_keys:
            lines.append(f"{key}={val}")

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for key, val in vars_to_update.items():
        os.environ[key] = val


# ── Config API ───────────────────────────────────────────────────────────

@router.get("/api/settings/config")
async def get_config(request: Request):
    """获取当前所有可配置参数的状态。"""
    return {
        "llm": {
            "provider": os.environ.get("HORIZON_LLM__PROVIDER", "deepseek"),
            "model": os.environ.get("HORIZON_LLM__MODEL", "deepseek-chat"),
            "base_url": os.environ.get("HORIZON_LLM__BASE_URL", "https://api.deepseek.com"),
            "temperature": os.environ.get("HORIZON_LLM__TEMPERATURE", "0.3"),
            "max_tokens": os.environ.get("HORIZON_LLM__MAX_TOKENS", "4096"),
        },
        "lightweight_llm": {
            "model": os.environ.get("HORIZON_LIGHTWEIGHT_LLM__MODEL", "deepseek-chat"),
            "base_url": os.environ.get("HORIZON_LIGHTWEIGHT_LLM__BASE_URL", "https://api.deepseek.com"),
        },
        "embedding": {
            "provider": os.environ.get("HORIZON_EMBEDDING__PROVIDER", "dashscope"),
            "model": os.environ.get("HORIZON_EMBEDDING__MODEL", "text-embedding-v4"),
            "base_url": os.environ.get("HORIZON_EMBEDDING__BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        },
        "agent": {
            "max_steps": os.environ.get("HORIZON_AGENT__MAX_STEPS", "30"),
            "worker_semaphore_limit": os.environ.get("HORIZON_AGENT__WORKER_SEMAPHORE_LIMIT", "3"),
            "task_timeout": os.environ.get("HORIZON_AGENT__TASK_TIMEOUT", "120"),
            "llm_call_timeout": os.environ.get("HORIZON_AGENT__LLM_CALL_TIMEOUT", "30"),
            "tool_call_timeout": os.environ.get("HORIZON_AGENT__TOOL_CALL_TIMEOUT", "12"),
            "max_retries_per_task": os.environ.get("HORIZON_AGENT__MAX_RETRIES_PER_TASK", "3"),
        },
        "tools": {
            "web_search_engine": os.environ.get("HORIZON_TOOLS__WEB_SEARCH__ENGINE", "auto"),
        },
        "memory": {
            "l3_backend": os.environ.get("HORIZON_MEMORY__L3_BACKEND", "chromadb"),
        },
        "models_available": _MODELS,
        "embedding_models_available": _EMBEDDING_MODELS,
        "search_engines_available": _SEARCH_ENGINES,
    }


@router.post("/api/settings/config")
async def save_config(request: Request):
    """保存配置参数。Body: {key: value, ...}，键名如 llm__model。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})

    updates: dict[str, str] = {}
    for key, value in body.items():
        if not isinstance(value, str):
            continue
        env_key = f"HORIZON_{key.upper()}"
        updates[env_key] = value

    if updates:
        _write_env(updates)

    return {"ok": True, "updated": list(updates.keys())}


# ── API Key ──────────────────────────────────────────────────────────────

@router.get("/api/settings/keys")
async def list_keys(request: Request):
    env = _read_env()
    items = []
    for provider_id, info in _PROVIDERS.items():
        value = env.get(info["env"], "") or os.environ.get(info["env"], "")
        items.append({
            "provider": provider_id, "label": info["label"],
            "env_var": info["env"], "url": info["url"],
            "configured": bool(value),
            "masked": _mask_key(value) if value else "",
        })
    return {"keys": items}


@router.post("/api/settings/keys")
async def save_key(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})
    provider = body.get("provider", "")
    key_value = body.get("key", "").strip()
    if provider not in _PROVIDERS:
        return JSONResponse(status_code=400, content={"error": f"未知提供商: {provider}"})
    if not key_value:
        return JSONResponse(status_code=400, content={"error": "Key 不能为空"})
    _write_env({_PROVIDERS[provider]["env"]: key_value})
    return {"ok": True, "provider": provider, "masked": _mask_key(key_value)}


@router.delete("/api/settings/keys/{provider}")
async def delete_key(provider: str, request: Request):
    if provider not in _PROVIDERS:
        return JSONResponse(status_code=400, content={"error": f"未知提供商: {provider}"})
    env_var = _PROVIDERS[provider]["env"]
    _write_env({env_var: ""})
    os.environ.pop(env_var, None)
    return {"ok": True, "provider": provider, "deleted": True}
