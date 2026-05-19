"""API Key 管理 + 配置预设 — /api/settings/*。"""

from __future__ import annotations

import os
import re
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

# 配置预设
PRESETS = {
    "deep": {
        "id": "deep", "name": "深度研究", "icon": "🔬", "active": True,
        "desc": "DeepSeek-V3, 5后端搜索, 最多3 Agent 并发",
        "env": {
            "HORIZON_LLM__MODEL": "deepseek-chat",
            "HORIZON_LLM__TEMPERATURE": "0.3",
            "HORIZON_AGENT__WORKER_SEMAPHORE_LIMIT": "3",
            "HORIZON_TOOLS__WEB_SEARCH__ENGINE": "auto",
            "HORIZON_AGENT__MAX_STEPS": "30",
        },
    },
    "quick": {
        "id": "quick", "name": "快速对话", "icon": "💬", "active": False,
        "desc": "DeepSeek-V3, 仅 DDGS 搜索, 单 Agent, 低延迟",
        "env": {
            "HORIZON_LLM__MODEL": "deepseek-chat",
            "HORIZON_LLM__TEMPERATURE": "0.7",
            "HORIZON_AGENT__WORKER_SEMAPHORE_LIMIT": "1",
            "HORIZON_TOOLS__WEB_SEARCH__ENGINE": "duckduckgo",
            "HORIZON_AGENT__MAX_STEPS": "10",
        },
    },
    "eval": {
        "id": "eval", "name": "评测模式", "icon": "🧪", "active": False,
        "desc": "确定性温度 0.0, Mock 数据, 严格验证, 4 Agent 并发",
        "env": {
            "HORIZON_LLM__TEMPERATURE": "0.0",
            "HORIZON_AGENT__WORKER_SEMAPHORE_LIMIT": "4",
            "HORIZON_TOOLS__WEB_SEARCH__ENGINE": "mock",
            "HORIZON_VERIFIER__STRICT_MODE": "true",
            "HORIZON_VERIFIER__MIN_EVIDENCE_COUNT": "3",
            "HORIZON_AGENT__MAX_STEPS": "30",
        },
    },
}


def _mask_key(key: str) -> str:
    """遮蔽 Key 中间部分: sk-xxx...xxx"""
    if len(key) <= 12:
        return key[:4] + "***" + key[-4:]
    return key[:6] + "***" + key[-4:]


def _read_env_file() -> dict[str, str]:
    """读取 .env 文件中的所有环境变量。"""
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


def _write_env_file(vars_to_update: dict[str, str]) -> None:
    """更新 .env 文件中的指定变量，保留其他行不变。"""
    env_path = Path(".env")
    # 如果 .env 不存在，从 .env.example 复制
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
                    val = vars_to_update[key]
                    lines.append(f'{key}={val}')
                    updated_keys.add(key)
                else:
                    lines.append(line.rstrip("\n"))

    # 追加新 key
    for key, val in vars_to_update.items():
        if key not in updated_keys:
            lines.append(f"{key}={val}")

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 同步到 os.environ
    for key, val in vars_to_update.items():
        os.environ[key] = val


@router.get("/api/settings/keys")
async def list_keys(request: Request):
    """获取所有 API Key 状态 (已配置/未配置，值已遮蔽)。"""
    env = _read_env_file()
    items = []
    for provider_id, info in _PROVIDERS.items():
        env_var = info["env"]
        value = env.get(env_var, "") or os.environ.get(env_var, "")
        items.append({
            "provider": provider_id,
            "label": info["label"],
            "env_var": env_var,
            "url": info["url"],
            "configured": bool(value),
            "masked": _mask_key(value) if value else "",
        })
    return {"keys": items}


@router.post("/api/settings/keys")
async def save_key(request: Request):
    """保存 API Key。Body: {provider: "deepseek", key: "sk-xxx"}"""
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
    if not re.match(r"^[a-zA-Z0-9\-_.]+$", key_value[:20]):
        return JSONResponse(status_code=400, content={"error": "Key 格式无效"})

    env_var = _PROVIDERS[provider]["env"]
    _write_env_file({env_var: key_value})

    return {"ok": True, "provider": provider, "masked": _mask_key(key_value)}


@router.delete("/api/settings/keys/{provider}")
async def delete_key(provider: str, request: Request):
    """删除 API Key (从 .env 中移除并清空)。"""
    if provider not in _PROVIDERS:
        return JSONResponse(status_code=400, content={"error": f"未知提供商: {provider}"})

    env_var = _PROVIDERS[provider]["env"]
    _write_env_file({env_var: ""})
    os.environ.pop(env_var, None)

    return {"ok": True, "provider": provider, "deleted": True}


@router.get("/api/settings/presets")
async def list_presets(request: Request):
    """获取所有配置预设及当前激活状态。"""
    items = []
    for pid, preset in PRESETS.items():
        # 检查预设是否与当前环境变量匹配
        match_count = 0
        total = len(preset["env"])
        for k, v in preset["env"].items():
            if os.environ.get(k, "") == v:
                match_count += 1
        items.append({
            "id": pid,
            "name": preset["name"],
            "icon": preset["icon"],
            "desc": preset["desc"],
            "active": match_count == total,
            "match": f"{match_count}/{total}",
        })
    return {"presets": items}


@router.post("/api/settings/presets/{preset_id}")
async def apply_preset(preset_id: str, request: Request):
    """应用配置预设 — 写入 .env + 同步 os.environ。"""
    if preset_id not in PRESETS:
        return JSONResponse(status_code=400, content={"error": f"未知预设: {preset_id}"})

    preset = PRESETS[preset_id]
    _write_env_file(preset["env"])

    # 清除其他预设的激活状态在前端展示
    return {"ok": True, "preset": preset_id, "name": preset["name"],
            "applied": list(preset["env"].keys())}
