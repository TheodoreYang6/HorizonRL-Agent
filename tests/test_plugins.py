"""插件系统测试 — base / registry / manager / worker 集成。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from horizonrl.plugins.base import (
    PluginConfig,
    PluginEvidence,
    PluginParams,
    ToolPlugin,
    clean_search_query,
)
from horizonrl.plugins.registry import PluginRegistry
from horizonrl.schemas.task import TaskSpec


# ── 具体测试插件类 ────────────────────────────────────────────────────────


class _TestPluginParams(PluginParams):
    query: str = ""


class _TestPlugin(ToolPlugin):
    name = "test_tool"
    description = "测试工具"
    param_schema = _TestPluginParams

    async def execute(self, query: str = "", **kwargs) -> str:
        return json.dumps({"result": query.upper()})


class _TestPluginWithConfig(ToolPlugin):
    name = "test_config_tool"
    description = "带配置的测试工具"
    param_schema = _TestPluginParams

    def __init__(self, config=None):
        super().__init__(config)

    async def execute(self, query: str = "", **kwargs) -> str:
        prefix = getattr(self.config, "prefix", "")
        return json.dumps({"result": f"{prefix}{query}"})


class _TestPluginWithProvider(ToolPlugin):
    name = "test_provider_tool"
    description = "带 provider 信息的测试工具"

    async def execute(self, **kwargs) -> str:
        return "{}"

    @classmethod
    def get_provider_info(cls) -> dict:
        return {
            "provider_id": "test_api",
            "env_var": "TEST_API_KEY",
            "label": "Test API",
            "url": "https://api.test.example.com",
        }


# ── 可序列化任务创建辅助函数 ──────────────────────────────────────────────


def _make_task(**kwargs) -> TaskSpec:
    defaults = {
        "id": "t1",
        "name": "测试任务",
        "description": "搜索测试数据",
        "tool_names": ["test_tool"],
        "depends_on": [],
        "context": "",
    }
    defaults.update(kwargs)
    return TaskSpec(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# Base 类测试
# ═══════════════════════════════════════════════════════════════════════════


class TestPluginConfig:
    def test_default_values(self):
        cfg = PluginConfig()
        assert cfg.enabled is True
        assert cfg.timeout == 12.0

    def test_custom_values(self):
        cfg = PluginConfig(enabled=False, timeout=30.0)
        assert cfg.enabled is False
        assert cfg.timeout == 30.0


class TestPluginEvidence:
    def test_default_fields(self):
        ev = PluginEvidence(content="hello")
        assert ev.content == "hello"
        assert ev.source_type == "api"
        assert ev.is_mock is False


class TestToolPlugin:
    def test_build_params_default(self):
        plugin = _TestPlugin()
        params = plugin.build_params("搜索AI技术", "")
        assert params == {"query": "搜索AI技术"}

    def test_build_params_falls_back_to_input(self):
        """无 param_schema 字段时退回到 input 键。"""

        class _NoFieldPlugin(_TestPlugin):
            param_schema = PluginParams

            async def execute(self, **kwargs) -> str:
                return ""

        plugin = _NoFieldPlugin()
        params = plugin.build_params("hello", "")
        assert params == {"input": "hello"}

    def test_extract_evidence_json_list(self):
        plugin = _TestPlugin()
        output = json.dumps([
            {"title": "Item 1", "url": "http://a.com"},
            {"title": "Item 2", "url": "http://b.com", "is_mock": True},
        ])
        evs = plugin.extract_evidence(output)
        assert len(evs) == 2
        assert evs[0].content == "Item 1"
        assert evs[0].source == "http://a.com"
        assert evs[1].is_mock is True

    def test_extract_evidence_fallback_text(self):
        plugin = _TestPlugin()
        evs = plugin.extract_evidence("plain text output")
        assert len(evs) == 1
        assert evs[0].content == "plain text output"
        assert evs[0].source_type == "test_tool"

    def test_execute_returns_string(self):
        """确保 execute() 返回字符串。"""
        import asyncio

        plugin = _TestPlugin()
        result = asyncio.get_event_loop().run_until_complete(
            plugin.execute(query="hello")
        )
        data = json.loads(result)
        assert data["result"] == "HELLO"

    def test_get_provider_info_default(self):
        plugin = _TestPlugin()
        assert plugin.get_provider_info() == {}

    def test_get_provider_info_custom(self):
        info = _TestPluginWithProvider.get_provider_info()
        assert info["provider_id"] == "test_api"
        assert info["env_var"] == "TEST_API_KEY"


class TestCleanSearchQuery:
    def test_removes_prefixes(self):
        result = clean_search_query("搜索Transformer注意力机制")
        assert "搜索" not in result
        assert "Transformer" in result

    def test_truncates_long_query(self):
        long_text = "这是一个非常长的搜索查询" * 20
        result = clean_search_query(long_text)
        assert len(result) <= 120

    def test_preserves_short_query(self):
        result = clean_search_query("RLHF")
        assert result == "RLHF"


# ═══════════════════════════════════════════════════════════════════════════
# PluginRegistry 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestPluginRegistry:
    def test_discover_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = PluginRegistry()
            result = registry.discover(tmp)
            assert result == {}

    def test_discover_finds_plugin(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin_file = Path(tmp) / "my_plugin.py"
            plugin_file.write_text("""
from horizonrl.plugins.base import ToolPlugin, PluginParams

class MyParams(PluginParams):
    query: str = ""

class MyPlugin(ToolPlugin):
    name = "my_tool"
    description = "my custom tool"
    param_schema = MyParams

    async def execute(self, query: str = "", **kwargs) -> str:
        import json
        return json.dumps({"result": query})
""", encoding="utf-8")

            registry = PluginRegistry()
            result = registry.discover(tmp)
            assert "my_tool" in result
            assert result["my_tool"].name == "my_tool"

    def test_discover_skips_underscore_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "_private.py").write_text("""
from horizonrl.plugins.base import ToolPlugin
class PrivatePlugin(ToolPlugin):
    name = "private"
    async def execute(self, **kw): return ""
""", encoding="utf-8")

            registry = PluginRegistry()
            result = registry.discover(tmp)
            assert "private" not in result

    def test_discover_tolerates_syntax_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 一个语法错误文件
            (Path(tmp) / "bad.py").write_text("this is not valid python {{{", encoding="utf-8")
            # 一个有效文件
            (Path(tmp) / "good.py").write_text("""
from horizonrl.plugins.base import ToolPlugin, PluginParams
class GoodParams(PluginParams):
    q: str = ""
class GoodPlugin(ToolPlugin):
    name = "good_tool"
    param_schema = GoodParams
    async def execute(self, q: str = "", **kw) -> str:
        import json
        return json.dumps({"q": q})
""", encoding="utf-8")

            registry = PluginRegistry()
            result = registry.discover(tmp)
            # 不应因错误文件而崩溃
            assert "good_tool" in result

    def test_register_and_get(self):
        registry = PluginRegistry()
        registry.register("test_tool", _TestPlugin)
        assert registry.get("test_tool") is _TestPlugin
        assert registry.get("nonexistent") is None

    def test_register_rejects_non_plugin(self):
        registry = PluginRegistry()
        with pytest.raises(TypeError):
            registry.register("bad", dict)  # type: ignore

    def test_list_plugins(self):
        registry = PluginRegistry()
        registry.register("test_tool", _TestPlugin)
        registry.register("test_config_tool", _TestPluginWithConfig)
        plugins = registry.list_plugins()
        assert len(plugins) == 2

    def test_instantiate_all(self):
        registry = PluginRegistry()
        registry.register("test_tool", _TestPlugin)
        registry.register("test_config_tool", _TestPluginWithConfig)
        instances = registry.instantiate_all()
        assert len(instances) == 2
        assert all(isinstance(i, ToolPlugin) for i in instances)


# ═══════════════════════════════════════════════════════════════════════════
# ToolManager 插件集成测试
# ═══════════════════════════════════════════════════════════════════════════


class TestToolManagerPluginIntegration:
    def test_register_plugin(self):
        from horizonrl.tools.manager import ToolManager

        mgr = ToolManager()
        plugin = _TestPlugin()
        mgr.register_plugin("test_tool", plugin)
        assert mgr.is_registered("test_tool")
        assert mgr.get_plugin_meta("test_tool") is plugin

    def test_get_plugin_meta_none_for_builtin(self):
        from horizonrl.tools.manager import ToolManager

        mgr = ToolManager()
        from horizonrl.tools.mock import MockWebSearch
        mgr.register("web_search", MockWebSearch())
        assert mgr.get_plugin_meta("web_search") is None

    def test_register_plugins_from_registry(self):
        from horizonrl.tools.manager import ToolManager

        mgr = ToolManager()
        registry = PluginRegistry()
        registry.register("test_tool", _TestPlugin)
        registry.register("test_config_tool", _TestPluginWithConfig)

        registered = mgr.register_plugins_from_registry(registry)
        assert len(registered) == 2
        assert "test_tool" in registered
        assert mgr.is_registered("test_tool")
        assert mgr.is_registered("test_config_tool")
        assert mgr.get_plugin_meta("test_tool") is not None
        assert mgr.get_plugin_meta("test_config_tool") is not None


# ═══════════════════════════════════════════════════════════════════════════
# AgentWorker 插件分发测试
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentWorkerPluginDispatch:
    def test_build_params_uses_plugin(self):
        from horizonrl.agent.worker import AgentWorker
        from horizonrl.tools.manager import ToolManager

        mgr = ToolManager()
        plugin = _TestPlugin()
        mgr.register_plugin("test_tool", plugin)

        worker = AgentWorker(worker_id="w1", tool_manager=mgr)
        task = _make_task(tool_names=["test_tool"], description="搜索AI")
        params = worker._build_params("test_tool", task)
        assert params == {"query": "搜索AI"}

    def test_build_params_falls_back_to_builtin(self):
        from horizonrl.agent.worker import AgentWorker
        from horizonrl.tools.manager import ToolManager

        mgr = ToolManager()
        worker = AgentWorker(worker_id="w1", tool_manager=mgr)
        task = _make_task(tool_names=["web_search"], description="搜索AI")
        params = worker._build_params("web_search", task)
        assert "query" in params
        assert "num_results" in params

    def test_extract_evidence_uses_plugin(self):
        from horizonrl.agent.worker import AgentWorker
        from horizonrl.tools.manager import ToolManager

        mgr = ToolManager()
        plugin = _TestPlugin()
        mgr.register_plugin("test_tool", plugin)

        worker = AgentWorker(worker_id="w1", tool_manager=mgr)
        output = json.dumps([
            {"title": "Result 1", "url": "http://x.com"},
        ])
        evs = worker._extract_evidence("test_tool", output, "t1", "搜索")
        assert len(evs) == 1
        assert evs[0].source_type == "test_tool"
        assert evs[0].provider == "test_tool"

    def test_extract_evidence_falls_back_to_builtin(self):
        from horizonrl.agent.worker import AgentWorker
        from horizonrl.tools.manager import ToolManager

        mgr = ToolManager()
        worker = AgentWorker(worker_id="w1", tool_manager=mgr)
        output = json.dumps([
            {"title": "Web Result", "url": "http://web.com", "snippet": "desc"},
        ])
        evs = worker._extract_evidence("web_search", output, "t1", "搜索")
        assert len(evs) == 1
        assert evs[0].source_type == "web"

    def test_execute_with_plugin_tool(self):
        """端到端: Worker.execute() 使用插件工具。"""
        import asyncio
        from horizonrl.agent.worker import AgentWorker
        from horizonrl.tools.manager import ToolManager

        mgr = ToolManager()
        plugin = _TestPlugin()
        mgr.register_plugin("test_tool", plugin)

        worker = AgentWorker(worker_id="w1", tool_manager=mgr)
        task = _make_task(tool_names=["test_tool"], description="hello")
        result = asyncio.get_event_loop().run_until_complete(
            worker.execute(task)
        )
        assert result.success
        assert len(result.evidence) > 0
        # 应该通过插件提取证据
        assert result.evidence[0].source_type == "test_tool"


# ═══════════════════════════════════════════════════════════════════════════
# 示例插件端到端测试
# ═══════════════════════════════════════════════════════════════════════════


class TestExamplePluginEndToEnd:
    def test_example_plugin_discovery(self):
        """验证 plugins/ 目录中的示例插件可被发现。"""
        plugin_dir = Path("plugins")
        if not plugin_dir.is_dir():
            pytest.skip("plugins/ 目录不存在")

        registry = PluginRegistry()
        result = registry.discover(plugin_dir)
        assert "echo_tool" in result, f"未发现 echo_tool，已发现: {list(result.keys())}"

    def test_example_plugin_execution(self):
        """验证示例插件可执行。"""
        import asyncio
        from plugins.example_plugin import EchoPlugin

        plugin = EchoPlugin()
        result = asyncio.get_event_loop().run_until_complete(
            plugin.execute(message="hello", repeat=2)
        )
        data = json.loads(result)
        assert len(data["result"]) == 2
        assert "hello" in data["result"][0]
