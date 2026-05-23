"""
Plugin Registry — 插件发现、加载、注册。

扫描文件系统目录中的 .py 文件，用 importlib 动态加载，
通过 inspect 查找 ToolPlugin 子类，缓存为类注册表。
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path

from horizonrl.plugins.base import PluginConfig, ToolPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """插件类注册表。发现、缓存、实例化 ToolPlugin 子类。"""

    def __init__(self):
        self._classes: dict[str, type[ToolPlugin]] = {}
        self._discovered_dirs: list[Path] = []

    def discover(self, *plugin_dirs: str | Path) -> dict[str, type[ToolPlugin]]:
        """扫描目录中的 .py 文件，加载并查找 ToolPlugin 子类。

        跳过以 _ 开头的文件（私有模块）和 base.py / registry.py。

        Args:
            plugin_dirs: 要扫描的目录路径。

        Returns:
            {plugin_name: plugin_class} 字典。
        """
        for plugin_dir in plugin_dirs:
            dir_path = Path(plugin_dir).resolve()
            if not dir_path.is_dir():
                logger.debug("插件目录不存在: %s", dir_path)
                continue

            self._discovered_dirs.append(dir_path)
            py_files = sorted(dir_path.glob("*.py"))
            skip_names = ("_", "base.py", "registry.py", "__init__.py")

            for py_file in py_files:
                if py_file.name in skip_names or py_file.name.startswith("_"):
                    continue

                try:
                    self._load_file(py_file)
                except Exception:
                    logger.warning("加载插件文件失败: %s", py_file, exc_info=True)

        return dict(self._classes)

    def register(self, name: str, plugin_cls: type[ToolPlugin]) -> None:
        """手动注册一个插件类。"""
        if not issubclass(plugin_cls, ToolPlugin):
            raise TypeError(f"{plugin_cls} 不是 ToolPlugin 子类")
        self._classes[name] = plugin_cls

    def get(self, name: str) -> type[ToolPlugin] | None:
        """获取指定名称的插件类。"""
        return self._classes.get(name)

    def list_plugins(self) -> dict[str, type[ToolPlugin]]:
        """返回所有已发现的插件类。"""
        return dict(self._classes)

    def instantiate_all(
        self, configs: dict[str, PluginConfig] | None = None
    ) -> list[ToolPlugin]:
        """实例化所有已发现的插件。

        Args:
            configs: {name: PluginConfig} 可选的配置覆盖。

        Returns:
            实例化后的插件列表。
        """
        configs = configs or {}
        instances: list[ToolPlugin] = []
        for name, cls in self._classes.items():
            try:
                cfg = configs.get(name)
                instances.append(cls(cfg))
            except Exception:
                logger.warning("插件实例化失败: %s", name, exc_info=True)
        return instances

    def _load_file(self, py_file: Path) -> None:
        """用 importlib 动态加载单个 .py 文件，查找 ToolPlugin 子类。"""
        module_name = f"horizonrl_plugin_{py_file.stem}"
        full_name = f"{module_name}_{id(py_file)}"

        spec = importlib.util.spec_from_file_location(full_name, str(py_file))
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(full_name, None)
            raise

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, ToolPlugin) or obj is ToolPlugin:
                continue
            if not obj.name:
                logger.warning("插件类 %s 未设置 name，跳过", obj.__name__)
                continue
            self._classes[obj.name] = obj
