"""Horizon-Agent 工具插件系统。

插件开发者只需:
  1. 继承 ToolPlugin
  2. 设置 name / description
  3. 实现 async execute()
  4. 放入 plugins/ 目录
"""

from horizonrl.plugins.base import (
    PluginConfig,
    PluginEvidence,
    PluginParams,
    ToolPlugin,
    clean_search_query,
)
from horizonrl.plugins.registry import PluginRegistry

__all__ = [
    "ToolPlugin",
    "PluginConfig",
    "PluginParams",
    "PluginEvidence",
    "PluginRegistry",
    "clean_search_query",
]
