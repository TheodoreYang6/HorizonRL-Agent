"""
EchoPlugin — 示例工具插件。

演示 HorizonRL-Agent 插件开发模式：
  1. 继承 ToolPlugin
  2. 设置 name / description 类变量
  3. 声明 param_schema / config_schema
  4. 实现 async execute()
  5. （可选）覆盖 build_params / extract_evidence

放入 plugins/ 目录后，启动时自动被发现和注册。
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import Field

from horizonrl.plugins.base import (
    PluginConfig,
    PluginEvidence,
    PluginParams,
    ToolPlugin,
)


class EchoPluginConfig(PluginConfig):
    """Echo 插件配置。"""

    prefix: str = Field(default="[Echo] ", description="输出前缀")


class EchoPluginParams(PluginParams):
    """Echo 插件参数。"""

    message: str = Field(..., description="要回显的消息")
    repeat: int = Field(default=1, ge=1, le=5, description="重复次数")


class EchoPlugin(ToolPlugin):
    """回显插件 — 完整演示插件开发模式。

    此插件无需任何外部依赖，可直接运行验证插件系统是否正常工作。
    """

    name: ClassVar[str] = "echo_tool"
    description: ClassVar[str] = "回显输入消息（示例插件）"
    version: ClassVar[str] = "1.0.0"
    author: ClassVar[str] = "HorizonRL Team"
    param_schema: ClassVar[type[PluginParams]] = EchoPluginParams
    config_schema: ClassVar[type[PluginConfig]] = EchoPluginConfig

    async def execute(self, message: str = "", repeat: int = 1, **kwargs: Any) -> str:
        prefix = getattr(self.config, "prefix", "[Echo] ")
        lines = [f"{prefix}{message}" for _ in range(repeat)]
        return json.dumps({
            "result": lines,
            "length": len(message),
            "query": message,
        }, ensure_ascii=False)

    def build_params(self, task_description: str, task_context: str = "") -> dict[str, Any]:
        from horizonrl.plugins.base import clean_search_query

        query = clean_search_query(task_description)
        return {"message": query, "repeat": 1}

    def extract_evidence(
        self, output: str, task_description: str = ""
    ) -> list[PluginEvidence]:
        try:
            data = json.loads(output)
            return [
                PluginEvidence(
                    content=data.get("query", output),
                    source="echo_tool",
                    source_type="echo",
                )
            ]
        except (json.JSONDecodeError, TypeError):
            return [PluginEvidence(content=output[:2000], source_type="echo")]
