"""
Tool Plugin 抽象基类 — 插件开发者的唯一入口。

插件只需继承 ToolPlugin，实现 execute()，设置 name/description，
放入 plugins/ 目录即自动发现、注册、可用。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class PluginConfig(BaseModel):
    """插件配置基类。插件作者可子类化添加自定义字段（api_key, endpoint 等）。"""

    enabled: bool = Field(default=True, description="是否启用此插件")
    timeout: float = Field(default=12.0, ge=1.0, description="调用超时（秒）")


class PluginParams(BaseModel):
    """插件调用参数基类。子类声明 execute() 接受的参数字段。

    AgentWorker.build_params() 默认将 task.description 放入第一个字段。
    """


@dataclass
class PluginEvidence:
    """插件从工具输出中提取的证据条目。

    由 ToolPlugin.extract_evidence() 返回，AgentWorker 转换为内部 EvidenceItem。
    """

    content: str
    source: str = ""
    source_type: str = "api"
    relevance_score: float = 0.0
    is_mock: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolPlugin(abc.ABC):
    """工具插件抽象基类。

    插件开发者只需：
      1. 继承 ToolPlugin
      2. 设置 name / description 类变量
      3. 实现 async execute(**params) -> str
      4. （可选）覆盖 build_params / extract_evidence
      5. 放入 plugins/ 目录
    """

    # ── 类变量：插件声明自身身份 ──
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    version: ClassVar[str] = "1.0.0"
    author: ClassVar[str] = ""

    # ── Schema 声明 ──
    param_schema: ClassVar[type[PluginParams]] = PluginParams
    config_schema: ClassVar[type[PluginConfig]] = PluginConfig

    def __init__(self, config: PluginConfig | None = None):
        self.config = config or self.config_schema()

    @abc.abstractmethod
    async def execute(self, **params: Any) -> str:
        """执行工具调用。插件作者必须实现此方法。"""

    def build_params(self, task_description: str, task_context: str = "") -> dict[str, Any]:
        """从任务描述构建 execute() 参数。

        默认：将 task_description 放入 param_schema 第一个字段。
        插件可覆盖此方法实现自定义参数映射（如解析、清洗搜索词）。
        """
        fields = list(self.param_schema.model_fields.keys())
        if fields:
            return {fields[0]: task_description}
        return {"input": task_description}

    def extract_evidence(
        self, output: str, task_description: str = ""
    ) -> list[PluginEvidence]:
        """从工具输出中提取证据条目。

        默认：尝试 JSON list 解析，失败则整段输出装入单个条目。
        """
        import json

        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return [
                PluginEvidence(
                    content=output[:2000],
                    source_type=self.name,
                )
            ]

        if isinstance(parsed, list):
            return [
                PluginEvidence(
                    content=(
                        entry.get("title", str(entry))
                        if isinstance(entry, dict)
                        else str(entry)
                    ),
                    source=entry.get("url", "") if isinstance(entry, dict) else "",
                    source_type=self.name,
                    is_mock=entry.get("is_mock", False) if isinstance(entry, dict) else False,
                )
                for entry in parsed
            ]

        return [
            PluginEvidence(
                content=str(parsed)[:2000],
                source_type=self.name,
            )
        ]

    @classmethod
    def get_provider_info(cls) -> dict[str, str]:
        """返回插件的 API Key 提供商信息（用于 Web 设置页）。

        默认：返回空信息。有 API Key 需求的插件应覆盖此方法。
        """
        return {}


def clean_search_query(description: str) -> str:
    """从任务描述中提取干净的搜索查询词。

    可被插件 build_params() 复用。
    """
    import re

    text = description.strip()
    for prefix in ("搜索", "检索", "查找", "调研", "了解", "分析", "探讨"):
        text = re.sub(rf"^{prefix}", "", text, count=1).strip()
    text = text.rstrip("。，.。")
    if len(text) > 120:
        cut = text[:120]
        last_comma = max(cut.rfind("，"), cut.rfind(","), cut.rfind(" "))
        if last_comma > 60:
            text = cut[:last_comma]
        else:
            text = cut
    return text.strip() or description[:120]
