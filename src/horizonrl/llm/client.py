"""
LLM Client —— OpenAI-compatible API 封装。

支持 OpenAI / DeepSeek / vLLM 等所有兼容 OpenAI SDK 的后端。
切换后端只需改 base_url 和 model 配置。

使用方式：
    client = LLMClient(config.llm)
    result = await client.chat("你好")
    print(result.content)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from horizonrl.config.settings import LLMConfig as LLMConfigModel


@dataclass
class LLMCallResult:
    """单次 LLM 调用的返回结果。"""

    content: str = ""
    model: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    elapsed: float = 0.0
    finish_reason: str = ""
    error: str = ""

    @property
    def is_success(self) -> bool:
        return self.error == ""


@dataclass
class EmbedResult:
    """单次 Embedding 调用的返回结果。"""

    embedding: list[float] | None = None
    tokens_used: int = 0
    elapsed: float = 0.0
    error: str = ""

    @property
    def is_success(self) -> bool:
        return self.error == "" and self.embedding is not None


class LLMClient:
    """OpenAI-compatible LLM 调用客户端。

    封装了同步/异步调用、超时、错误处理、Token 统计。
    支持 OpenAI、DeepSeek、vLLM 及任何 OpenAI-compatible API。

    Examples:
        >>> from horizonrl.config.settings import LLMConfig
        >>> cfg = LLMConfig(model="deepseek-chat", base_url="https://api.deepseek.com")
        >>> client = LLMClient(cfg)
        >>> result = await client.chat("将以下任务拆解为子任务: ...")
        >>> print(result.content)
    """

    def __init__(self, config: LLMConfigModel):
        self.config = config
        self._client = None  # 延迟初始化

    def _get_client(self):
        """延迟创建 OpenAI client，避免 import 时就需要 API key。"""
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        """异步调用 LLM Chat Completion。

        Args:
            prompt: 用户消息。
            system_prompt: 系统提示词。
            temperature: 温度，None 则使用配置值。
            max_tokens: 最大 token 数，None 则使用配置值。

        Returns:
            LLMCallResult（含内容、token 数、耗时）。
        """
        start = time.monotonic()
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                ),
                timeout=getattr(self.config, "llm_call_timeout", 30),
            )

            choice = response.choices[0]
            usage = response.usage

            return LLMCallResult(
                content=choice.message.content or "",
                model=response.model,
                tokens_prompt=usage.prompt_tokens if usage else 0,
                tokens_completion=usage.completion_tokens if usage else 0,
                tokens_total=usage.total_tokens if usage else 0,
                elapsed=time.monotonic() - start,
                finish_reason=choice.finish_reason or "",
            )

        except asyncio.TimeoutError:
            return LLMCallResult(
                elapsed=time.monotonic() - start,
                error="LLM 调用超时",
            )
        except Exception as exc:
            return LLMCallResult(
                elapsed=time.monotonic() - start,
                error=f"LLM 调用失败: {exc}",
            )

    async def chat_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """流式调用 LLM Chat Completion — 逐 token yield。

        Yields:
            str: 每个 delta token 的文本内容。
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def chat_sync(self, prompt: str, system_prompt: str = "") -> LLMCallResult:
        """同步调用（内部用 asyncio.run 包装）。"""
        return asyncio.run(self.chat(prompt, system_prompt))

    async def embed(self, text: str) -> EmbedResult:
        """调用 OpenAI-compatible Embedding API 生成向量。

        Args:
            text: 要嵌入的文本。

        Returns:
            EmbedResult（含向量、token 数、耗时）。
        """
        start = time.monotonic()
        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.embeddings.create(
                    model=self.config.model,
                    input=text,
                ),
                timeout=getattr(self.config, "llm_call_timeout", 10),
            )
            return EmbedResult(
                embedding=list(response.data[0].embedding),
                tokens_used=response.usage.total_tokens if response.usage else 0,
                elapsed=time.monotonic() - start,
            )
        except asyncio.TimeoutError:
            return EmbedResult(
                elapsed=time.monotonic() - start,
                error="Embedding 调用超时",
            )
        except Exception as exc:
            return EmbedResult(
                elapsed=time.monotonic() - start,
                error=f"Embedding 调用失败: {exc}",
            )
