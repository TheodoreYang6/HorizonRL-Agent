"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
统一工具管理器 —— Tool Manager
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

提供所有工具调用的统一入口：超时控制、重试、熔断、错误码规范化。

使用方式：
    mgr = ToolManager(config.tools)
    mgr.register("web_search", web_search_tool)
    result = await mgr.call(ToolCallRequest(tool_name="web_search", params={"query": "Transformer"}))

── 被哪些模块依赖 ──
    agent/worker.py     — Worker 通过 ToolManager 调用所有工具
    orchestration/dag_workflow.py — 编排层注入配置
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from horizonrl.schemas.result import ToolCall

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ToolErrorType — 工具错误分类                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ToolErrorType(str, Enum):
    """工具调用失败的错误分类。Worker 和 Verifier 据此决定恢复策略。"""

    TIMEOUT = "timeout"            # 工具调用超时
    NETWORK = "network"            # 网络错误（DNS/连接失败）
    AUTH = "auth"                  # 鉴权失败（API Key 无效）
    RATE_LIMIT = "rate_limit"      # 被限流（429）
    INVALID_PARAMS = "invalid_params"  # 参数不合法
    SANDBOX_ERROR = "sandbox_error"    # 代码沙箱错误
    CIRCUIT_OPEN = "circuit_open"      # 熔断器开启，拒绝调用
    UNKNOWN = "unknown"            # 未分类错误


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ToolCallRequest — 工具调用请求                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class ToolCallRequest:
    """标准化工具调用请求。

    Attributes:
        tool_name: 工具名称（需已注册）
        params: 传给工具的参数
        timeout: 本次调用的超时时间（秒），0 表示使用默认值
        max_retries: 本次调用的最大重试次数，0 表示不重试
        task_id: 关联的 TaskSpec.id（用于追踪）
    """

    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    timeout: float = 0.0
    max_retries: int = 0
    task_id: str = ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ToolStats — 工具调用统计                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class ToolStats:
    """单个工具的运行时统计。"""

    total_calls: int = 0
    success_calls: int = 0
    failure_calls: int = 0
    timeout_calls: int = 0
    circuit_open_rejects: int = 0
    total_latency: float = 0.0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CircuitBreaker — 熔断器                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CircuitBreaker:
    """工具级别的熔断器，防止重复调用已知不可用的工具。

    三态模型：
      CLOSED    — 正常，请求放行
      OPEN      — 熔断，拒绝所有请求
      HALF_OPEN — 冷却后允许一个探测请求

    用法：
        cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)
        if not cb.allow(): raise CircuitOpenError
        try:
            result = await call_tool()
            cb.on_success()
        except Exception:
            cb.on_failure()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._opened_at = 0.0

    def allow(self) -> bool:
        """检查是否允许通过。

        Returns:
            True 如果允许调用，False 如果熔断器开启。
        """
        if self._state == "CLOSED":
            return True
        if self._state == "HALF_OPEN":
            return True
        # OPEN: 检查冷却时间是否已过
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            self._state = "HALF_OPEN"
            return True
        return False

    def on_success(self) -> None:
        """记录成功，重置熔断器。"""
        self._failure_count = 0
        self._state = "CLOSED"

    def on_failure(self) -> None:
        """记录失败，连续失败达到阈值时打开熔断器。"""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == "HALF_OPEN":
            # 探测请求也失败了，重新打开
            self._state = "OPEN"
            self._opened_at = time.monotonic()
        elif self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            self._opened_at = time.monotonic()

    @property
    def state(self) -> str:
        """当前熔断器状态。"""
        return self._state

    @property
    def failure_count(self) -> int:
        """连续失败计数。"""
        return self._failure_count


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ToolManager — 统一工具管理器                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ToolManager:
    """统一工具调用入口，提供超时、重试、熔断、统计。

    所有 Worker 通过 ToolManager 调用工具，不直接调用工具实例。
    这样可以统一日志、统一错误处理、统一限流。

    Examples:
        >>> mgr = ToolManager()
        >>> mgr.register("web_search", WebSearchTool())
        >>> req = ToolCallRequest(tool_name="web_search", params={"query": "RLHF"})
        >>> result = await mgr.call(req)
        >>> print(result.tool_name, result.is_success)
    """

    def __init__(self, tools_config=None):
        """
        Args:
            tools_config: ToolsConfig from settings.py，为 None 时使用默认值。
        """
        self._tools: dict[str, Any] = {}  # tool_name → tool_instance
        self._stats: dict[str, ToolStats] = {}  # tool_name → ToolStats
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # 默认超时和重试配置
        self._default_timeout: float = 12.0
        self._default_max_retries: int = 1
        self._circuit_failure_threshold: int = 5
        self._circuit_cooldown_seconds: float = 15.0

        if tools_config is not None:
            self._apply_config(tools_config)

    def _apply_config(self, tools_config) -> None:
        """从 ToolsConfig 读取配置，为已注册工具设置超时。

        各工具超时优先用工具自身配置，未配置时统一用 web_search.timeout。
        """
        self._default_timeout = float(
            getattr(getattr(tools_config, "web_search", None), "timeout", 12)
        )
        self._default_max_retries = 2
        # 存储各工具类型的超时，供 call() 按工具名查询
        self._tool_timeouts: dict[str, float] = {}
        for tool_type in ("web_search", "arxiv_search", "code_execution", "retrieval"):
            tool_cfg = getattr(tools_config, tool_type, None)
            if tool_cfg is not None:
                self._tool_timeouts[tool_type] = float(
                    getattr(tool_cfg, "timeout", self._default_timeout)
                )

    # ── 工具注册 ────────────────────────────────────────────────────────

    def register(self, name: str, tool: Any) -> None:
        """注册一个工具实例。

        Args:
            name: 工具名称（与 tool.name 保持一致）。
            tool: 工具实例，必须有可调用的 execute 或 search 方法。
        """
        self._tools[name] = tool
        self._stats[name] = ToolStats()
        self._circuit_breakers[name] = CircuitBreaker(
            failure_threshold=self._circuit_failure_threshold,
            cooldown_seconds=self._circuit_cooldown_seconds,
        )

    def is_registered(self, name: str) -> bool:
        """检查工具是否已注册。"""
        return name in self._tools

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名称。"""
        return list(self._tools.keys())

    # ── 搜索结果规范化 ──────────────────────────────────────────────────

    def normalize_search_results(
        self,
        tool_name: str,
        raw_output: str,
        query: str = "",
    ) -> list[dict]:
        """将工具返回的原始 JSON 规范化为带 provenance 的 dict 列表。

        ToolManager 负责统一填入 provider/query/timestamp/is_mock，
        避免不同 provider 返回格式不一致。

        Args:
            tool_name: 工具名 (web_search / arxiv_search).
            raw_output: 工具返回的原始 JSON 字符串.
            query: 实际搜索 query.

        Returns:
            规范化后的 dict 列表，每条含 provider/query/is_mock/timestamp.
        """
        import json as _json
        import time as _time

        now = _time.time()
        results: list[dict] = []

        # 解析原始输出
        try:
            parsed = _json.loads(raw_output)
        except (_json.JSONDecodeError, TypeError):
            import ast
            try:
                parsed = ast.literal_eval(raw_output)
            except (ValueError, SyntaxError):
                parsed = raw_output

        if isinstance(parsed, list):
            for entry in parsed:
                if isinstance(entry, dict):
                    results.append({
                        "title": entry.get("title", ""),
                        "url": entry.get("url", entry.get("href", "")),
                        "snippet": entry.get("snippet", entry.get("body", str(entry))),
                        "provider": entry.get("provider", tool_name),
                        "is_mock": entry.get("is_mock", False),
                        "query": query,
                        "timestamp": now,
                    })
        else:
            results.append({
                "title": "",
                "url": "",
                "snippet": str(raw_output)[:2000],
                "provider": tool_name,
                "is_mock": "mock" in str(raw_output).lower(),
                "query": query,
                "timestamp": now,
            })

        return results

    # ── 工具调用 ────────────────────────────────────────────────────────

    async def call(self, request: ToolCallRequest) -> ToolCall:
        """执行工具调用，包含超时、重试、熔断。

        执行流程：
          1. 检查熔断器是否放行
          2. 通过 ToolCallRequest 解析工具名、参数、超时、重试次数
          3. 在 asyncio.wait_for 中调用工具
          4. 失败时根据重试次数重试
          5. 返回标准化的 ToolCall 记录

        Args:
            request: 工具调用请求。

        Returns:
            ToolCall 记录（包含输入/输出/耗时/错误信息）。
        """
        tool_name = request.tool_name
        if tool_name not in self._tools:
            return ToolCall(
                tool_name=tool_name,
                input=request.params,
                output="",
                elapsed=0.0,
                error=f"工具未注册: {tool_name}",
            )

        cb = self._circuit_breakers[tool_name]
        stats = self._stats[tool_name]

        # 熔断检查
        if not cb.allow():
            stats.circuit_open_rejects += 1
            return ToolCall(
                tool_name=tool_name,
                input=request.params,
                output="",
                elapsed=0.0,
                error=f"[{ToolErrorType.CIRCUIT_OPEN.value}] 熔断器开启，拒绝调用",
            )

        timeout = request.timeout or self._default_timeout
        max_retries = request.max_retries if request.max_retries > 0 else self._default_max_retries

        start = time.monotonic()

        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._invoke(tool_name, request.params),
                    timeout=timeout,
                )
                elapsed = time.monotonic() - start
                cb.on_success()
                stats.total_calls += 1
                stats.success_calls += 1
                stats.total_latency += elapsed

                return ToolCall(
                    tool_name=tool_name,
                    input=request.params,
                    output=str(result) if not isinstance(result, str) else result,
                    elapsed=elapsed,
                    error="",
                )

            except asyncio.TimeoutError:
                stats.total_calls += 1
                stats.timeout_calls += 1
                elapsed = time.monotonic() - start

                # 超时不退避：超时通常是服务端无响应而非过载，退避无意义。
                # 最多重试 1 次（立即重试），避免长时间阻塞。
                if attempt < min(max_retries, 1):
                    continue

                cb.on_failure()
                return ToolCall(
                    tool_name=tool_name,
                    input=request.params,
                    output="",
                    elapsed=elapsed,
                    error=f"[{ToolErrorType.TIMEOUT.value}] 超时 ({timeout}s)，"
                          f"已重试 {attempt} 次",
                )

            except Exception as exc:
                stats.total_calls += 1
                elapsed = time.monotonic() - start

                error_type = self._classify_error(exc)

                # 网络/连接错误不重试 — DNS/连接拒绝是持久性错误
                if error_type == ToolErrorType.NETWORK:
                    cb.on_failure()
                    stats.failure_calls += 1
                    return ToolCall(
                        tool_name=tool_name,
                        input=request.params,
                        output="",
                        elapsed=elapsed,
                        error=f"[{error_type.value}] {exc}",
                    )

                # 限流/鉴权/未知错误可重试（带指数退避）
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)
                    continue

                cb.on_failure()
                stats.failure_calls += 1
                return ToolCall(
                    tool_name=tool_name,
                    input=request.params,
                    output="",
                    elapsed=elapsed,
                    error=f"[{error_type.value}] {exc}",
                )

        # 不可达（循环内已 return），但类型检查器需要
        elapsed = time.monotonic() - start
        return ToolCall(
            tool_name=tool_name,
            input=request.params,
            output="",
            elapsed=elapsed,
            error=f"[{ToolErrorType.UNKNOWN.value}] 未知错误",
        )

    async def _invoke(self, tool_name: str, params: dict[str, Any]) -> str:
        """调用工具实例的入口方法。

        工具实例需要实现以下方法之一（按优先级）：
          1. async execute(**params)  — 首选
          2. async search(**params)   — Web/Arxiv 搜索工具
          3. __call__(**params)       — 同步回退
        """
        tool = self._tools[tool_name]

        # 1. 尝试 async execute
        if hasattr(tool, "execute") and callable(tool.execute):
            if asyncio.iscoroutinefunction(tool.execute):
                return await tool.execute(**params)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: tool.execute(**params))

        # 2. 尝试 async search (WebSearchTool / ArxivSearchTool)
        if hasattr(tool, "search") and callable(tool.search):
            if asyncio.iscoroutinefunction(tool.search):
                return await tool.search(**params)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: tool.search(**params))

        # 3. 回退到 __call__
        if callable(tool):
            if asyncio.iscoroutinefunction(tool):
                return await tool(**params)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: tool(**params))

        raise RuntimeError(f"工具 {tool_name} 无可用调用接口")

    def _classify_error(self, exc: Exception) -> ToolErrorType:
        """根据异常类型分类工具错误。"""
        msg = str(exc).lower()

        if "timeout" in msg or "timed out" in msg:
            return ToolErrorType.TIMEOUT
        if any(k in msg for k in ("401", "403", "unauthorized", "api key", "apikey")):
            return ToolErrorType.AUTH
        if any(k in msg for k in ("429", "rate limit", "too many requests")):
            return ToolErrorType.RATE_LIMIT
        if any(k in msg for k in ("connection", "dns", "resolve", "refused", "network")):
            return ToolErrorType.NETWORK
        if any(k in msg for k in ("sandbox", "exec", "subprocess")):
            return ToolErrorType.SANDBOX_ERROR
        if any(k in msg for k in ("param", "invalid", "argument", "required")):
            return ToolErrorType.INVALID_PARAMS

        return ToolErrorType.UNKNOWN

    # ── 统计查询 ────────────────────────────────────────────────────────

    def get_stats(self, tool_name: str) -> ToolStats | None:
        """获取单个工具的调用统计。"""
        return self._stats.get(tool_name)

    def get_all_stats(self) -> dict[str, ToolStats]:
        """获取所有工具的调用统计。"""
        return dict(self._stats)

    def get_circuit_state(self, tool_name: str) -> str:
        """获取工具的熔断器状态。"""
        cb = self._circuit_breakers.get(tool_name)
        return cb.state if cb else "UNREGISTERED"

    def reset_circuit(self, tool_name: str) -> None:
        """手动重置工具的熔断器。"""
        cb = self._circuit_breakers.get(tool_name)
        if cb:
            cb._failure_count = 0
            cb._state = "CLOSED"
