"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Horizon-Agent 配置管理系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本文件是项目中唯一被所有模块依赖的配置层。使用 Pydantic V2 + YAML + .env
三级配置体系：

    .env 环境变量 (最高优先级, 用于 API Key)
       ↓ 覆盖
    YAML 文件 (dev.yaml 覆盖 default.yaml)
       ↓ 覆盖
    Pydantic 默认值 (代码中的 field(default=...))

── 公开 API 一览 ──

    load_config(path=None) -> AgentConfig
        加载完整配置。按以下优先级合并：
        1. 代码默认值
        2. configs/default.yaml
        3. 可选的 overlay YAML (如 configs/dev.yaml)
        4. .env 文件中的 HORIZON_* 环境变量
        5. 直接设置的环境变量

    AgentConfig        — 顶层配置对象，getattr 访问所有子配置
    LLMConfig          — LLM 端点配置 (provider/model/temperature/tokens)
    MemoryConfig       — 分层记忆配置 (L1/L2/L3 参数)
    RolloutConfig      — RL 训练配置 (Phase 3+)

── 使用方式 ──

    # 最简单：全用默认值
    cfg = load_config()

    # 开发环境
    cfg = load_config("configs/dev.yaml")

    # 评测环境
    cfg = load_config("configs/eval.yaml")

    # 无论如何配置，都可以通过环境变量临时覆盖：
    # export HORIZON_LLM_MODEL=gpt-4o
    # export HORIZON_DEBUG=true

── 被哪些模块依赖 ──
    agent/planner.py     — 读取 agent.max_steps, llm 配置
    agent/worker.py      — 读取 llm, tool 超时配置
    agent/verifier.py    — 读取 verifier 配置
    memory/hierarchical_memory.py — 读取 memory 配置
    orchestration/dag_workflow.py — 读取 agent.semaphore_limit
    tools/manager.py     — 读取 tool 超时配置
    logging/trajectory_logger.py  — 读取 logging 配置
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LLMConfig — LLM 端点配置                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 每个 LLM 端点（主推理、轻量摘要、Embedding）都用此结构描述。
# API Key 不从配置文件读取，而是从环境变量注入（安全考虑）。
#
# 环境变量覆盖：
#   HORIZON_LLM_PROVIDER  → provider
#   HORIZON_LLM_MODEL     → model
#   HORIZON_LLM_TEMPERATURE → temperature
#   HORIZON_LLM_MAX_TOKENS  → max_tokens
#   HORIZON_LLM_BASE_URL    → base_url


class LLMConfig(BaseModel):
    """单个 LLM 端点的配置。

    Examples:
        >>> cfg = LLMConfig(provider="deepseek", model="deepseek-chat")
        >>> cfg.model
        'deepseek-chat'
    """

    provider: str = Field(
        default="deepseek",
        description="LLM 提供商: deepseek | openai | anthropic | dashscope | vllm",
    )
    model: str = Field(
        default="deepseek-chat",
        description="模型名称，如 deepseek-chat, deepseek-reasoner, gpt-4o",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="生成温度。Agent 推荐 0.0-0.3（低温度=更确定性）",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="单次生成的最大 token 数",
    )
    base_url: str | None = Field(
        default=None,
        description="自定义 API 地址。None=官方地址, DeepSeek=https://api.deepseek.com",
    )
    api_key: str = Field(
        default="",
        description="API Key。优先从环境变量读取，不写在 YAML 中",
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MemoryConfig — 分层记忆配置                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 控制 L1（最近窗口）、L2（语义摘要）、L3（经验归档）的行为参数。
#
# 环境变量覆盖：
#   HORIZON_MEMORY_L1_MAX_TOKENS, HORIZON_MEMORY_L2_MAX_ENTRIES,
#   HORIZON_MEMORY_RETRIEVAL_TOP_K


class MemoryConfig(BaseModel):
    """分层记忆 (L1/L2/L3) 的配置。

    Examples:
        >>> cfg = MemoryConfig()
        >>> cfg.l1_max_tokens
        8000
    """

    # ── L1: 最近窗口 ──
    l1_max_tokens: int = Field(
        default=8000,
        gt=0,
        description="L1 最近窗口的 token 上限。超限时触发 L1→L2 压缩",
    )
    auto_compress_threshold: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="L1 使用率超过此比例时自动触发压缩（0.8=80%）",
    )

    # ── L2: 语义摘要 ──
    l2_summary_model: str = Field(
        default="gpt-4o-mini",
        description="L2 语义压缩使用的轻量模型",
    )
    l2_max_entries: int = Field(
        default=50,
        gt=0,
        description="L2 最多保留的语义摘要条目数。超限时 FIFO 淘汰",
    )

    # ── L3: 经验归档 ──
    l3_backend: str = Field(
        default="faiss",
        description="L3 向量存储后端: faiss (默认, 零依赖) | chromadb (推荐生产使用, 支持元数据过滤)",
    )
    l3_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="L3 向量嵌入使用的模型",
    )
    l3_index_path: str = Field(
        default=".memory/episodic_index",
        description="FAISS/ChromaDB 索引持久化路径",
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="单次检索返回的最大结果数",
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  AgentConfig — Agent 执行参数                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 控制 Agent 运行时的执行参数：步数限制、并发数、超时、重试等。
#
# 环境变量覆盖：
#   HORIZON_AGENT_MAX_STEPS, HORIZON_AGENT_WORKER_SEMAPHORE_LIMIT,
#   HORIZON_AGENT_TASK_TIMEOUT, HORIZON_AGENT_LLM_CALL_TIMEOUT,
#   HORIZON_AGENT_TOOL_CALL_TIMEOUT, HORIZON_AGENT_MAX_RETRIES_PER_TASK


class AgentRuntimeConfig(BaseModel):
    """Agent 运行时执行参数。"""

    max_steps: int = Field(
        default=30,
        gt=0,
        description="单任务最大执行步数。超限则终止",
    )
    worker_semaphore_limit: int = Field(
        default=3,
        ge=1,
        description="同时执行的 Worker 最大数量（asyncio.Semaphore）",
    )
    task_timeout: int = Field(
        default=120,
        gt=0,
        description="单个子任务总超时（秒）",
    )
    llm_call_timeout: int = Field(
        default=30,
        gt=0,
        description="单次 LLM 调用超时（秒）",
    )
    tool_call_timeout: int = Field(
        default=12,
        gt=0,
        description="单次工具调用超时（秒）",
    )
    max_retries_per_task: int = Field(
        default=3,
        ge=0,
        description="单个子任务最多重试次数",
    )
    replan_max_iterations: int = Field(
        default=5,
        ge=0,
        description="单次任务最多重规划次数（防止死循环）",
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LoggingConfig — 轨迹日志配置                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class LoggingConfig(BaseModel):
    """轨迹日志配置。"""

    format: str = Field(
        default="jsonl",
        description="输出格式: jsonl | parquet",
    )
    output_dir: str = Field(
        default="data/traces",
        description="轨迹文件输出目录",
    )
    log_to_console: bool = Field(
        default=False,
        description="是否同时输出到控制台",
    )
    buffer_size: int = Field(
        default=100,
        ge=1,
        description="内存缓冲事件数。攒够此数量再写入磁盘",
    )
    flush_interval: int = Field(
        default=5,
        ge=1,
        description="强制刷新间隔（秒），即使 buffer 未满也写入",
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ToolConfig — 工具配置                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class WebSearchConfig(BaseModel):
    """Web 搜索工具配置。"""

    engine: str = Field(default="duckduckgo", description="搜索引擎")
    max_results: int = Field(default=10, ge=1, description="最大结果数")
    timeout: int = Field(default=12, gt=0, description="超时（秒）")


class ArxivSearchConfig(BaseModel):
    """Arxiv 搜索工具配置。"""

    max_results: int = Field(default=20, ge=1)
    timeout: int = Field(default=12, gt=0)


class PaperSearchConfig(BaseModel):
    """学术论文搜索工具配置。"""

    max_results: int = Field(default=20, ge=1, description="最大返回论文数")
    timeout: int = Field(default=15, gt=0, description="单次搜索超时(秒)")
    rate_limit_per_minute: int = Field(default=10, ge=1, description="每分钟最大请求数")


class CodeExecutionConfig(BaseModel):
    """代码执行工具配置。"""

    sandbox: str = Field(default="subprocess", description="沙箱方式: subprocess | docker")
    timeout: int = Field(default=30, gt=0)
    max_output_chars: int = Field(default=10000, gt=0)


class VerifierConfig(BaseModel):
    """验证器配置。"""

    strict_mode: bool = Field(default=False, description="严格模式, 提高证据数量和质量要求")
    min_evidence_count: int = Field(default=1, ge=0, description="最少证据数量, 低于此值判定为证据不足")


class ToolsConfig(BaseModel):
    """工具总配置。"""

    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    arxiv_search: ArxivSearchConfig = Field(default_factory=ArxivSearchConfig)
    paper_search: PaperSearchConfig = Field(default_factory=PaperSearchConfig)
    code_execution: CodeExecutionConfig = Field(default_factory=CodeExecutionConfig)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TrainingConfig — RL 训练配置 (Phase 3+)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class RewardConfig(BaseModel):
    """RL 奖励函数配置。"""

    task_success: float = Field(default=100.0, description="任务成功奖励")
    tool_call_success: float = Field(default=5.0, description="有效工具调用奖励")
    step_efficiency: float = Field(default=1.0, description="步数效率奖励")
    dead_loop_penalty: float = Field(default=-10.0, description="死循环惩罚")
    timeout_penalty: float = Field(default=-20.0, description="超时惩罚")
    hallucination_penalty: float = Field(default=-5.0, description="幻觉惩罚")


class TrainingConfig(BaseModel):
    """RL 训练配置（Phase 3+，当前仅占位）。"""

    model: str = Field(default="Qwen2.5-3B", description="RL 训练用的基础模型")
    learning_rate: float = Field(default=1.0e-6, gt=0)
    kl_penalty: float = Field(default=0.01, ge=0)
    grpo_epochs: int = Field(default=3, ge=1)
    trajectory_buffer_size: int = Field(default=1000, ge=1)
    num_envs: int = Field(default=4, ge=1)
    reward: RewardConfig = Field(default_factory=RewardConfig)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RootConfig — 顶层配置                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 这是整个配置体系的根。所有子配置通过属性访问。
#
# 环境变量覆盖（HORIZON_ 前缀，双下划线分隔）：
#   HORIZON_DEBUG=true
#   HORIZON_LOG_LEVEL=DEBUG
#   HORIZON_LLM__MODEL=gpt-4o           ← 注意双下划线表示嵌套
#   HORIZON_AGENT__MAX_STEPS=20


class RootConfig(BaseModel):
    """Horizon-Agent 顶层配置。

    聚合所有子配置。通过 getattr 访问：
        cfg.llm.provider
        cfg.memory.l1_max_tokens
        cfg.agent.max_steps

    Examples:
        >>> cfg = RootConfig()
        >>> cfg.llm.model
        'deepseek-chat'
        >>> cfg.llm.provider
        'deepseek'
        >>> cfg.memory.l1_max_tokens
        8000
        >>> cfg.agent.worker_semaphore_limit
        3
    """

    # ── 子配置 ──
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(
        base_url="https://api.deepseek.com",
    ))
    lightweight_llm: LLMConfig = Field(default_factory=lambda: LLMConfig(
        model="deepseek-chat", temperature=0.0, max_tokens=1024,
        base_url="https://api.deepseek.com",
    ))
    embedding: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider="dashscope",
        model="text-embedding-v4",
        temperature=0.0,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ))
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # ── 顶层字段 ──
    debug: bool = Field(default=False, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别: DEBUG|INFO|WARNING|ERROR")
    log_dir: str = Field(default="logs", description="日志文件目录")


# ─── 向后兼容别名 ───────────────────────────────────────────────────────
# 旧代码中使用的类名，映射到新名称。

LLMConfig = LLMConfig       # 不变
MemoryConfig = MemoryConfig  # 不变

# 旧 AgentConfig → 新 RootConfig
# 旧 RolloutConfig → 新 TrainingConfig
AgentConfig = RootConfig     # 向后兼容
RolloutConfig = TrainingConfig  # 向后兼容


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  load_config() — 配置加载入口                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 这是项目中使用配置的唯一入口。加载逻辑：
#
#   1. 从 Pydantic 默认值创建 RootConfig
#   2. 加载 configs/default.yaml（如果存在）
#   3. 加载用户指定的 overlay YAML（深度合并覆盖）
#   4. 加载 .env 文件中的环境变量（HORIZON_ 前缀）
#   5. 覆盖直接设置的环境变量
#
# 深度合并规则：
#   - 嵌套 dict 递归合并
#   - 列表直接替换（不拼接）
#   - 叶子值：overlay 覆盖 base


def load_config(path: str | Path | None = None) -> RootConfig:
    """加载完整配置。

    按优先级从低到高合并：代码默认值 → default.yaml → overlay YAML → .env → 环境变量。

    Args:
        path: 可选的 overlay YAML 文件路径（如 configs/dev.yaml）。
              如果为 None，只加载 default.yaml。

    Returns:
        合并后的 RootConfig 对象。

    Raises:
        FileNotFoundError: 指定的 YAML 文件不存在。
        ValueError: YAML 格式错误或字段验证失败。

    Examples:
        >>> cfg = load_config()                              # 仅默认值
        >>> cfg = load_config("configs/dev.yaml")            # 开发环境
        >>> cfg = load_config("configs/eval.yaml")           # 评测环境
        >>> print(cfg.llm.model)
        'gpt-4o'
    """
    # Step 1: 创建默认配置
    config = RootConfig()

    # Step 2: 加载 default.yaml
    default_yaml = Path("configs/default.yaml")
    if default_yaml.exists():
        config = _merge_yaml(config, default_yaml)

    # Step 3: 加载用户指定的 overlay YAML
    if path is not None:
        overlay_path = Path(path)
        if not overlay_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {overlay_path}")
        config = _merge_yaml(config, overlay_path)

    # Step 4: 加载 .env 文件
    _load_dotenv()

    # Step 5: 覆盖环境变量（最高优先级）
    config = _apply_env_overrides(config)

    # Step 6: 自动注入 API Key（从标准环境变量名）
    config = _inject_api_keys(config)

    return config


def _inject_api_keys(config: RootConfig) -> RootConfig:
    """按 provider 匹配环境变量，自动注入 API Key。

    匹配规则 (对 llm / lightweight_llm / embedding 三个槽位各自独立):
      provider 含 "deepseek"    → DEEPSEEK_API_KEY
      provider 含 "dashscope"   → DASHSCOPE_API_KEY
      provider 含 "openai"      → OPENAI_API_KEY
      provider 含 "anthropic"   → ANTHROPIC_API_KEY
      未匹配到任何 provider      → 尝试 OPENAI_API_KEY 兜底
    """
    import os

    _PROVIDER_KEY_MAP = {
        "deepseek": "DEEPSEEK_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    api_keys = {name: os.environ.get(env_var, "")
                for name, env_var in _PROVIDER_KEY_MAP.items()}

    for slot in ("llm", "lightweight_llm", "embedding"):
        slot_cfg = getattr(config, slot)
        if slot_cfg.api_key:
            continue  # 已设置，不覆盖

        # 按 provider 匹配
        provider = (slot_cfg.provider or "").lower()
        for name, key in api_keys.items():
            if key and name in provider:
                slot_cfg.api_key = key
                break
        else:
            # 兜底：任意可用 Key
            if not slot_cfg.api_key and api_keys.get("openai"):
                slot_cfg.api_key = api_keys["openai"]

    return config


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  内部辅助函数                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _merge_yaml(config: RootConfig, yaml_path: Path) -> RootConfig:
    """将 YAML 文件深度合并到现有配置中。

    合并规则：
      - dict → 递归合并
      - list → 直接替换
      - 标量 → overlay 覆盖 base
      - null 值 → 保留原值（null 表示"不覆盖"）

    Args:
        config: 当前配置对象。
        yaml_path: YAML 文件路径。

    Returns:
        合并后的新配置对象（不修改原对象）。
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        overlay_data = yaml.safe_load(f)

    if not overlay_data:
        return config

    # 获取当前配置的 dict 表示
    base_data = config.model_dump()

    # 深度合并
    merged_data = _deep_merge(base_data, overlay_data)

    # 从合并后的 dict 重建配置对象
    return RootConfig(**merged_data)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个字典。

    overlay 中的值覆盖 base 中的值。嵌套字典递归合并。
    列表和标量直接替换。null 值保留原值。

    Args:
        base: 基础字典。
        overlay: 覆盖字典。

    Returns:
        合并后的新字典。
    """
    result = dict(base)  # 浅拷贝

    for key, value in overlay.items():
        if value is None:
            # null 表示"不覆盖"，保留原值
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 两边都是字典 → 递归合并
            result[key] = _deep_merge(result[key], value)
        else:
            # 列表或标量 → 直接替换
            result[key] = value

    return result


def _load_dotenv() -> None:
    """加载 .env 文件中的环境变量。

    只在 .env 文件存在时加载，不存在时静默跳过。
    使用 python-dotenv 库（如果已安装）。
    """
    env_path = Path(".env")
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv as _load
        _load(env_path, override=True)
    except ImportError:
        _load_dotenv_manual(env_path, override=True)


def _load_dotenv_manual(env_path: Path, override: bool = False) -> None:
    """手动解析 .env 文件（不依赖 python-dotenv 库）。

    Args:
        env_path: .env 文件路径。
        override: 是否覆盖已存在的环境变量。
    """
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value


def _apply_env_overrides(config: RootConfig) -> RootConfig:
    """将 HORIZON_ 前缀的环境变量覆盖到配置中。

    环境变量命名规则：
      HORIZON_DEBUG=true              → config.debug = True
      HORIZON_LOG_LEVEL=DEBUG         → config.log_level = "DEBUG"
      HORIZON_LLM__MODEL=gpt-4o       → config.llm.model = "gpt-4o"
      HORIZON_MEMORY__L1_MAX_TOKENS=6000  → config.memory.l1_max_tokens = 6000
      HORIZON_AGENT__MAX_STEPS=20     → config.agent.max_steps = 20

    注意：双下划线 __ 表示嵌套层级（兼容 Pydantic 的 env_prefix 风格）。
    单下划线用于单词分隔（如 max_steps）。

    Args:
        config: 当前配置对象。

    Returns:
        应用环境变量覆盖后的新配置对象。
    """
    prefix = "HORIZON_"
    overrides: dict[str, Any] = {}

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # 去掉 HORIZON_ 前缀
        config_path = env_key[len(prefix):].lower()

        # HORIZON_LLM__MODEL → llm.model (双下划线=嵌套边界)
        # HORIZON_AGENT__MAX_STEPS → agent.max_steps
        parts = config_path.split("__")

        # 对每个部分，将单下划线转为 Pydantic 字段名（单下划线保留）
        # 例如: max_steps → max_steps (不变)
        #       l1_max_tokens → l1_max_tokens (不变)
        # 实际上 Pydantic model_dump 使用 field name，和 YAML key 一致

        # 转换值为正确类型
        typed_value = _coerce_env_value(env_value)

        # 设置嵌套值
        _set_nested(overrides, parts, typed_value)

    if overrides:
        # 深度合并覆盖
        base_data = config.model_dump()
        merged = _deep_merge(base_data, overrides)
        config = RootConfig(**merged)

    return config


def _coerce_env_value(value: str) -> Any:
    """将环境变量字符串转换为适当的 Python 类型。

    转换规则：
      - "true"/"false" (不区分大小写) → bool
      - 纯数字字符串 → int
      - 带小数点的数字 → float
      - "null"/"none" (不区分大小写) → None
      - 其他 → 保持字符串

    Args:
        value: 环境变量值（字符串）。

    Returns:
        转换后的 Python 值。
    """
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _set_nested(data: dict[str, Any], keys: list[str], value: Any) -> None:
    """在字典中设置嵌套值，自动创建中间层级。

    Args:
        data: 目标字典（原地修改）。
        keys: 嵌套键路径，如 ["llm", "model"]。
        value: 要设置的值。
    """
    for key in keys[:-1]:
        if key not in data:
            data[key] = {}
        data = data[key]
    data[keys[-1]] = value
