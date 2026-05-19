"""配置系统测试 — 三级合并、环境变量覆盖、新配置模型。"""

from __future__ import annotations

import os

from horizonrl.config.settings import (
    RootConfig,
    load_config,
)


class TestConfigModels:
    """所有配置模型可正常构造且默认值正确。"""

    def test_root_config_defaults(self):
        cfg = RootConfig()
        assert cfg.llm.provider == "deepseek"
        assert cfg.llm.model == "deepseek-chat"
        assert cfg.memory.l1_max_tokens == 8000
        assert cfg.agent.max_steps == 30
        assert cfg.debug is False
        assert cfg.log_level == "INFO"

    def test_verifier_config_exists(self):
        cfg = RootConfig()
        assert hasattr(cfg, "verifier")
        assert cfg.verifier.strict_mode is False
        assert cfg.verifier.min_evidence_count == 1

    def test_paper_search_config_exists(self):
        cfg = RootConfig()
        assert hasattr(cfg.tools, "paper_search")
        assert cfg.tools.paper_search.max_results == 20
        assert cfg.tools.paper_search.timeout == 15
        assert cfg.tools.paper_search.rate_limit_per_minute == 10

    def test_memory_l3_backend_default(self):
        cfg = RootConfig()
        assert cfg.memory.l3_backend == "chromadb"

    def test_all_configs_in_root(self):
        cfg = RootConfig()
        assert hasattr(cfg, "llm")
        assert hasattr(cfg, "lightweight_llm")
        assert hasattr(cfg, "embedding")
        assert hasattr(cfg, "memory")
        assert hasattr(cfg, "agent")
        assert hasattr(cfg, "verifier")
        assert hasattr(cfg, "logging")
        assert hasattr(cfg, "tools")
        assert hasattr(cfg, "training")


class TestEnvOverrides:
    """HORIZON_ 环境变量覆盖测试。"""

    def setup_method(self):
        self._cleanup_env()

    def teardown_method(self):
        self._cleanup_env()

    @staticmethod
    def _cleanup_env():
        for k in list(os.environ):
            if k.startswith("HORIZON_"):
                del os.environ[k]

    def test_override_llm_model(self):
        os.environ["HORIZON_LLM__MODEL"] = "gpt-4o"
        cfg = load_config()
        assert cfg.llm.model == "gpt-4o"

    def test_override_memory_l3_backend(self):
        os.environ["HORIZON_MEMORY__L3_BACKEND"] = "chromadb"
        cfg = load_config()
        assert cfg.memory.l3_backend == "chromadb"

    def test_override_verifier_strict_mode(self):
        os.environ["HORIZON_VERIFIER__STRICT_MODE"] = "true"
        cfg = load_config()
        assert cfg.verifier.strict_mode is True

    def test_override_agent_semaphore(self):
        os.environ["HORIZON_AGENT__WORKER_SEMAPHORE_LIMIT"] = "8"
        cfg = load_config()
        assert cfg.agent.worker_semaphore_limit == 8

    def test_override_tools_paper_search(self):
        os.environ["HORIZON_TOOLS__PAPER_SEARCH__MAX_RESULTS"] = "50"
        cfg = load_config()
        assert cfg.tools.paper_search.max_results == 50

    def test_override_debug(self):
        os.environ["HORIZON_DEBUG"] = "true"
        cfg = load_config()
        assert cfg.debug is True

    def test_override_log_level(self):
        os.environ["HORIZON_LOG_LEVEL"] = "DEBUG"
        cfg = load_config()
        assert cfg.log_level == "DEBUG"

    def test_multiple_overrides(self):
        os.environ["HORIZON_LLM__MODEL"] = "gpt-4o"
        os.environ["HORIZON_AGENT__MAX_STEPS"] = "20"
        os.environ["HORIZON_MEMORY__L3_BACKEND"] = "chromadb"
        os.environ["HORIZON_VERIFIER__MIN_EVIDENCE_COUNT"] = "5"
        cfg = load_config()
        assert cfg.llm.model == "gpt-4o"
        assert cfg.agent.max_steps == 20
        assert cfg.memory.l3_backend == "chromadb"
        assert cfg.verifier.min_evidence_count == 5


class TestYAMLLoading:
    """YAML 配置文件加载测试。"""

    def test_load_default_yaml(self):
        cfg = load_config()
        assert cfg.llm.provider == "deepseek"
        assert cfg.memory.l1_max_tokens == 10000  # from default.yaml

    def test_load_dev_yaml(self):
        cfg = load_config("configs/dev.yaml")
        assert cfg.debug is True
        assert cfg.log_level == "DEBUG"
        assert cfg.llm.temperature == 0.5  # dev override

    def test_load_eval_yaml(self):
        cfg = load_config("configs/eval.yaml")
        assert cfg.log_level == "WARNING"
        assert cfg.llm.temperature == 0.0
        # eval.yaml verifier section (was silently ignored before VerifierConfig)
        assert cfg.verifier.strict_mode is True
        assert cfg.verifier.min_evidence_count == 3

    def test_yaml_deep_merge_preserves_unset(self):
        """dev.yaml 未设置的字段保留 default.yaml 的值。"""
        cfg = load_config("configs/dev.yaml")
        assert cfg.llm.model == "deepseek-chat"  # from default.yaml
        assert cfg.memory.l2_max_entries == 50   # from default.yaml
        assert cfg.agent.task_timeout == 120      # from default.yaml
