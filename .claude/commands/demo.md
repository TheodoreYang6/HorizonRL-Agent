# /demo — 运行 Demo

启动 HorizonRL-Agent 的三个 Demo：

- `02_simple_agent.py` — 端到端 Demo（无需 API Key，模板规划 + 模拟工具）
- `03_llm_demo.py` — LLM 驱动 Demo（需 API Key，LLM 智能规划 + 全链路执行）

用法：
```
/demo              → 运行 02_simple_agent.py
/demo llm          → 运行 03_llm_demo.py (需 .env 配置)
/demo llm 你的问题  → LLM Demo + 自定义问题
```

运行前确认：
- `.env` 文件已配置 (仅 LLM Demo 需要)
- `configs/dev.yaml` 中 base_url 和 model 正确
