# /train — 启动训练

启动 HorizonRL-Agent 的 RL 训练流程：

1. 检查训练配置是否完整（`src/horizonrl/config/settings.py`）
2. 确认 GPU 可用（vLLM 服务是否运行）
3. 运行训练脚本 `scripts/train_grpo.py`
4. 监控训练指标：reward 曲线、成功率、平均步数
5. 训练完成后保存 checkpoint 并生成训练报告

参数：
- `--task`：训练任务类型 (research/coding/data_analysis)
- `--model`：基础模型 (默认 Qwen2.5-3B)
- `--epochs`：训练轮数 (默认 3)
- `--lr`：学习率 (默认 1e-6)
