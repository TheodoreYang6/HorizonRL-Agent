# /eval — 运行评测

对训练好的 Agent 进行基准评测：

1. 加载指定 checkpoint
2. 运行3个基准任务（research/coding/data_analysis）
3. 收集指标：成功率、Pass@k、平均步数、Token消耗
4. 与 Baseline/No-Memory/No-Replan 对比
5. 生成评测报告

参数：
- `--checkpoint`：模型 checkpoint 路径
- `--task`：评测任务 (all/research/coding/data_analysis)
- `--runs`：每个样本运行次数 (默认 5)
