# 贡献指南

欢迎贡献！本项目是一个学术研究型的开源项目，目标是将 LLM Agent 的长链路稳定执行能力推进到新的水平。

## 如何贡献

### 报告 Bug

在 Issues 中提交，请包含：
- 运行环境和 Python 版本
- 完整的错误信息
- 复现步骤

### 提交代码

1. Fork 本仓库
2. 创建分支: `git checkout -b feature/your-feature`
3. 提交前运行测试: `python -m pytest tests/`
4. 提交 PR，描述你改了什么、为什么改

### 开发规范

- Python 3.10+ 语法
- 遵循 PEP 8
- 所有公开函数需要类型注解
- 新模块需要对应的测试文件

### 模块依赖方向

```
schemas/  → config/  → tools/  → agent/  → orchestration/
                              → memory/
                              → logging/
```

不要引入循环依赖。所有模块通过 `schemas/` 中定义的数据结构通信。

## 许可证

贡献的代码默认采用 MIT License。
