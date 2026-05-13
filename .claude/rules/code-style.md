# Code Style — Python 编码规范

## 基本原则
- 遵循 PEP 8，ruff 自动检查
- 行宽上限 100 字符
- Python 3.10+ 语法（`str | None` 而非 `Optional[str]`）

## 类型注解
- 所有公开函数必须有完整类型注解
- 使用 `from __future__ import annotations` 启用延迟求值
- TYPE_CHECKING 块用于避免循环导入

## 命名规范
- 类名：PascalCase（`AgentWorker`, `HierarchicalMemory`）
- 函数/方法：snake_case（`execute_task`, `get_context`）
- 常量：UPPER_SNAKE_CASE（`MAX_STEPS`）
- 私有方法：_leading_underscore

## 导入顺序
1. `from __future__ import annotations`
2. 标准库
3. 第三方库
4. 本地模块

## 文档字符串
- 公开类和函数需要单行 docstring 描述用途
- 不要写多段 docstring，保持简洁
- 复杂参数用 Args/Returns 注释

## 禁止事项
- 不要用 `import *`
- 不要用裸 `except:`
- 不要在函数内部写多行注释解释逻辑（好的命名就够了）
- 不要提交 print 调试语句（用 logging）
