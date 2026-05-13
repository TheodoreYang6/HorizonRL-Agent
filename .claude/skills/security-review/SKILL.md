# Security Review Skill

自动审查代码变更中的安全问题。

## 触发条件
- 文件涉及 API key 处理
- 文件涉及 subprocess / code execution
- 文件涉及用户输入处理
- 文件涉及网络请求

## 检查项
1. API Key 是否硬编码（必须是环境变量或配置注入）
2. subprocess 调用是否使用 `shell=False`（或参数列表形式）
3. 用户输入是否经过验证（Pydantic 验证）
4. HTTP 请求是否有超时设置
5. 文件路径是否有路径遍历风险
6. 是否有 eval/exec 使用（除非明确需要）

## 输出
安全风险列表 + 修复建议，按严重度排序（Critical > High > Medium > Low）
