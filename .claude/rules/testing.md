# Testing — 测试规范

## 测试框架
- pytest + pytest-asyncio（async 测试）
- 测试文件放在 `tests/` 目录
- 文件命名：`test_<module>.py`

## 每条规则
1. 每个公开模块至少有一个对应的测试文件
2. 核心路径（happy path）必须有测试
3. 边界情况：空输入、超时、并发冲突
4. 使用 `pytest.mark.skip` 标记可选依赖的测试
5. 测试函数命名：`test_<what>_<condition>` 如 `test_worker_execute_empty_task`

## 运行测试
```bash
python -m pytest tests/ -v          # 全部
python -m pytest tests/test_imports.py -v  # 导入检查
```

## CI 门槛（Phase 4 启用）
- 测试覆盖率 > 70%
- 无失败测试
