# /test — 运行测试

运行 HorizonRL-Agent 测试套件 (138 tests)：

```bash
# 全部测试
python -m pytest tests/ -v

# 单模块
python -m pytest tests/test_verifier.py -v
python -m pytest tests/test_dag_workflow.py -v
python -m pytest tests/test_tools_manager.py -v

# 快速冒烟 (仅导入 + 核心)
python -m pytest tests/test_imports.py tests/test_planner.py tests/test_worker.py -v
```

当前状态: 138 passed, 4 skipped, 0 failed
