# /check-imports — 检查包导入

运行导入检查测试，验证所有依赖包是否正确安装：

```bash
python -m pytest tests/test_imports.py -v
```

检查范围：
- Core 依赖（Phase 1）：langgraph, langchain, openai, anthropic, httpx, aiohttp, numpy, pydantic, yaml, tiktoken, faiss, rich, tqdm
- RL 依赖（Phase 3, 可选）：trl, torch, accelerate, verl
- 推理依赖（可选）：vllm
- 评测依赖（Phase 4, 可选）：scipy, sklearn, pandas, matplotlib, seaborn
- 内部模块：horizonrl 全部子模块
