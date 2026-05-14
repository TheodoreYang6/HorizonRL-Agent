"""
标准化 Mock 工具 —— Demo 和测试的共享回退。

当外部依赖不可用或不需要真实 API 调用时，使用这些 Mock 工具
保证流程能从起点跑到终点。所有 Demo 和测试统一引用此模块。

使用方式：
    from horizonrl.tools.mock import register_mock_tools
    mgr = ToolManager()
    register_mock_tools(mgr)  # 一键注册全部 3 个 mock 工具
"""

from __future__ import annotations

import json
import time


class MockWebSearch:
    """模拟网页搜索，返回预设结果。"""

    name = "web_search"

    async def search(self, query: str, num_results: int = 5) -> str:
        results = []
        for i in range(1, min(num_results + 1, 4)):
            results.append({
                "title": f"[Mock] 搜索结果 {i}: {query[:30]}",
                "url": f"https://mock-search.local/result-{i}",
                "snippet": (
                    f"这是关于 '{query[:50]}' 的模拟搜索结果 #{i}。"
                    f"内容涵盖相关概念、方法、应用场景与最新进展。"
                ),
            })
        return json.dumps(results, ensure_ascii=False)


class MockArxivSearch:
    """模拟 Arxiv 搜索，返回预设论文。"""

    name = "arxiv_search"

    async def search(self, query: str, max_results: int = 5) -> str:
        papers = [
            {
                "title": f"A Comprehensive Survey on {query[:30]}",
                "authors": ["Zhang, W.", "Li, X.", "Wang, H."],
                "abstract": (
                    f"本文全面综述了 {query[:50]} 领域的最新进展，"
                    f"涵盖主流方法和应用场景。在多个公开基准上进行了系统对比实验。"
                ),
                "url": "https://arxiv.org/abs/2501.00001",
                "pdf_url": "https://arxiv.org/pdf/2501.00001",
                "published": "2025-01-15",
                "categories": ["cs.AI", "cs.CL"],
            },
            {
                "title": f"Advances in {query[:30]}: A New Approach",
                "authors": ["Chen, Y.", "Liu, J."],
                "abstract": (
                    f"提出了一种针对 {query[:50]} 的创新方法，"
                    f"在性能和效率上显著优于现有方案。代码已开源。"
                ),
                "url": "https://arxiv.org/abs/2502.00002",
                "pdf_url": "https://arxiv.org/pdf/2502.00002",
                "published": "2025-02-20",
                "categories": ["cs.LG"],
            },
        ]
        return json.dumps(papers[:max_results], ensure_ascii=False)


class MockCodeExecution:
    """模拟代码执行，返回预设输出（与 CodeExecutionTool 返回格式一致）。"""

    name = "code_execution"

    async def execute(self, code: str = "", **kwargs) -> dict:
        if not code or not code.strip():
            return {
                "stdout": "",
                "stderr": "",
                "success": True,
                "error": None,
            }
        return {
            "stdout": (
                f"[Mock 执行] 代码长度: {len(code)} 字符\n"
                f"输出: 代码运行成功，结果符合预期。\n"
                f"时间戳: {time.time()}"
            ),
            "stderr": "",
            "success": True,
            "error": None,
        }


def register_mock_tools(manager) -> None:
    """向 ToolManager 一键注册全部 3 个模拟工具。

    注册后可通过 manager.call() 统一调用，行为与真实工具一致。

    Args:
        manager: ToolManager 实例。

    Examples:
        >>> mgr = ToolManager()
        >>> register_mock_tools(mgr)
        >>> "web_search" in mgr.list_tools()
        True
    """
    manager.register("web_search", MockWebSearch())
    manager.register("arxiv_search", MockArxivSearch())
    manager.register("code_execution", MockCodeExecution())
