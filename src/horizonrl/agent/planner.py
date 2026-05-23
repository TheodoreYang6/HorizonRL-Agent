"""
Planner —— 任务分解模块。

将用户自然语言任务拆解为结构化的 TaskSpec 列表和 PlanGraph DAG。
MVP 阶段使用模板分解，Phase 2+ 接入 LLM 做语义分解。

输入：UserTask
输出：PlanGraph（含 TaskSpec[] + 依赖边）
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from horizonrl.schemas.task import (
    PlanGraph,
    PlanNode,
    TaskPriority,
    TaskSpec,
    UserTask,
)

if TYPE_CHECKING:
    from horizonrl.config.settings import RootConfig


def _short_id() -> str:
    """生成短 ID：task_ + UUID 前 8 位。"""
    return f"task_{uuid.uuid4().hex[:8]}"


def _build_plan_graph(specs: list[TaskSpec]) -> PlanGraph:
    """从 TaskSpec 列表构建 PlanGraph。Planner 和 LLMPlanner 共用。"""
    nodes: dict[str, PlanNode] = {}
    edges: dict[str, list[str]] = {}
    root_ids: list[str] = []

    for spec in specs:
        node = PlanNode(spec=spec)
        nodes[spec.id] = node
        edges[spec.id] = list(spec.depends_on)
        if not spec.depends_on:
            root_ids.append(spec.id)

    return PlanGraph(
        nodes=nodes,
        edges=edges,
        root_ids=root_ids,
        total_tokens_spent=0,
        created_at=time.time(),
    )


# ─── 任务分解模板 ─────────────────────────────────────────────────────────
# MVP 阶段使用静态模板。每个模板定义一组 TaskSpec，包含依赖关系。
# Phase 2+ 会替换为 LLM 驱动的动态分解。


_RESEARCH_TEMPLATE: list[dict] = [
    {
        "name": "检索背景信息",
        "description": "搜索 '{topic}' 的基础概念、定义、历史背景",
        "tool_names": ["web_search"],
        "depends_on": [],
        "priority": TaskPriority.P0,
    },
    {
        "name": "检索最新进展",
        "description": "搜索 '{topic}' 在 {year} 年的最新研究和方法",
        "tool_names": ["web_search", "paper_search"],
        "depends_on": [],
        "priority": TaskPriority.P0,
    },
    {
        "name": "分析对比方法",
        "description": "搜索并对比 '{topic}' 中不同方法的优劣",
        "tool_names": ["web_search"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P1,
    },
    {
        "name": "检索局限性",
        "description": "搜索 '{topic}' 当前方法的局限和挑战",
        "tool_names": ["web_search"],
        "depends_on": [],
        "priority": TaskPriority.P1,
    },
    {
        "name": "综合汇总",
        "description": "搜索 '{topic}' 的未来展望，将所有发现综合为结构化摘要",
        "tool_names": ["web_search"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P2,
    },
]

_CODE_TEMPLATE: list[dict] = [
    {
        "name": "理解代码结构",
        "description": "阅读项目中的 '{topic}' 相关代码，理解逻辑和依赖",
        "tool_names": ["code_execution"],
        "depends_on": [],
        "priority": TaskPriority.P0,
    },
    {
        "name": "定位问题",
        "description": "根据 '{topic}' 描述定位具体的 bug 或问题位置",
        "tool_names": ["web_search"],
        "depends_on": [],
        "priority": TaskPriority.P0,
    },
    {
        "name": "运行现有代码",
        "description": "运行 '{topic}' 相关代码，记录错误信息",
        "tool_names": ["code_execution"],
        "depends_on": [],
        "priority": TaskPriority.P1,
    },
    {
        "name": "修复代码",
        "description": "根据错误信息和搜索结果修复代码",
        "tool_names": ["code_execution"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P1,
    },
    {
        "name": "验证修复",
        "description": "运行修复后的代码，确认问题已解决",
        "tool_names": ["code_execution"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P2,
    },
]

_COMPARISON_TEMPLATE: list[dict] = [
    {
        "name": "检索对象A信息",
        "description": "搜索 '{topic}' 中第一个对比对象的详细信息",
        "tool_names": ["web_search"],
        "depends_on": [],
        "priority": TaskPriority.P0,
    },
    {
        "name": "检索对象B信息",
        "description": "搜索 '{topic}' 中第二个对比对象的详细信息",
        "tool_names": ["web_search"],
        "depends_on": [],
        "priority": TaskPriority.P0,
    },
    {
        "name": "提取对比维度",
        "description": "搜索 '{topic}' 的通用对比指标和评价标准",
        "tool_names": ["web_search"],
        "depends_on": [],
        "priority": TaskPriority.P1,
    },
    {
        "name": "逐维度对比",
        "description": "按提取的维度对 '{topic}' 中的两者做定量/定性对比",
        "tool_names": ["web_search"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P1,
    },
    {
        "name": "生成对比结论",
        "description": "基于对比结果生成 '{topic}' 的优劣分析及选型建议",
        "tool_names": ["web_search"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P2,
    },
]

_SUMMARY_TEMPLATE: list[dict] = [
    {
        "name": "多源收集",
        "description": "从多个来源收集 '{topic}' 的相关信息",
        "tool_names": ["web_search", "paper_search"],
        "depends_on": [],
        "priority": TaskPriority.P0,
    },
    {
        "name": "信息去重与排序",
        "description": "对收集到的信息去重并按相关性排序",
        "tool_names": ["web_search"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P1,
    },
    {
        "name": "结构化汇总",
        "description": "将 '{topic}' 的信息按主题/时间线/重要性组织为结构化摘要",
        "tool_names": ["web_search"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P2,
    },
]

_FACTUAL_QA_TEMPLATE: list[dict] = [
    {
        "name": "权威来源检索",
        "description": "从权威来源搜索 '{topic}' 的事实性信息",
        "tool_names": ["web_search"],
        "depends_on": [],
        "priority": TaskPriority.P0,
    },
    {
        "name": "交叉验证",
        "description": "用不同来源交叉验证 '{topic}' 的关键事实",
        "tool_names": ["web_search", "paper_search"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P1,
    },
    {
        "name": "生成答案",
        "description": "基于验证后的事实生成 '{topic}' 的准确答案，标注置信度",
        "tool_names": ["web_search"],
        "depends_on": [],  # 动态填充
        "priority": TaskPriority.P2,
    },
]


class Planner:
    """任务分解器 —— 将 UserTask 拆解为可执行的 PlanGraph。

    MVP 阶段使用模板匹配做任务分解（无需 LLM）。
    Phase 2+ 接入 LLM 做语义分解和动态工具选择。

    Examples:
        >>> planner = Planner()
        >>> task = UserTask(description="调研 Transformer 注意力机制")
        >>> plan = planner.plan(task)
        >>> len(plan.nodes) >= 3
        True
        >>> plan.root_ids  # 第一批可并行执行的任务
        [...]
    """

    def __init__(self, config: RootConfig | None = None):
        self.config = config
        self._templates = {
            "research": _RESEARCH_TEMPLATE,
            "code": _CODE_TEMPLATE,
            "comparison": _COMPARISON_TEMPLATE,
            "summary": _SUMMARY_TEMPLATE,
            "factual_qa": _FACTUAL_QA_TEMPLATE,
        }

    def plan(self, task: UserTask) -> PlanGraph:
        """将用户任务分解为 PlanGraph。

        根据任务类型选择模板，生成 TaskSpec 列表，构建 DAG。

        Args:
            task: 用户输入的原始任务。

        Returns:
            完整的 PlanGraph，包含节点、依赖边、根节点列表。
        """
        # 判断任务类型
        task_type = self._classify_task(task)
        template = self._templates.get(task_type, _RESEARCH_TEMPLATE)

        # 生成所有 TaskSpec
        import time as _time
        current_year = _time.strftime('%Y')
        specs: list[TaskSpec] = []
        spec_ids: list[str] = []
        for tmpl in template:
            desc = tmpl["description"]
            if callable(desc):
                desc = desc(topic=task.description, year=current_year)
            else:
                desc = desc.format(topic=task.description, year=current_year)
            spec = TaskSpec(
                id=_short_id(),
                name=tmpl["name"],
                description=desc,
                tool_names=list(tmpl["tool_names"]),
                depends_on=[],  # 稍后填充
                priority=tmpl.get("priority", TaskPriority.P1),
            )
            specs.append(spec)
            spec_ids.append(spec.id)

        # 建立依赖关系（模板中的 depends_on 用索引引用前序任务）
        for i, tmpl in enumerate(template):
            deps = []
            for dep_idx in tmpl["depends_on"]:
                if 0 <= dep_idx < len(spec_ids):
                    deps.append(spec_ids[dep_idx])
            specs[i].depends_on = deps

        # 根据模板类型设置具体依赖
        if task_type == "research":
            # 分析对比 → 依赖 检索背景 + 最新进展
            specs[2].depends_on = [spec_ids[0], spec_ids[1]]
            # 综合汇总 → 依赖 分析对比 + 检索局限性
            specs[4].depends_on = [spec_ids[2], spec_ids[3]]
        elif task_type == "code":
            # 修复代码 → 依赖 运行代码 + 定位问题
            specs[3].depends_on = [spec_ids[1], spec_ids[2]]
            # 验证修复 → 依赖 修复代码
            specs[4].depends_on = [spec_ids[3]]
        elif task_type == "comparison":
            # 逐维度对比 → 依赖 对象A + 对象B + 对比维度
            specs[3].depends_on = [spec_ids[0], spec_ids[1], spec_ids[2]]
            # 生成对比结论 → 依赖 逐维度对比
            specs[4].depends_on = [spec_ids[3]]
        elif task_type == "summary":
            # 信息去重与排序 → 依赖 多源收集
            specs[1].depends_on = [spec_ids[0]]
            # 结构化汇总 → 依赖 信息去重与排序
            specs[2].depends_on = [spec_ids[1]]
        elif task_type == "factual_qa":
            # 交叉验证 → 依赖 权威来源检索
            specs[1].depends_on = [spec_ids[0]]
            # 生成答案 → 依赖 交叉验证
            specs[2].depends_on = [spec_ids[1]]

        return _build_plan_graph(specs)

    def _classify_task(self, task: UserTask) -> str:
        """根据任务描述和 required_tools 判断任务类型。

        支持类型: code, comparison, summary, factual_qa, research (默认).
        """
        # 强制工具优先
        if "code_execution" in task.required_tools:
            return "code"

        desc = task.description.lower()

        # code: 代码/调试/修复相关
        code_kw = ("代码", "修复", "bug", "debug", "编程", "函数", "测试",
                   "报错", "异常", "重构", "refactor", "编译", "运行")
        if any(kw in desc for kw in code_kw):
            return "code"

        # comparison: 对比/比较/选型相关
        cmp_kw = ("对比", "比较", "区别", "差异", "优劣", "选型", "哪个更好",
                  "哪个更", "区别是什么", "有何不同", "compare", "vs", "versus",
                  "差别", "优缺点", "利弊", "抉择")
        if any(kw in desc for kw in cmp_kw):
            return "comparison"

        # summary: 汇总/总结/概述相关
        sum_kw = ("汇总", "总结", "概述", "梳理", "归纳", "综述", "概览",
                  "一览", "整理", "汇总一下", "总结一下", "summarize",
                  "overview", "梳理一下")
        if any(kw in desc for kw in sum_kw):
            return "summary"

        # factual_qa: 事实问答/定义/查询
        qa_kw = ("是什么", "什么是", "定义", "谁", "何时", "哪里", "多少",
                 "哪个", "列出", "列举", "查询", "百科", "what is",
                 "who is", "when", "where", "define")
        if any(kw in desc for kw in qa_kw):
            return "factual_qa"

        return "research"


class LLMPlanner:
    """LLM 驱动的任务分解器 —— 用大模型将 UserTask 拆解为 PlanGraph。

    相比模板 Planner，LLMPlanner 能理解任意任务类型，动态选择工具，
    生成更合理的依赖关系和优先级。

    Examples:
        >>> from horizonrl.llm import LLMClient
        >>> client = LLMClient(config.llm)
        >>> planner = LLMPlanner(client)
        >>> plan = await planner.plan(UserTask(description="调研 LLaMA 架构"))
        >>> print(plan.total_count(), "个子任务")
    """

    def __init__(self, llm_client, config=None, tool_manager=None):
        self.llm = llm_client
        self.config = config
        self.tool_manager = tool_manager

    async def plan(self, task: UserTask) -> PlanGraph:
        """用 LLM 将用户任务分解为 PlanGraph。

        Args:
            task: 用户输入的原始任务。

        Returns:
            完整的 PlanGraph，包含节点、依赖边、根节点列表。
        """
        prompt = self._build_prompt(task)
        result = await self.llm.chat(prompt, system_prompt=self._system_prompt())

        if not result.is_success:
            # LLM 调用失败，回退到模板 Planner
            fallback = Planner(self.config)
            return fallback.plan(task)

        specs = self._parse_response(result.content, task)
        return self._build_graph(specs)

    def _system_prompt(self) -> str:
        import time as _time
        today = _time.strftime('%Y年%m月%d日')
        return (
            f"你是一个任务规划专家。将用户的问题拆解为可并行执行的子任务。"
            f"注意：当前日期是 {today}。如用户问\"最新进展\"，请在搜索描述中加入最近的年份。"
            f"核心原则：能并行的任务绝不串行。只有真正依赖前一步结果的任务才加依赖。"
            f"简单问题 3-4 个任务即可，复杂问题 5-6 个。只输出 JSON，不解释。"
        )

    def _build_prompt(self, task: UserTask) -> str:
        tools_hint = ""
        if task.required_tools:
            tools_hint = f"必须使用的工具: {', '.join(task.required_tools)}"
        elif self.tool_manager is not None:
            all_tools = self.tool_manager.list_tools()
            tools_hint = f"可选工具: {', '.join(all_tools)}"
        else:
            tools_hint = "可选工具: web_search, paper_search, code_execution"

        return f"""将以下问题拆解为子任务。重点是最大化并行度——互相不依赖的任务必须设为空依赖数组。

问题: {task.description}
{tools_hint}

输出 JSON 数组，每个元素:
{{"name": "任务名", "description": "详细描述", "tool_names": ["web_search"], "depends_on": [], "priority": "p0"}}

关键规则:
- depends_on 用整数索引，空数组[]表示可并行执行
- 只有真正需要前一步输出结果的任务才加依赖，否则全部设为[]
- p0=关键路径(先执行), p1=正常, p2=可选后置
- 简单问题拆3-4个，复杂问题拆5-6个
- 只输出JSON数组

JSON:"""

    def _parse_response(self, content: str, task: UserTask) -> list[TaskSpec]:
        """解析 LLM 返回的 JSON，生成 TaskSpec 列表。"""
        import json
        import re

        # 提取 JSON 数组
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            return []

        try:
            raw_items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        # 先创建所有 TaskSpec（depends_on 稍后用索引填充）
        specs: list[TaskSpec] = []
        spec_ids: list[str] = []
        for item in raw_items:
            sid = _short_id()
            priority = TaskPriority.P1
            if isinstance(item.get("priority"), str):
                pval = item["priority"].lower()
                if pval in ("p0", "p1", "p2"):
                    priority = TaskPriority(pval)

            # 动态获取可用工具列表（含插件）
            valid_tools = set(
                self.tool_manager.list_tools()
            ) if self.tool_manager else {"web_search", "paper_search", "code_execution"}

            spec = TaskSpec(
                id=sid,
                name=str(item.get("name", f"子任务{len(specs)+1}")),
                description=str(item.get("description", "")),
                tool_names=[t for t in item.get("tool_names", [])
                           if t in valid_tools],
                depends_on=[],  # 稍后填充
                priority=priority,
            )
            specs.append(spec)
            spec_ids.append(sid)

        # 用 spec_ids 替换索引依赖
        for i, item in enumerate(raw_items):
            raw_deps = item.get("depends_on", [])
            if isinstance(raw_deps, list):
                deps = []
                for d in raw_deps:
                    if isinstance(d, int) and 0 <= d < len(spec_ids) and d != i:
                        deps.append(spec_ids[d])
                specs[i].depends_on = deps

        return specs

    def _build_graph(self, specs: list[TaskSpec]) -> PlanGraph:
        """从 TaskSpec 列表构建 PlanGraph（委托共享函数）。"""
        return _build_plan_graph(specs)
