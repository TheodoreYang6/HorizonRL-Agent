"""
共享 Research Service —— CLI / Web / Benchmark 统一入口。

所有入口 (examples/04, 05, 07, benchmark runner) 通过此模块调用
同一个 ResearchOrchestrator，保证报告一致性。

使用方式:
    from horizonrl.services import run_research_session, SessionArtifacts

    artifacts = await run_research_session(
        query="Transformer注意力机制",
        mode="deep",
        llm_client=client,
    )
    print(artifacts.final_answer_text)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from horizonrl.agent.planner import Planner
from horizonrl.agent.writer import Writer, WriterConfig
from horizonrl.config.settings import RootConfig, load_config
from horizonrl.llm.client import LLMClient
from horizonrl.logging.trajectory_logger import TrajectoryLogger
from horizonrl.memory.hierarchical_memory import HierarchicalMemory
from horizonrl.orchestration.dag_workflow import (
    ResearchOrchestrator,
    create_orchestrator,
    _to_dict,
    _from_dict,
)
from horizonrl.schemas.event import EventType, TrajectoryEvent
from horizonrl.schemas.result import StepResult
from horizonrl.schemas.task import PlanGraph


@dataclass
class TaskDetail:
    """单个子任务的执行详情。"""
    task_id: str = ""
    name: str = ""
    status: str = ""
    score: float = 0.0
    passed: bool = False
    evidence_count: int = 0
    tool_calls: int = 0
    elapsed: float = 0.0
    error_type: str = ""
    feedback: str = ""


@dataclass
class SessionArtifacts:
    """一次研究会话的完整产出。

    Attributes:
        session_id: 会话唯一标识
        mode_resolved: 实际采用的模式 (chat / deep)
        final_answer_text: final_answer.md 的文本内容
        final_answer_path: final_answer.md 磁盘路径
        debug_report_path: debug_report.md 磁盘路径
        trajectory_path: 轨迹 JSONL 磁盘路径
        task_details: 每个子任务的执行详情
        stats: 执行统计 {total_count, success_count, rounds, tool_calls, replans, elapsed}
        mock_ratio: 模拟数据占比 (0.0 ~ 1.0)
        tool_calls_count: 工具调用总次数
        runtime_ms: 总耗时 (毫秒)
        used_search_provider: 实际使用的搜索提供商
        error: 异常信息 (成功时为 "")
    """

    session_id: str = ""
    mode_resolved: str = "deep"
    final_answer_text: str = ""
    final_answer_path: str = ""
    debug_report_path: str = ""
    trajectory_path: str = ""
    task_details: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    mock_ratio: float = 0.0
    tool_calls_count: int = 0
    runtime_ms: float = 0.0
    used_search_provider: str = ""
    error: str = ""


# ─── 模式判断 ────────────────────────────────────────────────────────────────

_CHAT_KEYWORDS = (
    "你好", "你是谁", "谢谢", "再见", "帮助", "help",
    "hello", "hi", "what are you", "who are you",
)

_DEEP_KEYWORDS = (
    "研究", "调研", "分析", "对比", "比较", "总结", "综述",
    "搜索", "查找", "检索", "最新", "进展", "原理", "机制",
    "是什么", "什么是", "如何", "为什么", "区别",
    "research", "analyze", "compare", "survey", "summarize",
    "search", "find", "latest", "explain", "investigate",
)


def resolve_mode(query: str, explicit: str = "auto") -> str:
    """根据查询内容自动判断 chat / deep 模式。

    Args:
        query: 用户输入。
        explicit: 显式指定的模式，非 auto 时直接返回。

    Returns:
        "chat" 或 "deep"
    """
    if explicit not in ("auto", ""):
        return explicit

    query_lower = query.lower().strip()

    # 短问候 → chat
    if len(query_lower) < 10:
        for kw in _CHAT_KEYWORDS:
            if kw in query_lower:
                return "chat"

    # 研究型关键词 → deep
    for kw in _DEEP_KEYWORDS:
        if kw in query_lower:
            return "deep"

    # 默认: 短问题 chat, 长问题 deep
    return "deep" if len(query) > 30 else "chat"


# ─── 主入口 ───────────────────────────────────────────────────────────────────


async def run_research_session(
    query: str,
    mode: str = "auto",
    session_id: str | None = None,
    llm_client=None,
    tool_manager=None,
    config: RootConfig | None = None,
    writer: Writer | None = None,
    search_provider: str = "auto",
    offline: bool = False,
    semaphore_limit: int = 3,
    max_iterations: int = 10,
    export_dir: str = "reports",
    on_token=None,  # async callable(str) for token streaming
) -> SessionArtifacts:
    """执行一次完整的研究会话 —— CLI / Web / Benchmark 统一入口。

    内部调用 ResearchOrchestrator.run() 走 LangGraph 全链路:
    plan_task → mark_ready → execute_batch → verify_batch → replan → finalize

    Args:
        query: 用户问题 (自然语言)。
        mode: "auto" | "chat" | "deep" — 执行模式。
        session_id: 指定会话 ID，None 则自动生成。
        llm_client: LLM 客户端，None 则使用模板模式。
        tool_manager: 工具管理器，None 则自动创建。
        config: 全局配置，None 则使用默认值。
        writer: Writer 实例，None 则自动创建。
        search_provider: 搜索提供商偏好 (auto/bocha/brave/duckduckgo/mock)。
        offline: 是否强制离线/Mock 模式。
        semaphore_limit: 最大并发 Worker 数。
        max_iterations: 最大执行轮数。
        export_dir: 报告输出目录。

    Returns:
        SessionArtifacts 包含完整会话产出。
    """
    t_start = time.monotonic()
    mode_resolved = resolve_mode(query, mode)
    sid = session_id or f"session_{uuid.uuid4().hex[:12]}"

    artifacts = SessionArtifacts(
        session_id=sid,
        mode_resolved=mode_resolved,
    )

    # ── chat 模式: 不走 Agent 管道 ──
    if mode_resolved == "chat":
        if llm_client is not None:
            result = await llm_client.chat(
                query,
                system_prompt="你是一个友好、专业的AI助手。用流畅的中文回答用户。",
            )
            artifacts.final_answer_text = result.content if result.is_success else query
        else:
            artifacts.final_answer_text = (
                "你好！我是 HorizonRL-Agent。\n\n"
                "我可以帮你做深度研究：搜索资料、对比分析、汇总报告。\n"
                "试试输入一个研究问题，比如「Transformer注意力机制的最新进展」。"
            )
        artifacts.runtime_ms = (time.monotonic() - t_start) * 1000
        return artifacts

    # ── deep 模式: 走全链路 ──
    try:
        # 1. 配置
        if config is None:
            try:
                config = load_config(
                    Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None
                )
            except Exception:
                config = RootConfig()

        if offline:
            search_provider = "mock"

        # 2. ToolManager
        if tool_manager is None:
            tool_manager = _build_default_tool_manager(search_provider, offline)

        # 3. Planner
        planner: Planner = Planner(config)

        # 4. LLMPlanner (如果有 LLM)
        if llm_client is not None:
            from horizonrl.agent.planner import LLMPlanner
            planner = LLMPlanner(llm_client)

        # 5. Writer
        if writer is None:
            writer_mode = "llm" if llm_client is not None else "template"
            writer = Writer(
                mode=writer_mode,
                llm_client=llm_client,
                config=WriterConfig(export_dir=export_dir),
            )

        # 6. 轨迹日志 — 创建并启动
        traj_logger = TrajectoryLogger(output_dir="trajectories")
        traj_sid = await traj_logger.start_session(query)
        # 统一 sid: logger → service → orchestrator → writer
        sid = session_id or traj_sid
        artifacts.session_id = sid

        # 7. Embedding 客户端 — L3 向量检索 (有 Key 则用真实 API, 否则 n-gram)
        embedding_client = None
        if config.embedding.api_key:
            try:
                embedding_client = LLMClient(config.embedding)
            except Exception:
                pass

        # 8. Orchestrator — 注入 logger + embedding 实现 per-node 事件 + L3 真实向量
        orchestrator = ResearchOrchestrator(
            planner=planner,
            tool_manager=tool_manager,
            semaphore_limit=semaphore_limit,
            max_iterations=max_iterations,
            writer=writer,
            embedding_client=embedding_client,
            trajectory_logger=traj_logger,
            on_token=on_token,
        )
        state = await orchestrator.run(query, session_id=sid)

        # 9. 收集产出
        plan = state.get("plan")
        raw_results = state.get("results", {})
        replan_count = state.get("replan_count", 0)
        iteration = state.get("iteration", 0)
        error = state.get("error", "")
        total_elapsed = time.monotonic() - t_start

        # 检测空 PlanGraph (LLM 返回无法解析或 Planner 失败)
        if plan is not None and plan.total_count() == 0 and not error:
            error = (
                f"任务规划失败: 未能生成有效的子任务。"
                f"{'LLM 返回结果无法解析，' if llm_client else ''}"
                f"已回退到模板规划但仍失败，请检查任务描述。"
            )

        # 10. 结束轨迹日志 — 注入真实统计
        if traj_logger._session is not None:
            total_calls = sum(
                len(r.get("tool_calls", [])) for r in raw_results.values()
                if isinstance(r, dict)
            )
            traj_logger._session.total_tool_calls = total_calls
            traj_logger._session.replan_count = replan_count
            traj_logger._session.total_tokens = sum(
                r.get("tokens_used", 0) for r in raw_results.values()
                if isinstance(r, dict)
            )
        success = plan.success_count() == plan.total_count() if plan else False
        await traj_logger.end_session(success=success)

        # 统计
        total_count = plan.total_count() if plan else 0
        success_count = plan.success_count() if plan else 0
        total_tool_calls = 0
        total_evidence = 0
        mock_evidence = 0

        for r_dict in raw_results.values():
            if isinstance(r_dict, dict):
                # 工具调用
                for tc in r_dict.get("tool_calls", []):
                    total_tool_calls += 1
                # 证据 — 兼容 EvidenceItem 对象和 dict
                for ev in r_dict.get("evidence", []):
                    total_evidence += 1
                    is_mock = False
                    if isinstance(ev, dict):
                        is_mock = ev.get("is_mock", False)
                    elif hasattr(ev, "is_mock"):
                        is_mock = ev.is_mock
                    if is_mock:
                        mock_evidence += 1

        mock_ratio = mock_evidence / max(total_evidence, 1)

        # 搜索提供商 — 从工具实例获取实际运行时 provider
        used_provider = search_provider
        if tool_manager is not None and "web_search" in tool_manager._tools:
            ws_tool = tool_manager._tools["web_search"]
            actual = getattr(ws_tool, "actual_provider", "")
            if actual:
                used_provider = actual

        # 收集每个子任务的执行详情
        task_details = []
        if plan is not None:
            for node in plan.nodes.values():
                r_dict = raw_results.get(node.spec.id, {})
                v_dict = (
                    state.get("verifications", {}).get(node.id, {})
                    if isinstance(state.get("verifications", {}), dict)
                    else {}
                )
                node_elapsed = 0.0
                if node.finished_at and node.started_at:
                    node_elapsed = node.finished_at - node.started_at
                td = TaskDetail(
                    task_id=node.id,
                    name=node.spec.name,
                    status=node.status.value if hasattr(node.status, "value") else str(node.status),
                    score=v_dict.get("score", 0.0) if isinstance(v_dict, dict) else 0.0,
                    passed=node.status.value == "success" if hasattr(node.status, "value") else False,
                    evidence_count=len(r_dict.get("evidence", [])) if isinstance(r_dict, dict) else 0,
                    tool_calls=len(r_dict.get("tool_calls", [])) if isinstance(r_dict, dict) else 0,
                    elapsed=node_elapsed,
                    error_type=v_dict.get("error_type", "") if isinstance(v_dict, dict) else "",
                    feedback=v_dict.get("feedback", "") if isinstance(v_dict, dict) else "",
                )
                task_details.append(td)
        artifacts.task_details = task_details

        # Report 路径
        final_path = Path(export_dir) / sid / "final_answer.md"
        debug_path = Path(export_dir) / sid / "debug_report.md"
        trajectory_path = traj_logger.output_dir / f"{sid}.jsonl"

        final_text = state.get("final_output", "")
        if not final_text and final_path.exists():
            final_text = final_path.read_text(encoding="utf-8")

        artifacts.final_answer_text = final_text
        artifacts.final_answer_path = str(final_path)
        artifacts.debug_report_path = str(debug_path)
        artifacts.trajectory_path = str(trajectory_path)
        artifacts.stats = {
            "total_count": total_count,
            "success_count": success_count,
            "rounds": iteration,
            "total_tool_calls": total_tool_calls,
            "total_replans": replan_count,
            "total_evidence": total_evidence,
            "total_elapsed": f"{total_elapsed:.1f}s",
        }
        artifacts.mock_ratio = mock_ratio
        artifacts.tool_calls_count = total_tool_calls
        artifacts.runtime_ms = total_elapsed * 1000
        artifacts.used_search_provider = used_provider
        artifacts.error = error

    except Exception as exc:
        artifacts.error = str(exc)
        artifacts.runtime_ms = (time.monotonic() - t_start) * 1000
        try:
            await traj_logger.end_session(success=False)
        except Exception:
            pass

    return artifacts


async def stream_research_session(
    query: str,
    mode: str = "deep",
    session_id: str | None = None,
    llm_client=None,
    tool_manager=None,
    config: RootConfig | None = None,
    writer: Writer | None = None,
    search_provider: str = "auto",
    offline: bool = False,
    semaphore_limit: int = 3,
    max_iterations: int = 10,
    export_dir: str = "reports",
    on_token=None,  # async callable(str) for token streaming
) -> AsyncIterator[dict]:
    """流式执行研究会话 — 每完成一个 LangGraph 节点就 yield。

    Web SSE 接口调用此函数，逐阶段推送进度。
    若提供 on_token 回调，Writer LLM 模式时逐 token 调用。

    Yields:
        {"event": "stage"|"token"|"done"|"error", "data": {...}}
    """
    mode_resolved = resolve_mode(query, mode)
    t_start = time.monotonic()

    # ── chat 模式 ──
    if mode_resolved == "chat":
        if llm_client is not None:
            result = await llm_client.chat(query, system_prompt="你是一个友好、专业的AI助手。")
            text = result.content if result.is_success else query
        else:
            text = "你好！我是 HorizonRL-Agent。试试输入一个研究问题。"
        yield {"event": "stage", "data": {"session_id": "chat", "stage": "chat", "label": "对话模式", "progress": 1.0}}
        yield {"event": "done", "data": {"mode_resolved": "chat", "final_answer_text": text, "runtime_ms": (time.monotonic() - t_start) * 1000}}
        return

    # ── deep 模式: 走 Orchestrator.stream() ──
    traj_logger = TrajectoryLogger(output_dir="trajectories")
    traj_sid = await traj_logger.start_session(query)
    sid = session_id or traj_sid  # 使用 logger 的 sid 确保一致
    try:
        if config is None:
            try:
                config = load_config(
                    Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None
                )
            except Exception:
                config = RootConfig()

        if offline:
            search_provider = "mock"

        if tool_manager is None:
            tool_manager = _build_default_tool_manager(search_provider, offline)

        planner = Planner(config)
        if llm_client is not None:
            from horizonrl.agent.planner import LLMPlanner
            planner = LLMPlanner(llm_client)

        if writer is None:
            writer_mode = "llm" if llm_client is not None else "template"
            writer = Writer(
                mode=writer_mode,
                llm_client=llm_client,
                config=WriterConfig(export_dir=export_dir),
            )

        # Embedding 客户端 — L3 真实向量检索
        embedding_client = None
        if config.embedding.api_key:
            try:
                embedding_client = LLMClient(config.embedding)
            except Exception:
                pass

        orchestrator = ResearchOrchestrator(
            planner=planner,
            tool_manager=tool_manager,
            semaphore_limit=semaphore_limit,
            max_iterations=max_iterations,
            writer=writer,
            embedding_client=embedding_client,
            trajectory_logger=traj_logger,
            on_token=on_token,
        )

        stage_map = {
            "plan_task": ("planning", "正在规划任务", 0.10),
            "mark_ready": ("scheduling", "正在调度任务", 0.20),
            "execute_batch": ("executing", "正在执行子任务", 0.50),
            "verify_batch": ("verifying", "正在验证结果", 0.70),
            "replan": ("replanning", "正在重规划", 0.80),
            "finalize": ("writing", "正在撰写报告", 0.90),
        }

        async for node_name, node_state in orchestrator.stream(query, session_id=sid):
            info = stage_map.get(node_name, (node_name, node_name, 0.5))
            yield {
                "event": "stage",
                "data": {
                    "session_id": sid,
                    "stage": info[0],
                    "label": info[1],
                    "progress": info[2],
                    "node": node_name,
                },
            }
            # execute_batch 完成后推送工具调用详情
            if node_name == "execute_batch":
                results = node_state.get("results", {})
                for task_id, r in results.items():
                    if not isinstance(r, dict):
                        continue
                    for tc in r.get("tool_calls", []):
                        yield {
                            "event": "tool",
                            "data": {
                                "session_id": sid,
                                "task_id": task_id,
                                "tool_name": tc.get("tool_name", ""),
                                "success": tc.get("error", "") == "",
                                "elapsed": tc.get("elapsed", 0),
                                "tokens": tc.get("tokens_used", 0),
                            },
                        }
            # verify_batch 完成后推送验证结果
            elif node_name == "verify_batch":
                verifications = node_state.get("verifications", {})
                for node_id, v in verifications.items():
                    if not isinstance(v, dict):
                        continue
                    yield {
                        "event": "verify",
                        "data": {
                            "session_id": sid,
                            "task_id": node_id,
                            "pass": v.get("pass_", v.get("pass", False)),
                            "score": v.get("score", 0),
                            "error_type": v.get("error_type", ""),
                        },
                    }

        # 收集最终产出
        elapsed = time.monotonic() - t_start
        final_path = Path(export_dir) / sid / "final_answer.md"

        final_text = ""
        if final_path.exists():
            final_text = final_path.read_text(encoding="utf-8")

        await traj_logger.end_session(success=True)
        yield {"event": "report_ready", "data": {"session_id": sid, "final_answer_path": str(final_path), "debug_report_path": str(Path(export_dir) / sid / "debug_report.md")}}
        yield {"event": "done", "data": {"session_id": sid, "mode_resolved": "deep", "final_answer_text": final_text[:500], "runtime_ms": elapsed * 1000}}

    except Exception as exc:
        try:
            await traj_logger.end_session(success=False)
        except Exception:
            pass
        yield {"event": "error", "data": {"session_id": sid, "error": str(exc), "runtime_ms": (time.monotonic() - t_start) * 1000}}


# ─── 内部工具 ─────────────────────────────────────────────────────────────────


def _build_default_tool_manager(search_provider: str = "auto", offline: bool = False):
    """构建默认 ToolManager，自动注册可用工具。"""
    from horizonrl.tools.manager import ToolManager

    mgr = ToolManager()

    if offline or search_provider == "mock":
        from horizonrl.tools.mock import register_mock_tools
        register_mock_tools(mgr)
        return mgr

    # Web Search
    try:
        from horizonrl.tools.web_search import WebSearchTool
        mgr.register("web_search", WebSearchTool(provider=search_provider))
    except Exception:
        from horizonrl.tools.mock import MockWebSearch
        mgr.register("web_search", MockWebSearch())

    # Arxiv Search
    try:
        from horizonrl.tools.paper_search import PaperSearchTool
        mgr.register("paper_search", PaperSearchTool(max_results=5))
    except Exception:
        from horizonrl.tools.mock import MockPaperSearch
        mgr.register("paper_search", MockPaperSearch())

    # Code Execution
    try:
        from horizonrl.tools.code_execution import CodeExecutionTool
        mgr.register("code_execution", CodeExecutionTool(timeout=10.0))
    except Exception:
        from horizonrl.tools.mock import MockCodeExecution
        mgr.register("code_execution", MockCodeExecution())

    return mgr
