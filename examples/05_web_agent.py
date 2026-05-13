"""
=======================================================================
05_web_agent.py — HorizonRL-Agent Web 交互界面
=======================================================================

自包含 Web 应用：一个 Python 文件启动服务器，浏览器打开即可使用。
前端用纯 HTML/CSS/JS 构建，后端用 aiohttp 异步处理。

运行方式:
    python examples/05_web_agent.py
    然后打开浏览器访问 http://localhost:8080

功能:
    - 输入研究问题，点击"开始研究"
    - 实时显示 Pipeline 6 个阶段的进度
    - 展示 DAG 任务结构、执行结果、证据列表
    - 支持离线模式（无需 API Key）

依赖:
    aiohttp (pip install aiohttp)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiohttp import web

from horizonrl.config.settings import load_config, RootConfig
from horizonrl.schemas.task import UserTask, TaskStatus
from horizonrl.schemas.result import ErrorType
from horizonrl.schemas.event import EventType, TrajectoryEvent
from horizonrl.agent.planner import Planner
from horizonrl.agent.worker import AgentWorker
from horizonrl.agent.verifier import Verifier
from horizonrl.agent.replanner import Replanner
from horizonrl.agent.writer import Writer
from horizonrl.tools.manager import ToolManager
from horizonrl.memory.hierarchical_memory import HierarchicalMemory
from horizonrl.logging.trajectory_logger import TrajectoryLogger


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HTML 前端                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HorizonRL-Agent — 多 Agent 研究系统</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ea; --text2: #9ca3af; --accent: #6c8cff;
    --success: #4ade80; --fail: #f87171; --warn: #fbbf24;
    --radius: 8px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 960px; margin: 0 auto; padding: 24px; }

  header { text-align: center; padding: 40px 0 24px; }
  header h1 { font-size: 28px; background: linear-gradient(135deg, #6c8cff, #a78bfa);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  header p { color: var(--text2); margin-top: 8px; font-size: 14px; }

  .search-box { display: flex; gap: 12px; margin-bottom: 32px; }
  .search-box input { flex: 1; padding: 14px 18px; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--radius); color: var(--text);
    font-size: 15px; outline: none; }
  .search-box input:focus { border-color: var(--accent); }
  .search-box button { padding: 14px 28px; background: var(--accent); color: #fff;
    border: none; border-radius: var(--radius); font-size: 15px; cursor: pointer;
    font-weight: 600; white-space: nowrap; }
  .search-box button:hover { opacity: 0.9; }
  .search-box button:disabled { opacity: 0.4; cursor: not-allowed; }

  .status-bar { display: flex; gap: 8px; margin-bottom: 24px; flex-wrap: wrap; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--border);
                 animation: pulse 1.5s infinite; display: none; }
  .status-dot.active { display: block; background: var(--accent); }
  .status-dot.done { display: block; background: var(--success); animation: none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .stage-badge { padding: 4px 12px; border-radius: 20px; font-size: 12px;
                 background: var(--surface); color: var(--text2); border: 1px solid var(--border); }
  .stage-badge.active { border-color: var(--accent); color: var(--accent); }
  .stage-badge.done { border-color: var(--success); color: var(--success); }

  .results { display: none; }
  .results.show { display: block; }

  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: var(--radius); padding: 20px; margin-bottom: 16px; }
  .card h3 { font-size: 16px; margin-bottom: 12px; color: var(--accent); }

  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr));
                  gap: 12px; }
  .summary-item { text-align: center; padding: 12px; background: var(--bg);
                  border-radius: var(--radius); }
  .summary-item .num { font-size: 28px; font-weight: 700; }
  .summary-item .num.ok { color: var(--success); }
  .summary-item .num.warn { color: var(--warn); }
  .summary-item .label { font-size: 12px; color: var(--text2); margin-top: 4px; }

  .task-row { display: flex; align-items: center; gap: 12px; padding: 10px 0;
              border-bottom: 1px solid var(--border); }
  .task-row:last-child { border-bottom: none; }
  .task-icon { width: 32px; height: 32px; border-radius: 50%; display: flex;
               align-items: center; justify-content: center; font-size: 14px; }
  .task-icon.ok { background: #1a3a2a; color: var(--success); }
  .task-icon.fail { background: #3a1a1a; color: var(--fail); }
  .task-info { flex: 1; }
  .task-name { font-weight: 600; font-size: 14px; }
  .task-meta { font-size: 12px; color: var(--text2); margin-top: 2px; }
  .task-score { font-size: 13px; font-weight: 600; }

  .evidence-list { max-height: 300px; overflow-y: auto; }
  .evidence-item { padding: 8px 12px; margin-bottom: 6px; background: var(--bg);
                   border-radius: 4px; font-size: 13px; border-left: 3px solid var(--accent); }
  .evidence-item .src { font-size: 11px; color: var(--text2);
                         background: var(--surface); padding: 2px 6px; border-radius: 3px;
                         margin-right: 8px; }

  .report-text { white-space: pre-wrap; font-size: 14px; line-height: 1.6;
                 max-height: 600px; overflow-y: auto; }

  .empty-state { text-align: center; padding: 60px 20px; color: var(--text2); }
  .empty-state .icon { font-size: 48px; margin-bottom: 16px; }
  footer { text-align: center; padding: 40px 0; color: var(--text2); font-size: 13px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>HorizonRL-Agent</h1>
    <p>多 Agent 长链路研究系统 — 输入一个问题，AI Agent 自动分解 → 搜索 → 验证 → 总结</p>
  </header>

  <div class="search-box">
    <input type="text" id="query" placeholder="输入你想研究的问题..."
           value="Transformer 多头注意力机制的最新进展">
    <button id="btn" onclick="startResearch()">开始研究</button>
  </div>

  <div id="stages" class="status-bar"></div>

  <div id="results" class="results">
    <div class="card">
      <h3>执行概要</h3>
      <div id="summary" class="summary-grid"></div>
    </div>
    <div class="card">
      <h3>DAG 任务执行</h3>
      <div id="tasks"></div>
    </div>
    <div class="card">
      <h3>收集证据</h3>
      <div id="evidence" class="evidence-list"></div>
    </div>
    <div class="card">
      <h3>研究报告</h3>
      <div id="report" class="report-text"></div>
    </div>
  </div>

  <div id="empty" class="empty-state">
    <div class="icon">🤖</div>
    <p>输入研究问题，让 AI Agent 为你工作</p>
    <p style="font-size:12px;margin-top:8px">
       支持中文/英文问题 · Planner 拆解 → Worker 搜索 → Verifier 验证 → Replanner 修复 · 全程记录</p>
  </div>

  <footer>HorizonRL-Agent v0.1.0 · 杨启铎 · NWPU · 2026</footer>
</div>

<script>
const STAGES = ['加载基础设施', '任务规划', '并发执行', '质量验证', '记忆总结', '生成报告'];

function startResearch() {
  const q = document.getElementById('query').value.trim();
  if (!q) return;

  document.getElementById('btn').disabled = true;
  document.getElementById('btn').textContent = '研究中...';
  document.getElementById('empty').style.display = 'none';
  document.getElementById('results').classList.remove('show');

  // Show stages
  const stagesDiv = document.getElementById('stages');
  stagesDiv.innerHTML = STAGES.map((s,i) =>
    `<span class="stage-badge" id="stage-${i}">${s}</span>`
  ).join('');

  fetch('/api/research', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: q})
  })
  .then(r => r.json())
  .then(data => {
    renderResults(data);
    document.getElementById('btn').disabled = false;
    document.getElementById('btn').textContent = '开始研究';
  })
  .catch(err => {
    alert('研究失败: ' + err.message);
    document.getElementById('btn').disabled = false;
    document.getElementById('btn').textContent = '开始研究';
  });
}

function markStage(i, done) {
  const el = document.getElementById('stage-' + i);
  if (!el) return;
  el.classList.add('active');
  if (done) { el.classList.remove('active'); el.classList.add('done'); }
}

function renderResults(data) {
  // Mark all stages done
  STAGES.forEach((_, i) => markStage(i, true));

  const results = document.getElementById('results');
  results.classList.add('show');

  // Summary
  const stats = data.stats || {};
  const scoreColor = stats.success_rate >= 0.8 ? 'ok' : stats.success_rate >= 0.4 ? 'warn' : '';
  document.getElementById('summary').innerHTML = `
    <div class="summary-item"><div class="num ok">${stats.success_count||0}/${stats.total_count||0}</div><div class="label">任务完成</div></div>
    <div class="summary-item"><div class="num">${stats.rounds||0}</div><div class="label">执行轮次</div></div>
    <div class="summary-item"><div class="num">${stats.tool_calls||0}</div><div class="label">工具调用</div></div>
    <div class="summary-item"><div class="num">${stats.replans||0}</div><div class="label">重规划</div></div>
    <div class="summary-item"><div class="num">${data.total_evidence||0}</div><div class="label">收集证据</div></div>
    <div class="summary-item"><div class="num">${data.plan_count||0}</div><div class="label">子任务数</div></div>
  `;

  // Tasks
  const tasks = data.tasks || [];
  document.getElementById('tasks').innerHTML = tasks.map(t => `
    <div class="task-row">
      <div class="task-icon ${t.status==='success'?'ok':'fail'}">${t.status==='success'?'✓':'✗'}</div>
      <div class="task-info">
        <div class="task-name">${t.name}</div>
        <div class="task-meta">工具: ${t.tools||'无'} | 依赖: ${t.deps||'无'} | ${t.evidence||0}条证据 | ${t.elapsed||0}s</div>
      </div>
      <div class="task-score" style="color:${t.score>=0.7?'var(--success)':t.score>=0.3?'var(--warn)':'var(--fail)'}">
        评分 ${t.score?.toFixed(1)||'?'}
      </div>
    </div>
  `).join('');

  // Evidence
  const evidence = data.evidence || [];
  document.getElementById('evidence').innerHTML = evidence.length > 0
    ? evidence.slice(0, 10).map(e =>
        `<div class="evidence-item"><span class="src">[${e.type}]</span>${e.content}</div>`
      ).join('')
    : '<div style="color:var(--text2)">(无证据)</div>';

  // Report
  document.getElementById('report').textContent = data.report || '(无报告)';
}
</script>
</body>
</html>"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Pipeline 引擎                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def run_research(query: str) -> dict:
    """运行完整研究管道，返回 JSON 可序列化的结果。"""
    t0 = time.time()

    # ── 基础设施 ──
    try:
        cfg = load_config(Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None)
    except Exception:
        cfg = RootConfig()

    mgr = ToolManager()
    # Web Search（真实优先）
    try:
        from horizonrl.tools.web_search import WebSearchTool
        mgr.register("web_search", WebSearchTool())
    except Exception:
        from horizonrl.tools.mock import MockWebSearch
        mgr.register("web_search", MockWebSearch())
    # Arxiv Search（真实优先）
    try:
        from horizonrl.tools.arxiv_search import ArxivSearchTool
        mgr.register("arxiv_search", ArxivSearchTool(max_results=5))
    except Exception:
        from horizonrl.tools.mock import MockArxivSearch
        mgr.register("arxiv_search", MockArxivSearch())
    # Code Execution（始终可用）
    try:
        from horizonrl.tools.code_execution import CodeExecutionTool
        mgr.register("code_execution", CodeExecutionTool(timeout=10.0))
    except Exception:
        from horizonrl.tools.mock import MockCodeExecution
        mgr.register("code_execution", MockCodeExecution())
    memory = HierarchicalMemory(cfg.memory)
    logger = TrajectoryLogger(output_dir="trajectories")
    await logger.start_session(query)

    # ── 规划 ──
    planner = Planner()
    plan = planner.plan(UserTask(description=query, max_steps=20))

    await logger.log(TrajectoryEvent(
        module="planner", event_type=EventType.PLAN_COMPLETE,
        payload={"num_subtasks": plan.total_count(), "root_ids": plan.root_ids},
    ))

    # ── DAG 执行 + 验证 + 重规划 ──
    verifier = Verifier(mode="rule")
    replanner = Replanner(max_retries_per_task=3, max_total_replans=5)
    sem = asyncio.Semaphore(3)
    results: dict = {}
    verifications: dict = {}
    task_details: list[dict] = []
    all_evidence: list[dict] = []
    round_num = 0
    total_tool_calls = 0
    total_replans = 0

    while plan.has_pending_work():
        round_num += 1

        for node in plan.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            deps_ok = all(
                plan.nodes[d].status == TaskStatus.SUCCESS
                for d in node.depends_on
            )
            if deps_ok:
                node.status = TaskStatus.READY

        ready = plan.get_ready_nodes()
        if not ready:
            pending = [n for n in plan.nodes.values()
                       if n.status in (TaskStatus.PENDING, TaskStatus.READY)]
            if pending:
                break
            break

        async def exec_one(node):
            node.status = TaskStatus.RUNNING
            node.started_at = time.time()
            async with sem:
                worker = AgentWorker(worker_id=f"wrk_{node.id}", tool_manager=mgr)
                result = await worker.execute(node.spec)
            node.finished_at = time.time()

            await logger.log(TrajectoryEvent(
                module="worker",
                event_type=EventType.WORKER_COMPLETE if result.success else EventType.WORKER_ERROR,
                payload={"task_id": node.id, "success": result.success},
                cost=result.tokens_used, latency=result.elapsed,
            ))
            return node, result

        batch = await asyncio.gather(*[exec_one(n) for n in ready])

        for node, result in batch:
            results[result.task_id] = result
            vr = await verifier.verify(result, node.spec)
            verifications[node.id] = vr

            await logger.log(TrajectoryEvent(
                module="verifier",
                event_type=EventType.VERIFY_COMPLETE if vr.pass_ else EventType.VERIFY_FAIL,
                payload={"task_id": node.id, "pass": vr.pass_, "score": vr.score},
            ))

            if vr.pass_:
                node.status = TaskStatus.SUCCESS
                memory.record(result, vr)
            else:
                patch = replanner.replan(vr, plan, node.id)
                if patch is not None:
                    replanner.apply_patch(plan, patch)
                    total_replans += 1
                    memory.record_replan()
                    await logger.log(TrajectoryEvent(
                        module="replanner", event_type=EventType.REPLAN_PATCH,
                        payload={"target_node": node.id, "patch_type": patch.patch_type.value},
                    ))
                else:
                    node.status = TaskStatus.FAILED
                    memory.record(result, vr)

            total_tool_calls += len(result.tool_calls)

            deps_list = node.depends_on if node.depends_on else []
            deps_names = [plan.nodes[d].spec.name if d in plan.nodes else d for d in deps_list]

            task_details.append({
                "id": node.id,
                "name": node.spec.name,
                "tools": ", ".join(node.spec.tool_names) or "无",
                "deps": ", ".join(deps_names) or "无",
                "status": node.status.value,
                "score": vr.score,
                "evidence": len(result.evidence),
                "elapsed": f"{result.elapsed:.1f}",
                "output": result.output[:200],
            })

            for ev in result.evidence:
                all_evidence.append({
                    "type": ev.source_type or "unknown",
                    "content": ev.content[:250],
                })

        memory.auto_compress()

    # ── 记忆压缩 ──
    if memory.l1.count > 0:
        memory.compress(query)

    # ── 报告 ──
    ctx = memory.get_context()
    writer = Writer(mode="template")
    report_text = writer.synthesize(
        query=query,
        plan=plan,
        results=results,
        verifications=verifications,
        memory_ctx=ctx,
    )

    await logger.end_session(success=(plan.success_count() == plan.total_count()))

    return {
        "query": query,
        "plan_count": plan.total_count(),
        "rounds": round_num,
        "stats": {
            "success_count": plan.success_count(),
            "total_count": plan.total_count(),
            "rounds": round_num,
            "tool_calls": total_tool_calls,
            "replans": total_replans,
            "success_rate": plan.success_count() / max(plan.total_count(), 1),
        },
        "tasks": task_details,
        "evidence": all_evidence,
        "total_evidence": len(all_evidence),
        "report": report_text,
        "elapsed": f"{time.time() - t0:.1f}s",
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HTTP 路由                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=HTML_PAGE, content_type="text/html", charset="utf-8")


async def handle_api_research(request: web.Request) -> web.Response:
    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        return web.json_response({"error": "请提供研究问题"}, status=400)
    if len(query) > 500:
        return web.json_response({"error": "问题太长，请控制在500字以内"}, status=400)

    result = await run_research(query)
    return web.json_response(result, dumps=lambda o: json.dumps(o, ensure_ascii=False))


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_post("/api/research", handle_api_research)
    # 静态 favicon 避免 404
    app.router.add_get("/favicon.ico", lambda r: web.Response(status=204))
    return app


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  入口                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def main():
    import os

    app = create_app()
    port = int(os.environ.get("PORT", 8080))

    print(f"""
==============================================================
  HorizonRL-Agent Web 界面
  http://localhost:{port}
  在浏览器中打开上面的地址，输入研究问题即可体验。
  按 Ctrl+C 停止服务。
  (如果端口被占用，执行: taskkill /F /IM python.exe)
==============================================================
""")
    try:
        web.run_app(app, host="127.0.0.1", port=port, print=lambda *a: None)
    except OSError as e:
        print(f"\n端口 {port} 被占用: {e}")
        print("请先关闭之前的进程: taskkill /F /IM python.exe")
        sys.exit(1)


if __name__ == "__main__":
    main()
