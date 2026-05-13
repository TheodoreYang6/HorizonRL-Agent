"""
=======================================================================
05_web_agent.py — HorizonRL-Agent 对话式 Web 界面
=======================================================================

自包含 Web 应用：启动后浏览器打开即可使用。
类似 ChatGPT 的对话体验，背后是完整的 Agent 管道。

运行方式:
    python examples/05_web_agent.py
    浏览器打开 http://localhost:8080

特性:
    - 对话式自然语言输出（像 ChatGPT）
    - 自动检测 LLM，有 API Key 用 LLM 合成，没有则用模板
    - 真实联网搜索 (DDGS + Wikipedia + Arxiv)
    - 后台运行完整 6-Stage Pipeline
    - 过程细节可展开查看
"""

from __future__ import annotations

import asyncio, json, sys, time, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiohttp import web

from horizonrl.config.settings import load_config, RootConfig
from horizonrl.schemas.task import UserTask, TaskStatus
from horizonrl.schemas.event import EventType, TrajectoryEvent
from horizonrl.agent.planner import Planner, LLMPlanner
from horizonrl.agent.worker import AgentWorker
from horizonrl.agent.verifier import Verifier
from horizonrl.agent.replanner import Replanner
from horizonrl.agent.writer import Writer, WriterConfig
from horizonrl.tools.manager import ToolManager
from horizonrl.memory.hierarchical_memory import HierarchicalMemory
from horizonrl.logging.trajectory_logger import TrajectoryLogger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HTML 前端 — 对话式界面                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HorizonRL-Agent — AI 研究助手</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ea; --text2: #9ca3af; --accent: #6c8cff;
    --success: #4ade80; --fail: #f87171;
    --radius: 12px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 800px; margin: 0 auto; padding: 20px; }

  header { text-align: center; padding: 30px 0 20px; }
  header h1 { font-size: 24px; background: linear-gradient(135deg, #6c8cff, #a78bfa);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  header p { color: var(--text2); font-size: 13px; margin-top: 4px; }

  .chat-area { min-height: 60vh; }

  .message { margin-bottom: 20px; animation: fadeIn 0.3s; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

  .msg-user { display: flex; justify-content: flex-end; }
  .msg-user .bubble { background: var(--accent); color: #fff; max-width: 80%; padding: 12px 18px;
    border-radius: var(--radius) var(--radius) 4px var(--radius); font-size: 15px; line-height: 1.5; }

  .msg-agent .bubble { background: var(--surface); border: 1px solid var(--border); max-width: 100%;
    padding: 20px 24px; border-radius: var(--radius); font-size: 14px; line-height: 1.7; }
  .msg-agent .bubble h1 { font-size: 20px; margin: 16px 0 8px; color: var(--accent); }
  .msg-agent .bubble h2 { font-size: 16px; margin: 14px 0 6px; color: #a78bfa; }
  .msg-agent .bubble h3 { font-size: 14px; margin: 10px 0 4px; }
  .msg-agent .bubble p { margin: 6px 0; }
  .msg-agent .bubble ul, .msg-agent .bubble ol { margin: 6px 0 6px 20px; }
  .msg-agent .bubble table { border-collapse: collapse; margin: 8px 0; width: 100%; }
  .msg-agent .bubble th, .msg-agent .bubble td { border: 1px solid var(--border); padding: 6px 10px;
    text-align: left; font-size: 13px; }
  .msg-agent .bubble th { background: var(--border); }
  .msg-agent .bubble blockquote { border-left: 3px solid var(--accent); padding-left: 12px;
    color: var(--text2); margin: 8px 0; }
  .msg-agent .bubble code { background: #00000030; padding: 2px 6px; border-radius: 4px; font-size: 13px; }

  .typing { color: var(--text2); font-size: 13px; padding: 8px 0; display: none; }
  .typing.show { display: block; }
  .typing span { animation: blink 1.4s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

  .process-toggle { font-size: 12px; color: var(--text2); cursor: pointer; margin-top: 8px;
    user-select: none; }
  .process-toggle:hover { color: var(--accent); }
  .process-detail { display: none; margin-top: 12px; padding: 12px; background: #00000020;
    border-radius: 8px; font-size: 12px; color: var(--text2); max-height: 200px; overflow-y: auto; }
  .process-detail.show { display: block; }

  .input-area { position: sticky; bottom: 0; background: var(--bg); padding: 16px 0; border-top: 1px solid var(--border); }
  .input-row { display: flex; gap: 10px; }
  .input-row input { flex: 1; padding: 12px 16px; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--radius); color: var(--text);
    font-size: 14px; outline: none; }
  .input-row input:focus { border-color: var(--accent); }
  .input-row button { padding: 12px 24px; background: var(--accent); color: #fff;
    border: none; border-radius: var(--radius); font-size: 14px; cursor: pointer; font-weight: 600; }
  .input-row button:hover { opacity: 0.9; }
  .input-row button:disabled { opacity: 0.4; cursor: not-allowed; }

  .mode-badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: var(--surface);
    border: 1px solid var(--border); color: var(--text2); display: inline-block; margin-bottom: 12px; }
  .mode-badge.llm { border-color: var(--success); color: var(--success); }

  footer { text-align: center; padding: 20px 0; color: var(--text2); font-size: 12px; }

  .markdown-body { word-wrap: break-word; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>HorizonRL-Agent</h1>
    <p>AI 研究助手 — 多 Agent 协作搜索 + 验证 + 合成</p>
    <span class="mode-badge" id="modeBadge">离线模式</span>
  </header>

  <div class="chat-area" id="chatArea">
    <div class="message msg-agent">
      <div class="bubble">
        你好！我是一个 AI 研究助手。<br><br>
        输入你想研究的问题，我会：
        <br>1. 自动分解任务
        <br>2. 并行搜索网络和学术论文
        <br>3. 验证结果质量
        <br>4. 合成为自然语言研究报告<br><br>
        试试：<b>"Transformer 注意力机制最新进展"</b>
      </div>
    </div>
  </div>

  <div class="typing" id="typing">Agent 正在研究中<span>...</span></div>

  <div class="input-area">
    <div class="input-row">
      <input type="text" id="query" placeholder="输入你的研究问题..."
             value="Transformer 注意力机制最新进展"
             onkeydown="if(event.key==='Enter')startResearch()">
      <button id="btn" onclick="startResearch()">发送</button>
    </div>
  </div>

  <footer>HorizonRL-Agent v0.1.0 · NWPU · 2026</footer>
</div>

<script>
function startResearch() {
  const q = document.getElementById('query').value.trim();
  if (!q) return;

  const btn = document.getElementById('btn');
  btn.disabled = true;
  btn.textContent = '研究中...';

  // 添加用户消息
  addMessage('user', q);

  // 显示输入中
  document.getElementById('typing').classList.add('show');

  // 滚动到底部
  document.getElementById('chatArea').scrollIntoView({behavior: 'smooth'});

  fetch('/api/research', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: q})
  })
  .then(r => r.json())
  .then(data => {
    document.getElementById('typing').classList.remove('show');
    if (data.error) {
      addMessage('agent', '抱歉，研究过程出错：' + data.error);
    } else {
      // 添加 Agent 回答（自然语言报告）
      addMessage('agent', data.report || '研究完成，但未能生成报告。', data.process);
      if (data.llm_mode) {
        document.getElementById('modeBadge').textContent = 'LLM 模式';
        document.getElementById('modeBadge').classList.add('llm');
      }
    }
    btn.disabled = false;
    btn.textContent = '发送';
    document.getElementById('chatArea').scrollIntoView({behavior: 'smooth'});
  })
  .catch(err => {
    document.getElementById('typing').classList.remove('show');
    addMessage('agent', '网络错误：' + err.message);
    btn.disabled = false;
    btn.textContent = '发送';
  });
}

function addMessage(role, content, process) {
  const div = document.createElement('div');
  div.className = 'message msg-' + (role === 'user' ? 'user' : 'agent');

  if (role === 'user') {
    div.innerHTML = '<div class="bubble">' + escapeHtml(content) + '</div>';
  } else {
    // 渲染 Markdown
    let html = renderMarkdown(content);
    let bubble = '<div class="bubble markdown-body">' + html + '</div>';

    // 过程细节（可展开）
    if (process) {
      let procText = '';
      if (process.tasks) {
        procText += '任务执行：\n';
        process.tasks.forEach(t => {
          let icon = t.status === 'success' ? 'OK' : 'FAIL';
          procText += `  [${icon}] ${t.name} — 工具:${t.tools} 评分:${t.score?.toFixed(1)} ${t.evidence}条证据\n`;
        });
      }
      procText += '\n统计：' + (process.success||'?') + '/' + (process.total||'?') + ' 成功, '
               + (process.rounds||'?') + '轮, ' + (process.tool_calls||'?') + '次工具调用, '
               + (process.replans||'?') + '次重规划, ' + (process.elapsed||'?') + 's';

      bubble += '<div class="process-toggle" onclick="this.nextElementSibling.classList.toggle(\'show\')">'
             + '查看执行过程</div>';
      bubble += '<div class="process-detail"><pre>' + escapeHtml(procText) + '</pre></div>';
    }

    div.innerHTML = bubble;
  }

  document.getElementById('chatArea').appendChild(div);
  div.scrollIntoView({behavior: 'smooth'});
}

function escapeHtml(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// 简单 Markdown 渲染
function renderMarkdown(md) {
  if (!md) return '';
  let html = escapeHtml(md);

  // 标题
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // 粗体
  html = html.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');

  // 表格（简化）
  html = html.replace(/\|(.+)\|/g, function(m) {
    if (m.includes('---')) return '';
    let cells = m.split('|').filter(c => c.trim());
    let row = '<tr>' + cells.map(c => '<td>' + c.trim() + '</td>').join('') + '</tr>';
    return row;
  });

  // 引用
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  // 分隔线
  html = html.replace(/^---$/gm, '<hr>');

  // 换行
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = '<p>' + html + '</p>';

  // 清理空标签
  html = html.replace(/<p><\/p>/g, '');
  html = html.replace(/<p><br><\/p>/g, '');

  return html;
}
</script>
</body>
</html>"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Pipeline 引擎                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def run_research(query: str) -> dict:
    """运行完整研究管道，返回自然语言报告 + 过程数据。"""
    t0 = time.time()

    # ── 基础设施 ──
    try:
        cfg = load_config(Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None)
    except Exception:
        cfg = RootConfig()

    # LLM 可用性检测
    llm_client = None
    if cfg.llm.api_key:
        try:
            from horizonrl.llm.client import LLMClient
            llm_client = LLMClient(cfg.llm)
        except Exception:
            pass
    llm_mode = llm_client is not None

    # ── 工具注册 ──
    mgr = ToolManager()
    try:
        from horizonrl.tools.web_search import WebSearchTool
        mgr.register("web_search", WebSearchTool())
    except Exception:
        from horizonrl.tools.mock import MockWebSearch
        mgr.register("web_search", MockWebSearch())
    try:
        from horizonrl.tools.arxiv_search import ArxivSearchTool
        mgr.register("arxiv_search", ArxivSearchTool(max_results=5))
    except Exception:
        from horizonrl.tools.mock import MockArxivSearch
        mgr.register("arxiv_search", MockArxivSearch())
    try:
        from horizonrl.tools.code_execution import CodeExecutionTool
        mgr.register("code_execution", CodeExecutionTool(timeout=10.0))
    except Exception:
        from horizonrl.tools.mock import MockCodeExecution
        mgr.register("code_execution", MockCodeExecution())

    memory = HierarchicalMemory(cfg.memory)

    # ── 规划（简单问题用模板，复杂问题用 LLM）──
    is_complex = any(kw in query for kw in ["对比", "比较", "区别", "vs", "优劣", "分析", "总结", "综述", "展望"])
    use_llm = llm_mode and is_complex

    if use_llm:
        planner = LLMPlanner(llm_client)
        plan = await planner.plan(UserTask(description=query, max_steps=20))
    else:
        planner = Planner()
        plan = planner.plan(UserTask(description=query, max_steps=20))

    # ── DAG 执行 + 验证 + 重规划 ──
    verifier = Verifier(mode="rule")
    replanner = Replanner(max_retries_per_task=3, max_total_replans=5)
    sem = asyncio.Semaphore(3)
    results: dict = {}
    verifications: dict = {}
    task_details = []
    round_num = 0
    total_tool_calls = 0
    total_replans = 0

    while plan.has_pending_work():
        round_num += 1
        for node in plan.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            if all(plan.nodes[d].status == TaskStatus.SUCCESS for d in node.depends_on):
                node.status = TaskStatus.READY

        ready = plan.get_ready_nodes()
        if not ready:
            break

        async def exec_one(node):
            node.status = TaskStatus.RUNNING
            async with sem:
                worker = AgentWorker(worker_id=f"wrk_{node.id}", tool_manager=mgr)
                return node, await worker.execute(node.spec)

        batch = await asyncio.gather(*[exec_one(n) for n in ready])

        for node, result in batch:
            results[result.task_id] = result
            vr = await verifier.verify(result, node.spec)
            verifications[node.id] = vr

            if vr.pass_:
                node.status = TaskStatus.SUCCESS
                memory.record(result, vr)
            else:
                patch = replanner.replan(vr, plan, node.id)
                if patch is not None:
                    replanner.apply_patch(plan, patch)
                    total_replans += 1
                    memory.record_replan()
                else:
                    node.status = TaskStatus.FAILED
                    memory.record(result, vr)

            total_tool_calls += len(result.tool_calls)
            task_details.append({
                "name": node.spec.name,
                "tools": ", ".join(node.spec.tool_names) or "无",
                "status": node.status.value,
                "score": vr.score,
                "evidence": len(result.evidence),
            })

        memory.auto_compress()

    # ── 报告合成 ──
    ctx = memory.get_context()
    writer_mode = "llm" if llm_mode else "template"
    writer = Writer(mode=writer_mode, llm_client=llm_client)
    report_text = await writer.synthesize_async(
        query=query, plan=plan, results=results,
        verifications=verifications, memory_ctx=ctx,
    )

    elapsed = time.time() - t0

    return {
        "report": report_text,
        "llm_mode": llm_mode,
        "process": {
            "tasks": task_details,
            "success": plan.success_count(),
            "total": plan.total_count(),
            "rounds": round_num,
            "tool_calls": total_tool_calls,
            "replans": total_replans,
            "elapsed": f"{elapsed:.1f}s",
        },
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
        return web.json_response({"error": "问题太长"}, status=400)

    result = await run_research(query)
    return web.json_response(result, dumps=lambda o: json.dumps(o, ensure_ascii=False))


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_post("/api/research", handle_api_research)
    app.router.add_get("/favicon.ico", lambda r: web.Response(status=204))
    return app


def main():
    import os
    app = create_app()
    port = int(os.environ.get("PORT", 8080))

    print(f"""
==============================================================
  HorizonRL-Agent Web 界面 — 对话式 AI 研究助手
  http://localhost:{port}
  按 Ctrl+C 停止服务
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
