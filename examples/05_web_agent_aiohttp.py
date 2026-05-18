"""
=======================================================================
05_web_agent.py — HorizonRL-Agent Web 界面 (v3: SSE + 实时进度)
=======================================================================

自包含 Web 应用: aiohttp 后端 + SSE 实时推送 + 原生 HTML/JS 前端。

路由:
    POST /api/chat              — 对话入口 (chat/auto/deep 三模式)
    GET  /api/stream/{sid}      — SSE 实时进度推送 (替代轮询)
    GET  /api/report/{sid}      — 报告状态查询 (页面刷新恢复)
    GET  /api/download/{sid}/{kind} — 下载 final/debug markdown

v3 变更:
    - SSE 端点替代 500ms 轮询 → 实时推送 stage/tool/report/done 事件
    - 全新 UI: 阶段时间线 + 工具调用面板 + 进度条 + markdown 预览
    - 更好的错误处理 + 心跳保活

运行:
    python examples/05_web_agent.py
    http://localhost:8080
"""

from __future__ import annotations

import asyncio, json, sys, time, uuid, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiohttp import web

from horizonrl.config.settings import load_config, RootConfig
from horizonrl.services.research_service import (
    stream_research_session,
    resolve_mode,
)
from horizonrl.llm.client import LLMClient

# ─── 全局会话状态 ────────────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}


# ─── 后台执行 + SSE 推送 ─────────────────────────────────────────────────────

async def run_deep_pipeline(sid: str, query: str, stream: web.StreamResponse):
    """执行 deep 管道并通过 SSE 推送实时进度。

    完成后更新 _sessions[sid] 供页面刷新恢复。
    """
    _sessions[sid] = {"status": "running", "query": query, "events": []}

    try:
        cfg = load_config(
            Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None
        )
    except Exception:
        cfg = RootConfig()

    llm_client = None
    if cfg.llm.api_key:
        try:
            llm_client = LLMClient(cfg.llm)
        except Exception:
            pass

    # Token 流式回调 → SSE token 事件 (流关闭时静默跳过)
    async def _push_token(token: str):
        await _sse_write(stream, "token", {"delta": token})

    try:
        last_heartbeat = time.monotonic()
        async for event in stream_research_session(
            query=query, mode="deep", llm_client=llm_client, config=cfg,
            export_dir="reports", on_token=_push_token,
        ):
            # Heartbeat 每 15s 防超时断连
            now = time.monotonic()
            if now - last_heartbeat > 15:
                if not await _sse_write(stream, "heartbeat", {"ts": now}):
                    break  # 流已关闭，停止
                last_heartbeat = now

            evt_type = event["event"]
            data = event["data"]
            _sessions[sid]["events"].append({"type": evt_type, "data": data})

            if evt_type == "stage":
                _sessions[sid]["phase"] = data.get("stage", "")
                _sessions[sid]["label"] = data.get("label", "")
            elif evt_type == "report_ready":
                _sessions[sid]["final_path"] = data.get("final_answer_path", "")
                _sessions[sid]["debug_path"] = data.get("debug_report_path", "")
            elif evt_type == "done":
                _sessions[sid].update({
                    "status": "completed",
                    "final_answer": data.get("final_answer_text", ""),
                    "runtime_ms": data.get("runtime_ms", 0),
                })
            elif evt_type == "error":
                _sessions[sid]["status"] = "failed"
                _sessions[sid]["error"] = data.get("error", "")

            if not await _sse_write(stream, evt_type, data):
                break  # 流已关闭，停止

        # 确保最终状态
        if _sessions[sid].get("status") == "running":
            final_text = ""
            final_path = _sessions[sid].get("final_path", "")
            if final_path and Path(final_path).exists():
                final_text = Path(final_path).read_text(encoding="utf-8")[:500]
            _sessions[sid]["status"] = "completed"
            _sessions[sid]["final_answer"] = final_text

    except Exception as exc:
        _sessions[sid]["status"] = "failed"
        _sessions[sid]["error"] = str(exc)
        # 流可能已关闭，忽略写入失败
        await _sse_write(stream, "sse_error", {"error": str(exc)})

    return _sessions[sid]


async def _sse_write(stream: web.StreamResponse, event: str, data: dict) -> bool:
    """写入一条 SSE 事件。流已关闭时返回 False 不抛异常。"""
    try:
        payload = json.dumps(data, ensure_ascii=False)
        await stream.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        return True
    except (RuntimeError, ConnectionResetError, ConnectionAbortedError):
        return False


# ─── LLM 对话 ────────────────────────────────────────────────────────────────

async def run_chat(query: str) -> str:
    try:
        cfg = load_config(
            Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None
        )
    except Exception:
        cfg = RootConfig()

    if not cfg.llm.api_key:
        return (
            "你好！我是 HorizonRL-Agent。\n\n"
            "我可以帮你做深度研究：搜索资料、对比分析、汇总报告。\n"
            "试试输入一个研究问题，比如「Transformer注意力机制的最新进展」。"
        )

    try:
        client = LLMClient(cfg.llm)
        result = await client.chat(
            query,
            system_prompt="你是一个友好、专业的AI助手。用简洁流畅的中文回答。",
            max_tokens=1000,
        )
        return result.content if result.is_success else f"LLM 调用失败: {result.error}"
    except Exception as e:
        return f"LLM 错误: {e}"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HTTP 路由                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=HTML_PAGE, content_type="text/html", charset="utf-8")


async def handle_api_chat(request: web.Request) -> web.Response:
    """POST /api/chat — 统一对话入口。"""
    body = await request.json()
    message = body.get("message", "").strip()
    mode = body.get("mode", "auto")

    if not message or len(message) > 500:
        return web.json_response({"error": "无效问题"}, status=400)

    resolved = resolve_mode(message, mode)

    if resolved == "chat":
        answer = await run_chat(message)
        return web.json_response(
            {"mode": "chat", "answer": answer},
            dumps=lambda o: json.dumps(o, ensure_ascii=False),
        )

    # deep 模式: 返回 session_id, 前端通过 SSE 订阅进度
    sid = f"session_{uuid.uuid4().hex[:12]}"
    _sessions[sid] = {"status": "queued", "query": message, "events": []}
    return web.json_response(
        {"mode": "agent", "session_id": sid, "status": "queued"},
        dumps=lambda o: json.dumps(o, ensure_ascii=False),
    )


async def handle_api_stream(request: web.Request) -> web.StreamResponse:
    """GET /api/stream/{sid} — SSE 实时进度推送。

    事件类型: stage | token | report_ready | done | sse_error | heartbeat
    """
    sid = request.match_info["session_id"]
    session = _sessions.get(sid)

    if not session:
        return web.json_response({"error": "session not found"}, status=404)

    query = session["query"]

    # SSE 响应
    stream = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await stream.prepare(request)

    # 直接内联执行管道，不通过后台 task
    await run_deep_pipeline(sid, query, stream)
    return stream


async def handle_api_report(request: web.Request) -> web.Response:
    """GET /api/report/{sid} — 查询会话状态 (页面刷新恢复)。"""
    sid = request.match_info["session_id"]
    session = _sessions.get(sid)
    if not session:
        return web.json_response({"error": "session not found"}, status=404)

    resp = {
        "status": session.get("status", ""),
        "phase": session.get("phase", ""),
        "label": session.get("label", ""),
        "events": session.get("events", []),
    }
    if session.get("status") == "completed":
        resp["final_answer"] = session.get("final_answer", "")
        resp["download_url_final"] = f"/api/download/{sid}/final"
        resp["download_url_debug"] = f"/api/download/{sid}/debug"
        resp["runtime_ms"] = session.get("runtime_ms", 0)
        resp["final_path"] = session.get("final_path", "")
        resp["debug_path"] = session.get("debug_path", "")
    elif session.get("status") == "failed":
        resp["error"] = session.get("error", "")

    return web.json_response(resp, dumps=lambda o: json.dumps(o, ensure_ascii=False))


async def handle_api_download(request: web.Request) -> web.Response:
    """GET /api/download/{sid}/{kind} — 下载 Markdown 文件。"""
    sid = request.match_info["session_id"]
    kind = request.match_info["kind"]
    session = _sessions.get(sid)

    if not session or session.get("status") != "completed":
        return web.json_response({"error": "not found"}, status=404)

    path_key = f"{kind}_path"
    filepath = session.get(path_key)
    if not filepath or not Path(filepath).exists():
        return web.json_response({"error": "file not found"}, status=404)

    filename = f"{kind}_answer.md"
    return web.FileResponse(filepath, headers={
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "text/markdown; charset=utf-8",
    })


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_post("/api/chat", handle_api_chat)
    app.router.add_get("/api/stream/{session_id}", handle_api_stream)
    app.router.add_get("/api/report/{session_id}", handle_api_report)
    app.router.add_get("/api/download/{session_id}/{kind}", handle_api_download)
    app.router.add_get("/favicon.ico", lambda r: web.Response(status=204))
    return app


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HTML 前端 (v3: SSE + 实时进度 + 工具面板)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HorizonRL-Agent</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff;
  --success: #3fb950; --fail: #f85149; --warn: #d2991d;
  --radius: 8px; --font: -apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--font);background:var(--bg);color:var(--text);height:100vh;display:flex}
/* Sidebar */
.sidebar{width:320px;background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow-y:auto;flex-shrink:0}
.sidebar-header{padding:16px;border-bottom:1px solid var(--border);
  font-size:13px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px}
.stage-list{padding:8px}
.stage-item{display:flex;align-items:flex-start;gap:10px;padding:10px 12px;
  border-radius:var(--radius);margin-bottom:2px;transition:background .2s}
.stage-item.active{background:rgba(88,166,255,.08)}
.stage-dot{width:8px;height:8px;border-radius:50%;margin-top:4px;flex-shrink:0;
  background:var(--border);transition:background .3s}
.stage-dot.done{background:var(--success)}
.stage-dot.running{background:var(--accent);animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.stage-info{flex:1;min-width:0}
.stage-name{font-size:13px;font-weight:500}
.stage-label{font-size:11px;color:var(--text2);margin-top:2px}
.stage-progress{height:3px;background:var(--border);border-radius:2px;margin-top:6px}
.stage-progress-bar{height:100%;background:var(--accent);border-radius:2px;
  transition:width .5s ease;width:0%}
/* Tool Log */
.tool-log{padding:8px;border-top:1px solid var(--border);flex:1;overflow-y:auto}
.tool-log .section-title{font-size:11px;color:var(--text2);padding:8px 12px 4px;
  text-transform:uppercase;letter-spacing:.5px}
.tool-entry{display:flex;align-items:center;gap:8px;font-size:12px;padding:5px 12px;
  border-left:2px solid var(--border);margin:2px 0 2px 8px;color:var(--text2)}
.tool-entry .t-icon{width:16px;text-align:center;flex-shrink:0;font-size:11px}
.tool-entry .t-name{color:var(--text);font-weight:500;flex-shrink:0}
.tool-entry .t-meta{font-size:11px;color:var(--text2)}
.tool-entry.success{border-left-color:var(--success)}
.tool-entry.success .t-icon{color:var(--success)}
.tool-entry.fail{border-left-color:var(--fail)}
.tool-entry.fail .t-icon{color:var(--fail)}
/* Main Area */
.main{flex:1;display:flex;flex-direction:column;min-width:0}
.chat-header{padding:16px 24px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between}
.chat-header h1{font-size:18px;background:linear-gradient(135deg,var(--accent),#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.chat-header .badge{font-size:11px;padding:3px 8px;border-radius:12px;font-weight:500}
.badge-idle{background:var(--border);color:var(--text2)}
.badge-running{background:rgba(88,166,255,.15);color:var(--accent)}
.badge-done{background:rgba(63,185,80,.15);color:var(--success)}
.badge-error{background:rgba(248,81,73,.15);color:var(--fail)}
.chat-area{flex:1;overflow-y:auto;padding:20px 24px}
.message{margin-bottom:20px;animation:fadeIn .3s}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg-user{display:flex;justify-content:flex-end}
.msg-user .bubble{background:var(--accent);color:#fff;max-width:75%;
  padding:10px 16px;border-radius:var(--radius) var(--radius) 4px var(--radius);
  font-size:14px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}
.msg-agent .bubble{background:var(--surface);border:1px solid var(--border);
  padding:16px 20px;border-radius:var(--radius);font-size:14px;line-height:1.7;
  max-width:100%;word-wrap:break-word}
.msg-agent .bubble h1,.msg-agent .bubble h2{font-size:17px;margin:12px 0 6px;color:var(--accent)}
.msg-agent .bubble h3{font-size:15px;margin:10px 0 4px;color:#a78bfa}
.msg-agent .bubble ul,.msg-agent .bubble ol{padding-left:20px;margin:6px 0}
.msg-agent .bubble li{margin:3px 0}
.msg-agent .bubble code{background:rgba(110,118,129,.2);padding:2px 6px;border-radius:4px;font-size:12px}
.msg-agent .bubble blockquote{border-left:3px solid var(--accent);padding:4px 12px;
  color:var(--text2);margin:8px 0}
.msg-agent .bubble strong{color:var(--text)}
.msg-agent .bubble em{color:var(--text2)}
/* Markdown separator */
.msg-agent .bubble hr{border:none;border-top:1px solid var(--border);margin:12px 0}
.msg-agent .bubble table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
.msg-agent .bubble th,.msg-agent .bubble td{border:1px solid var(--border);padding:6px 10px;text-align:left}
.msg-agent .bubble th{background:rgba(88,166,255,.08);font-weight:600}
.msg-agent .bubble pre{background:rgba(22,27,34,.8);padding:12px 16px;border-radius:var(--radius);
  overflow-x:auto;font-size:12px;line-height:1.5;margin:8px 0}
.msg-agent .bubble pre code{background:none;padding:0}
.msg-agent .bubble a{color:var(--accent);text-decoration:none}
.msg-agent .bubble a:hover{text-decoration:underline}
/* Status bar inline */
.status-row{display:flex;align-items:center;gap:8px;margin-bottom:12px;font-size:13px;color:var(--text2)}
.status-row .spinner{width:14px;height:14px;border:2px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.download-row{margin-top:16px;display:flex;gap:10px;flex-wrap:wrap}
.dl-btn{display:flex;flex-direction:column;padding:12px 16px;
  background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  text-decoration:none;cursor:pointer;transition:all .2s;min-width:180px}
.dl-btn:hover{border-color:var(--accent);box-shadow:0 0 0 1px rgba(88,166,255,.12)}
.dl-btn .dl-label{font-size:13px;font-weight:500;color:var(--text)}
.dl-btn .dl-hint{font-size:11px;color:var(--text2);margin-top:2px}
.dl-btn.dl-final:hover{border-color:var(--accent);background:rgba(88,166,255,.04)}
.dl-btn.dl-debug:hover{border-color:#a78bfa;background:rgba(167,139,250,.04)}
/* Input */
.input-area{background:var(--bg);border-top:1px solid var(--border);padding:16px 24px}
.input-row{display:flex;gap:8px}
.input-row input{flex:1;padding:10px 16px;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);color:var(--text);font-size:14px;outline:none;transition:border-color .2s}
.input-row input:focus{border-color:var(--accent)}
.input-row select{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);color:var(--text);font-size:12px;padding:8px 12px;outline:none;cursor:pointer}
.input-row button{padding:10px 24px;background:var(--accent);color:#fff;border:none;
  border-radius:var(--radius);font-size:14px;font-weight:600;cursor:pointer;transition:opacity .2s}
.input-row button:disabled{opacity:.4;cursor:not-allowed}
footer{text-align:center;padding:8px;color:var(--text2);font-size:11px}
/* Mobile */
@media(max-width:768px){
  body{flex-direction:column}
  .sidebar{width:100%;max-height:200px;border-right:none;border-bottom:1px solid var(--border)}
  .chat-area{padding:12px 16px}
  .input-area{padding:12px 16px}
}
</style>
</head>
<body>
<!-- Sidebar -->
<div class="sidebar" id="sidebar">
  <div class="sidebar-header">研究进度</div>
  <div class="stage-list" id="stageList">
    <div class="stage-item" id="stage-plan"><div class="stage-dot"></div>
      <div class="stage-info"><div class="stage-name">任务规划</div>
      <div class="stage-label">等待开始</div></div></div>
    <div class="stage-item" id="stage-exec"><div class="stage-dot"></div>
      <div class="stage-info"><div class="stage-name">执行子任务</div>
      <div class="stage-label">等待中</div>
      <div class="stage-progress"><div class="stage-progress-bar" id="execProgress"></div></div></div></div>
    <div class="stage-item" id="stage-verify"><div class="stage-dot"></div>
      <div class="stage-info"><div class="stage-name">验证结果</div>
      <div class="stage-label">等待中</div></div></div>
    <div class="stage-item" id="stage-write"><div class="stage-dot"></div>
      <div class="stage-info"><div class="stage-name">撰写报告</div>
      <div class="stage-label">等待中</div></div></div>
  </div>
  <div class="tool-log" id="toolLog">
    <div class="section-title">工具调用日志</div>
  </div>
</div>

<!-- Main -->
<div class="main">
  <div class="chat-header">
    <h1>HorizonRL-Agent</h1>
    <span class="badge badge-idle" id="statusBadge">就绪</span>
  </div>
  <div class="chat-area" id="chatArea">
    <div class="message msg-agent"><div class="bubble">
👋 你好！我是 <b>HorizonRL-Agent</b>，一个 AI 研究助手。<br><br>
<b>对话模式</b>：直接问我任何问题<br>
<b>深度研究</b>：输入研究类问题，我会自动搜索网络和论文，验证后写报告<br><br>
试试问：<b>"Transformer注意力机制最新进展"</b> 或 <b>"对比PyTorch和TensorFlow"</b>
    </div></div>
  </div>
  <div class="input-area">
    <div class="input-row">
      <input id="query" placeholder="输入你想研究的问题..." onkeydown="if(event.key==='Enter')send()">
      <select id="modeSel">
        <option value="auto">自动</option>
        <option value="chat">对话</option>
        <option value="deep">深度研究</option>
      </select>
      <button id="btn" onclick="send()">发送</button>
    </div>
  </div>
  <footer>HorizonRL-Agent v0.3.0 · NWPU · 2026</footer>
</div>

<script>
// ── State ──
let isRunning = false, currentSid = null, es = null;

// ── Stage mapping ──
const STAGES = {
  planning:    {el:'stage-plan',    name:'任务规划',  icon:'📋'},
  scheduling:  {el:'stage-plan',    name:'任务调度',  icon:'📋'},
  executing:   {el:'stage-exec',    name:'执行子任务',icon:'🔧'},
  verifying:   {el:'stage-verify',  name:'验证结果',  icon:'✅'},
  replanning:  {el:'stage-exec',    name:'局部重规划',icon:'🔄'},
  writing:     {el:'stage-write',   name:'撰写报告',  icon:'📝'},
};

function send(){
  if(isRunning) return;
  // 清理上一次连接
  if(es){ es.close(); es = null; }
  window._tokenBuf = null;
  const preview = document.getElementById('tokenPreview');
  if(preview && preview.parentNode) preview.parentNode.removeChild(preview);

  const inp = document.getElementById('query');
  const q = inp.value.trim();
  if(!q) return;
  inp.value = '';
  const mode = document.getElementById('modeSel').value;
  const btn = document.getElementById('btn');
  btn.disabled = true; btn.textContent = '...';
  addMessage('user', q);

  fetch('/api/chat',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:q, mode})
  })
  .then(r => r.json())
  .then(data => {
    if(data.mode === 'chat'){
      addMessage('agent', data.answer || '(空)');
      btn.disabled = false; btn.textContent = '发送';
    } else if(data.mode === 'agent'){
      // 启动 SSE 连接
      currentSid = data.session_id;
      connectSSE(currentSid);
    }
  })
  .catch(e => {
    addMessage('agent', '❌ 请求失败: ' + e.message);
    btn.disabled = false; btn.textContent = '发送';
  });
}

function connectSSE(sid){
  isRunning = true;
  window._tokenBuf = null;
  resetSidebar();
  setStatus('running', '研究中...');
  addMessage('agent', '🔍 正在启动深度研究管道...\n\n> 规划 → 搜索 → 验证 → 写作');

  // 创建状态行
  const statusDiv = document.createElement('div');
  statusDiv.className = 'status-row';
  statusDiv.innerHTML = '<div class="spinner"></div><span id="liveStatus">正在规划...</span>';
  const chatArea = document.getElementById('chatArea');
  const wrapper = document.createElement('div');
  wrapper.className = 'message msg-agent';
  wrapper.appendChild(statusDiv);
  chatArea.appendChild(wrapper);
  wrapper.scrollIntoView({behavior:'smooth'});

  // 清理函数
  function cleanup(){
    isRunning = false;
    document.getElementById('btn').disabled = false;
    document.getElementById('btn').textContent = '发送';
    if(wrapper && wrapper.parentNode) wrapper.parentNode.removeChild(wrapper);
    if(es){ es.close(); es = null; }
  }

  es = new EventSource('/api/stream/' + sid);

  es.addEventListener('stage', e => {
    const d = JSON.parse(e.data);
    updateStage(d.stage, d.label, d.progress);
    const ls = document.getElementById('liveStatus');
    if(ls) ls.textContent = d.label || d.stage;
  });

  // Token 流式渲染
  es.addEventListener('token', e => {
    const d = JSON.parse(e.data);
    if(!window._tokenBuf){ window._tokenBuf = ''; }
    window._tokenBuf += (d.delta || '');
    let preview = document.getElementById('tokenPreview');
    if(!preview){
      const div = document.createElement('div');
      div.className = 'message msg-agent';
      div.innerHTML = '<div class="bubble" id="tokenPreview" style="min-height:20px"></div>';
      chatArea.appendChild(div);
      preview = document.getElementById('tokenPreview');
    }
    preview.innerHTML = renderMD(escapeHtml(window._tokenBuf));
    preview.scrollIntoView({behavior:'smooth'});
  });

  // 工具调用事件 — 填充侧栏工具日志
  es.addEventListener('tool', e => {
    const d = JSON.parse(e.data);
    const nameMap = {web_search:'网页搜索', arxiv_search:'学术论文', code_execution:'代码执行', retrieval:'知识检索'};
    const toolLabel = nameMap[d.tool_name] || d.tool_name;
    const icon = d.success ? '✓' : '✗';
    const cssClass = d.success ? 'success' : 'fail';
    addToolEntry(toolLabel, icon, d.elapsed, d.tokens, cssClass);
    // 同步更新按钮状态
    const ls = document.getElementById('liveStatus');
    if(ls) ls.textContent = `正在调用: ${toolLabel}`;
  });

  // 验证事件
  es.addEventListener('verify', e => {
    const d = JSON.parse(e.data);
    const scorePct = Math.round(d.score * 100);
    addToolEntry(`验证: ${d.task_id.slice(-8)}`, d.pass ? '✓' : '✗', 0, 0, d.pass ? 'success' : 'fail', `评分 ${scorePct}%`);
  });

  es.addEventListener('report_ready', e => {
    const d = JSON.parse(e.data);
    updateStage('writing', '报告已生成', 1.0);
    const ls = document.getElementById('liveStatus');
    if(ls) ls.textContent = '报告已生成，正在整理...';
  });

  es.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    completeAllStages();
    setStatus('done', '完成');
    cleanup();

    let finalText = window._tokenBuf || d.final_answer_text || '报告生成完毕。';
    window._tokenBuf = null;

    let dl = '<div class="download-row">';
    dl += `<a class="dl-btn dl-final" href="/api/download/${sid}/final" download>`;
    dl += `<span class="dl-label">下载研究报告</span>`;
    dl += `<span class="dl-hint">final_answer.md</span>`;
    dl += `</a>`;
    dl += `<a class="dl-btn dl-debug" href="/api/download/${sid}/debug" download>`;
    dl += `<span class="dl-label">下载调试报告</span>`;
    dl += `<span class="dl-hint">debug_report.md</span>`;
    dl += `</a>`;
    dl += '</div>';

    let preview = document.getElementById('tokenPreview');
    if(preview){
      preview.innerHTML = renderMD(escapeHtml(finalText)) + dl;
    } else {
      addMessage('agent', finalText, dl);
    }
  });

  // 仅捕获服务器推送的 SSE error 事件 (e.data 存在); 浏览器原生错误走 onerror
  es.addEventListener('sse_error', e => {
    let errMsg = '未知错误';
    try { const d = JSON.parse(e.data); errMsg = d.error || errMsg; } catch(_) {}
    setStatus('error', '失败');
    cleanup();
    addMessage('agent', '❌ 研究失败: ' + errMsg);
  });

  es.addEventListener('heartbeat', e => { /* keepalive */ });

  // 浏览器原生错误: 连接失败/断连 → 不重连
  es.onerror = (evt) => {
    if(es && es.readyState === EventSource.CLOSED){
      // 正常关闭 (done 已触发)
    } else {
      setStatus('error', '连接中断');
      cleanup();
      addMessage('agent', '❌ SSE 连接中断，请重试');
    }
  };
}

function resetSidebar(){
  document.querySelectorAll('.stage-dot').forEach(d => {d.className='stage-dot'});
  document.querySelectorAll('.stage-item').forEach(i => {i.classList.remove('active')});
  document.querySelectorAll('.stage-label').forEach(l => {l.textContent='等待中'});
  document.getElementById('execProgress').style.width = '0%';
  document.getElementById('toolLog').innerHTML = '<div class="section-title">工具调用日志</div>';
}

function updateStage(stage, label, progress){
  const s = STAGES[stage];
  if(!s) return;
  const el = document.getElementById(s.el);
  if(!el) return;
  el.classList.add('active');
  const dot = el.querySelector('.stage-dot');
  if(dot){ dot.className = 'stage-dot running'; }
  const lbl = el.querySelector('.stage-label');
  if(lbl && label) lbl.textContent = label;
  // 标记已完成的前面阶段
  markDoneBefore(stage);
  // 进度条
  if(stage === 'executing' && progress){
    document.getElementById('execProgress').style.width = (progress * 100) + '%';
  }
}

function markDoneBefore(currentStage){
  const order = ['planning','scheduling','executing','verifying','replanning','writing'];
  const idx = order.indexOf(currentStage);
  document.querySelectorAll('.stage-item').forEach(el => {
    const sid = el.id.replace('stage-','');
    // map 'plan' to 'planning', etc
    const mapped = {plan:'planning',exec:'executing',verify:'verifying',write:'writing'};
    const s = mapped[sid] || sid;
    const si = order.indexOf(s);
    if(si >= 0 && si < idx){
      const dot = el.querySelector('.stage-dot');
      if(dot && !dot.classList.contains('running')) dot.className = 'stage-dot done';
    }
  });
}

function completeAllStages(){
  document.querySelectorAll('.stage-dot').forEach(d => {d.className='stage-dot done'});
  document.querySelectorAll('.stage-item').forEach(i => {i.classList.add('active')});
  document.querySelectorAll('.stage-label').forEach(l => {l.textContent='已完成'});
  document.getElementById('execProgress').style.width = '100%';
}

function addToolEntry(name, icon, elapsed, tokens, cssClass, extraInfo){
  const log = document.getElementById('toolLog');
  const entry = document.createElement('div');
  entry.className = 'tool-entry ' + (cssClass || 'success');
  let meta = '';
  if(elapsed > 0) meta += `${elapsed.toFixed(1)}s`;
  if(tokens > 0) meta += (meta?' · ':'') + `${tokens} tokens`;
  if(extraInfo) meta += (meta?' · ':'') + extraInfo;
  entry.innerHTML = `<span class="t-icon">${icon||'✓'}</span>`
    + `<span class="t-name">${escapeHtml(name)}</span>`
    + (meta ? `<span class="t-meta">${meta}</span>` : '');
  log.appendChild(entry);
  entry.scrollIntoView({behavior:'smooth'});
}

function setStatus(status, text){
  const badge = document.getElementById('statusBadge');
  badge.textContent = text;
  badge.className = 'badge badge-' + status;
}

function addMessage(role, text, downloadHtml){
  const div = document.createElement('div');
  div.className = 'message msg-' + (role === 'user' ? 'user' : 'agent');
  let bubble = '';
  if(text){
    bubble += '<div class="bubble">' + renderMD(escapeHtml(text)) + '</div>';
  }
  if(downloadHtml) bubble += downloadHtml;
  if(!bubble) return;  // 无内容不创建空消息
  div.innerHTML = bubble;
  document.getElementById('chatArea').appendChild(div);
  div.scrollIntoView({behavior:'smooth'});
}

function escapeHtml(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function renderMD(md){
  if(!md) return '';
  let h = md;
  // 代码块 (```...```)
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g,'<pre><code>$2</code></pre>');
  // 表格 (| a | b | → <table>)
  h = h.replace(/(^\|.+\|\n\|[-| ]+\|\n((^\|.+\|\n?)*))/gm, function(m){
    let rows = m.trim().split('\n').filter(r => r.includes('|') && !r.match(/^[\|\- :]+$/));
    let html = '<table>';
    rows.forEach((row, i) => {
      let cells = row.split('|').filter(c => c.trim()).map(c => i===0?'<th>'+c.trim()+'</th>':'<td>'+c.trim()+'</td>').join('');
      html += '<tr>'+cells+'</tr>';
    });
    return html+'</table>';
  });
  // Headers
  h = h.replace(/^#### (.+)$/gm,'<h4>$1</h4>');
  h = h.replace(/^### (.+)$/gm,'<h3>$1</h3>');
  h = h.replace(/^## (.+)$/gm,'<h2>$1</h2>');
  h = h.replace(/^# (.+)$/gm,'<h1>$1</h1>');
  // Bold / italic
  h = h.replace(/\*\*(.+?)\*\*/g,'<b>$1</b>');
  h = h.replace(/\*(.+?)\*/g,'<em>$1</em>');
  // Links
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank">$1</a>');
  // Blockquote
  h = h.replace(/^&gt; (.+)$/gm,'<blockquote>$1</blockquote>');
  // Horizontal rule
  h = h.replace(/^---$/gm,'<hr>');
  // Ordered lists (1. item)
  h = h.replace(/(^(\d+)\. .+$\n?)+/gm, function(m){
    let items = m.trim().split('\n').map(s => '<li>'+s.replace(/^\d+\. /,'')+'</li>').join('');
    return '<ol>'+items+'</ol>';
  });
  // Unordered lists
  h = h.replace(/(^[-*] .+$\n?)+/gm, function(m){
    let items = m.trim().split('\n').map(s => '<li>'+s.replace(/^[-*] /,'')+'</li>').join('');
    return '<ul>'+items+'</ul>';
  });
  // Code inline
  h = h.replace(/`(.+?)`/g,'<code>$1</code>');
  // Paragraphs
  h = h.replace(/\n\n/g,'</p><p>');
  h = h.replace(/\n/g,'<br>');
  return '<p>'+h+'</p>';
}
</script>
</body>
</html>"""

# ─── 入口 ────────────────────────────────────────────────────────────────────

def main():
    app = create_app()
    port = int(os.environ.get("PORT", 8080))
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  HorizonRL-Agent Web v3 (SSE 实时推送)                       ║
║  http://localhost:{port}                                         ║
║  Ctrl+C 停止                                                 ║
╚══════════════════════════════════════════════════════════════╝
""")
    try:
        web.run_app(app, host="127.0.0.1", port=port, print=lambda *a: None)
    except OSError as e:
        print(f"\n端口 {port} 被占用: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
