"""
=======================================================================
05_web_agent.py — HorizonRL-Agent 对话式 Web 界面 (v2: 共享 Service)
=======================================================================

自包含 Web 应用: aiohttp 后端 + 原生 HTML/JS 前端。

路由:
    POST /api/chat              — 对话入口 (chat/auto/deep 三模式)
    GET  /api/report/{sid}      — 轮询深度研究报告状态
    GET  /api/download/{sid}/{kind} — 下载 final/debug markdown

v2 变更 (Day 6):
    - 改为调用共享 research_service，CLI/Web 执行路径完全一致
    - 删除手动编排循环，走 ResearchOrchestrator 全链路
    - 使用 resolve_mode() 统一模式判断

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
    run_research_session,
    resolve_mode,
    SessionArtifacts,
)
from horizonrl.llm.client import LLMClient

# ─── 全局状态 ────────────────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}


# ─── 后台 Agent 管道 ─────────────────────────────────────────────────────────

async def run_agent_pipeline(session_id: str, query: str):
    """后台执行 Agent 管道 — 通过共享 service 走全链路。

    完成后更新 _sessions[session_id] 供前端轮询。
    """
    _sessions[session_id]["status"] = "running"
    _sessions[session_id]["phase"] = "starting"
    _sessions[session_id]["progress_messages"] = []
    t0 = time.time()

    def emit(phase: str, message: str):
        _sessions[session_id]["phase"] = phase
        _sessions[session_id]["progress_messages"].append({
            "phase": phase, "message": message, "ts": time.time(),
        })

    try:
        emit("planning", "正在加载配置与工具...")

        # ── 配置 ──
        try:
            cfg = load_config(
                Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None
            )
        except Exception:
            cfg = RootConfig()

        # ── LLM 客户端 ──
        llm_client = None
        if cfg.llm.api_key:
            try:
                llm_client = LLMClient(cfg.llm)
            except Exception:
                pass

        emit("planning", "正在规划任务...")

        # ── 调用共享 Service (唯一执行入口) ──
        artifacts = await run_research_session(
            query=query,
            mode="deep",
            session_id=session_id,
            llm_client=llm_client,
            config=cfg,
            export_dir="reports",
        )

        elapsed = time.time() - t0

        if artifacts.error:
            _sessions[session_id].update({
                "status": "failed",
                "error": artifacts.error,
            })
            return

        # ── 完成 ──
        emit("completed", "研究报告已完成!")
        _sessions[session_id].update({
            "status": "completed",
            "final_path": str(artifacts.final_answer_path),
            "debug_path": str(artifacts.debug_report_path),
            "final_answer": artifacts.final_answer_text,
            "process": {
                "success": artifacts.stats.get("success_count", 0),
                "total": artifacts.stats.get("total_count", 0),
                "rounds": artifacts.stats.get("rounds", 0),
                "tool_calls": artifacts.tool_calls_count,
                "replans": artifacts.stats.get("total_replans", 0),
                "mock_ratio": f"{artifacts.mock_ratio:.0%}",
                "elapsed": f"{elapsed:.1f}s",
                "provider": artifacts.used_search_provider,
            },
        })

    except Exception as e:
        _sessions[session_id].update({"status": "failed", "error": str(e)})


# ─── LLM 对话 ────────────────────────────────────────────────────────────────

async def run_chat(query: str) -> str:
    """直接调用 LLM 对话，不触发 Agent 管道。"""
    try:
        cfg = load_config(
            Path("configs/dev.yaml") if Path("configs/dev.yaml").exists() else None
        )
    except Exception:
        cfg = RootConfig()

    if not cfg.llm.api_key:
        return (
            "我没有配置 LLM API Key，无法进行对话。\n\n"
            "请复制 `.env.example` 为 `.env` 并填入你的 DeepSeek 或 OpenAI Key。\n\n"
            "当前可以：输入学术/研究类问题自动触发 Agent 离线研究管道。"
        )

    try:
        client = LLMClient(cfg.llm)
        result = await client.chat(
            query,
            system_prompt="你是一个友好的AI助手。用简洁流畅的中文回答。",
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

    # deep 模式: 启动后台 Agent 管道
    sid = f"session_{uuid.uuid4().hex[:12]}"
    _sessions[sid] = {"status": "queued", "phase": "", "query": message}
    asyncio.create_task(run_agent_pipeline(sid, message))
    return web.json_response(
        {"mode": "agent", "session_id": sid, "status": "queued"},
        dumps=lambda o: json.dumps(o, ensure_ascii=False),
    )


async def handle_api_report(request: web.Request) -> web.Response:
    """GET /api/report/{session_id} — 轮询深度研究状态。"""
    sid = request.match_info["session_id"]
    session = _sessions.get(sid)
    if not session:
        return web.json_response({"error": "session not found"}, status=404)

    resp = {
        "status": session["status"],
        "phase": session.get("phase", ""),
        "progress_messages": session.get("progress_messages", []),
    }
    if session["status"] == "completed":
        resp["final_answer"] = session.get("final_answer", "")
        resp["download_url_final"] = f"/api/download/{sid}/final"
        resp["download_url_debug"] = f"/api/download/{sid}/debug"
        resp["process"] = session.get("process", {})
    elif session["status"] == "failed":
        resp["error"] = session.get("error", "未知错误")

    return web.json_response(resp, dumps=lambda o: json.dumps(o, ensure_ascii=False))


async def handle_api_download(request: web.Request) -> web.Response:
    """GET /api/download/{session_id}/{kind} — 下载 Markdown 文件。"""
    sid = request.match_info["session_id"]
    kind = request.match_info["kind"]
    session = _sessions.get(sid)

    if not session or session["status"] != "completed":
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
    app.router.add_get("/api/report/{session_id}", handle_api_report)
    app.router.add_get("/api/download/{session_id}/{kind}", handle_api_download)
    app.router.add_get("/favicon.ico", lambda r: web.Response(status=204))
    return app


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HTML 前端                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HorizonRL-Agent</title>
<style>
  :root{--bg:#0f1117;--surface:#1a1d27;--border:#2a2d3a;--text:#e1e4ea;--text2:#9ca3af;--accent:#6c8cff;--success:#4ade80;--fail:#f87171;--warn:#fbbf24;--radius:12px}
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
  .container{max-width:800px;margin:0 auto;padding:16px}
  header{text-align:center;padding:24px 0 16px}
  header h1{font-size:22px;background:linear-gradient(135deg,#6c8cff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  header p{color:var(--text2);font-size:12px;margin-top:4px}
  .chat-area{min-height:55vh}
  .message{margin-bottom:16px;animation:fadeIn .3s}
  @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .msg-user{display:flex;justify-content:flex-end}
  .msg-user .bubble{background:var(--accent);color:#fff;max-width:80%;padding:10px 16px;border-radius:var(--radius) var(--radius) 4px var(--radius);font-size:14px;line-height:1.5}
  .msg-agent .bubble{background:var(--surface);border:1px solid var(--border);padding:16px 20px;border-radius:var(--radius);font-size:14px;line-height:1.7;max-width:100%;word-wrap:break-word}
  .msg-agent .bubble h1,.msg-agent .bubble h2{font-size:16px;margin:10px 0 4px;color:var(--accent)}
  .msg-agent .bubble h3{font-size:14px;margin:8px 0 2px;color:#a78bfa}
  .msg-agent .bubble blockquote{border-left:3px solid var(--accent);padding-left:10px;color:var(--text2);margin:8px 0}
  .msg-agent .bubble code{background:#0003;padding:1px 5px;border-radius:4px;font-size:12px}
  .process-badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:4px;margin:2px}
  .badge-mock{background:var(--warn);color:#000}
  .badge-real{background:var(--success);color:#000}
  .status-bar{padding:8px 0;display:none;font-size:12px;color:var(--text2)}
  .status-bar.show{display:flex;align-items:center;gap:8px}
  .spinner{width:12px;height:12px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .dl-btn{display:inline-block;margin:4px 6px 0 0;padding:4px 12px;background:var(--accent);color:#fff;border-radius:6px;font-size:12px;text-decoration:none;cursor:pointer}
  .dl-btn:hover{opacity:.8}
  .input-area{position:sticky;bottom:0;background:var(--bg);padding:12px 0;border-top:1px solid var(--border)}
  .input-row{display:flex;gap:8px}
  .input-row input{flex:1;padding:10px 14px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-size:14px;outline:none}
  .input-row input:focus{border-color:var(--accent)}
  .input-row select{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-size:12px;padding:8px;outline:none}
  .input-row button{padding:10px 20px;background:var(--accent);color:#fff;border:none;border-radius:var(--radius);font-size:14px;cursor:pointer;font-weight:600}
  .input-row button:disabled{opacity:.4;cursor:not-allowed}
  footer{text-align:center;padding:16px 0;color:var(--text2);font-size:11px}
</style>
</head>
<body>
<div class="container">
<header><h1>HorizonRL-Agent</h1><p>AI 研究助手 · 对话 + 深度研究自动切换</p></header>
<div class="chat-area" id="chatArea">
<div class="message msg-agent"><div class="bubble">
你好！我是一个 AI 研究助手。<br><br>
<b>普通对话</b>: 直接问任何问题，我即时回答。<br>
<b>深度研究</b>: 输入学术/综述/对比类问题，我会自动搜索网络和论文，验证后写报告。<br><br>
试试: <b>"Transformer注意力机制最新进展"</b> 或 <b>"你好，介绍一下自己"</b>
</div></div></div>
<div class="status-bar" id="statusBar"><div class="spinner"></div><span id="statusText">Agent 研究中...</span></div>
<div class="input-area">
<div class="input-row">
<input id="query" placeholder="输入你想研究的问题..." onkeydown="if(event.key==='Enter')send()">
<select id="modeSel"><option value="auto">自动</option><option value="chat">对话</option><option value="deep">深度研究</option></select>
<button id="btn" onclick="send()">发送</button>
</div></div>
<footer>HorizonRL-Agent v0.2.1 · NWPU · 2026</footer>
</div>
<script>
let isPolling=false;

function send(){
  const inp=document.getElementById('query');
  const q=inp.value.trim();
  if(!q||isPolling)return;
  inp.value='';
  const mode=document.getElementById('modeSel').value;
  const btn=document.getElementById('btn');
  btn.disabled=true;btn.textContent='...';
  addMessage('user',q);
  fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:q,mode:mode})})
  .then(r=>r.json())
  .then(data=>{
    if(data.mode==='chat'){
      addMessage('agent',data.answer||'(空)');
      btn.disabled=false;btn.textContent='发送';
    }else if(data.mode==='agent'){
      addMessage('agent','正在启动深度研究管道...\n\n规划 → 搜索 → 验证 → 写作');
      startPolling(data.session_id);
    }
  })
  .catch(e=>{addMessage('agent','错误: '+e.message);btn.disabled=false;btn.textContent='发送'});
}

function startPolling(sid){
  isPolling=true;
  const bar=document.getElementById('statusBar');
  bar.classList.add('show');
  const phases={planning:'正在规划任务...',searching:'正在搜索资料...',executing:'正在执行子任务...',
    verifying:'正在验证结果...',replanning:'正在重规划...',writing:'正在撰写报告...',completed:'完成!'};
  let shownMsgs=0;

  const interval=setInterval(()=>{
    fetch('/api/report/'+sid).then(r=>r.json()).then(data=>{
      const msgs=data.progress_messages||[];
      while(shownMsgs<msgs.length){
        const m=msgs[shownMsgs];
        document.getElementById('statusText').textContent=
          phases[m.phase]||m.message||m.phase;
        shownMsgs++;
      }
      if(data.status==='completed'){
        clearInterval(interval);isPolling=false;
        bar.classList.remove('show');
        document.getElementById('btn').disabled=false;
        document.getElementById('btn').textContent='发送';
        const p=data.process||{};
        let html='深度研究完成!\n\n';
        html+='<span class="process-badge badge-'+(p.mock_ratio==='100%'?'mock':'real')+'">'+
          (p.mock_ratio==='100%'?'Mock Demo':'真实数据')+'</span> ';
        html+='<span style="font-size:11px;color:var(--text2)">'+
          p.success+'/'+p.total+' 成功 · '+p.tool_calls+' 工具 · '+p.elapsed+'</span>\n\n';
        html+=(data.final_answer||'');
        let dl='<div style="margin-top:12px">';
        dl+='<a class="dl-btn" href="'+data.download_url_final+'" download>📥 下载 final_answer.md</a> ';
        dl+='<a class="dl-btn" href="'+data.download_url_debug+'" download>📥 下载 debug_report.md</a></div>';
        addMessage('agent',html,null,dl);
      }else if(data.status==='failed'){
        clearInterval(interval);isPolling=false;
        bar.classList.remove('show');
        document.getElementById('btn').disabled=false;
        document.getElementById('btn').textContent='发送';
        addMessage('agent','研究失败: '+(data.error||''));
      }
    });
  },500);
}

function addMessage(role,text,process,downloadHtml){
  const div=document.createElement('div');
  div.className='message msg-'+(role==='user'?'user':'agent');
  let bubble='<div class="bubble">'+renderMD(escapeHtml(text))+'</div>';
  if(downloadHtml)bubble+=downloadHtml;
  div.innerHTML=bubble;
  document.getElementById('chatArea').appendChild(div);
  div.scrollIntoView({behavior:'smooth'});
}

function escapeHtml(s){return(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function renderMD(md){
  if(!md)return'';
  let h=escapeHtml(md);
  h=h.replace(/^### (.+)$/gm,'<h3>$1</h3>');
  h=h.replace(/^## (.+)$/gm,'<h2>$1</h2>');
  h=h.replace(/^# (.+)$/gm,'<h1>$1</h1>');
  h=h.replace(/\*\*(.+?)\*\*/g,'<b>$1</b>');
  h=h.replace(/^&gt; (.+)$/gm,'<blockquote>$1</blockquote>');
  h=h.replace(/^---$/gm,'<hr>');
  h=h.replace(/\n\n/g,'</p><p>');
  h=h.replace(/\n/g,'<br>');
  return'<p>'+h+'</p>';
}
</script>
</body>
</html>"""


def main():
    app = create_app()
    port = int(os.environ.get("PORT", 8080))
    print(f"""
==============================================================
  HorizonRL-Agent Web — 对话+深度研究双路由 (v2: 共享Service)
  http://localhost:{port}
  Ctrl+C 停止
==============================================================
""")
    try:
        web.run_app(app, host="127.0.0.1", port=port, print=lambda *a: None)
    except OSError as e:
        print(f"\n端口 {port} 被占用: {e}\ntaskkill /F /IM python.exe")
        sys.exit(1)


if __name__ == "__main__":
    main()
