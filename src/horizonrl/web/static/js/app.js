/**
 * HorizonRL-Agent Web UI — 主控制器
 *
 * 管理全局状态、UI 更新、SSE 事件处理。
 * 依赖: SSEClient, MarkdownRenderer
 */

// ═══════════════════════════════════════════════════════════════════════════
// Global State
// ═══════════════════════════════════════════════════════════════════════════
const STATE = {
  isRunning: false,
  currentSid: null,
  sseClient: null,
  tokenBuf: null,
  stageOrder: ['planning', 'scheduling', 'executing', 'verifying', 'replanning', 'writing'],
  stageMap: {
    planning:   { el: 'stage-plan',    name: '任务规划',  icon: '📋' },
    scheduling: { el: 'stage-plan',    name: '任务调度',  icon: '📋' },
    executing:  { el: 'stage-exec',    name: '执行子任务', icon: '🔧' },
    verifying:  { el: 'stage-verify',  name: '验证结果',  icon: '✅' },
    replanning: { el: 'stage-exec',    name: '局部重规划', icon: '🔄' },
    writing:    { el: 'stage-write',   name: '撰写报告',  icon: '📝' },
  },
  stats: {
    toolCalls: 0,
    toolSuccess: 0,
    toolFail: 0,
    verifications: 0,
    verPassed: 0,
    startTime: 0,
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
  await MarkdownRenderer.init();
  bindEvents();
});

function bindEvents() {
  document.getElementById('queryInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  document.getElementById('sendBtn').addEventListener('click', sendMessage);
}

// ═══════════════════════════════════════════════════════════════════════════
// Message Sending
// ═══════════════════════════════════════════════════════════════════════════
function sendMessage() {
  if (STATE.isRunning) return;

  // Cleanup previous session
  if (STATE.sseClient) {
    STATE.sseClient.disconnect();
    STATE.sseClient = null;
  }
  STATE.tokenBuf = null;
  removeTokenPreview();

  const input = document.getElementById('queryInput');
  const query = input.value.trim();
  if (!query) return;

  input.value = '';
  const mode = document.getElementById('modeSel').value;
  const btn = document.getElementById('sendBtn');

  setButtonLoading(true);
  addMessage('user', query);

  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: query, mode }),
  })
    .then(r => r.json())
    .then(data => {
      if (data.mode === 'chat') {
        addMessage('agent', data.answer || '(空)');
        setButtonLoading(false);
      } else if (data.mode === 'agent') {
        STATE.currentSid = data.session_id;
        startResearchSession(data.session_id);
      }
    })
    .catch(err => {
      showToast('请求失败: ' + err.message, 'error');
      setButtonLoading(false);
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Research Session (SSE)
// ═══════════════════════════════════════════════════════════════════════════
function startResearchSession(sid) {
  STATE.isRunning = true;
  STATE.tokenBuf = null;
  STATE.stats = { toolCalls: 0, toolSuccess: 0, toolFail: 0, verifications: 0, verPassed: 0, startTime: Date.now() };

  resetSidebar();
  resetDetailPanel();
  setStatus('running', '研究中...');

  // Create status row
  const statusDiv = document.createElement('div');
  statusDiv.className = 'status-row';
  statusDiv.id = 'statusRow';
  statusDiv.innerHTML = '<div class="spinner"></div><span id="liveStatus">正在规划...</span>';
  const chatArea = document.getElementById('chatArea');
  const wrapper = document.createElement('div');
  wrapper.className = 'message msg-agent';
  wrapper.id = 'statusWrapper';
  wrapper.appendChild(statusDiv);
  chatArea.appendChild(wrapper);
  wrapper.scrollIntoView({ behavior: 'smooth' });

  STATE.sseClient = new SSEClient('/api/stream/' + sid, {
    onStage: handleStage,
    onToken: handleToken,
    onTool: handleTool,
    onVerify: handleVerify,
    onReportReady: handleReportReady,
    onDone: handleDone,
    onSseError: handleSseError,
    onDisconnect: handleDisconnect,
  });
  STATE.sseClient.connect();
}

// ═══════════════════════════════════════════════════════════════════════════
// SSE Event Handlers
// ═══════════════════════════════════════════════════════════════════════════
function handleStage(data) {
  updateStage(data.stage, data.label, data.progress);
  const ls = document.getElementById('liveStatus');
  if (ls) ls.textContent = data.label || data.stage;
}

function handleToken(data) {
  if (!STATE.tokenBuf) STATE.tokenBuf = '';
  STATE.tokenBuf += (data.delta || '');

  let preview = document.getElementById('tokenPreview');
  if (!preview) {
    const div = document.createElement('div');
    div.className = 'message msg-agent';
    div.id = 'tokenMsg';
    div.innerHTML = '<div class="bubble" id="tokenPreview" style="min-height:20px"></div>';
    document.getElementById('chatArea').appendChild(div);
    preview = document.getElementById('tokenPreview');
  }
  preview.innerHTML = MarkdownRenderer.render(STATE.tokenBuf);
  preview.scrollIntoView({ behavior: 'smooth' });
}

function handleTool(data) {
  STATE.stats.toolCalls++;
  if (data.success) STATE.stats.toolSuccess++;
  else STATE.stats.toolFail++;

  const nameMap = {
    web_search: '网页搜索', paper_search: '学术论文',
    code_execution: '代码执行', retrieval: '知识检索',
  };
  const label = nameMap[data.tool_name] || data.tool_name;
  const icon = data.success ? '✓' : '✗';
  const cssClass = data.success ? 'success' : 'fail';

  addToolEntry(label, icon, data.elapsed, data.tokens, cssClass);
  updateDetailStats();

  const ls = document.getElementById('liveStatus');
  if (ls) ls.textContent = `正在调用: ${label}`;
}

function handleVerify(data) {
  STATE.stats.verifications++;
  if (data.pass) STATE.stats.verPassed++;

  const scorePct = Math.round((data.score || 0) * 100);
  addToolEntry(
    `验证: ${(data.task_id || '').slice(-8)}`,
    data.pass ? '✓' : '✗',
    0, 0,
    data.pass ? 'success' : 'fail',
    `评分 ${scorePct}%`
  );
  updateDetailStats();
}

function handleReportReady(data) {
  updateStage('writing', '报告已生成', 1.0);
  const ls = document.getElementById('liveStatus');
  if (ls) ls.textContent = '报告已生成，正在整理...';
}

function handleDone(data) {
  completeAllStages();
  setStatus('done', '完成');

  let finalText = STATE.tokenBuf || data.final_answer_text || '报告生成完毕。';
  STATE.tokenBuf = null;

  const runtime = data.runtime_ms ? (data.runtime_ms / 1000).toFixed(1) + 's' : '';
  updateDetailStats({ runtime });

  const dlHtml = buildDownloadButtons(STATE.currentSid);

  const preview = document.getElementById('tokenPreview');
  if (preview) {
    preview.innerHTML = MarkdownRenderer.render(finalText) + dlHtml;
  } else {
    addMessage('agent', finalText, dlHtml);
  }

  // 标记完成并关闭 SSE，阻止重连导致重新执行
  if (STATE.sseClient) {
    STATE.sseClient.markFinished();
  }
  finishSession();
}

function handleSseError(data) {
  const errMsg = data.error || '未知错误';
  setStatus('error', '失败');
  showToast('研究失败: ' + errMsg, 'error');
  addMessage('agent', '❌ 研究失败: ' + errMsg);

  // 标记失败并关闭 SSE，阻止重连
  if (STATE.sseClient) {
    STATE.sseClient.markFinished();
  }
  finishSession();
}

function handleDisconnect(msg) {
  setStatus('error', '连接中断');
  showToast(msg, 'error');

  if (STATE.sseClient) {
    STATE.sseClient.markFinished();
  }
  finishSession();
}

// ═══════════════════════════════════════════════════════════════════════════
// Session Lifecycle
// ═══════════════════════════════════════════════════════════════════════════
function finishSession() {
  STATE.isRunning = false;
  STATE.sseClient = null;
  setButtonLoading(false);

  // Remove status wrapper
  const wrapper = document.getElementById('statusWrapper');
  if (wrapper) wrapper.remove();
}

function removeTokenPreview() {
  const msg = document.getElementById('tokenMsg');
  if (msg) msg.remove();
  const preview = document.getElementById('tokenPreview');
  if (preview && preview.parentNode) {
    // Already handled via tokenMsg removal
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sidebar: Stage Timeline
// ═══════════════════════════════════════════════════════════════════════════
function resetSidebar() {
  document.querySelectorAll('.stage-dot').forEach(d => {
    d.className = 'stage-dot';
  });
  document.querySelectorAll('.stage-line').forEach(l => {
    l.classList.remove('done');
  });
  document.querySelectorAll('.stage-item').forEach(i => {
    i.classList.remove('active');
  });
  document.querySelectorAll('.stage-label').forEach(l => {
    l.textContent = '等待中';
  });
  document.getElementById('execProgress').style.width = '0%';
  document.getElementById('toolLog').innerHTML =
    '<div class="section-title">工具调用日志</div>';

  document.getElementById('liveDot')?.classList.add('active');
}

function updateStage(stage, label, progress) {
  const s = STATE.stageMap[stage];
  if (!s) return;

  const el = document.getElementById(s.el);
  if (!el) return;

  el.classList.add('active');
  const dot = el.querySelector('.stage-dot');
  if (dot) { dot.className = 'stage-dot running'; }
  const lbl = el.querySelector('.stage-label');
  if (lbl && label) lbl.textContent = label;

  // Mark completed stages before current
  const idx = STATE.stageOrder.indexOf(stage);
  document.querySelectorAll('.stage-item').forEach(item => {
    const elId = item.id.replace('stage-', '');
    const mapped = { plan: 'planning', exec: 'executing', verify: 'verifying', write: 'writing' };
    const sName = mapped[elId] || elId;
    const si = STATE.stageOrder.indexOf(sName);
    if (si >= 0 && si < idx) {
      const d = item.querySelector('.stage-dot');
      if (d && !d.classList.contains('running')) d.className = 'stage-dot done';
      const line = item.querySelector('.stage-line');
      if (line) line.classList.add('done');
    }
  });

  if (stage === 'executing' && progress != null) {
    document.getElementById('execProgress').style.width = (progress * 100) + '%';
  }
}

function completeAllStages() {
  document.querySelectorAll('.stage-dot').forEach(d => { d.className = 'stage-dot done'; });
  document.querySelectorAll('.stage-line').forEach(l => { l.classList.add('done'); });
  document.querySelectorAll('.stage-item').forEach(i => { i.classList.add('active'); });
  document.querySelectorAll('.stage-label').forEach(l => { l.textContent = '已完成'; });
  document.getElementById('execProgress').style.width = '100%';
  document.getElementById('liveDot')?.classList.remove('active');
}

// ═══════════════════════════════════════════════════════════════════════════
// Sidebar: Tool Log
// ═══════════════════════════════════════════════════════════════════════════
function addToolEntry(name, icon, elapsed, tokens, cssClass, extraInfo) {
  const log = document.getElementById('toolLog');
  const entry = document.createElement('div');
  entry.className = 'tool-entry ' + (cssClass || 'success');

  let meta = '';
  if (elapsed > 0) meta += `${elapsed.toFixed(1)}s`;
  if (tokens > 0) meta += (meta ? ' · ' : '') + `${tokens} tokens`;
  if (extraInfo) meta += (meta ? ' · ' : '') + extraInfo;

  entry.innerHTML =
    `<span class="t-icon">${icon || '✓'}</span>` +
    `<span class="t-name">${escapeHtml(name)}</span>` +
    (meta ? `<span class="t-meta">${meta}</span>` : '');

  log.appendChild(entry);
  entry.scrollIntoView({ behavior: 'smooth' });
}

// ═══════════════════════════════════════════════════════════════════════════
// Detail Panel
// ═══════════════════════════════════════════════════════════════════════════
function resetDetailPanel() {
  document.getElementById('statTools').textContent = '0';
  document.getElementById('statSuccess').textContent = '0';
  document.getElementById('statFail').textContent = '0';
  document.getElementById('statVerifications').textContent = '0';
  document.getElementById('statRuntime').textContent = '--';
}

function updateDetailStats(extra = {}) {
  document.getElementById('statTools').textContent = STATE.stats.toolCalls;
  document.getElementById('statSuccess').textContent = STATE.stats.toolSuccess;
  document.getElementById('statFail').textContent = STATE.stats.toolFail;
  document.getElementById('statVerifications').textContent = STATE.stats.verifications;
  if (extra.runtime) {
    document.getElementById('statRuntime').textContent = extra.runtime;
  } else if (STATE.stats.startTime) {
    const elapsed = ((Date.now() - STATE.stats.startTime) / 1000).toFixed(1);
    document.getElementById('statRuntime').textContent = elapsed + 's';
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Chat Messages
// ═══════════════════════════════════════════════════════════════════════════
function addMessage(role, text, downloadHtml) {
  const div = document.createElement('div');
  div.className = 'message msg-' + (role === 'user' ? 'user' : 'agent');

  let html = '';
  if (text) {
    const rendered = role === 'user' ? escapeHtml(text) : MarkdownRenderer.render(text);
    html += '<div class="bubble">' + rendered + '</div>';
  }
  if (downloadHtml) html += downloadHtml;
  if (!html) return;

  div.innerHTML = html;
  document.getElementById('chatArea').appendChild(div);
  div.scrollIntoView({ behavior: 'smooth' });
}

function buildDownloadButtons(sid) {
  return `
    <div class="download-row">
      <a class="dl-btn dl-final" href="/api/download/${sid}/final" download>
        <span class="dl-label">📥 下载研究报告</span>
        <span class="dl-hint">final_answer.md</span>
      </a>
      <a class="dl-btn dl-debug" href="/api/download/${sid}/debug" download>
        <span class="dl-label">🔍 下载调试报告</span>
        <span class="dl-hint">debug_report.md</span>
      </a>
    </div>`;
}

// ═══════════════════════════════════════════════════════════════════════════
// UI Helpers
// ═══════════════════════════════════════════════════════════════════════════
function setStatus(status, text) {
  const badge = document.getElementById('statusBadge');
  badge.textContent = text;
  badge.className = 'badge badge-' + status;
}

function setButtonLoading(loading) {
  const btn = document.getElementById('sendBtn');
  btn.disabled = loading;
  btn.textContent = loading ? '⏳' : '发送';
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = 'toast ' + type;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(40px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
