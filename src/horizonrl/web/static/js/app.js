/**
 * Horizon-Agent · React SPA
 * 依赖: React 18 + htm + marked (通过全局变量)
 */
(function() {
var diag = document.getElementById('diagMsg');
function step(msg) { if (diag) diag.textContent = msg; }

try { step('Step 1/6: React...');
var _React = (typeof React !== 'undefined') ? React : null;
if (!_React) throw new Error('React global not found');
step('Step 2/6: htm...');
var _htm = (typeof htm !== 'undefined') ? htm : null;
if (!_htm) throw new Error('htm global not found');
step('Step 3/6: marked...');
var _marked = (typeof marked !== 'undefined') ? marked : null;
if (!_marked) throw new Error('marked global not found');

step('Step 4/6: Init...');
var h = _React.createElement;
var useState = _React.useState;
var useEffect = _React.useEffect;
var useRef = _React.useRef;
var useCallback = _React.useCallback;
var useReducer = _React.useReducer;
var html = _htm.bind(h);
if (_marked.setOptions) _marked.setOptions({ breaks: true, gfm: true });
var VERSION = window.APP_VERSION || '0.1.0';

// ── Constants ──────────────────────────────────────────────────────────
const PAGE_SIZE = 20;
const TOOL_NAMES = { web_search: '网页搜索', paper_search: '学术论文', code_execution: '代码执行', retrieval: '知识检索' };
const STATUS_LABELS = { completed: '已完成', failed: '失败', running: '运行中', queued: '排队中' };
const STAGE_MAP = { planning: 'plan', scheduling: 'plan', executing: 'exec', verifying: 'verify', replanning: 'exec', writing: 'write' };
const STAGE_ORDER = ['planning', 'scheduling', 'executing', 'verifying', 'replanning', 'writing'];
const HINTS = [
  'Transformer注意力机制的最新进展',
  '对比PyTorch和TensorFlow的优缺点',
  '强化学习在LLM中的应用综述',
];

const TEMPLATES = [
  { label: '论文综述', icon: ' ', prompt: '请对 {topic} 领域的最新研究进行综述，包括主要方法、关键进展和未来方向' },
  { label: '技术对比', icon: ' ', prompt: '请对比 {topic} 中不同技术方案的优缺点，包括性能、易用性和适用场景' },
  { label: '新闻摘要', icon: ' ', prompt: '请总结 {topic} 领域最近的重要新闻和动态，并按主题分类' },
  { label: '概念解释', icon: ' ', prompt: '请详细解释 {topic}，包括定义、原理、应用场景和相关概念' },
  { label: '代码分析', icon: ' ', prompt: '请分析 {topic} 的实现方式，提供代码示例并解释关键部分' },
];

// ── Initial State ──────────────────────────────────────────────────────
const init = {
  tab: 'progress',
  running: false,
  sid: null,
  queryText: '',
  queryMode: 'auto',
  messages: [],
  toasts: [],
  confirm: null,
  showSettings: false,
  apiKeys: [],
  settingsLoading: false,
  sessions: [],
  histPage: 0,
  histTotal: 0,
  histLoading: false,
  stats: { toolCalls: 0, toolSuccess: 0, toolFail: 0, verifications: 0, runtime: '--' },
  stages: [
    { id: 'plan',   name: '任务规划',   label: '等待开始', dot: '', lineDone: false, showBar: false, prog: 0, last: false },
    { id: 'exec',   name: '执行子任务', label: '等待中',   dot: '', lineDone: false, showBar: true,  prog: 0, last: false },
    { id: 'verify', name: '验证结果',   label: '等待中',   dot: '', lineDone: false, showBar: false, prog: 0, last: false },
    { id: 'write',  name: '撰写报告',   label: '等待中',   dot: '', lineDone: false, showBar: false, prog: 0, last: true },
  ],
  toolLog: [],
  sbDot: '',
};

// ── Reducer ────────────────────────────────────────────────────────────
function reducer(state, action) {
  switch (action.type) {
    case 'SET':
      return { ...state, ...action.payload };
    case 'PUSH_MSG':
      return { ...state, messages: [...state.messages, action.payload] };
    case 'POP_MSG_TYPE':
      return { ...state, messages: state.messages.filter(function(m) { return m.type !== action.payload; }) };
    case 'UPDATE_LAST_MSG': {
      if (!state.messages.length) return state;
      var msgs = state.messages.slice();
      msgs[msgs.length - 1] = Object.assign({}, msgs[msgs.length - 1], action.payload);
      return { ...state, messages: msgs };
    }
    case 'PUSH_TOAST':
      return { ...state, toasts: [...state.toasts, action.payload] };
    case 'POP_TOAST': {
      var t = state.toasts.slice();
      t.splice(action.payload, 1);
      return { ...state, toasts: t };
    }
    case 'PUSH_TOOL':
      return { ...state, toolLog: [...state.toolLog, action.payload] };
    case 'RESET_SIDEBAR':
      return {
        ...state,
        stages: init.stages.map(function(s) { return Object.assign({}, s); }),
        toolLog: [],
        stats: Object.assign({}, init.stats, { startTime: Date.now() }),
        sbDot: 'live',
      };
    case 'UPDATE_STAGE': {
      var stage = action.payload.stage;
      var label = action.payload.label;
      var progress = action.payload.progress;
      var mapped = STAGE_MAP[stage] || 'plan';
      var idx = STAGE_ORDER.indexOf(stage);
      return {
        ...state,
        stages: state.stages.map(function(s) {
          if (s.id === mapped) {
            return Object.assign({}, s, {
              active: true, dot: 'run',
              label: label || s.label,
              prog: (progress != null && s.showBar) ? progress : s.prog,
            });
          }
          // Mark completed stages before current
          var stageForThis = null;
          for (var k in STAGE_MAP) {
            if (STAGE_MAP[k] === s.id) { stageForThis = k; break; }
          }
          var si = stageForThis ? STAGE_ORDER.indexOf(stageForThis) : -1;
          if (si >= 0 && si < idx && !s.dot) {
            return Object.assign({}, s, { dot: 'done', lineDone: true, active: true });
          }
          return s;
        }),
      };
    }
    case 'COMPLETE_STAGES':
      return {
        ...state,
        sbDot: 'done',
        stages: state.stages.map(function(s) {
          return Object.assign({}, s, { dot: 'done', lineDone: true, active: true, label: '已完成', prog: s.showBar ? 1 : s.prog });
        }),
      };
    case 'FAIL_STAGES':
      return {
        ...state,
        sbDot: 'err',
        stages: state.stages.map(function(s) { return s.dot === 'run' ? Object.assign({}, s, { dot: 'fail' }) : s; }),
      };
    default:
      return state;
  }
}

// ── Helpers ────────────────────────────────────────────────────────────
function esc(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function timeAgo(ts) {
  if (!ts) return '';
  var d = Date.now() / 1000 - ts;
  if (d < 60) return '刚刚';
  if (d < 3600) return Math.floor(d / 60) + '分前';
  if (d < 86400) return Math.floor(d / 3600) + '时前';
  if (d < 604800) return Math.floor(d / 86400) + '天前';
  return new Date(ts * 1000).toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' });
}
function downloadsHtml(sid) {
  return '<a class="dl-btn final" href="/api/download/' + sid + '/final" download="final_answer.md">'
    + '<span class="dl-title">下载研究报告</span><span class="dl-sub">Markdown</span>'
    + '</a><a class="dl-btn pdf" href="/api/download/' + sid + '/pdf" download="report.pdf">'
    + '<span class="dl-title">导出 PDF</span><span class="dl-sub">pdf</span>'
    + '</a><a class="dl-btn debug" href="/api/download/' + sid + '/debug" download="debug_report.md">'
    + '<span class="dl-title">调试报告</span><span class="dl-sub">debug_report.md</span></a>';
}
function uid() { return Date.now() + Math.random(); }

// ═══════════════════════════════════════════════════════════════════════════
// App Component
// ═══════════════════════════════════════════════════════════════════════════
function App() {
  var _a = useReducer(reducer, init), s = _a[0], dispatch = _a[1];
  var msgRef = useRef(null);
  var esRef = useRef(null);
  var msgIdxRef = useRef(-1);
  var bufRef = useRef('');
  var statsRef = useRef(s.stats);
  statsRef.current = s.stats;

  // 主题切换
  var _theme = useState(localStorage.getItem('theme') || 'dark'), theme = _theme[0], setTheme = _theme[1];
  useEffect(function() {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);
  function toggleTheme() { setTheme(theme === 'dark' ? 'light' : 'dark'); }

  var scrollDown = useCallback(function() {
    setTimeout(function() {
      if (msgRef.current) msgRef.current.scrollTop = msgRef.current.scrollHeight;
    }, 50);
  }, []);

  useEffect(function() {
    if (s.messages.length) scrollDown();
  }, [s.messages.length]);

  // ── Toast ────────────────────────────────────────────────────────
  var toast = useCallback(function(msg, type) {
    var t = { id: uid(), msg: msg, type: type || 'info' };
    dispatch({ type: 'PUSH_TOAST', payload: t });
    setTimeout(function() {
      dispatch({ type: 'POP_TOAST', payload: 0 });
    }, 4000);
  }, []);

  // ── History ──────────────────────────────────────────────────────
  var loadHistory = useCallback(async function(page) {
    var p = page !== undefined ? page : s.histPage;
    dispatch({ type: 'SET', payload: { histLoading: true } });
    try {
      var r = await fetch('/api/sessions?limit=' + PAGE_SIZE + '&offset=' + (p * PAGE_SIZE));
      if (!r.ok) throw new Error('API error');
      var d = await r.json();
      dispatch({ type: 'SET', payload: { sessions: d.sessions, histTotal: d.total, histPage: p, histLoading: false } });
    } catch (e) {
      dispatch({ type: 'SET', payload: { histLoading: false } });
      if (!s.sessions.length) toast('加载失败，请重试', 'err');
    }
  }, [s.histPage]);

  useEffect(function() { loadHistory(0); }, []);

  // ── Messages ─────────────────────────────────────────────────────
  var addUserMsg = useCallback(function(text) {
    dispatch({ type: 'PUSH_MSG', payload: { type: 'user', role: 'user', html: '<p>' + esc(text) + '</p>' } });
  }, []);

  var addBotMsg = useCallback(function(text, dls) {
    dispatch({ type: 'PUSH_MSG', payload: { type: 'bot', role: 'bot', html: marked.parse(text || ''), downloads: dls || '' } });
  }, []);

  var addStatus = useCallback(function(text) {
    dispatch({ type: 'POP_MSG_TYPE', payload: 'status' });
    dispatch({ type: 'PUSH_MSG', payload: { type: 'status', text: text } });
  }, []);

  // ── Session View ─────────────────────────────────────────────────
  var viewSession = useCallback(function(session) {
    if (s.running && session.session_id !== s.sid) {
      toast('当前有研究正在进行中', 'info');
      return;
    }
    if (session.status === 'completed' || session.status === 'failed') {
      // 不清除当前对话 — 在下方追加分隔线和历史会话
      addStatus('正在加载历史会话...');
      var rt = session.runtime_ms > 0 ? (session.runtime_ms / 1000).toFixed(1) + 's' : '--';
      // 分隔线
      dispatch({ type: 'PUSH_MSG', payload: { type:'status', text:'——— 历史会话: ' + (session.query||'').slice(0,40) + ' ———' } });
      addUserMsg(session.query);
      if (session.status === 'completed') {
        var content = session.final_answer || (session.final_answer_path ? '(报告已生成，点击下方按钮下载)' : '');
        addBotMsg(content, downloadsHtml(session.session_id));
        dispatch({ type: 'SET', payload: {
          stats: Object.assign({}, s.stats, { runtime: rt }),
          tab: 'progress', sbDot: 'done',
        } });
      } else {
        addBotMsg('研究失败: ' + (session.error || '未知错误'));
      }
      // 移除加载状态
      dispatch({ type: 'POP_MSG_TYPE', payload: 'status' });
    } else if (session.status === 'running') {
      toast('该会话正在运行中', 'info');
    } else {
      dispatch({ type: 'SET', payload: { sid: session.session_id, queryText: '', tab: 'progress' } });
      startSSE(session.query, session.session_id);
    }
  }, [s.running, s.sid, s.stats, addUserMsg, addBotMsg, toast]);

  var confirmDel = useCallback(function(session) {
    dispatch({ type: 'SET', payload: { confirm: { sid: session.session_id, query: session.query } } });
  }, []);

  var doDelete = useCallback(async function(sid) {
    dispatch({ type: 'SET', payload: { confirm: null } });
    try {
      var r = await fetch('/api/sessions/' + sid, { method: 'DELETE' });
      if (!r.ok) { toast('删除失败', 'err'); return; }
      toast('已删除', 'ok');
      var newTotal = s.histTotal - 1;
      var maxPage = Math.max(0, Math.ceil(newTotal / PAGE_SIZE) - 1);
      var page = Math.min(s.histPage, maxPage);
      dispatch({ type: 'SET', payload: {
        sid: s.sid === sid ? null : s.sid,
        messages: s.sid === sid ? [] : s.messages,
        stats: s.sid === sid ? Object.assign({}, s.stats, { runtime: '--' }) : s.stats,
      } });
      await loadHistory(page);
    } catch (e) { toast('删除失败: ' + e.message, 'err'); }
  }, [s.histTotal, s.histPage, s.sid, s.messages, s.stats, loadHistory, toast]);

  // ── SSE Research ─────────────────────────────────────────────────
  function startSSE(query, sid) {
    dispatch({ type: 'RESET_SIDEBAR' });
    addStatus('正在规划研究任务...');
    dispatch({ type: 'SET', payload: { running: true, sid: sid } });
    bufRef.current = '';
    msgIdxRef.current = -1;

    var es = new EventSource('/api/stream/' + sid);
    esRef.current = es;

    es.addEventListener('stage', function(e) {
      var d = JSON.parse(e.data);
      dispatch({ type: 'UPDATE_STAGE', payload: { stage: d.stage, label: d.label, progress: d.progress } });
    });

    es.addEventListener('token', function(e) {
      var d = JSON.parse(e.data);
      bufRef.current += (d.delta || '');
      if (msgIdxRef.current < 0) {
        dispatch({ type: 'POP_MSG_TYPE', payload: 'status' });
        dispatch({ type: 'PUSH_MSG', payload: { type: 'bot', role: 'bot', html: '', downloads: '' } });
        msgIdxRef.current = s.messages.length;
      }
      dispatch({ type: 'UPDATE_LAST_MSG', payload: { html: marked.parse(bufRef.current) } });
    });

    es.addEventListener('tool', function(e) {
      var d = JSON.parse(e.data);
      var cur = statsRef.current;
      dispatch({ type: 'SET', payload: { stats: Object.assign({}, cur, {
        toolCalls: cur.toolCalls + 1,
        toolSuccess: cur.toolSuccess + (d.success ? 1 : 0),
        toolFail: cur.toolFail + (d.success ? 0 : 1),
      })}});
      var label = TOOL_NAMES[d.tool_name] || d.tool_name;
      var meta = '';
      if (d.elapsed > 0) meta += d.elapsed.toFixed(1) + 's';
      if (d.tokens > 0) meta += (meta ? ' · ' : '') + d.tokens + ' tokens';
      dispatch({ type: 'PUSH_TOOL', payload: { ok: d.success, name: label, meta: meta } });
    });

    es.addEventListener('verify', function(e) {
      var d = JSON.parse(e.data);
      var cur = statsRef.current;
      dispatch({ type: 'SET', payload: { stats: Object.assign({}, cur, { verifications: cur.verifications + 1 }) }});
      dispatch({ type: 'PUSH_TOOL', payload: {
        ok: d.pass,
        name: '验证: ' + (d.task_id || '').slice(-8),
        meta: '评分 ' + Math.round((d.score || 0) * 100) + '%',
      }});
    });

    es.addEventListener('report_ready', function() {
      dispatch({ type: 'UPDATE_STAGE', payload: { stage: 'writing', label: '报告已生成', progress: 1 } });
    });

    es.addEventListener('done', function(e) {
      var d = JSON.parse(e.data);
      dispatch({ type: 'COMPLETE_STAGES' });
      var cur = statsRef.current;
      dispatch({ type: 'SET', payload: { stats: Object.assign({}, cur, { runtime: d.runtime_ms ? (d.runtime_ms / 1000).toFixed(1) + 's' : '' }) }});
      var finalText = bufRef.current || d.final_answer_text || '报告生成完毕。';
      var dl = downloadsHtml(sid);
      dispatch({ type: 'POP_MSG_TYPE', payload: 'status' });
      dispatch({ type: 'UPDATE_LAST_MSG', payload: { html: marked.parse(finalText), downloads: dl } });
      es.close();
      dispatch({ type: 'SET', payload: { running: false } });
      loadHistory();
    });

    es.addEventListener('sse_error', function(e) {
      var d = JSON.parse(e.data);
      dispatch({ type: 'POP_MSG_TYPE', payload: 'status' });
      dispatch({ type: 'PUSH_MSG', payload: { type: 'bot', role: 'bot', html: marked.parse('研究失败: ' + (d.error || '未知错误')), downloads: '' } });
      dispatch({ type: 'FAIL_STAGES' });
      es.close();
      dispatch({ type: 'SET', payload: { running: false } });
      loadHistory();
    });

    es.onerror = function() {
      if (es.readyState === EventSource.CLOSED) {
        dispatch({ type: 'POP_MSG_TYPE', payload: 'status' });
        dispatch({ type: 'FAIL_STAGES' });
        es.close();
        dispatch({ type: 'SET', payload: { running: false } });
        loadHistory();
      }
    };
  }

  useEffect(function() {
    return function() { if (esRef.current) esRef.current.close(); };
  }, []);

  // ── Send ─────────────────────────────────────────────────────────
  var send = useCallback(async function() {
    var q = s.queryText.trim();
    if (!q || s.running) return;
    dispatch({ type: 'SET', payload: { queryText: '' } });
    addUserMsg(q);
    dispatch({ type: 'SET', payload: { running: true } });
    addStatus('正在分析问题...');

    // 多轮对话：如果已有会话且已完成，作为追问发送
    var isFollowUp = s.sid && s.stats.runtime !== '--' && !s.running;
    var body = { message: q, mode: s.queryMode };
    if (isFollowUp) {
      body.session_id = s.sid;
      // 添加分隔提示
      addStatus('正在基于之前的上下文研究追问...');
    }

    try {
      var r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var d = await r.json();
      if (d.mode === 'chat') {
        dispatch({ type: 'POP_MSG_TYPE', payload: 'status' });
        addBotMsg(d.answer || '(空)');
        dispatch({ type: 'SET', payload: { running: false } });
      } else if (d.mode === 'agent') {
        dispatch({ type: 'SET', payload: { tab: 'progress' } });
        startSSE(q, d.session_id);
        // 多轮对话：链式显示上下文
        if (isFollowUp && d.parent_query) {
          // 在状态消息中显示这是追问
        }
      }
    } catch (e) {
      dispatch({ type: 'POP_MSG_TYPE', payload: 'status' });
      addBotMsg('请求失败: ' + e.message);
      dispatch({ type: 'SET', payload: { running: false } });
    }
  }, [s.queryText, s.queryMode, s.running, s.sid, s.stats.runtime, addUserMsg, addBotMsg, addStatus]);

  // 新对话 — 清除上下文，重新开始
  function newConversation() {
    if (s.running) { toast('当前研究进行中，请等待完成', 'info'); return; }
    dispatch({ type: 'SET', payload: {
      sid: null, messages: [], toolLog: [],
      stats: { toolCalls:0, toolSuccess:0, toolFail:0, verifications:0, runtime:'--' },
      stages: init.stages.map(function(x) { return Object.assign({}, x); }),
      sbDot: '',
    }});
    toast('已开启新对话', 'ok');
  }

  // ── Settings: API Keys ─────────────────────────────────────────
  async function loadApiKeys() {
    dispatch({ type: 'SET', payload: { settingsLoading: true } });
    try {
      var r = await fetch('/api/settings/keys');
      var d = await r.json();
      dispatch({ type: 'SET', payload: { apiKeys: d.keys || [], settingsLoading: false } });
    } catch(e) {
      dispatch({ type: 'SET', payload: { settingsLoading: false } });
      toast('加载API配置失败', 'err');
    }
  }
  function openSettings() { dispatch({ type: 'SET', payload: { showSettings: true } }); loadApiKeys(); }
  function closeSettings() { dispatch({ type: 'SET', payload: { showSettings: false } }); }

  async function saveApiKey(provider, keyElId) {
    var el = document.getElementById(keyElId);
    var key = el ? el.value.trim() : '';
    if (!key) { toast('请输入Key', 'err'); return; }
    try {
      var r = await fetch('/api/settings/keys', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({provider:provider, key:key}),
      });
      var d = await r.json();
      if (d.ok) { toast(d.masked + ' 已保存', 'ok'); if (el) el.value = ''; loadApiKeys(); }
      else { toast(d.error || '保存失败', 'err'); }
    } catch(e) { toast('保存失败: '+e.message, 'err'); }
  }

  async function deleteApiKey(provider) {
    try {
      var r = await fetch('/api/settings/keys/'+provider, {method:'DELETE'});
      var d = await r.json();
      if (d.ok) { toast('已删除', 'ok'); loadApiKeys(); }
      else { toast(d.error||'删除失败', 'err'); }
    } catch(e) { toast('删除失败: '+e.message, 'err'); }
  }

  // ── Badge ────────────────────────────────────────────────────────
  var badgeClass = s.running ? 'badge-run' : (s.stats.runtime !== '--' ? 'badge-done' : 'badge-idle');
  var badgeText = s.running ? '研究中...' : (s.stats.runtime !== '--' ? '完成' : '就绪');

  // ═══════════════════════════════════════════════════════════════════════
  // Render helpers
  // ═══════════════════════════════════════════════════════════════════════
  function rStage(st) {
    return html`
      <div key=${st.id} className=${'stage-item' + (st.active ? ' act' : '')}>
        <div className="stage-sig">
          <div className=${'stage-dot ' + st.dot}></div>
          ${!st.last && html`<div className=${'stage-line' + (st.lineDone ? ' done' : '')}></div>`}
        </div>
        <div className="stage-body">
          <div className="stage-name">${st.name}</div>
          <div className="stage-label">${st.label}</div>
          ${st.showBar && html`
            <div className="stage-bar-wrap">
              <div className="stage-bar-fill" style=${{ width: (st.prog * 100) + '%' }}></div>
            </div>`}
        </div>
      </div>`;
  }

  function rTool(t, i) {
    return html`
      <div key=${i} className=${'tool-e ' + (t.ok ? 'ok' : 'err')}>
        <span className="t-icon">${t.ok ? '✓' : '✗'}</span>
        <span className="t-name">${t.name}</span>
        ${t.meta && html`<span className="t-meta">${t.meta}</span>`}
      </div>`;
  }

  function rHistItem(hs) {
    return html`
      <div key=${hs.session_id}
           className=${'hist-item' + (hs.session_id === s.sid ? ' act' : '')}
           onClick=${function() { viewSession(hs); }}>
        <div className=${'hi-dot ' + hs.status}></div>
        <div className="hi-body">
          <div className="hi-query" title=${hs.query}>${hs.query}</div>
          <div className="hi-meta">
            <span className=${'hi-tag ' + hs.status}>${STATUS_LABELS[hs.status] || hs.status}</span>
            ${hs.runtime_ms > 0 && html`<span>${(hs.runtime_ms / 1000).toFixed(1)}s</span>`}
            <span>${timeAgo(hs.created_at)}</span>
          </div>
        </div>
        <button className="hi-del" onClick=${function(e) { e.stopPropagation(); confirmDel(hs); }} title="删除">×</button>
      </div>`;
  }

  function rMsg(m, i) {
    if (m.type === 'status') {
      return html`<div key=${i} className="status-row"><div className="spinner"></div><span>${m.text}</span></div>`;
    }
    return html`
      <div key=${i} className=${'msg ' + (m.role === 'user' ? 'msg-user' : 'msg-bot')}>
        <div className="bubble" dangerouslySetInnerHTML=${{ __html: m.html }}></div>
        ${m.downloads && html`<div className="dl-row" dangerouslySetInnerHTML=${{ __html: m.downloads }}></div>`}
      </div>`;
  }

  function rPager() {
    var total = Math.max(1, Math.ceil(s.histTotal / PAGE_SIZE));
    if (total <= 1) return null;
    var pages = [];
    var start = Math.max(0, s.histPage - 2);
    var end = Math.min(total, start + 5);
    for (var i = start; i < end; i++) pages.push(i);
    return html`
      <div className="hist-pager">
        <button onClick=${function() { loadHistory(s.histPage - 1); }} disabled=${s.histPage <= 0}>&#8249;</button>
        ${pages.map(function(p) {
          return html`<button key=${p} className=${p === s.histPage ? 'on' : ''} onClick=${function() { loadHistory(p); }}>${p + 1}</button>`;
        })}
        <button onClick=${function() { loadHistory(s.histPage + 1); }} disabled=${s.histPage >= total - 1}>&#8250;</button>
      </div>`;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Main Render
  // ═══════════════════════════════════════════════════════════════════════
  return html`
    <div className="app-shell">

      <div className="toast-ctr">
        ${s.toasts.map(function(t, i) {
          return html`<div key=${t.id} className=${'toast ' + t.type} onClick=${function() { dispatch({ type: 'POP_TOAST', payload: i }); }}>${t.msg}</div>`;
        })}
      </div>

      ${s.confirm && html`
        <div className="dlg-overlay" onClick=${function() { dispatch({ type: 'SET', payload: { confirm: null } }); }}>
          <div className="dlg-box" onClick=${function(e) { e.stopPropagation(); }}>
            <h3>删除会话</h3>
            <p>确定删除「${esc(s.confirm.query.slice(0, 60))}${s.confirm.query.length > 60 ? '...' : ''}」？<br/>报告文件也会一并删除，此操作不可撤销。</p>
            <div className="dlg-acts">
              <button className="btn-c" onClick=${function() { dispatch({ type: 'SET', payload: { confirm: null } }); }}>取消</button>
              <button className="btn-d" onClick=${function() { doDelete(s.confirm.sid); }}>确认删除</button>
            </div>
          </div>
        </div>`}

      ${s.showSettings && html`
        <div className="dlg-overlay" onClick=${function(e) { if (e.target.className==='dlg-overlay') closeSettings(); }}>
          <div className="dlg-box" style=${{maxWidth:'560px'}} onClick=${function(e) { e.stopPropagation(); }}>
            <h3>API Key 配置</h3>
            <p style=${{marginBottom:'14px'}}>配置各提供商的 API Key 后即可使用真实 LLM 和搜索引擎。Key 保存在 .env 文件中。</p>
            ${s.settingsLoading ? html`<div className="skel" style=${{height:'200px'}}></div>` : html`
              <div style=${{maxHeight:'400px',overflowY:'auto'}}>
                ${s.apiKeys.map(function(ak) {
                  var inputId = 'keyInput_' + ak.provider;
                  return html`<div key=${ak.provider} style=${{display:'flex',alignItems:'center',gap:'8px',padding:'8px 0',borderBottom:'1px solid var(--border-subtle)'}}>
                    <div style=${{flex:1,minWidth:0}}>
                      <div style=${{fontSize:'12px',fontWeight:600,color:'var(--text-primary)'}}>${ak.label}</div>
                      <div style=${{fontSize:'10px',color:'var(--text-muted)',marginTop:'2px'}}>
                        ${ak.configured ? html`<span style=${{color:'var(--success)'}}>已配置: ${ak.masked}</span>` : html`<span style=${{color:'var(--fail)'}}>未配置</span>`}
                      </div>
                    </div>
                    <input id=${inputId} type="password" placeholder="输入新Key..."
                           style=${{width:'160px',padding:'5px 8px',fontSize:'11px',background:'var(--bg-input)',border:'1px solid var(--border-normal)',borderRadius:'4px',color:'var(--text-primary)',outline:'none'}} />
                    <button onClick=${function() { saveApiKey(ak.provider, inputId); }}
                            style=${{padding:'4px 10px',fontSize:'10px',borderRadius:'4px',border:'1px solid var(--accent)',background:'var(--accent-soft)',color:'var(--accent)',cursor:'pointer',fontWeight:600,whiteSpace:'nowrap'}}>保存</button>
                    ${ak.configured && html`<button onClick=${function() { deleteApiKey(ak.provider); }}
                            style=${{padding:'4px 8px',fontSize:'10px',borderRadius:'4px',border:'1px solid var(--fail)',background:'transparent',color:'var(--fail)',cursor:'pointer'}}>删除</button>`}
                  </div>`;
                })}
              </div>`}
            <div className="dlg-acts" style=${{marginTop:'14px'}}>
              <button className="btn-c" onClick=${closeSettings}>关闭</button>
            </div>
          </div>
        </div>`}

      <aside className="sidebar">
        <div className="sidebar-head">
          <h2>${s.tab === 'progress' ? '研究进度' : '历史会话'}</h2>
          <div className=${'sb-dot ' + s.sbDot}></div>
        </div>

        <div className="sidebar-tabs">
          <button className=${'sidebar-tab' + (s.tab === 'progress' ? ' act' : '')}
                  onClick=${function() { dispatch({ type: 'SET', payload: { tab: 'progress' } }); }}>当前进度</button>
          <button className=${'sidebar-tab' + (s.tab === 'history' ? ' act' : '')}
                  onClick=${function() { dispatch({ type: 'SET', payload: { tab: 'history' } }); loadHistory(); }}>历史会话</button>
        </div>

        ${s.tab === 'progress' && html`
          <div className="sidebar-panel">
            <div className="stage-list">${s.stages.map(function(st) { return rStage(st); })}</div>
            <div className="tool-log">
              <div className="tool-log-title">工具调用日志</div>
              ${s.toolLog.length === 0
                ? html`<div style=${{ padding: '14px', textAlign: 'center', fontSize: '10px', color: 'var(--text-muted)' }}>暂无工具调用</div>`
                : s.toolLog.map(function(t, i) { return rTool(t, i); })}
            </div>
          </div>`}

        ${s.tab === 'history' && html`
          <div className="sidebar-panel">
            <div className="hist-list">
              ${s.histLoading
                ? Array.from({ length: 5 }, function(_, i) { return html`<div key=${i} className="skel" style=${{ height: '48px', marginBottom: '4px' }}></div>`; })
                : s.sessions.length === 0
                  ? html`<div className="hist-empty">
                      <div className="he-icon">~</div>
                      <div className="he-text">暂无历史会话</div>
                      <div className="he-text" style=${{ fontSize: '9px' }}>发起深度研究后自动出现在这里</div>
                    </div>`
                  : html`${s.sessions.map(function(hs) { return rHistItem(hs); })}${rPager()}`}
            </div>
          </div>`}
      </aside>

      <div className="main-area">
        <header className="header">
          <div className="header-brand">
            <div className="hb-icon">V</div>
            <h1>Horizon-Agent</h1>
          </div>
          <div style=${{display:'flex',alignItems:'center',gap:'8px'}}>
            <button className="theme-toggle" onClick=${openSettings} title="API Key 配置">⚙</button>
            <button className="theme-toggle" onClick=${toggleTheme}
                    title=${theme==='dark'?'切换亮色主题':'切换暗色主题'}>
              ${theme==='dark'?'☀':' \u{1F319}'}
            </button>
            ${(s.sid && s.stats.runtime !== '--') && html`
              <button className="btn-new-chat" onClick=${newConversation} title="开启新对话">
                新对话
              </button>`}
            <div className=${'badge ' + badgeClass}>${badgeText}</div>
          </div>
        </header>

        <div className="msg-area" ref=${msgRef}>
          <div className="msg-area-inner">
            ${s.messages.length === 0
              ? html`
                <div className="welcome">
                  <div className="w-icon">~</div>
                  <h2>溯证智搜 · 多 Agent 协同研究</h2>
                  <p>多 Agent 协同搜索网络和学术论文、交叉验证信息来源、撰写结构化报告。每个结论都可追溯到原始出处。</p>
                  <div style=${{marginTop:'12px',fontSize:'11px',color:'var(--text-muted)',marginBottom:'6px'}}>快捷提问</div>
                  <div className="w-hints">
                    ${HINTS.map(function(h) {
                      return html`<span key=${h} className="w-hint" onClick=${function() {
                        dispatch({ type: 'SET', payload: { queryText: h } });
                        setTimeout(function() { send(); }, 50);
                      }}>${h}</span>`;
                    })}
                  </div>
                  <div style=${{marginTop:'16px',fontSize:'11px',color:'var(--text-muted)',marginBottom:'6px'}}>研究模板 (点击后替换 {topic})</div>
                  <div className="w-hints">
                    ${TEMPLATES.map(function(t) {
                      return html`<span key=${t.label} className="w-hint" onClick=${function() {
                        dispatch({ type: 'SET', payload: { queryText: t.prompt.replace('{topic}', '') } });
                        // Focus input so user can type their topic
                        setTimeout(function() {
                          var inp = document.querySelector('.input-row input');
                          if (inp) { inp.focus(); inp.setSelectionRange(t.prompt.indexOf('{topic}'), t.prompt.indexOf('{topic}') + 7); }
                        }, 100);
                      }}>${t.icon} ${t.label}</span>`;
                    })}
                  </div>
                </div>`
              : s.messages.map(function(m, i) { return rMsg(m, i); })
            }
          </div>
        </div>

        <div className="input-bar">
          <div className="input-row">
            <input type="text" value=${s.queryText}
                   placeholder="输入你想研究的问题..."
                   maxLength="500"
                   onInput=${function(e) { dispatch({ type: 'SET', payload: { queryText: e.target.value } }); }}
                   onKeyDown=${function(e) { if (e.key === 'Enter') send(); }}
                   disabled=${s.running} />
            <select value=${s.queryMode}
                    onChange=${function(e) { dispatch({ type: 'SET', payload: { queryMode: e.target.value } }); }}
                    disabled=${s.running}>
              <option value="auto">自动</option>
              <option value="chat">对话</option>
              <option value="deep">深度研究</option>
            </select>
            <button className="btn-send" onClick=${send} disabled=${s.running || !s.queryText.trim()}>
              ${s.running ? '研究中...' : '发送'}
            </button>
          </div>
        </div>
      </div>

      <aside className="detail">
        <div className="sec">
          <h3>会话统计</h3>
          <div className="stat-row"><span className="sl">工具调用</span><span className="sv" style=${{ color: 'var(--accent)' }}>${s.stats.toolCalls}</span></div>
          <div className="stat-row"><span className="sl">成功</span><span className="sv" style=${{ color: 'var(--success)' }}>${s.stats.toolSuccess}</span></div>
          <div className="stat-row"><span className="sl">失败</span><span className="sv" style=${{ color: 'var(--fail)' }}>${s.stats.toolFail}</span></div>
          <div className="stat-row"><span className="sl">验证次数</span><span className="sv" style=${{ color: 'var(--accent)' }}>${s.stats.verifications}</span></div>
          <div className="stat-row"><span className="sl">运行时间</span><span className="sv">${s.stats.runtime}</span></div>
        </div>
        <div className="sec">
          <h3>可用工具</h3>
          <div className="stat-row"><span className="sl">网页搜索</span><span className="sv" style=${{fontSize:'10px',color:'var(--text-muted)'}}>5 后端竞速</span></div>
          <div className="stat-row"><span className="sl">学术论文</span><span className="sv" style=${{fontSize:'10px',color:'var(--text-muted)'}}>OpenAlex + S2 + Arxiv</span></div>
          <div className="stat-row"><span className="sl">代码执行</span><span className="sv" style=${{fontSize:'10px',color:'var(--text-muted)'}}>AST 安全沙箱</span></div>
          <div className="stat-row"><span className="sl">知识检索</span><span className="sv" style=${{fontSize:'10px',color:'var(--text-muted)'}}>L3 向量搜索</span></div>
        </div>
        <div className="sec">
          <h3>系统信息</h3>
          <div className="stat-row"><span className="sl">版本</span><span className="sv">v${VERSION}</span></div>
          <div className="stat-row"><span className="sl">引擎</span><span className="sv">FastAPI + SSE</span></div>
          <div className="stat-row"><span className="sl">模型</span><span className="sv">DeepSeek</span></div>
          <div className="stat-row"><span className="sl">验证</span><span className="sv">9 规则 Hybrid</span></div>
        </div>
      </aside>

    </div>`;
}

// ── Mount with error boundary ──────────────────────────────────────────
step('Step 5/6: Mount...');
var rootEl = document.getElementById('root');
if (!rootEl) throw new Error('#root element not found');

var root = ReactDOM.createRoot(rootEl);
root.render(html`<${App} />`);
step('Step 6/6: Done!');
console.log('[Horizon-Agent] App mounted successfully');

} catch(e) {
  console.error('[HorizonRL] Failed:', e);
  if (diag) diag.textContent = 'FAIL: ' + e.message;
  if (rootEl) {
    rootEl.innerHTML = '<div style="color:#f85149;background:#0a0e14;padding:24px;font-family:sans-serif;min-height:100vh">'
      + '<h2 style="font-size:18px;margin-bottom:8px">应用加载失败</h2>'
      + '<p style="color:#8b949e;margin-bottom:12px">' + e.message + '</p>'
      + '<pre style="color:#4a5568;font-size:12px;white-space:pre-wrap;line-height:1.6">' + (e.stack||'').replace(/</g,'&lt;') + '</pre>'
      + '<p style="color:var(--text-muted);margin-top:16px;font-size:11px">请刷新页面重试，或检查浏览器控制台 (F12) 获取更多信息</p>'
      + '</div>';
  }
}
})();
