/**
 * SSE 客户端封装 — EventSource 管理、自动重连、事件分发。
 *
 * 用法:
 *   const client = new SSEClient('/api/stream/session_abc123', {
 *     onStage: (data) => { ... },
 *     onToken: (data) => { ... },
 *     onDone: (data) => { ... },
 *     onError: (err) => { ... },
 *   });
 *   client.connect();
 */
class SSEClient {
  constructor(url, handlers = {}) {
    this.url = url;
    this.handlers = handlers;
    this.es = null;
    this.connected = false;
    this.finished = false;  // 标记会话是否已完成（阻止重连）
    this.reconnectAttempts = 0;
    this.maxReconnects = 3;
  }

  connect() {
    // 已完成则不再连接
    if (this.finished) return;

    if (this.es) this.disconnect();

    this.es = new EventSource(this.url);

    // 注册所有标准事件
    const events = ['stage', 'token', 'tool', 'verify', 'report_ready', 'done', 'sse_error'];
    events.forEach(name => {
      this.es.addEventListener(name, (e) => {
        try {
          const data = JSON.parse(e.data);
          // 事件名 → handler 名: stage→onStage, report_ready→onReportReady, sse_error→onSseError
          const handlerName = 'on' + name.charAt(0).toUpperCase() + name.slice(1).replace(/_./g, x => x[1].toUpperCase());
          const handler = this.handlers[handlerName];
          if (handler) handler(data);
        } catch (_) { /* JSON parse error - ignore malformed events */ }
      });
    });

    // 心跳保活
    this.es.addEventListener('heartbeat', () => { /* keepalive, no-op */ });

    this.es.onopen = () => {
      this.connected = true;
      this.reconnectAttempts = 0;
    };

    this.es.onerror = (evt) => {
      // 已完成 → 不重连（done 事件后服务端正常关闭连接）
      if (this.finished) return;

      if (this.es && this.es.readyState === EventSource.CLOSED) {
        this.connected = false;
        if (this.reconnectAttempts < this.maxReconnects) {
          this.reconnectAttempts++;
          setTimeout(() => this.connect(), 1000 * this.reconnectAttempts);
        } else if (this.handlers.onDisconnect) {
          this.handlers.onDisconnect('SSE 连接中断，请刷新重试');
        }
      }
    };
  }

  /** 标记会话完成，关闭连接且不再重连。 */
  markFinished() {
    this.finished = true;
    this.disconnect();
  }

  disconnect() {
    if (this.es) {
      this.es.close();
      this.es = null;
    }
    this.connected = false;
  }
}
