/**
 * Markdown 渲染器 — 加载 marked.js CDN + 自定义配置。
 *
 * 用法:
 *   await MarkdownRenderer.init();
 *   const html = MarkdownRenderer.render(mdText);
 */
const MarkdownRenderer = (() => {
  let ready = false;

  async function init() {
    if (ready) return;
    if (typeof marked !== 'undefined') {
      configureMarked();
      ready = true;
      return;
    }

    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js';
      script.onload = () => {
        configureMarked();
        ready = true;
        resolve();
      };
      script.onerror = () => {
        // Fallback: use simple renderer
        console.warn('marked.js CDN 加载失败，使用简易渲染器');
        ready = true;
        resolve();
      };
      document.head.appendChild(script);
    });
  }

  function configureMarked() {
    if (typeof marked === 'undefined') return;

    marked.setOptions({
      gfm: true,
      breaks: false,
    });

    // Custom renderer for better styling hooks
    const renderer = new marked.Renderer();

    renderer.code = function({ text, lang }) {
      const langAttr = lang ? ` data-lang="${lang}"` : '';
      return `<pre${langAttr}><code>${escapeHtml(text)}</code></pre>`;
    };

    renderer.table = function({ header, rows }) {
      let html = '<table>';
      if (header && header.length) {
        html += '<thead><tr>';
        header.forEach(cell => { html += `<th>${cell}</th>`; });
        html += '</tr></thead>';
      }
      if (rows && rows.length) {
        html += '<tbody>';
        rows.forEach(row => {
          html += '<tr>';
          row.forEach(cell => { html += `<td>${cell}</td>`; });
          html += '</tr>';
        });
        html += '</tbody>';
      }
      html += '</table>';
      return html;
    };

    marked.use({ renderer });
  }

  function render(md) {
    if (!md) return '';

    if (typeof marked !== 'undefined') {
      try {
        return marked.parse(md);
      } catch (_) {
        return simpleRender(md);
      }
    }
    return simpleRender(md);
  }

  /** 简易渲染器 — marked.js 不可用时的回退 */
  function simpleRender(md) {
    let h = escapeHtml(md);

    // Code blocks
    h = h.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // Inline code
    h = h.replace(/`(.+?)`/g, '<code>$1</code>');
    // Headers
    h = h.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    // Bold / italic
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Links
    h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    // Blockquote
    h = h.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
    // Horizontal rule
    h = h.replace(/^---$/gm, '<hr>');
    // Lists
    h = h.replace(/(^(\d+)\. .+$\n?)+/gm, m => {
      const items = m.trim().split('\n').map(s => '<li>' + s.replace(/^\d+\. /, '') + '</li>').join('');
      return '<ol>' + items + '</ol>';
    });
    h = h.replace(/(^[-*] .+$\n?)+/gm, m => {
      const items = m.trim().split('\n').map(s => '<li>' + s.replace(/^[-*] /, '') + '</li>').join('');
      return '<ul>' + items + '</ul>';
    });
    // Tables
    h = h.replace(/(^\|.+\|\n\|[-| ]+\|\n((^\|.+\|\n?)*))/gm, m => {
      const rows = m.trim().split('\n').filter(r => r.includes('|') && !r.match(/^[\|\- :]+$/));
      let html = '<table>';
      rows.forEach((row, i) => {
        const cells = row.split('|').filter(c => c.trim()).map(c => i === 0 ? '<th>' + c.trim() + '</th>' : '<td>' + c.trim() + '</td>').join('');
        html += '<tr>' + cells + '</tr>';
      });
      return html + '</table>';
    });
    // Paragraphs
    h = h.replace(/\n\n/g, '</p><p>');
    h = h.replace(/\n/g, '<br>');
    return '<p>' + h + '</p>';
  }

  function escapeHtml(s) {
    return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  return { init, render };
})();

// 暴露 escapeHtml 供外部使用
function escapeHtml(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
