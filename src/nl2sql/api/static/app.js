'use strict';

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const EXAMPLES = [
  '2025 年 6 月华东区的有效销售额是多少？',
  '各渠道的销售额分别是多少？',
  '销售额最高的 5 个客户是哪几个？',
  '各品类的销售额，从高到低',
  '2025 年 12 月的销售额环比增长率是多少？',
  '各收货大区的销售额占比',
  '数据里最新的一个月是哪个月？它的销售额是多少？',
  '在售商品的动销率是多少？',
];

const ATTACKS = [
  'DROP TABLE sales_orders',
  'dRoP /* 注释绕过 */ TABLE sales_orders',
  "SELECT 1; DROP TABLE sales_orders",
  "SELECT * FROM read_csv('/etc/passwd')",
  "COPY (SELECT * FROM sales_orders) TO '/tmp/leak.csv'",
  "ATTACH 'other.db' AS other",
  'SELECT * FROM sales_orders LIMIT 5',
  'SELECT a.order_id FROM sales_orders a JOIN customers b ON 1=1 LIMIT 5',
  'SELECT * FROM duckdb_tables()',
  'SELECT * FROM information_schema.tables',
  'SELECT order_id FROM sales_orders',
  '-- 正常查询，应当通过 --\nSELECT ship_region, ROUND(SUM(order_amount),2) AS amt\nFROM v_order_fact WHERE status IN (\'paid\',\'shipped\',\'completed\')\nGROUP BY 1 ORDER BY amt DESC',
];

// ─────────────────────────────────────────────
// Tab 切换
// ─────────────────────────────────────────────
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $('#tab-' + tab.dataset.tab).classList.add('active');
  });
});

// ─────────────────────────────────────────────
// 初始化
// ─────────────────────────────────────────────
async function init() {
  const chips = $('#examples');
  EXAMPLES.forEach((q) => {
    const c = el('button', 'chip', q);
    c.addEventListener('click', () => {
      $('#question').value = q;
      ask();
    });
    chips.appendChild(c);
  });

  const list = $('#attacks');
  ATTACKS.forEach((sql) => {
    const li = el('li', '', sql.replace(/\n/g, ' ⏎ '));
    li.title = sql;
    li.addEventListener('click', () => {
      $('#raw-sql').value = sql;
      runRawSQL();
    });
    list.appendChild(li);
  });

  await refreshHealth();
  loadStats();
  loadSchema();
}

async function refreshHealth() {
  const badges = $('#badges');
  try {
    const h = await (await fetch('/health')).json();
    badges.innerHTML = '';
    if (h.rows) {
      badges.appendChild(el('span', 'badge ok', `订单 ${fmt(h.rows[0])}`));
      badges.appendChild(el('span', 'badge ok', `客户 ${fmt(h.rows[1])}`));
    }
    const llm = h.llm || {};
    badges.appendChild(el('span', llm.ready ? 'badge ok' : 'badge err',
      llm.ready ? `模型 ${llm.model || ''}` : 'LLM 未配置（提问不可用）'));
    if (llm.error) console.warn('LLM:', llm.error);
  } catch (e) {
    badges.innerHTML = '';
    badges.appendChild(el('span', 'badge err', '服务未就绪'));
  }
}

// ─────────────────────────────────────────────
// 提问
// ─────────────────────────────────────────────
async function ask() {
  const question = $('#question').value.trim();
  if (!question) return;
  const btn = $('#ask-btn');
  btn.disabled = true;
  btn.textContent = '分析中…';
  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, with_answer: $('#with-answer').checked }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || data.error || resp.statusText);
    renderResult(data);
  } catch (e) {
    const box = $('#ask-result');
    box.classList.remove('hidden');
    const v = $('#verdict');
    v.className = 'verdict err';
    v.textContent = '请求失败：' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = '分析';
  }
}

function renderResult(d) {
  $('#ask-result').classList.remove('hidden');

  const v = $('#verdict');
  if (d.ok) {
    v.className = 'verdict ok';
    v.textContent = `✓ 执行成功 · 返回 ${d.row_count} 行 · 用时 ${Math.round(d.total_ms)} ms · 调用 ${d.attempts} 次`
      + (d.repairs ? ` · 自我修复 ${d.repairs} 次` : '');
  } else {
    v.className = 'verdict err';
    const stage = { guard: '安全校验', execute: '执行', generate: '生成' }[d.error_stage] || d.error_stage;
    v.textContent = `✗ ${stage}阶段失败：${d.error}`;
  }

  $('#out-sql').textContent = d.sql || '（未生成 SQL）';

  const table = $('#out-table');
  table.innerHTML = '';
  if (d.rows && d.rows.length) {
    const thead = el('thead');
    const tr = el('tr');
    d.columns.forEach((c) => tr.appendChild(el('th', '', c)));
    thead.appendChild(tr);
    table.appendChild(thead);
    const tbody = el('tbody');
    d.rows.slice(0, 200).forEach((row) => {
      const r = el('tr');
      row.forEach((cell, i) => {
        const num = typeof cell === 'number';
        r.appendChild(el('td', num ? 'num' : '', fmt(cell)));
      });
      tbody.appendChild(r);
    });
    table.appendChild(tbody);
  }
  $('#out-meta').textContent = d.truncated ? '（已截断）' : '';

  const answerBlock = $('#answer-block');
  if (d.answer) {
    answerBlock.style.display = '';
    $('#out-answer').textContent = d.answer;
    const tag = $('#grounding-tag');
    const g = (d.steps || []).find((s) => s.stage === 'grounding');
    if (g) {
      tag.className = 'tag ' + (g.ok ? 'ok' : 'err');
      tag.textContent = g.ok ? '数字已核对' : '含无法核实的数字';
      tag.title = g.detail || '';
    } else {
      tag.className = 'tag';
      tag.textContent = '';
    }
  } else {
    answerBlock.style.display = 'none';
  }

  const trace = $('#out-trace');
  trace.innerHTML = '';
  (d.steps || []).forEach((s) => {
    const li = el('li');
    li.appendChild(el('span', 'stage', s.stage));
    li.appendChild(el('span', s.ok ? 'ok' : 'err', s.ok ? '通过' : '失败'));
    li.appendChild(el('span', 'ms', `${Math.round(s.elapsed_ms)} ms`));
    if (s.detail) li.appendChild(el('span', '', ' — ' + s.detail));
    trace.appendChild(li);
  });
}

// ─────────────────────────────────────────────
// 护栏演示
// ─────────────────────────────────────────────
async function runRawSQL() {
  const sql = $('#raw-sql').value.trim();
  if (!sql) return;
  const out = $('#guard-out');
  out.textContent = '执行中…';
  try {
    const resp = await fetch('/sql', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sql }),
    });
    const d = await resp.json();
    if (d.blocked) {
      const vs = (d.violations || [])
        .map((v) => `  • [${v.rule}] ${v.detail}\n    建议：${v.hint}`)
        .join('\n');
      out.textContent = `⛔ 已拦截（未执行）\n\n${vs}`;
      out.style.borderColor = 'var(--err)';
    } else if (!d.ok) {
      out.textContent = `❌ 执行失败：${d.error}`;
      out.style.borderColor = 'var(--warn)';
    } else {
      const head = d.columns.join(' | ');
      const body = d.rows.slice(0, 50)
        .map((r) => r.map(fmt).join(' | ')).join('\n');
      out.textContent = `✅ 放行并执行 · ${d.row_count} 行 · ${Math.round(d.elapsed_ms)} ms\n`
        + `使用表：${(d.tables_used || []).join(', ')}\n\n${head}\n${'-'.repeat(head.length)}\n${body}`;
      out.style.borderColor = 'var(--ok)';
    }
  } catch (e) {
    out.textContent = '请求失败：' + e.message;
  }
}

// ─────────────────────────────────────────────
// 数据集与口径
// ─────────────────────────────────────────────
async function loadStats() {
  try {
    const s = await (await fetch('/stats')).json();
    const box = $('#stats');
    box.innerHTML = '';
    Object.entries(s).forEach(([k, v]) => {
      if (k === 'channel_sales') return;
      const d = el('div', 'stat');
      d.appendChild(el('div', 'k', k));
      d.appendChild(el('div', 'v', typeof v === 'number' ? fmt(v) : String(v)));
      box.appendChild(d);
    });
    (s.channel_sales || []).forEach((c) => {
      const d = el('div', 'stat');
      d.appendChild(el('div', 'k', `渠道 · ${c.channel}`));
      d.appendChild(el('div', 'v', fmt(c.amount)));
      box.appendChild(d);
    });
  } catch (e) {
    console.warn('stats failed', e);
  }
}

async function loadSchema() {
  try {
    const s = await (await fetch('/schema')).json();
    const host = $('#tables');
    host.innerHTML = '';
    s.tables.forEach((t) => {
      const table = el('table', 'schema');
      const thead = el('thead');
      const tr = el('tr');
      tr.appendChild(el('th', '', t.name));
      tr.appendChild(el('th', '', '类型'));
      tr.appendChild(el('th', '', '业务口径说明'));
      thead.appendChild(tr);
      table.appendChild(thead);
      const tbody = el('tbody');
      t.columns.forEach((c) => {
        const r = el('tr');
        r.appendChild(el('td', '', c.name));
        r.appendChild(el('td', '', c.dtype));
        r.appendChild(el('td', '', c.comment));
        tbody.appendChild(r);
      });
      table.appendChild(tbody);
      host.appendChild(table);
    });

    const g = $('#glossary');
    g.innerHTML = '';
    s.glossary.forEach((e) => {
      const d = el('div', 'gloss');
      d.appendChild(el('div', 'term', e.term));
      d.appendChild(el('div', 'def', e.definition));
      const code = el('code', '', e.sql_hint);
      d.appendChild(code);
      g.appendChild(d);
    });
  } catch (e) {
    console.warn('schema failed', e);
  }
}

// ─────────────────────────────────────────────
// 工具
// ─────────────────────────────────────────────
function fmt(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return v.toLocaleString('zh-CN');
    return v.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
  }
  if (typeof v === 'string' && /^-?\d+(\.\d+)?$/.test(v)) {
    const n = Number(v);
    return Number.isInteger(n) ? n.toLocaleString('zh-CN')
      : n.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
  }
  return String(v);
}

$('#ask-btn').addEventListener('click', ask);
$('#question').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask();
});
$('#run-sql-btn').addEventListener('click', runRawSQL);

init();
