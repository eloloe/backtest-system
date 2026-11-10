import { applyWindow, rebaseSeries, replayFixedTrades } from "./replay_runtime.js";

async function loadData() {
  const resp = await fetch("./data.json");
  if (!resp.ok) throw new Error("讀不到 data.json");
  return resp.json();
}

function fmtPct(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(2)}%`;
}

function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString("zh-TW", { maximumFractionDigits: 2 });
}

function drawCharts(state) {
  const baseEq = rebaseSeries(applyWindow(state.original.equity, state.start, state.end));
  const eq = rebaseSeries(applyWindow(state.preview.equity, state.start, state.end));
  const dd = applyWindow(state.preview.drawdown, state.start, state.end);
  const bench = rebaseSeries(applyWindow(state.benchmark, state.start, state.end));
  Plotly.newPlot(
    "equityChart",
    [
      { x: baseEq.map((p) => p.date), y: baseEq.map((p) => p.value), type: "scatter", name: "原始回測（重播）", line: { color: "#475569", dash: "dot" } },
      { x: eq.map((p) => p.date), y: eq.map((p) => p.value), type: "scatter", name: "目前控制值（重播）", line: { color: "#60a5fa" } },
      { x: bench.map((p) => p.date), y: bench.map((p) => p.value), type: "scatter", name: "基準（同區間 rebase）", line: { color: "#f59e0b", dash: "dot" } },
    ],
    {
      template: "plotly_dark",
      title: "淨值（區間起點=1.0，避免跨窗比較錯位）",
      yaxis: { title: "倍數" },
      margin: { t: 36, b: 36, l: 36, r: 18 },
    },
    { responsive: true }
  );
  Plotly.newPlot(
    "drawdownChart",
    [{ x: dd.map((p) => p.date), y: dd.map((p) => p.value), type: "scatter", fill: "tozeroy", name: "回撤", line: { color: "#f87171" } }],
    { template: "plotly_dark", title: "回撤", margin: { t: 36, b: 36, l: 36, r: 18 } },
    { responsive: true }
  );
}

function renderProvenance(data) {
  document.getElementById("title").textContent = data.meta.title;
  document.getElementById("subtitle").textContent = data.meta.subtitle;
  const root = document.getElementById("provenance");
  root.innerHTML = "";
  const rows = [
    ["標的", data.provenance.symbols.join(", ")],
    ["區間", `${data.provenance.start} ~ ${data.provenance.end}`],
    ["基準", data.provenance.benchmark || "無"],
    ["資料來源", data.provenance.provider || "未註明"],
    ["試驗數", String(data.provenance.trial_count)],
    ["驗證狀態", data.provenance.validation_label],
    ["資料型態", data.provenance.data_label],
    ["即時連線", data.provenance.live_connectivity_verified ? "已驗證" : "未驗證"],
  ];
  for (const [k, v] of rows) {
    const dl = document.createElement("dl");
    dl.className = "item";
    dl.innerHTML = `<dt>${k}</dt><dd>${v}</dd>`;
    root.appendChild(dl);
  }
}

function renderFindings(data) {
  const list = document.getElementById("findingList");
  const detail = document.getElementById("findingDetail");
  const baseline = document.getElementById("findingScope");
  baseline.textContent = "以下 finding 來自原始基準回測，不隨成本滑桿即時改寫；滑桿影響請看上方『目前控制值（重播）』。";
  list.innerHTML = "";
  data.diagnostics.findings.forEach((f, idx) => {
    const btn = document.createElement("button");
    btn.className = `finding-btn ${f.status}`;
    btn.textContent = `${f.status.toUpperCase()}｜${f.title}`;
    btn.onclick = () => {
      detail.textContent =
        `${f.question}\n\n` +
        `等級：${f.severity}\n` +
        `判定：${f.status}\n\n` +
        `為什麼重要：\n${f.why}\n\n` +
        `證據：\n${JSON.stringify(f.evidence, null, 2)}\n\n` +
        `下一步：\n${f.next_step}`;
    };
    list.appendChild(btn);
    if (idx === 0) btn.click();
  });
}

function renderShioaji(data) {
  const root = document.getElementById("shioajiPanel");
  if (!data.shioaji) {
    root.textContent = "這份示範沒有附 Shioaji fixture。";
    return;
  }
  const summary = data.shioaji.summary;
  let html = `<p class="muted">${summary.note}</p>`;
  html += `<p>快照數 ${summary.snapshot_count}，假設滑價 ${summary.assumed_slippage_bps.toFixed(2)} bps，價差違反 ${summary.spread_violations} 筆。</p>`;
  html += "<table><thead><tr><th>symbol</th><th>time</th><th>bid</th><th>ask</th><th>spread(bps)</th><th>lot</th><th>判定</th></tr></thead><tbody>";
  data.shioaji.checks.forEach((row) => {
    html += `<tr><td>${row.symbol}</td><td>${row.timestamp}</td><td>${fmtNum(row.bid)}</td><td>${fmtNum(row.ask)}</td><td>${row.spread_bps.toFixed(2)}</td><td>${row.lot_size}</td><td>${row.within_spread ? "通過" : "超出"}</td></tr>`;
  });
  html += "</tbody></table>";
  root.innerHTML = html;
}

function stats(points) {
  if (!points.length) return { totalReturn: null, mdd: null };
  const start = points[0].value;
  const end = points[points.length - 1].value;
  let peak = start;
  let mdd = 0;
  for (const p of points) {
    peak = Math.max(peak, p.value);
    mdd = Math.min(mdd, p.value / peak - 1);
  }
  return { totalReturn: end / start - 1, mdd };
}

function renderPreviewKpis(state) {
  const seg = applyWindow(state.preview.equity, state.start, state.end);
  const s = stats(seg);
  const root = document.getElementById("previewKpis");
  root.innerHTML = `
    <div class="kpi"><div class="muted">區間報酬</div><strong>${fmtPct(s.totalReturn)}</strong></div>
    <div class="kpi"><div class="muted">區間最大回撤</div><strong>${fmtPct(s.mdd)}</strong></div>
    <div class="kpi"><div class="muted">重播成交筆數</div><strong>${state.preview.trades_applied}</strong></div>
    <div class="kpi"><div class="muted">拒單筆數（現金/部位不足）</div><strong>${state.preview.trades_denied}</strong></div>
    <div class="kpi"><div class="muted">重播問題數</div><strong>${state.preview.problems.length}</strong></div>
  `;
  const issue = document.getElementById("previewIssue");
  issue.textContent = state.preview.problems.length
    ? `重播異常：${state.preview.problems.slice(0, 2).join("；")}${state.preview.problems.length > 2 ? "…" : ""}`
    : "";
}

function setupControls(data, state) {
  const start = document.getElementById("startDate");
  const end = document.getElementById("endDate");
  const comm = document.getElementById("commissionPct");
  const slip = document.getElementById("slippagePct");
  const btn = document.getElementById("replayBtn");
  const minDate = data.series.calendar[0];
  const maxDate = data.series.calendar[data.series.calendar.length - 1];
  start.min = minDate;
  start.max = maxDate;
  end.min = minDate;
  end.max = maxDate;
  start.value = state.start;
  end.value = state.end;
  comm.value = (data.meta.base_commission_rate * 100).toFixed(4);
  slip.value = (data.meta.base_slippage_rate * 100).toFixed(4);

  btn.onclick = () => {
    state.start = start.value || minDate;
    state.end = end.value || maxDate;
    if (state.start > state.end) {
      document.getElementById("previewIssue").textContent = "日期區間錯誤：起日不可晚於訖日。";
      return;
    }
    const c = Math.max(0, Number(comm.value || 0) / 100);
    const s = Math.max(0, Number(slip.value || 0) / 100);
    state.preview = replayFixedTrades(data, c, s);
    drawCharts(state);
    renderPreviewKpis(state);
  };
}

loadData()
  .then((data) => {
    renderProvenance(data);
    renderFindings(data);
    renderShioaji(data);
    const state = {
      start: data.series.calendar[0],
      end: data.series.calendar[data.series.calendar.length - 1],
      benchmark: data.series.benchmark,
      original: data.series.base_preview,
      preview: data.series.base_preview,
    };
    setupControls(data, state);
    drawCharts(state);
    renderPreviewKpis(state);
  })
  .catch((err) => {
    document.body.innerHTML = `<main class="wrap"><section class="card"><h1>載入失敗</h1><p>${err.message}</p></section></main>`;
  });
