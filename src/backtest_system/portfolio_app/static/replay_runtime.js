export function replayFixedTrades(data, commissionRate, slippageRate) {
  const calendar = data.series.calendar;
  const prices = data.series.prices;
  const trades = data.series.trades;
  const positions = Object.create(null);
  const lastClose = Object.create(null);
  const byDate = new Map();
  for (const trade of trades) {
    const rows = byDate.get(trade.date) || [];
    rows.push(trade);
    byDate.set(trade.date, rows);
  }
  let cash = Number(data.meta.initial_capital || 0);
  let denied = 0;
  const problems = [];
  const equity = [];
  const drawdown = [];
  let peak = cash;
  let applied = 0;

  for (const day of calendar) {
    const rows = byDate.get(day) || [];
    for (const trade of rows) {
      const bar = prices[trade.symbol] ? prices[trade.symbol][day] : null;
      if (!bar) {
        problems.push(`${trade.symbol} ${day} missing Open/Close for trade replay`);
        continue;
      }
      const open = Number(bar.open);
      if (!Number.isFinite(open) || open <= 0) {
        problems.push(`${trade.symbol} ${day} invalid open price ${String(bar.open)}`);
        continue;
      }
      const fill = trade.action === "BUY" ? open * (1 + slippageRate) : open * (1 - slippageRate);
      const shares = Number(trade.shares);
      const notional = fill * shares;
      const marketText = String(trade.market || "").toUpperCase();
      const tw = marketText === "TW" || trade.symbol.endsWith(".TW") || trade.symbol.endsWith(".TWO");
      let fee = notional * commissionRate + Number(data.meta.commission_flat || 0);
      if (tw) {
        fee = Math.max(notional * Number(data.meta.tw_brokerage_rate), Number(data.meta.tw_brokerage_min));
        if (trade.action === "SELL") {
          fee += notional * Number(data.meta.tw_sell_tax_rate);
        }
      }
      if (trade.action === "BUY") {
        const needed = notional + fee;
        if (needed > cash + 1e-9) {
          denied += 1;
          problems.push(`${trade.symbol} ${day} buy denied: need ${needed.toFixed(4)}, cash ${cash.toFixed(4)}`);
          continue;
        }
        cash -= needed;
        positions[trade.symbol] = Number(positions[trade.symbol] || 0) + shares;
      } else {
        const held = Number(positions[trade.symbol] || 0);
        if (shares > held + 1e-9) {
          denied += 1;
          problems.push(`${trade.symbol} ${day} sell denied: shares ${shares.toFixed(6)} > held ${held.toFixed(6)}`);
          continue;
        }
        positions[trade.symbol] = held - shares;
        cash += notional - fee;
      }
      applied += 1;
    }

    let mark = cash;
    for (const [symbol, qty] of Object.entries(positions)) {
      if (Math.abs(qty) <= 1e-12) continue;
      const bar = prices[symbol] ? prices[symbol][day] : null;
      if (bar && Number.isFinite(Number(bar.close)) && Number(bar.close) > 0) {
        lastClose[symbol] = Number(bar.close);
      }
      if (lastClose[symbol] !== undefined) {
        mark += qty * Number(lastClose[symbol]);
      }
    }
    peak = Math.max(peak, mark);
    equity.push({ date: day, value: mark });
    drawdown.push({ date: day, value: mark / peak - 1 });
  }

  return {
    equity,
    drawdown,
    trades_applied: applied,
    trades_denied: denied,
    problems: Array.from(new Set(problems)),
  };
}

export function applyWindow(series, start, end) {
  if (!start || !end || start > end) {
    return [];
  }
  return series.filter((point) => point.date >= start && point.date <= end);
}

export function rebaseSeries(series) {
  if (!series.length) return [];
  const base = Number(series[0].value);
  if (!Number.isFinite(base) || base === 0) return series.map((point) => ({ ...point, value: null }));
  return series.map((point) => ({ date: point.date, value: Number(point.value) / base }));
}
