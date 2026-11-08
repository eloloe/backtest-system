"""Build a portable static mini app bundle from one research run."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from backtest_system.engine.costs import DEFAULT_TW_BROKERAGE_MIN, DEFAULT_TW_BROKERAGE_RATE, DEFAULT_TW_STOCK_SELL_TAX
from backtest_system.portfolio_app.diagnostics import default_replay_costs, diagnose_result, replay_preview
from backtest_system.portfolio_app.shioaji_fixture import check_execution_assumptions


def build_portfolio_bundle(
    result,
    *,
    output_dir: str | Path,
    title: str = "Leo｜回測診斷與假設檢查",
    subtitle: str = "證據導向：先看數字，再下結論",
    data_label: str = "歷史回測",
    validation_label: str = "樣本內",
    shioaji_fixture: str | None = None,
    shioaji_slippage_bps: float = 5.0,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    static_dir = Path(__file__).with_name("static")
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(static_dir / name, out / name)

    diagnostics = diagnose_result(result)
    preview = replay_preview(
        result,
        commission=float(result.config.commission),
        slippage=float(result.config.slippage),
    )
    payload = {
        "meta": {
            "title": title,
            "subtitle": subtitle,
            "initial_capital": float(result.config.initial_capital),
            "base_commission_rate": float(result.config.commission),
            "base_slippage_rate": float(result.config.slippage),
            "tw_brokerage_rate": float(DEFAULT_TW_BROKERAGE_RATE),
            "tw_brokerage_min": float(DEFAULT_TW_BROKERAGE_MIN),
            "tw_sell_tax_rate": float(DEFAULT_TW_STOCK_SELL_TAX),
            "preview_scope": "固定成交重播，不含訊號重算與再優化",
        },
        "provenance": _provenance(result, data_label=data_label, validation_label=validation_label),
        "diagnostics": diagnostics,
        "series": _series_pack(result, preview),
        "shioaji": (
            check_execution_assumptions(shioaji_fixture, assumed_slippage_bps=shioaji_slippage_bps)
            if shioaji_fixture
            else None
        ),
    }
    data_path = out / "data.json"
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = out / "README_INTEGRATION.md"
    readme.write_text(_integration_readme(), encoding="utf-8")
    return {
        "bundle_dir": out,
        "index_html": out / "index.html",
        "data_json": data_path,
        "integration_readme": readme,
    }


def _provenance(result, *, data_label: str, validation_label: str) -> dict:
    selection = getattr(result, "selection", None) or {}
    trial_count = int(selection.get("n_label") or 1) if isinstance(selection.get("n_label"), (int, float)) else 1
    return {
        "strategy": result.config.strategy,
        "symbols": list(result.symbols),
        "start": str(result.config.start),
        "end": str(result.config.end),
        "benchmark": result.config.benchmark,
        "provider": result.config.provider or "caller prices",
        "trial_count": trial_count,
        "validation_label": validation_label,
        "data_label": data_label,
        "sample_label": "sample" if "樣本" in validation_label else "historical",
        "live_connectivity_verified": False,
    }


def _series_pack(result, preview: Mapping) -> dict:
    calendar = [str(pd.Timestamp(day).date()) for day in result.equity.index]
    return {
        "calendar": calendar,
        "strategy_equity": _series(result.equity),
        "benchmark": _series(result.benchmark_equity) if result.benchmark_equity is not None else [],
        "strategy_drawdown": _series((result.equity / result.equity.cummax()) - 1.0),
        "base_preview": preview,
        "trades": _trades(result),
        "prices": _prices(result),
    }


def _series(series: pd.Series | None) -> list[dict]:
    if series is None:
        return []
    return [{"date": str(pd.Timestamp(day).date()), "value": float(value)} for day, value in series.items()]


def _trades(result) -> list[dict]:
    frame = result.trades.copy() if result.trades is not None else pd.DataFrame()
    if frame.empty:
        return []
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "symbol" not in frame.columns:
        fallback = result.symbols[0] if result.symbols else "UNKNOWN"
        frame["symbol"] = fallback
    if "market" not in frame.columns:
        frame["market"] = result.config.market or ""
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["action"] = frame["action"].astype(str).str.upper()
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
    out = frame[(frame["shares"] > 0) & frame["action"].isin({"BUY", "SELL"})]
    out = out.sort_values(["date", "symbol", "action"], kind="stable")
    rows = []
    for row in out.itertuples(index=False):
        rows.append(
            {
                "date": str(pd.Timestamp(row.date).date()),
                "symbol": row.symbol,
                "action": row.action,
                "shares": float(row.shares),
                "market": str(row.market or ""),
                "price": float(getattr(row, "price", 0.0)),
                "fee": float(getattr(row, "fee", 0.0)),
            }
        )
    return rows


def _prices(result) -> dict:
    packed = {}
    raw = getattr(result, "prices", None) or {}
    for symbol, frame in raw.items():
        if frame is None or frame.empty:
            continue
        safe = frame.copy()
        safe.index = pd.DatetimeIndex(safe.index).normalize()
        if "Open" not in safe.columns or "Close" not in safe.columns:
            continue
        packed[str(symbol).upper()] = {
            str(pd.Timestamp(day).date()): {"open": float(row["Open"]), "close": float(row["Close"])}
            for day, row in safe[["Open", "Close"]].iterrows()
        }
    return packed


def _integration_readme() -> str:
    return """# 回測診斷小 app（靜態嵌入）

這個資料夾可直接放到任何靜態網站（GitHub Pages、Netlify、個人作品集）。

## 檔案

- `index.html`：入口頁
- `styles.css` / `app.js`：前端程式
- `data.json`：這次回測的證據資料與可重播輸入

## 嵌入

### 1) 直接 iframe

```html
<iframe src="/portfolio-mini-app/index.html" width="100%" height="1600" style="border:0"></iframe>
```

### 2) 當成獨立頁

把整個資料夾部署在網站 `/portfolio-mini-app/` 路徑即可。

## 邊界說明

- 成本/滑價控制是「固定成交重播」，不是重新跑選股/訊號/參數搜尋。
- 若要完整重跑策略，請回到 CLI 用新設定再輸出一包。
- Shioaji 區塊只讀 fixture，沒有登入、下單或私鑰處理。
"""
