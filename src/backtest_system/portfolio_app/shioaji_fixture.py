"""Read-only Shioaji quote fixture checks for execution-assumption diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class QuoteCheck:
    symbol: str
    timestamp: str
    bid: float
    ask: float
    lot_size: int
    spread_bps: float
    assumed_slippage_bps: float
    within_spread: bool


def load_quote_fixture(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("quote fixture must be a JSON object")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("quote fixture needs snapshots[]")
    rows = []
    for row in snapshots:
        rows.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "timestamp": str(row.get("timestamp") or ""),
                "bid": float(row.get("bid")),
                "ask": float(row.get("ask")),
                "lot_size": int(row.get("lot_size", 1)),
            }
        )
    frame = pd.DataFrame(rows)
    if (frame["bid"] <= 0).any() or (frame["ask"] <= 0).any():
        raise ValueError("bid/ask must be > 0")
    if (frame["ask"] < frame["bid"]).any():
        raise ValueError("ask must be >= bid")
    return {"meta": payload.get("meta", {}), "snapshots": frame}


def check_execution_assumptions(path: str | Path, *, assumed_slippage_bps: float) -> dict:
    loaded = load_quote_fixture(path)
    frame: pd.DataFrame = loaded["snapshots"]
    checks: list[QuoteCheck] = []
    for row in frame.itertuples(index=False):
        mid = (row.bid + row.ask) / 2.0
        spread_bps = ((row.ask - row.bid) / mid) * 10_000.0 if mid > 0 else 0.0
        checks.append(
            QuoteCheck(
                symbol=row.symbol,
                timestamp=row.timestamp,
                bid=float(row.bid),
                ask=float(row.ask),
                lot_size=int(row.lot_size),
                spread_bps=float(spread_bps),
                assumed_slippage_bps=float(assumed_slippage_bps),
                within_spread=assumed_slippage_bps <= spread_bps,
            )
        )
    mismatch = [item for item in checks if not item.within_spread]
    return {
        "meta": loaded["meta"],
        "summary": {
            "snapshot_count": len(checks),
            "assumed_slippage_bps": float(assumed_slippage_bps),
            "spread_violations": len(mismatch),
            "all_within_spread": len(mismatch) == 0,
            "connectivity_verified": False,
            "note": "此檢查只使用離線 fixture，未驗證雲端環境可連實際 Shioaji 帳號。",
        },
        "checks": [item.__dict__ for item in checks],
    }
