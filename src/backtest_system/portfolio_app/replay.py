"""Deterministic fixed-trade replay for portable diagnostics.

This replay does not re-optimize parameters or regenerate signals. It applies
the existing trade tape to embedded Open/Close inputs so the website demo can
recompute curves for changed costs/slippage with no server.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReplayCosts:
    commission_rate: float
    commission_flat: float
    slippage_rate: float
    tw_brokerage_rate: float
    tw_brokerage_min: float
    tw_sell_tax_rate: float


@dataclass(frozen=True)
class ReplayResult:
    equity: pd.Series
    drawdown: pd.Series
    trades_applied: int
    problems: tuple[str, ...]


def replay_fixed_trades(
    *,
    trades: pd.DataFrame,
    prices: Mapping[str, pd.DataFrame],
    calendar: pd.Index,
    initial_capital: float,
    costs: ReplayCosts,
) -> ReplayResult:
    if initial_capital < 0:
        raise ValueError("initial_capital must be >= 0")
    if not prices:
        raise ValueError("prices are required for fixed-trade replay")

    tape = _normalize_tape(trades, prices)
    bars = _normalize_prices(prices)
    index = pd.DatetimeIndex(calendar).normalize()
    if len(index) == 0:
        return ReplayResult(pd.Series(dtype="float64"), pd.Series(dtype="float64"), 0, ("empty calendar",))

    grouped = defaultdict(list)
    for row in tape.itertuples(index=False):
        grouped[row.date].append(row)

    positions: dict[str, float] = defaultdict(float)
    closes: dict[str, float] = {}
    cash = float(initial_capital)
    marks: list[float] = []
    problems: list[str] = []
    applied = 0

    for day in index:
        date = pd.Timestamp(day).normalize()
        rows = grouped.get(date, ())
        for row in rows:
            frame = bars.get(row.symbol)
            if frame is None or date not in frame.index:
                problems.append(f"{row.symbol} {date.date()} missing Open/Close for trade replay")
                continue
            open_price = float(frame.at[date, "Open"])
            if open_price <= 0 or not pd.notna(open_price):
                problems.append(f"{row.symbol} {date.date()} invalid open price {open_price!r}")
                continue
            fill = _fill_price(open_price, row.action, costs.slippage_rate)
            notional = float(fill * row.shares)
            fee = _fee(notional, row.action, row.market, row.symbol, costs)
            if row.action == "BUY":
                cash -= notional + fee
                positions[row.symbol] += float(row.shares)
            else:
                positions[row.symbol] -= float(row.shares)
                if positions[row.symbol] < -1e-9:
                    problems.append(
                        f"{row.symbol} {date.date()} sell exceeds long position by {abs(positions[row.symbol]):.6f}"
                    )
                cash += notional - fee
            applied += 1

        equity = cash
        for symbol, qty in positions.items():
            if abs(qty) <= 1e-12:
                continue
            frame = bars.get(symbol)
            if frame is None:
                continue
            if date in frame.index:
                close = float(frame.at[date, "Close"])
                closes[symbol] = close
            close = closes.get(symbol)
            if close is None or close <= 0 or not pd.notna(close):
                continue
            equity += qty * close
        marks.append(float(equity))

    equity_curve = pd.Series(marks, index=index, dtype="float64")
    drawdown = (equity_curve / equity_curve.cummax()) - 1.0
    return ReplayResult(
        equity=equity_curve,
        drawdown=drawdown,
        trades_applied=applied,
        problems=tuple(dict.fromkeys(problems)),
    )


def _fill_price(open_price: float, side: str, slippage_rate: float) -> float:
    if side == "BUY":
        return open_price * (1.0 + slippage_rate)
    return open_price * (1.0 - slippage_rate)


def _fee(notional: float, side: str, market: str, symbol: str, costs: ReplayCosts) -> float:
    market_text = (market or "").strip().upper()
    if not market_text:
        market_text = "TW" if symbol.upper().endswith((".TW", ".TWO")) else "US"
    if market_text == "TW":
        brokerage = max(notional * costs.tw_brokerage_rate, costs.tw_brokerage_min)
        tax = notional * costs.tw_sell_tax_rate if side == "SELL" else 0.0
        return brokerage + tax
    return notional * costs.commission_rate + costs.commission_flat


def _normalize_tape(trades: pd.DataFrame, prices: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if trades is None or trades.empty:
        columns = ["date", "symbol", "action", "shares", "market"]
        return pd.DataFrame(columns=columns)
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "symbol" not in frame.columns:
        inferred = next(iter(prices))
        frame["symbol"] = inferred
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["action"] = frame["action"].astype(str).str.upper()
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
    if "market" not in frame.columns:
        frame["market"] = ""
    frame["market"] = frame["market"].fillna("").astype(str)
    keep = frame[["date", "symbol", "action", "shares", "market"]]
    keep = keep[keep["action"].isin({"BUY", "SELL"}) & (keep["shares"] > 0)]
    return keep.sort_values(["date", "symbol", "action"], kind="stable").reset_index(drop=True)


def _normalize_prices(prices: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    mapped: dict[str, pd.DataFrame] = {}
    for symbol, raw in prices.items():
        frame = raw.copy()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        for col in ("Open", "Close"):
            if col not in frame.columns:
                raise ValueError(f"{symbol} prices miss {col}")
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        mapped[str(symbol).upper()] = frame[["Open", "Close"]].sort_index()
    return mapped
