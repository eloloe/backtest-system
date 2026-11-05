"""Turn signals or target weights into an equity curve.

Single-name strategies still go through ``Backtester``: one account per
name, equal capital, all-in or flat. Target-weight books (the cross-section)
share one cash balance. Fills use the same next-open, slippage, and fee
helpers as ``Backtester``. A weight decided on the close is traded at the
next bar's open, or dropped if that open never arrives.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtester import TRADE_COLUMNS, Backtester, _execution_price, _fee, resolve_costs

BOOK_TRADE_COLUMNS = TRADE_COLUMNS + ["symbol"]


@dataclass
class BookResult:
    equity: pd.Series
    trades: pd.DataFrame
    initial_capital: float
    holdings: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


def month_end_dates(index, start, end, *, every: int = 1) -> list[pd.Timestamp]:
    """Close of a month whose next session falls in a later month.

    A sample that stops on the 6th does not pretend that day is month-end.
    The last bar of the index is never a decision: there is no later session
    to confirm the month rolled, and there would be no open to fill. ``every``
    keeps the 1st, 1+N, ... of those dates, so 1 is monthly and 3 is every
    third completed month, counting from the first one in ``[start, end)``.
    """
    if isinstance(every, bool) or int(every) < 1:
        raise ValueError("rebalance_months must be an integer >= 1")
    start_day = _day(start)
    end_day = _day(end)
    stamps = _day_index(index).unique().sort_values()
    dates: list[pd.Timestamp] = []
    for i in range(len(stamps) - 1):
        day = _day(stamps[i])
        nxt = _day(stamps[i + 1])
        if (day.year, day.month) != (nxt.year, nxt.month):
            if start_day <= day < end_day:
                dates.append(day)
    return dates[:: int(every)]


def run_sleeve(
    strategy,
    prices: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    start,
    end,
    *,
    initial_capital: float,
    commission: float = 0.001,
    slippage: float = 0.0005,
    market: str | None = None,
    backtester_kwargs: Mapping | None = None,
) -> BookResult:
    """Run one sleeve. Book strategies size themselves; signals use ``Backtester``."""
    if getattr(strategy, "manages_book", False):
        return strategy.run_book(
            prices,
            symbols,
            start,
            end,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            market=market,
            backtester_kwargs=backtester_kwargs,
        )
    return simulate_signals(
        strategy,
        prices,
        symbols,
        start,
        end,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        market=market,
        backtester_kwargs=backtester_kwargs,
    )


def simulate_signals(
    strategy,
    prices: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    start,
    end,
    *,
    initial_capital: float,
    commission: float = 0.001,
    slippage: float = 0.0005,
    market: str | None = None,
    backtester_kwargs: Mapping | None = None,
) -> BookResult:
    """Equal-weight, all-in accounts. Same split the Streamlit loop used.

    Capital is divided by the number of requested names. A name with no
    prices contributes nothing: that slice is not reassigned. Signals
    before ``start`` are cleared so a warmup history can build an indicator
    without leaving a position on the books. The reported curve is
    ``[start, end)``.
    """
    start_day = _day(start)
    end_day = _day(end)
    names = [str(symbol) for symbol in symbols]
    if not names:
        raise ValueError("no symbols to simulate")
    per = float(initial_capital) / len(names)
    market, extra = _split_market(market, backtester_kwargs)
    curves: list[pd.Series] = []
    trades: list[pd.DataFrame] = []

    for symbol in names:
        history = _frame(prices.get(symbol))
        if history is None:
            continue
        signals = strategy.generate_signals(history)
        signals = signals.reindex(history.index).fillna(0).astype(int)
        signals = signals.copy()
        signals.loc[signals.index < start_day] = 0
        bt = Backtester(
            initial_capital=per,
            commission=commission,
            slippage=slippage,
            market=market,
            **extra,
        )
        portfolio, filled = bt.run(history, signals, symbol=symbol, market=market)
        view = portfolio.loc[(portfolio.index >= start_day) & (portfolio.index < end_day)]
        if view.empty:
            continue
        curves.append(view["portfolio_value"].rename(symbol))
        if not filled.empty:
            kept = _clip_trades(filled, start_day, end_day)
            if not kept.empty:
                kept = kept.copy()
                kept["symbol"] = symbol
                trades.append(kept)

    if not curves:
        equity = _cash_curve(prices, names, start_day, end_day, float(initial_capital))
        return BookResult(equity, _empty_trades(), float(initial_capital), notes=["沒有可交易的報價，全程現金。"])

    wide = pd.concat(curves, axis=1).sort_index().ffill().fillna(0.0)
    equity = wide.sum(axis=1)
    equity.name = "equity"
    return BookResult(
        equity=equity,
        trades=_stack_trades(trades),
        initial_capital=float(initial_capital),
    )


def simulate_target_weights(
    prices: Mapping[str, pd.DataFrame],
    weights: pd.DataFrame,
    symbols: Sequence[str],
    start,
    end,
    *,
    initial_capital: float,
    commission: float = 0.001,
    slippage: float = 0.0005,
    market: str | None = None,
    backtester_kwargs: Mapping | None = None,
) -> BookResult:
    """Trade toward target weights at the next open.

    ``weights`` is indexed by the decision close. Missing names are target
    zero. Between decisions the share count is held, so weights drift.
    Sells print before buys. Buy size is scaled so notional plus fees fit
    in the cash those sells left behind.
    """
    start_day = _day(start)
    end_day = _day(end)
    names = [str(symbol) for symbol in symbols]
    frames = {symbol: _frame(prices.get(symbol)) for symbol in names}
    frames = {symbol: frame for symbol, frame in frames.items() if frame is not None}
    calendar = _cash_index(frames, start_day, end_day)
    if len(calendar) < 2:
        raise ValueError(f"no price bars in [{start_day.date()}, {end_day.date()})")

    decisions = _decision_map(weights)
    cost_of = {
        symbol: _costs(symbol, market, commission, slippage, backtester_kwargs)
        for symbol in names
    }
    cash = float(initial_capital)
    shares = {symbol: 0.0 for symbol in names}
    last_close: dict[str, float] = {}
    pending: dict[str, float] | None = None
    equity_values: list[float] = []
    trades: list[dict] = []

    for i, day in enumerate(calendar):
        if i > 0 and pending is not None:
            cash = _rebalance(day, pending, frames, cost_of, shares, trades, cash)
            pending = None
        for symbol, frame in frames.items():
            close = _print(frame, day, "Close")
            if close is not None:
                last_close[symbol] = close
        marked = cash
        for symbol, qty in shares.items():
            if qty <= 0 or symbol not in last_close:
                continue
            marked += qty * last_close[symbol] * cost_of[symbol].contract_multiplier
        equity_values.append(marked)
        if day in decisions:
            pending = decisions[day]

    equity = pd.Series(equity_values, index=calendar, name="equity")
    return BookResult(
        equity=equity,
        trades=_rows_to_frame(trades),
        initial_capital=float(initial_capital),
    )


def _rebalance(day, pending, frames, cost_of, shares, trades, cash: float) -> float:
    """Sell first, then buy. Returns the cash left after the open."""
    opens: dict[str, float] = {}
    for symbol in shares:
        frame = frames.get(symbol)
        if frame is None:
            continue
        price = _print(frame, day, "Open")
        if price is not None and price > 0:
            opens[symbol] = price

    equity = cash
    for symbol, qty in shares.items():
        if qty > 0 and symbol in opens:
            equity += qty * opens[symbol] * cost_of[symbol].contract_multiplier

    targets: dict[str, float] = {}
    for symbol in shares:
        weight = float(pending.get(symbol, 0.0) or 0.0)
        if weight < 0:
            weight = 0.0
        if symbol not in opens:
            continue
        targets[symbol] = weight * equity

    for symbol, target_notional in targets.items():
        costs = cost_of[symbol]
        unit = opens[symbol] * costs.contract_multiplier
        if unit <= 0:
            continue
        target_shares = target_notional / unit
        held = shares[symbol]
        if held <= target_shares + 1e-8:
            continue
        sell_qty = held if target_shares <= 1e-8 else held - target_shares
        if costs.min_lot and target_shares > 1e-8:
            lots = math.floor(sell_qty / costs.min_lot + 1e-9)
            sell_qty = lots * costs.min_lot
        sell_qty = min(sell_qty, held)
        if sell_qty <= 1e-12:
            continue
        fill = _execution_price(opens[symbol], False, costs.slippage_rate)
        notional = sell_qty * fill * costs.contract_multiplier
        fee = _fee(notional, costs, "SELL")
        proceeds = notional - fee
        cash += proceeds
        shares[symbol] = held - sell_qty
        trades.append(
            _trade(day, "SELL", fill, sell_qty, proceeds, fee, costs.market, symbol)
        )

    gaps = []
    for symbol, target_notional in targets.items():
        costs = cost_of[symbol]
        fill = _execution_price(opens[symbol], True, costs.slippage_rate)
        unit = fill * costs.contract_multiplier
        if unit <= 0:
            continue
        # Target share count is still measured off the unslipped open, so a
        # full investment matches Backtester's all-in share count once fees
        # and slippage scale the order down.
        open_unit = opens[symbol] * costs.contract_multiplier
        target_shares = target_notional / open_unit
        gap = target_shares - shares[symbol]
        if gap <= 1e-8:
            continue
        gaps.append((symbol, gap, fill, costs))

    for symbol, qty, notional, fee, fill in _scale_buys(gaps, cash):
        cash -= notional + fee
        shares[symbol] += qty
        trades.append(_trade(day, "BUY", fill, qty, notional, fee, cost_of[symbol].market, symbol))

    return cash


def _scale_buys(gaps, cash: float) -> list[tuple]:
    """Shrink buy quantities until notional + fee fits in ``cash``."""
    if not gaps or cash <= 0:
        return []
    scale = 1.0
    planned: list[tuple] = []
    for _ in range(8):
        planned = []
        total = 0.0
        for symbol, qty, fill, costs in gaps:
            sized = qty * scale
            if costs.min_lot:
                lots = math.floor(sized / costs.min_lot + 1e-9)
                sized = lots * costs.min_lot
            if sized <= 1e-12:
                continue
            notional = sized * fill * costs.contract_multiplier
            fee = _fee(notional, costs, "BUY")
            planned.append((symbol, sized, notional, fee, fill))
            total += notional + fee
        if not planned:
            return []
        if total <= cash + 1e-6:
            return planned
        if total <= 0:
            return []
        scale *= cash / total
    # Still over after a few passes (a minimum commission can do that).
    # Drop orders from the end until the rest fit.
    while planned:
        total = sum(row[2] + row[3] for row in planned)
        if total <= cash + 1e-6:
            return planned
        planned.pop()
    return []


def combine_books(books: Sequence[BookResult]) -> BookResult:
    """Add sleeve equity. A sleeve with no mark yet is still its starting cash."""
    if not books:
        raise ValueError("no sleeves to combine")
    index = pd.DatetimeIndex([])
    for book in books:
        index = index.union(book.equity.index)
    index = index.sort_values()
    columns = []
    for book in books:
        series = book.equity.reindex(index).astype(float)
        series = series.ffill()
        series = series.fillna(float(book.initial_capital))
        columns.append(series)
    equity = pd.concat(columns, axis=1).sum(axis=1)
    equity.name = "equity"
    trades = _stack_trades(
        [book.trades for book in books if book.trades is not None and not book.trades.empty]
    )
    holdings = [book.holdings for book in books if book.holdings is not None and not book.holdings.empty]
    notes: list[str] = []
    for book in books:
        notes.extend(book.notes)
    return BookResult(
        equity=equity,
        trades=trades,
        initial_capital=sum(float(book.initial_capital) for book in books),
        holdings=pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame(),
        notes=notes,
    )


def _split_market(market, extra):
    cleaned = dict(extra or {})
    if market is None and cleaned.get("market") is not None:
        market = cleaned.pop("market")
    else:
        cleaned.pop("market", None)
    return market, cleaned


def _costs(symbol, market, commission, slippage, extra):
    market, extra = _split_market(market, extra)
    return resolve_costs(
        symbol=symbol,
        market=market,
        commission=commission,
        slippage=slippage,
        commission_flat=float(extra.get("commission_flat", 0.0)),
        commission_bps=extra.get("commission_bps"),
        slippage_bps=extra.get("slippage_bps"),
        brokerage_rate=extra.get("brokerage_rate"),
        brokerage_min=extra.get("brokerage_min"),
        sell_tax_rate=extra.get("sell_tax_rate"),
        dividend_withholding=extra.get("dividend_withholding"),
        contract_multiplier=float(extra.get("contract_multiplier", 1.0)),
        min_lot=extra.get("min_lot"),
    )


def _decision_map(weights: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    if weights is None or weights.empty:
        return {}
    out: dict[pd.Timestamp, dict[str, float]] = {}
    for stamp, row in weights.iterrows():
        out[_day(stamp)] = {str(column): float(row[column]) for column in weights.columns}
    return out


def _frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or len(frame) == 0:
        return None
    if "Open" not in frame.columns or "Close" not in frame.columns:
        return None
    out = frame.copy()
    out.index = _day_index(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _print(frame: pd.DataFrame, day: pd.Timestamp, column: str):
    if day not in frame.index:
        return None
    value = frame.at[day, column]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _cash_index(frames: Mapping[str, pd.DataFrame], start, end) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex([])
    for frame in frames.values():
        stamps = frame.index[(frame.index >= start) & (frame.index < end)]
        index = index.union(stamps)
    return index.sort_values()


def _cash_curve(prices, symbols, start, end, capital: float) -> pd.Series:
    frames = {symbol: _frame(prices.get(symbol)) for symbol in symbols}
    frames = {symbol: frame for symbol, frame in frames.items() if frame is not None}
    index = _cash_index(frames, start, end)
    if len(index) < 2:
        raise ValueError(f"no price bars in [{start.date()}, {end.date()})")
    equity = pd.Series(float(capital), index=index, name="equity")
    return equity


def _clip_trades(trades: pd.DataFrame, start, end) -> pd.DataFrame:
    dates = _day_index(trades["date"])
    mask = np.asarray((dates >= start) & (dates < end), dtype=bool)
    return trades.loc[mask].copy()


def _stack_trades(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return _empty_trades()
    stacked = pd.concat(list(frames), ignore_index=True)
    if "date" in stacked.columns:
        stacked = stacked.sort_values("date").reset_index(drop=True)
    return stacked


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=BOOK_TRADE_COLUMNS)


def _rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty_trades()
    return pd.DataFrame(rows, columns=BOOK_TRADE_COLUMNS)


def prepare_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Naive daily index, last row wins on a duplicate date. Shared by the CLI and the UI."""
    return _frame(frame)


def _trade(day, action, price, shares, value, fee, market, symbol) -> dict:
    return {
        "date": day,
        "action": action,
        "price": price,
        "shares": shares,
        "value": value,
        "fee": fee,
        "market": market,
        "symbol": symbol,
    }


def _day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"invalid date: {value!r}")
    if stamp.tzinfo is not None:
        stamp = pd.Timestamp(stamp.strftime("%Y-%m-%d"))
    return stamp.normalize()


def _day_index(index) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex(pd.to_datetime(index))
    if getattr(stamps, "tz", None) is not None:
        stamps = pd.DatetimeIndex(stamps.strftime("%Y-%m-%d"))
    return stamps.normalize()
