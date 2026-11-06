"""Compare this engine's equity to another fill engine.

The default checker rebuilds the equity curve from the trade list and the
closes. It does not re-decide signals, so it will not notice a wrong trade
that the rest of the tests already missed; it will notice an equity path
that does not add up to those trades.

``nautilus`` submits that same trade list to a Nautilus cash account and
marks the resulting position at these closes. The package is optional and
is not imported unless a caller asks for it. See
:mod:`backtest_system.engine.nautilus_adapter` for the fill and fee limits.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class OptionalEngineError(RuntimeError):
    """The requested external engine is not installed or not wired up."""


@dataclass(frozen=True)
class CrossCheckReport:
    """Result of lining two books up.

    ``total_pnl`` is the primary last equity minus the cash that was passed
    in. ``other_pnl`` is the same difference for the second book. Fill
    fields are set when the second book priced its own orders. A Nautilus
    report also carries the limits of that adapter in ``notes``.
    """

    ok: bool
    max_abs_diff: float
    compared_bars: int
    notes: tuple[str, ...] = ()
    total_pnl: float | None = None
    other_pnl: float | None = None
    trade_count: int | None = None
    other_trade_count: int | None = None
    max_fill_price_diff: float | None = None
    material_fills: int | None = None
    max_abs_return_diff: float | None = None


@dataclass
class CrossCheckRun:
    """The report plus the second equity, when the checker built one."""

    report: CrossCheckReport
    equity: pd.Series
    fills: pd.DataFrame | None = None


# One cent per fill for USD rounding, plus one 1e-4 price step on exposure.
# The price step is the grid the Nautilus adapter sends bars on.
def nautilus_atol(n_fills: int, max_exposure: float) -> float:
    """Absolute equity band for a Nautilus cross-check.

    Cash is rounded to the cent, and each open is sent on a 1e-4 grid.
    ``max_exposure`` should be the largest absolute equity or notional the
    book holds, so one grid step on that exposure is inside the band.
    """
    fills = max(int(n_fills), 0)
    exposure = max(float(max_exposure), 0.0)
    return 0.01 * (fills + 1) + 1e-4 * exposure + 1e-9


NAUTILUS_LIMITATIONS = (
    "Nautilus prints the market order at the session open: the bar it sees is flat at that open, so high and low are not walked.",
    "Slippage stays on the primary fill. A non-zero slippage is a fill-price gap, not a second slip model.",
    "Cash is rounded to the cent. Fees are the US rate plus the flat fee, or Taiwan brokerage, its minimum, and the sell-side tax.",
    "The Nautilus instrument is a cash CFD so size can be fractional. A contract multiplier is folded into the quantity.",
    "Bar volume is lifted so the order can fill. Historical volume is not a constraint.",
)


class PnlCrossCheck(Protocol):
    """Something that can judge an equity curve against a trade list."""

    name: str

    def describe(self) -> str:
        """One sentence on what this checker compares."""

    def check(
        self,
        trades: pd.DataFrame,
        closes: pd.Series | pd.DataFrame,
        initial_capital: float,
        equity: pd.Series,
        *,
        contract_multiplier: float | Mapping[str, float] = 1.0,
        symbol: str | None = None,
        atol: float = 1e-6,
        rtol: float = 1e-8,
        bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
        costs=None,
        mode: str = "pooled",
        symbols: Sequence[str] | None = None,
    ) -> CrossCheckReport:
        """Return whether ``equity`` matches this checker's own marks."""


def compare_equity(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-8,
) -> CrossCheckReport:
    """True when ``abs(left - right) <= atol + rtol * abs(right)`` on the overlap."""
    pair = pd.concat(
        [_as_float(left).rename("left"), _as_float(right).rename("right")],
        axis=1,
    ).dropna()
    notes: list[str] = []
    if pair.empty:
        return CrossCheckReport(False, math.inf, 0, ("no overlapping dates",))
    if len(pair) != len(left) or len(pair) != len(right):
        notes.append(
            f"compared {len(pair)} shared dates; left has {len(left)}, right has {len(right)}"
        )
    diff = (pair["left"] - pair["right"]).abs()
    limit = atol + rtol * pair["right"].abs()
    ok = bool((diff <= limit).all())
    return CrossCheckReport(ok, float(diff.max()), int(len(pair)), tuple(notes))


def replay_equity(
    trades: pd.DataFrame,
    closes: pd.Series | pd.DataFrame,
    initial_capital: float,
    *,
    contract_multiplier: float | Mapping[str, float] = 1.0,
    symbol: str | None = None,
) -> pd.Series:
    """Cash ledger implied by the fills.

    A BUY spends ``value + fee`` and adds ``shares``. A SELL adds ``value``
    (already net of fees) and subtracts ``shares``. The book is then marked
    at that session's close. Missing closes forward-fill. A name with no
    close yet contributes nothing. Cash does not earn interest.
    """
    frame = _close_frame(closes, symbol)
    if frame.shape[1] == 0:
        raise ValueError("closes must name at least one symbol")
    default_symbol = str(frame.columns[0]) if frame.shape[1] == 1 else None
    multipliers = _multipliers(frame.columns, contract_multiplier)
    cash = float(initial_capital)
    held = {str(column): 0.0 for column in frame.columns}
    by_day = _trades_by_day(trades, default_symbol if symbol is None else symbol)
    values: list[float] = []
    for stamp in frame.index:
        for row in by_day.get(_day(stamp), ()):
            name = row["symbol"]
            if name not in held:
                known = ", ".join(held) or "(none)"
                raise ValueError(f"trade symbol {name!r} is not in the close marks ({known})")
            side = str(row["action"]).upper()
            qty = float(row["shares"])
            if side == "BUY":
                cash -= float(row["value"]) + float(row["fee"])
                held[name] += qty
            elif side == "SELL":
                cash += float(row["value"])
                held[name] -= qty
            else:
                raise ValueError(f"unknown trade action {row['action']!r}")
        marked = cash
        prices = frame.loc[stamp]
        for name, qty in held.items():
            if qty == 0.0:
                continue
            price = float(prices[name])
            if not math.isfinite(price):
                continue
            marked += qty * price * multipliers[name]
        values.append(marked)
    equity = pd.Series(values, index=frame.index, name="equity")
    return equity


class ReferenceReplay:
    """Rebuild equity from this engine's own trade list. No extra dependency."""

    name = "reference"

    def describe(self) -> str:
        return "Replay buys and sells onto the closes and compare that equity to the engine."

    def check(
        self,
        trades: pd.DataFrame,
        closes: pd.Series | pd.DataFrame,
        initial_capital: float,
        equity: pd.Series,
        *,
        contract_multiplier: float | Mapping[str, float] = 1.0,
        symbol: str | None = None,
        atol: float = 1e-6,
        rtol: float = 1e-8,
        bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
        costs=None,
        mode: str = "pooled",
        symbols: Sequence[str] | None = None,
    ) -> CrossCheckReport:
        del bars, costs, mode, symbols
        rebuilt = replay_equity(
            trades,
            closes,
            initial_capital,
            contract_multiplier=contract_multiplier,
            symbol=symbol,
        )
        report = compare_equity(equity, rebuilt, atol=atol, rtol=rtol)
        if report.ok:
            return report
        return CrossCheckReport(
            False,
            report.max_abs_diff,
            report.compared_bars,
            report.notes + ("replayed equity does not match the engine",),
        )


class NautilusCrossCheck:
    """Submit the primary trade list to a Nautilus cash account.

    The import happens inside :meth:`check`. A missing install raises
    :class:`OptionalEngineError` and does not touch the primary engine.
    The last successful or failed run is kept on ``last_run`` so a caller
    can plot that equity without running Nautilus twice.
    """

    name = "nautilus"

    def __init__(self) -> None:
        self.last_run: CrossCheckRun | None = None

    def describe(self) -> str:
        return (
            "Optional Nautilus book. Same orders, filled at the session open, "
            "marked at these closes. Not a default dependency."
        )

    def available(self) -> bool:
        return importlib.util.find_spec("nautilus_trader") is not None

    def check(
        self,
        trades: pd.DataFrame,
        closes: pd.Series | pd.DataFrame,
        initial_capital: float,
        equity: pd.Series,
        *,
        contract_multiplier: float | Mapping[str, float] = 1.0,
        symbol: str | None = None,
        atol: float = 1e-6,
        rtol: float = 1e-8,
        bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
        costs=None,
        mode: str = "pooled",
        symbols: Sequence[str] | None = None,
    ) -> CrossCheckReport:
        run = _nautilus_run(
            trades,
            closes,
            initial_capital,
            equity,
            bars=bars,
            costs=costs,
            contract_multiplier=contract_multiplier,
            symbol=symbol,
            atol=atol,
            rtol=rtol,
            mode=mode,
            symbols=symbols,
        )
        self.last_run = run
        return run.report


def load_crosscheck(name: str = "reference") -> PnlCrossCheck:
    """``reference`` replays our trades. ``nautilus`` runs them in Nautilus."""
    key = (name or "reference").strip().lower()
    if key in {"reference", "replay"}:
        return ReferenceReplay()
    if key in {"nautilus", "nautilus_trader"}:
        return NautilusCrossCheck()
    raise ValueError(
        f"unknown cross-check {name!r}; expected 'reference' or 'nautilus'"
    )


def _as_float(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    out.index = series.index
    return out


def _day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"invalid date: {value!r}")
    if stamp.tzinfo is not None:
        stamp = pd.Timestamp(stamp.strftime("%Y-%m-%d"))
    return stamp.normalize()


def _close_frame(closes: pd.Series | pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    if isinstance(closes, pd.Series):
        name = symbol if symbol else closes.name
        if name is None or str(name) == "":
            name = "asset"
        frame = closes.to_frame(name=str(name))
    else:
        frame = closes.copy()
        if symbol is not None and frame.shape[1] == 1 and str(frame.columns[0]) != str(symbol):
            frame = frame.rename(columns={frame.columns[0]: str(symbol)})
    frame.columns = [str(column) for column in frame.columns]
    frame.index = pd.DatetimeIndex(_day(stamp) for stamp in frame.index)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.ffill()


def _multipliers(
    columns: Sequence[str], contract_multiplier: float | Mapping[str, float]
) -> dict[str, float]:
    if isinstance(contract_multiplier, Mapping):
        return {str(column): float(contract_multiplier.get(str(column), 1.0)) for column in columns}
    value = float(contract_multiplier)
    return {str(column): value for column in columns}


def run_crosscheck(
    name: str,
    trades: pd.DataFrame,
    closes: pd.Series | pd.DataFrame,
    initial_capital: float,
    equity: pd.Series,
    *,
    bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
    costs=None,
    contract_multiplier: float | Mapping[str, float] = 1.0,
    symbol: str | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    mode: str = "pooled",
    symbols: Sequence[str] | None = None,
) -> CrossCheckRun:
    """Run ``reference`` or ``nautilus`` and return the second equity too.

    ``mode="sleeves"`` funds each requested name with an equal slice and
    sums the curves, dropping a name that has no bars. ``mode="pooled"``
    is one cash account. Target-weight books are pooled. Equal-weight
    signal books are sleeves.
    """
    key = (name or "reference").strip().lower()
    if key in {"reference", "replay"}:
        return _reference_run(
            trades,
            closes,
            initial_capital,
            equity,
            bars=bars,
            contract_multiplier=contract_multiplier,
            symbol=symbol,
            atol=1e-6 if atol is None else atol,
            rtol=1e-8 if rtol is None else rtol,
            mode=mode,
            symbols=symbols,
        )
    if key in {"nautilus", "nautilus_trader"}:
        return _nautilus_run(
            trades,
            closes,
            initial_capital,
            equity,
            bars=bars,
            costs=costs,
            contract_multiplier=contract_multiplier,
            symbol=symbol,
            atol=atol,
            rtol=0.0 if rtol is None else rtol,
            mode=mode,
            symbols=symbols,
        )
    raise ValueError(f"unknown cross-check {name!r}; expected 'reference' or 'nautilus'")


def _reference_run(
    trades,
    closes,
    initial_capital,
    equity,
    *,
    bars,
    contract_multiplier,
    symbol,
    atol,
    rtol,
    mode,
    symbols,
) -> CrossCheckRun:
    kind = (mode or "pooled").strip().lower()
    if kind == "sleeves":
        rebuilt = _replay_sleeves(
            trades,
            _close_map(closes, bars, symbol),
            initial_capital,
            list(symbols) if symbols is not None else _names_of(closes, bars, symbol),
            contract_multiplier,
            equity.index,
        )
    elif kind == "pooled":
        rebuilt = replay_equity(
            trades,
            closes if bars is None else _closes_frame(closes, bars, symbol),
            initial_capital,
            contract_multiplier=contract_multiplier,
            symbol=symbol,
        )
    else:
        raise ValueError(f"unknown cross-check mode {mode!r}; expected pooled or sleeves")
    report = compare_equity(equity, rebuilt.reindex(equity.index), atol=atol, rtol=rtol)
    notes = list(report.notes)
    if not report.ok:
        notes.append("replayed equity does not match the engine")
    counted = 0 if trades is None or trades.empty else int(len(trades))
    return CrossCheckRun(
        _with_pnl(
            report,
            equity,
            rebuilt.reindex(equity.index),
            initial_capital,
            notes,
            trade_count=counted,
            other_trade_count=counted,
        ),
        rebuilt,
        None,
    )


def _nautilus_run(
    trades,
    closes,
    initial_capital,
    equity,
    *,
    bars,
    costs,
    contract_multiplier,
    symbol,
    atol,
    rtol,
    mode,
    symbols,
) -> CrossCheckRun:
    if importlib.util.find_spec("nautilus_trader") is None:
        raise OptionalEngineError(
            "nautilus_trader is not installed. It is not a default dependency. "
            'Install the optional extra (pip install -e ".[nautilus]") to cross-check '
            "fills in Nautilus. load_crosscheck('reference') replays this engine's "
            "trades without it."
        )
    if bars is None:
        raise OptionalEngineError(
            "nautilus cross-check needs the Open/Close bars the fills were made from. "
            "Pass bars= (one frame, or a symbol to frame mapping)."
        )
    exposure = float(pd.Series(equity).abs().max()) if len(equity) else abs(float(initial_capital))
    n_fills = 0 if trades is None or getattr(trades, "empty", True) else int(len(trades))
    if atol is None:
        atol = nautilus_atol(n_fills, exposure)
    try:
        from backtest_system.engine.nautilus_adapter import execute
    except ImportError as exc:
        raise OptionalEngineError(
            "nautilus_trader is not installed. It is not a default dependency. "
            'Install the optional extra (pip install -e ".[nautilus]").'
        ) from exc
    try:
        execution = execute(
            trades,
            bars,
            equity.index,
            float(initial_capital),
            costs=costs,
            contract_multiplier=contract_multiplier,
            symbol=symbol,
            mode=mode,
            symbols=symbols,
        )
    except ImportError as exc:
        raise OptionalEngineError(
            "nautilus_trader is not installed. It is not a default dependency. "
            'Install the optional extra (pip install -e ".[nautilus]").'
        ) from exc
    other = _align_index(execution.equity, equity.index)
    compared = compare_equity(equity, other, atol=float(atol), rtol=float(rtol))
    max_px, material = _fill_gap(trades, execution.fills, symbol, symbols)
    n_right = 0 if execution.fills is None or execution.fills.empty else int(len(execution.fills))
    notes = list(compared.notes)
    notes.extend(execution.problems)
    if n_fills != n_right:
        notes.append(f"trade count {n_fills} vs nautilus {n_right}")
    if material:
        notes.append(
            f"{material} fill(s) differ by more than the 1e-4 price grid, the 1e-6 size grid, or one cent of fees"
        )
    length_ok = compared.compared_bars == len(equity) and compared.compared_bars > 0
    if not length_ok:
        notes.append("nautilus equity does not cover the primary curve")
    if not compared.ok:
        notes.append("nautilus equity does not match the primary book")
    notes.extend(NAUTILUS_LIMITATIONS)
    ok = bool(compared.ok and length_ok and n_fills == n_right and material == 0 and not execution.problems)
    report = _with_pnl(
        CrossCheckReport(ok, compared.max_abs_diff, compared.compared_bars, tuple(notes)),
        equity,
        other,
        initial_capital,
        notes,
        trade_count=n_fills,
        other_trade_count=n_right,
        max_fill_price_diff=max_px,
        material_fills=material,
        max_abs_return_diff=_return_gap(equity, other),
    )
    return CrossCheckRun(report, other, execution.fills)


def _with_pnl(
    report: CrossCheckReport,
    equity: pd.Series,
    other: pd.Series,
    initial_capital: float,
    notes,
    *,
    trade_count,
    other_trade_count,
    max_fill_price_diff=None,
    material_fills=None,
    max_abs_return_diff=None,
) -> CrossCheckReport:
    def _pnl(series: pd.Series) -> float | None:
        if series is None or len(series) == 0:
            return None
        last = float(series.iloc[-1])
        if not math.isfinite(last):
            return None
        return last - float(initial_capital)

    return CrossCheckReport(
        report.ok,
        report.max_abs_diff,
        report.compared_bars,
        tuple(notes),
        total_pnl=_pnl(equity),
        other_pnl=_pnl(other),
        trade_count=trade_count,
        other_trade_count=other_trade_count,
        max_fill_price_diff=max_fill_price_diff,
        material_fills=material_fills,
        max_abs_return_diff=max_abs_return_diff if max_abs_return_diff is not None else _return_gap(equity, other),
    )


def _return_gap(equity: pd.Series, other: pd.Series) -> float | None:
    if equity is None or other is None or len(equity) < 2 or len(other) < 2:
        return None
    gap = (equity.pct_change() - other.reindex(equity.index).pct_change()).abs()
    gap = gap.replace([math.inf, -math.inf], pd.NA).dropna()
    if gap.empty:
        return None
    return float(gap.max())


def _align_index(series: pd.Series, index: pd.Index) -> pd.Series:
    values = series.copy()
    values.index = pd.DatetimeIndex(_day(stamp) for stamp in values.index)
    values = values[~values.index.duplicated(keep="last")].sort_index()
    target = pd.DatetimeIndex(_day(stamp) for stamp in index)
    aligned = values.reindex(target)
    aligned.index = index
    return aligned


def _fill_gap(trades: pd.DataFrame, fills: pd.DataFrame | None, symbol: str | None, symbols: Sequence[str] | None):
    left = _fill_records(trades, symbol if symbols is None or len(list(symbols or [])) <= 1 else None)
    right = _fill_records(fills, None)
    paired = min(len(left), len(right))
    max_px = 0.0
    material = abs(len(left) - len(right))
    for i in range(paired):
        primary = left[i]
        other = right[i]
        price_gap = abs(primary["price"] - other["price"])
        max_px = max(max_px, price_gap)
        disagreed = (
            primary["action"] != other["action"]
            or primary["symbol"] != other["symbol"]
            or primary["date"] != other["date"]
            or price_gap > 1e-4 + 1e-9
            or abs(primary["shares"] - other["shares"]) > 1e-6 + 1e-12
            or abs(primary["fee"] - other["fee"]) > 0.01 + 1e-9
        )
        if disagreed:
            material += 1
    if not left and not right:
        max_px = 0.0
    elif paired == 0 and (left or right):
        max_px = math.inf
    return max_px, material


def _fill_records(frame: pd.DataFrame | None, default_symbol: str | None) -> list[dict]:
    if frame is None or frame.empty:
        return []
    rows = []
    counts: dict[tuple, int] = {}
    for record in frame.to_dict(orient="records"):
        name = record.get("symbol")
        if name is None or (isinstance(name, float) and math.isnan(name)) or str(name) == "":
            name = default_symbol or "asset"
        action = str(record.get("action") or "").upper()
        day = _day(record["date"])
        key = (day, str(name), action)
        seq = counts.get(key, 0)
        counts[key] = seq + 1
        rows.append(
            (
                (day, str(name), action, seq),
                {
                    "date": day,
                    "symbol": str(name),
                    "action": action,
                    "price": float(record.get("price") or 0.0),
                    "shares": float(record.get("shares") or 0.0),
                    "fee": float(record.get("fee") or 0.0),
                },
            )
        )
    rows.sort(key=lambda item: item[0])
    return [row for _, row in rows]


def _replay_sleeves(trades, close_map, capital, symbols, contract_multiplier, index) -> pd.Series:
    if not symbols:
        raise ValueError("sleeves need the requested symbol list")
    per = float(capital) / len(symbols)
    curves = []
    for name in symbols:
        series = close_map.get(str(name))
        if series is None:
            continue
        part = _trades_for_symbol(trades, str(name), symbols)
        scale = _one_scale(str(name), contract_multiplier)
        rebuilt = replay_equity(part, series.dropna(), per, contract_multiplier=scale, symbol=str(name))
        curves.append(rebuilt.rename(str(name)))
    target = pd.DatetimeIndex(_day(stamp) for stamp in index)
    if not curves:
        return pd.Series(0.0, index=target, name="replay")
    wide = pd.concat(curves, axis=1).sort_index().reindex(target).ffill().fillna(0.0)
    equity = wide.sum(axis=1)
    equity.name = "replay"
    if len(equity) == len(index):
        equity.index = index
    return equity


def _trades_for_symbol(trades: pd.DataFrame, symbol: str, symbols: Sequence[str]) -> pd.DataFrame:
    if trades is None or trades.empty:
        return trades if trades is not None else pd.DataFrame()
    if "symbol" not in trades.columns:
        if len(symbols) == 1:
            out = trades.copy()
            out["symbol"] = symbol
            return out
        raise ValueError("sleeves need a symbol column when more than one name was requested")
    return trades.loc[trades["symbol"].astype(str) == symbol].copy()


def _close_map(closes, bars, symbol: str | None) -> dict[str, pd.Series]:
    if isinstance(bars, Mapping) and not isinstance(bars, pd.DataFrame):
        out = {}
        for name, frame in bars.items():
            if frame is None or "Close" not in frame.columns:
                continue
            out[str(name)] = frame["Close"]
        if out:
            return out
    if isinstance(closes, pd.DataFrame):
        return {str(column): closes[column] for column in closes.columns}
    name = symbol or (closes.name if isinstance(closes, pd.Series) and closes.name else "asset")
    return {str(name): closes}


def _closes_frame(closes, bars, symbol):
    if bars is None:
        return closes
    if isinstance(bars, pd.DataFrame):
        if "Close" in bars.columns:
            return bars["Close"] if symbol is None else bars["Close"].rename(symbol)
        return closes
    columns = {}
    for name, frame in bars.items():
        if frame is not None and "Close" in frame.columns:
            columns[str(name)] = frame["Close"]
    if not columns:
        return closes
    return pd.DataFrame(columns)


def _names_of(closes, bars, symbol) -> list[str]:
    return list(_close_map(closes, bars, symbol))


def _one_scale(symbol: str, contract_multiplier) -> float:
    if isinstance(contract_multiplier, Mapping):
        return float(contract_multiplier.get(symbol, 1.0))
    return float(contract_multiplier)


def _trades_by_day(trades: pd.DataFrame, default_symbol: str | None) -> dict[pd.Timestamp, tuple[dict, ...]]:
    grouped: dict[pd.Timestamp, list[dict]] = {}
    if trades is None or trades.empty:
        return {}
    for row in trades.to_dict(orient="records"):
        name = row.get("symbol")
        if name is None or (isinstance(name, float) and math.isnan(name)) or str(name) == "":
            if default_symbol is None:
                raise ValueError("trade is missing a symbol and the close marks have several columns")
            name = default_symbol
        item = {
            "symbol": str(name),
            "action": row.get("action"),
            "shares": row.get("shares"),
            "value": row.get("value"),
            "fee": row.get("fee", 0.0),
        }
        grouped.setdefault(_day(row["date"]), []).append(item)
    return {day: tuple(rows) for day, rows in grouped.items()}
