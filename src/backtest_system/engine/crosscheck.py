"""Compare this engine's equity to another fill engine.

The default checker rebuilds the equity curve from the trade list and the
closes. It does not re-decide signals, so it will not notice a wrong trade
that the rest of the tests already missed; it will notice an equity path
that does not add up to those trades.

Nautilus is optional and is not imported unless a caller asks for it.
``requirements.txt`` does not list ``nautilus_trader``. The hook raises a
clear error until an adapter is plugged in; subclass :class:`PnlCrossCheck`
when that work happens.
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
    """Result of lining two equity curves up on shared dates."""

    ok: bool
    max_abs_diff: float
    compared_bars: int
    notes: tuple[str, ...] = ()


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
    ) -> CrossCheckReport:
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
    """Placeholder for a later Nautilus backtest on the same next-open schedule.

    Asking for this checker does not import ``nautilus_trader`` when the
    package is absent. When it is present, ``check`` still refuses: wiring
    orders, venues, and fee models is a separate piece of work, and a silent
    half-adapter would be worse than a loud one.
    """

    name = "nautilus"

    def describe(self) -> str:
        return (
            "Optional Nautilus comparison. Not a default dependency, and not wired "
            "to a live Nautilus engine in this repo."
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
    ) -> CrossCheckReport:
        del trades, closes, initial_capital, equity, contract_multiplier, symbol, atol, rtol
        if not self.available():
            raise OptionalEngineError(
                "nautilus_trader is not installed. It is not in the default "
                "requirements. Install the optional extra "
                '(pip install -e ".[nautilus]") only when an adapter exists. '
                "Use load_crosscheck('reference') to replay this engine's trades."
            )
        raise OptionalEngineError(
            "nautilus_trader is installed, but this repo only ships the comparison "
            "interface. Subclass PnlCrossCheck to submit the same next-open orders "
            "and return that engine's equity. The reference replay does not need Nautilus."
        )


def load_crosscheck(name: str = "reference") -> PnlCrossCheck:
    """``reference`` replays our trades. ``nautilus`` is the optional external hook."""
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
