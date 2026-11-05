"""Buy-and-hold benchmark and a noise baseline for one finished run.

The benchmark is the declared ticker's close, bought on the first date of
the strategy curve and held, without fees. Strategy equity already includes
fees. Taiwan names (``.TW`` / ``.TWO``, or ``market="TW"``) default to
``0050.TW``; everything else defaults to ``SPY``. An explicit ticker wins.
An empty benchmark is an opt-out.

Noise is a shuffled-label book when the strategy emits a 0/1 signal: the
in-window signals are permuted, then filled with the same next-open costs.
A book that sizes its own weights has no single label series, so the noise
path is random entry, long on half the days, same costs. The seed is fixed.
A signal that never changes is reported as degenerate and is not redrawn.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from books import simulate_signals
from metrics import calculate_metrics
from strategies import BaseStrategy

_DEFAULT_PERMS = 16
_DEFAULT_SEED = 0
_RANDOM_LONG = 0.5


@dataclass
class NoiseBaseline:
    kind: str
    seed: int
    n_perm: int
    degenerate: bool
    sharpe_median: float | None
    sharpe_p05: float | None
    sharpe_p95: float | None
    total_return_median: float | None
    max_drawdown_median: float | None
    fraction_ge_strategy: float | None
    equity: pd.Series | None
    note: str

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "n_perm": self.n_perm,
            "degenerate": self.degenerate,
            "sharpe_median": self.sharpe_median,
            "sharpe_p05": self.sharpe_p05,
            "sharpe_p95": self.sharpe_p95,
            "total_return_median": self.total_return_median,
            "max_drawdown_median": self.max_drawdown_median,
            "fraction_ge_strategy": self.fraction_ge_strategy,
            "note": self.note,
        }


class _ShuffledSignals(BaseStrategy):
    """Permute in-window labels. Warmup stays flat so a position cannot leak in."""

    def __init__(self, inner: BaseStrategy, rng: np.random.Generator, start, end):
        self.inner = inner
        self.rng = rng
        self.start = pd.Timestamp(start).normalize()
        self.end = pd.Timestamp(end).normalize()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = self.inner.generate_signals(data)
        signals = signals.reindex(data.index).fillna(0).astype(int)
        out = signals.copy()
        out.loc[out.index < self.start] = 0
        mask = (out.index >= self.start) & (out.index < self.end)
        values = out.loc[mask].to_numpy().copy()
        self.rng.shuffle(values)
        out.loc[mask] = values
        return out


class _RandomEntry(BaseStrategy):
    """Independent long/flat draws. Each name gets its own stream from the seed."""

    def __init__(self, probability: float, seed: int):
        self.probability = float(probability)
        self.seed = int(seed)
        self._calls = 0

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self._calls += 1
        rng = np.random.default_rng((self.seed + self._calls * 1_000_003) % (2**32))
        draws = rng.random(len(data))
        return pd.Series((draws < self.probability).astype(int), index=data.index)


def infer_benchmark(tickers: Sequence[str], market: str | None = None) -> str:
    """``0050.TW`` for a Taiwan book, otherwise ``SPY``."""
    if str(market or "").strip().upper() == "TW":
        return "0050.TW"
    for symbol in tickers:
        upper = str(symbol).upper()
        if upper.endswith(".TW") or upper.endswith(".TWO"):
            return "0050.TW"
    return "SPY"


def buy_and_hold_on_index(close: pd.Series, index, capital: float) -> pd.Series | None:
    """Scale ``close`` onto ``index``, starting at ``capital`` on the first shared date.

    Later dates use the last close at or before that date. Nothing is filled
    from a future print.
    """
    if close is None or len(close) == 0:
        return None
    prices = pd.Series(close, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if prices.empty:
        return None
    prices.index = _day_index(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    stamps = _day_index(index)
    aligned = prices.reindex(prices.index.union(stamps)).sort_index().ffill().reindex(stamps)
    valid = aligned.dropna()
    if valid.empty or float(valid.iloc[0]) <= 0:
        return None
    curve = aligned / float(valid.iloc[0]) * float(capital)
    curve.name = "benchmark"
    return curve


def noise_baseline_single(
    strategy,
    prices: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    start,
    end,
    *,
    initial_capital: float,
    commission: float,
    slippage: float,
    market: str | None,
    backtester_kwargs: Mapping | None,
    risk_free_rate: float,
    strategy_sharpe: float | None,
    n_perm: int = _DEFAULT_PERMS,
    seed: int = _DEFAULT_SEED,
) -> NoiseBaseline:
    """Shuffle labels, or enter at random when the strategy manages its own book."""
    bookish = bool(getattr(strategy, "manages_book", False))
    kind = "random_entry" if bookish else "shuffled_label"
    if not bookish and _signals_constant(strategy, prices, symbols, start, end):
        return _degenerate(kind, seed, n_perm, strategy_sharpe)
    rng = np.random.default_rng(seed)
    sharpes: list[float] = []
    totals: list[float] = []
    drawdowns: list[float] = []
    equities: list[pd.Series] = []
    for _ in range(int(n_perm)):
        if bookish:
            wrapped = _RandomEntry(_RANDOM_LONG, int(rng.integers(0, 2**31 - 1)))
        else:
            wrapped = _ShuffledSignals(strategy, rng, start, end)
        equity = _simulate(
            wrapped,
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
        scored = _score(equity, risk_free_rate)
        if scored is None:
            continue
        sharpe, total, drawdown = scored
        sharpes.append(sharpe)
        totals.append(total)
        drawdowns.append(drawdown)
        equities.append(equity)
    return _assemble(
        kind,
        seed,
        n_perm,
        sharpes,
        totals,
        drawdowns,
        equities,
        strategy_sharpe,
        bookish=bookish,
    )


def noise_baseline_walk_forward(
    prices: Mapping[str, pd.DataFrame],
    result,
    *,
    strategy_sharpe: float | None,
    n_perm: int = _DEFAULT_PERMS,
    seed: int = _DEFAULT_SEED,
) -> NoiseBaseline:
    """Same noise idea on each test window, then stitched like the out-of-sample book."""
    from walk_forward import WalkForwardError, _equity_for, _stitch

    context = result.context
    active = [fold for fold in result.folds if fold.params is not None and fold.symbols]
    if not active:
        return _degenerate("shuffled_label", seed, n_perm, strategy_sharpe, empty=True)
    bookish = False
    constant = True
    for fold in active:
        strategy = context.factory(dict(fold.params))
        if getattr(strategy, "manages_book", False):
            bookish = True
            constant = False
            break
        if not _signals_constant(strategy, prices, fold.symbols, fold.test_start, fold.test_end):
            constant = False
    kind = "random_entry" if bookish else "shuffled_label"
    if constant and not bookish:
        baseline = _degenerate(kind, seed, n_perm, strategy_sharpe)
        baseline.equity = result.oos_equity
        return baseline

    rng = np.random.default_rng(seed)
    sharpes: list[float] = []
    totals: list[float] = []
    drawdowns: list[float] = []
    equities: list[pd.Series] = []
    for _ in range(int(n_perm)):
        pieces: list[pd.Series] = []
        failed = False
        for fold in result.folds:
            if fold.params is None or not fold.symbols:
                pieces.append(fold.test_equity)
                continue
            strategy = context.factory(dict(fold.params))
            try:
                if getattr(strategy, "manages_book", False):
                    piece = _simulate(
                        _RandomEntry(_RANDOM_LONG, int(rng.integers(0, 2**31 - 1))),
                        prices,
                        fold.symbols,
                        fold.test_start,
                        fold.test_end,
                        initial_capital=context.initial_capital,
                        commission=context.commission,
                        slippage=context.slippage,
                        market=context.backtester_kwargs.get("market"),
                        backtester_kwargs=context.backtester_kwargs,
                    )
                else:
                    stream = rng

                    def _permute(signals, start=fold.test_start, end=fold.test_end, stream=stream):
                        out = signals.copy()
                        mask = (out.index >= start) & (out.index < end)
                        values = out.loc[mask].to_numpy().copy()
                        stream.shuffle(values)
                        out.loc[mask] = values
                        return out

                    piece = _equity_for(
                        prices,
                        fold.symbols,
                        context,
                        fold.params,
                        fold.test_start,
                        fold.test_end,
                        context.universe,
                        signal_transform=_permute,
                    )
            except (WalkForwardError, ValueError):
                failed = True
                break
            pieces.append(piece)
        if failed or not pieces:
            continue
        try:
            stitched = _stitch(pieces, context.initial_capital)
        except WalkForwardError:
            continue
        scored = _score(stitched, context.risk_free_rate)
        if scored is None:
            continue
        sharpe, total, drawdown = scored
        sharpes.append(sharpe)
        totals.append(total)
        drawdowns.append(drawdown)
        equities.append(stitched)
    return _assemble(
        kind,
        seed,
        n_perm,
        sharpes,
        totals,
        drawdowns,
        equities,
        strategy_sharpe,
        bookish=bookish,
    )


def _assemble(kind, seed, n_perm, sharpes, totals, drawdowns, equities, strategy_sharpe, *, bookish: bool) -> NoiseBaseline:
    if not sharpes:
        return NoiseBaseline(
            kind=kind,
            seed=int(seed),
            n_perm=int(n_perm),
            degenerate=False,
            sharpe_median=None,
            sharpe_p05=None,
            sharpe_p95=None,
            total_return_median=None,
            max_drawdown_median=None,
            fraction_ge_strategy=None,
            equity=None,
            note="噪音路徑沒有留下可評的淨值。",
        )
    values = np.asarray(sharpes, dtype=float)
    median = float(np.median(values))
    closest = int(np.argmin(np.abs(values - median)))
    fraction = None
    if strategy_sharpe is not None and math.isfinite(float(strategy_sharpe)):
        fraction = float(np.mean(values >= float(strategy_sharpe) - 1e-12))
    if bookish:
        note = (
            f"這條策略自己決定權重，沒有單一名稱的 0/1 訊號。"
            f"噪音對照改成各檔獨立、以 {_RANDOM_LONG:.0%} 的日子持有，成本與成交不變。"
            f"種子 {int(seed)}，{len(values)} 次。"
        )
    else:
        note = (
            "區間內的 0/1 訊號被打亂後，仍用下一根開盤與同一套成本成交。"
            f"種子 {int(seed)}，{len(values)} 次。圖上是最接近中位數夏普的那一條。"
        )
    if fraction is not None:
        note += f" 其中 {fraction:.0%} 的夏普不低於策略。"
    return NoiseBaseline(
        kind=kind,
        seed=int(seed),
        n_perm=int(n_perm),
        degenerate=False,
        sharpe_median=median,
        sharpe_p05=float(np.quantile(values, 0.05)),
        sharpe_p95=float(np.quantile(values, 0.95)),
        total_return_median=float(np.median(np.asarray(totals, dtype=float))),
        max_drawdown_median=float(np.median(np.asarray(drawdowns, dtype=float))),
        fraction_ge_strategy=fraction,
        equity=equities[closest],
        note=note,
    )


def _degenerate(kind: str, seed: int, n_perm: int, strategy_sharpe: float | None, empty: bool = False) -> NoiseBaseline:
    sharpe = None if strategy_sharpe is None or not math.isfinite(float(strategy_sharpe)) else float(strategy_sharpe)
    if empty:
        note = "這一輪沒有持倉可以打亂。"
    else:
        note = "訊號在區間內沒有變化，打亂之後還是同一條路徑。"
    return NoiseBaseline(
        kind=kind,
        seed=int(seed),
        n_perm=int(n_perm),
        degenerate=True,
        sharpe_median=sharpe,
        sharpe_p05=sharpe,
        sharpe_p95=sharpe,
        total_return_median=None,
        max_drawdown_median=None,
        fraction_ge_strategy=1.0 if sharpe is not None else None,
        equity=None,
        note=note,
    )


def _signals_constant(strategy, prices, symbols, start, end) -> bool:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    saw = False
    for symbol in symbols:
        frame = prices.get(symbol)
        if frame is None or frame.empty:
            continue
        signals = strategy.generate_signals(frame)
        signals = signals.reindex(frame.index).fillna(0)
        window = signals.loc[(signals.index >= start_day) & (signals.index < end_day)]
        if window.empty:
            continue
        saw = True
        if int(window.nunique(dropna=True)) > 1:
            return False
    return saw


def _simulate(
    strategy,
    prices,
    symbols,
    start,
    end,
    *,
    initial_capital,
    commission,
    slippage,
    market,
    backtester_kwargs,
) -> pd.Series:
    book = simulate_signals(
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
    return book.equity


def _score(equity: pd.Series | None, risk_free_rate: float):
    if equity is None or len(equity.dropna()) < 2:
        return None
    clean = equity.dropna().astype(float).sort_index()
    _display, raw = calculate_metrics(
        clean,
        clean.pct_change().dropna(),
        risk_free_rate=risk_free_rate,
    )
    sharpe = float(raw["sharpe"])
    if not math.isfinite(sharpe):
        return None
    return sharpe, float(raw["total_return"]), float(raw["max_drawdown"])


def _day_index(index) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex(pd.to_datetime(index))
    if stamps.tz is not None:
        stamps = pd.DatetimeIndex(stamps.strftime("%Y-%m-%d"))
    return stamps.normalize()
