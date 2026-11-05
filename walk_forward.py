"""Rolling walk-forward with a calendar embargo and a locked holdout.

Intervals follow the price feed: half-open ``[start, end)``. Fold ``k`` is

    origin     = start + k * test_months
    train      = [origin, origin + train_months)
    embargo    = [train_end, train_end + gap_days)     # calendar days
    test       = [train_end + gap_days, test_start + test_months)

The next origin is one test window later, so the test windows meet and do
not share a date. A later fold's train may cover an earlier fold's test;
that span is already in the past when the later fold is decided. A tail
that cannot hold a full test window is dropped rather than shortened.

``holdout_months`` reserves ``[end - holdout_months, end)``. Folds are cut
only from ``[start, holdout_start)``. Nothing in this module scores the
holdout unless ``evaluate_locked_holdout(..., open_holdout=True)`` is
called. That function freezes the parameters chosen on the last fold that
actually selected some and does not search again. Leaving
``holdout_months`` at 0 means there is no reserved tail.

Parameter search, when the grid has more than one point, sees prices
strictly inside that fold's train window (plus an optional warmup that
ends before the train starts). The embargo and the test are not in the
objective. Buy & Hold has a one-point grid, so each fold is evaluated and
not searched. The score is a metric from ``metrics.calculate_metrics`` on
the cost-aware equity from ``Backtester``. Sharpe is the default. Ties
keep the earlier grid row.

Screens, when supplied, are called at the train window's first day. The
same names are used for that fold's train and test. ``pit_screen`` reads
``fetch_fundamentals_pit`` and does not call the live Yahoo snapshot.
Opening the holdout re-screens at the holdout start; the parameters stay
the ones frozen above.

Each test book starts in cash. Test-window returns are compounded into one
out-of-sample equity curve. Gap days are not inserted as flat returns.
Multiple-testing corrections (Deflated Sharpe, Holm) are out of scope.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go

from backtester import Backtester
from fundamentals_pit import fetch_fundamentals_pit
from metrics import calculate_metrics
from screener import apply_factor_score, filter_stocks
from strategies import BaseStrategy, build_strategy
from strategies import canonical_strategy as _strategy_key

DEFAULT_TRAIN_MONTHS = 24
DEFAULT_TEST_MONTHS = 6
# Calendar days, not sessions. Twenty-one days is about a month of embargo
# between the last train close and the first test day.
DEFAULT_GAP_DAYS = 21
DEFAULT_WARMUP_BARS = 252
DEFAULT_METRIC = "sharpe"

SELECTION_METRICS = ("sharpe", "sortino", "calmar", "cagr", "total_return")

# Modest grids. The UI sliders are the single-run parameters; walk-forward
# does not search around whatever the slider happens to show.
DEFAULT_PARAM_GRIDS: dict[str, tuple[dict, ...]] = {
    "buy_and_hold": ({},),
    "ma_cross": (
        {"short_window": 10, "long_window": 30, "ma_type": "SMA"},
        {"short_window": 20, "long_window": 60, "ma_type": "SMA"},
        {"short_window": 50, "long_window": 100, "ma_type": "SMA"},
        {"short_window": 20, "long_window": 60, "ma_type": "EMA"},
    ),
    "rsi": (
        {"period": 14, "oversold": 30, "overbought": 70},
        {"period": 14, "oversold": 20, "overbought": 80},
        {"period": 7, "oversold": 30, "overbought": 70},
        {"period": 21, "oversold": 30, "overbought": 70},
    ),
    "bollinger": (
        {"period": 20, "std_dev": 2.0},
        {"period": 20, "std_dev": 2.5},
        {"period": 20, "std_dev": 1.5},
        {"period": 10, "std_dev": 2.0},
    ),
    "macd": (
        {"fast": 12, "slow": 26, "signal_period": 9},
        {"fast": 8, "slow": 21, "signal_period": 9},
        {"fast": 12, "slow": 26, "signal_period": 5},
        {"fast": 5, "slow": 35, "signal_period": 5},
    ),
    # Shorter momentum than a single full-sample run: each fold only sees
    # the warmup bars in front of its own window.
    "trend_stop": (
        {"trend_window": 50, "ma_type": "SMA", "stop_mode": "percent", "stop_pct": 0.08},
        {"trend_window": 100, "ma_type": "SMA", "stop_mode": "percent", "stop_pct": 0.10},
        {"trend_window": 50, "ma_type": "SMA", "stop_mode": "atr", "atr_period": 14, "atr_mult": 2.5},
        {"trend_window": 100, "ma_type": "SMA", "stop_mode": "atr", "atr_period": 14, "atr_mult": 3.0},
    ),
    "cross_section": (
        {
            "top_n": 2,
            "weighting": "equal",
            "rebalance_months": 3,
            "momentum_lookback": 63,
            "momentum_skip": 0,
            "momentum_weight": 0.5,
            "value_weight": 0.5,
        },
        {
            "top_n": 3,
            "weighting": "equal",
            "rebalance_months": 3,
            "momentum_lookback": 126,
            "momentum_skip": 21,
            "momentum_weight": 1.0,
            "value_weight": 0.5,
        },
        {
            "top_n": 2,
            "weighting": "risk",
            "rebalance_months": 1,
            "momentum_lookback": 63,
            "momentum_skip": 0,
            "momentum_weight": 0.5,
            "value_weight": 0.5,
        },
    ),
    # Weights stay 70/30. The grid only changes the sleeves, not the mix.
    "core_satellite": (
        {
            "core_weight": 0.70,
            "satellite_weight": 0.30,
            "core": "buy_and_hold",
            "satellite": "trend_stop",
            "satellite_params": {"trend_window": 50, "stop_mode": "percent", "stop_pct": 0.08},
        },
        {
            "core_weight": 0.70,
            "satellite_weight": 0.30,
            "core": "buy_and_hold",
            "satellite": "trend_stop",
            "satellite_params": {"trend_window": 100, "stop_mode": "percent", "stop_pct": 0.12},
        },
        {
            "core_weight": 0.70,
            "satellite_weight": 0.30,
            "core": "trend_stop",
            "core_params": {"trend_window": 100, "stop_mode": "percent", "stop_pct": 0.10},
            "satellite": "ma_cross",
            "satellite_params": {"short_window": 20, "long_window": 60, "ma_type": "SMA"},
        },
    ),
}

_COLOR_EQUITY = "#58A6FF"
_COLOR_DRAWDOWN = "#F85149"
_COLOR_DRAWDOWN_FILL = "rgba(248, 81, 73, 0.25)"

ScreenFn = Callable[[pd.Timestamp, Sequence[str]], Sequence[str]]
StrategyFactory = Callable[[Mapping], BaseStrategy]


class WalkForwardError(ValueError):
    """The requested windows cannot be built or priced."""


class HoldoutLockedError(RuntimeError):
    """Holdout evaluation was requested without an explicit open."""


@dataclass(frozen=True)
class FoldWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def as_of(self) -> pd.Timestamp:
        """Decision date for a screen: the first day of the train window."""
        return self.train_start


@dataclass(frozen=True)
class SampleSplit:
    folds: tuple[FoldWindow, ...]
    holdout_start: pd.Timestamp | None
    holdout_end: pd.Timestamp | None
    research_end: pd.Timestamp
    gap_days: int


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    as_of: pd.Timestamp
    params: dict | None
    searched: bool
    train_score: float | None
    test_sharpe: float | None
    test_total_return: float | None
    symbols: list[str]
    note: str
    test_equity: pd.Series = field(repr=False)


@dataclass
class HoldoutResult:
    start: pd.Timestamp
    end: pd.Timestamp
    as_of: pd.Timestamp
    params: dict
    symbols: list[str]
    equity: pd.Series
    returns: pd.Series
    metrics_display: dict
    metrics: dict
    equity_figure: go.Figure
    drawdown_figure: go.Figure


@dataclass
class _RunContext:
    strategy: str
    factory: StrategyFactory
    screen: ScreenFn | None
    universe: tuple[str, ...]
    initial_capital: float
    commission: float
    slippage: float
    risk_free_rate: float
    warmup_bars: int
    backtester_kwargs: dict
    benchmark: pd.Series | None
    param_grid: tuple[dict, ...]


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    oos_equity: pd.Series
    oos_returns: pd.Series
    metrics_display: dict
    metrics: dict
    equity_figure: go.Figure
    drawdown_figure: go.Figure
    metric: str
    gap_days: int
    train_months: int
    test_months: int
    holdout_start: pd.Timestamp | None
    holdout_end: pd.Timestamp | None
    holdout: HoldoutResult | None
    strategy: str
    context: _RunContext = field(repr=False)


def canonical_strategy(name: str) -> str:
    try:
        return _strategy_key(name)
    except ValueError as exc:
        raise WalkForwardError(str(exc)) from exc


def default_param_grid(strategy: str) -> tuple[dict, ...]:
    """Grid used when the caller does not pass one.

    Buy & Hold is a single empty parameter set: there is nothing to search.
    """
    return tuple(dict(row) for row in DEFAULT_PARAM_GRIDS[canonical_strategy(strategy)])


def make_strategy(strategy: str, params: Mapping | None = None) -> BaseStrategy:
    try:
        return build_strategy(strategy, params)
    except ValueError as exc:
        raise WalkForwardError(str(exc)) from exc


def pit_screen(
    *,
    pe_max: float | None = None,
    roe_min: float | None = None,
    rev_growth_min: float | None = None,
    mcap_min: float | None = None,
    mcap_max: float | None = None,
    active_factors: Mapping[str, float] | None = None,
    top_n: int | None = None,
    method: str = "zscore",
    lag_days: int | None = None,
    use_thresholds: bool = True,
) -> ScreenFn:
    """Screen at a fold's ``as_of`` with point-in-time fundamentals.

    The returned callable never reads the live Yahoo snapshot. Price factors
    (momentum, volatility) stop on ``as_of`` via ``apply_factor_score``.
    """
    factors = {
        str(name): float(weight)
        for name, weight in (active_factors or {}).items()
        if weight
    }

    def screen(as_of, symbols: Sequence[str]) -> list[str]:
        names = [str(symbol) for symbol in symbols]
        frame = fetch_fundamentals_pit(names, as_of, lag_days=lag_days)
        if use_thresholds:
            frame = filter_stocks(
                frame,
                pe_max=pe_max,
                roe_min=roe_min,
                rev_growth_min=rev_growth_min,
                mcap_min=mcap_min,
                mcap_max=mcap_max,
            )
        picked = frame["symbol"].tolist()
        if not factors:
            return picked
        if not picked:
            return []
        limit = len(picked) if top_n is None else min(int(top_n), len(picked))
        if limit <= 0:
            return []
        ranked = apply_factor_score(
            frame,
            as_of=as_of,
            active_factors=factors,
            top_n=limit,
            method=method,
        )
        return ranked["symbol"].tolist()

    return screen


def build_folds(
    start,
    end,
    *,
    train_months: int = DEFAULT_TRAIN_MONTHS,
    test_months: int = DEFAULT_TEST_MONTHS,
    gap_days: int = DEFAULT_GAP_DAYS,
    holdout_months: int = 0,
) -> SampleSplit:
    """Cut ``[start, end)`` into rolling folds and an optional locked tail."""
    start_day = _day(start)
    end_day = _day(end)
    train_months = _as_int("train_months", train_months, 1)
    test_months = _as_int("test_months", test_months, 1)
    gap_days = _as_int("gap_days", gap_days, 1)
    holdout_months = _as_int("holdout_months", holdout_months, 0)
    if end_day <= start_day:
        raise WalkForwardError("end must be after start")

    if holdout_months:
        holdout_start = _add_months(end_day, -holdout_months)
        if holdout_start <= start_day:
            raise WalkForwardError("holdout_months covers the whole sample")
        research_end = holdout_start
        holdout_end = end_day
    else:
        holdout_start = None
        holdout_end = None
        research_end = end_day

    folds: list[FoldWindow] = []
    for k in range(10_000):
        origin = _add_months(start_day, test_months * k)
        train_end = _add_months(origin, train_months)
        test_start = train_end + pd.Timedelta(days=gap_days)
        test_end = _add_months(test_start, test_months)
        if test_end > research_end or origin >= research_end:
            break
        folds.append(
            FoldWindow(
                fold_id=k,
                train_start=origin,
                train_end=train_end,
                test_start=_day(test_start),
                test_end=test_end,
            )
        )
    return SampleSplit(
        folds=tuple(folds),
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        research_end=research_end,
        gap_days=gap_days,
    )


def run_walk_forward(
    prices: Mapping[str, pd.DataFrame],
    strategy: str,
    start,
    end,
    *,
    train_months: int = DEFAULT_TRAIN_MONTHS,
    test_months: int = DEFAULT_TEST_MONTHS,
    gap_days: int = DEFAULT_GAP_DAYS,
    holdout_months: int = 0,
    open_holdout: bool = False,
    metric: str = DEFAULT_METRIC,
    param_grid: Sequence[Mapping] | None = None,
    strategy_factory: StrategyFactory | None = None,
    screen: ScreenFn | None = None,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.04,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
    benchmark: pd.Series | None = None,
    backtester_kwargs: Mapping | None = None,
) -> WalkForwardResult:
    """Search on each train window, evaluate on that fold's test, stitch OOS.

    ``open_holdout`` defaults to false. When it is true and a holdout was
    reserved, the holdout is reported on ``result.holdout`` only. It is not
    appended to the out-of-sample curve and it is not used to pick parameters.
    """
    if open_holdout and _as_int("holdout_months", holdout_months, 0) <= 0:
        raise WalkForwardError(
            "no holdout was reserved; set holdout_months > 0 before opening it"
        )
    metric_name = str(metric).strip().lower()
    if metric_name not in SELECTION_METRICS:
        raise WalkForwardError(
            f"metric must be one of {', '.join(SELECTION_METRICS)}"
        )
    warmup_bars = _as_int("warmup_bars", warmup_bars, 0)
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise WalkForwardError("initial_capital must be a finite number > 0")

    if strategy_factory is None:
        key = canonical_strategy(strategy)

        def factory(params, name=key):
            return make_strategy(name, params)

        grid = default_param_grid(key) if param_grid is None else _copy_grid(param_grid)
        if key == "buy_and_hold":
            # One empty row. A caller-supplied grid would imply a search
            # this strategy does not have.
            grid = ({},)
    else:
        if param_grid is None:
            raise WalkForwardError("strategy_factory requires param_grid")
        key = str(strategy)
        factory = strategy_factory
        grid = _copy_grid(param_grid)
    if not grid:
        raise WalkForwardError("param_grid is empty")

    prepared = _prepare_prices(prices)
    if not prepared:
        raise WalkForwardError("prices are empty")
    universe = tuple(prepared)

    split = build_folds(
        start,
        end,
        train_months=train_months,
        test_months=test_months,
        gap_days=gap_days,
        holdout_months=holdout_months,
    )
    if not split.folds:
        raise WalkForwardError(
            "not enough room for one full train window, embargo, and test window"
        )

    bench = None if benchmark is None else _prepare_benchmark(benchmark)
    context = _RunContext(
        strategy=key,
        factory=factory,
        screen=screen,
        universe=universe,
        initial_capital=float(initial_capital),
        commission=float(commission),
        slippage=float(slippage),
        risk_free_rate=float(risk_free_rate),
        warmup_bars=warmup_bars,
        backtester_kwargs=dict(backtester_kwargs or {}),
        benchmark=bench,
        param_grid=grid,
    )

    folds: list[FoldResult] = []
    pieces: list[pd.Series] = []
    for window in split.folds:
        fold = _run_fold(prepared, window, context, metric_name)
        folds.append(fold)
        pieces.append(fold.test_equity)

    oos_equity = _stitch(pieces, context.initial_capital)
    oos_returns = oos_equity.pct_change().dropna()
    display, raw = _metrics(oos_equity, oos_returns, context)
    result = WalkForwardResult(
        folds=folds,
        oos_equity=oos_equity,
        oos_returns=oos_returns,
        metrics_display=display,
        metrics=raw,
        equity_figure=equity_figure(oos_equity, "樣本外淨值"),
        drawdown_figure=drawdown_figure(oos_equity, "樣本外回撤"),
        metric=metric_name,
        gap_days=split.gap_days,
        train_months=int(train_months),
        test_months=int(test_months),
        holdout_start=split.holdout_start,
        holdout_end=split.holdout_end,
        holdout=None,
        strategy=key,
        context=context,
    )
    if open_holdout:
        result.holdout = evaluate_locked_holdout(
            result, prepared, open_holdout=True
        )
    return result


def evaluate_locked_holdout(
    result: WalkForwardResult,
    prices: Mapping[str, pd.DataFrame],
    *,
    open_holdout: bool,
) -> HoldoutResult:
    """Score the reserved tail with parameters already chosen.

    This is the only entry point that reads holdout prices. It does not
    grid-search. ``open_holdout`` must be true; a default or a forgotten
    flag raises ``HoldoutLockedError`` so the tail is not a silent extra fold.
    """
    if not open_holdout:
        raise HoldoutLockedError(
            "Holdout stays locked unless open_holdout=True. "
            "The report uses parameters chosen before this interval and "
            "does not search. Do not feed it back into the grid or the windows."
        )
    if result.holdout_start is None or result.holdout_end is None:
        raise WalkForwardError("no holdout was reserved")

    prepared = _prepare_prices(prices)
    params = _frozen_params(result)
    symbols = _names_for(result.context, result.holdout_start, result.context.universe)
    equity = _equity_for(
        prepared,
        symbols,
        result.context,
        params,
        result.holdout_start,
        result.holdout_end,
        result.context.universe,
    )
    returns = equity.pct_change().dropna()
    display, raw = _metrics(equity, returns, result.context)
    return HoldoutResult(
        start=result.holdout_start,
        end=result.holdout_end,
        as_of=result.holdout_start,
        params=params,
        symbols=list(symbols),
        equity=equity,
        returns=returns,
        metrics_display=display,
        metrics=raw,
        equity_figure=equity_figure(equity, "鎖定 holdout 淨值"),
        drawdown_figure=drawdown_figure(equity, "鎖定 holdout 回撤"),
    )


def equity_figure(equity: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity,
            name="淨值",
            line=dict(color=_COLOR_EQUITY, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.06)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="投資組合價值",
        template="plotly_dark",
        height=420,
        margin=dict(t=60, b=40),
    )
    return fig


def drawdown_figure(equity: pd.Series, title: str) -> go.Figure:
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name="回撤",
            line=dict(color=_COLOR_DRAWDOWN, width=1.5),
            fill="tozeroy",
            fillcolor=_COLOR_DRAWDOWN_FILL,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="回撤 (%)",
        template="plotly_dark",
        height=240,
        margin=dict(t=50, b=40),
    )
    return fig


def _run_fold(
    prices: Mapping[str, pd.DataFrame],
    window: FoldWindow,
    context: _RunContext,
    metric: str,
) -> FoldResult:
    symbols = _names_for(context, window.as_of, context.universe)
    note = ""
    searched = False
    train_score: float | None = None
    params: dict | None = None

    if not symbols:
        note = "沒有標的通過這一折的篩選，測試段以現金計"
        equity = _cash_on_calendar(
            prices, context.universe, window.test_start, window.test_end, context.initial_capital
        )
    else:
        searched = len(context.param_grid) > 1
        if searched:
            params, train_score = _select(prices, symbols, window, context, metric)
        else:
            params = dict(context.param_grid[0])
        equity = _equity_for(
            prices,
            symbols,
            context,
            params,
            window.test_start,
            window.test_end,
            context.universe,
        )

    _display, raw = _metrics(equity, equity.pct_change().dropna(), context, with_benchmark=False)
    return FoldResult(
        fold_id=window.fold_id,
        train_start=window.train_start,
        train_end=window.train_end,
        test_start=window.test_start,
        test_end=window.test_end,
        as_of=window.as_of,
        params=None if params is None else dict(params),
        searched=searched,
        train_score=train_score,
        test_sharpe=raw.get("sharpe"),
        test_total_return=raw.get("total_return"),
        symbols=list(symbols),
        note=note,
        test_equity=equity,
    )


def _select(
    prices: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    window: FoldWindow,
    context: _RunContext,
    metric: str,
) -> tuple[dict, float]:
    best_params: dict | None = None
    best_score = -math.inf
    for row in context.param_grid:
        try:
            equity = _equity_for(
                prices,
                symbols,
                context,
                row,
                window.train_start,
                window.train_end,
                symbols,
            )
            _display, raw = _metrics(
                equity, equity.pct_change().dropna(), context, with_benchmark=False
            )
            score = float(raw[metric])
            if not math.isfinite(score):
                score = -math.inf
        except WalkForwardError:
            score = -math.inf
        if best_params is None or score > best_score + 1e-12:
            best_score = score
            best_params = dict(row)
    if best_params is None:
        raise WalkForwardError("param_grid is empty")
    reported = None if best_score == -math.inf else best_score
    return best_params, -math.inf if reported is None else reported


def _equity_for(
    prices: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    context: _RunContext,
    params: Mapping,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    calendar_symbols: Sequence[str],
) -> pd.Series:
    """Equal-weight book. Signals before ``window_start`` are cleared.

    Warmup bars, if any, sit strictly before the window so an indicator can
    look back. The embargo is not warmup for a train window: train warmup
    stops at ``train_start``, and the frame itself stops before ``train_end``.
    """
    strategy = context.factory(dict(params))
    if getattr(strategy, "manages_book", False):
        return _book_equity(
            prices, symbols, strategy, context, window_start, window_end, calendar_symbols
        )
    ready: list[tuple[str, pd.DataFrame, pd.Series]] = []
    for symbol in symbols:
        frame = prices.get(symbol)
        if frame is None or frame.empty:
            continue
        history = _window_frame(frame, window_start, window_end, context.warmup_bars)
        in_window = (history.index >= window_start) & (history.index < window_end)
        if not bool(in_window.any()):
            continue
        signals = strategy.generate_signals(history)
        signals = signals.reindex(history.index).fillna(0).astype("int64")
        signals = signals.copy()
        signals.loc[signals.index < window_start] = 0
        ready.append((symbol, history, signals))

    if not ready:
        return _cash_on_calendar(
            prices, calendar_symbols, window_start, window_end, context.initial_capital
        )

    per = context.initial_capital / len(ready)
    valued: list[pd.Series] = []
    for symbol, history, signals in ready:
        bt = Backtester(
            initial_capital=per,
            commission=context.commission,
            slippage=context.slippage,
            **context.backtester_kwargs,
        )
        portfolio, _trades = bt.run(history, signals, symbol=symbol)
        valued.append(
            portfolio.loc[
                (portfolio.index >= window_start) & (portfolio.index < window_end),
                "portfolio_value",
            ].rename(symbol)
        )
        if valued[-1].empty:
            raise WalkForwardError(f"{symbol} produced no marks in the window")

    wide = pd.concat(valued, axis=1).sort_index().ffill().fillna(per)
    equity = wide.sum(axis=1)
    equity.name = "equity"
    return equity


def _book_equity(
    prices: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    strategy,
    context: _RunContext,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    calendar_symbols: Sequence[str],
) -> pd.Series:
    """A book strategy gets the whole sleeve, not one all-in account per name.

    Prices stop at ``window_end``. Warmup bars, if any, sit strictly before
    ``window_start``. The core symbol of a core-satellite sleeve is included
    in the price map even when a screen did not keep it in ``symbols``.
    """
    wanted = [str(symbol) for symbol in symbols]
    extra = getattr(strategy, "core_symbol", None)
    if extra and extra not in wanted:
        wanted.append(str(extra))
    windowed: dict[str, pd.DataFrame] = {}
    for symbol in wanted:
        frame = prices.get(symbol)
        if frame is None or frame.empty:
            continue
        history = _window_frame(frame, window_start, window_end, context.warmup_bars)
        if not history.empty:
            windowed[symbol] = history
    if not windowed:
        return _cash_on_calendar(
            prices, calendar_symbols, window_start, window_end, context.initial_capital
        )
    try:
        book = strategy.run_book(
            windowed,
            list(symbols),
            window_start,
            window_end,
            initial_capital=context.initial_capital,
            commission=context.commission,
            slippage=context.slippage,
            backtester_kwargs=context.backtester_kwargs,
        )
    except ValueError as exc:
        raise WalkForwardError(str(exc)) from exc
    equity = book.equity.dropna().astype(float).sort_index()
    equity = equity.loc[(equity.index >= window_start) & (equity.index < window_end)]
    equity = equity[~equity.index.duplicated(keep="last")]
    if len(equity) < 2:
        raise WalkForwardError("book produced fewer than two marks in the window")
    equity.name = "equity"
    return equity


def _names_for(context: _RunContext, as_of: pd.Timestamp, universe: Sequence[str]) -> list[str]:
    if context.screen is None:
        return list(universe)
    picked = [str(symbol) for symbol in context.screen(as_of, list(universe))]
    allowed = set(universe)
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in picked:
        if symbol in allowed and symbol not in seen:
            ordered.append(symbol)
            seen.add(symbol)
    return ordered


def _frozen_params(result: WalkForwardResult) -> dict:
    """Last parameter set a fold actually used. Never the first grid row by default."""
    for fold in reversed(result.folds):
        if fold.params is not None:
            return dict(fold.params)
    raise WalkForwardError(
        "no fold produced parameters, so the holdout is not scored with an arbitrary grid row"
    )


def _window_frame(
    frame: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    warmup_bars: int,
) -> pd.DataFrame:
    window = frame.loc[(frame.index >= window_start) & (frame.index < window_end)]
    if warmup_bars > 0:
        pre = frame.loc[frame.index < window_start].tail(warmup_bars)
        history = pd.concat([pre, window])
    else:
        history = window
    history = history[~history.index.duplicated(keep="last")].sort_index()
    if not history.empty and history.index.max() >= window_end:
        raise WalkForwardError("price slice crossed the window end")
    return history


def _cash_on_calendar(
    prices: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    capital: float,
) -> pd.Series:
    index = pd.DatetimeIndex([])
    for symbol in symbols:
        frame = prices.get(symbol)
        if frame is None or frame.empty:
            continue
        index = index.union(
            frame.loc[(frame.index >= window_start) & (frame.index < window_end)].index
        )
    index = index.sort_values()
    if len(index) < 2:
        raise WalkForwardError(
            f"no price bars in [{window_start.date()}, {window_end.date()})"
        )
    equity = pd.Series(float(capital), index=index, name="equity")
    return equity


def _stitch(pieces: Sequence[pd.Series], capital: float) -> pd.Series:
    parts: list[pd.Series] = []
    level = float(capital)
    for equity in pieces:
        cleaned = equity.dropna().astype(float).sort_index()
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        if len(cleaned) < 2:
            raise WalkForwardError("a test window has fewer than two marks")
        base = float(cleaned.iloc[0])
        if not math.isfinite(base) or base <= 0:
            raise WalkForwardError("test equity must start above zero")
        scaled = cleaned / base * level
        parts.append(scaled if not parts else scaled.iloc[1:])
        level = float(scaled.iloc[-1])
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.name = "equity"
    if len(out) < 2:
        raise WalkForwardError("out-of-sample equity is too short to score")
    return out


def _metrics(equity, returns, context: _RunContext, with_benchmark: bool = True):
    bench = None
    if with_benchmark and context.benchmark is not None:
        bench = context.benchmark.reindex(returns.index)
    return calculate_metrics(
        equity_curve=equity,
        returns=returns,
        benchmark_returns=bench,
        risk_free_rate=context.risk_free_rate,
    )


def _prepare_prices(prices: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    for symbol, frame in prices.items():
        if frame is None or frame.empty:
            continue
        cleaned = frame.copy()
        cleaned.index = _day_index(cleaned.index)
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
        if "Open" not in cleaned.columns or "Close" not in cleaned.columns:
            raise WalkForwardError(f"{symbol} is missing Open or Close")
        prepared[str(symbol)] = cleaned
    return prepared


def _prepare_benchmark(benchmark: pd.Series) -> pd.Series:
    series = benchmark.copy()
    series.index = _day_index(series.index)
    series = series[~series.index.duplicated(keep="last")].sort_index().astype(float)
    return series.pct_change()


def _copy_grid(grid: Sequence[Mapping]) -> tuple[dict, ...]:
    rows = tuple(dict(row) for row in grid)
    if not rows:
        raise WalkForwardError("param_grid is empty")
    return rows


def _day(value) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise WalkForwardError(f"invalid date: {value!r}") from exc
    if pd.isna(stamp):
        raise WalkForwardError(f"invalid date: {value!r}")
    if stamp.tzinfo is not None:
        stamp = pd.Timestamp(stamp.strftime("%Y-%m-%d"))
    return stamp.normalize()


def _day_index(index) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex(pd.to_datetime(index))
    if stamps.tz is not None:
        stamps = pd.DatetimeIndex(stamps.strftime("%Y-%m-%d"))
    return stamps.normalize()


def _add_months(day: pd.Timestamp, months: int) -> pd.Timestamp:
    return _day(day + pd.DateOffset(months=int(months)))


def _as_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool):
        raise WalkForwardError(f"{name} must be an integer >= {minimum}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WalkForwardError(f"{name} must be an integer >= {minimum}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise WalkForwardError(f"{name} must be an integer >= {minimum}")
    whole = int(number)
    if whole < minimum:
        raise WalkForwardError(f"{name} must be an integer >= {minimum}")
    return whole
