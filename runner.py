"""One research path for the CLI and the Streamlit app.

The UI should only collect choices and draw. Fetching, signals, the
backtest, walk-forward, and the equity / drawdown figures live here.
API keys are not a config field; data providers read them from the
environment when a run actually fetches.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from baselines import (
    buy_and_hold_on_index,
    infer_benchmark,
    noise_baseline_single,
    noise_baseline_walk_forward,
)
from books import prepare_frame, run_sleeve
from metrics import calculate_annual_returns, calculate_metrics, calculate_monthly_returns
from multiple_testing import summarize_selection
from report import comparison_equity_figure, comparison_lines, rolling_sharpe_figure, write_report as _write_report
from strategies import build_strategy, canonical_strategy
from walk_forward import (
    default_param_grid,
    drawdown_figure,
    equity_figure,
    pit_screen,
    run_walk_forward,
)

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "token",
    "password",
    "private_key",
    "tiingo_api_key",
    "polygon_api_key",
    "fmp_api_key",
    "shioaji_api_key",
    "shioaji_secret_key",
}

_TOP_LEVEL = {
    "strategy",
    "tickers",
    "universe",
    "start",
    "end",
    "benchmark",
    "initial_capital",
    "capital",
    "commission",
    "slippage",
    "risk_free_rate",
    "market",
    "provider",
    "output_dir",
    "walk_forward",
    "train_months",
    "test_months",
    "gap_days",
    "holdout_months",
    "open_holdout",
    "warmup_bars",
    "params",
    "cross_section",
    "core_satellite",
    "screen",
}


class ConfigError(ValueError):
    """The yaml or the CLI flags cannot be run."""


class ResearchError(RuntimeError):
    """Prices or the resulting book cannot be scored."""


@dataclass
class RunConfig:
    strategy: str
    tickers: list[str]
    start: str
    end: str
    benchmark: str | None = None
    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    risk_free_rate: float = 0.04
    market: str | None = None
    provider: str | None = None
    output_dir: str = "output"
    walk_forward: bool = False
    train_months: int = 24
    test_months: int = 6
    gap_days: int = 21
    holdout_months: int = 0
    open_holdout: bool = False
    warmup_bars: int | None = None
    params: dict = field(default_factory=dict)
    cross_section: dict = field(default_factory=dict)
    core_satellite: dict = field(default_factory=dict)
    use_screener: bool = False
    use_factor_scoring: bool = False
    pe_max: float | None = None
    roe_min: float | None = None
    rev_growth_min: float | None = None
    mcap_min: float | None = None
    active_factors: dict = field(default_factory=dict)
    factor_top_n: int | None = None
    norm_method: str = "zscore"


@dataclass
class ResearchResult:
    config: RunConfig
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics_display: dict
    metrics: dict
    equity_figure: go.Figure
    drawdown_figure: go.Figure
    symbols: list[str]
    annual_returns: pd.Series
    monthly_returns: pd.DataFrame
    benchmark_equity: pd.Series | None
    benchmark_returns: pd.Series | None
    walk_forward: WalkForwardResult | None
    holdings: pd.DataFrame | None
    notes: list[str]
    chart_symbol: str | None
    chart_prices: pd.DataFrame | None
    rolling_sharpe_figure: go.Figure | None = None
    comparison: dict = field(default_factory=dict)
    selection: dict | None = None


def load_run_config(path, overrides: Mapping | None = None) -> RunConfig:
    """Read a yaml file. ``overrides`` wins field by field. Secrets are rejected."""
    payload = _read_yaml(path)
    _reject_secrets(payload)
    unknown = sorted(set(payload) - _TOP_LEVEL)
    if unknown:
        raise ConfigError("unknown config keys: " + ", ".join(unknown))

    data = dict(payload)
    for key, value in dict(overrides or {}).items():
        if value is not None:
            data[key] = value

    strategy = data.get("strategy")
    if not strategy or not str(strategy).strip():
        raise ConfigError("strategy is required")
    try:
        canonical_strategy(str(strategy))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    tickers = _tickers(data.get("tickers", data.get("universe")))
    start = _date_text(data.get("start"), "start")
    end = _date_text(data.get("end"), "end")
    if pd.Timestamp(end) <= pd.Timestamp(start):
        raise ConfigError("end must be after start")

    market = data.get("market")
    market_text = None if market is None or str(market).strip() == "" else str(market).strip()
    if "benchmark" not in data:
        benchmark_text = infer_benchmark(tickers, market_text)
    else:
        benchmark = data.get("benchmark")
        if benchmark is False or benchmark is None or str(benchmark).strip() == "":
            benchmark_text = None
        else:
            benchmark_text = str(benchmark).strip().upper()
    provider = data.get("provider")
    provider_text = None if provider is None or str(provider).strip() == "" else str(provider).strip().lower()

    params = _mapping(data.get("params") or {}, "params")
    cross_section = _mapping(data.get("cross_section") or {}, "cross_section")
    core_satellite = _mapping(data.get("core_satellite") or {}, "core_satellite")
    screen = _mapping(data.get("screen") or {}, "screen")
    factors = _mapping(screen.get("factors") or screen.get("active_factors") or {}, "screen.factors")
    threshold_keys = ("pe_max", "roe_min", "rev_growth_min", "mcap_min")
    use_screener = any(screen.get(key) is not None for key in threshold_keys)

    capital = data.get("initial_capital", data.get("capital", 100_000.0))
    return RunConfig(
        strategy=str(strategy).strip(),
        tickers=tickers,
        start=start,
        end=end,
        benchmark=benchmark_text,
        initial_capital=_number("initial_capital", capital, minimum=0.0, strict=True),
        commission=_number("commission", data.get("commission", 0.001), minimum=0.0),
        slippage=_number("slippage", data.get("slippage", 0.0005), minimum=0.0),
        risk_free_rate=_number("risk_free_rate", data.get("risk_free_rate", 0.04), minimum=0.0),
        market=market_text,
        provider=provider_text,
        output_dir=str(data.get("output_dir") or "output"),
        walk_forward=_as_bool(data.get("walk_forward", False), "walk_forward"),
        train_months=int(_number("train_months", data.get("train_months", 24), minimum=1.0)),
        test_months=int(_number("test_months", data.get("test_months", 6), minimum=1.0)),
        gap_days=int(_number("gap_days", data.get("gap_days", 21), minimum=1.0)),
        holdout_months=int(_number("holdout_months", data.get("holdout_months", 0), minimum=0.0)),
        open_holdout=_as_bool(data.get("open_holdout", False), "open_holdout"),
        warmup_bars=(
            None
            if data.get("warmup_bars") is None
            else int(_number("warmup_bars", data.get("warmup_bars"), minimum=0.0))
        ),
        params=params,
        cross_section=cross_section,
        core_satellite=core_satellite,
        use_screener=use_screener,
        use_factor_scoring=bool(factors),
        pe_max=_optional_number(screen.get("pe_max")),
        roe_min=_optional_number(screen.get("roe_min")),
        rev_growth_min=_optional_number(screen.get("rev_growth_min")),
        mcap_min=_optional_number(screen.get("mcap_min")),
        active_factors={str(name): float(weight) for name, weight in factors.items()},
        factor_top_n=None if screen.get("top_n") is None else int(screen.get("top_n")),
        norm_method=str(screen.get("method") or "zscore"),
    )


def run_research(
    config: RunConfig,
    *,
    prices: Mapping[str, pd.DataFrame] | None = None,
    benchmark_close: pd.Series | None = None,
    fundamentals_loader=None,
) -> ResearchResult:
    """Run the same path the buttons use. Pass ``prices`` in tests to stay offline."""
    key = canonical_strategy(config.strategy)
    start = pd.Timestamp(config.start).normalize()
    end = pd.Timestamp(config.end).normalize()
    symbols = list(config.tickers)
    fetch_symbols = list(symbols)
    core_symbol = None
    if key == "core_satellite":
        core_symbol = config.core_satellite.get("core_symbol") or config.params.get("core_symbol")
        if core_symbol:
            core_symbol = str(core_symbol).strip().upper()
            if core_symbol and core_symbol not in fetch_symbols:
                fetch_symbols.append(core_symbol)

    notes: list[str] = []
    injected = prices is not None
    if prices is None:
        _apply_provider(config.provider)
        fetch_start = _prefetch_start(config, key)
        prices = _fetch_prices(fetch_symbols, fetch_start, config.end)
        if config.benchmark and benchmark_close is None:
            held = prices.get(config.benchmark)
            if held is not None and not held.empty and "Close" in held.columns:
                benchmark_close = held["Close"]
            else:
                benchmark_close = _fetch_close(config.benchmark, fetch_start, config.end)
    else:
        notes.append("這次用呼叫端給的價格，沒有再向資料源要。")

    prepared = {}
    for symbol in fetch_symbols:
        frame = prepare_frame(prices.get(symbol))
        if frame is None:
            notes.append(f"{symbol} 沒有報價，這一段不交易。")
            continue
        prepared[symbol] = frame.loc[frame.index < end]
    if not prepared:
        raise ResearchError("沒有任何標的有報價，回測沒有開始。")

    benchmark_close = _resolve_benchmark_close(
        config, prices, prepared, benchmark_close, injected
    )
    bench_equity, bench_returns = _benchmark(benchmark_close, config.initial_capital)

    if config.walk_forward:
        return _run_walk(
            config,
            key,
            prepared,
            symbols,
            bench_returns,
            benchmark_close,
            fundamentals_loader,
            notes,
        )

    strategy = build_strategy(key, strategy_params(config), fundamentals_loader=fundamentals_loader)
    extra = {}
    if config.market:
        extra["market"] = config.market
    try:
        book = run_sleeve(
            strategy,
            prepared,
            symbols,
            start,
            end,
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
            market=config.market,
            backtester_kwargs=extra,
        )
    except ValueError as exc:
        raise ResearchError(str(exc)) from exc

    notes.extend(book.notes)
    packed = _pack(
        config,
        book.equity,
        book.trades,
        symbols,
        bench_equity,
        bench_returns,
        notes,
        holdings=book.holdings,
        chart_prices=_chart_prices(prepared, symbols, start, end),
        walk_forward=None,
    )
    return _overlay(packed, prepared, benchmark_close, strategy=strategy)


def strategy_params(config: RunConfig) -> dict:
    """Single-run parameters. Walk-forward uses its own small grid instead."""
    key = canonical_strategy(config.strategy)
    if key == "cross_section":
        return {**config.cross_section, **config.params}
    if key == "core_satellite":
        return {**config.core_satellite, **config.params}
    return dict(config.params)


def write_report(result: ResearchResult, output_dir) -> dict[str, Path]:
    """Write the shared pack: equity, drawdown, rolling Sharpe, metrics, baselines."""
    return _write_report(result, output_dir)


def format_metrics(display: Mapping[str, str]) -> str:
    """Plain table for the terminal. The same strings land in metrics.csv."""
    if not display:
        return ""
    width = max(len(str(key)) for key in display)
    lines = [f"{key:<{width}}  {display[key]}" for key in display]
    return "\n".join(lines)


def _run_walk(config, key, prices, symbols, bench_returns, benchmark_close, fundamentals_loader, notes):
    screen = None
    if key == "cross_section":
        notes.append("橫截面策略在每個測試窗裡自己排名，這次沒有再套一層篩選。")
    elif config.use_screener or config.use_factor_scoring:
        screen = pit_screen(
            pe_max=config.pe_max if config.use_screener else None,
            roe_min=config.roe_min if config.use_screener else None,
            rev_growth_min=config.rev_growth_min if config.use_screener else None,
            mcap_min=config.mcap_min if config.use_screener else None,
            active_factors=config.active_factors if config.use_factor_scoring else {},
            top_n=config.factor_top_n if config.use_factor_scoring else None,
            method=config.norm_method,
            use_thresholds=config.use_screener,
        )
        notes.append("Walk-forward 的選股日是該折訓練窗的第一天，用 point-in-time，不是今天的快照。")

    factory = None
    grid = None
    if fundamentals_loader is not None or key in {"cross_section", "core_satellite", "trend_stop"}:
        # Built-in grid. The single-run params block is not what gets searched.
        grid = default_param_grid(key)

        def factory(params, loader=fundamentals_loader):
            return build_strategy(key, params, fundamentals_loader=loader)

    benchmark = None
    if bench_returns is not None:
        # run_walk_forward turns a close series into returns. Rebuild closes
        # from the returns we already have so a caller can pass either.
        benchmark = (1.0 + bench_returns.fillna(0.0)).cumprod()

    extra = {}
    if config.market:
        extra["market"] = config.market
    wf = run_walk_forward(
        prices,
        config.strategy,
        config.start,
        config.end,
        train_months=config.train_months,
        test_months=config.test_months,
        gap_days=config.gap_days,
        holdout_months=config.holdout_months,
        open_holdout=config.open_holdout,
        metric="sharpe",
        param_grid=grid,
        strategy_factory=factory,
        screen=screen,
        initial_capital=config.initial_capital,
        commission=config.commission,
        slippage=config.slippage,
        risk_free_rate=config.risk_free_rate,
        warmup_bars=252 if config.warmup_bars is None else config.warmup_bars,
        benchmark=benchmark,
        backtester_kwargs=extra,
    )
    notes.append("這條淨值是各折測試窗接起來的，不是整段樣本內的績效。")
    if any(fold.searched for fold in wf.folds):
        notes.append("參數用的是內建小格點，不是這份設定裡的單一參數。")
    packed = _pack(
        config,
        wf.oos_equity,
        pd.DataFrame(),
        symbols,
        None,
        bench_returns,
        notes,
        holdings=None,
        chart_prices=_chart_prices(prices, symbols, pd.Timestamp(config.start), pd.Timestamp(config.end)),
        walk_forward=wf,
    )
    # Metrics stay the out-of-sample table, including DSR when the grid was searched.
    # The equity figure is rebuilt with the benchmark and the noise path.
    packed.equity_figure = wf.equity_figure
    packed.drawdown_figure = wf.drawdown_figure
    packed.metrics_display = wf.metrics_display
    packed.metrics = wf.metrics
    packed.selection = wf.selection
    return _overlay(packed, prices, benchmark_close, strategy=None)


def _pack(
    config,
    equity,
    trades,
    symbols,
    bench_equity,
    bench_returns,
    notes,
    *,
    holdings,
    chart_prices,
    walk_forward,
) -> ResearchResult:
    equity = equity.dropna().astype(float).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    equity.name = "equity"
    if len(equity) < 2:
        raise ResearchError("淨值少於兩點，指標算不出來。")
    returns = equity.pct_change().dropna()
    aligned_bench = None
    if bench_returns is not None:
        aligned_bench = bench_returns.reindex(returns.index)
    display, raw = calculate_metrics(
        equity_curve=equity,
        returns=returns,
        benchmark_returns=aligned_bench,
        risk_free_rate=config.risk_free_rate,
    )
    title = "樣本外淨值" if config.walk_forward else "淨值"
    figure = equity_figure(equity, title)
    if bench_equity is not None and not config.walk_forward:
        aligned = bench_equity.reindex(equity.index).ffill()
        figure.add_trace(
            go.Scatter(
                x=aligned.index,
                y=aligned,
                name=config.benchmark or "基準",
                line=dict(color="#F0883E", width=2, dash="dash"),
            )
        )
    chart_symbol = None
    chart_frame = None
    if chart_prices:
        chart_symbol = next(iter(chart_prices))
        chart_frame = chart_prices[chart_symbol]
    return ResearchResult(
        config=config,
        equity=equity,
        returns=returns,
        trades=trades if trades is not None else pd.DataFrame(),
        metrics_display=display,
        metrics=raw,
        equity_figure=figure,
        drawdown_figure=drawdown_figure(equity, "樣本外回撤" if config.walk_forward else "回撤"),
        symbols=list(symbols),
        annual_returns=calculate_annual_returns(equity),
        monthly_returns=calculate_monthly_returns(equity),
        benchmark_equity=bench_equity,
        benchmark_returns=bench_returns,
        walk_forward=walk_forward,
        holdings=holdings,
        notes=list(notes),
        chart_symbol=chart_symbol,
        chart_prices=chart_frame,
    )


def _prefetch_start(config: RunConfig, key: str) -> str:
    """Calendar pad so a slow average or a momentum window exists on the first decision.

    ``warmup_bars: 0`` really means no pad. Omitting it lets a trend or
    cross-section run look back on its own.
    """
    start = pd.Timestamp(config.start)
    if config.walk_forward:
        return (start - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    if config.warmup_bars is not None:
        bars = config.warmup_bars
    elif key == "trend_stop":
        bars = int(strategy_params(config).get("trend_window", 100) or 100)
    elif key == "cross_section":
        params = strategy_params(config)
        bars = int(params.get("momentum_lookback", 252)) + int(params.get("momentum_skip", 21))
    elif key == "core_satellite":
        bars = 120
    else:
        bars = 0
    if bars <= 0:
        return config.start
    return (start - pd.Timedelta(days=int(bars * 1.8) + 10)).strftime("%Y-%m-%d")


def _apply_provider(provider: str | None) -> None:
    if provider:
        os.environ["BACKTEST_DATA_PROVIDER"] = provider


def _fetch_prices(symbols: Sequence[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    from data_fetcher import fetch_data

    frames = {}
    for symbol in symbols:
        frame = fetch_data(symbol, start, end)
        if frame is not None and not frame.empty:
            frames[symbol] = frame
    return frames


def _fetch_close(symbol: str, start: str, end: str) -> pd.Series | None:
    from data_fetcher import fetch_data

    frame = fetch_data(symbol, start, end)
    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    return frame["Close"]


def _overlay(result: ResearchResult, prices, benchmark_close, strategy) -> ResearchResult:
    """Attach the benchmark, the noise path, rolling Sharpe, and selection lines."""
    comparison: dict = {}
    extras = []
    label = result.config.benchmark
    if benchmark_close is not None and label:
        curve = buy_and_hold_on_index(benchmark_close, result.equity.index, result.config.initial_capital)
        if curve is not None:
            scored = _score_curve(curve, result.config.risk_free_rate)
            comparison["benchmark"] = {
                "symbol": label,
                "kind": "buy_and_hold",
                "sharpe": _plain(scored.get("sharpe")),
                "total_return": _plain(scored.get("total_return")),
                "max_drawdown": _plain(scored.get("max_drawdown")),
                "note": (
                    f"{label} 是策略同一組交易日上的收盤價買進持有，不加手續費。"
                    "策略淨值含成本。"
                ),
            }
            extras.append((f"{label} 買進持有", curve, "dash", "#F0883E"))
            result.benchmark_equity = curve
    elif label:
        result.notes.append(f"這次的價格裡沒有 {label}，基準買進持有先空著。")

    if result.walk_forward is not None:
        noise = noise_baseline_walk_forward(
            prices,
            result.walk_forward,
            strategy_sharpe=result.metrics.get("sharpe"),
        )
        if result.selection is None:
            result.selection = result.walk_forward.selection
    else:
        extra = {}
        if result.config.market:
            extra["market"] = result.config.market
        noise = noise_baseline_single(
            strategy,
            prices,
            result.symbols,
            result.config.start,
            result.config.end,
            initial_capital=result.config.initial_capital,
            commission=result.config.commission,
            slippage=result.config.slippage,
            market=result.config.market,
            backtester_kwargs=extra,
            risk_free_rate=result.config.risk_free_rate,
            strategy_sharpe=result.metrics.get("sharpe"),
        )
        result.selection = summarize_selection(
            result.returns,
            [],
            risk_free_rate=result.config.risk_free_rate,
            sample="全樣本",
        )

    comparison["noise"] = noise.as_dict()
    if noise.equity is not None and not noise.degenerate:
        name = "隨機進場" if noise.kind == "random_entry" else "打亂標籤"
        extras.append((name, noise.equity, "dot", "#8B949E"))
    result.comparison = comparison
    title = "淨值"
    if result.equity_figure is not None and result.equity_figure.layout.title.text:
        title = result.equity_figure.layout.title.text
    trades = None if result.config.walk_forward else result.trades
    result.equity_figure = comparison_equity_figure(result.equity, title, extras, trades=trades)
    result.rolling_sharpe_figure = rolling_sharpe_figure(
        result.returns, result.config.risk_free_rate
    )
    merged = dict(result.metrics_display)
    if result.selection:
        merged.update(result.selection.get("display") or {})
    merged.update(comparison_lines(comparison))
    result.metrics_display = merged
    if comparison.get("benchmark"):
        result.notes.append(comparison["benchmark"]["note"])
    if noise.note:
        result.notes.append(noise.note)
    if result.selection and result.selection.get("note"):
        result.notes.append(result.selection["note"])
    return result


def _resolve_benchmark_close(config, fetched, prepared, provided, injected: bool):
    if not config.benchmark:
        return None
    if provided is not None and len(provided) > 0:
        return provided
    for source in (prepared, fetched):
        if not source:
            continue
        frame = source.get(config.benchmark)
        if isinstance(frame, pd.DataFrame) and "Close" in frame.columns and not frame.empty:
            return frame["Close"]
        if isinstance(frame, pd.Series) and len(frame) > 0:
            return frame
    if not injected:
        return _fetch_close(config.benchmark, config.start, config.end)
    return None


def _plain(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _score_curve(equity: pd.Series, risk_free_rate: float) -> dict:
    clean = equity.dropna().astype(float).sort_index()
    if len(clean) < 2:
        return {}
    _display, raw = calculate_metrics(
        clean,
        clean.pct_change().dropna(),
        risk_free_rate=risk_free_rate,
    )
    return raw


def _benchmark(close: pd.Series | None, capital: float):
    if close is None or len(close) == 0:
        return None, None
    prepared = prepare_frame(pd.DataFrame({"Open": close, "Close": close}))
    if prepared is None:
        return None, None
    returns = prepared["Close"].pct_change().dropna()
    equity = (1.0 + returns).cumprod() * float(capital)
    equity.name = "benchmark"
    return equity, returns


def _chart_prices(prices, symbols, start, end) -> dict[str, pd.DataFrame]:
    out = {}
    for symbol in symbols:
        frame = prices.get(symbol)
        if frame is None or frame.empty:
            continue
        view = frame.loc[(frame.index >= start) & (frame.index < end)]
        if not view.empty:
            out[symbol] = view
    return out


def _read_yaml(path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required. pip install -r requirements.txt") from exc
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"config not found: {file}")
    loaded = yaml.safe_load(file.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("config must be a mapping")
    return loaded


def _reject_secrets(node, where: str = "config") -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SECRET_KEYS or normalized.endswith("_api_key") or normalized.endswith("_secret"):
                raise ConfigError(
                    f"{where}.{key} looks like a secret. "
                    "Leave API keys in the environment; do not write them into the yaml."
                )
            _reject_secrets(value, f"{where}.{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _reject_secrets(item, f"{where}[{index}]")


def _tickers(value) -> list[str]:
    if value is None:
        raise ConfigError("tickers is required")
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    else:
        raise ConfigError("tickers must be a list or a comma-separated string")
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        symbol = part.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    if not out:
        raise ConfigError("tickers is empty")
    return out


def _date_text(value, name: str) -> str:
    if value is None or str(value).strip() == "":
        raise ConfigError(f"{name} is required")
    if hasattr(value, "isoformat") and not isinstance(value, str):
        text = value.isoformat()
    else:
        text = str(value).strip()
    try:
        stamp = pd.Timestamp(text)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} is not a date: {value!r}") from exc
    if pd.isna(stamp):
        raise ConfigError(f"{name} is not a date: {value!r}")
    return stamp.strftime("%Y-%m-%d")


def _mapping(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return dict(value)


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"{name} must be true or false")


def _number(name: str, value, *, minimum: float, strict: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number < minimum or (strict and number <= minimum):
        op = ">" if strict else ">="
        raise ConfigError(f"{name} must be {op} {minimum}")
    return number


def _optional_number(value):
    if value is None or value == "":
        return None
    return float(value)
