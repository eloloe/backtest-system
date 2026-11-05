"""New strategies and the CLI. Synthetic prices only — no network, no broker keys."""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from books import simulate_target_weights
from runner import ConfigError, RunConfig, load_run_config, run_research, write_report
from strategies import (
    CoreSatellite,
    TrendStop,
    build_strategy,
    canonical_strategy,
    rank_cross_section,
)
from walk_forward import default_param_grid, run_walk_forward

ROOT = Path(__file__).resolve().parents[1]


def _ohlcv(close, opens=None, high=None, low=None) -> pd.DataFrame:
    close = pd.Series(close, dtype=float)
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.bdate_range("2020-01-02", periods=len(close))
    open_ = close.copy() if opens is None else pd.Series(opens, index=close.index, dtype=float)
    high_ = np.maximum(open_, close) if high is None else pd.Series(high, index=close.index, dtype=float)
    low_ = np.minimum(open_, close) if low is None else pd.Series(low, index=close.index, dtype=float)
    return pd.DataFrame(
        {"Open": open_, "High": high_, "Low": low_, "Close": close, "Volume": 1.0},
        index=close.index,
    )


def test_trend_stop_signal_shape_and_warmup_is_flat():
    data = _ohlcv(np.linspace(10, 12, 30))
    signals = TrendStop(trend_window=50, stop_mode="percent", stop_pct=0.08).generate_signals(data)
    assert list(signals.index) == list(data.index)
    assert signals.dtype == int
    assert set(signals.unique()).issubset({0, 1})
    assert int(signals.sum()) == 0


def test_percent_stop_exits_and_does_not_reenter_while_the_trend_is_still_on():
    # Slow average stays under the pullback. 17.2 is through a 10% stop from
    # the high at 20, and it is still above the average, so the book does not
    # re-arm. The next three closes stay flat.
    close = list(np.linspace(10, 20, 25)) + [17.2, 17.4, 17.6, 17.8]
    data = _ohlcv(close)
    signals = TrendStop(trend_window=20, stop_mode="percent", stop_pct=0.10).generate_signals(data)

    assert list(signals.index) == list(data.index)
    assert signals.dtype == int
    assert set(signals.unique()) <= {0, 1}
    assert signals.iloc[24] == 1
    assert signals.iloc[25] == 0
    assert signals.iloc[26:].tolist() == [0, 0, 0]


def test_percent_stop_can_reenter_only_after_the_trend_resets():
    close = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 17, 12, 11, 10, 14, 18, 22]
    data = _ohlcv(close)
    signals = TrendStop(trend_window=8, stop_mode="percent", stop_pct=0.10).generate_signals(data)
    # The dip through the average re-arms. A later close back above it is a new trade.
    assert signals.iloc[11] == 0
    assert signals.iloc[-1] == 1


def test_atr_stop_cuts_a_loser_the_percent_stop_still_holds():
    # ATR(1) on the last bar is the gap from 62 to 56. Half an ATR sits above
    # that close. A 20% stop from the same high does not. The average is ~53.
    close = np.array([50.0] * 20 + [55, 58, 60, 62, 56], dtype=float)
    high = close + 0.2
    low = close - 0.2
    high[-1] = 56.2
    low[-1] = 55.8
    data = _ohlcv(close, high=high, low=low)
    atr = TrendStop(
        trend_window=15, stop_mode="atr", atr_period=1, atr_mult=0.5, stop_pct=0.20
    ).generate_signals(data)
    percent = TrendStop(
        trend_window=15, stop_mode="percent", stop_pct=0.20
    ).generate_signals(data)
    assert atr.iloc[-2] == 1
    assert atr.iloc[-1] == 0
    assert percent.iloc[-1] == 1


def test_trend_stop_exit_fills_on_the_next_open():
    close = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 17, 17]
    opens = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 19, 17]
    data = _ohlcv(close, opens=opens)
    signals = TrendStop(trend_window=8, stop_mode="percent", stop_pct=0.10).generate_signals(data)
    _, trades = Backtester(initial_capital=10_000, commission=0.0, slippage=0.0).run(
        data, signals, symbol="SPY"
    )
    sells = trades[trades["action"] == "SELL"]
    assert len(sells) == 1
    # Signal goes flat on the 17 close (index 11). The sell is the next open.
    assert sells.iloc[0]["date"] == data.index[12]
    assert sells.iloc[0]["price"] == pytest.approx(17.0)


def test_rank_uses_only_prices_on_or_before_as_of_and_the_given_fundamentals():
    index = pd.bdate_range("2020-01-02", periods=40)
    as_of = index[25]
    base = pd.Series(np.linspace(100, 110, len(index)), index=index)
    cheap = _ohlcv(base)
    expensive = _ohlcv(base.copy())
    # Identical history through as_of. Anything after must not move the rank.
    expensive.loc[expensive.index > as_of, "Close"] = 500.0
    fundamentals = pd.DataFrame({"symbol": ["CHEAP", "RICH"], "PE": [8.0, 40.0]})
    kwargs = dict(
        top_n=1,
        weighting="equal",
        momentum_lookback=10,
        momentum_skip=0,
        momentum_weight=0.5,
        value_weight=0.5,
    )
    first = rank_cross_section(
        {"CHEAP": cheap, "RICH": expensive}, fundamentals, as_of, **kwargs
    )
    flipped = expensive.copy()
    flipped.loc[flipped.index > as_of, "Close"] = 1.0
    second = rank_cross_section(
        {"CHEAP": cheap, "RICH": flipped}, fundamentals, as_of, **kwargs
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["symbol"] == "CHEAP"
    assert first["weight"].sum() == pytest.approx(1.0)
    assert first.iloc[0]["as_of"] == as_of

    # A worse PE, still the only frame the rank is allowed to see, changes the winner.
    revised = pd.DataFrame({"symbol": ["CHEAP", "RICH"], "PE": [80.0, 5.0]})
    reranked = rank_cross_section(
        {"CHEAP": cheap, "RICH": expensive}, revised, as_of, **kwargs
    )
    assert reranked.iloc[0]["symbol"] == "RICH"


def test_missing_pe_is_left_out_instead_of_filled_with_the_median():
    index = pd.bdate_range("2020-01-02", periods=30)
    prices = {
        "A": _ohlcv(np.linspace(10, 12, len(index)), ),
        "B": _ohlcv(np.linspace(10, 11, len(index))),
    }
    prices["A"].index = index
    prices["B"].index = index
    fundamentals = pd.DataFrame({"symbol": ["A", "B"], "PE": [12.0, np.nan]})
    ranked = rank_cross_section(
        prices,
        fundamentals,
        index[-1],
        top_n=2,
        momentum_lookback=5,
        momentum_skip=0,
        momentum_weight=0.5,
        value_weight=0.5,
    )
    assert ranked["symbol"].tolist() == ["A"]
    assert ranked.iloc[0]["weight"] == pytest.approx(1.0)


def test_risk_budget_is_inverse_volatility_and_sums_to_one():
    index = pd.bdate_range("2020-01-02", periods=40)
    calm = np.array([100.0, 100.5] * 20)
    wild = np.array([100.0, 120.0, 80.0, 120.0] * 10)
    prices = {
        "CALM": _ohlcv(pd.Series(calm, index=index)),
        "WILD": _ohlcv(pd.Series(wild, index=index)),
    }
    fundamentals = pd.DataFrame({"symbol": ["CALM", "WILD"], "PE": [15.0, 15.0]})
    ranked = rank_cross_section(
        prices,
        fundamentals,
        index[-1],
        top_n=2,
        weighting="risk",
        momentum_lookback=5,
        momentum_skip=0,
        vol_lookback=20,
        momentum_weight=0.0,
        value_weight=1.0,
    )
    table = ranked.set_index("symbol")
    assert set(table.index) == {"CALM", "WILD"}
    assert table["weight"].sum() == pytest.approx(1.0)
    assert table.loc["CALM", "weight"] > table.loc["WILD", "weight"]
    assert table.loc["CALM", "weight"] / table.loc["WILD", "weight"] == pytest.approx(
        table.loc["WILD", "volatility"] / table.loc["CALM", "volatility"]
    )
    assert table["weighting"].unique().tolist() == ["risk"]


def test_cross_section_buys_the_next_open_and_matches_all_in_costs():
    index = pd.bdate_range("2020-01-02", "2020-02-06")
    rising = np.linspace(100, 130, len(index))
    flat = np.full(len(index), 50.0)
    prices = {"WIN": _ohlcv(pd.Series(rising, index=index)), "FLAT": _ohlcv(pd.Series(flat, index=index))}
    seen = []

    def loader(symbols, as_of):
        seen.append(pd.Timestamp(as_of).normalize())
        return pd.DataFrame({"symbol": list(symbols), "PE": [10.0, 10.0]})

    strategy = build_strategy(
        "橫截面多因子 (Cross-Section)",
        {
            "top_n": 1,
            "weighting": "equal",
            "rebalance_months": 1,
            "momentum_lookback": 5,
            "momentum_skip": 0,
            "momentum_weight": 1.0,
            "value_weight": 0.0,
        },
        fundamentals_loader=loader,
    )
    book = strategy.run_book(
        prices,
        ["WIN", "FLAT"],
        index[0],
        index[-1] + pd.Timedelta(days=1),
        initial_capital=100_000,
        commission=0.0,
        slippage=0.0,
    )
    assert seen  # the loader is what the book asked; nothing else is fetched
    assert all(stamp <= pd.Timestamp("2020-01-31") for stamp in seen)
    buys = book.trades[book.trades["action"] == "BUY"]
    assert len(buys) == 1
    assert buys.iloc[0]["symbol"] == "WIN"
    assert buys.iloc[0]["date"] == pd.Timestamp("2020-02-03")
    assert buys.iloc[0]["price"] == pytest.approx(float(prices["WIN"].loc[pd.Timestamp("2020-02-03"), "Open"]))
    assert book.equity.loc[pd.Timestamp("2020-01-31")] == pytest.approx(100_000)
    assert not book.holdings.empty
    assert book.holdings.iloc[0]["as_of"] == pd.Timestamp("2020-01-31")

    # A 100% weight from cash uses the same notional, fee, and shares as Backtester.
    cost_index = pd.bdate_range("2021-03-01", periods=3)
    frame = _ohlcv(
        pd.Series([10.0, 110.0, 120.0], index=cost_index),
        opens=[10.0, 100.0, 100.0],
    )
    weights = pd.DataFrame({"AAA": [1.0]}, index=[cost_index[0]])
    weighted = simulate_target_weights(
        {"AAA": frame},
        weights,
        ["AAA"],
        cost_index[0],
        cost_index[-1] + pd.Timedelta(days=1),
        initial_capital=100_000,
        commission=0.001,
        slippage=0.0005,
    )
    _, engine = Backtester(initial_capital=100_000, commission=0.001, slippage=0.0005).run(
        frame, pd.Series([1, 1, 1], index=cost_index), symbol="AAA"
    )
    assert weighted.trades.iloc[0]["date"] == engine.iloc[0]["date"]
    assert weighted.trades.iloc[0]["shares"] == pytest.approx(engine.iloc[0]["shares"])
    assert weighted.trades.iloc[0]["fee"] == pytest.approx(engine.iloc[0]["fee"])
    assert weighted.trades.iloc[0]["price"] == pytest.approx(engine.iloc[0]["price"])


def test_core_satellite_is_a_one_time_split_with_no_optimizer():
    with pytest.raises(ValueError, match="sum to 1"):
        CoreSatellite(core_weight=0.8, satellite_weight=0.3)

    index = pd.bdate_range("2020-03-02", periods=8)
    close = pd.Series([10, 10, 11, 12, 13, 14, 15, 16], index=index, dtype=float)
    prices = {"AAA": _ohlcv(close)}
    book = build_strategy(
        "core_satellite",
        {
            "core_weight": 0.7,
            "satellite_weight": 0.3,
            "core": "buy_and_hold",
            "satellite": "rsi",
            "satellite_params": {"period": 3, "oversold": 0, "overbought": 70},
        },
    ).run_book(
        prices,
        ["AAA"],
        index[0],
        index[-1] + pd.Timedelta(days=1),
        initial_capital=100_000,
        commission=0.0,
        slippage=0.0,
    )
    core, _ = Backtester(initial_capital=70_000, commission=0.0, slippage=0.0).run(
        prices["AAA"], pd.Series(1, index=index), symbol="AAA"
    )
    expected = core["portfolio_value"] + 30_000
    assert book.equity.tolist() == pytest.approx(expected.tolist())
    assert set(book.trades["action"]) == {"BUY"}
    assert any("沒有優化器" in note for note in book.notes)


def test_new_strategy_grids_stay_small():
    for name in ("trend_stop", "cross_section", "core_satellite"):
        grid = default_param_grid(name)
        assert 1 < len(grid) <= 6
    weights = {(row["core_weight"], row["satellite_weight"]) for row in default_param_grid("核心衛星 (Core-Satellite)")}
    assert weights == {(0.70, 0.30)}


def test_trend_stop_walk_forward_uses_the_existing_engine():
    index = pd.date_range("2019-01-01", "2019-09-01", freq="B")
    close = pd.Series(np.linspace(100, 140, len(index)), index=index)
    prices = {"AAA": _ohlcv(close)}
    result = run_walk_forward(
        prices,
        "趨勢停損 (Trend + Stop)",
        "2019-01-01",
        "2019-09-01",
        train_months=2,
        test_months=1,
        gap_days=1,
        holdout_months=0,
        warmup_bars=0,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        param_grid=({"trend_window": 5, "stop_mode": "percent", "stop_pct": 0.2},),
    )
    assert result.folds
    assert result.folds[0].params["stop_mode"] == "percent"
    assert result.oos_equity.iloc[-1] > 0
    assert len(result.equity_figure.data) == 1


def test_cross_section_walk_forward_stays_inside_the_window_and_off_the_live_snapshot(monkeypatch):
    calls = []

    def loader(symbols, as_of):
        calls.append(pd.Timestamp(as_of).normalize())
        return pd.DataFrame({"symbol": list(symbols), "PE": 12.0})

    def boom(*args, **kwargs):
        raise AssertionError("live snapshot must not be read")

    monkeypatch.setattr("fundamentals.fetch_fundamentals", boom)
    monkeypatch.setattr("fundamentals_pit.fetch_fundamentals_pit", boom)

    def factory(params):
        return build_strategy("cross_section", params, fundamentals_loader=loader)

    index = pd.date_range("2020-01-01", "2020-08-01", freq="B")
    prices = {
        "AAA": _ohlcv(pd.Series(np.linspace(50, 80, len(index)), index=index)),
        "BBB": _ohlcv(pd.Series(np.linspace(40, 45, len(index)), index=index)),
    }
    grid = (
        {
            "top_n": 1,
            "weighting": "equal",
            "rebalance_months": 1,
            "momentum_lookback": 10,
            "momentum_skip": 0,
            "momentum_weight": 1.0,
            "value_weight": 0.0,
        },
    )
    result = run_walk_forward(
        prices,
        "cross_section",
        "2020-01-01",
        "2020-08-01",
        train_months=2,
        test_months=1,
        gap_days=1,
        holdout_months=0,
        warmup_bars=15,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        strategy_factory=factory,
        param_grid=grid,
    )
    assert calls
    for fold in result.folds:
        inside = [stamp for stamp in calls if fold.test_start <= stamp < fold.test_end]
        assert inside
        assert all(stamp < fold.test_end for stamp in inside)


def test_yaml_parses_and_cli_flags_override_it(tmp_path):
    loaded = load_run_config(
        ROOT / "examples" / "trend_stop.yaml",
        {"start": "2019-06-01", "tickers": "spy, qqq"},
    )
    assert loaded.start == "2019-06-01"
    assert loaded.end == "2024-01-01"
    assert loaded.tickers == ["SPY", "QQQ"]
    assert loaded.params["stop_mode"] == "atr"
    assert loaded.provider == "yahoo"
    assert loaded.walk_forward is False
    assert canonical_strategy(loaded.strategy) == "trend_stop"

    secret = tmp_path / "secret.yaml"
    secret.write_text(
        "strategy: buy_and_hold\ntickers: [SPY]\nstart: 2020-01-01\nend: 2021-01-01\n"
        "tiingo_api_key: nope\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="secret"):
        load_run_config(secret)


def test_run_research_writes_equity_and_drawdown_charts(tmp_path):
    index = pd.bdate_range("2020-01-02", periods=12)
    close = pd.Series(np.linspace(10, 16, len(index)), index=index)
    config = RunConfig(
        strategy="buy_and_hold",
        tickers=["AAA"],
        start=index[0].date().isoformat(),
        end=(index[-1] + pd.Timedelta(days=1)).date().isoformat(),
        benchmark=None,
        initial_capital=50_000,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
        output_dir=str(tmp_path),
    )
    result = run_research(config, prices={"AAA": _ohlcv(close)})
    written = write_report(result, tmp_path)
    assert result.metrics["total_return"] > 0
    assert result.equity_figure.layout.title.text == "淨值"
    assert result.drawdown_figure.layout.title.text == "回撤"
    equity_html = written["equity_html"].read_text(encoding="utf-8")
    drawdown_html = written["drawdown_html"].read_text(encoding="utf-8")
    # Plotly writes the layout title as a JSON escape inside the chart file.
    assert "\\u6de8\\u503c" in equity_html
    assert "\\u56de\\u64a4" in drawdown_html
    assert "總報酬率" in written["metrics_json"].read_text(encoding="utf-8")


def test_core_modules_do_not_import_streamlit():
    modules = [
        "strategies.py",
        "factor_engine.py",
        "books.py",
        "runner.py",
        "cli.py",
        "walk_forward.py",
        "backtester.py",
        "screener.py",
        "fundamentals_pit.py",
        "metrics.py",
    ]
    for name in modules:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module.split(".")[0]] if node.module else []
            else:
                continue
            assert "streamlit" not in imported, name

    code = (
        "import books, cli, factor_engine, runner, strategies, walk_forward, sys; "
        "raise SystemExit(0 if 'streamlit' not in sys.modules else 1)"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=False)
    assert completed.returncode == 0
