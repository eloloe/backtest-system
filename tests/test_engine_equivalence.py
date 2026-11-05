"""Vectorized engine versus the bar-by-bar definition. No network."""

import math

import numpy as np
import pandas as pd
import pytest

from backtest_system.engine.backtester import Backtester
from backtest_system.engine.crosscheck import (
    OptionalEngineError,
    compare_equity,
    load_crosscheck,
    replay_equity,
)
from backtest_system.engine.portfolio import (
    run_equal_weight,
    simulate_target_weights,
    simulate_target_weights_iterative,
)
from backtest_system.strategies import BaseStrategy


def _ohlcv(opens, closes, start="2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(opens))
    open_s = pd.Series(opens, index=idx, dtype=float)
    close_s = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "Open": open_s,
            "High": np.maximum(open_s, close_s),
            "Low": np.minimum(open_s, close_s),
            "Close": close_s,
            "Volume": 1.0,
        },
        index=idx,
    )


def _signals(data: pd.DataFrame, values) -> pd.Series:
    return pd.Series(values, index=data.index, dtype=float)


def _assert_same_book(left, right):
    pd.testing.assert_series_equal(
        left[0]["portfolio_value"],
        right[0]["portfolio_value"],
        check_exact=False,
        rtol=1e-12,
        atol=1e-8,
    )
    pd.testing.assert_series_equal(left[0]["signal"], right[0]["signal"], check_names=False)
    pd.testing.assert_series_equal(left[0]["price"], right[0]["price"], check_names=False)
    if left[1].empty and right[1].empty:
        return
    assert list(left[1]["action"]) == list(right[1]["action"])
    assert list(left[1]["date"]) == list(right[1]["date"])
    for column in ("price", "shares", "value", "fee"):
        assert left[1][column].tolist() == pytest.approx(right[1][column].tolist(), rel=1e-12, abs=1e-8)


def test_golden_round_trip_with_us_costs():
    """Frozen equity. A joint edit of both engines still has to hit these numbers.

    Signal 1, 0, 0: buy the second open, sell the third open. The flat signal
    has to land on the middle bar; a 0 only on the last bar never fills.
    """
    data = _ohlcv([10.0, 10.0, 12.0], [10.0, 11.0, 12.0])
    portfolio, trades = Backtester(
        initial_capital=1_000, commission=0.001, slippage=0.0
    ).run(data, _signals(data, [1, 0, 0]), symbol="AAPL")
    # Buy spends 1000/1.001 at the second open (10). Sell the next open (12) net of 10 bps.
    notional = 1_000 / 1.001
    shares = notional / 10.0
    proceeds = shares * 12.0 * (1.0 - 0.001)
    assert portfolio["portfolio_value"].tolist() == pytest.approx(
        [1_000.0, shares * 11.0, proceeds], abs=1e-9
    )
    assert [round(value, 6) for value in portfolio["portfolio_value"]] == [
        1000.0,
        1098.901099,
        1197.602398,
    ]
    assert list(trades["action"]) == ["BUY", "SELL"]
    assert trades.iloc[0]["shares"] == pytest.approx(shares)


def test_vectorized_matches_reference_on_random_bars():
    rng = np.random.default_rng(7)
    schedules = [
        dict(symbol="AAPL", market=None, commission=0.001, slippage=0.0005),
        dict(symbol="2330.TW", market=None, commission=0.05, slippage=0.0005),
        dict(symbol="0050.TW", market=None, commission=0.0, slippage=0.0, brokerage_min=20),
        dict(
            symbol="ES=F",
            market="FUTURES",
            commission=0.0,
            slippage=0.0,
            contract_multiplier=50,
            min_lot=1,
        ),
        dict(symbol="MSFT", market="US", commission_bps=10, slippage_bps=5, commission_flat=1.0),
    ]
    for case, schedule in enumerate(schedules):
        for draw in range(12):
            n = int(rng.integers(2, 60))
            opens = rng.uniform(8.0, 180.0, n)
            closes = opens * rng.uniform(0.85, 1.15, n)
            if rng.random() < 0.7:
                opens[int(rng.integers(1, n))] = np.nan
            if rng.random() < 0.3:
                closes[int(rng.integers(0, n))] = np.nan
            signals = rng.integers(0, 2, n).astype(float)
            if rng.random() < 0.4:
                signals[int(rng.integers(0, n))] = np.nan
            data = _ohlcv(opens, closes, start=f"202{case % 4}-0{(draw % 8) + 1}-02")
            capital = float(rng.choice([20.0, 1_000.0, 50_000.0, 1_000_000.0]))
            kwargs = dict(schedule)
            symbol = kwargs.pop("symbol")
            market = kwargs.pop("market")
            bt = Backtester(initial_capital=capital, **kwargs)
            fast = bt.run(data, _signals(data, signals), symbol=symbol, market=market)
            slow = bt.run_reference(data, _signals(data, signals), symbol=symbol, market=market)
            _assert_same_book(fast, slow)
            assert np.isfinite(fast[0]["portfolio_value"].to_numpy()).all()


def test_replay_matches_engine_equity_including_a_multiplier():
    data = _ohlcv([80.0, 80.0, 80.0, 100.0], [80.0, 90.0, 100.0, 100.0])
    bt = Backtester(
        initial_capital=10_000,
        commission=0.0,
        slippage=0.0,
        contract_multiplier=50,
        min_lot=1,
    )
    portfolio, trades = bt.run(data, _signals(data, [1, 1, 0, 0]), market="FUTURES")
    replayed = replay_equity(
        trades,
        portfolio["price"],
        10_000,
        contract_multiplier=50,
    )
    report = load_crosscheck("reference").check(
        trades,
        portfolio["price"],
        10_000,
        portfolio["portfolio_value"],
        contract_multiplier=50,
    )
    assert report.ok
    assert report.compared_bars == len(portfolio)
    assert replayed.tolist() == pytest.approx(portfolio["portfolio_value"].tolist())


def test_compare_equity_flags_a_shifted_path():
    index = pd.bdate_range("2024-01-02", periods=4)
    left = pd.Series([100.0, 110.0, 90.0, 95.0], index=index)
    right = left * 1.01
    report = compare_equity(left, right, atol=1e-8, rtol=0.0)
    assert not report.ok
    assert report.max_abs_diff == pytest.approx(1.1)
    same = compare_equity(left, left.copy())
    assert same.ok
    assert same.compared_bars == 4


def test_nautilus_hook_is_optional_and_not_required_to_import():
    checker = load_crosscheck("nautilus")
    assert checker.name == "nautilus"
    assert "Nautilus" in checker.describe()
    with pytest.raises(OptionalEngineError, match="nautilus"):
        checker.check(pd.DataFrame(), pd.Series([1.0]), 1.0, pd.Series([1.0]))
    with pytest.raises(ValueError, match="unknown cross-check"):
        load_crosscheck("bloomberg")


def test_equal_weight_sleeves_match_separate_accounts():
    class _AlwaysLong(BaseStrategy):
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            return pd.Series(1, index=data.index, dtype=int)

    index = pd.bdate_range("2021-04-01", periods=6)
    prices = {
        "AAA": _ohlcv(np.linspace(10, 15, len(index)), np.linspace(10, 16, len(index)), start="2021-04-01"),
        "2330.TW": _ohlcv(np.linspace(500, 520, len(index)), np.linspace(500, 530, len(index)), start="2021-04-01"),
    }
    # Give both frames the same index the helper just built.
    for frame in prices.values():
        frame.index = index
    book = run_equal_weight(
        _AlwaysLong(),
        prices,
        ["AAA", "2330.TW", "MISSING"],
        index[0],
        index[-1] + pd.Timedelta(days=1),
        initial_capital=90_000,
        commission=0.001,
        slippage=0.0005,
    )
    curves = []
    for symbol in ("AAA", "2330.TW"):
        portfolio, _ = Backtester(initial_capital=30_000, commission=0.001, slippage=0.0005).run(
            prices[symbol], pd.Series(1, index=index), symbol=symbol
        )
        curves.append(portfolio["portfolio_value"])
    expected = pd.concat(curves, axis=1).sum(axis=1)
    assert book.equity.tolist() == pytest.approx(expected.tolist())
    # Three names were requested, so each sleeve is funded with 30k.
    # MISSING has no prices: that slice is not reassigned and is not in the sum.
    assert book.equity.iloc[0] == pytest.approx(60_000)


def test_weight_series_matches_the_bar_loop_with_a_hole():
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2022-01-03", periods=40)
    prices = {}
    for symbol, level in (("AAA", 20.0), ("BBB", 50.0), ("CCC.TW", 80.0)):
        noise = rng.normal(0, 0.01, len(index)).cumsum()
        close = level * np.exp(noise)
        opens = close * rng.uniform(0.98, 1.02, len(index))
        frame = _ohlcv(opens, close, start="2022-01-03")
        frame.index = index
        prices[symbol] = frame
    # BBB does not print on two sessions. The mark uses the previous close.
    prices["BBB"] = prices["BBB"].drop(index[5:7])
    decisions = [index[0], index[10], index[20], index[-1]]
    rows = []
    for _ in decisions:
        raw = rng.random(3)
        raw = raw / raw.sum() * float(rng.choice([0.0, 0.5, 1.0]))
        rows.append(raw)
    weights = pd.DataFrame(rows, index=decisions, columns=["AAA", "BBB", "CCC.TW"])
    common = dict(
        prices=prices,
        weights=weights,
        symbols=["AAA", "BBB", "CCC.TW"],
        start=index[0],
        end=index[-1] + pd.Timedelta(days=1),
        initial_capital=250_000,
        commission=0.001,
        slippage=0.0005,
    )
    fast = simulate_target_weights(**common)
    slow = simulate_target_weights_iterative(**common)
    assert fast.equity.tolist() == pytest.approx(slow.equity.tolist(), rel=1e-12, abs=1e-6)
    assert list(fast.trades["action"]) == list(slow.trades["action"])
    if not fast.trades.empty:
        assert fast.trades["shares"].tolist() == pytest.approx(slow.trades["shares"].tolist())
        assert fast.trades["fee"].tolist() == pytest.approx(slow.trades["fee"].tolist())
    replayed = replay_equity(fast.trades, _close_panel(prices, fast.equity.index), 250_000)
    report = compare_equity(fast.equity, replayed.reindex(fast.equity.index))
    assert report.ok, report


def _close_panel(prices, index) -> pd.DataFrame:
    columns = {}
    for symbol, frame in prices.items():
        columns[symbol] = frame["Close"].reindex(index)
    return pd.DataFrame(columns, index=index)


def test_missing_open_does_not_rest_an_order_past_the_next_signal():
    data = _ohlcv([10.0, math.nan, 12.0], [10.0, 11.0, 12.0])
    portfolio, trades = Backtester(initial_capital=1_000, commission=0.0, slippage=0.0).run(
        data, _signals(data, [1, 0, 0])
    )
    assert trades.empty
    assert portfolio["portfolio_value"].tolist() == pytest.approx([1_000, 1_000, 1_000])
