"""Walk-forward splits, embargo, and a locked holdout. No network."""

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from metrics import calculate_metrics
from strategies import BaseStrategy
from walk_forward import (
    DEFAULT_GAP_DAYS,
    HoldoutLockedError,
    WalkForwardError,
    build_folds,
    default_param_grid,
    evaluate_locked_holdout,
    pit_screen,
    run_walk_forward,
)


class FixedPosition(BaseStrategy):
    """Always long or always flat. The grid can prefer one on a rising train."""

    def __init__(self, position: int):
        self.position = int(position)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self.position, index=data.index, dtype=int)


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    opened = close.shift(1)
    opened.iloc[0] = close.iloc[0]
    return pd.DataFrame(
        {
            "Open": opened.astype(float),
            "High": np.maximum(opened, close),
            "Low": np.minimum(opened, close),
            "Close": close.astype(float),
            "Volume": 1.0,
        },
        index=close.index,
    )


def _daily_close(start, end, value) -> pd.Series:
    index = pd.date_range(start, end, freq="D", inclusive="left")
    if np.isscalar(value):
        return pd.Series(float(value), index=index)
    values = value(index)
    return pd.Series(values, index=index, dtype=float)


def _spy(monkeypatch):
    seen = []
    real = Backtester.run

    def wrapped(self, data, signals, symbol=None, market=None):
        seen.append((symbol, pd.DatetimeIndex(pd.to_datetime(data.index)).normalize()))
        return real(self, data, signals, symbol=symbol, market=market)

    monkeypatch.setattr(Backtester, "run", wrapped)
    return seen


def _days(start, end):
    return set(pd.date_range(start, end, freq="D", inclusive="left"))


def test_default_gap_is_twenty_one_calendar_days():
    assert DEFAULT_GAP_DAYS == 21


def test_gap_must_be_at_least_one_calendar_day():
    with pytest.raises(WalkForwardError, match="gap_days"):
        build_folds("2018-01-01", "2021-01-01", gap_days=0)


def test_folds_are_half_open_and_test_windows_do_not_overlap():
    split = build_folds(
        "2016-01-01",
        "2020-01-01",
        train_months=12,
        test_months=6,
        gap_days=21,
        holdout_months=6,
    )

    assert split.holdout_start == pd.Timestamp("2019-07-01")
    assert split.holdout_end == pd.Timestamp("2020-01-01")
    assert split.research_end == pd.Timestamp("2019-07-01")
    assert [(str(f.test_start.date()), str(f.test_end.date())) for f in split.folds] == [
        ("2017-01-22", "2017-07-22"),
        ("2017-07-22", "2018-01-22"),
        ("2018-01-22", "2018-07-22"),
        ("2018-07-22", "2019-01-22"),
    ]

    covered = set()
    holdout = _days(split.holdout_start, split.holdout_end)
    for fold, nxt in zip(split.folds, list(split.folds[1:]) + [None]):
        assert fold.train_start < fold.train_end < fold.test_start < fold.test_end
        assert fold.test_start - fold.train_end == pd.Timedelta(days=21)
        assert fold.as_of == fold.train_start
        train, gap, test = (
            _days(fold.train_start, fold.train_end),
            _days(fold.train_end, fold.test_start),
            _days(fold.test_start, fold.test_end),
        )
        assert train.isdisjoint(gap) and train.isdisjoint(test) and gap.isdisjoint(test)
        assert len(gap) == 21
        assert test.isdisjoint(covered)
        assert train.isdisjoint(holdout) and gap.isdisjoint(holdout) and test.isdisjoint(holdout)
        covered |= test
        if nxt is not None:
            assert fold.test_end == nxt.test_start

    # The tail after the last full test stays out of both the tests and the holdout.
    assert pd.Timestamp("2019-03-01") not in covered
    assert pd.Timestamp("2019-03-01") < split.holdout_start


def test_short_sample_does_not_invent_a_partial_fold():
    prices = {"AAA": _ohlcv(_daily_close("2020-01-01", "2020-06-01", 100.0))}
    with pytest.raises(WalkForwardError):
        run_walk_forward(
            prices,
            "買進持有 (Buy & Hold)",
            "2020-01-01",
            "2020-06-01",
            train_months=12,
            test_months=6,
            gap_days=21,
        )


def test_default_grids_are_small_and_buy_and_hold_is_not_searched():
    assert default_param_grid("買進持有 (Buy & Hold)") == ({},)
    moving_average = default_param_grid("均線交叉 (MA Cross)")
    assert 1 < len(moving_average) <= 6
    assert all(row["short_window"] < row["long_window"] for row in moving_average)
    assert {row["ma_type"] for row in moving_average} <= {"SMA", "EMA"}
    rsi = default_param_grid("RSI 策略")
    bands = default_param_grid("布林通道 (Bollinger Bands)")
    macd = default_param_grid("MACD 策略")
    assert 1 < len(rsi) <= 6
    assert 1 < len(bands) <= 6
    assert 1 < len(macd) <= 6
    assert all(row["fast"] < row["slow"] for row in macd)
    assert all(row["oversold"] < row["overbought"] for row in rsi)


def test_buy_and_hold_evaluates_tests_only_and_builds_charts(monkeypatch):
    seen = _spy(monkeypatch)
    prices = {"AAA": _ohlcv(_daily_close("2016-01-01", "2020-01-01", lambda idx: np.linspace(100, 160, len(idx))))}
    result = run_walk_forward(
        prices,
        "買進持有 (Buy & Hold)",
        "2016-01-01",
        "2020-01-01",
        train_months=12,
        test_months=6,
        gap_days=21,
        holdout_months=6,
        open_holdout=False,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
    )

    assert result.holdout is None
    assert all(not fold.searched and fold.params == {} for fold in result.folds)
    assert len(seen) == len(result.folds)
    for fold, (symbol, index) in zip(result.folds, seen):
        assert symbol == "AAA"
        assert index.min() >= fold.test_start
        assert index.max() < fold.test_end
        assert index.max() < result.holdout_start
    assert result.oos_equity.index.max() < result.holdout_start
    assert all(
        any(fold.test_start <= ts < fold.test_end for fold in result.folds)
        for ts in result.oos_equity.index
    )
    display, raw = calculate_metrics(
        result.oos_equity, result.oos_returns, risk_free_rate=0.0
    )
    assert raw["sharpe"] == pytest.approx(result.metrics["sharpe"])
    assert raw["total_return"] == pytest.approx(result.metrics["total_return"])
    assert display["夏普比率"] == result.metrics_display["夏普比率"]
    assert result.metrics["total_return"] > 0
    assert len(result.equity_figure.data) == 1
    assert len(result.drawdown_figure.data) == 1
    assert result.equity_figure.layout.title.text == "樣本外淨值"
    assert result.drawdown_figure.layout.title.text == "樣本外回撤"
    assert list(result.equity_figure.data[0].y) == pytest.approx(list(result.oos_equity))


def test_search_ignores_the_embargo_the_test_and_the_locked_holdout(monkeypatch):
    """A crash that starts on the embargo would flip the winner if it leaked.

    Train drifts from 100 to 160, so always-long beats flat on total return.
    From the train end onward the price is cut to the 30s and then lower.
    Flat wins on any slice that includes that crash. The embargo, the test,
    and the holdout all contain it.
    """
    seen = _spy(monkeypatch)
    index = pd.date_range("2020-01-01", "2020-12-01", freq="D", inclusive="left")
    close = pd.Series(index=index, dtype=float)
    train = index < "2020-07-01"
    after = ~train
    close.loc[train] = np.linspace(100, 160, int(train.sum()))
    close.loc[after] = np.linspace(40, 10, int(after.sum()))
    prices = {"AAA": _ohlcv(close)}

    def factory(params):
        return FixedPosition(params["position"])

    result = run_walk_forward(
        prices,
        "custom",
        "2020-01-01",
        "2020-12-01",
        train_months=6,
        test_months=2,
        gap_days=10,
        holdout_months=2,
        open_holdout=False,
        metric="total_return",
        param_grid=({"position": 0}, {"position": 1}),
        strategy_factory=factory,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
    )

    assert len(result.folds) == 1
    fold = result.folds[0]
    assert fold.train_end == pd.Timestamp("2020-07-01")
    assert fold.test_start == pd.Timestamp("2020-07-11")
    assert fold.test_end == pd.Timestamp("2020-09-11")
    assert result.holdout_start == pd.Timestamp("2020-10-01")
    assert fold.searched is True
    assert fold.params == {"position": 1}
    assert fold.train_score > 0

    assert len(seen) == 3
    for _symbol, index in seen[:2]:
        assert index.min() >= fold.train_start
        assert index.max() < fold.train_end
        assert index.max() < fold.test_start
    _symbol, test_index = seen[2]
    assert test_index.min() >= fold.test_start
    assert test_index.max() < fold.test_end
    for _symbol, index in seen:
        assert index.max() < result.holdout_start
        assert not ((index >= fold.train_end) & (index < fold.test_start)).any()

    with pytest.raises(HoldoutLockedError, match="open_holdout"):
        evaluate_locked_holdout(result, prices, open_holdout=False)

    seen.clear()
    holdout = evaluate_locked_holdout(result, prices, open_holdout=True)
    assert len(seen) == 1
    assert holdout.params == {"position": 1}
    assert holdout.metrics["total_return"] < 0
    assert seen[0][1].min() >= result.holdout_start
    assert seen[0][1].max() < result.holdout_end
    assert holdout.equity.index.min() >= result.holdout_start
    assert result.oos_equity.index.max() < holdout.equity.index.min()
    assert len(holdout.equity_figure.data) == 1
    assert len(holdout.drawdown_figure.data) == 1


def test_opening_the_holdout_does_not_change_the_out_of_sample_path():
    prices = {
        "AAA": _ohlcv(
            _daily_close("2016-01-01", "2020-01-01", lambda idx: np.linspace(80, 140, len(idx)))
        )
    }
    common = dict(
        train_months=12,
        test_months=6,
        gap_days=21,
        holdout_months=6,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
    )
    locked = run_walk_forward(
        prices, "買進持有 (Buy & Hold)", "2016-01-01", "2020-01-01", open_holdout=False, **common
    )
    opened = run_walk_forward(
        prices, "買進持有 (Buy & Hold)", "2016-01-01", "2020-01-01", open_holdout=True, **common
    )
    assert locked.holdout is None
    assert opened.holdout is not None
    assert opened.holdout.params == {}
    assert list(locked.oos_equity) == pytest.approx(list(opened.oos_equity))
    assert [fold.params for fold in locked.folds] == [fold.params for fold in opened.folds]
    assert opened.oos_equity.index.max() < opened.holdout.start
    assert opened.holdout.equity_figure.layout.title.text == "鎖定 holdout 淨值"


def test_each_fold_search_stays_inside_that_fold(monkeypatch):
    seen = _spy(monkeypatch)
    prices = {"AAA": _ohlcv(_daily_close("2016-01-01", "2020-01-01", 100.0))}

    def factory(params):
        return FixedPosition(params["position"])

    result = run_walk_forward(
        prices,
        "custom",
        "2016-01-01",
        "2020-01-01",
        train_months=12,
        test_months=6,
        gap_days=21,
        holdout_months=6,
        metric="total_return",
        param_grid=({"position": 0}, {"position": 1}),
        strategy_factory=factory,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
    )

    assert len(seen) == len(result.folds) * 3
    for i, fold in enumerate(result.folds):
        train_a, train_b, test = (item[1] for item in seen[3 * i : 3 * i + 3])
        gap = _days(fold.train_end, fold.test_start)
        for index in (train_a, train_b):
            assert index.min() >= fold.train_start
            assert index.max() < fold.train_end
            assert set(index).isdisjoint(gap)
            assert index.max() < result.holdout_start
        assert test.min() >= fold.test_start
        assert test.max() < fold.test_end
        assert set(test).isdisjoint(gap)
        assert test.max() < result.holdout_start


def test_warmup_can_precede_train_but_still_stops_before_the_embargo(monkeypatch):
    seen = _spy(monkeypatch)
    prices = {
        "AAA": _ohlcv(
            _daily_close("2017-01-01", "2020-01-01", lambda idx: np.linspace(100, 150, len(idx)))
        )
    }

    def factory(params):
        return FixedPosition(params["position"])

    result = run_walk_forward(
        prices,
        "custom",
        "2018-01-01",
        "2020-01-01",
        train_months=12,
        test_months=3,
        gap_days=15,
        holdout_months=3,
        metric="total_return",
        param_grid=({"position": 0}, {"position": 1}),
        strategy_factory=factory,
        warmup_bars=30,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
    )
    fold = result.folds[0]
    train_slices = [index for _symbol, index in seen if index.max() < fold.train_end]
    assert train_slices
    for index in train_slices:
        assert index.min() < fold.train_start
        assert index.max() < fold.train_end
        assert set(index).isdisjoint(_days(fold.train_end, fold.test_start))
        assert index.max() < result.holdout_start


def test_screen_uses_pit_at_train_start_and_not_the_live_snapshot(monkeypatch):
    calls = []

    def fake_pit(symbols, as_of, lag_days=None):
        calls.append(pd.Timestamp(as_of).normalize())
        rows = []
        for symbol in symbols:
            rows.append(
                {
                    "symbol": symbol,
                    "PE": 10.0 if symbol == "AAA" else 90.0,
                    "PB": 1.0,
                    "ROE": 0.25,
                    "RevenueGrowth": 0.2,
                    "EarningsGrowth": 0.1,
                    "MarketCap": 40_000_000_000.0,
                }
            )
        return pd.DataFrame(rows)

    def live(*_args, **_kwargs):
        raise AssertionError("live snapshot was used")

    monkeypatch.setattr("walk_forward.fetch_fundamentals_pit", fake_pit)
    monkeypatch.setattr("fundamentals.fetch_fundamentals", live)
    seen = _spy(monkeypatch)

    index = pd.date_range("2020-01-01", "2020-12-01", freq="D", inclusive="left")
    close = pd.Series(np.linspace(100, 130, len(index)), index=index)
    prices = {"AAA": _ohlcv(close), "BBB": _ohlcv(close)}
    screen = pit_screen(
        pe_max=50,
        roe_min=0.1,
        rev_growth_min=0.0,
        mcap_min=1_000_000_000,
        use_thresholds=True,
    )
    result = run_walk_forward(
        prices,
        "買進持有 (Buy & Hold)",
        "2020-01-01",
        "2020-12-01",
        train_months=6,
        test_months=2,
        gap_days=10,
        holdout_months=2,
        open_holdout=False,
        screen=screen,
        commission=0.0,
        slippage=0.0,
        warmup_bars=0,
    )

    assert calls == [result.folds[0].train_start]
    assert result.folds[0].as_of == result.folds[0].train_start
    assert result.folds[0].symbols == ["AAA"]
    assert [symbol for symbol, _index in seen] == ["AAA"]
    assert result.holdout_start not in calls


def test_opened_holdout_rescreens_at_its_own_start(monkeypatch):
    calls = []

    def fake_pit(symbols, as_of, lag_days=None):
        calls.append(pd.Timestamp(as_of).normalize())
        return pd.DataFrame(
            {
                "symbol": list(symbols),
                "PE": 12.0,
                "PB": 1.0,
                "ROE": 0.2,
                "RevenueGrowth": 0.1,
                "EarningsGrowth": 0.1,
                "MarketCap": 20_000_000_000.0,
            }
        )

    scored = []

    def fake_score(df_fundamentals, as_of, active_factors, top_n, method):
        scored.append(pd.Timestamp(as_of).normalize())
        out = df_fundamentals.head(1).copy()
        out["Composite_Score"] = 1.0
        return out

    monkeypatch.setattr("walk_forward.fetch_fundamentals_pit", fake_pit)
    monkeypatch.setattr("walk_forward.apply_factor_score", fake_score)
    monkeypatch.setattr(
        "fundamentals.fetch_fundamentals",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("live snapshot")),
    )

    index = pd.date_range("2020-01-01", "2020-12-01", freq="D", inclusive="left")
    close = pd.Series(np.linspace(50, 80, len(index)), index=index)
    prices = {"AAA": _ohlcv(close), "BBB": _ohlcv(close)}
    screen = pit_screen(
        use_thresholds=False,
        active_factors={"roe": 1.0},
        top_n=1,
        method="zscore",
    )
    result = run_walk_forward(
        prices,
        "買進持有 (Buy & Hold)",
        "2020-01-01",
        "2020-12-01",
        train_months=6,
        test_months=2,
        gap_days=10,
        holdout_months=2,
        open_holdout=True,
        screen=screen,
        commission=0.0,
        slippage=0.0,
        warmup_bars=0,
    )

    assert calls == [result.folds[0].train_start, result.holdout_start]
    assert scored == calls
    assert result.folds[0].symbols == ["AAA"]
    assert result.holdout.symbols == ["AAA"]
    assert result.holdout.as_of == result.holdout_start


def test_equal_weight_copies_of_one_path_match_a_single_name():
    close = _daily_close("2018-01-01", "2021-01-01", lambda idx: np.linspace(90, 130, len(idx)))
    frame = _ohlcv(close)
    common = dict(
        train_months=12,
        test_months=6,
        gap_days=21,
        holdout_months=0,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
    )
    one = run_walk_forward({"AAA": frame}, "買進持有 (Buy & Hold)", "2018-01-01", "2021-01-01", **common)
    two = run_walk_forward(
        {"AAA": frame, "BBB": frame.copy()},
        "買進持有 (Buy & Hold)",
        "2018-01-01",
        "2021-01-01",
        **common,
    )
    assert two.metrics["total_return"] == pytest.approx(one.metrics["total_return"])
    assert two.metrics["sharpe"] == pytest.approx(one.metrics["sharpe"])
    assert list(two.folds[0].symbols) == ["AAA", "BBB"]


def test_empty_screen_stays_in_cash_and_still_draws(monkeypatch):
    seen = _spy(monkeypatch)
    prices = {"AAA": _ohlcv(_daily_close("2020-01-01", "2020-12-01", 100.0))}
    result = run_walk_forward(
        prices,
        "均線交叉 (MA Cross)",
        "2020-01-01",
        "2020-12-01",
        train_months=6,
        test_months=2,
        gap_days=10,
        holdout_months=2,
        screen=lambda _as_of, _symbols: [],
        commission=0.0,
        slippage=0.0,
        warmup_bars=0,
    )
    assert seen == []
    assert result.folds[0].symbols == []
    assert result.folds[0].params is None
    assert result.metrics["total_return"] == pytest.approx(0.0)
    assert len(result.equity_figure.data) == 1
    assert len(result.drawdown_figure.data) == 1
    with pytest.raises(WalkForwardError, match="holdout"):
        evaluate_locked_holdout(result, prices, open_holdout=True)
