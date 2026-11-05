"""Benchmarks, noise, and the shared report. Synthetic prices, no network."""

import json

import numpy as np
import pandas as pd
import pytest

from baselines import infer_benchmark, noise_baseline_single
from runner import RunConfig, run_research, write_report
from strategies import BaseStrategy, CrossSection
from walk_forward import run_walk_forward


class Timed(BaseStrategy):
    """Long only after an up close. The label series is not constant."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        up = data["Close"].pct_change().fillna(0.0) > 0
        return up.astype(int)


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


def test_infer_benchmark_follows_the_market():
    assert infer_benchmark(["AAPL", "MSFT"]) == "SPY"
    assert infer_benchmark(["2330.TW"]) == "0050.TW"
    assert infer_benchmark(["SPY"], market="TW") == "0050.TW"
    assert infer_benchmark(["0050.TWO"]) == "0050.TW"


def test_single_run_without_a_search_reports_n_equals_one_and_writes_the_pack(tmp_path):
    index = pd.bdate_range("2019-01-02", periods=80)
    close = pd.Series(np.linspace(20, 30, len(index)), index=index)
    bench = pd.Series(np.linspace(100, 110, len(index)), index=index)
    config = RunConfig(
        strategy="buy_and_hold",
        tickers=["AAA"],
        start=index[0].date().isoformat(),
        end=(index[-1] + pd.Timedelta(days=1)).date().isoformat(),
        benchmark="SPY",
        initial_capital=50_000,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
        output_dir=str(tmp_path),
    )
    result = run_research(
        config,
        prices={"AAA": _ohlcv(close), "SPY": _ohlcv(bench)},
    )
    assert result.selection["multiple_testing"] is False
    assert result.selection["n_trials"] == 1
    assert result.selection["dsr"] is None
    assert result.metrics_display["多重檢定"] == "不適用（N=1）"
    assert "PSR (SR>0)" in result.metrics_display
    assert result.comparison["benchmark"]["symbol"] == "SPY"
    assert result.comparison["benchmark"]["sharpe"] > 0
    assert result.comparison["noise"]["degenerate"] is True
    assert "打亂標籤 Sharpe" in result.metrics_display
    names = [trace.name for trace in result.equity_figure.data]
    assert "SPY 買進持有" in names
    assert result.equity_figure.layout.title.text == "淨值"
    assert result.rolling_sharpe_figure.layout.title.text.startswith("滾動夏普")

    written = write_report(result, tmp_path)
    payload = json.loads(written["metrics_json"].read_text(encoding="utf-8"))
    assert payload["selection"]["multiple_testing"] is False
    assert "總報酬率" in payload["metrics"]
    assert "rolling_sharpe.html" in written["rolling_sharpe_html"].name
    assert "SPY" in written["baselines_csv"].read_text(encoding="utf-8")
    assert "trials_csv" not in written


def test_a_book_strategy_is_compared_with_random_entry():
    index = pd.bdate_range("2020-01-02", periods=40)
    close = pd.Series(np.linspace(10, 14, len(index)), index=index)
    frame = _ohlcv(close)
    strategy = CrossSection(
        top_n=1,
        momentum_lookback=5,
        momentum_skip=0,
        vol_lookback=5,
        rebalance_months=1,
    )
    noise = noise_baseline_single(
        strategy,
        {"AAA": frame, "BBB": frame.copy()},
        ["AAA", "BBB"],
        index[0],
        index[-1] + pd.Timedelta(days=1),
        initial_capital=10_000,
        commission=0.0,
        slippage=0.0,
        market=None,
        backtester_kwargs={},
        risk_free_rate=0.0,
        strategy_sharpe=0.0,
        n_perm=4,
        seed=1,
    )
    assert noise.kind == "random_entry"
    assert noise.degenerate is False
    assert noise.sharpe_median is not None
    assert noise.equity is not None


def test_shuffled_labels_move_and_stay_on_the_seed():
    index = pd.bdate_range("2018-01-02", periods=120)
    wobble = np.sin(np.linspace(0, 18, len(index)))
    close = pd.Series(50 + np.linspace(0, 8, len(index)) + wobble, index=index)
    config = RunConfig(
        strategy="ma_cross",
        tickers=["AAA"],
        start=index[0].date().isoformat(),
        end=(index[-1] + pd.Timedelta(days=1)).date().isoformat(),
        benchmark=None,
        initial_capital=10_000,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
        params={"short_window": 5, "long_window": 20, "ma_type": "SMA"},
    )
    prices = {"AAA": _ohlcv(close)}
    first = run_research(config, prices=prices)
    second = run_research(config, prices=prices)
    noise = first.comparison["noise"]
    assert noise["kind"] == "shuffled_label"
    assert noise["degenerate"] is False
    assert noise["seed"] == 0
    assert noise["sharpe_median"] == pytest.approx(second.comparison["noise"]["sharpe_median"])
    assert first.equity_figure.layout.title.text == "淨值"


def test_walk_forward_grid_counts_each_cell_on_each_fold(tmp_path):
    index = pd.bdate_range("2018-01-02", periods=420)
    close = pd.Series(100 + np.linspace(0, 30, len(index)) + np.sin(np.linspace(0, 40, len(index))), index=index)
    bench = pd.Series(np.linspace(50, 70, len(index)), index=index)
    prices = {"AAA": _ohlcv(close), "SPY": _ohlcv(bench)}

    def factory(params):
        return Timed() if params["side"] == 1 else BaseStrategyHold()

    searched = run_walk_forward(
        prices,
        "custom",
        index[0].date().isoformat(),
        (index[-1] + pd.Timedelta(days=1)).date().isoformat(),
        train_months=6,
        test_months=3,
        gap_days=5,
        holdout_months=3,
        open_holdout=True,
        metric="sharpe",
        param_grid=({"side": 0}, {"side": 1}),
        strategy_factory=factory,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
    )
    folds_searched = sum(1 for fold in searched.folds if fold.searched)
    assert folds_searched >= 1
    assert len(searched.trials) == folds_searched * 2
    assert searched.selection["multiple_testing"] is True
    assert searched.selection["n_trials"] == len(searched.trials)
    assert searched.selection["dsr"] is not None
    assert 0.0 <= searched.selection["dsr"] <= 1.0
    assert "Deflated Sharpe (機率)" in searched.metrics_display
    assert searched.holdout.selection["n_trials"] == len(searched.trials)
    assert searched.holdout.selection["multiple_testing"] is True
    # The untouched single-trace figure on the walk-forward result stays the test-window book.
    assert len(searched.equity_figure.data) == 1

    quiet = run_walk_forward(
        {"AAA": prices["AAA"]},
        "買進持有 (Buy & Hold)",
        index[0].date().isoformat(),
        (index[-1] + pd.Timedelta(days=1)).date().isoformat(),
        train_months=6,
        test_months=3,
        gap_days=5,
        holdout_months=0,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
    )
    assert quiet.trials == []
    assert quiet.selection["multiple_testing"] is False
    assert quiet.metrics_display["多重檢定"] == "不適用（N=1）"

    config = RunConfig(
        strategy="ma_cross",
        tickers=["AAA"],
        start=index[0].date().isoformat(),
        end=(index[-1] + pd.Timedelta(days=1)).date().isoformat(),
        benchmark="SPY",
        initial_capital=20_000,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
        walk_forward=True,
        train_months=6,
        test_months=3,
        gap_days=5,
        holdout_months=0,
        output_dir=str(tmp_path),
    )
    research = run_research(config, prices=prices)
    assert research.selection["multiple_testing"] is True
    assert research.comparison["benchmark"]["symbol"] == "SPY"
    assert research.comparison["noise"]["kind"] == "shuffled_label"
    assert any(trace.name == "SPY 買進持有" for trace in research.equity_figure.data)
    assert research.equity_figure.layout.title.text == "樣本外淨值"
    written = write_report(research, tmp_path)
    trials = pd.read_csv(written["trials_csv"])
    assert len(trials) == research.selection["n_trials"]
    assert "holm_reject" in trials.columns
    assert "bh_reject" in trials.columns
    payload = json.loads(written["metrics_json"].read_text(encoding="utf-8"))
    assert payload["selection"]["dsr"] == pytest.approx(research.selection["dsr"])


class BaseStrategyHold(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index, dtype=int)
