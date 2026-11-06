"""Second-engine P&L check. Nautilus cases skip when the extra is not installed."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_system.engine.backtester import Backtester
from backtest_system.engine.crosscheck import (
    OptionalEngineError,
    load_crosscheck,
    nautilus_atol,
    replay_equity,
    run_crosscheck,
)
from backtest_system.research.runner import ConfigError, RunConfig, attach_crosscheck, load_run_config, run_research, write_report

ROOT = Path(__file__).resolve().parents[1]


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


def _signals(data, values) -> pd.Series:
    return pd.Series(values, index=data.index, dtype=float)


def _nautilus():
    pytest.importorskip("nautilus_trader")


def test_crosscheck_import_does_not_pull_nautilus():
    code = (
        "import backtest_system.engine, backtest_system.engine.crosscheck, sys; "
        "raise SystemExit(0 if 'nautilus_trader' not in sys.modules else 1)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")])
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=False, env=env)
    assert completed.returncode == 0


def test_missing_nautilus_is_a_clear_install_error(monkeypatch):
    real = importlib.util.find_spec

    def _hidden(name, package=None):
        if name == "nautilus_trader":
            return None
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _hidden)
    checker = load_crosscheck("nautilus")
    assert checker.available() is False
    with pytest.raises(OptionalEngineError, match=r"\.\[nautilus\]"):
        checker.check(
            pd.DataFrame(),
            pd.Series([1.0]),
            1.0,
            pd.Series([1.0]),
            bars=_ohlcv([1.0], [1.0]),
        )


def test_nautilus_without_bars_explains_what_it_needs():
    if importlib.util.find_spec("nautilus_trader") is None:
        pytest.skip("nautilus_trader is not installed")
    checker = load_crosscheck("nautilus")
    with pytest.raises(OptionalEngineError, match="Open/Close"):
        checker.check(pd.DataFrame(), pd.Series([1.0], index=pd.bdate_range("2024-01-02", periods=1)), 1.0, pd.Series([1.0]))


def test_reference_crosscheck_still_writes_equity_and_drawdown(tmp_path):
    index = pd.bdate_range("2020-01-02", periods=8)
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
        cross_check="reference",
        output_dir=str(tmp_path),
    )
    result = run_research(config, prices={"AAA": _ohlcv(close, close, start="2020-01-02")})
    report = attach_crosscheck(result)
    written = write_report(result, tmp_path)
    assert report.ok
    assert report.trade_count == report.other_trade_count
    assert result.equity_figure.layout.title.text == "淨值"
    assert result.drawdown_figure.layout.title.text == "回撤"
    equity_html = written["equity_html"].read_text(encoding="utf-8")
    drawdown_html = written["drawdown_html"].read_text(encoding="utf-8")
    assert "\\u6de8\\u503c" in equity_html
    assert "\\u56de\\u64a4" in drawdown_html
    # Plotly stores the trace name as a JSON unicode escape, same as the title.
    assert "\\u6210\\u4ea4\\u91cd\\u64ad" in equity_html
    payload = written["crosscheck_json"].read_text(encoding="utf-8")
    assert '"ok": true' in payload
    assert written["crosscheck_equity_csv"].is_file()


def test_yaml_cross_check_and_unknown_name(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "strategy: buy_and_hold\ntickers: [AAA]\nstart: 2020-01-01\nend: 2021-01-01\n"
        "cross_check: nautilus\n",
        encoding="utf-8",
    )
    loaded = load_run_config(path, {"cross_check": "reference"})
    assert loaded.cross_check == "reference"
    path.write_text(
        "strategy: buy_and_hold\ntickers: [AAA]\nstart: 2020-01-01\nend: 2021-01-01\n"
        "cross_check: bloomberg\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="cross_check"):
        load_run_config(path)


def test_nautilus_matches_a_round_trip_with_zero_costs():
    _nautilus()
    data = _ohlcv([10.0, 10.0, 12.0, 11.0], [9.0, 11.0, 13.0, 12.0])
    bt = Backtester(initial_capital=10_000, commission=0.0, slippage=0.0)
    portfolio, trades = bt.run(data, _signals(data, [1, 1, 0, 0]), symbol="AAPL")
    run = run_crosscheck(
        "nautilus",
        trades,
        portfolio["price"],
        10_000,
        portfolio["portfolio_value"],
        bars=data,
        costs=bt.cost_model(symbol="AAPL"),
        symbol="AAPL",
        atol=1e-6,
        rtol=0.0,
    )
    assert run.report.ok, run.report
    assert run.report.max_abs_diff < 1e-6
    assert run.report.trade_count == 2
    assert run.report.other_trade_count == 2
    assert run.report.material_fills == 0
    assert run.report.max_fill_price_diff == pytest.approx(0.0, abs=1e-9)
    assert run.report.total_pnl == pytest.approx(run.report.other_pnl, abs=1e-6)
    assert run.equity.tolist() == pytest.approx(portfolio["portfolio_value"].tolist(), abs=1e-6)


def test_nautilus_matches_us_commission_inside_the_documented_band():
    _nautilus()
    data = _ohlcv([10.0, 10.0, 12.0, 11.0], [10.0, 11.0, 13.0, 12.0])
    bt = Backtester(initial_capital=10_000, commission=0.001, slippage=0.0, commission_flat=1.0)
    portfolio, trades = bt.run(data, _signals(data, [1, 1, 0, 0]), symbol="MSFT")
    exposure = float(portfolio["portfolio_value"].abs().max())
    atol = nautilus_atol(len(trades), exposure)
    run = run_crosscheck(
        "nautilus",
        trades,
        portfolio["price"],
        10_000,
        portfolio["portfolio_value"],
        bars=data,
        costs=bt.cost_model(symbol="MSFT"),
        symbol="MSFT",
        atol=atol,
    )
    assert run.report.ok, run.report
    assert run.report.material_fills == 0
    assert abs(run.report.total_pnl - run.report.other_pnl) <= atol


def test_nautilus_matches_taiwan_minimum_and_sell_tax():
    _nautilus()
    data = _ohlcv([100.0, 100.0, 110.0, 108.0], [100.0, 105.0, 112.0, 108.0])
    bt = Backtester(initial_capital=1_000, commission=0.0, slippage=0.0)
    portfolio, trades = bt.run(data, _signals(data, [1, 1, 0, 0]), symbol="2330.TW")
    atol = nautilus_atol(len(trades), float(portfolio["portfolio_value"].abs().max()))
    run = run_crosscheck(
        "nautilus",
        trades,
        portfolio["price"],
        1_000,
        portfolio["portfolio_value"],
        bars=data,
        costs=bt.cost_model(symbol="2330.TW"),
        symbol="2330.TW",
        atol=atol,
    )
    assert run.report.ok, run.report
    assert list(trades["action"]) == ["BUY", "SELL"]
    assert float(trades.iloc[0]["fee"]) == pytest.approx(20.0)
    assert run.report.material_fills == 0


def test_nautilus_matches_a_futures_multiplier_by_scaling_quantity():
    _nautilus()
    data = _ohlcv([80.0, 80.0, 90.0, 100.0], [80.0, 90.0, 100.0, 100.0])
    bt = Backtester(
        initial_capital=10_000,
        commission=0.0,
        slippage=0.0,
        contract_multiplier=50,
        min_lot=1,
    )
    portfolio, trades = bt.run(data, _signals(data, [1, 1, 0, 0]), market="FUTURES")
    run = run_crosscheck(
        "nautilus",
        trades,
        portfolio["price"],
        10_000,
        portfolio["portfolio_value"],
        bars=data,
        costs=bt.cost_model(market="FUTURES"),
        contract_multiplier=50,
        symbol="ES",
        atol=1e-6,
    )
    assert run.report.ok, run.report
    assert trades.iloc[0]["shares"] == pytest.approx(2.0)
    assert run.fills.iloc[0]["shares"] == pytest.approx(2.0)
    assert run.fills.iloc[0]["price"] == pytest.approx(80.0)


def test_nautilus_matches_two_sleeves_and_drops_a_name_without_prices(tmp_path):
    _nautilus()
    index = pd.bdate_range("2021-04-01", periods=6)
    frames = {}
    for symbol, level in (("AAA", 10.0), ("BBB", 20.0)):
        close = np.linspace(level, level + 4, len(index))
        opens = close.copy()
        opens[1:] = close[:-1]
        frames[symbol] = _ohlcv(opens, close, start="2021-04-01")
        frames[symbol].index = index
    config = RunConfig(
        strategy="buy_and_hold",
        tickers=["AAA", "BBB", "MISSING"],
        start=index[0].date().isoformat(),
        end=(index[-1] + pd.Timedelta(days=1)).date().isoformat(),
        benchmark=None,
        initial_capital=90_000,
        commission=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        warmup_bars=0,
        cross_check="nautilus",
    )
    result = run_research(config, prices=frames)
    report = attach_crosscheck(result)
    written = write_report(result, tmp_path)
    assert result.equity.iloc[0] == pytest.approx(60_000)
    assert report.ok, report
    assert report.material_fills == 0
    assert report.max_abs_diff < 1e-6
    assert result.drawdown_figure.layout.title.text == "回撤"
    equity_html = written["equity_html"].read_text(encoding="utf-8")
    assert "Nautilus" in equity_html
    assert "\\u56de\\u64a4" in written["drawdown_html"].read_text(encoding="utf-8")
    assert '"ok": true' in written["crosscheck_json"].read_text(encoding="utf-8")


def test_nautilus_sell_then_buy_on_the_same_open_uses_the_sale_proceeds():
    _nautilus()
    data = _ohlcv([10.0, 10.0, 12.0, 11.0], [10.0, 11.0, 13.0, 12.0])
    day2 = data.index[2]
    trades = pd.DataFrame(
        [
            {"date": data.index[1], "action": "BUY", "price": 10.0, "shares": 10.0, "value": 100.0, "fee": 0.0},
            {"date": day2, "action": "SELL", "price": 12.0, "shares": 10.0, "value": 120.0, "fee": 0.0},
            {"date": day2, "action": "BUY", "price": 12.0, "shares": 4.0, "value": 48.0, "fee": 0.0},
        ]
    )
    equity = replay_equity(trades, data["Close"], 1_000, symbol="AAA")
    run = run_crosscheck(
        "nautilus",
        trades,
        data["Close"],
        1_000,
        equity,
        bars=data,
        costs=Backtester(commission=0.0, slippage=0.0).cost_model(symbol="AAA"),
        symbol="AAA",
        atol=1e-6,
    )
    assert run.report.ok, run.report
    assert list(run.fills["action"]) == ["BUY", "SELL", "BUY"]
    assert run.report.other_trade_count == 3


def test_slippage_is_a_material_fill_gap_even_if_the_equity_band_is_wide():
    _nautilus()
    data = _ohlcv([10.0, 10.0, 12.0, 11.0], [9.0, 11.0, 13.0, 12.0])
    bt = Backtester(initial_capital=10_000, commission=0.0, slippage=0.01)
    portfolio, trades = bt.run(data, _signals(data, [1, 1, 0, 0]), symbol="AAPL")
    run = run_crosscheck(
        "nautilus",
        trades,
        portfolio["price"],
        10_000,
        portfolio["portfolio_value"],
        bars=data,
        costs=bt.cost_model(symbol="AAPL"),
        symbol="AAPL",
        atol=1e12,
        rtol=0.0,
    )
    assert not run.report.ok
    assert run.report.material_fills > 0
    # Sell fills the last open (11), slipped by 1%. Nautilus prints the raw open.
    assert run.report.max_fill_price_diff == pytest.approx(0.11, abs=1e-9)
    assert "Nautilus prints the market order at the session open" in " ".join(run.report.notes)


def test_a_shifted_equity_is_flagged_when_the_fills_still_agree():
    _nautilus()
    data = _ohlcv([10.0, 10.0, 12.0, 11.0], [9.0, 11.0, 13.0, 12.0])
    bt = Backtester(initial_capital=10_000, commission=0.0, slippage=0.0)
    portfolio, trades = bt.run(data, _signals(data, [1, 1, 0, 0]), symbol="AAPL")
    bumped = portfolio["portfolio_value"] + 500.0
    run = run_crosscheck(
        "nautilus",
        trades,
        portfolio["price"],
        10_000,
        bumped,
        bars=data,
        costs=bt.cost_model(symbol="AAPL"),
        symbol="AAPL",
        atol=1e-6,
    )
    assert not run.report.ok
    assert run.report.material_fills == 0
    assert run.report.max_abs_diff == pytest.approx(500.0, abs=1e-6)
    assert run.report.trade_count == run.report.other_trade_count


def test_empty_trade_list_against_a_moving_curve_diverges():
    _nautilus()
    data = _ohlcv([10.0, 10.0, 12.0, 11.0], [9.0, 11.0, 13.0, 12.0])
    bt = Backtester(initial_capital=10_000, commission=0.0, slippage=0.0)
    portfolio, trades = bt.run(data, _signals(data, [1, 1, 0, 0]), symbol="AAPL")
    run = run_crosscheck(
        "nautilus",
        trades.iloc[0:0],
        portfolio["price"],
        10_000,
        portfolio["portfolio_value"],
        bars=data,
        costs=bt.cost_model(symbol="AAPL"),
        symbol="AAPL",
        atol=1e-6,
    )
    assert not run.report.ok
    assert run.report.trade_count == 0
    assert run.report.other_trade_count == 0
    assert run.report.max_abs_diff > 100
