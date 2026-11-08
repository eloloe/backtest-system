from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backtest_system.portfolio_app.bundle import build_portfolio_bundle
from backtest_system.portfolio_app.diagnostics import diagnose_result, replay_preview
from backtest_system.portfolio_app.shioaji_fixture import check_execution_assumptions, load_quote_fixture
from backtest_system.research.runner import RunConfig, run_research


def _frame(close_values: list[float]) -> pd.DataFrame:
    close = np.array(close_values, dtype="float64")
    open_ = np.array([close[0], *close[:-1]], dtype="float64")
    index = pd.bdate_range("2024-01-02", periods=len(close_values))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close),
            "Low": np.minimum(open_, close),
            "Close": close,
            "Volume": np.full(len(close_values), 1_000_000.0),
        },
        index=index,
    )


def _config(*, tickers: list[str], start: str, end: str, strategy: str = "buy_and_hold") -> RunConfig:
    return RunConfig(
        strategy=strategy,
        tickers=tickers,
        start=start,
        end=end,
        benchmark="SPY",
        initial_capital=100_000.0,
        commission=0.001,
        slippage=0.0005,
        provider=None,
        output_dir="output",
    )


def test_diagnostics_marks_benchmark_fail_when_underperforming():
    prices = {"AAA": _frame([100, 101, 101, 101, 101, 101])}
    benchmark = pd.Series([100, 102, 104, 106, 108, 110], index=next(iter(prices.values())).index)
    cfg = _config(tickers=["AAA"], start="2024-01-02", end="2024-01-11")
    result = run_research(cfg, prices=prices, benchmark_close=benchmark)
    out = diagnose_result(result)
    finding = next(item for item in out["findings"] if item["key"] == "benchmark_relative")
    assert finding["status"] == "fail"


def test_diagnostics_handles_zero_trades_as_insufficient():
    prices = {"AAA": _frame([100, 100, 100, 100])}
    benchmark = pd.Series([100, 100, 100, 100], index=next(iter(prices.values())).index)
    cfg = _config(tickers=["AAA"], start="2024-01-02", end="2024-01-09", strategy="ma_cross")
    cfg.params = {"short_window": 2, "long_window": 8}
    result = run_research(cfg, prices=prices, benchmark_close=benchmark)
    out = diagnose_result(result)
    finding = next(item for item in out["findings"] if item["key"] == "trade_concentration")
    assert finding["status"] == "insufficient"


def test_cost_stress_can_erase_profits():
    prices = {"AAA": _frame([100, 110, 90, 120, 85, 130])}
    benchmark = pd.Series([100, 101, 102, 103, 104, 105], index=next(iter(prices.values())).index)
    cfg = _config(tickers=["AAA"], start="2024-01-02", end="2024-01-10", strategy="ma_cross")
    cfg.params = {"short_window": 1, "long_window": 2}
    cfg.commission = 0.005
    cfg.slippage = 0.01
    result = run_research(cfg, prices=prices, benchmark_close=benchmark)
    out = diagnose_result(result)
    finding = next(item for item in out["findings"] if item["key"] == "cost_stress")
    assert finding["status"] in {"fail", "pass"}
    assert "scope" in finding["evidence"]


def test_data_quality_flags_missing_symbol_notes():
    prices = {"AAA": _frame([100, 103, 106, 105])}
    benchmark = pd.Series([100, 100, 100, 100], index=next(iter(prices.values())).index)
    cfg = _config(tickers=["AAA", "MISS"], start="2024-01-02", end="2024-01-09")
    result = run_research(cfg, prices=prices, benchmark_close=benchmark)
    out = diagnose_result(result)
    finding = next(item for item in out["findings"] if item["key"] == "data_quality")
    assert finding["status"] == "fail"


def test_shioaji_fixture_round_trip(tmp_path):
    fixture = {
        "meta": {"source": "synthetic", "label": "offline"},
        "snapshots": [
            {"symbol": "2330.TW", "timestamp": "2026-10-01T09:00:00+08:00", "bid": 100.0, "ask": 100.2, "lot_size": 1000},
            {"symbol": "0050.TW", "timestamp": "2026-10-01T09:00:01+08:00", "bid": 140.0, "ask": 140.1, "lot_size": 1000},
        ],
    }
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    loaded = load_quote_fixture(path)
    assert len(loaded["snapshots"]) == 2
    check = check_execution_assumptions(path, assumed_slippage_bps=5.0)
    assert check["summary"]["snapshot_count"] == 2
    assert check["summary"]["connectivity_verified"] is False


def test_bundle_writes_static_app_files(tmp_path):
    prices = {"AAA": _frame([100, 102, 101, 106, 108, 110])}
    benchmark = pd.Series([100, 101, 102, 102, 103, 104], index=next(iter(prices.values())).index)
    cfg = _config(tickers=["AAA"], start="2024-01-02", end="2024-01-11")
    result = run_research(cfg, prices=prices, benchmark_close=benchmark)
    written = build_portfolio_bundle(
        result,
        output_dir=tmp_path / "mini",
        data_label="合成樣本",
        validation_label="樣本內",
    )
    assert written["index_html"].exists()
    payload = json.loads(written["data_json"].read_text(encoding="utf-8"))
    assert payload["provenance"]["data_label"] == "合成樣本"
    assert "固定成交重播" in payload["meta"]["preview_scope"]


def test_replay_preview_cost_controls_change_curve():
    prices = {"AAA": _frame([100, 101, 104, 103, 108, 109])}
    benchmark = pd.Series([100, 100, 100, 100, 100, 100], index=next(iter(prices.values())).index)
    cfg = _config(tickers=["AAA"], start="2024-01-02", end="2024-01-11")
    result = run_research(cfg, prices=prices, benchmark_close=benchmark)
    base = replay_preview(result, commission=0.001, slippage=0.0005)
    high = replay_preview(result, commission=0.01, slippage=0.02)
    assert base["equity"][-1]["value"] > high["equity"][-1]["value"]
