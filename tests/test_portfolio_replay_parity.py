from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from backtest_system.portfolio_app.replay import ReplayCosts, replay_fixed_trades


def _payload():
    calendar = ["2024-01-02", "2024-01-03", "2024-01-04"]
    prices = {
        "AAA": {
            "2024-01-02": {"open": 100.0, "close": 100.0},
            "2024-01-03": {"open": 110.0, "close": 111.0},
            "2024-01-04": {"open": 95.0, "close": 98.0},
        }
    }
    trades = [
        {"date": "2024-01-02", "symbol": "AAA", "action": "BUY", "shares": 3.0, "market": "US"},
        {"date": "2024-01-03", "symbol": "AAA", "action": "SELL", "shares": 5.0, "market": "US"},
        {"date": "2024-01-04", "symbol": "AAA", "action": "BUY", "shares": 9999.0, "market": "US"},
    ]
    return {
        "meta": {
            "initial_capital": 1_000.0,
            "commission_flat": 0.0,
            "tw_brokerage_rate": 0.001425,
            "tw_brokerage_min": 20.0,
            "tw_sell_tax_rate": 0.003,
        },
        "series": {"calendar": calendar, "prices": prices, "trades": trades},
    }


def _python_result(payload, commission, slippage):
    calendar = pd.DatetimeIndex(payload["series"]["calendar"])
    prices = {}
    for symbol, rows in payload["series"]["prices"].items():
        frame = pd.DataFrame(
            {"Open": {k: v["open"] for k, v in rows.items()}, "Close": {k: v["close"] for k, v in rows.items()}}
        )
        frame.index = pd.to_datetime(frame.index)
        prices[symbol] = frame
    trades = pd.DataFrame(payload["series"]["trades"])
    costs = ReplayCosts(
        commission_rate=commission,
        commission_flat=0.0,
        slippage_rate=slippage,
        tw_brokerage_rate=0.001425,
        tw_brokerage_min=20.0,
        tw_sell_tax_rate=0.003,
    )
    return replay_fixed_trades(trades=trades, prices=prices, calendar=calendar, initial_capital=1_000.0, costs=costs)


def _js_result(payload, commission, slippage):
    module = Path("/agent/repos/backtest-system/src/backtest_system/portfolio_app/static/replay_runtime.js")
    script = (
        "import { replayFixedTrades } from 'file://"
        + str(module)
        + "';"
        + "const payload = JSON.parse(process.env.PAYLOAD);"
        + "const out = replayFixedTrades(payload, Number(process.env.C), Number(process.env.S));"
        + "process.stdout.write(JSON.stringify(out));"
    )
    env = dict(os.environ)
    env.update({"PAYLOAD": json.dumps(payload), "C": str(commission), "S": str(slippage)})
    proc = subprocess.run(["node", "--input-type=module", "-e", script], text=True, capture_output=True, check=True, env=env)
    return json.loads(proc.stdout)


def test_js_and_python_replay_parity_for_denials():
    payload = _payload()
    commission = 0.01
    slippage = 0.02
    py = _python_result(payload, commission, slippage)
    js = _js_result(payload, commission, slippage)
    assert py.trades_denied == js["trades_denied"]
    assert py.trades_applied == js["trades_applied"]
    assert len(py.equity) == len(js["equity"])
    assert py.equity.iloc[-1] == pytest.approx(js["equity"][-1]["value"], abs=1e-9)
    assert py.drawdown.iloc[-1] == pytest.approx(js["drawdown"][-1]["value"], abs=1e-9)


def test_replay_handles_no_trades():
    payload = _payload()
    payload["series"]["trades"] = []
    py = _python_result(payload, commission=0.001, slippage=0.001)
    js = _js_result(payload, commission=0.001, slippage=0.001)
    assert py.trades_applied == js["trades_applied"] == 0
    assert py.trades_denied == js["trades_denied"] == 0
    assert py.equity.iloc[0] == pytest.approx(1_000.0)
