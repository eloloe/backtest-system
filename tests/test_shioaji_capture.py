from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from backtest_system.portfolio_app.shioaji_capture import CaptureConfig, collect_quotes
from backtest_system.providers.base import MissingApiKeyError


class _FakeQuote:
    def __init__(self):
        self.callback = None
        self.subscriptions = []
        self.unsubscriptions = []

    def set_on_bidask_stk_v1_callback(self, cb):
        self.callback = cb

    def subscribe(self, contract, **kwargs):
        self.subscriptions.append((contract, kwargs))
        quote = {
            "code": contract,
            "datetime": datetime(2026, 11, 8, 9, 0, tzinfo=timezone.utc),
            "bid_price": [100.0, 99.9],
            "ask_price": [100.1, 100.2],
            "bid_volume": [120, 80],
            "ask_volume": [90, 70],
        }
        self.callback(f"QUT/{contract}", quote)

    def unsubscribe(self, contract, **kwargs):
        self.unsubscriptions.append((contract, kwargs))


class _FakeContracts:
    def __init__(self):
        self.Stocks = {"2330": "2330", "0050": "0050"}
        self.Futures = {"TXF": {"TXFR1": "TXFR1"}}


class _FakeApi:
    def __init__(self, simulation=True):
        self.simulation = simulation
        self.quote = _FakeQuote()
        self.Contracts = _FakeContracts()
        self.login_called = False
        self.logout_called = False

    def login(self, api_key, secret_key):
        self.login_called = bool(api_key and secret_key)

    def logout(self):
        self.logout_called = True


class _FakeSj:
    def __init__(self):
        self.created = []

    def Shioaji(self, simulation=True):
        api = _FakeApi(simulation=simulation)
        self.created.append(api)
        return api


def test_collect_quotes_requires_env_keys(monkeypatch):
    monkeypatch.delenv("SHIOAJI_API_KEY", raising=False)
    monkeypatch.delenv("SHIOAJI_SECRET_KEY", raising=False)
    monkeypatch.setattr("backtest_system.portfolio_app.shioaji_capture._load_shioaji", lambda: _FakeSj())
    with pytest.raises(MissingApiKeyError):
        collect_quotes(CaptureConfig(symbols=("2330.TW",), duration_seconds=0.01))


def test_collect_quotes_subscribe_and_cleanup(monkeypatch):
    fake = _FakeSj()
    monkeypatch.setenv("SHIOAJI_API_KEY", "k")
    monkeypatch.setenv("SHIOAJI_SECRET_KEY", "s")
    monkeypatch.setenv("SHIOAJI_SIMULATION", "1")
    monkeypatch.setattr("backtest_system.portfolio_app.shioaji_capture._load_shioaji", lambda: fake)
    out = collect_quotes(CaptureConfig(symbols=("2330.TW", "TXF"), duration_seconds=0.01))
    assert out["meta"]["simulation"] is True
    assert out["meta"]["sdk_connection_smoke_test"] == "not_run_in_ci_requires_local_account"
    assert len(out["snapshots"]) >= 2
    api = fake.created[0]
    assert api.login_called
    assert api.logout_called
    assert len(api.quote.subscriptions) == 2
    assert len(api.quote.unsubscriptions) == 2
    assert out["snapshots"][0]["bid"] > 0
    assert "depth" in out["snapshots"][0]


def test_collect_quotes_no_depth(monkeypatch):
    fake = _FakeSj()
    monkeypatch.setenv("SHIOAJI_API_KEY", "k")
    monkeypatch.setenv("SHIOAJI_SECRET_KEY", "s")
    monkeypatch.setattr("backtest_system.portfolio_app.shioaji_capture._load_shioaji", lambda: fake)
    out = collect_quotes(CaptureConfig(symbols=("2330",), duration_seconds=0.01, include_depth=False, simulation=False))
    assert out["meta"]["simulation"] is False
    assert "depth" not in out["snapshots"][0]
