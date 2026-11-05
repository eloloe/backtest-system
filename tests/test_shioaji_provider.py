"""Shioaji provider: symbols, daily aggregation, keys. No network."""

import sys

import pandas as pd
import pytest

from data_fetcher import fetch_data
from providers import CachingProvider, DataProviderConfigError, MissingApiKeyError, get_provider
from providers.base import ProviderError
from providers.shioaji import ShioajiProvider, contract_code


def _wall_ns(text):
    # The SDK stores Taiwan wall time in the UTC fields of a ns stamp.
    return int(pd.Timestamp(text, tz="UTC").value)


def _bar(when, open_, high, low, close, volume):
    return {
        "ts": pd.Timestamp(when),
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }


class _Bars:
    def __init__(self, payload):
        self._payload = payload

    def dict(self):
        return self._payload


class _Api:
    def __init__(self, rows):
        self.rows = rows
        self.missing = set()
        self.simulation = None
        self.logins = []
        self.gets = []
        self.calls = []
        self.fail_logins = 0

    def login(self, api_key, secret_key, subscribe_trade):
        self.logins.append(
            {
                "api_key": api_key,
                "secret_key": secret_key,
                "subscribe_trade": subscribe_trade,
            }
        )
        if self.fail_logins:
            self.fail_logins -= 1
            raise RuntimeError(f"login rejected {secret_key} / {api_key}")

    @property
    def contracts(self):
        return self

    def get(self, code):
        self.gets.append(code)
        if code in self.missing:
            return None
        return {"code": code}

    def kbars(self, contract, start, end, timeout):
        self.calls.append(
            {"contract": contract, "start": start, "end": end, "timeout": timeout}
        )
        start_ts = pd.Timestamp(start)
        # Shioaji's end date is inclusive.
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
        kept = [row for row in self.rows if start_ts <= row["ts"] < end_ts]
        payload = {col: [] for col in ("ts", "Open", "High", "Low", "Close", "Volume")}
        for row in kept:
            payload["ts"].append(_wall_ns(row["ts"]))
            for col in ("Open", "High", "Low", "Close", "Volume"):
                payload[col].append(row[col])
        return _Bars(payload)


class _Module:
    def __init__(self, api):
        self.api = api

    def Shioaji(self, simulation):
        self.api.simulation = simulation
        return self.api


def _install(monkeypatch, rows):
    api = _Api(rows)
    monkeypatch.setattr("providers.shioaji._load_shioaji", lambda: _Module(api))
    return api


def _keys(monkeypatch, api_key="api-key-value", secret="super-secret"):
    monkeypatch.setenv("SHIOAJI_API_KEY", api_key)
    monkeypatch.setenv("SHIOAJI_SECRET_KEY", secret)


def test_shioaji_requires_both_keys(monkeypatch):
    with pytest.raises(MissingApiKeyError, match="SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY"):
        get_provider("shioaji")

    monkeypatch.setenv("SHIOAJI_API_KEY", "id")
    with pytest.raises(MissingApiKeyError, match="SHIOAJI_SECRET_KEY"):
        get_provider("Shioaji")

    monkeypatch.delenv("SHIOAJI_API_KEY")
    monkeypatch.setenv("SHIOAJI_SECRET_KEY", "   ")
    with pytest.raises(MissingApiKeyError, match="SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY"):
        ShioajiProvider()


def test_blank_secret_is_missing(monkeypatch):
    monkeypatch.setenv("SHIOAJI_API_KEY", "id")
    monkeypatch.setenv("SHIOAJI_SECRET_KEY", "   ")
    with pytest.raises(MissingApiKeyError, match="SHIOAJI_SECRET_KEY is not set"):
        ShioajiProvider()


def test_simulation_flag_must_be_boolean(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setenv("SHIOAJI_SIMULATION", "maybe")
    with pytest.raises(DataProviderConfigError, match="SHIOAJI_SIMULATION"):
        ShioajiProvider()


def test_cache_keeps_the_shioaji_name(monkeypatch, tmp_path):
    _keys(monkeypatch)
    monkeypatch.setenv("BACKTEST_DATA_CACHE", "1")
    monkeypatch.setenv("BACKTEST_DATA_CACHE_DIR", str(tmp_path))
    provider = get_provider("shioaji")
    assert isinstance(provider, CachingProvider)
    assert provider.name == "shioaji"
    assert provider.cache_dir == tmp_path


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("2330.TW", "2330"),
        ("2330.tw", "2330"),
        ("2330.TWO", "2330"),
        (" 0050.TWO ", "0050"),
        ("0050", "0050"),
        ("00631L", "00631L"),
        ("TXF", "TXFR1"),
        ("txf", "TXFR1"),
        ("TXFR2", "TXFR2"),
        ("MXFA6", "MXFA6"),
    ],
)
def test_contract_code_strips_yahoo_suffixes_and_maps_the_near_month(symbol, expected):
    assert contract_code(symbol) == expected


@pytest.mark.parametrize("symbol", ["", "   ", ".TW", "AAPL", "BTC-USD", "BRK.B"])
def test_contract_code_rejects_empty_and_non_taiwan_symbols(symbol):
    with pytest.raises(ProviderError):
        contract_code(symbol)


def test_daily_bars_from_minutes_drop_the_exclusive_end(monkeypatch):
    rows = [
        _bar("2020-01-02 09:00", 10, 12, 9, 11, 100),
        _bar("2020-01-02 09:01", 11, 13, 10, 12, 50),
        # 15:30 stays on the stock's calendar day. Futures would roll it.
        _bar("2020-01-02 15:30", 12, 14, 11, 13, 5),
        _bar("2020-01-03 09:00", 20, 21, 19, 20.5, 7),
        _bar("2020-01-06 09:00", 99, 99, 99, 99, 1),
    ]
    api = _install(monkeypatch, rows)
    _keys(monkeypatch)
    provider = ShioajiProvider()
    frame = provider.fetch_ohlcv(" 2330.TW ", "2020-01-01", "2020-01-06")

    assert api.simulation is True
    assert api.logins == [
        {
            "api_key": "api-key-value",
            "secret_key": "super-secret",
            "subscribe_trade": False,
        }
    ]
    assert api.gets == ["2330"]
    assert api.calls == [
        {
            "contract": {"code": "2330"},
            "start": "2020-01-01",
            "end": "2020-01-05",
            "timeout": 30000,
        }
    ]
    assert frame.index.tz is None
    assert frame.index.name == "Date"
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert list(frame.dtypes) == [float] * 5
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03"]
    assert frame.loc[pd.Timestamp("2020-01-02"), "Open"] == 10
    assert frame.loc[pd.Timestamp("2020-01-02"), "High"] == 14
    assert frame.loc[pd.Timestamp("2020-01-02"), "Low"] == 9
    assert frame.loc[pd.Timestamp("2020-01-02"), "Close"] == 13
    assert frame.loc[pd.Timestamp("2020-01-02"), "Volume"] == 155
    assert frame.loc[pd.Timestamp("2020-01-03"), "Close"] == 20.5

    again = provider.fetch_ohlcv("2330.TWO", "2020-01-01", "2020-01-06")
    assert len(api.logins) == 1
    assert api.gets == ["2330", "2330"]
    pd.testing.assert_frame_equal(frame, again)
    assert "super-secret" not in repr(api.calls)
    assert "super-secret" not in repr(api.gets)


def test_empty_window_does_not_log_in(monkeypatch):
    api = _install(monkeypatch, [_bar("2020-01-02 09:00", 1, 1, 1, 1, 1)])
    _keys(monkeypatch)

    def _boom():
        raise AssertionError("shioaji should not be imported")

    monkeypatch.setattr("providers.shioaji._load_shioaji", _boom)
    frame = ShioajiProvider().fetch_ohlcv("2330.TW", "2020-02-01", "2020-01-01")
    assert frame.empty
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert api.logins == []


def test_empty_kbars_are_an_empty_ohlcv_frame(monkeypatch):
    api = _install(monkeypatch, [])
    _keys(monkeypatch)
    frame = ShioajiProvider().fetch_ohlcv("0050.TW", "2020-01-01", "2020-02-01")
    assert frame.empty
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert api.gets == ["0050"]


def test_long_history_is_split_into_30_day_kbars(monkeypatch):
    api = _install(monkeypatch, [])
    _keys(monkeypatch)
    frame = ShioajiProvider().fetch_ohlcv("2330", "2020-01-01", "2020-03-01")
    assert frame.empty
    assert [(call["start"], call["end"]) for call in api.calls] == [
        ("2020-01-01", "2020-01-30"),
        ("2020-01-31", "2020-02-29"),
    ]


def test_txf_near_month_rolls_friday_night_onto_monday(monkeypatch):
    # 2020-01-03 is a Friday. 15:00 onward, including Saturday morning,
    # belongs to Monday 2020-01-06.
    rows = [
        _bar("2020-01-03 13:44", 100, 101, 99, 100, 1),
        _bar("2020-01-03 15:01", 110, 112, 108, 111, 2),
        _bar("2020-01-04 02:00", 111, 113, 107, 109, 3),
        _bar("2020-01-06 08:46", 109, 115, 106, 114, 4),
    ]
    api = _install(monkeypatch, rows)
    _keys(monkeypatch)
    frame = ShioajiProvider().fetch_ohlcv("txf", "2020-01-06", "2020-01-07")

    assert api.gets == ["TXFR1"]
    assert api.calls[0]["start"] == "2020-01-03"
    assert api.calls[0]["end"] == "2020-01-06"
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2020-01-06"]
    assert frame.iloc[0].to_dict() == {
        "Open": 110.0,
        "High": 115.0,
        "Low": 106.0,
        "Close": 114.0,
        "Volume": 9.0,
    }


def test_us_symbol_does_not_log_in(monkeypatch):
    api = _install(monkeypatch, [])
    _keys(monkeypatch)
    with pytest.raises(ProviderError, match="AAPL"):
        ShioajiProvider().fetch_ohlcv("AAPL", "2020-01-01", "2020-02-01")
    assert api.logins == []


def test_missing_contract(monkeypatch):
    api = _install(monkeypatch, [])
    api.missing.add("9999")
    _keys(monkeypatch)
    with pytest.raises(ProviderError, match="no contract for 9999") as exc:
        ShioajiProvider().fetch_ohlcv("9999", "2020-01-01", "2020-02-01")
    assert "super-secret" not in str(exc.value)


def test_login_failure_hides_the_secret_and_retries(monkeypatch):
    api = _install(monkeypatch, [])
    api.fail_logins = 1
    _keys(monkeypatch)
    provider = ShioajiProvider()
    with pytest.raises(ProviderError, match="login failed") as exc:
        provider.fetch_ohlcv("2330", "2020-01-01", "2020-02-01")
    assert "super-secret" not in str(exc.value)
    assert "api-key-value" not in str(exc.value)

    frame = provider.fetch_ohlcv("2330", "2020-01-01", "2020-02-01")
    assert frame.empty
    assert len(api.logins) == 2


def test_production_flag_turns_simulation_off(monkeypatch):
    api = _install(monkeypatch, [])
    _keys(monkeypatch)
    monkeypatch.setenv("SHIOAJI_SIMULATION", "0")
    ShioajiProvider().fetch_ohlcv("2330", "2020-01-01", "2020-01-03")
    assert api.simulation is False


def test_missing_package_is_a_config_error(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setitem(sys.modules, "shioaji", None)
    with pytest.raises(DataProviderConfigError, match="not installed"):
        fetch_data("2330.TW", "2020-01-01", "2020-02-01", provider="shioaji")


def test_fetch_data_swallows_kbars_errors(monkeypatch, capsys):
    api = _install(monkeypatch, [])
    _keys(monkeypatch)

    def _boom(**kwargs):
        raise RuntimeError("kbars down")

    api.kbars = _boom
    frame = fetch_data("2330.TW", "2020-01-01", "2020-02-01", provider="shioaji")
    assert frame.empty
    message = "Error fetching 2330.TW from shioaji: Shioaji kbars failed: kbars down"
    assert message in capsys.readouterr().out
