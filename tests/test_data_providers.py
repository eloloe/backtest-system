"""Provider selection, HTTP parsing, cache, and the fetch_data entry point."""

import pandas as pd
import pytest

from providers import (
    CachingProvider,
    DataProviderConfigError,
    MissingApiKeyError,
    PolygonProvider,
    TiingoProvider,
    YahooProvider,
    get_provider,
)
from providers.base import ProviderError, empty_ohlcv
from providers.yahoo import yf


class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def _patch_http(monkeypatch, payload, status=200):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["url"] = url
        seen["params"] = params or {}
        seen["headers"] = headers or {}
        seen["timeout"] = timeout
        return _Response(status, payload)

    monkeypatch.setattr("providers.base.requests.get", fake_get)
    return seen


def test_default_provider_is_yahoo():
    provider = get_provider()
    assert isinstance(provider, YahooProvider)
    assert provider.name == "yahoo"


def test_name_overrides_env(monkeypatch):
    monkeypatch.setenv("BACKTEST_DATA_PROVIDER", "polygon")
    monkeypatch.setenv("POLYGON_API_KEY", "polygon-key")
    assert isinstance(get_provider(), PolygonProvider)
    assert isinstance(get_provider("yahoo"), YahooProvider)


def test_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "tiingo-key")
    provider = get_provider("Tiingo")
    assert isinstance(provider, TiingoProvider)
    assert provider.name == "tiingo"


def test_unknown_provider_raises():
    with pytest.raises(DataProviderConfigError, match="Unknown data provider"):
        get_provider("bloomberg")


def test_tiingo_and_polygon_require_keys():
    with pytest.raises(MissingApiKeyError, match="TIINGO_API_KEY"):
        get_provider("tiingo")
    with pytest.raises(MissingApiKeyError, match="POLYGON_API_KEY"):
        get_provider("polygon")


def test_blank_api_key_is_missing(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "   ")
    with pytest.raises(MissingApiKeyError, match="TIINGO_API_KEY"):
        TiingoProvider()


def test_invalid_cache_flag_raises(monkeypatch):
    monkeypatch.setenv("BACKTEST_DATA_CACHE", "maybe")
    with pytest.raises(DataProviderConfigError, match="BACKTEST_DATA_CACHE"):
        get_provider("yahoo")


def test_cache_wraps_when_enabled(monkeypatch, tmp_path):
    monkeypatch.delenv("BACKTEST_DATA_CACHE", raising=False)
    monkeypatch.setenv("BACKTEST_DATA_CACHE_DIR", str(tmp_path))
    provider = get_provider("yahoo")
    assert isinstance(provider, CachingProvider)
    assert provider.name == "yahoo"
    assert provider.cache_dir == tmp_path


def test_yahoo_normalizes_and_uses_auto_adjust(monkeypatch):
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"], tz="America/New_York")
    history = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 99.0],
            "High": [10.5, 11.5, 99.5],
            "Low": [9.5, 10.5, 98.5],
            "Close": [10.2, 11.2, 99.2],
            "Volume": [100, 110, 120],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=index,
    )
    seen = {}

    class _Ticker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def history(self, start, end, auto_adjust):
            seen["start"] = start
            seen["end"] = end
            seen["auto_adjust"] = auto_adjust
            return history

    monkeypatch.setattr(yf, "Ticker", _Ticker)
    frame = YahooProvider().fetch_ohlcv(" AAPL ", "2020-01-01", "2020-01-06")

    assert seen == {
        "symbol": "AAPL",
        "start": "2020-01-01",
        "end": "2020-01-06",
        "auto_adjust": True,
    }
    assert frame.index.tz is None
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03"]
    assert frame.loc[pd.Timestamp("2020-01-03"), "Close"] == 11.2


def test_yahoo_empty_window_skips_the_client(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("yfinance should not be called")

    monkeypatch.setattr(yf, "Ticker", _boom)
    frame = YahooProvider().fetch_ohlcv("AAPL", "2020-02-01", "2020-01-01")
    assert frame.empty
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_tiingo_uses_adjusted_prices_and_keeps_the_key_out_of_the_url(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "super-secret")
    payload = [
        {
            "date": "2020-01-02T00:00:00.000Z",
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "volume": 100,
            "adjOpen": 8,
            "adjHigh": 9,
            "adjLow": 7,
            "adjClose": 8.5,
            "adjVolume": 80,
        },
        {
            "date": "2020-01-06T00:00:00.000Z",
            "open": 12,
            "high": 13,
            "low": 11,
            "close": 12.5,
            "volume": 100,
            "adjOpen": 12,
            "adjHigh": 13,
            "adjLow": 11,
            "adjClose": 12.5,
            "adjVolume": 100,
        },
    ]
    seen = _patch_http(monkeypatch, payload)
    frame = TiingoProvider().fetch_ohlcv("AAPL", "2020-01-01", "2020-01-06")

    assert seen["headers"]["Authorization"] == "Token super-secret"
    assert "super-secret" not in seen["url"]
    assert "super-secret" not in repr(seen["params"])
    assert seen["params"]["startDate"] == "2020-01-01"
    assert seen["timeout"] == 30
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2020-01-02"]
    assert frame.iloc[0]["Close"] == 8.5
    assert frame.iloc[0]["Volume"] == 80


def test_tiingo_404_is_empty(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "k")
    _patch_http(monkeypatch, {"detail": "Not found."}, status=404)
    frame = TiingoProvider().fetch_ohlcv("NOPE", "2020-01-01", "2020-02-01")
    assert frame.empty


def test_tiingo_rejected_key_does_not_echo_the_secret(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "super-secret")
    _patch_http(monkeypatch, {"detail": "Invalid token super-secret"}, status=401)
    with pytest.raises(ProviderError, match="rejected the API key") as exc:
        TiingoProvider().fetch_ohlcv("AAPL", "2020-01-01", "2020-02-01")
    assert "super-secret" not in str(exc.value)


def test_tiingo_empty_symbol_and_empty_window(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "k")

    def _boom(*args, **kwargs):
        raise AssertionError("http should not be called")

    monkeypatch.setattr("providers.base.requests.get", _boom)
    assert TiingoProvider().fetch_ohlcv("AAPL", "2020-02-01", "2020-01-01").empty
    with pytest.raises(ProviderError, match="symbol is empty"):
        TiingoProvider().fetch_ohlcv("   ", "2020-01-01", "2020-02-01")


def test_polygon_session_dates_adjustment_and_auth(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "poly-secret")
    # Published sample: 1577941200000 ms is 2020-01-02 00:00 America/New_York.
    winter = 1577941200000
    summer = int(pd.Timestamp("2020-07-02", tz="America/New_York").timestamp() * 1000)
    end_bar = int(pd.Timestamp("2020-07-03", tz="America/New_York").timestamp() * 1000)
    payload = {
        "status": "OK",
        "adjusted": True,
        "resultsCount": 3,
        "results": [
            {"o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10, "t": winter},
            {"o": 3, "h": 4, "l": 2.5, "c": 3.5, "v": 20, "t": summer},
            {"o": 9, "h": 9, "l": 9, "c": 9, "v": 1, "t": end_bar},
        ],
    }
    seen = _patch_http(monkeypatch, payload)
    frame = PolygonProvider().fetch_ohlcv("AAPL", "2020-01-01", "2020-07-03")

    assert seen["headers"]["Authorization"] == "Bearer poly-secret"
    assert "apiKey" not in seen["params"]
    assert "poly-secret" not in seen["url"]
    assert seen["params"]["adjusted"] == "true"
    assert seen["params"]["sort"] == "asc"
    assert seen["params"]["limit"] == 50000
    assert "/v2/aggs/ticker/AAPL/range/1/day/2020-01-01/2020-07-03" in seen["url"]
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-07-02"]
    assert frame.index.tz is None
    assert frame.loc[pd.Timestamp("2020-07-02"), "Close"] == 3.5


def test_polygon_truncated_page_is_an_error(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    payload = {
        "status": "OK",
        "resultsCount": 1,
        "next_url": "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/more",
        "results": [{"o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "t": 1577941200000}],
    }
    _patch_http(monkeypatch, payload)
    with pytest.raises(ProviderError, match="50,000"):
        PolygonProvider().fetch_ohlcv("AAPL", "1900-01-01", "2020-02-01")


class _Counting:
    def __init__(self, name, frame=None, empty=False):
        self.name = name
        self.calls = 0
        self.frame = frame if frame is not None else empty_ohlcv()
        self.empty = empty

    def fetch_ohlcv(self, symbol, start, end):
        self.calls += 1
        if self.empty:
            return empty_ohlcv()
        return self.frame.copy()


def _one_bar(close):
    index = pd.to_datetime(["2020-01-02"])
    return pd.DataFrame(
        {
            "Open": [close],
            "High": [close + 1],
            "Low": [close - 1],
            "Close": [close],
            "Volume": [1000.0],
        },
        index=index,
    )


def test_cache_hits_and_skips_empty(tmp_path):
    inner = _Counting("yahoo", frame=_one_bar(10.0))
    cached = CachingProvider(inner, tmp_path)
    first = cached.fetch_ohlcv("AAPL", "2020-01-01", "2020-02-01")
    second = cached.fetch_ohlcv("AAPL", "2020-01-01", "2020-02-01")
    assert inner.calls == 1
    pd.testing.assert_frame_equal(first, second)

    empty = _Counting("yahoo", empty=True)
    cached_empty = CachingProvider(empty, tmp_path)
    assert cached_empty.fetch_ohlcv("MSFT", "2020-01-01", "2020-02-01").empty
    assert cached_empty.fetch_ohlcv("MSFT", "2020-01-01", "2020-02-01").empty
    assert empty.calls == 2


def test_cache_separates_providers_and_refetches_corrupt_files(tmp_path):
    yahoo = _Counting("yahoo", frame=_one_bar(1.0))
    tiingo = _Counting("tiingo", frame=_one_bar(2.0))
    yahoo_cache = CachingProvider(yahoo, tmp_path)
    tiingo_cache = CachingProvider(tiingo, tmp_path)

    assert yahoo_cache.fetch_ohlcv("AAPL", "2020-01-01", "2020-02-01")["Close"].iloc[0] == 1.0
    assert tiingo_cache.fetch_ohlcv("AAPL", "2020-01-01", "2020-02-01")["Close"].iloc[0] == 2.0

    path = yahoo_cache._path("AAPL", "2020-01-01", "2020-02-01")
    path.write_bytes(b"not parquet")
    again = yahoo_cache.fetch_ohlcv("AAPL", "2020-01-01", "2020-02-01")
    assert yahoo.calls == 2
    assert again["Close"].iloc[0] == 1.0
    # The replacement file is readable, so the next call is a hit.
    yahoo_cache.fetch_ohlcv("AAPL", "2020-01-01", "2020-02-01")
    assert yahoo.calls == 2
    assert tiingo.calls == 1


def test_fetch_data_delegates_swallows_request_errors_and_raises_config(monkeypatch, capsys):
    from data_fetcher import fetch_data

    class _Ok:
        name = "yahoo"

        def fetch_ohlcv(self, symbol, start, end):
            return _one_bar(4.0)

    monkeypatch.setattr("data_fetcher.get_provider", lambda name=None: _Ok())
    frame = fetch_data("AAPL", "2020-01-01", "2020-02-01")
    assert frame["Close"].iloc[0] == 4.0

    class _Boom:
        name = "polygon"

        def fetch_ohlcv(self, symbol, start, end):
            raise ProviderError("nope")

    monkeypatch.setattr("data_fetcher.get_provider", lambda name=None: _Boom())
    empty = fetch_data("AAPL", "2020-01-01", "2020-02-01")
    assert empty.empty
    assert "Error fetching AAPL from polygon: nope" in capsys.readouterr().out

    def _missing(name=None):
        raise MissingApiKeyError("Polygon is selected but POLYGON_API_KEY is not set.")

    monkeypatch.setattr("data_fetcher.get_provider", _missing)
    with pytest.raises(MissingApiKeyError, match="POLYGON_API_KEY"):
        fetch_data("AAPL", "2020-01-01", "2020-02-01")


def test_price_factors_use_the_fetcher(monkeypatch):
    import numpy as np

    from factor_engine import _fetch_price_factors_cached, fetch_price_factors

    _fetch_price_factors_cached.clear()
    index = pd.bdate_range("2020-01-01", periods=25)
    close = pd.Series(100.0, index=index)
    close.iloc[-1] = 125.0
    frame = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}
    )
    seen = {}

    def fake_fetch(symbol, start, end, provider=None):
        seen["symbol"] = symbol
        seen["start"] = start
        seen["end"] = end
        seen["provider"] = provider
        return frame

    monkeypatch.setattr("factor_engine.fetch_data", fake_fetch)
    monkeypatch.setenv("BACKTEST_DATA_PROVIDER", "Tiingo")

    def _yf_should_not_run(*args, **kwargs):
        raise AssertionError("price factors must not call yfinance")

    monkeypatch.setattr(yf, "Ticker", _yf_should_not_run)
    result = fetch_price_factors(("AAPL",), "2020-01-01", "2021-01-01")

    assert seen == {
        "symbol": "AAPL",
        "start": "2020-01-01",
        "end": "2021-01-01",
        "provider": "tiingo",
    }
    assert result.loc[0, "Momentum"] == pytest.approx(0.25)
    assert result.loc[0, "Volatility"] > 0
    assert np.isfinite(result.loc[0, "Volatility"])


def test_price_factors_propagate_config_errors_and_fill_gaps(monkeypatch):
    from factor_engine import _fetch_price_factors_cached, fetch_price_factors

    _fetch_price_factors_cached.clear()

    def _missing(symbol, start, end, provider=None):
        raise MissingApiKeyError("Tiingo is selected but TIINGO_API_KEY is not set.")

    monkeypatch.setattr("factor_engine.fetch_data", _missing)
    with pytest.raises(MissingApiKeyError):
        fetch_price_factors(("AAPL",), "2020-01-01", "2020-06-01")

    _fetch_price_factors_cached.clear()
    monkeypatch.setattr(
        "factor_engine.fetch_data",
        lambda symbol, start, end, provider=None: empty_ohlcv(),
    )
    result = fetch_price_factors(("ZZZ",), "2019-01-01", "2019-06-01")
    assert result.loc[0, "symbol"] == "ZZZ"
    assert pd.isna(result.loc[0, "Momentum"])
    assert pd.isna(result.loc[0, "Volatility"])
