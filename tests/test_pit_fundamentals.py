"""Point-in-time fundamentals. No network and no API keys."""

import math
from datetime import date

import pandas as pd
import pytest

from factor_engine import price_factor_bounds
from fundamentals_pit import (
    compute_ratios,
    fetch_fundamentals_pit,
    reporting_lag_days,
)
from providers.base import DataProviderConfigError
from screener import apply_factor_score


def _is_nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _quarters():
    """Four quarters of 2019. Q4 is unseen until 60 days after 2019-12-31."""
    return pd.DataFrame(
        [
            {"period_end": "2019-03-31", "period_type": "Q", "net_income": 10.0, "equity": 100.0},
            {"period_end": "2019-06-30", "period_type": "Q", "net_income": 10.0, "equity": 100.0},
            {"period_end": "2019-09-30", "period_type": "Q", "net_income": 10.0, "equity": 100.0},
            {"period_end": "2019-12-31", "period_type": "Q", "net_income": 10.0, "equity": 100.0},
        ]
    )


def test_statement_stays_hidden_until_the_reporting_lag():
    panel = _quarters()
    # 2019-12-31 + 60 days is 2020-02-29.
    before = compute_ratios(panel, "2020-02-28", price=None, lag_days=60)
    on_lag = compute_ratios(panel, "2020-02-29", price=None, lag_days=60)

    assert _is_nan(before["ROE"])
    assert on_lag["ROE"] == pytest.approx(0.4)


def test_restatement_does_not_replace_the_figure_already_filed():
    base = _quarters()
    base["filing_date"] = ["2019-05-01", "2019-08-01", "2019-11-01", "2020-02-01"]
    restated = pd.DataFrame(
        [
            {
                "period_end": "2019-12-31",
                "period_type": "Q",
                "net_income": 20.0,
                "equity": 100.0,
                "filing_date": "2020-05-01",
                "revision": 1,
            },
            {
                "period_end": "2019-12-31",
                "period_type": "Q",
                "net_income": 40.0,
                "equity": 100.0,
                "filing_date": "2020-08-01",
                "revision": 2,
            },
        ]
    )
    panel = pd.concat([base, restated], ignore_index=True)

    original = compute_ratios(panel, "2020-03-01", price=None, lag_days=60)
    middle = compute_ratios(panel, "2020-06-01", price=None, lag_days=60)
    latest = compute_ratios(panel, "2020-09-01", price=None, lag_days=60)

    assert original["ROE"] == pytest.approx(0.4)
    assert middle["ROE"] == pytest.approx(0.5)
    assert latest["ROE"] == pytest.approx(0.7)


def test_earnings_date_is_used_when_it_is_earlier_than_the_lag():
    panel = _quarters()
    panel["filing_date"] = ["2019-05-01", "2019-08-01", "2019-11-01", None]
    panel["earnings_date"] = [None, None, None, "2020-01-15"]

    hidden = compute_ratios(panel, "2020-01-14", price=None, lag_days=60)
    visible = compute_ratios(panel, "2020-01-15", price=None, lag_days=60)

    assert _is_nan(hidden["ROE"])
    assert visible["ROE"] == pytest.approx(0.4)


def test_late_earnings_date_is_not_pulled_forward_by_the_lag():
    panel = _quarters()
    panel["earnings_date"] = [None, None, None, "2020-04-15"]
    panel["filing_date"] = ["2019-05-01", "2019-08-01", "2019-11-01", None]

    # Lag alone would have released Q4 on 2020-02-29.
    too_early = compute_ratios(panel, "2020-03-01", price=None, lag_days=60)
    on_date = compute_ratios(panel, "2020-04-15", price=None, lag_days=60)

    assert _is_nan(too_early["ROE"])
    assert on_date["ROE"] == pytest.approx(0.4)


def test_earnings_announcements_cover_pe_when_statements_are_missing():
    panel = pd.DataFrame(
        [
            {"period_end": "2019-02-01", "period_type": "E", "eps": 1.0, "earnings_date": "2019-02-01"},
            {"period_end": "2019-05-01", "period_type": "E", "eps": 1.0, "earnings_date": "2019-05-01"},
            {"period_end": "2019-08-01", "period_type": "E", "eps": 1.0, "earnings_date": "2019-08-01"},
            {"period_end": "2019-11-01", "period_type": "E", "eps": 1.0, "earnings_date": "2019-11-01"},
        ]
    )
    before = compute_ratios(panel, "2019-10-31", price=10.0, lag_days=60)
    after = compute_ratios(panel, "2019-11-01", price=10.0, lag_days=60)

    assert _is_nan(before["PE"])
    assert _is_nan(after["ROE"])
    assert after["PE"] == pytest.approx(2.5)


def test_published_quarters_replace_the_earnings_series():
    earnings = pd.DataFrame(
        [
            {"period_end": "2019-02-01", "period_type": "E", "eps": 1.0, "earnings_date": "2019-02-01"},
            {"period_end": "2019-05-01", "period_type": "E", "eps": 1.0, "earnings_date": "2019-05-01"},
            {"period_end": "2019-08-01", "period_type": "E", "eps": 1.0, "earnings_date": "2019-08-01"},
            {"period_end": "2019-11-01", "period_type": "E", "eps": 1.0, "earnings_date": "2019-11-01"},
        ]
    )
    quarters = _quarters()
    quarters["eps"] = 2.0
    panel = pd.concat([quarters, earnings], ignore_index=True)

    # Q3 2019 is still inside the lag, so PE comes from the four announcements.
    early = compute_ratios(panel, "2019-11-01", price=10.0, lag_days=60)
    # All four 2019 quarters are public. Trailing EPS is 8, not the announcement sum of 4.
    later = compute_ratios(panel, "2020-02-29", price=16.0, lag_days=60)

    assert early["PE"] == pytest.approx(2.5)
    assert later["PE"] == pytest.approx(2.0)


def test_annual_statement_is_used_when_quarters_are_not_available_yet():
    panel = pd.DataFrame(
        [
            {
                "period_end": "2018-12-31",
                "period_type": "A",
                "net_income": 100.0,
                "eps": 5.0,
                "revenue": 400.0,
                "equity": 200.0,
                "shares": 20.0,
            },
            {
                "period_end": "2019-12-31",
                "period_type": "A",
                "net_income": 150.0,
                "eps": 7.0,
                "revenue": 500.0,
                "equity": 250.0,
                "shares": 20.0,
            },
        ]
    )
    before = compute_ratios(panel, "2020-02-28", price=70.0, lag_days=60)
    after = compute_ratios(panel, "2020-02-29", price=70.0, lag_days=60)

    assert before["PE"] == pytest.approx(14.0)
    assert after["PE"] == pytest.approx(10.0)
    assert after["RevenueGrowth"] == pytest.approx(0.25)


def test_reporting_lag_env_and_override(monkeypatch):
    assert reporting_lag_days() == 60
    monkeypatch.setenv("BACKTEST_FUNDAMENTALS_LAG_DAYS", "45")
    assert reporting_lag_days() == 45
    assert reporting_lag_days(10) == 10
    monkeypatch.setenv("BACKTEST_FUNDAMENTALS_LAG_DAYS", "soon")
    with pytest.raises(ValueError, match="BACKTEST_FUNDAMENTALS_LAG_DAYS"):
        reporting_lag_days()


class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def _point(day, raw):
    return {"asOfDate": day, "periodType": "3M", "reportedValue": {"raw": raw}}


def _yahoo_payload():
    days = ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]

    def series(key, values):
        return {"meta": {"type": [key]}, key: [_point(day, value) for day, value in zip(days, values)]}

    return {
        "timeseries": {
            "result": [
                series("quarterlyTotalRevenue", [500.0, 500.0, 500.0, 500.0]),
                series("quarterlyNetIncome", [100.0, 100.0, 100.0, 100.0]),
                series("quarterlyDilutedEPS", [2.0, 2.0, 2.0, 2.0]),
                series("quarterlyDilutedAverageShares", [50.0, 50.0, 50.0, 50.0]),
                series("quarterlyOrdinarySharesNumber", [40.0, 40.0, 40.0, 40.0]),
                series("quarterlyStockholdersEquity", [400.0, 400.0, 400.0, 400.0]),
            ]
        }
    }


def _price_frame(close, day):
    idx = pd.DatetimeIndex([pd.Timestamp(day)])
    value = pd.Series([close], index=idx, dtype=float)
    return pd.DataFrame(
        {"Open": value, "High": value, "Low": value, "Close": value, "Volume": 1.0},
        index=idx,
    )


class _NoEarnings:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_earnings_dates(self, limit=100):
        return None


def _patch_yahoo(monkeypatch, payload):
    seen = []

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append(url)
        assert "financialmodelingprep.com" not in url
        return _Response(200, payload)

    monkeypatch.setattr("fundamentals_pit.requests.get", fake_get)
    monkeypatch.setattr("fundamentals_pit.yf.Ticker", _NoEarnings)
    monkeypatch.setattr(
        "fundamentals_pit.fetch_data",
        lambda symbol, start, end, provider=None: _price_frame(40.0, "2021-03-01"),
    )
    return seen


def test_yahoo_path_uses_lag_and_caches_the_panel(monkeypatch, tmp_path):
    monkeypatch.setenv("BACKTEST_DATA_CACHE", "1")
    monkeypatch.setenv("BACKTEST_DATA_CACHE_DIR", str(tmp_path))
    seen = _patch_yahoo(monkeypatch, _yahoo_payload())

    early = fetch_fundamentals_pit(["AAPL"], "2021-02-28", lag_days=60)
    late = fetch_fundamentals_pit(["AAPL"], "2021-03-01", lag_days=60)

    assert len(seen) == 1
    assert _is_nan(early.loc[0, "PE"])
    assert late.loc[0, "PE"] == pytest.approx(5.0)
    # 40 * ordinary shares (40), not the diluted average (50).
    assert late.loc[0, "PB"] == pytest.approx(4.0)
    assert late.loc[0, "ROE"] == pytest.approx(1.0)
    assert late.loc[0, "MarketCap"] == pytest.approx(1600.0)
    assert list(tmp_path.rglob("*.parquet"))


def test_blank_fmp_key_stays_on_yahoo(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "   ")
    seen = _patch_yahoo(monkeypatch, _yahoo_payload())
    frame = fetch_fundamentals_pit(["AAPL"], "2021-03-01", lag_days=60)
    assert frame.loc[0, "PE"] == pytest.approx(5.0)
    assert seen and all("yahoo" in url for url in seen)


def test_fmp_filing_date_hides_the_quarter_until_it_is_filed(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    hosts = []

    def fake_get(url, params=None, headers=None, timeout=None):
        hosts.append(url)
        assert params["apikey"] == "test-key"
        if url.endswith("/income-statement"):
            rows = []
            for day, filed in (
                ("2020-03-31", "2020-05-01"),
                ("2020-06-30", "2020-08-01"),
                ("2020-09-30", "2020-11-01"),
                ("2020-12-31", "2021-04-01"),
            ):
                rows.append(
                    {
                        "date": day,
                        "period": "Q1",
                        "filingDate": filed,
                        "revenue": 500.0,
                        "netIncome": 100.0,
                        "epsdiluted": 2.0,
                        "weightedAverageShsOutDil": 50.0,
                    }
                )
            return _Response(200, rows)
        if url.endswith("/balance-sheet-statement"):
            return _Response(
                200,
                [
                    {"date": "2020-03-31", "period": "Q1", "filingDate": "2020-05-01", "totalStockholdersEquity": 400.0},
                    {"date": "2020-06-30", "period": "Q1", "filingDate": "2020-08-01", "totalStockholdersEquity": 400.0},
                    {"date": "2020-09-30", "period": "Q1", "filingDate": "2020-11-01", "totalStockholdersEquity": 400.0},
                    {"date": "2020-12-31", "period": "Q1", "filingDate": "2021-04-30", "totalStockholdersEquity": 400.0},
                ],
            )
        raise AssertionError(url)

    monkeypatch.setattr("fundamentals_pit.requests.get", fake_get)
    monkeypatch.setattr("fundamentals_pit.yf.Ticker", _NoEarnings)
    monkeypatch.setattr(
        "fundamentals_pit.fetch_data",
        lambda symbol, start, end, provider=None: _price_frame(40.0, "2021-04-30"),
    )

    # Lag would have released 2020-12-31 on 2021-03-01. The balance sheet
    # is not public until 2021-04-30, which is the later filing.
    early = fetch_fundamentals_pit(["AAPL"], "2021-04-15", lag_days=60)
    late = fetch_fundamentals_pit(["AAPL"], "2021-04-30", lag_days=60)

    assert hosts and all("financialmodelingprep.com" in url for url in hosts)
    assert _is_nan(early.loc[0, "ROE"])
    assert late.loc[0, "PE"] == pytest.approx(5.0)
    assert late.loc[0, "ROE"] == pytest.approx(1.0)


def test_fmp_failure_falls_back_to_yahoo(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")

    def fake_get(url, params=None, headers=None, timeout=None):
        if "financialmodelingprep.com" in url:
            return _Response(500, {"Error Message": "nope"})
        return _Response(200, _yahoo_payload())

    monkeypatch.setattr("fundamentals_pit.requests.get", fake_get)
    monkeypatch.setattr("fundamentals_pit.yf.Ticker", _NoEarnings)
    monkeypatch.setattr(
        "fundamentals_pit.fetch_data",
        lambda symbol, start, end, provider=None: _price_frame(40.0, "2021-03-01"),
    )

    frame = fetch_fundamentals_pit(["AAPL"], "2021-03-01", lag_days=60)
    assert frame.loc[0, "PE"] == pytest.approx(5.0)


def test_thin_symbol_is_nan_and_does_not_raise(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        return _Response(200, {"timeseries": {"result": []}})

    monkeypatch.setattr("fundamentals_pit.requests.get", fake_get)
    monkeypatch.setattr("fundamentals_pit.yf.Ticker", _NoEarnings)
    monkeypatch.setattr(
        "fundamentals_pit.fetch_data",
        lambda symbol, start, end, provider=None: _price_frame(10.0, "2020-01-02"),
    )

    frame = fetch_fundamentals_pit(["2330.TW"], "2020-01-02", lag_days=60)
    assert frame.loc[0, "symbol"] == "2330.TW"
    for col in ("PE", "PB", "ROE", "RevenueGrowth", "EarningsGrowth", "MarketCap"):
        assert _is_nan(frame.loc[0, col])


def test_price_provider_misconfig_is_not_swallowed(monkeypatch):
    _patch_yahoo(monkeypatch, _yahoo_payload())

    def boom(*args, **kwargs):
        raise DataProviderConfigError("missing key")

    monkeypatch.setattr("fundamentals_pit.fetch_data", boom)
    with pytest.raises(DataProviderConfigError, match="missing key"):
        fetch_fundamentals_pit(["AAPL"], "2021-03-01", lag_days=60)


def test_price_factors_end_on_as_of_and_ignore_the_later_sample(monkeypatch):
    as_of = pd.Timestamp("2020-06-15")
    seen = []

    def fake_fetch(symbol, start, end, provider=None):
        seen.append((symbol, start, end))
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        idx = pd.bdate_range(start_ts, end_ts - pd.Timedelta(days=1))
        close = pd.Series(100.0, index=idx)
        close = close.where(close.index <= as_of, 500.0)
        return pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}
        )

    monkeypatch.setattr("factor_engine.fetch_data", fake_fetch)
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "PE": [10.0, 12.0],
            "PB": [1.0, 1.0],
            "ROE": [0.1, 0.2],
            "RevenueGrowth": [0.1, 0.1],
            "EarningsGrowth": [0.1, 0.1],
            "MarketCap": [1.0e9, 1.0e9],
        }
    )
    scored = apply_factor_score(
        fundamentals,
        as_of=date(2020, 6, 15),
        active_factors={"momentum": 1.0},
        top_n=2,
    )

    assert seen
    assert {item[1:] for item in seen} == {("2019-06-15", "2020-06-16")}
    assert scored["Momentum"].max() == pytest.approx(0.0)
    assert price_factor_bounds("2020-06-15") == ("2019-06-15", "2020-06-16")


def test_fundamental_score_does_not_request_prices(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("price history should not be loaded")

    monkeypatch.setattr("factor_engine.fetch_data", boom)
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "ROE": [0.1, 0.4],
            "PE": [10.0, 10.0],
        }
    )
    scored = apply_factor_score(
        fundamentals,
        as_of="2020-01-02",
        active_factors={"roe": 1.0},
        top_n=1,
    )
    assert scored.iloc[0]["symbol"] == "BBB"
