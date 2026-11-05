"""Point-in-time fundamentals. Synthetic statements only; no live HTTP."""

import math

import pandas as pd
import pytest

from fundamentals.fmp import statements_from_fmp
from fundamentals.pit import (
    fetch_pit_fundamentals,
    ratios_as_of,
    resolve_lag_days,
)
from fundamentals.snapshot import fetch_fundamentals
from fundamentals.statements import (
    empty_statements,
    match_earnings_dates,
    reporting_date,
)
from fundamentals.yahoo_statements import statements_from_yahoo_frames
from providers.base import DataProviderConfigError, MissingApiKeyError, ProviderError


def _quarter(period, revenue, net_income, equity, shares=10.0, available=None, symbol="AAA"):
    return {
        "symbol": symbol,
        "period_end": period,
        "available_at": available if available is not None else reporting_date(period, 60),
        "frequency": "Q",
        "revenue": revenue,
        "net_income": net_income,
        "diluted_shares": shares,
        "equity": equity,
        "shares_outstanding": shares,
    }


def _history():
    specs = [
        ("2018-03-31", 80, 8, 170),
        ("2018-06-30", 85, 9, 180),
        ("2018-09-30", 90, 9, 190),
        ("2018-12-31", 95, 10, 195),
        ("2019-03-31", 100, 10, 200),
        ("2019-06-30", 110, 12, 210),
        ("2019-09-30", 120, 14, 220),
        ("2019-12-31", 130, 16, 230),
        ("2020-03-31", 140, 18, 240),
    ]
    return pd.DataFrame([_quarter(*spec) for spec in specs])


def _prices():
    return pd.Series(
        [50.0, 50.0, 50.0, 999.0],
        index=pd.to_datetime(["2020-05-29", "2020-05-30", "2020-07-31", "2020-08-03"]),
    )


def test_reporting_lag_hides_a_period_until_the_lag_date():
    assert reporting_date("2019-12-31", 60) == pd.Timestamp("2020-02-29")
    assert reporting_date("2020-03-31", 60, "2020-03-01") == pd.Timestamp("2020-05-30")
    assert reporting_date("2020-03-31", 60, "2020-04-28") == pd.Timestamp("2020-04-28")

    income = pd.DataFrame(
        [[500.0], [50.0], [10.0]],
        index=["Total Revenue", "Net Income", "Diluted Average Shares"],
        columns=[pd.Timestamp("2019-12-31")],
    )
    balance = pd.DataFrame(
        [[250.0], [10.0]],
        index=["Stockholders Equity", "Ordinary Shares Number"],
        columns=[pd.Timestamp("2019-12-31")],
    )
    statements = statements_from_yahoo_frames(
        "AAA",
        income_annual=income,
        balance_annual=balance,
        lag_days=60,
    )
    prices = pd.Series([40.0, 40.0], index=pd.to_datetime(["2020-02-28", "2020-02-29"]))

    hidden = ratios_as_of(statements, "2020-02-28", symbol="AAA", prices=prices)
    visible = ratios_as_of(statements, "2020-02-29", symbol="AAA", prices=prices)

    assert math.isnan(hidden["PE"])
    assert hidden["statement_period_end"] is None
    assert visible["PE"] == pytest.approx(8.0)
    assert visible["PB"] == pytest.approx(1.6)
    assert visible["ROE"] == pytest.approx(0.2)
    assert visible["MarketCap"] == pytest.approx(400.0)
    assert math.isnan(visible["RevenueGrowth"])
    assert visible["statement_period_end"] == "2019-12-31"
    assert visible["statement_available_at"] == "2020-02-29"


def test_new_quarter_and_later_price_do_not_leak_early():
    statements = _history()
    prices = _prices()
    before = ratios_as_of(statements, "2020-05-29", symbol="AAA", prices=prices)
    after = ratios_as_of(statements, "2020-05-30", symbol="AAA", prices=prices)

    assert before["statement_period_end"] == "2019-12-31"
    assert before["PE"] == pytest.approx(50 / 5.2)
    assert before["ROE"] == pytest.approx(52 / 230)
    assert before["RevenueGrowth"] == pytest.approx(460 / 350 - 1)
    assert before["EarningsGrowth"] == pytest.approx(52 / 36 - 1)
    assert before["MarketCap"] == pytest.approx(500.0)

    assert after["statement_period_end"] == "2020-03-31"
    assert after["statement_available_at"] == "2020-05-30"
    assert after["PE"] == pytest.approx(50 / 6)
    assert after["ROE"] == pytest.approx(0.25)
    assert after["PB"] == pytest.approx(500 / 240)
    assert after["RevenueGrowth"] == pytest.approx(500 / 370 - 1)
    assert after["EarningsGrowth"] == pytest.approx(60 / 38 - 1)
    # 999 is after this decision date.
    assert after["MarketCap"] == pytest.approx(500.0)


def test_restatement_is_invisible_until_its_own_date():
    statements = _history()
    restated = _quarter("2019-12-31", 200, 40, 230, available="2020-08-01")
    panel = pd.concat([statements, pd.DataFrame([restated])], ignore_index=True)
    prices = _prices()

    before = ratios_as_of(panel, "2020-07-31", symbol="AAA", prices=prices)
    after = ratios_as_of(panel, "2020-08-01", symbol="AAA", prices=prices)

    assert before["PE"] == pytest.approx(50 / 6)
    assert before["RevenueGrowth"] == pytest.approx(500 / 370 - 1)
    assert after["PE"] == pytest.approx(50 / 8.4)
    assert after["RevenueGrowth"] == pytest.approx(570 / 370 - 1)
    assert after["MarketCap"] == pytest.approx(500.0)


def test_earnings_date_beats_the_fixed_lag_and_is_not_shared_with_the_prior_quarter():
    mapping = match_earnings_dates(
        ["2020-03-31", "2020-06-30"],
        ["2020-04-30", "2020-07-15"],
    )
    assert mapping[pd.Timestamp("2020-03-31")] == pd.Timestamp("2020-04-30")
    assert mapping[pd.Timestamp("2020-06-30")] == pd.Timestamp("2020-07-15")

    income = pd.DataFrame(
        [[140.0, 130.0], [18.0, 16.0], [10.0, 10.0]],
        index=["Total Revenue", "Net Income", "Diluted Average Shares"],
        columns=[pd.Timestamp("2020-03-31"), pd.Timestamp("2019-12-31")],
    )
    earnings = pd.DataFrame(
        {"Reported EPS": [1.6, 1.8, None]},
        index=pd.to_datetime(["2020-02-20", "2020-08-15", "2099-01-15"]),
    )
    # 2020-02-20 matches 2019-12-31. 2020-08-15 is outside 120 days of 2020-03-31.
    statements = statements_from_yahoo_frames(
        "AAA",
        income_quarterly=income,
        earnings_dates=earnings,
        lag_days=60,
    )
    by_period = statements.set_index("period_end")["available_at"]
    assert by_period.loc[pd.Timestamp("2019-12-31")] == pd.Timestamp("2020-02-20")
    assert by_period.loc[pd.Timestamp("2020-03-31")] == pd.Timestamp("2020-05-30")


def test_gap_in_quarters_falls_back_to_annual():
    quarters = pd.DataFrame(
        [
            _quarter("2019-03-31", 100, 10, 200),
            _quarter("2019-06-30", 110, 12, 210),
            _quarter("2019-12-31", 130, 16, 230),
            _quarter("2020-03-31", 140, 99, 240, available="2020-05-30"),
        ]
    )
    annuals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "period_end": "2018-12-31",
                "available_at": "2019-03-01",
                "frequency": "A",
                "revenue": 400,
                "net_income": 40,
                "diluted_shares": 10,
                "equity": 200,
                "shares_outstanding": 10,
            },
            {
                "symbol": "AAA",
                "period_end": "2019-12-31",
                "available_at": "2020-03-01",
                "frequency": "A",
                "revenue": 500,
                "net_income": 50,
                "diluted_shares": 10,
                "equity": 250,
                "shares_outstanding": 10,
            },
        ]
    )
    panel = pd.concat([quarters, annuals], ignore_index=True)
    prices = pd.Series([80.0], index=pd.to_datetime(["2020-06-01"]))
    row = ratios_as_of(panel, "2020-06-01", symbol="AAA", prices=prices)
    # A gapped trail would have used the 99 of 2020-Q1 and a much lower PE.
    assert row["PE"] == pytest.approx(16.0)
    assert row["RevenueGrowth"] == pytest.approx(0.25)


def test_fmp_accepted_date_is_used_and_a_later_filing_stays_hidden():
    def item(period, accepted, revenue, net_income, equity):
        income = {
            "date": period,
            "acceptedDate": f"{accepted} 18:04:00",
            "filingDate": "2099-01-01",
            "revenue": revenue,
            "netIncome": net_income,
            "weightedAverageShsOutDil": 10,
        }
        balance = {
            "date": period,
            "acceptedDate": f"{accepted} 18:04:00",
            "totalStockholdersEquity": equity,
            "commonStockSharesOutstanding": 10,
        }
        return income, balance

    specs = [
        ("2019-03-31", "2019-05-15", 100, 10, 200),
        ("2019-06-30", "2019-08-14", 110, 12, 210),
        ("2019-09-30", "2019-11-14", 120, 14, 220),
        ("2019-12-31", "2020-02-14", 130, 16, 230),
    ]
    income, balance = zip(*(item(*spec) for spec in specs))
    statements = statements_from_fmp(
        "AAA",
        income_quarterly=list(income),
        balance_quarterly=list(balance),
        lag_days=60,
    )
    q4 = statements[statements["period_end"] == pd.Timestamp("2019-12-31")].iloc[0]
    assert q4["available_at"] == pd.Timestamp("2020-02-14")

    prices = pd.Series([50.0], index=pd.to_datetime(["2020-02-14"]))
    hidden = ratios_as_of(statements, "2020-02-13", symbol="AAA", prices=prices)
    visible = ratios_as_of(statements, "2020-02-14", symbol="AAA", prices=prices)
    assert math.isnan(hidden["PE"])
    assert visible["PE"] == pytest.approx(50 / 5.2)
    assert visible["statement_available_at"] == "2020-02-14"


def test_fmp_http_hides_the_key_and_reads_the_filing_date(monkeypatch):
    seen = []

    class _Response:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append((url, dict(params or {})))
        period = (params or {}).get("period")
        if url.endswith("/income-statement") and period == "quarter":
            return _Response(
                200,
                [
                    {
                        "date": "2020-03-31",
                        "acceptedDate": "2020-04-30 18:04:00",
                        "filingDate": "2020-05-01",
                        "revenue": 140,
                        "netIncome": 18,
                        "weightedAverageShsOutDil": 10,
                    }
                ],
            )
        if url.endswith("/balance-sheet-statement") and period == "quarter":
            return _Response(
                200,
                [
                    {
                        "date": "2020-03-31",
                        "acceptedDate": "2020-04-30 18:04:00",
                        "totalStockholdersEquity": 240,
                    }
                ],
            )
        return _Response(200, [])

    monkeypatch.setattr("providers.base.requests.get", fake_get)
    monkeypatch.setenv("FMP_API_KEY", "secret-key")
    from fundamentals.fmp import load_fmp_statements

    frame = load_fmp_statements("AAA", lag_days=60)
    assert frame.loc[0, "available_at"] == pd.Timestamp("2020-04-30")
    assert frame.loc[0, "net_income"] == 18
    assert {url.rsplit("/", 1)[-1] for url, _params in seen} >= {
        "income-statement",
        "balance-sheet-statement",
    }
    assert all(params.get("apikey") == "secret-key" for _url, params in seen)
    assert all("secret-key" not in url for url, _params in seen)

    def rejected(url, params=None, headers=None, timeout=None):
        response = _Response(401, {})
        response.text = "secret-key"
        return response

    monkeypatch.setattr("providers.base.requests.get", rejected)
    with pytest.raises(ProviderError, match="rejected the API key") as caught:
        load_fmp_statements("AAA", lag_days=60)
    assert "secret-key" not in str(caught.value)


def test_missing_fmp_key_keeps_the_yahoo_path(monkeypatch):
    def _fmp_should_not_run(*args, **kwargs):
        raise AssertionError("FMP should not be called")

    def _yahoo(symbol, lag_days):
        return pd.DataFrame([_quarter("2019-12-31", 100, 10, 200, available="2020-03-01")])

    monkeypatch.setattr("fundamentals.pit.load_fmp_statements", _fmp_should_not_run)
    monkeypatch.setattr("fundamentals.pit.load_yahoo_statements", _yahoo)
    monkeypatch.setattr(
        "fundamentals.pit.fetch_data",
        lambda symbol, start, end, provider=None: pd.DataFrame(
            {"Close": [20.0]}, index=pd.to_datetime(["2020-03-01"])
        ),
    )

    frame = fetch_pit_fundamentals(["aaa"], as_of="2020-03-01")
    assert frame.loc[0, "source"] == "yahoo_pit"
    assert frame.loc[0, "symbol"] == "aaa"

    with pytest.raises(MissingApiKeyError, match="FMP_API_KEY"):
        fetch_pit_fundamentals(["AAA"], as_of="2020-03-01", source="fmp")


def test_auto_fmp_failure_falls_back_without_printing_the_key(monkeypatch, capsys):
    def _boom(symbol, lag_days, api_key=None):
        raise ProviderError("fmp request failed: timeout")

    def _yahoo(symbol, lag_days):
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "period_end": "2019-12-31",
                    "available_at": "2020-03-01",
                    "frequency": "A",
                    "revenue": 500,
                    "net_income": 50,
                    "diluted_shares": 10,
                    "equity": 250,
                    "shares_outstanding": 10,
                }
            ]
        )

    monkeypatch.setenv("FMP_API_KEY", "secret-key")
    monkeypatch.setattr("fundamentals.pit.load_fmp_statements", _boom)
    monkeypatch.setattr("fundamentals.pit.load_yahoo_statements", _yahoo)
    monkeypatch.setattr(
        "fundamentals.pit.fetch_data",
        lambda symbol, start, end, provider=None: pd.DataFrame(
            {"Close": [80.0]}, index=pd.to_datetime(["2020-03-01"])
        ),
    )
    frame = fetch_pit_fundamentals(["AAA"], as_of="2020-03-01")
    assert frame.loc[0, "source"] == "yahoo_pit"
    assert frame.loc[0, "PE"] == pytest.approx(16.0)
    assert "secret-key" not in capsys.readouterr().out


def test_explicit_fmp_auth_failure_does_not_silently_use_yahoo(monkeypatch):
    def _boom(symbol, lag_days, api_key=None):
        raise ProviderError("fmp rejected the API key (HTTP 401).")

    def _yahoo(*args, **kwargs):
        raise AssertionError("Yahoo should not backfill an explicit FMP failure")

    monkeypatch.setenv("FMP_API_KEY", "secret-key")
    monkeypatch.setattr("fundamentals.pit.load_fmp_statements", _boom)
    monkeypatch.setattr("fundamentals.pit.load_yahoo_statements", _yahoo)
    with pytest.raises(ProviderError, match="rejected the API key"):
        fetch_pit_fundamentals(["AAA"], as_of="2020-03-01", source="fmp")


def test_thin_statements_stay_missing(monkeypatch):
    monkeypatch.setattr(
        "fundamentals.pit.load_yahoo_statements",
        lambda symbol, lag_days: empty_statements(),
    )
    monkeypatch.setattr(
        "fundamentals.pit.fetch_data",
        lambda symbol, start, end, provider=None: pd.DataFrame(
            {"Close": [100.0]}, index=pd.to_datetime(["2020-01-02"])
        ),
    )
    frame = fetch_pit_fundamentals(["2330.TW"], as_of="2020-01-02", source="yahoo")
    assert frame.loc[0, "symbol"] == "2330.TW"
    assert frame.loc[0, "source"] == "yahoo_pit"
    for column in ("PE", "PB", "ROE", "RevenueGrowth", "EarningsGrowth", "MarketCap"):
        assert pd.isna(frame.loc[0, column])

    from screener import filter_stocks

    kept = filter_stocks(frame, pe_max=50)
    assert kept.empty


def test_fetch_pit_prices_stop_at_as_of(monkeypatch):
    seen = {}

    def _yahoo(symbol, lag_days):
        assert lag_days == 45
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "period_end": "2019-12-31",
                    "available_at": "2020-02-15",
                    "frequency": "A",
                    "revenue": 500,
                    "net_income": 50,
                    "diluted_shares": 10,
                    "equity": 250,
                    "shares_outstanding": 10,
                }
            ]
        )

    def _prices(symbol, start, end, provider=None):
        seen["start"] = start
        seen["end"] = end
        return pd.DataFrame(
            {"Close": [80.0, 500.0]},
            index=pd.to_datetime(["2020-03-01", "2020-06-01"]),
        )

    monkeypatch.setattr("fundamentals.pit.load_yahoo_statements", _yahoo)
    monkeypatch.setattr("fundamentals.pit.fetch_data", _prices)
    frame = fetch_pit_fundamentals(["AAA"], as_of="2020-03-01", lag_days=45, source="yahoo")
    assert seen == {"start": "2020-01-31", "end": "2020-03-02"}
    assert frame.loc[0, "PE"] == pytest.approx(16.0)
    assert frame.loc[0, "MarketCap"] == pytest.approx(800.0)


def test_statement_cache_skips_the_second_download(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _yahoo(symbol, lag_days):
        calls["n"] += 1
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "period_end": "2019-12-31",
                    "available_at": "2020-03-01",
                    "frequency": "A",
                    "revenue": 100,
                    "net_income": 20,
                    "diluted_shares": 10,
                    "equity": 200,
                    "shares_outstanding": 10,
                }
            ]
        )

    monkeypatch.setenv("BACKTEST_DATA_CACHE", "1")
    monkeypatch.setenv("BACKTEST_DATA_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("fundamentals.pit.load_yahoo_statements", _yahoo)
    monkeypatch.setattr(
        "fundamentals.pit.fetch_data",
        lambda symbol, start, end, provider=None: pd.DataFrame(
            {"Close": [20.0]}, index=pd.to_datetime(["2020-03-01"])
        ),
    )
    first = fetch_pit_fundamentals(["AAA"], as_of="2020-03-01", source="yahoo")
    second = fetch_pit_fundamentals(["AAA"], as_of="2020-06-01", source="yahoo")
    assert calls["n"] == 1
    # EPS is 2. The mocked close is 20 on both calls, so the cached statements
    # produce the same PE. The second call must not download again.
    assert first.loc[0, "PE"] == pytest.approx(10.0)
    assert second.loc[0, "PE"] == pytest.approx(10.0)
    assert list(tmp_path.glob("fundamentals/yahoo/*.parquet"))


def test_bad_lag_and_unknown_source_raise(monkeypatch):
    monkeypatch.setenv("BACKTEST_FUNDAMENTALS_LAG_DAYS", "soon")
    with pytest.raises(DataProviderConfigError, match="BACKTEST_FUNDAMENTALS_LAG_DAYS"):
        resolve_lag_days()
    monkeypatch.delenv("BACKTEST_FUNDAMENTALS_LAG_DAYS")
    with pytest.raises(DataProviderConfigError, match="Unknown fundamentals source"):
        fetch_pit_fundamentals(["AAA"], as_of="2020-01-01", source="compustat")


def test_live_snapshot_is_labeled_and_not_used_as_pit(monkeypatch):
    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            return {
                "trailingPE": 12,
                "priceToBook": 3,
                "returnOnEquity": 0.2,
                "revenueGrowth": 0.1,
                "earningsGrowth": 0.05,
                "marketCap": 1_000_000_000,
            }

    monkeypatch.setattr("fundamentals.snapshot.yf.Ticker", _Ticker)
    fetch_fundamentals.clear()
    frame = fetch_fundamentals(["AAPL"])
    assert frame.loc[0, "source"] == "live_snapshot"
    assert frame.loc[0, "PE"] == 12
    assert frame.loc[0, "as_of"] is None


def test_factor_score_window_ends_at_as_of(monkeypatch):
    from screener import apply_factor_score

    seen = {}

    def _factors(symbols, start, end, as_of=None):
        seen["symbols"] = symbols
        seen["start"] = start
        seen["end"] = end
        seen["as_of"] = as_of
        return pd.DataFrame({"symbol": ["AAA"], "Momentum": [0.2], "Volatility": [0.1]})

    monkeypatch.setattr("screener.fetch_price_factors", _factors)
    fundamentals = pd.DataFrame({"symbol": ["AAA"], "ROE": [0.2]})
    ranked = apply_factor_score(
        fundamentals,
        "2010-01-01",
        "2020-01-01",
        {"momentum": 1.0, "roe": 1.0},
        top_n=1,
        as_of="2018-01-01",
    )
    assert seen == {
        "symbols": ("AAA",),
        "start": "2017-01-01",
        "end": "2018-01-02",
        "as_of": "2018-01-01",
    }
    assert ranked.loc[0, "symbol"] == "AAA"
    assert "Composite_Score" in ranked.columns
