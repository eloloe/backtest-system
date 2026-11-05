"""Point-in-time fundamentals for a historical screen.

Ratios are built from period statements and a price on ``as_of``. A row is
usable only once its filing date, earnings date, or a fixed reporting lag
has arrived. This is not a Compustat-style revision database: when the
vendor sends one number per period, that number is whatever they publish
today. If two versions of the same period are both in the panel, an
``as_of`` date sees only the version already filed.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

from backtest_system.data import fetch_data
from backtest_system.providers.base import DataProviderConfigError

DEFAULT_LAG_DAYS = 60
RATIO_COLUMNS = ["PE", "PB", "ROE", "RevenueGrowth", "EarningsGrowth", "MarketCap"]
STATEMENT_COLUMNS = [
    "period_end",
    "period_type",
    "filing_date",
    "earnings_date",
    "revenue",
    "net_income",
    "eps",
    "shares",
    "equity",
    "revision",
]

_YAHOO_TYPES = [
    "quarterlyTotalRevenue",
    "quarterlyNetIncome",
    "quarterlyDilutedEPS",
    "quarterlyDilutedAverageShares",
    "quarterlyBasicAverageShares",
    "quarterlyOrdinarySharesNumber",
    "quarterlyStockholdersEquity",
    "quarterlyCommonStockEquity",
    "annualTotalRevenue",
    "annualNetIncome",
    "annualDilutedEPS",
    "annualDilutedAverageShares",
    "annualBasicAverageShares",
    "annualOrdinarySharesNumber",
    "annualStockholdersEquity",
    "annualCommonStockEquity",
]
_YAHOO_FIELDS = {
    "TotalRevenue": "revenue",
    "NetIncome": "net_income",
    "DilutedEPS": "eps",
    "DilutedAverageShares": "shares_diluted",
    "BasicAverageShares": "shares_basic",
    "OrdinarySharesNumber": "shares_ordinary",
    "StockholdersEquity": "equity",
    "CommonStockEquity": "equity_common",
}
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def reporting_lag_days(explicit: int | None = None) -> int:
    """Calendar days after period end when no filing or earnings date exists.

    ``BACKTEST_FUNDAMENTALS_LAG_DAYS`` overrides the default of 60. An
    explicit argument wins over the environment.
    """
    if explicit is None:
        raw = os.environ.get("BACKTEST_FUNDAMENTALS_LAG_DAYS")
        if raw is None or raw.strip() == "":
            return DEFAULT_LAG_DAYS
        try:
            lag = int(raw.strip())
        except ValueError as exc:
            raise ValueError(
                "BACKTEST_FUNDAMENTALS_LAG_DAYS must be a non-negative integer, "
                f"got {raw!r}."
            ) from exc
    else:
        try:
            lag = int(explicit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"lag_days must be a non-negative integer, got {explicit!r}.") from exc
    if lag < 0:
        raise ValueError(f"lag_days must be >= 0, got {lag}.")
    return lag


def empty_ratios() -> dict[str, float]:
    return {col: math.nan for col in RATIO_COLUMNS}


def empty_statements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_end": pd.Series(dtype="datetime64[ns]"),
            "period_type": pd.Series(dtype="object"),
            "filing_date": pd.Series(dtype="datetime64[ns]"),
            "earnings_date": pd.Series(dtype="datetime64[ns]"),
            "revenue": pd.Series(dtype="float64"),
            "net_income": pd.Series(dtype="float64"),
            "eps": pd.Series(dtype="float64"),
            "shares": pd.Series(dtype="float64"),
            "equity": pd.Series(dtype="float64"),
            "revision": pd.Series(dtype="int64"),
        }
    )


def available_on(period_end, filing_date, earnings_date, lag_days: int) -> pd.Timestamp:
    """First day a statement may be used.

    Filing date wins, then the earnings announcement, then period end plus
    ``lag_days``. A known announcement that is later than the lag stays
    hidden until that announcement: the lag is only a stand-in.
    """
    if pd.notna(filing_date):
        return pd.Timestamp(filing_date).normalize()
    if pd.notna(earnings_date):
        return pd.Timestamp(earnings_date).normalize()
    return pd.Timestamp(period_end).normalize() + pd.Timedelta(days=int(lag_days))


def finalize_statements(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return empty_statements()
    work = frame.copy()
    for col in STATEMENT_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA
    work = work.loc[:, STATEMENT_COLUMNS]
    for col in ("period_end", "filing_date", "earnings_date"):
        work[col] = _normalize_days(work[col])
    for col in ("revenue", "net_income", "eps", "shares", "equity"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["revision"] = pd.to_numeric(work["revision"], errors="coerce").fillna(0).astype("int64")
    work["period_type"] = work["period_type"].astype(object)
    work = work.dropna(subset=["period_end"])
    work = work[work["period_type"].isin(["Q", "A", "E"])]
    work = work.sort_values(["period_type", "period_end", "revision"]).reset_index(drop=True)
    if work.empty:
        return empty_statements()
    return work


def select_visible(panel: pd.DataFrame, as_of, lag_days: int) -> pd.DataFrame:
    """Statements whose availability date is on or before ``as_of``.

    Two rows for the same period keep the one with the later availability
    date (then the higher ``revision``). A restatement filed after
    ``as_of`` is dropped, so it cannot replace the earlier figure.
    """
    prepared = finalize_statements(panel)
    if prepared.empty:
        return prepared
    as_of_day = _one_day(as_of)
    available = [
        available_on(row.period_end, row.filing_date, row.earnings_date, lag_days)
        for row in prepared.itertuples(index=False)
    ]
    prepared = prepared.copy()
    prepared["available"] = available
    visible = prepared[prepared["available"] <= as_of_day]
    if visible.empty:
        return empty_statements()
    visible = visible.sort_values(["period_type", "period_end", "available", "revision"])
    visible = visible.drop_duplicates(["period_type", "period_end"], keep="last")
    return finalize_statements(visible.drop(columns=["available"]))


def compute_ratios(panel: pd.DataFrame, as_of, price, lag_days: int = DEFAULT_LAG_DAYS) -> dict:
    """PE, PB, ROE, growth, and market cap knowable on ``as_of``.

    Quarterly statements are preferred. Earnings-date EPS (one row per
    announcement) is used for PE and earnings growth when the quarterly
    window is not available yet. Annual figures fill ROE, book, and
    revenue in that case. Missing inputs stay NaN.
    """
    visible = select_visible(panel, as_of, lag_days)
    if visible.empty:
        return empty_ratios()

    quarters = visible[visible["period_type"] == "Q"]
    annuals = visible[visible["period_type"] == "A"]
    earnings = visible[visible["period_type"] == "E"]

    quarter_block = _last_block(quarters, 4, 240, 400)
    if quarter_block is not None:
        trail = _trail(quarters, quarter_block)
        if trail is not None:
            equity, shares = _balance(quarters, quarter_block)
            return _pack(trail, equity, shares, price, earnings_from_eps=False)

    earnings_block = _last_block(earnings, 4, 240, 400)
    earnings_trail = _trail(earnings, earnings_block) if earnings_block is not None else None
    annual_trail = _annual_trail(annuals)
    if earnings_trail is not None:
        trail = {
            "ni": None if annual_trail is None else annual_trail["ni"],
            "rev": None if annual_trail is None else annual_trail["rev"],
            "eps": earnings_trail["eps"],
            "growth_ni": None,
            "growth_rev": None if annual_trail is None else annual_trail["growth_rev"],
            "growth_eps": earnings_trail["growth_eps"],
        }
        equity, shares = _balance(annuals if annual_trail is not None else quarters)
        return _pack(trail, equity, shares, price, earnings_from_eps=True)

    if annual_trail is not None:
        equity, shares = _balance(annuals)
        return _pack(annual_trail, equity, shares, price, earnings_from_eps=False)
    return empty_ratios()


def fetch_fundamentals_pit(
    symbols: list,
    as_of,
    lag_days: int | None = None,
) -> pd.DataFrame:
    """Fundamentals that would have been knowable on ``as_of``.

    Same columns as the live snapshot: symbol, PE, PB, ROE, RevenueGrowth,
    EarningsGrowth, MarketCap. A thin or failed symbol is a row of NaNs.
    """
    lag = reporting_lag_days(lag_days)
    _cache_enabled()
    rows = [_one_symbol(str(symbol).strip(), as_of, lag) for symbol in symbols]
    if not rows:
        return pd.DataFrame(columns=["symbol", *RATIO_COLUMNS])
    frame = pd.DataFrame(rows)
    for col in RATIO_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def load_statement_panel(symbol: str) -> pd.DataFrame:
    """Cached statement panel for one symbol.

    Uses Financial Modeling Prep when ``FMP_API_KEY`` is set and the
    request returns rows. Otherwise Yahoo period financials, plus earnings
    announcement dates when Yahoo will give them. A missing key, a blank
    key, or an FMP error stays on the Yahoo path.
    """
    symbol = symbol.strip()
    key = _fmp_api_key()
    if key:
        cached = _read_cache(symbol, "fmp")
        if cached is not None:
            return cached
        try:
            panel = fetch_fmp_panel(symbol, key)
        except Exception as exc:
            # The exception text can contain the request URL, and the key
            # is a query parameter. Keep it out of the log.
            print(
                f"FMP fundamentals failed for {symbol}, using Yahoo ({type(exc).__name__})."
            )
            panel = empty_statements()
        if not panel.empty:
            _write_cache(symbol, "fmp", panel)
            return panel

    cached = _read_cache(symbol, "yahoo")
    if cached is not None:
        return cached
    panel = fetch_yahoo_panel(symbol)
    if not panel.empty:
        _write_cache(symbol, "yahoo", panel)
    return panel


def fetch_yahoo_panel(symbol: str) -> pd.DataFrame:
    try:
        payload = _yahoo_get(symbol)
        statements = parse_yahoo_timeseries(payload)
    except Exception as exc:
        print(f"Yahoo fundamentals failed for {symbol}: {exc}")
        statements = empty_statements()
    earnings = _yahoo_earnings(symbol)
    return _with_earnings(statements, earnings)


def fetch_fmp_panel(symbol: str, api_key: str) -> pd.DataFrame:
    income = _fmp_get(
        "https://financialmodelingprep.com/stable/income-statement",
        symbol,
        api_key,
    )
    balance = _fmp_get(
        "https://financialmodelingprep.com/stable/balance-sheet-statement",
        symbol,
        api_key,
    )
    if not isinstance(income, list) or not isinstance(balance, list):
        return empty_statements()
    return parse_fmp_statements(income, balance)


def parse_yahoo_timeseries(payload: dict | None) -> pd.DataFrame:
    result = ((payload or {}).get("timeseries") or {}).get("result") or []
    buckets: dict[tuple[str, pd.Timestamp], dict] = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        for key, series in item.items():
            parsed = _yahoo_key(key)
            if parsed is None or not isinstance(series, list):
                continue
            period_type, field = parsed
            for point in series:
                if not isinstance(point, dict):
                    continue
                period_end = _maybe_day(point.get("asOfDate"))
                number = _number((point.get("reportedValue") or {}).get("raw"))
                if period_end is None or number is None:
                    continue
                buckets.setdefault((period_type, period_end), {})[field] = number

    rows = []
    for (period_type, period_end), fields in buckets.items():
        # Ordinary shares for market cap. Diluted average shares are only
        # a fallback when Yahoo did not send a share count.
        shares = fields.get("shares_ordinary")
        if shares is None:
            shares = fields.get("shares_basic", fields.get("shares_diluted"))
        equity = fields.get("equity", fields.get("equity_common"))
        rows.append(
            {
                "period_end": period_end,
                "period_type": period_type,
                "revenue": fields.get("revenue"),
                "net_income": fields.get("net_income"),
                "eps": fields.get("eps"),
                "shares": shares,
                "equity": equity,
            }
        )
    return finalize_statements(pd.DataFrame(rows))


def parse_fmp_statements(income: list, balance: list) -> pd.DataFrame:
    """Join FMP income and balance rows on period end.

    The row becomes visible on the later of the two filing dates, so book
    value is not used before the balance sheet was filed.
    """
    combined: dict[str, dict] = {}
    for source, kind in ((income, "income"), (balance, "balance")):
        for raw in source:
            if not isinstance(raw, dict):
                continue
            period_end = _maybe_day(raw.get("date"))
            if period_end is None:
                continue
            slot = combined.setdefault(period_end.strftime("%Y-%m-%d"), {"period_end": period_end})
            filing = _fmp_filing(raw)
            slot["filing_date"] = _later(slot.get("filing_date"), filing)
            period = str(raw.get("period") or "")
            slot["period_type"] = "A" if period.upper() in {"FY", "ANNUAL"} else "Q"
            if kind == "income":
                slot["revenue"] = _number(raw.get("revenue"))
                slot["net_income"] = _number(raw.get("netIncome"))
                eps = _number(raw.get("epsdiluted"))
                slot["eps"] = eps if eps is not None else _number(raw.get("eps"))
                shares = _number(raw.get("weightedAverageShsOut"))
                if shares is None or shares <= 0:
                    shares = _number(raw.get("weightedAverageShsOutDil"))
                slot["shares"] = shares if shares and shares > 0 else None
            else:
                equity = _number(raw.get("totalStockholdersEquity"))
                slot["equity"] = equity if equity not in (None, 0) else None
    return finalize_statements(pd.DataFrame(list(combined.values())))


def price_as_of(symbol: str, as_of) -> float:
    """Last close on or before ``as_of`` from the price provider.

    ``end`` is exclusive, so the request stops the morning after ``as_of``.
    """
    day = _one_day(as_of)
    start = (day - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hist = fetch_data(symbol, start, end)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return math.nan
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if close.empty:
        return math.nan
    index = close.index
    if getattr(index, "tz", None) is not None:
        dates = pd.to_datetime(index.strftime("%Y-%m-%d"))
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    close = close.copy()
    close.index = pd.DatetimeIndex(dates)
    close = close[close.index <= day]
    if close.empty:
        return math.nan
    return float(close.iloc[-1])


def _one_symbol(symbol: str, as_of, lag: int) -> dict:
    blank = {"symbol": symbol, **empty_ratios()}
    try:
        panel = load_statement_panel(symbol)
        price = price_as_of(symbol, as_of)
        ratios = compute_ratios(panel, as_of, price, lag)
    except DataProviderConfigError:
        raise
    except Exception as exc:
        print(f"Error fetching point-in-time fundamentals for {symbol}: {exc}")
        return blank
    return {"symbol": symbol, **ratios}


def _with_earnings(statements: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    statements = finalize_statements(statements)
    if earnings is None or earnings.empty:
        return statements
    dates = earnings["period_end"]
    if not statements.empty:
        matched = []
        for row in statements.itertuples(index=False):
            if row.period_type != "Q":
                matched.append(pd.NaT)
                continue
            matched.append(_match_earnings(row.period_end, dates))
        statements = statements.copy()
        statements["earnings_date"] = matched
    extra = earnings.copy()
    extra["period_type"] = "E"
    return finalize_statements(pd.concat([statements, extra], ignore_index=True))


def _match_earnings(period_end, dates: pd.Series):
    start = pd.Timestamp(period_end).normalize()
    end = start + pd.Timedelta(days=120)
    hits = dates[(dates >= start) & (dates <= end)]
    if len(hits) == 0:
        return pd.NaT
    return pd.Timestamp(hits.min()).normalize()


def _yahoo_earnings(symbol: str) -> pd.DataFrame:
    try:
        frame = yf.Ticker(symbol).get_earnings_dates(limit=100)
    except Exception as exc:
        print(f"Yahoo earnings dates failed for {symbol}: {exc}")
        return empty_statements()
    if frame is None or len(frame) == 0 or "Reported EPS" not in frame.columns:
        return empty_statements()
    index = frame.index
    if getattr(index, "tz", None) is not None:
        days = pd.to_datetime(index.strftime("%Y-%m-%d"))
    else:
        days = pd.to_datetime(pd.Series(index).dt.strftime("%Y-%m-%d"))
    eps = pd.to_numeric(frame["Reported EPS"], errors="coerce")
    table = pd.DataFrame({"period_end": pd.DatetimeIndex(days), "eps": eps.to_numpy()})
    table = table.dropna(subset=["eps"])
    if table.empty:
        return empty_statements()
    table = table.groupby("period_end", as_index=False)["eps"].last()
    table["period_type"] = "E"
    table["earnings_date"] = table["period_end"]
    return finalize_statements(table)


def _yahoo_get(symbol: str) -> dict:
    url = (
        "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/"
        + quote(symbol, safe="")
    )
    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "type": ",".join(_YAHOO_TYPES),
            "period1": "0",
            "period2": str(int(time.time()) + 86400),
        },
        headers=_HEADERS,
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Yahoo fundamentals HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Yahoo fundamentals returned an unexpected payload")
    return payload


def _yahoo_key(key: str) -> tuple[str, str] | None:
    if key.startswith("quarterly"):
        period_type, name = "Q", key[len("quarterly") :]
    elif key.startswith("annual"):
        period_type, name = "A", key[len("annual") :]
    else:
        return None
    field = _YAHOO_FIELDS.get(name)
    if field is None:
        return None
    return period_type, field


def _fmp_get(url: str, symbol: str, api_key: str):
    response = requests.get(
        url,
        params={"symbol": symbol, "period": "quarter", "limit": 120, "apikey": api_key},
        headers=_HEADERS,
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"FMP HTTP {response.status_code}")
    return response.json()


def _fmp_api_key() -> str | None:
    raw = os.environ.get("FMP_API_KEY")
    if raw is None:
        return None
    key = raw.strip()
    return key or None


def _fmp_filing(row: dict):
    for key in ("filingDate", "fillingDate", "acceptedDate"):
        day = _maybe_day(row.get(key))
        if day is not None:
            return day
    return None


def _last_block(frame: pd.DataFrame, n: int, min_span: int, max_span: int) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    part = frame.sort_values(["period_end", "revision"]).drop_duplicates("period_end", keep="last")
    if len(part) < n:
        return None
    tail = part.tail(n)
    if not _span_ok(tail, min_span, max_span):
        return None
    return tail


def _span_ok(frame: pd.DataFrame, min_span: int, max_span: int) -> bool:
    span = int((frame["period_end"].iloc[-1] - frame["period_end"].iloc[0]).days)
    return min_span <= span <= max_span


def _sum_col(block: pd.DataFrame | None, col: str):
    if block is None or col not in block.columns:
        return None
    total = pd.to_numeric(block[col], errors="coerce").sum(min_count=len(block))
    if pd.isna(total):
        return None
    return float(total)


def _trail(frame: pd.DataFrame, block: pd.DataFrame) -> dict | None:
    ni = _sum_col(block, "net_income")
    rev = _sum_col(block, "revenue")
    eps = _sum_col(block, "eps")
    if ni is None and rev is None and eps is None:
        return None
    return {
        "ni": ni,
        "rev": rev,
        "eps": eps,
        "growth_ni": _growth(frame, "net_income"),
        "growth_rev": _growth(frame, "revenue"),
        "growth_eps": _growth(frame, "eps"),
    }


def _growth(frame: pd.DataFrame, col: str):
    block = _last_block(frame, 8, 540, 750)
    if block is None:
        return None
    recent, prior = block.tail(4), block.head(4)
    if not _span_ok(recent, 240, 400) or not _span_ok(prior, 240, 400):
        return None
    new = _sum_col(recent, col)
    old = _sum_col(prior, col)
    if new is None or old is None or old <= 0:
        return None
    return new / old - 1.0


def _annual_trail(frame: pd.DataFrame) -> dict | None:
    if frame is None or frame.empty:
        return None
    part = frame.sort_values(["period_end", "revision"]).drop_duplicates("period_end", keep="last")
    latest = part.iloc[-1]
    previous = part.iloc[-2] if len(part) >= 2 else None

    def value(row, col):
        if row is None or pd.isna(row[col]):
            return None
        return float(row[col])

    def growth(col):
        if previous is None:
            return None
        gap = int((latest.period_end - previous.period_end).days)
        if gap < 300 or gap > 430:
            return None
        new, old = value(latest, col), value(previous, col)
        if new is None or old is None or old <= 0:
            return None
        return new / old - 1.0

    ni, rev, eps = value(latest, "net_income"), value(latest, "revenue"), value(latest, "eps")
    if ni is None and rev is None and eps is None:
        return None
    return {
        "ni": ni,
        "rev": rev,
        "eps": eps,
        "growth_ni": growth("net_income"),
        "growth_rev": growth("revenue"),
        "growth_eps": growth("eps"),
    }


def _balance(frame: pd.DataFrame, block: pd.DataFrame | None = None) -> tuple[float, float]:
    equity = _latest_value(block if block is not None else frame, "equity")
    shares = _latest_value(block if block is not None else frame, "shares")
    if block is not None and math.isnan(equity):
        equity = _latest_value(frame, "equity")
    if block is not None and math.isnan(shares):
        shares = _latest_value(frame, "shares")
    return equity, shares


def _latest_value(frame: pd.DataFrame | None, col: str) -> float:
    if frame is None or frame.empty or col not in frame.columns:
        return math.nan
    part = frame.dropna(subset=[col]).sort_values("period_end")
    if part.empty:
        return math.nan
    return float(part.iloc[-1][col])


def _pack(trail: dict, equity: float, shares: float, price, earnings_from_eps: bool) -> dict:
    px = _as_float(price)
    market_cap = px * shares if _positive(px) and _positive(shares) else math.nan
    eps, ni = trail.get("eps"), trail.get("ni")
    if _positive(eps) and _positive(px):
        pe = px / eps
    elif _positive(ni) and _positive(market_cap):
        pe = market_cap / ni
    else:
        pe = math.nan
    pb = market_cap / equity if _positive(market_cap) and _positive(equity) else math.nan
    roe = ni / equity if ni is not None and _positive(equity) else math.nan
    if earnings_from_eps:
        earnings_growth = trail.get("growth_eps")
    else:
        earnings_growth = trail.get("growth_ni")
        if earnings_growth is None:
            earnings_growth = trail.get("growth_eps")
    revenue_growth = trail.get("growth_rev")
    return {
        "PE": pe,
        "PB": pb,
        "ROE": roe if roe is not None else math.nan,
        "RevenueGrowth": math.nan if revenue_growth is None else revenue_growth,
        "EarningsGrowth": math.nan if earnings_growth is None else earnings_growth,
        "MarketCap": market_cap,
    }


def _cache_enabled() -> bool:
    raw = os.environ.get("BACKTEST_DATA_CACHE")
    if raw is None or raw.strip() == "":
        return True
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise DataProviderConfigError(f"BACKTEST_DATA_CACHE must be 0/1/true/false, got {raw!r}.")


def _cache_path(symbol: str, source: str) -> Path | None:
    if not _cache_enabled():
        return None
    root = Path(os.environ.get("BACKTEST_DATA_CACHE_DIR", "data_cache"))
    digest = hashlib.sha256(f"v1|{source}|{symbol}".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", symbol)[:40] or "symbol"
    return root / "fundamentals" / "v1" / source / f"{safe}_{digest}.parquet"


def _read_cache(symbol: str, source: str) -> pd.DataFrame | None:
    path = _cache_path(symbol, source)
    if path is None or not path.exists():
        return None
    try:
        frame = pd.read_parquet(path, engine="pyarrow")
        frame = finalize_statements(frame)
    except Exception:
        return None
    if frame.empty:
        return None
    return frame


def _write_cache(symbol: str, source: str, frame: pd.DataFrame) -> None:
    path = _cache_path(symbol, source)
    if path is None or frame.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        finalize_statements(frame).to_parquet(path, engine="pyarrow")
    except Exception:
        return


def _normalize_days(values) -> pd.Series:
    stamps = pd.to_datetime(values, errors="coerce")
    if isinstance(stamps.dtype, pd.DatetimeTZDtype):
        stamps = pd.to_datetime(stamps.dt.strftime("%Y-%m-%d"))
    return stamps.dt.normalize()


def _one_day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"invalid as_of date: {value!r}")
    if stamp.tzinfo is not None:
        stamp = pd.Timestamp(stamp.strftime("%Y-%m-%d"))
    return stamp.normalize()


def _maybe_day(value):
    if value is None or value == "":
        return None
    text = str(value)[:10]
    try:
        stamp = pd.Timestamp(text)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.normalize()


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _as_float(value) -> float:
    number = _number(value)
    if number is None:
        return math.nan
    return number


def _positive(value) -> bool:
    number = _as_float(value)
    return not math.isnan(number) and number > 0


def _later(left, right):
    if left is None or pd.isna(left):
        return right
    if right is None or pd.isna(right):
        return left
    return max(pd.Timestamp(left), pd.Timestamp(right))
