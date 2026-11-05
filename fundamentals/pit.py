"""Point-in-time ratios from period statements and an as-of price.

Only statement rows with ``available_at`` on or before ``as_of`` are used.
A later restatement of the same period replaces the earlier row once its
own ``available_at`` has been reached, and not before.

Trailing figures prefer four contiguous quarters. One missing quarter
drops the quarterly trail and the latest annual statement is used instead.
Growth needs the prior trailing year, or the prior fiscal year. The prior
base has to be positive.

PE is price / trailing diluted EPS, and only when EPS is positive.
PB is price / book value per share, and only when equity is positive.
ROE is trailing net income / that latest positive equity.
Market cap is the as-of close times shares outstanding, or diluted shares
when outstanding shares are missing.
"""

from __future__ import annotations

import math
import os

import pandas as pd

from data_fetcher import fetch_data
from fundamentals.cache import read_statement_cache, write_statement_cache
from fundamentals.fmp import load_fmp_statements
from fundamentals.statements import as_timestamp, empty_statements, normalize_statements
from fundamentals.yahoo_statements import load_yahoo_statements
from providers.base import DataProviderConfigError, MissingApiKeyError, ProviderError

DEFAULT_LAG_DAYS = 60
PRICE_LOOKBACK_DAYS = 30

_METRIC_COLUMNS = [
    "symbol",
    "PE",
    "PB",
    "ROE",
    "RevenueGrowth",
    "EarningsGrowth",
    "MarketCap",
    "as_of",
    "source",
    "statement_period_end",
    "statement_available_at",
    "lag_days",
]

_warned: set[str] = set()


def fetch_pit_fundamentals(
    symbols,
    as_of,
    lag_days: int | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Ratios knowable at ``as_of`` for each symbol.

    ``source`` is ``yahoo``, ``fmp``, or ``auto`` (the default). ``auto``
    uses FMP when ``FMP_API_KEY`` is set and otherwise Yahoo. A missing key
    does not raise on the default path. An empty Yahoo panel stays empty;
    numbers are not filled in.
    """
    lag = resolve_lag_days(lag_days)
    chosen, allow_fallback = resolve_fundamentals_source(source)
    as_of_ts = as_timestamp(as_of)
    if as_of_ts is None:
        raise ValueError(f"invalid as_of: {as_of!r}")

    rows = []
    for symbol in symbols:
        label = str(symbol).strip()
        if not label:
            continue
        try:
            statements, used = _load_statements(label, lag, chosen, allow_fallback)
            row = ratios_as_of(
                statements,
                as_of_ts,
                symbol=label,
                prices=_closes(label, as_of_ts),
            )
            row["source"] = used
        except ProviderError:
            raise
        except Exception as exc:
            print(f"Error fetching point-in-time fundamentals for {label}: {exc}")
            row = _blank_metrics(label)
            row["source"] = "unavailable"
        row["as_of"] = as_of_ts.strftime("%Y-%m-%d")
        row["lag_days"] = lag
        rows.append(row)

    frame = pd.DataFrame(rows, columns=_METRIC_COLUMNS)
    for column in ("PE", "PB", "ROE", "RevenueGrowth", "EarningsGrowth", "MarketCap", "lag_days"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def resolve_lag_days(lag_days: int | None = None) -> int:
    """Reporting lag in calendar days. Default 60, or ``BACKTEST_FUNDAMENTALS_LAG_DAYS``."""
    if lag_days is None:
        raw = os.environ.get("BACKTEST_FUNDAMENTALS_LAG_DAYS", str(DEFAULT_LAG_DAYS))
        raw = raw.strip() or str(DEFAULT_LAG_DAYS)
        try:
            lag_days = int(raw)
        except ValueError as exc:
            raise DataProviderConfigError(
                f"BACKTEST_FUNDAMENTALS_LAG_DAYS must be an integer, got {raw!r}."
            ) from exc
    if int(lag_days) < 0:
        raise DataProviderConfigError("Reporting lag must be >= 0 days.")
    return int(lag_days)


def resolve_fundamentals_source(source: str | None = None) -> tuple[str, bool]:
    """Return ``(yahoo|fmp, allow Yahoo fallback)``."""
    raw = source if source is not None else os.environ.get("BACKTEST_FUNDAMENTALS_SOURCE", "auto")
    key = (raw or "auto").strip().lower() or "auto"
    if key not in {"auto", "yahoo", "fmp"}:
        raise DataProviderConfigError(
            f"Unknown fundamentals source {raw!r}. Expected auto, yahoo, or fmp."
        )
    if key == "yahoo":
        return "yahoo", False
    api_key = os.environ.get("FMP_API_KEY", "").strip()
    if key == "fmp":
        if not api_key:
            raise MissingApiKeyError("FMP is selected but FMP_API_KEY is not set.")
        return "fmp", False
    if api_key:
        return "fmp", True
    return "yahoo", False


def resolve_vintages(statements: pd.DataFrame | None, as_of) -> pd.DataFrame:
    """Latest knowable version of each period. Later vintages stay hidden."""
    as_of_ts = as_timestamp(as_of)
    if as_of_ts is None:
        raise ValueError(f"invalid as_of: {as_of!r}")
    frame = normalize_statements(statements)
    if frame.empty:
        return frame
    known = frame[frame["available_at"] <= as_of_ts]
    if known.empty:
        return known.reset_index(drop=True)
    known = known.sort_values(
        ["symbol", "frequency", "period_end", "available_at"],
        kind="mergesort",
    )
    return known.drop_duplicates(
        ["symbol", "frequency", "period_end"],
        keep="last",
    ).reset_index(drop=True)


def ratios_as_of(statements, as_of, *, symbol: str, prices: pd.Series | None = None) -> dict:
    """One symbol's ratios from synthetic or vendor statement rows."""
    label = symbol.strip()
    as_of_ts = as_timestamp(as_of)
    if as_of_ts is None:
        raise ValueError(f"invalid as_of: {as_of!r}")

    resolved = resolve_vintages(statements, as_of_ts)
    if not resolved.empty:
        resolved = resolved[resolved["symbol"].str.upper() == label.upper()].reset_index(drop=True)
    quarters = _frequency(resolved, "Q")
    annuals = _frequency(resolved, "A")
    price = _price_as_of(prices, as_of_ts)

    pe = roe = None
    used_period = None
    used_available = None

    def mark(period, available) -> None:
        nonlocal used_period, used_available
        if period is None:
            return
        period_ts = as_timestamp(period)
        if period_ts is None:
            return
        if used_period is None or period_ts >= used_period:
            used_period = period_ts
            used_available = as_timestamp(available)

    earnings = _level(quarters, annuals, ("net_income", "diluted_shares"))
    income = _level(quarters, annuals, ("net_income",))
    if income is not None:
        ni = _total(income, "net_income")
        end = _end(income)
        mark(end, _available(income))
        equity_row = _latest_row(resolved, "equity", end)
        if equity_row is not None and float(equity_row["equity"]) > 0:
            roe = ni / float(equity_row["equity"])
    if earnings is not None and price is not None:
        eps = _eps(earnings)
        if eps is not None and eps > 0:
            pe = price / eps
        mark(_end(earnings), _available(earnings))

    rev_g = _growth_metric(quarters, annuals, "revenue", mark)
    earn_g = _growth_metric(quarters, annuals, "net_income", mark)
    pb, mcap = _valuation(resolved, price, mark)

    return {
        "symbol": label,
        "PE": _finite(pe),
        "PB": _finite(pb),
        "ROE": _finite(roe),
        "RevenueGrowth": _finite(rev_g),
        "EarningsGrowth": _finite(earn_g),
        "MarketCap": _finite(mcap),
        "statement_period_end": None if used_period is None else used_period.strftime("%Y-%m-%d"),
        "statement_available_at": None if used_available is None else used_available.strftime("%Y-%m-%d"),
    }


def _load_statements(symbol: str, lag_days: int, source: str, allow_fallback: bool):
    if source == "yahoo":
        return _cached("yahoo", symbol, lag_days, load_yahoo_statements), "yahoo_pit"
    try:
        frame = _cached("fmp", symbol, lag_days, load_fmp_statements)
    except MissingApiKeyError:
        raise
    except Exception as exc:
        if allow_fallback:
            _warn_once(f"FMP fundamentals unavailable ({exc}). Using Yahoo statements.")
            return _cached("yahoo", symbol, lag_days, load_yahoo_statements), "yahoo_pit"
        if _fatal_fmp(exc):
            raise
        print(f"FMP fundamentals failed for {symbol}: {exc}")
        return empty_statements(), "fmp"
    if frame.empty and allow_fallback:
        yahoo = _cached("yahoo", symbol, lag_days, load_yahoo_statements)
        if not yahoo.empty:
            return yahoo, "yahoo_pit"
    return frame, "fmp"


def _cached(source: str, symbol: str, lag_days: int, loader):
    cached = read_statement_cache(source, symbol, lag_days)
    if cached is not None:
        return cached
    frame = loader(symbol, lag_days)
    write_statement_cache(source, symbol, lag_days, frame)
    return frame


def _closes(symbol: str, as_of: pd.Timestamp) -> pd.Series:
    start = (as_of - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frame = fetch_data(symbol, start, end)
    if frame.empty or "Close" not in frame.columns:
        return pd.Series(dtype=float)
    return frame["Close"]


def _frequency(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame["frequency"] == frequency]


def _level(quarters: pd.DataFrame, annuals: pd.DataFrame, columns: tuple[str, ...]):
    """Four contiguous quarters, unless a newer annual statement exists."""
    window = _quarter_window(quarters, columns, 4)
    latest, _prior = _annual_pair(annuals, columns)
    if window is None:
        return latest
    if latest is not None and as_timestamp(latest["period_end"]) > _end(window):
        return latest
    return window


def _growth_metric(quarters, annuals, column: str, mark) -> float | None:
    pair = _ttm_pair(quarters, (column,))
    latest, prior = _annual_pair(annuals, (column,))
    if pair is not None:
        previous, current = pair
        if latest is not None and as_timestamp(latest["period_end"]) > _end(current):
            pair = None
        else:
            mark(_end(current), _available(current))
            return _growth(_total(current, column), _total(previous, column))
    if latest is not None and prior is not None:
        mark(latest["period_end"], latest["available_at"])
        return _growth(float(latest[column]), float(prior[column]))
    return None


def _valuation(resolved: pd.DataFrame, price: float | None, mark):
    pb = mcap = None
    equity_row = _latest_row(resolved, "equity")
    if equity_row is not None:
        mark(equity_row["period_end"], equity_row["available_at"])
        shares = _shares(equity_row)
        if shares is None:
            share_row = _latest_row(resolved, "shares_outstanding", equity_row["period_end"])
            if share_row is None:
                share_row = _latest_row(resolved, "diluted_shares", equity_row["period_end"])
            shares = _shares(share_row)
        equity = float(equity_row["equity"])
        if price is not None and shares is not None and equity > 0:
            pb = price * shares / equity

    share_row = _latest_row(resolved, "shares_outstanding")
    shares = float(share_row["shares_outstanding"]) if share_row is not None else None
    if share_row is None:
        share_row = _latest_row(resolved, "diluted_shares")
        shares = float(share_row["diluted_shares"]) if share_row is not None else None
    if share_row is not None:
        mark(share_row["period_end"], share_row["available_at"])
    if price is not None and shares is not None:
        mcap = price * shares
    return pb, mcap


def _quarter_window(frame: pd.DataFrame, columns: tuple[str, ...], n: int) -> pd.DataFrame | None:
    usable = _complete(frame, columns)
    if len(usable) < n:
        return None
    window = usable.tail(n)
    if not _span_ok(window, n):
        return None
    return window


def _ttm_pair(frame: pd.DataFrame, columns: tuple[str, ...]):
    usable = _complete(frame, columns)
    if len(usable) < 8:
        return None
    prior = usable.iloc[-8:-4]
    current = usable.iloc[-4:]
    if not _span_ok(prior, 4) or not _span_ok(current, 4):
        return None
    # The two trails have to sit next to each other. A missing quarter in
    # between would make the "prior year" overlap or skip a period.
    gap = (as_timestamp(current["period_end"].iloc[0]) - _end(prior)).days
    if not 45 <= gap <= 140:
        return None
    return prior, current


def _annual_pair(frame: pd.DataFrame, columns: tuple[str, ...]):
    usable = _complete(frame, columns)
    if usable.empty:
        return None, None
    latest = usable.iloc[-1]
    prior = None
    if len(usable) >= 2:
        candidate = usable.iloc[-2]
        gap = (as_timestamp(latest["period_end"]) - as_timestamp(candidate["period_end"])).days
        if 300 <= gap <= 430:
            prior = candidate
    return latest, prior


def _complete(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame
    usable = frame
    for column in columns:
        usable = usable[usable[column].notna()]
    return usable.sort_values("period_end")


def _span_ok(window: pd.DataFrame, n: int) -> bool:
    if len(window) != n:
        return False
    if n <= 1:
        return True
    span = (_end(window) - as_timestamp(window["period_end"].iloc[0])).days
    expected = 91 * (n - 1)
    return expected - 40 <= span <= expected + 45


def _eps(level) -> float | None:
    if isinstance(level, pd.DataFrame):
        shares = level["diluted_shares"]
        if shares.isna().any() or (shares <= 0).any():
            return None
        return float((level["net_income"] / shares).sum())
    shares = float(level["diluted_shares"])
    if shares <= 0:
        return None
    return float(level["net_income"]) / shares


def _total(level, column: str) -> float:
    if isinstance(level, pd.DataFrame):
        return float(level[column].sum())
    return float(level[column])


def _end(level) -> pd.Timestamp:
    if isinstance(level, pd.DataFrame):
        return as_timestamp(level["period_end"].iloc[-1])
    return as_timestamp(level["period_end"])


def _available(level) -> pd.Timestamp:
    if isinstance(level, pd.DataFrame):
        return as_timestamp(level["available_at"].iloc[-1])
    return as_timestamp(level["available_at"])


def _latest_row(frame: pd.DataFrame, column: str, on_or_before=None):
    if frame.empty or column not in frame.columns:
        return None
    usable = frame[frame[column].notna()]
    if on_or_before is not None:
        cutoff = as_timestamp(on_or_before)
        usable = usable[usable["period_end"].map(as_timestamp) <= cutoff]
    if usable.empty:
        return None
    return usable.sort_values(["period_end", "available_at"]).iloc[-1]


def _shares(row) -> float | None:
    if row is None:
        return None
    if pd.notna(row["shares_outstanding"]):
        return float(row["shares_outstanding"])
    if pd.notna(row["diluted_shares"]):
        return float(row["diluted_shares"])
    return None


def _price_as_of(prices: pd.Series | None, as_of: pd.Timestamp) -> float | None:
    if prices is None or len(prices) == 0:
        return None
    values = pd.to_numeric(pd.Series(prices), errors="coerce")
    index = pd.to_datetime(values.index)
    if getattr(index, "tz", None) is not None:
        index = pd.DatetimeIndex([as_timestamp(stamp) for stamp in index])
    else:
        index = pd.DatetimeIndex([as_timestamp(stamp) for stamp in index])
    ordered = pd.Series(values.to_numpy(), index=index).sort_index().dropna()
    known = ordered.loc[:as_of]
    if known.empty:
        return None
    price = float(known.iloc[-1])
    if not math.isfinite(price):
        return None
    return price


def _growth(current: float, prior: float) -> float | None:
    if prior <= 0:
        return None
    return current / prior - 1.0


def _finite(value):
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(number):
        return math.nan
    return number


def _blank_metrics(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "PE": math.nan,
        "PB": math.nan,
        "ROE": math.nan,
        "RevenueGrowth": math.nan,
        "EarningsGrowth": math.nan,
        "MarketCap": math.nan,
        "statement_period_end": None,
        "statement_available_at": None,
    }


def _fatal_fmp(exc: Exception) -> bool:
    if not isinstance(exc, ProviderError):
        return False
    text = str(exc)
    return "rejected the API key" in text or "rate limit" in text


def _warn_once(message: str) -> None:
    if message in _warned:
        return
    _warned.add(message)
    print(message)
