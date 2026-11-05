"""Statement rows shared by the Yahoo and FMP loaders.

A row is one published version of a fiscal period. A later row with the
same symbol, frequency, and period end is a restatement, visible only
once ``available_at`` has been reached.
"""

from __future__ import annotations

import math

import pandas as pd

STATEMENT_COLUMNS = [
    "symbol",
    "period_end",
    "available_at",
    "frequency",
    "revenue",
    "net_income",
    "diluted_shares",
    "equity",
    "shares_outstanding",
]

_NUMBER_COLUMNS = [
    "revenue",
    "net_income",
    "diluted_shares",
    "equity",
    "shares_outstanding",
]

# A quarterly earnings release more than this many days after period end
# is not matched to that period. The fixed reporting lag is used instead.
EARNINGS_MATCH_DAYS = 120


def empty_statements() -> pd.DataFrame:
    return pd.DataFrame(columns=STATEMENT_COLUMNS)


def as_timestamp(value) -> pd.Timestamp | None:
    """Calendar date, or None when ``value`` is not a date.

    Time zones are dropped without shifting the wall-clock date.
    A datetime keeps that calendar day; the clock time is not used.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp.strftime("%Y-%m-%d"))


def reporting_date(period_end, lag_days: int, filing_date=None) -> pd.Timestamp:
    """When a period may be used.

    A filing or earnings date on or after the period end wins. An earlier
    stamp is ignored. Otherwise the statement is assumed public ``lag_days``
    calendar days after the period end.
    """
    period = as_timestamp(period_end)
    if period is None:
        raise ValueError(f"invalid period end: {period_end!r}")
    filed = as_timestamp(filing_date)
    if filed is not None and filed >= period:
        return filed
    return period + pd.Timedelta(days=int(lag_days))


def normalize_statements(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce a statement panel. Rows missing a date or frequency are dropped."""
    if frame is None or frame.empty:
        return empty_statements()

    out = frame.copy()
    for column in STATEMENT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out = out.loc[:, STATEMENT_COLUMNS]
    out["symbol"] = out["symbol"].map(lambda value: "" if value is None else str(value).strip())
    out["period_end"] = out["period_end"].map(as_timestamp)
    out["available_at"] = out["available_at"].map(as_timestamp)
    freq = out["frequency"].astype(str).str.upper().str[0]
    out["frequency"] = freq.where(freq.isin(["Q", "A"]))
    for column in _NUMBER_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("diluted_shares", "shares_outstanding"):
        out.loc[out[column] <= 0, column] = math.nan
    out = out.dropna(subset=["symbol", "period_end", "available_at", "frequency"])
    out = out[out["symbol"] != ""]
    out = out.sort_values(
        ["symbol", "frequency", "period_end", "available_at"],
        kind="mergesort",
    )
    return out.reset_index(drop=True)


def match_earnings_dates(period_ends, earnings_dates) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map each period to the earnings date that announced it.

    Each announcement is given to the latest period end that is strictly
    before it and at most ``EARNINGS_MATCH_DAYS`` earlier. One announcement
    is not reused for an older period.
    """
    periods = sorted({stamp for stamp in (as_timestamp(value) for value in period_ends) if stamp is not None})
    dates = sorted({stamp for stamp in (as_timestamp(value) for value in earnings_dates) if stamp is not None})
    window = pd.Timedelta(days=EARNINGS_MATCH_DAYS)
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for earned in dates:
        eligible = [period for period in periods if period < earned <= period + window]
        if not eligible:
            continue
        period = eligible[-1]
        current = mapping.get(period)
        if current is None or earned < current:
            mapping[period] = earned
    return mapping


def number(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)
