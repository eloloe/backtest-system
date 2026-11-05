"""Optional Financial Modeling Prep statements.

Used only when ``FMP_API_KEY`` is set. Filing timestamps become
``available_at``. The standardized feed can still be rewritten in place
by the vendor, so this is not a restatement archive.
"""

from __future__ import annotations

import os

import pandas as pd

from fundamentals.statements import (
    as_timestamp,
    empty_statements,
    normalize_statements,
    number,
    reporting_date,
)
from providers.base import MissingApiKeyError, ProviderError, request_json

_DEFAULT_BASE = "https://financialmodelingprep.com/stable"


def load_fmp_statements(symbol: str, lag_days: int, api_key: str | None = None) -> pd.DataFrame:
    raw = api_key if api_key is not None else os.environ.get("FMP_API_KEY", "")
    key = raw.strip()
    if not key:
        raise MissingApiKeyError("FMP is selected but FMP_API_KEY is not set.")
    symbol_s = symbol.strip()
    return statements_from_fmp(
        symbol_s,
        income_quarterly=_fetch(key, "income-statement", symbol_s, "quarter"),
        balance_quarterly=_fetch(key, "balance-sheet-statement", symbol_s, "quarter"),
        income_annual=_fetch(key, "income-statement", symbol_s, "annual"),
        balance_annual=_fetch(key, "balance-sheet-statement", symbol_s, "annual"),
        lag_days=lag_days,
    )


def statements_from_fmp(
    symbol: str,
    *,
    income_quarterly=None,
    balance_quarterly=None,
    income_annual=None,
    balance_annual=None,
    lag_days: int,
) -> pd.DataFrame:
    """Merge FMP income and balance payloads. No network."""
    rows: list[dict] = []
    rows.extend(_merge(symbol, "Q", income_quarterly or [], balance_quarterly or [], lag_days))
    rows.extend(_merge(symbol, "A", income_annual or [], balance_annual or [], lag_days))
    if not rows:
        return empty_statements()
    return normalize_statements(pd.DataFrame(rows))


def _fetch(key: str, path: str, symbol: str, period: str) -> list[dict]:
    base = os.environ.get("FMP_BASE_URL", _DEFAULT_BASE).strip().rstrip("/") or _DEFAULT_BASE
    limit = 80 if period == "quarter" else 25
    payload = request_json(
        f"{base}/{path}",
        params={"symbol": symbol, "period": period, "limit": limit, "apikey": key},
        headers=None,
        provider="fmp",
    )
    return _records(payload, key)


def _records(payload, key: str) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        message = payload.get("Error Message") or payload.get("error") or payload.get("message")
        if message:
            text = str(message).replace(key, "[redacted]")
            raise ProviderError(f"fmp: {text}")
    raise ProviderError("fmp returned an unexpected payload.")


def _merge(symbol: str, frequency: str, income_items, balance_items, lag_days: int) -> list[dict]:
    income_by = _by_period(income_items)
    balance_by = _by_period(balance_items)
    rows = []
    for period in sorted(set(income_by) | set(balance_by)):
        inc = income_by.get(period, {})
        bal = balance_by.get(period, {})
        available = _available(inc, bal, period, lag_days)
        net_income = _first(inc, "netIncome", "bottomLineNetIncome")
        diluted = _first(
            inc,
            "weightedAverageShsOutDil",
            "weightedAverageDilutedSharesOutstanding",
        )
        basic = _first(
            inc,
            "weightedAverageShsOut",
            "weightedAverageBasicSharesOutstanding",
        )
        eps = _first(inc, "epsDiluted", "epsdiluted", "dilutedEarningsPerShare", "eps")
        if diluted is None and eps not in (None, 0.0) and net_income is not None:
            diluted = net_income / eps
        shares_out = _first(bal, "commonStockSharesOutstanding", "ordinarySharesNumber")
        if shares_out is None:
            shares_out = basic
        rows.append(
            {
                "symbol": symbol,
                "period_end": period,
                "available_at": available,
                "frequency": frequency,
                "revenue": _first(inc, "revenue"),
                "net_income": net_income,
                "diluted_shares": diluted,
                "equity": _first(bal, "totalStockholdersEquity", "totalEquity"),
                "shares_outstanding": shares_out,
            }
        )
    return rows


def _by_period(items) -> dict[pd.Timestamp, dict]:
    found = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        period = as_timestamp(item.get("date"))
        if period is not None:
            found[period] = item
    return found


def _available(income: dict, balance: dict, period: pd.Timestamp, lag_days: int) -> pd.Timestamp:
    dates = []
    if income:
        dates.append(reporting_date(period, lag_days, _filed(income)))
    if balance:
        dates.append(reporting_date(period, lag_days, _filed(balance)))
    return max(dates)


def _filed(item: dict):
    for key in ("acceptedDate", "accepted_date", "filingDate", "fillingDate", "filing_date"):
        if item.get(key):
            return item[key]
    return None


def _first(item: dict, *keys) -> float | None:
    if not item:
        return None
    for key in keys:
        if key in item:
            value = number(item.get(key))
            if value is not None:
                return value
    return None
