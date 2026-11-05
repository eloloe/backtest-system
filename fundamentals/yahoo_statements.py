"""Build statement rows from Yahoo period financials.

Yahoo publishes one figure per period (the latest figure it has today),
and usually no filing date. Availability is the matched earnings date
when one exists, otherwise period end plus the reporting lag.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from fundamentals.statements import (
    as_timestamp,
    empty_statements,
    match_earnings_dates,
    normalize_statements,
    number,
    reporting_date,
)

_INCOME_FIELDS = {
    "revenue": ("total revenue", "operating revenue", "revenue"),
    "net_income": (
        "net income common stockholders",
        "net income",
        "net income including noncontrolling interests",
        "net income continuous operations",
        "net income from continuing operations",
    ),
    "diluted_shares": ("diluted average shares", "basic average shares"),
    "diluted_eps": ("diluted eps", "basic eps"),
}
_BALANCE_FIELDS = {
    "equity": (
        "stockholders equity",
        "common stock equity",
        "total equity gross minority interest",
        "total equity",
    ),
    "shares_outstanding": (
        "ordinary shares number",
        "share issued",
        "common stock shares outstanding",
    ),
}


def load_yahoo_statements(symbol: str, lag_days: int) -> pd.DataFrame:
    """Download Yahoo statements. Network or parse errors become an empty panel."""
    try:
        ticker = yf.Ticker(symbol.strip())
        return statements_from_yahoo_frames(
            symbol.strip(),
            income_quarterly=_read_frame(ticker, "quarterly_income_stmt", "quarterly_financials"),
            income_annual=_read_frame(ticker, "income_stmt", "financials"),
            balance_quarterly=_read_frame(ticker, "quarterly_balance_sheet", "quarterly_balancesheet"),
            balance_annual=_read_frame(ticker, "balance_sheet", "balancesheet"),
            earnings_dates=_earnings_frame(ticker),
            lag_days=lag_days,
        )
    except Exception as exc:
        print(f"Error fetching Yahoo statements for {symbol}: {exc}")
        return empty_statements()


def statements_from_yahoo_frames(
    symbol: str,
    *,
    income_quarterly=None,
    income_annual=None,
    balance_quarterly=None,
    balance_annual=None,
    earnings_dates=None,
    lag_days: int,
) -> pd.DataFrame:
    """Turn Yahoo-shaped frames into statement rows. No network."""
    announcements = published_earnings_dates(earnings_dates)
    rows: list[dict] = []
    rows.extend(
        _rows_for_frequency(
            symbol,
            "Q",
            income_quarterly,
            balance_quarterly,
            announcements,
            lag_days,
        )
    )
    rows.extend(
        _rows_for_frequency(
            symbol,
            "A",
            income_annual,
            balance_annual,
            announcements,
            lag_days,
        )
    )
    if not rows:
        return empty_statements()
    return normalize_statements(pd.DataFrame(rows))


def published_earnings_dates(frame, today=None) -> list[pd.Timestamp]:
    """Earnings dates that already have a reported EPS, on or before ``today``."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    cutoff = as_timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    if cutoff is None:
        return []
    keep = pd.Series(True, index=frame.index)
    for name in ("Reported EPS", "ReportedEPS"):
        if name in frame.columns:
            keep = pd.to_numeric(frame[name], errors="coerce").notna()
            break
    dates = []
    for stamp, reported in zip(frame.index, keep.to_numpy()):
        if not reported:
            continue
        parsed = as_timestamp(stamp)
        if parsed is not None and parsed <= cutoff:
            dates.append(parsed)
    return dates


def _rows_for_frequency(symbol, frequency, income, balance, announcements, lag_days) -> list[dict]:
    income_rows = _extract(income, _INCOME_FIELDS)
    balance_rows = _extract(balance, _BALANCE_FIELDS)
    periods = sorted(set(income_rows) | set(balance_rows))
    if not periods:
        return []
    matched = match_earnings_dates(periods, announcements)
    rows = []
    for period in periods:
        inc = income_rows.get(period, {})
        bal = balance_rows.get(period, {})
        net_income = inc.get("net_income")
        diluted = inc.get("diluted_shares")
        eps = inc.get("diluted_eps")
        if diluted is None and eps not in (None, 0.0) and net_income is not None:
            diluted = net_income / eps
        rows.append(
            {
                "symbol": symbol,
                "period_end": period,
                "available_at": reporting_date(period, lag_days, matched.get(period)),
                "frequency": frequency,
                "revenue": inc.get("revenue"),
                "net_income": net_income,
                "diluted_shares": diluted,
                "equity": bal.get("equity"),
                "shares_outstanding": bal.get("shares_outstanding"),
            }
        )
    return rows


def _extract(frame, field_aliases: dict[str, tuple[str, ...]]) -> dict[pd.Timestamp, dict]:
    oriented = _orient(frame)
    if oriented.empty:
        return {}
    labels = {_norm_label(label): label for label in oriented.index}
    chosen = {}
    for field, aliases in field_aliases.items():
        for alias in aliases:
            label = labels.get(alias)
            if label is not None:
                chosen[field] = label
                break
    records: dict[pd.Timestamp, dict] = {}
    for column in oriented.columns:
        period = column if isinstance(column, pd.Timestamp) else as_timestamp(column)
        if period is None:
            continue
        values = {field: number(oriented.at[label, column]) for field, label in chosen.items()}
        if any(value is not None for value in values.values()):
            records[period] = values
    return records


def _orient(frame) -> pd.DataFrame:
    """Return line items on the index and period ends as columns."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out = _collapse_multiindex(out)
    if _mostly_dates(out.index) and not _mostly_dates(out.columns):
        out = out.T
    parsed_columns = []
    keep = []
    for column in out.columns:
        stamp = as_timestamp(column)
        if stamp is None:
            continue
        parsed_columns.append(stamp)
        keep.append(column)
    if not keep:
        return pd.DataFrame()
    out = out.loc[:, keep]
    out.columns = parsed_columns
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    return out


def _collapse_multiindex(frame: pd.DataFrame) -> pd.DataFrame:
    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level)
        if _mostly_dates(values):
            collapsed = frame.copy()
            collapsed.columns = values
            return collapsed
    collapsed = frame.copy()
    collapsed.columns = frame.columns.get_level_values(0)
    return collapsed


def _mostly_dates(values) -> bool:
    parsed = [_is_date(value) for value in list(values)[:12]]
    if not parsed:
        return False
    return sum(parsed) / len(parsed) >= 0.5


def _is_date(value) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False
    return as_timestamp(value) is not None


def _norm_label(value) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _read_frame(ticker, *names):
    for name in names:
        try:
            value = getattr(ticker, name)
        except Exception:
            continue
        if isinstance(value, pd.DataFrame) and not value.empty:
            return value
    return None


def _earnings_frame(ticker):
    getter = getattr(ticker, "get_earnings_dates", None)
    if callable(getter):
        try:
            frame = getter(limit=60)
        except TypeError:
            try:
                frame = getter()
            except Exception:
                frame = None
        except Exception:
            frame = None
        if isinstance(frame, pd.DataFrame):
            return frame
    try:
        frame = ticker.earnings_dates
    except Exception:
        return None
    if isinstance(frame, pd.DataFrame):
        return frame
    return None
