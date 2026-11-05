"""Shared types and OHLCV normalization for price providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import requests

HTTP_TIMEOUT = 30

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class ProviderError(Exception):
    """A price request failed or the payload could not be used."""


class DataProviderConfigError(ProviderError):
    """The selected provider is misconfigured. Callers should not swallow this."""


class MissingApiKeyError(DataProviderConfigError):
    """The selected provider needs an API key and none is set."""


class PriceProvider(ABC):
    """Daily OHLCV source.

    ``end`` is exclusive, matching ``yfinance.Ticker.history``.
    The returned index is timezone-naive.
    """

    name: str

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return Open/High/Low/Close/Volume indexed by session date."""


def empty_ohlcv() -> pd.DataFrame:
    frame = pd.DataFrame(columns=OHLCV_COLUMNS)
    frame.index = pd.DatetimeIndex([], name="Date")
    return frame


def parse_bound(value: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"invalid date: {value!r}") from exc
    if pd.isna(stamp):
        raise ProviderError(f"invalid date: {value!r}")
    if stamp.tzinfo is not None:
        # Keep the wall-clock date the caller wrote down.
        stamp = pd.Timestamp(stamp.strftime("%Y-%m-%d %H:%M:%S"))
    return stamp


def request_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return ``[start, end)`` or None when the window is empty."""
    start_ts = parse_bound(start)
    end_ts = parse_bound(end)
    if start_ts >= end_ts:
        return None
    return start_ts, end_ts


def drop_timezone(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Drop tz info without moving the calendar date.

    ``tz_convert(None)`` rewrites stamps as UTC, which can push a
    midnight exchange bar onto the previous or next date.
    """
    if index.tz is None:
        return index
    return pd.DatetimeIndex(index.strftime("%Y-%m-%d %H:%M:%S"), name=index.name)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_ohlcv()

    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ProviderError("price data is missing a datetime index")

    frame.index = drop_timezone(frame.index)
    missing = [col for col in OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        raise ProviderError(f"price data missing columns: {', '.join(missing)}")

    frame = frame.loc[:, OHLCV_COLUMNS]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all")
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "Date"
    return frame


def finalize_ohlcv(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Normalize and clip to the half-open window ``[start, end)``."""
    window = request_window(start, end)
    frame = normalize_ohlcv(df)
    if window is None or frame.empty:
        return empty_ohlcv()
    start_ts, end_ts = window
    clipped = frame[(frame.index >= start_ts) & (frame.index < end_ts)]
    if clipped.empty:
        return empty_ohlcv()
    return clipped.copy()


def request_json(
    url: str,
    *,
    params: dict | None,
    headers: dict | None,
    provider: str,
):
    """GET JSON. Returns None on HTTP 404. Never includes the API key in errors."""
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ProviderError(f"{provider} request failed: {exc}") from exc

    if response.status_code == 404:
        return None
    if response.status_code in (401, 403):
        raise ProviderError(
            f"{provider} rejected the API key (HTTP {response.status_code})."
        )
    if response.status_code == 429:
        raise ProviderError(f"{provider} rate limit hit (HTTP 429).")
    if response.status_code >= 400:
        detail = " ".join(response.text.split())[:200]
        raise ProviderError(f"{provider} HTTP {response.status_code}: {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(f"{provider} returned non-JSON data.") from exc
