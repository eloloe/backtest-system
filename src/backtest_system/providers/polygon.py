"""Polygon daily aggregates for US equities and ETFs."""

from __future__ import annotations

import os
from urllib.parse import quote

import pandas as pd

from backtest_system.providers.base import (
    MissingApiKeyError,
    PriceProvider,
    ProviderError,
    empty_ohlcv,
    finalize_ohlcv,
    request_json,
    request_window,
)

_AGGS_URL = (
    "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
)
_PAGE_LIMIT = 50_000


class PolygonProvider(PriceProvider):
    """Polygon grouped daily bars.

    Reads ``POLYGON_API_KEY``. ``adjusted=true`` corrects splits only,
    not dividends, so closes will not match Yahoo or Tiingo
    total-return series. Symbols are sent through unchanged; this
    endpoint is for US listings.
    """

    name = "polygon"

    def __init__(self, api_key: str | None = None):
        raw = api_key if api_key is not None else os.environ.get("POLYGON_API_KEY", "")
        key = raw.strip()
        if not key:
            raise MissingApiKeyError(
                "Polygon is selected but POLYGON_API_KEY is not set."
            )
        self._api_key = key

    def fetch_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        window = request_window(start, end)
        if window is None:
            return empty_ohlcv()
        start_ts, end_ts = window

        ticker = quote(symbol.strip(), safe=".-")
        if not ticker:
            raise ProviderError("symbol is empty")

        url = _AGGS_URL.format(
            ticker=ticker,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
        )
        payload = request_json(
            url,
            params={
                "adjusted": "true",
                "sort": "asc",
                "limit": _PAGE_LIMIT,
            },
            headers={"Authorization": f"Bearer {self._api_key}"},
            provider="Polygon",
        )
        if not payload:
            return empty_ohlcv()
        if not isinstance(payload, dict):
            raise ProviderError("Polygon returned an unexpected payload.")

        status = str(payload.get("status", "")).upper()
        if status not in {"OK", "DELAYED"}:
            message = payload.get("error") or payload.get("message") or status or "request failed"
            raise ProviderError(f"Polygon: {message}")

        results = payload.get("results") or []
        if not results:
            return empty_ohlcv()
        # The API default limit is 5000, which cuts off a long daily history
        # without an error. Ask for the max page, and refuse to return a
        # truncated series if that is still not enough.
        if payload.get("next_url") or payload.get("resultsCount", 0) >= _PAGE_LIMIT:
            raise ProviderError(
                "Polygon returned the 50,000 bar page limit; narrow the date range."
            )

        frame = pd.DataFrame(results)
        missing = [col for col in ("o", "h", "l", "c", "v", "t") if col not in frame.columns]
        if missing:
            raise ProviderError(f"Polygon response missing {', '.join(missing)}")

        parsed = pd.to_datetime(frame["t"], unit="ms", utc=True, errors="coerce")
        if parsed.isna().any():
            raise ProviderError("Polygon returned a timestamp that could not be parsed.")
        # ``t`` is the start of the daily window: midnight US/Eastern,
        # stored as UTC (05:00 UTC in January, 04:00 UTC in July).
        local = parsed.dt.tz_convert("America/New_York")
        index = pd.DatetimeIndex(local.dt.strftime("%Y-%m-%d"), name="Date")
        # Values only: these columns are still indexed 0..n, and aligning
        # them onto the session dates would null out every row.
        ohlcv = pd.DataFrame(
            {
                "Open": frame["o"].to_numpy(),
                "High": frame["h"].to_numpy(),
                "Low": frame["l"].to_numpy(),
                "Close": frame["c"].to_numpy(),
                "Volume": frame["v"].to_numpy(),
            },
            index=index,
        )
        return finalize_ohlcv(ohlcv, start, end)
