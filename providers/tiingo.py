"""Tiingo end-of-day prices for US equities and ETFs."""

from __future__ import annotations

import os
from urllib.parse import quote

import pandas as pd

from providers.base import (
    MissingApiKeyError,
    PriceProvider,
    ProviderError,
    empty_ohlcv,
    finalize_ohlcv,
    request_json,
    request_window,
)

_PRICES_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
_ADJ_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


class TiingoProvider(PriceProvider):
    """Tiingo daily bars.

    Reads ``TIINGO_API_KEY``. Adjusted fields are split- and
    dividend-adjusted, which is the closest public match to Yahoo
    ``auto_adjust=True``. Symbols are sent through unchanged; this
    endpoint is for US listings, not ``.TW`` or ``BTC-USD``.
    """

    name = "tiingo"

    def __init__(self, api_key: str | None = None):
        raw = api_key if api_key is not None else os.environ.get("TIINGO_API_KEY", "")
        key = raw.strip()
        if not key:
            raise MissingApiKeyError(
                "Tiingo is selected but TIINGO_API_KEY is not set."
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

        payload = request_json(
            _PRICES_URL.format(ticker=ticker),
            params={
                "startDate": start_ts.strftime("%Y-%m-%d"),
                # Tiingo's endDate is inclusive. ``finalize_ohlcv`` drops
                # the exclusive end date if the API returns it.
                "endDate": end_ts.strftime("%Y-%m-%d"),
            },
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            },
            provider="Tiingo",
        )
        if not payload:
            return empty_ohlcv()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message") or "unexpected response"
            raise ProviderError(f"Tiingo: {detail}")
        if not isinstance(payload, list):
            raise ProviderError("Tiingo returned an unexpected payload.")

        frame = pd.DataFrame(payload)
        missing = [col for col in ("date", *_ADJ_COLUMNS) if col not in frame.columns]
        if missing:
            raise ProviderError(f"Tiingo response missing {', '.join(missing)}")

        parsed = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        if parsed.isna().any():
            raise ProviderError("Tiingo returned a date that could not be parsed.")
        # ``date`` is midnight UTC on the session date. Converting to
        # US/Eastern would move the bar onto the previous calendar day.
        index = pd.DatetimeIndex(parsed.dt.strftime("%Y-%m-%d"), name="Date")
        # Values only: these columns are still indexed 0..n, and aligning
        # them onto the session dates would null out every row.
        ohlcv = pd.DataFrame(
            {
                "Open": frame["adjOpen"].to_numpy(),
                "High": frame["adjHigh"].to_numpy(),
                "Low": frame["adjLow"].to_numpy(),
                "Close": frame["adjClose"].to_numpy(),
                "Volume": frame["adjVolume"].to_numpy(),
            },
            index=index,
        )
        return finalize_ohlcv(ohlcv, start, end)
