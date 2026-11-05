"""Yahoo Finance daily bars."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from backtest_system.providers.base import PriceProvider, empty_ohlcv, finalize_ohlcv, request_window


class YahooProvider(PriceProvider):
    """Yahoo Finance via yfinance.

    ``auto_adjust=True`` rewrites Open/High/Low/Close for splits and
    dividends, and Volume for splits. That matches the previous
    ``data_fetcher.fetch_data`` behavior. No API key is required.
    """

    name = "yahoo"

    def fetch_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        if request_window(start, end) is None:
            return empty_ohlcv()

        # yfinance treats ``end`` as exclusive, same as this interface.
        history = yf.Ticker(symbol.strip()).history(
            start=start,
            end=end,
            auto_adjust=True,
        )
        return finalize_ohlcv(history, start, end)
