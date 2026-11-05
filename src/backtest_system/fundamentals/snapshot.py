"""Today's Yahoo ``ticker.info`` snapshot.

These numbers are not what an investor could have known on a past date.
Historical screens use :func:`backtest_system.fundamentals.pit.fetch_fundamentals_pit`.
The Streamlit UI caches :func:`fetch_fundamentals` in the root ``fundamentals.py``
wrapper. This module does not import Streamlit.
"""

import pandas as pd
import yfinance as yf

__all__ = ["fetch_fundamentals"]


def fetch_fundamentals(symbols: list) -> pd.DataFrame:
    """Live Yahoo ``ticker.info`` snapshot.

    These are today's PE, PB, ROE, growth, and market cap. They are not
    what an investor could have known on a past date. Historical screens
    must use ``fetch_fundamentals_pit`` instead. The price-provider switch
    does not cover this snapshot; it stays on Yahoo.
    """
    data = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info

            data.append({
                "symbol": sym,
                "PE": info.get("trailingPE", None),
                "PB": info.get("priceToBook", None),
                "ROE": info.get("returnOnEquity", None),
                "RevenueGrowth": info.get("revenueGrowth", None),
                "EarningsGrowth": info.get("earningsGrowth", None),
                "MarketCap": info.get("marketCap", None)
            })
        except Exception as e:
            print(f"Error fetching fundamental data for {sym}: {e}")
            data.append({
                "symbol": sym,
                "PE": None,
                "PB": None,
                "ROE": None,
                "RevenueGrowth": None,
                "EarningsGrowth": None,
                "MarketCap": None
            })

    df = pd.DataFrame(data)
    for col in ["PE", "PB", "ROE", "RevenueGrowth", "EarningsGrowth", "MarketCap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
