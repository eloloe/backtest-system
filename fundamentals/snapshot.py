"""Live Yahoo ``ticker.info`` snapshot.

These are today's ratios. They are not a historical screen.
"""

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=86400)
def fetch_fundamentals(symbols: list) -> pd.DataFrame:
    """
    Fetch the current Yahoo snapshot for ``symbols``.

    This reads ``ticker.info`` as of the request, not as of a past date.
    Historical screens should use ``fetch_pit_fundamentals`` instead.

    Columns match the point-in-time frame. ``source`` is ``live_snapshot``.
    """
    data = []
    for sym in symbols:
        row = {
            "symbol": sym,
            "PE": None,
            "PB": None,
            "ROE": None,
            "RevenueGrowth": None,
            "EarningsGrowth": None,
            "MarketCap": None,
        }
        try:
            info = yf.Ticker(sym).info or {}
            row["PE"] = info.get("trailingPE")
            row["PB"] = info.get("priceToBook")
            row["ROE"] = info.get("returnOnEquity")
            row["RevenueGrowth"] = info.get("revenueGrowth")
            row["EarningsGrowth"] = info.get("earningsGrowth")
            row["MarketCap"] = info.get("marketCap")
        except Exception as exc:
            print(f"Error fetching fundamental data for {sym}: {exc}")
        row["as_of"] = None
        row["source"] = "live_snapshot"
        row["statement_period_end"] = None
        row["statement_available_at"] = None
        row["lag_days"] = None
        data.append(row)

    df = pd.DataFrame(data)
    for col in ["PE", "PB", "ROE", "RevenueGrowth", "EarningsGrowth", "MarketCap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
