import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(ttl=86400)
def fetch_fundamentals(symbols: list) -> pd.DataFrame:
    """
    Fetch fundamental data for a list of stock symbols.
    Returns a DataFrame with columns:
    symbol, PE, PB, ROE, RevenueGrowth, EarningsGrowth, MarketCap
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

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Optional: Fill empty numerical columns with NaN instead of None
    for col in ["PE", "PB", "ROE", "RevenueGrowth", "EarningsGrowth", "MarketCap"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
