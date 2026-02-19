import yfinance as yf
import pandas as pd


def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """從 Yahoo Finance 取得 OHLCV 歷史資料。"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # 移除時區資訊
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()
