"""
factor_engine.py
─────────────────────────────────────────────────────────────
Modular factor scoring system for the backtest platform.

Each factor is registered in FACTOR_REGISTRY with its metadata.
Factors can be freely enabled/disabled and weighted by the user.
─────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st


# ── Factor Registry ─────────────────────────────────────────
# Each entry: { col: str, label: str, higher_is_better: bool }
#   col              → column name in the merged factor DataFrame
#   label            → display name shown in UI
#   higher_is_better → True = higher raw value = better score
#                      False = lower raw value = better score (inverted z-score)

FACTOR_REGISTRY: dict[str, dict] = {
    "roe": {
        "col": "ROE",
        "label": "ROE (股東權益報酬率)",
        "higher_is_better": True,
    },
    "pe": {
        "col": "PE",
        "label": "PE (本益比)",
        "higher_is_better": False,
    },
    "pb": {
        "col": "PB",
        "label": "PB (股價淨值比)",
        "higher_is_better": False,
    },
    "rev_growth": {
        "col": "RevenueGrowth",
        "label": "營收成長率",
        "higher_is_better": True,
    },
    "earnings_growth": {
        "col": "EarningsGrowth",
        "label": "盈餘成長率",
        "higher_is_better": True,
    },
    "momentum": {
        "col": "Momentum",
        "label": "12 個月動能 (Momentum)",
        "higher_is_better": True,
    },
    "volatility": {
        "col": "Volatility",
        "label": "年化波動率 (Volatility)",
        "higher_is_better": False,
    },
}


# ── Price-based Factor Computation ──────────────────────────

@st.cache_data(ttl=86400)
def fetch_price_factors(symbols: tuple, start: str, end: str) -> pd.DataFrame:
    """
    Compute Momentum (12-month total return) and annualised Volatility
    for each symbol using historical price data.

    Parameters
    ----------
    symbols : tuple  (hashable so Streamlit cache works)
    start, end : str  ISO date strings

    Returns
    -------
    DataFrame with columns: symbol, Momentum, Volatility
    """
    rows = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start, end=end, auto_adjust=True)
            if hist.empty or len(hist) < 20:
                raise ValueError("insufficient data")

            close = hist["Close"].dropna()
            momentum = (close.iloc[-1] / close.iloc[0]) - 1.0
            daily_ret = close.pct_change().dropna()
            volatility = daily_ret.std() * np.sqrt(252)

            rows.append({"symbol": sym, "Momentum": momentum, "Volatility": volatility})
        except Exception:
            rows.append({"symbol": sym, "Momentum": np.nan, "Volatility": np.nan})

    df = pd.DataFrame(rows)
    for col in ["Momentum", "Volatility"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Normalisation ────────────────────────────────────────────

def normalize_zscore(series: pd.Series) -> pd.Series:
    """
    Standard Z-score: (x - mean) / std
    NaN values are preserved; they will be treated as 0 in scoring.
    """
    filled = series.fillna(series.median())
    std = filled.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (filled - filled.mean()) / std


def normalize_rank(series: pd.Series) -> pd.Series:
    """
    Percentile rank in [0, 1].
    NaN values are assigned the median rank (0.5).
    """
    ranked = series.rank(pct=True, na_option="keep")
    ranked = ranked.fillna(0.5)
    return ranked


# ── Composite Scoring ─────────────────────────────────────────

def compute_composite_score(
    df: pd.DataFrame,
    active_factors: dict[str, float],
    method: str = "zscore",
) -> pd.DataFrame:
    """
    Compute a composite factor score for each stock.

    Parameters
    ----------
    df             : DataFrame containing all factor columns (from merged fundamentals + price factors)
    active_factors : {factor_key: weight}  — only factors with weight > 0 are included
    method         : "zscore" | "rank"

    Returns
    -------
    DataFrame with an additional "Composite_Score" column and individual "z_<factor>" columns.
    """
    scored = df.copy()

    total_weight = sum(active_factors.values())
    if total_weight == 0:
        scored["Composite_Score"] = 0.0
        return scored

    composite = pd.Series(0.0, index=df.index)

    for factor_key, weight in active_factors.items():
        if weight == 0:
            continue
        meta = FACTOR_REGISTRY.get(factor_key)
        if meta is None:
            continue
        col = meta["col"]
        if col not in df.columns:
            continue

        # Normalise
        if method == "rank":
            norm = normalize_rank(df[col])
        else:
            norm = normalize_zscore(df[col])

        # Invert if lower raw value is better
        if not meta["higher_is_better"]:
            norm = -norm

        z_col = f"z_{factor_key}"
        scored[z_col] = norm

        composite += norm * (weight / total_weight)

    scored["Composite_Score"] = composite
    return scored


# ── Ranking ──────────────────────────────────────────────────

def rank_and_select(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Sort by Composite_Score descending, return top N rows.
    """
    if "Composite_Score" not in df.columns:
        return df.head(top_n)
    return df.sort_values("Composite_Score", ascending=False).head(top_n).reset_index(drop=True)
