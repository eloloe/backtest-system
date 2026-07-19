"""
screener.py
─────────────────────────────────────────────────────────────
Stock filtering and factor-based scoring utilities.

filter_stocks()      — hard threshold filters (unchanged)
apply_factor_score() — new: composite factor scoring via factor_engine
─────────────────────────────────────────────────────────────
"""

import pandas as pd

from factor_engine import compute_composite_score, fetch_price_factors, rank_and_select


# ── Hard-threshold Filter (unchanged) ───────────────────────

def filter_stocks(
    df: pd.DataFrame,
    pe_max: float = None,
    roe_min: float = None,
    rev_growth_min: float = None,
    mcap_min: float = None,
    mcap_max: float = None,
) -> pd.DataFrame:
    """
    Filter stocks by hard thresholds using AND logic.
    Stocks with missing data for a filtered column are excluded.
    """
    filtered_df = df.copy()

    if pe_max is not None:
        filtered_df = filtered_df[filtered_df["PE"].notna() & (filtered_df["PE"] < pe_max)]

    if roe_min is not None:
        filtered_df = filtered_df[filtered_df["ROE"].notna() & (filtered_df["ROE"] > roe_min)]

    if rev_growth_min is not None:
        filtered_df = filtered_df[
            filtered_df["RevenueGrowth"].notna()
            & (filtered_df["RevenueGrowth"] > rev_growth_min)
        ]

    if mcap_min is not None:
        filtered_df = filtered_df[
            filtered_df["MarketCap"].notna() & (filtered_df["MarketCap"] >= mcap_min)
        ]

    if mcap_max is not None:
        filtered_df = filtered_df[
            filtered_df["MarketCap"].notna() & (filtered_df["MarketCap"] <= mcap_max)
        ]

    return filtered_df


# ── Factor Score Integration ─────────────────────────────────

def apply_factor_score(
    df_fundamentals: pd.DataFrame,
    start: str,
    end: str,
    active_factors: dict,
    top_n: int,
    method: str = "zscore",
) -> pd.DataFrame:
    """
    Merge fundamental data with price-based factors, compute composite
    score, and return the top N ranked stocks.

    Parameters
    ----------
    df_fundamentals : output of fetch_fundamentals()
    start, end      : ISO date strings for price-factor calculation
    active_factors  : {factor_key: weight}
    top_n           : number of stocks to select
    method          : "zscore" | "rank"

    Returns
    -------
    DataFrame of top N stocks with factor z-scores and Composite_Score
    """
    # Determine which price-based factors are active
    price_factor_keys = {"momentum", "volatility"}
    needs_price = any(k in active_factors for k in price_factor_keys)

    merged = df_fundamentals.copy()

    if needs_price and not df_fundamentals.empty:
        symbols_tuple = tuple(df_fundamentals["symbol"].tolist())
        df_price = fetch_price_factors(symbols_tuple, start, end)
        merged = merged.merge(df_price, on="symbol", how="left")

    # Compute composite score
    scored = compute_composite_score(merged, active_factors, method=method)

    # Rank and select top N
    top = rank_and_select(scored, top_n)
    return top
