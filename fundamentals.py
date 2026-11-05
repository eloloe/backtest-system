"""Streamlit wrapper around the live Yahoo snapshot.

Research code imports ``backtest_system.fundamentals`` and does not import
Streamlit. ``streamlit run app.py`` imports this module so the snapshot
stays cached for a day.
"""

from _bootstrap import ensure_src

ensure_src()

import streamlit as st

from backtest_system.fundamentals.pit import fetch_fundamentals_pit, reporting_lag_days
from backtest_system.fundamentals.snapshot import fetch_fundamentals as _fetch_fundamentals

__all__ = ["fetch_fundamentals", "fetch_fundamentals_pit", "reporting_lag_days"]


@st.cache_data(ttl=86400)
def fetch_fundamentals(symbols: list):
    """Cached live snapshot. See ``backtest_system.fundamentals.snapshot``."""
    return _fetch_fundamentals(list(symbols))
