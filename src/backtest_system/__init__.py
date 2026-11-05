"""Research backtest package.

``providers`` fetches prices, ``strategies`` builds signals and books,
``engine`` fills them, ``research`` runs walk-forward and the report,
``fundamentals`` is the point-in-time screen. The Streamlit UI stays in
the repository-root ``app.py`` and is not imported here.
"""

__version__ = "0.4.0"
