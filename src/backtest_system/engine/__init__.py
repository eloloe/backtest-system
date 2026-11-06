"""Fills and books.

Signal on the close, fill at the next open. :class:`Backtester` is one
long-or-flat name. :func:`simulate_signals` splits cash into equal-weight
sleeves. :func:`simulate_target_weights` is one shared cash account.

:func:`load_crosscheck` compares an equity curve to a replay of the trade
list, or to a Nautilus cash account on the same orders. Nautilus is not
imported from here.
"""

from backtest_system.engine.backtester import (
    TRADE_COLUMNS,
    Backtester,
    MarketCosts,
    detect_market,
    resolve_costs,
)
from backtest_system.engine.crosscheck import (
    NAUTILUS_LIMITATIONS,
    CrossCheckReport,
    CrossCheckRun,
    NautilusCrossCheck,
    OptionalEngineError,
    ReferenceReplay,
    compare_equity,
    load_crosscheck,
    nautilus_atol,
    replay_equity,
    run_crosscheck,
)
from backtest_system.engine.portfolio import (
    BookResult,
    combine_books,
    run_equal_weight,
    run_sleeve,
    run_weight_series,
    simulate_signals,
    simulate_target_weights,
    simulate_target_weights_iterative,
)

__all__ = [
    "TRADE_COLUMNS",
    "Backtester",
    "BookResult",
    "NAUTILUS_LIMITATIONS",
    "CrossCheckReport",
    "CrossCheckRun",
    "MarketCosts",
    "NautilusCrossCheck",
    "OptionalEngineError",
    "ReferenceReplay",
    "combine_books",
    "compare_equity",
    "detect_market",
    "load_crosscheck",
    "nautilus_atol",
    "replay_equity",
    "run_crosscheck",
    "resolve_costs",
    "run_equal_weight",
    "run_sleeve",
    "run_weight_series",
    "simulate_signals",
    "simulate_target_weights",
    "simulate_target_weights_iterative",
]
