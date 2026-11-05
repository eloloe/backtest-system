"""Bar-by-bar definition of the long-or-flat book.

``Backtester.run`` does not call this. Tests compare the two on synthetic
bars. The loop is the economics: a signal on the close of bar t is an order
for the open of bar t+1, and a bar is marked at the close after that open.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest_system.engine.costs import (
    MarketCosts,
    _execution_price,
    _fee,
    _size_buy,
)


def run_long_flat_reference(
    initial_capital: float,
    costs: MarketCosts,
    index: pd.Index,
    opens: np.ndarray,
    closes: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """One Python iteration per bar. Same fills as the original engine."""
    cash = float(initial_capital)
    shares = 0.0
    portfolio_values = np.empty(len(opens), dtype=np.float64)
    trades: list[dict] = []

    for i in range(len(opens)):
        if i > 0:
            target_long = int(observed[i - 1]) == 1
            holding = shares > 0
            open_price = float(opens[i])
            can_fill = math.isfinite(open_price) and open_price > 0
            if can_fill and target_long and not holding:
                fill = _execution_price(open_price, True, costs.slippage_rate)
                quantity, notional, fee = _size_buy(cash, fill, costs)
                if quantity > 0:
                    cash -= notional + fee
                    shares = quantity
                    trades.append(
                        _row(index[i], "BUY", fill, quantity, notional, fee, costs.market)
                    )
            elif can_fill and holding and not target_long:
                fill = _execution_price(open_price, False, costs.slippage_rate)
                notional = shares * fill * costs.contract_multiplier
                fee = _fee(notional, costs, "SELL")
                proceeds = notional - fee
                trades.append(
                    _row(index[i], "SELL", fill, shares, proceeds, fee, costs.market)
                )
                cash += proceeds
                shares = 0.0

        marked = shares * float(closes[i]) * costs.contract_multiplier
        portfolio_values[i] = cash + marked

    return portfolio_values, trades


def _row(date, action: str, price: float, shares: float, value: float, fee: float, market: str) -> dict:
    return {
        "date": date,
        "action": action,
        "price": price,
        "shares": shares,
        "value": value,
        "fee": fee,
        "market": market,
    }
