"""Long-only backtest: signal at the close, fill at the next bar's open.

Costs depend on the market. Bare tickers use the US schedule. ``.TW`` and
``.TWO`` use Taiwan brokerage plus sell-side securities transaction tax.

``Backtester.run`` walks trade dates, not every session. Between two fills
the share count and the cash balance are constant, so the close-to-close
mark is a slice of the close array. On a book that trades a few dozen times
across tens of thousands of sessions, that mark is much cheaper than a
Python iteration per bar. A book that trades almost every session still
pays a Python call per fill, and is not faster than the bar loop.

Cash, shares, and missing bars
------------------------------
Cash does not earn interest and the book is fully funded, including the
futures placeholder (no margin). A buy spends the cash it can, after fees.
Without ``min_lot`` the share count is fractional (``notional / fill``).
Taiwan names are not rounded to a 1,000-share board lot. With ``min_lot``,
size is floored to that multiple and the leftover cash stays in the account.

Rows whose close is not finite are dropped before the first mark, so they
are neither a signal bar nor a valuation. A non-finite or non-positive open
skips that session's fill. The next session reads the previous kept bar's
signal again; there is no resting order beyond that one-bar lag. If the
signal has flipped before a usable open, nothing is filled.

Dividends are not cashflows. ``dividend_withholding`` is recorded on the
cost schedule and does not move equity.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest_system.engine.costs import (
    DEFAULT_TW_BROKERAGE_MIN,
    DEFAULT_TW_BROKERAGE_RATE,
    DEFAULT_TW_ETF_SELL_TAX,
    DEFAULT_TW_STOCK_SELL_TAX,
    DEFAULT_US_COMMISSION,
    DEFAULT_US_DIVIDEND_WITHHOLDING,
    DEFAULT_US_SLIPPAGE,
    TRADE_COLUMNS,
    MarketCosts,
    _execution_price,
    _fee,
    _size_buy,
    detect_market,
    resolve_costs,
)
from backtest_system.engine.reference import run_long_flat_reference

__all__ = [
    "DEFAULT_TW_BROKERAGE_MIN",
    "DEFAULT_TW_BROKERAGE_RATE",
    "DEFAULT_TW_ETF_SELL_TAX",
    "DEFAULT_TW_STOCK_SELL_TAX",
    "DEFAULT_US_COMMISSION",
    "DEFAULT_US_DIVIDEND_WITHHOLDING",
    "DEFAULT_US_SLIPPAGE",
    "TRADE_COLUMNS",
    "Backtester",
    "MarketCosts",
    "detect_market",
    "resolve_costs",
    "run_long_flat",
    "run_long_flat_reference",
]


def _next_true(mask: np.ndarray) -> np.ndarray:
    """For each i, the smallest j >= i with ``mask[j]``, else ``len(mask)``."""
    n = int(mask.shape[0])
    positions = np.arange(n, dtype=np.int64)
    flagged = np.where(mask, positions, n)
    return np.minimum.accumulate(flagged[::-1])[::-1]


def run_long_flat(
    initial_capital: float,
    costs: MarketCosts,
    index: pd.Index,
    opens: np.ndarray,
    closes: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Same fills as :func:`run_long_flat_reference`, marked in slices."""
    n = int(len(opens))
    equity = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return equity, []

    desired = np.zeros(n, dtype=bool)
    if n > 1:
        desired[1:] = observed[:-1] == 1
    finite_open = np.isfinite(opens)
    positive = opens > 0
    fillable = finite_open & positive
    buy_at = _next_true(desired & fillable)
    sell_at = _next_true(~desired & fillable)

    cash = float(initial_capital)
    shares = 0.0
    multiplier = float(costs.contract_multiplier)
    trades: list[dict] = []
    i = 0
    while i < n:
        if shares <= 0:
            j = int(buy_at[i])
            if j >= n:
                equity[i:] = cash
                break
            fill = _execution_price(float(opens[j]), True, costs.slippage_rate)
            quantity, notional, fee = _size_buy(cash, fill, costs)
            if quantity > 0:
                equity[i:j] = cash
                cash -= notional + fee
                shares = quantity
                equity[j] = cash + shares * float(closes[j]) * multiplier
                trades.append(_row(index[j], "BUY", fill, quantity, notional, fee, costs.market))
            else:
                equity[i : j + 1] = cash
            i = j + 1
            continue

        j = int(sell_at[i])
        if j >= n:
            equity[i:] = cash + shares * closes[i:].astype(np.float64) * multiplier
            break
        if j > i:
            equity[i:j] = cash + shares * closes[i:j].astype(np.float64) * multiplier
        fill = _execution_price(float(opens[j]), False, costs.slippage_rate)
        notional = shares * fill * multiplier
        fee = _fee(notional, costs, "SELL")
        proceeds = notional - fee
        sold = shares
        trades.append(_row(index[j], "SELL", fill, sold, proceeds, fee, costs.market))
        cash += proceeds
        shares = 0.0
        equity[j] = cash
        i = j + 1

    return equity, trades


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


def _prepare(
    data: pd.DataFrame, signals: pd.Series
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray] | None:
    if "Open" not in data.columns or "Close" not in data.columns:
        raise ValueError(
            "data must include Open and Close; "
            "orders fill at the next bar's Open, not the signal bar's Close"
        )
    work = pd.DataFrame(
        {
            "Open": pd.to_numeric(data["Open"], errors="coerce"),
            "Close": pd.to_numeric(data["Close"], errors="coerce"),
        },
        index=data.index,
    )
    work = work.loc[np.isfinite(work["Close"].to_numpy(dtype=float))]
    if work.empty:
        return None
    aligned = signals.reindex(work.index).ffill().fillna(0).astype(int)
    opens = work["Open"].to_numpy(dtype=float)
    closes = work["Close"].to_numpy(dtype=float)
    observed = aligned.to_numpy(dtype=int)
    return work, opens, closes, observed


def _as_frames(
    index: pd.Index,
    equity: np.ndarray,
    observed: np.ndarray,
    closes: np.ndarray,
    trades: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio = pd.DataFrame(
        {
            "portfolio_value": equity,
            "signal": observed,
            "price": closes,
        },
        index=index,
    )
    portfolio["returns"] = portfolio["portfolio_value"].pct_change()
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=TRADE_COLUMNS)
    return portfolio, trades_df


class Backtester:
    """Fill at the next bar's open. One name, long or flat, all-in then all-out.

    A signal observed at the close of bar t is an order for the open of bar
    t+1. Bar t is still marked with the previous position. If t is the last
    bar, the order is not filled and the position is marked at that close.
    There is no same-bar close fill.

    ``commission`` and ``slippage`` are fractions of price / notional and still
    drive the US schedule (and futures, which reuse it). They are ignored for
    Taiwan commission: ``.TW`` / ``.TWO`` pay brokerage and sell-side tax.
    Slippage still shifts the TW fill. Pass ``commission_bps`` or
    ``slippage_bps`` to set those rates in basis points instead.

    Equal-weight books are one ``Backtester`` per name (see
    ``simulate_signals``). Target-weight books share cash and do not use
    this all-in sizer (see ``simulate_target_weights``).
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission: float = DEFAULT_US_COMMISSION,
        slippage: float = DEFAULT_US_SLIPPAGE,
        commission_flat: float = 0.0,
        commission_bps: float | None = None,
        slippage_bps: float | None = None,
        brokerage_rate: float | None = None,
        brokerage_min: float | None = None,
        sell_tax_rate: float | None = None,
        dividend_withholding: float | None = None,
        contract_multiplier: float = 1.0,
        min_lot: float | None = None,
        symbol: str | None = None,
        market: str | None = None,
    ):
        if not math.isfinite(initial_capital) or initial_capital < 0:
            raise ValueError("initial_capital must be a finite number >= 0")
        self.initial_capital = float(initial_capital)
        self.commission = commission
        self.slippage = slippage
        self.commission_flat = commission_flat
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.brokerage_rate = brokerage_rate
        self.brokerage_min = brokerage_min
        self.sell_tax_rate = sell_tax_rate
        self.dividend_withholding = dividend_withholding
        self.contract_multiplier = contract_multiplier
        self.min_lot = min_lot
        self.symbol = symbol
        self.market = market

    def cost_model(
        self, symbol: str | None = None, market: str | None = None
    ) -> MarketCosts:
        """Resolve costs the same way ``run`` will, without simulating."""
        return self._costs(
            self.symbol if symbol is None else symbol,
            self.market if market is None else market,
        )

    def _costs(self, symbol: str | None, market: str | None) -> MarketCosts:
        return resolve_costs(
            symbol=symbol,
            market=market,
            commission=self.commission,
            slippage=self.slippage,
            commission_flat=self.commission_flat,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            brokerage_rate=self.brokerage_rate,
            brokerage_min=self.brokerage_min,
            sell_tax_rate=self.sell_tax_rate,
            dividend_withholding=self.dividend_withholding,
            contract_multiplier=self.contract_multiplier,
            min_lot=self.min_lot,
        )

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str | None = None,
        market: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate one name. See the class docstring for fill rules.

        Returns
        -------
        portfolio : DataFrame
            ``portfolio_value``, ``signal`` (the signal observed on that
            bar, not the position held), ``price`` (close), ``returns``.
        trades : DataFrame
            One row per fill. ``price`` is the open after slippage.
            ``value`` is traded notional on a buy and cash received after
            fees on a sell. ``fee`` is the cash fee for that fill.
        """
        symbol = self.symbol if symbol is None else symbol
        market = self.market if market is None else market
        costs = self._costs(symbol, market)
        prepared = _prepare(data, signals)
        if prepared is None:
            return _empty_result(data)
        work, opens, closes, observed = prepared
        equity, trades = run_long_flat(
            self.initial_capital, costs, work.index, opens, closes, observed
        )
        return _as_frames(work.index, equity, observed, closes, trades)

    def run_reference(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str | None = None,
        market: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Bar-by-bar oracle. ``run`` should match it on the same inputs."""
        symbol = self.symbol if symbol is None else symbol
        market = self.market if market is None else market
        costs = self._costs(symbol, market)
        prepared = _prepare(data, signals)
        if prepared is None:
            return _empty_result(data)
        work, opens, closes, observed = prepared
        equity, trades = run_long_flat_reference(
            self.initial_capital, costs, work.index, opens, closes, observed
        )
        return _as_frames(work.index, equity, observed, closes, trades)


def _empty_result(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No finite closes left. Columns match a normal run; both frames are empty."""
    work = pd.DataFrame(
        {
            "Open": pd.to_numeric(data["Open"], errors="coerce"),
            "Close": pd.to_numeric(data["Close"], errors="coerce"),
        },
        index=data.index,
    )
    work = work.loc[np.isfinite(work["Close"].to_numpy(dtype=float))]
    portfolio = pd.DataFrame(
        {
            "portfolio_value": np.array([], dtype=float),
            "signal": np.array([], dtype=int),
            "price": np.array([], dtype=float),
        },
        index=work.index,
    )
    portfolio["returns"] = portfolio["portfolio_value"].pct_change()
    return portfolio, pd.DataFrame(columns=TRADE_COLUMNS)
