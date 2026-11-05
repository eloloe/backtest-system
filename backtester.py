"""Long-only backtest: signal at the close, fill at the next bar's open.

Costs depend on the market. Bare tickers use the US schedule. ``.TW`` and
``.TWO`` use Taiwan brokerage plus sell-side securities transaction tax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# US schedule. Rates are fractions of traded notional (10 bps = 0.001).
DEFAULT_US_COMMISSION = 0.001
DEFAULT_US_SLIPPAGE = 0.0005
# Recorded only. This engine does not generate dividend cashflows.
DEFAULT_US_DIVIDEND_WITHHOLDING = 0.30

# Taiwan retail schedule. Brokerage is the statutory ceiling, not a
# discounted e-broker rate. The NT$20 floor is a common broker minimum.
DEFAULT_TW_BROKERAGE_RATE = 0.001425
DEFAULT_TW_BROKERAGE_MIN = 20.0
DEFAULT_TW_STOCK_SELL_TAX = 0.003
DEFAULT_TW_ETF_SELL_TAX = 0.001

_MARKET_ALIASES = {
    "US": "US",
    "USA": "US",
    "TW": "TW",
    "TWD": "TW",
    "TAIWAN": "TW",
    "FUTURES": "FUTURES",
    "FUTURE": "FUTURES",
    "FUT": "FUTURES",
}

TRADE_COLUMNS = ["date", "action", "price", "shares", "value", "fee", "market"]


@dataclass(frozen=True)
class MarketCosts:
    """Resolved cost schedule for one run.

    ``dividend_withholding`` is the US rate to apply if dividends are added
    later. Nothing in ``Backtester.run`` reads it.
    """

    market: str
    slippage_rate: float
    commission_rate: float
    commission_flat: float
    brokerage_rate: float
    brokerage_min: float
    sell_tax_rate: float
    dividend_withholding: float
    contract_multiplier: float
    min_lot: float | None


def detect_market(symbol: str | None = None, market: str | None = None) -> str:
    """Return ``US``, ``TW``, or ``FUTURES``.

    An explicit ``market`` wins. Otherwise ``.TW`` / ``.TWO`` is Taiwan and
    every other symbol, including a missing one, is US. Futures are not
    inferred from the ticker.
    """
    if market is not None and str(market).strip() != "":
        key = str(market).strip().upper()
        if key not in _MARKET_ALIASES:
            raise ValueError(
                f"unknown market {market!r}; expected US, TW, or FUTURES"
            )
        return _MARKET_ALIASES[key]

    if symbol is None or str(symbol).strip() == "":
        return "US"
    ticker = str(symbol).strip().upper()
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return "TW"
    return "US"


def _tw_code(symbol: str | None) -> str:
    if symbol is None:
        return ""
    ticker = str(symbol).strip().upper()
    for suffix in (".TWO", ".TW"):
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


def _default_tw_sell_tax(symbol: str | None) -> float:
    """Stock 0.3%, ETF-like codes 0.1%.

    Taiwan ETF tickers conventionally start with ``00`` (``0050``, ``00631L``).
    This is a ticker convention, not an exchange classification. Bond ETFs
    are often exempt and still get 0.1% here; pass ``sell_tax_rate`` to override.
    """
    code = _tw_code(symbol)
    if code.startswith("00"):
        return DEFAULT_TW_ETF_SELL_TAX
    return DEFAULT_TW_STOCK_SELL_TAX


def _rate_from_bps(bps: float | None, fraction: float, bps_name: str) -> float:
    if bps is None:
        _require_non_negative(bps_name.replace("_bps", ""), fraction)
        return fraction
    _require_non_negative(bps_name, bps)
    return bps / 10_000.0


def _require_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0")


def resolve_costs(
    *,
    symbol: str | None = None,
    market: str | None = None,
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
) -> MarketCosts:
    """Build the cost schedule. ``commission_bps`` / ``slippage_bps`` win when set."""
    resolved_market = detect_market(symbol, market)
    slippage_rate = _rate_from_bps(slippage_bps, slippage, "slippage_bps")
    if slippage_rate >= 1:
        raise ValueError("slippage must be < 1 so a sell still has a positive price")
    _require_non_negative("commission_flat", commission_flat)
    if not math.isfinite(contract_multiplier) or contract_multiplier <= 0:
        raise ValueError("contract_multiplier must be a finite number > 0")
    if min_lot is not None and (not math.isfinite(min_lot) or min_lot <= 0):
        raise ValueError("min_lot must be a finite number > 0")

    commission_rate = _rate_from_bps(commission_bps, commission, "commission_bps")
    if resolved_market == "TW":
        brokerage = (
            DEFAULT_TW_BROKERAGE_RATE if brokerage_rate is None else brokerage_rate
        )
        minimum = DEFAULT_TW_BROKERAGE_MIN if brokerage_min is None else brokerage_min
        tax = _default_tw_sell_tax(symbol) if sell_tax_rate is None else sell_tax_rate
        _require_non_negative("brokerage_rate", brokerage)
        _require_non_negative("brokerage_min", minimum)
        _require_non_negative("sell_tax_rate", tax)
        # US withholding does not apply to the TW schedule.
        withholding = 0.0 if dividend_withholding is None else dividend_withholding
        _require_non_negative("dividend_withholding", withholding)
        return MarketCosts(
            market="TW",
            slippage_rate=slippage_rate,
            commission_rate=0.0,
            commission_flat=0.0,
            brokerage_rate=brokerage,
            brokerage_min=minimum,
            sell_tax_rate=tax,
            dividend_withholding=withholding,
            contract_multiplier=contract_multiplier,
            min_lot=min_lot,
        )

    if brokerage_rate is not None or brokerage_min is not None or sell_tax_rate is not None:
        raise ValueError(
            "brokerage_rate, brokerage_min, and sell_tax_rate apply to market='TW' only"
        )
    withholding_default = (
        DEFAULT_US_DIVIDEND_WITHHOLDING if resolved_market == "US" else 0.0
    )
    withholding = withholding_default if dividend_withholding is None else dividend_withholding
    _require_non_negative("dividend_withholding", withholding)
    return MarketCosts(
        market=resolved_market,
        slippage_rate=slippage_rate,
        commission_rate=commission_rate,
        commission_flat=commission_flat,
        brokerage_rate=0.0,
        brokerage_min=0.0,
        sell_tax_rate=0.0,
        dividend_withholding=withholding,
        contract_multiplier=contract_multiplier,
        min_lot=min_lot,
    )


def _execution_price(open_price: float, buying: bool, slippage_rate: float) -> float:
    if buying:
        return open_price * (1.0 + slippage_rate)
    return open_price * (1.0 - slippage_rate)


def _fee(notional: float, costs: MarketCosts, side: str) -> float:
    """Fee in cash. TW tax is charged on sells only, on traded notional."""
    if costs.market == "TW":
        brokerage = max(notional * costs.brokerage_rate, costs.brokerage_min)
        tax = notional * costs.sell_tax_rate if side == "SELL" else 0.0
        return brokerage + tax
    return costs.commission_rate * notional + costs.commission_flat


def _size_buy(cash: float, fill_price: float, costs: MarketCosts) -> tuple[float, float, float]:
    """All-in buy. Returns ``(quantity, notional, fee)`` or zeros if it does not fit.

    Notional is ``quantity * fill_price * contract_multiplier`` (fully funded,
    including the futures placeholder). The fee is a fraction of that notional,
    plus the US flat fee or the TW minimum brokerage.
    """
    unit = fill_price * costs.contract_multiplier
    if cash <= 0 or unit <= 0:
        return 0.0, 0.0, 0.0

    if costs.market == "TW":
        notional = cash / (1.0 + costs.brokerage_rate)
        fee = notional * costs.brokerage_rate
        if fee < costs.brokerage_min:
            fee = costs.brokerage_min
            notional = cash - fee
    else:
        notional = (cash - costs.commission_flat) / (1.0 + costs.commission_rate)
        fee = _fee(notional, costs, "BUY")

    if notional <= 0:
        return 0.0, 0.0, 0.0

    quantity = notional / unit
    if costs.min_lot is not None:
        lots = math.floor(quantity / costs.min_lot + 1e-9)
        while lots > 0:
            quantity = lots * costs.min_lot
            notional = quantity * unit
            fee = _fee(notional, costs, "BUY")
            if notional + fee <= cash + 1e-6:
                return quantity, notional, fee
            lots -= 1
        return 0.0, 0.0, 0.0

    if notional + fee > cash + 1e-6:
        return 0.0, 0.0, 0.0
    return quantity, notional, fee


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
        """Simulate one name.

        Parameters
        ----------
        data : DataFrame
            Must include ``Open`` and ``Close``. Fills use ``Open``.
            ``Close`` is only used to mark the book.
        signals : Series
            1 = be long, 0 = be flat. Any other value is treated as flat.
            The value at bar t is applied at the open of bar t+1.
        symbol, market :
            Optional. Used to pick the cost schedule. ``market`` wins over
            the suffix. Defaults to the values passed to ``Backtester``.

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
        if "Open" not in data.columns or "Close" not in data.columns:
            raise ValueError(
                "data must include Open and Close; "
                "orders fill at the next bar's Open, not the signal bar's Close"
            )

        symbol = self.symbol if symbol is None else symbol
        market = self.market if market is None else market
        costs = self._costs(symbol, market)

        work = pd.DataFrame(
            {
                "Open": pd.to_numeric(data["Open"], errors="coerce"),
                "Close": pd.to_numeric(data["Close"], errors="coerce"),
            },
            index=data.index,
        )
        work = work.loc[np.isfinite(work["Close"].to_numpy(dtype=float))]
        aligned = signals.reindex(work.index).ffill().fillna(0).astype(int)

        cash = float(self.initial_capital)
        shares = 0.0
        portfolio_values: list[float] = []
        trades: list[dict] = []

        opens = work["Open"].to_numpy(dtype=float)
        closes = work["Close"].to_numpy(dtype=float)
        observed = aligned.to_numpy(dtype=int)
        index = work.index

        for i in range(len(work)):
            # The order for this open was decided at the previous close.
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
                            {
                                "date": index[i],
                                "action": "BUY",
                                "price": fill,
                                "shares": quantity,
                                "value": notional,
                                "fee": fee,
                                "market": costs.market,
                            }
                        )
                elif can_fill and holding and not target_long:
                    fill = _execution_price(open_price, False, costs.slippage_rate)
                    notional = shares * fill * costs.contract_multiplier
                    fee = _fee(notional, costs, "SELL")
                    proceeds = notional - fee
                    trades.append(
                        {
                            "date": index[i],
                            "action": "SELL",
                            "price": fill,
                            "shares": shares,
                            "value": proceeds,
                            "fee": fee,
                            "market": costs.market,
                        }
                    )
                    cash += proceeds
                    shares = 0.0

            marked = shares * float(closes[i]) * costs.contract_multiplier
            portfolio_values.append(cash + marked)

        portfolio = pd.DataFrame(
            {
                "portfolio_value": portfolio_values,
                "signal": aligned.to_numpy(dtype=int),
                "price": closes,
            },
            index=index,
        )
        portfolio["returns"] = portfolio["portfolio_value"].pct_change()

        trades_df = (
            pd.DataFrame(trades)
            if trades
            else pd.DataFrame(columns=TRADE_COLUMNS)
        )
        return portfolio, trades_df
