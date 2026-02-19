from typing import Tuple

import numpy as np
import pandas as pd


class Backtester:
    """
    簡易事件驅動回測引擎。
    支援多空信號 (1=做多, 0=空倉)，並考慮手續費與滑價。
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self, data: pd.DataFrame, signals: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        執行回測模擬。

        Parameters
        ----------
        data    : OHLCV DataFrame
        signals : 1=持有多頭, 0=空倉

        Returns
        -------
        portfolio : portfolio_value / signal / price / returns 的 DataFrame
        trades    : 交易記錄 DataFrame
        """
        close = data["Close"].ffill().dropna()
        signals = signals.reindex(close.index).ffill().fillna(0).astype(int)

        capital = float(self.initial_capital)
        shares = 0.0
        portfolio_values: list = []
        trades: list = []
        prev_signal = 0

        for i in range(len(close)):
            date = close.index[i]
            price = float(close.iloc[i])
            signal = int(signals.iloc[i])

            # 信號改變時執行交易
            if signal != prev_signal:
                if signal == 1 and capital > 0:
                    # 買入
                    buy_price = price * (1 + self.slippage)
                    shares = (capital * (1 - self.commission)) / buy_price
                    capital = 0.0
                    trades.append(
                        {
                            "date": date,
                            "action": "BUY",
                            "price": round(buy_price, 4),
                            "shares": round(shares, 6),
                            "value": round(shares * buy_price, 2),
                        }
                    )
                elif signal == 0 and shares > 0:
                    # 賣出
                    sell_price = price * (1 - self.slippage)
                    capital = shares * sell_price * (1 - self.commission)
                    trades.append(
                        {
                            "date": date,
                            "action": "SELL",
                            "price": round(sell_price, 4),
                            "shares": round(shares, 6),
                            "value": round(capital, 2),
                        }
                    )
                    shares = 0.0

            prev_signal = signal
            portfolio_values.append(capital + shares * price)

        portfolio = pd.DataFrame(
            {
                "portfolio_value": portfolio_values,
                "signal": signals.values,
                "price": close.values,
            },
            index=close.index,
        )
        portfolio["returns"] = portfolio["portfolio_value"].pct_change()

        trades_df = (
            pd.DataFrame(trades)
            if trades
            else pd.DataFrame(columns=["date", "action", "price", "shares", "value"])
        )
        return portfolio, trades_df
