import pandas as pd
import numpy as np


class BaseStrategy:
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class BuyAndHold(BaseStrategy):
    """買進持有：從頭到尾持有，不進行任何交易。"""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=data.index, dtype=int)


class MACross(BaseStrategy):
    """均線交叉：短期均線上穿長期均線時買入，下穿時出場。"""

    def __init__(self, short_window: int = 20, long_window: int = 60, ma_type: str = "SMA"):
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        if self.ma_type == "EMA":
            short_ma = close.ewm(span=self.short_window, adjust=False).mean()
            long_ma = close.ewm(span=self.long_window, adjust=False).mean()
        else:
            short_ma = close.rolling(self.short_window).mean()
            long_ma = close.rolling(self.long_window).mean()
        signal = pd.Series(0, index=data.index, dtype=int)
        signal[short_ma > long_ma] = 1
        return signal


class RSIStrategy(BaseStrategy):
    """RSI 策略：RSI 低於超賣區時買入，高於超買區時出場。"""

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rsi = self._compute_rsi(data["Close"])
        position = 0
        signals = []
        for val in rsi:
            if pd.isna(val):
                signals.append(0)
                continue
            if val < self.oversold:
                position = 1
            elif val > self.overbought:
                position = 0
            signals.append(position)
        return pd.Series(signals, index=data.index, dtype=int)


class BollingerBandsStrategy(BaseStrategy):
    """布林通道均值回歸：價格跌破下軌時買入，突破上軌時出場。"""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        ma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = ma + self.std_dev * std
        lower = ma - self.std_dev * std
        position = 0
        signals = []
        for i in range(len(close)):
            if pd.isna(ma.iloc[i]):
                signals.append(0)
                continue
            c = close.iloc[i]
            if c < lower.iloc[i]:
                position = 1
            elif c > upper.iloc[i]:
                position = 0
            signals.append(position)
        return pd.Series(signals, index=data.index, dtype=int)


class MACDStrategy(BaseStrategy):
    """MACD 交叉：MACD 上穿信號線時買入，下穿時出場。"""

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        position = 0
        signals = []
        prev_m = None
        prev_s = None

        for i in range(len(macd_line)):
            m = macd_line.iloc[i]
            s = signal_line.iloc[i]
            if pd.isna(m) or pd.isna(s):
                signals.append(0)
                prev_m, prev_s = m, s
                continue
            if prev_m is not None and not pd.isna(prev_m) and not pd.isna(prev_s):
                if prev_m < prev_s and m >= s:
                    position = 1  # 黃金交叉
                elif prev_m > prev_s and m <= s:
                    position = 0  # 死亡交叉
            signals.append(position)
            prev_m, prev_s = m, s

        return pd.Series(signals, index=data.index, dtype=int)
