"""Single-name signals and the two book strategies.

Signal strategies return 1 (long) or 0 (flat) on the close. ``Backtester``
fills that decision at the next bar's open. ``CrossSection`` and
``CoreSatellite`` set ``manages_book`` because a 0/1 series cannot express
a ranked book or two funded sleeves.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


class BaseStrategy:
    # True when the strategy sizes a book itself instead of emitting 0/1.
    manages_book = False

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


def _moving_average(close: pd.Series, window: int, ma_type: str) -> pd.Series:
    if ma_type == "EMA":
        return close.ewm(span=window, adjust=False).mean()
    return close.rolling(window).mean()


def wilder_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR. Missing High/Low falls back to the close, so the range is the gap."""
    close = data["Close"].astype(float)
    high = data["High"].astype(float) if "High" in data.columns else close
    low = data["Low"].astype(float) if "Low" in data.columns else close
    prev = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


class TrendStop(BaseStrategy):
    """順勢持有，停損把虧損的單砍掉。

    收盤在慢均線之上才做多。進場之後停損從這一段的最高收盤往下算，
    可以是百分比、ATR 倍數，或兩者取較緊的那一條（停損價較高的）。
    停損用收盤判斷，不是盤中觸價；成交仍是下一根開盤。

    停損出場時若收盤還在均線上，不會下一根就買回。要先收到均線下，
    再重新站上，才算下一段趨勢。樣本一開始若已經在均線上，允許進場。
    """

    def __init__(
        self,
        trend_window: int = 100,
        ma_type: str = "SMA",
        stop_mode: str = "atr",
        atr_period: int = 14,
        atr_mult: float = 3.0,
        stop_pct: float = 0.08,
    ):
        self.trend_window = _as_int("trend_window", trend_window, 2)
        self.ma_type = _ma_type(ma_type)
        self.stop_mode = _stop_mode(stop_mode)
        self.atr_period = _as_int("atr_period", atr_period, 1)
        self.atr_mult = _positive("atr_mult", atr_mult)
        self.stop_pct = _stop_pct(stop_pct)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"].astype(float)
        trend = _moving_average(close, self.trend_window, self.ma_type)
        atr = None if self.stop_mode == "percent" else wilder_atr(data, self.atr_period)

        position = 0
        armed = True
        peak: float | None = None
        signals: list[int] = []

        for i in range(len(close)):
            price = close.iloc[i]
            line = trend.iloc[i]
            if pd.isna(price) or pd.isna(line):
                signals.append(0)
                continue
            above = float(price) > float(line)
            if position == 0:
                if not above:
                    armed = True
                elif armed and above:
                    position = 1
                    peak = float(price)
                    armed = False
                signals.append(position)
                continue

            peak = float(price) if peak is None else max(peak, float(price))
            atr_value = None if atr is None else atr.iloc[i]
            stop = _stop_level(peak, atr_value, self.stop_mode, self.atr_mult, self.stop_pct)
            hit = stop is not None and float(price) < stop
            if hit or not above:
                position = 0
                peak = None
                armed = not above
            signals.append(position)

        return pd.Series(signals, index=data.index, dtype=int)


def rank_cross_section(
    prices: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame,
    as_of,
    *,
    top_n: int = 3,
    weighting: str = "equal",
    momentum_lookback: int = 252,
    momentum_skip: int = 21,
    vol_lookback: int = 63,
    momentum_weight: float = 0.5,
    value_weight: float = 0.5,
) -> pd.DataFrame:
    """Rank a small universe with what is knowable on ``as_of``.

    Momentum uses closes on or before ``as_of`` only: the close
    ``momentum_skip`` sessions back, over the ``momentum_lookback`` sessions
    before that. Earnings yield is ``1 / PE`` from the fundamentals frame
    the caller passes in. That frame is the point-in-time panel. This
    function does not fetch Yahoo's live snapshot, and it does not fill a
    missing PE with the cross-sectional median.

    A name that lacks a required leg is left out. ``top_n`` of the rest are
    weighted equally, or by inverse annualised volatility when ``weighting``
    is ``"risk"``. Inverse vol is a budget, not an optimiser: no covariance,
    no leverage, and the selected weights sum to 1. If any selected name has
    no usable volatility, that date falls back to equal weight.
    """
    top_n = _as_int("top_n", top_n, 1)
    lookback = _as_int("momentum_lookback", momentum_lookback, 1)
    skip = _as_int("momentum_skip", momentum_skip, 0)
    vol_window = _as_int("vol_lookback", vol_lookback, 1)
    method = _weighting(weighting)
    mom_w = _non_negative("momentum_weight", momentum_weight)
    val_w = _non_negative("value_weight", value_weight)
    if mom_w + val_w <= 0:
        raise ValueError("momentum_weight and value_weight cannot both be 0")

    as_of_day = _day(as_of)
    pe_by_symbol = _pe_table(fundamentals)
    rows: list[dict] = []
    for symbol in list(prices):
        close = _closes_through(prices.get(symbol), as_of_day)
        momentum = _momentum(close, lookback, skip)
        pe = pe_by_symbol.get(str(symbol), math.nan)
        earnings_yield = (1.0 / pe) if _positive_number(pe) else math.nan
        volatility = _annual_vol(close, vol_window)
        if mom_w > 0 and not math.isfinite(momentum):
            continue
        if val_w > 0 and not math.isfinite(earnings_yield):
            continue
        rows.append(
            {
                "symbol": str(symbol),
                "as_of": as_of_day,
                "momentum": momentum,
                "pe": pe,
                "earnings_yield": earnings_yield,
                "volatility": volatility,
            }
        )

    columns = [
        "symbol",
        "as_of",
        "momentum",
        "pe",
        "earnings_yield",
        "volatility",
        "score",
        "weight",
        "weighting",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    score = pd.Series(0.0, index=frame.index, dtype=float)
    if mom_w > 0:
        score = score + _zscore(frame["momentum"]) * mom_w
    if val_w > 0:
        score = score + _zscore(frame["earnings_yield"]) * val_w
    frame["score"] = score / (mom_w + val_w)
    frame = frame.sort_values(["score", "symbol"], ascending=[False, True])
    selected = frame.head(min(top_n, len(frame))).copy()
    vol_ok = bool(selected["volatility"].map(_positive_number).all())
    if method == "risk" and vol_ok:
        inverse = 1.0 / selected["volatility"].astype(float)
        selected["weight"] = (inverse / float(inverse.sum())).to_numpy()
        used = "risk"
    else:
        selected["weight"] = 1.0 / len(selected)
        used = "equal"
    selected["weighting"] = used
    return selected.reset_index(drop=True)


class CrossSection(BaseStrategy):
    """再平衡日用當時可知的動能與本益比排名，持有前 N 名。

    權重在再平衡日收盤決定，下一根開盤才調倉。兩次再平衡之間讓部位隨價格
    漂，不每日拉回。沒有另外傳載入函式時，基本面走 ``fetch_fundamentals_pit``。
    """

    manages_book = True

    def __init__(
        self,
        top_n: int = 3,
        weighting: str = "equal",
        rebalance_months: int = 1,
        momentum_lookback: int = 252,
        momentum_skip: int = 21,
        vol_lookback: int = 63,
        momentum_weight: float = 0.5,
        value_weight: float = 0.5,
        fundamentals_loader=None,
    ):
        self.top_n = _as_int("top_n", top_n, 1)
        self.weighting = _weighting(weighting)
        self.rebalance_months = _as_int("rebalance_months", rebalance_months, 1)
        self.momentum_lookback = _as_int("momentum_lookback", momentum_lookback, 1)
        self.momentum_skip = _as_int("momentum_skip", momentum_skip, 0)
        self.vol_lookback = _as_int("vol_lookback", vol_lookback, 1)
        self.momentum_weight = _non_negative("momentum_weight", momentum_weight)
        self.value_weight = _non_negative("value_weight", value_weight)
        if self.momentum_weight + self.value_weight <= 0:
            raise ValueError("momentum_weight and value_weight cannot both be 0")
        self.fundamentals_loader = fundamentals_loader

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise ValueError(
            "cross_section ranks a universe; it does not emit a single-name 0/1 signal"
        )

    def run_book(self, prices, symbols, start, end, **kwargs):
        from backtest_system.engine.portfolio import month_end_dates, simulate_target_weights

        start_day = _day(start)
        end_day = _day(end)
        names = [str(symbol) for symbol in symbols]
        loader = self.fundamentals_loader or _default_pit_loader
        calendar = _union_index(prices, names)
        decisions = [
            stamp
            for stamp in month_end_dates(
                calendar, start_day, end_day, every=self.rebalance_months
            )
            if start_day <= stamp < end_day
        ]
        holdings: list[pd.DataFrame] = []
        weight_rows: list[dict] = []
        notes = [
            "排名只用再平衡日當天或之前的收盤，以及呼叫端給的 point-in-time 財報。"
            "缺了所需因子的標的這期不排。權重在收盤決定，下一根開盤成交。"
        ]
        for as_of in decisions:
            fundamentals = loader(names, as_of)
            ranked = rank_cross_section(
                prices,
                fundamentals,
                as_of,
                top_n=self.top_n,
                weighting=self.weighting,
                momentum_lookback=self.momentum_lookback,
                momentum_skip=self.momentum_skip,
                vol_lookback=self.vol_lookback,
                momentum_weight=self.momentum_weight,
                value_weight=self.value_weight,
            )
            target = {symbol: 0.0 for symbol in names}
            if ranked.empty:
                notes.append(
                    f"{as_of.date().isoformat()} 沒有同時具備所需因子的標的，目標權重為 0。"
                )
            else:
                holdings.append(ranked)
                for row in ranked.itertuples(index=False):
                    target[str(row.symbol)] = float(row.weight)
            weight_rows.append(target)

        if weight_rows:
            weights = pd.DataFrame(weight_rows, index=pd.DatetimeIndex(decisions))
        else:
            weights = pd.DataFrame(columns=names)
        book = simulate_target_weights(
            prices,
            weights,
            names,
            start_day,
            end_day,
            initial_capital=kwargs["initial_capital"],
            commission=kwargs.get("commission", 0.001),
            slippage=kwargs.get("slippage", 0.0005),
            market=kwargs.get("market"),
            backtester_kwargs=kwargs.get("backtester_kwargs"),
        )
        book.holdings = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
        book.notes = notes + book.notes
        return book


class CoreSatellite(BaseStrategy):
    """核心與衛星用固定比例各拿一筆資金，之後各自漲跌。

    ``core_weight`` 與 ``satellite_weight`` 必須加總為 1。資金在一開始切開，
    不每期拉回 70/30，也沒有組合優化。核心只能是買進持有或趨勢停損。
    衛星可以是單一名稱的策略，或橫截面多因子。
    """

    manages_book = True

    def __init__(
        self,
        core_weight: float = 0.70,
        satellite_weight: float = 0.30,
        core: str = "buy_and_hold",
        satellite: str = "trend_stop",
        core_params: Mapping | None = None,
        satellite_params: Mapping | None = None,
        core_symbol: str | None = None,
        fundamentals_loader=None,
    ):
        self.core_weight = _unit_weight("core_weight", core_weight)
        self.satellite_weight = _unit_weight("satellite_weight", satellite_weight)
        if abs(self.core_weight + self.satellite_weight - 1.0) > 1e-6:
            raise ValueError("core_weight and satellite_weight must sum to 1")
        self.core = canonical_strategy(core)
        self.satellite = canonical_strategy(satellite)
        if self.core not in {"buy_and_hold", "trend_stop"}:
            raise ValueError("core sleeve must be buy_and_hold or trend_stop")
        if self.satellite == "core_satellite":
            raise ValueError("satellite cannot be core_satellite")
        self.core_params = dict(core_params or {})
        self.satellite_params = dict(satellite_params or {})
        symbol = "" if core_symbol is None else str(core_symbol).strip().upper()
        self.core_symbol = symbol or None
        self.fundamentals_loader = fundamentals_loader

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise ValueError("core_satellite funds two sleeves; it has no single 0/1 signal")

    def run_book(self, prices, symbols, start, end, **kwargs):
        from backtest_system.engine.portfolio import combine_books, run_sleeve

        names = [str(symbol) for symbol in symbols]
        core_names = [self.core_symbol] if self.core_symbol else list(names)
        capital = float(kwargs["initial_capital"])
        shared = {
            "commission": kwargs.get("commission", 0.001),
            "slippage": kwargs.get("slippage", 0.0005),
            "market": kwargs.get("market"),
            "backtester_kwargs": kwargs.get("backtester_kwargs"),
        }
        core = build_strategy(self.core, self.core_params)
        satellite = build_strategy(
            self.satellite,
            self.satellite_params,
            fundamentals_loader=self.fundamentals_loader,
        )
        core_book = run_sleeve(
            core,
            prices,
            core_names,
            start,
            end,
            initial_capital=capital * self.core_weight,
            **shared,
        )
        sat_book = run_sleeve(
            satellite,
            prices,
            names,
            start,
            end,
            initial_capital=capital * self.satellite_weight,
            **shared,
        )
        note = (
            f"核心 {self.core_weight:.0%}（{self.core}）、"
            f"衛星 {self.satellite_weight:.0%}（{self.satellite}）。"
            "比例只用於一開始切資金，之後不拉回，也沒有優化器。"
        )
        if self.core_symbol:
            note += f" 核心只交易 {self.core_symbol}。"
        combined = combine_books([core_book, sat_book])
        combined.notes.insert(0, note)
        combined.initial_capital = capital
        return combined


_STRATEGY_KEYS = {
    "buy_and_hold": "buy_and_hold",
    "buy & hold": "buy_and_hold",
    "買進持有 (buy & hold)": "buy_and_hold",
    "ma_cross": "ma_cross",
    "ma cross": "ma_cross",
    "均線交叉 (ma cross)": "ma_cross",
    "rsi": "rsi",
    "rsi 策略": "rsi",
    "bollinger": "bollinger",
    "bollinger bands": "bollinger",
    "布林通道 (bollinger bands)": "bollinger",
    "macd": "macd",
    "macd 策略": "macd",
    "trend_stop": "trend_stop",
    "trend + stop": "trend_stop",
    "trend stop": "trend_stop",
    "趨勢停損 (trend + stop)": "trend_stop",
    "cross_section": "cross_section",
    "cross-section": "cross_section",
    "cross section": "cross_section",
    "橫截面多因子 (cross-section)": "cross_section",
    "core_satellite": "core_satellite",
    "core-satellite": "core_satellite",
    "core satellite": "core_satellite",
    "核心衛星 (core-satellite)": "core_satellite",
}

_PARAM_FIELDS = {
    "buy_and_hold": set(),
    "ma_cross": {"short_window", "long_window", "ma_type"},
    "rsi": {"period", "oversold", "overbought"},
    "bollinger": {"period", "std_dev"},
    "macd": {"fast", "slow", "signal_period"},
    "trend_stop": {"trend_window", "ma_type", "stop_mode", "atr_period", "atr_mult", "stop_pct"},
    "cross_section": {
        "top_n",
        "weighting",
        "rebalance_months",
        "momentum_lookback",
        "momentum_skip",
        "vol_lookback",
        "momentum_weight",
        "value_weight",
    },
    "core_satellite": {
        "core_weight",
        "satellite_weight",
        "core",
        "satellite",
        "core_params",
        "satellite_params",
        "core_symbol",
    },
}


def canonical_strategy(name: str) -> str:
    """Map a UI label or a yaml name onto one strategy key."""
    key = " ".join(str(name).strip().casefold().split())
    if key not in _STRATEGY_KEYS:
        raise ValueError(f"unknown strategy: {name}")
    return _STRATEGY_KEYS[key]


def build_strategy(
    name: str,
    params: Mapping | None = None,
    *,
    fundamentals_loader=None,
) -> BaseStrategy:
    """Build a strategy. Unknown parameter names are rejected."""
    key = canonical_strategy(name)
    raw = dict(params or {})
    raw.pop("fundamentals_loader", None)
    unknown = sorted(set(raw) - _PARAM_FIELDS[key])
    if unknown:
        raise ValueError(f"unknown parameters for {key}: {', '.join(unknown)}")
    if key == "buy_and_hold":
        return BuyAndHold()
    if key == "ma_cross":
        return MACross(
            short_window=int(raw.get("short_window", 20)),
            long_window=int(raw.get("long_window", 60)),
            ma_type=str(raw.get("ma_type", "SMA")),
        )
    if key == "rsi":
        return RSIStrategy(
            period=int(raw.get("period", 14)),
            oversold=int(raw.get("oversold", 30)),
            overbought=int(raw.get("overbought", 70)),
        )
    if key == "bollinger":
        return BollingerBandsStrategy(
            period=int(raw.get("period", 20)),
            std_dev=float(raw.get("std_dev", 2.0)),
        )
    if key == "macd":
        return MACDStrategy(
            fast=int(raw.get("fast", 12)),
            slow=int(raw.get("slow", 26)),
            signal_period=int(raw.get("signal_period", 9)),
        )
    if key == "trend_stop":
        return TrendStop(
            trend_window=int(raw.get("trend_window", 100)),
            ma_type=str(raw.get("ma_type", "SMA")),
            stop_mode=str(raw.get("stop_mode", "atr")),
            atr_period=int(raw.get("atr_period", 14)),
            atr_mult=float(raw.get("atr_mult", 3.0)),
            stop_pct=float(raw.get("stop_pct", 0.08)),
        )
    if key == "cross_section":
        return CrossSection(
            top_n=int(raw.get("top_n", 3)),
            weighting=str(raw.get("weighting", "equal")),
            rebalance_months=int(raw.get("rebalance_months", 1)),
            momentum_lookback=int(raw.get("momentum_lookback", 252)),
            momentum_skip=int(raw.get("momentum_skip", 21)),
            vol_lookback=int(raw.get("vol_lookback", 63)),
            momentum_weight=float(raw.get("momentum_weight", 0.5)),
            value_weight=float(raw.get("value_weight", 0.5)),
            fundamentals_loader=fundamentals_loader,
        )
    return CoreSatellite(
        core_weight=float(raw.get("core_weight", 0.70)),
        satellite_weight=float(raw.get("satellite_weight", 0.30)),
        core=str(raw.get("core", "buy_and_hold")),
        satellite=str(raw.get("satellite", "trend_stop")),
        core_params=raw.get("core_params"),
        satellite_params=raw.get("satellite_params"),
        core_symbol=raw.get("core_symbol"),
        fundamentals_loader=fundamentals_loader,
    )


def _default_pit_loader(symbols: Sequence[str], as_of):
    from backtest_system.fundamentals.pit import fetch_fundamentals_pit

    return fetch_fundamentals_pit(list(symbols), as_of)


def _closes_through(frame: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.Series:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna().copy()
    close.index = _day_index(close.index)
    close = close[close.index <= as_of]
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.astype(float)


def _momentum(close: pd.Series, lookback: int, skip: int) -> float:
    # close[t - skip] / close[t - skip - lookback] - 1, t = last bar on or before as_of.
    if len(close) <= lookback + skip:
        return math.nan
    end = close.iloc[-1 - skip]
    start = close.iloc[-1 - skip - lookback]
    if not _positive_number(start) or not math.isfinite(float(end)):
        return math.nan
    return float(end) / float(start) - 1.0


def _annual_vol(close: pd.Series, window: int) -> float:
    returns = close.pct_change().dropna()
    if len(returns) < 2:
        return math.nan
    sample = returns.iloc[-window:]
    if len(sample) < 2:
        return math.nan
    vol = float(sample.std(ddof=0) * math.sqrt(252))
    if not math.isfinite(vol) or vol <= 0:
        return math.nan
    return vol


def _pe_table(fundamentals: pd.DataFrame | None) -> dict[str, float]:
    if fundamentals is None or len(fundamentals) == 0 or "symbol" not in fundamentals.columns:
        return {}
    frame = fundamentals.copy()
    if "PE" not in frame.columns:
        frame["PE"] = math.nan
    out: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        symbol = getattr(row, "symbol", None)
        if symbol is None:
            continue
        try:
            pe = float(getattr(row, "PE"))
        except (TypeError, ValueError):
            pe = math.nan
        out[str(symbol)] = pe
    return out


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = float(values.std(ddof=0))
    if not math.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=series.index)
    return (values - float(values.mean())) / std


def _union_index(prices: Mapping[str, pd.DataFrame], symbols: Sequence[str]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex([])
    for symbol in symbols:
        frame = prices.get(symbol)
        if frame is None or frame.empty:
            continue
        index = index.union(_day_index(frame.index))
    return index.sort_values()


def _stop_level(peak, atr_value, mode: str, atr_mult: float, stop_pct: float) -> float | None:
    levels: list[float] = []
    if mode in {"percent", "tighter"}:
        levels.append(float(peak) * (1.0 - stop_pct))
    if mode in {"atr", "tighter"} and atr_value is not None and pd.notna(atr_value):
        atr = float(atr_value)
        if math.isfinite(atr) and atr > 0:
            levels.append(float(peak) - atr_mult * atr)
    if not levels:
        return None
    if mode == "tighter":
        return max(levels)
    return levels[0]


def _day(value) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid date: {value!r}") from exc
    if pd.isna(stamp):
        raise ValueError(f"invalid date: {value!r}")
    if stamp.tzinfo is not None:
        stamp = pd.Timestamp(stamp.strftime("%Y-%m-%d"))
    return stamp.normalize()


def _day_index(index) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex(pd.to_datetime(index))
    if stamps.tz is not None:
        stamps = pd.DatetimeIndex(stamps.strftime("%Y-%m-%d"))
    return stamps.normalize()


def _as_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer >= {minimum}")
    whole = int(number)
    if whole < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return whole


def _positive(name: str, value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number > 0") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number > 0")
    return number


def _non_negative(name: str, value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number >= 0") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return number


def _unit_weight(name: str, value) -> float:
    number = _non_negative(name, value)
    if number > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _stop_pct(value) -> float:
    number = _positive("stop_pct", value)
    if number >= 1:
        raise ValueError("stop_pct must be < 1")
    return number


def _positive_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _ma_type(value: str) -> str:
    kind = str(value).strip().upper()
    if kind not in {"SMA", "EMA"}:
        raise ValueError("ma_type must be SMA or EMA")
    return kind


def _stop_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"atr", "percent", "tighter"}:
        raise ValueError("stop_mode must be atr, percent, or tighter")
    return mode


def _weighting(value: str) -> str:
    method = str(value).strip().lower()
    if method not in {"equal", "risk"}:
        raise ValueError("weighting must be equal or risk")
    return method
