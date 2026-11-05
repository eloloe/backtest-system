"""Taiwan stocks, ETFs, and index futures via SinoPac Shioaji.

The SDK is imported on the first request that needs it. Selecting Yahoo,
Tiingo, or Polygon does not require the package.
"""

from __future__ import annotations

import os
import re

import pandas as pd

from providers.base import (
    DataProviderConfigError,
    MissingApiKeyError,
    PriceProvider,
    ProviderError,
    drop_timezone,
    empty_ohlcv,
    finalize_ohlcv,
    request_window,
)

# ``kbars`` rejects a start/end range longer than 30 days, and both
# ends are inclusive. Stay inside that with 30 calendar dates.
_KBAR_CHUNK_DAYS = 30
# The SDK default of 5 seconds is tight for a full chunk of minute bars.
_KBAR_TIMEOUT_MS = 30_000
# Domestic index futures open the night session at 15:00. That session
# belongs to the next regular trading day (Friday night -> Monday).
_NIGHT_SESSION_HOUR = 15
# Friday 15:00 through Saturday 05:00 is Monday's session, so a Monday
# bar needs the three calendar days before it.
_FUTURES_LOOKBACK_DAYS = 3

# TAIFEX month letters run A=January through L=December, then a one-digit year.
# TXFH6 is August 2026.
_EQUITY = re.compile(r"\d{4,6}[A-Z]?")
_FUTURES_ROOT = re.compile(r"[A-Z]{2,3}")
_FUTURES_CONTINUOUS = re.compile(r"[A-Z]{2,3}R[12]")
_FUTURES_MONTH = re.compile(r"[A-Z]{2,3}[A-L]\d")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def contract_code(symbol: str) -> str:
    """Map a caller symbol to a Shioaji contract code.

    ``2330.TW`` / ``2330.TWO`` / ``2330`` become ``2330``. A futures
    root such as ``TXF`` becomes the continuous near month ``TXFR1``.
    ``TXFR1``, ``TXFR2``, and a delivery code such as ``TXFH6`` are
    already contract codes and are returned unchanged.
    """
    text = symbol.strip().upper()
    if text.endswith(".TWO"):
        text = text[: -len(".TWO")]
    elif text.endswith(".TW"):
        text = text[: -len(".TW")]
    if not text:
        raise ProviderError("symbol is empty")

    if _EQUITY.fullmatch(text):
        return text
    if _FUTURES_CONTINUOUS.fullmatch(text) or _FUTURES_MONTH.fullmatch(text):
        return text
    if _FUTURES_ROOT.fullmatch(text):
        return f"{text}R1"
    raise ProviderError(
        f"Shioaji does not cover {symbol.strip()!r}. "
        "Use yahoo, tiingo, or polygon for anything that is not a Taiwan "
        "stock, ETF, or index future."
    )


def _is_futures(code: str) -> bool:
    return bool(_FUTURES_CONTINUOUS.fullmatch(code) or _FUTURES_MONTH.fullmatch(code))


def _simulation_from_env(raw: str | None) -> bool:
    """Default is simulation. ``SHIOAJI_SIMULATION=0`` selects production."""
    if raw is None or raw.strip() == "":
        return True
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise DataProviderConfigError(
        f"SHIOAJI_SIMULATION must be 0/1/true/false, got {raw!r}."
    )


def _load_shioaji():
    try:
        import shioaji as sj
    except ImportError as exc:
        raise DataProviderConfigError(
            "Shioaji is selected but the shioaji package is not installed. "
            "Install it with: pip install -r requirements-shioaji.txt"
        ) from exc
    return sj


def _kbar_windows(first: pd.Timestamp, last: pd.Timestamp):
    cursor = first.normalize()
    last = last.normalize()
    while cursor <= last:
        chunk_last = min(last, cursor + pd.Timedelta(days=_KBAR_CHUNK_DAYS - 1))
        yield cursor, chunk_last
        cursor = chunk_last + pd.Timedelta(days=1)


def _daily_bounds(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    """First and last session midnights inside the half-open window."""
    first = start_ts.normalize()
    if first < start_ts:
        first += pd.Timedelta(days=1)
    last = (end_ts - pd.Timedelta(nanoseconds=1)).normalize()
    return first, last


def _session_dates(index: pd.DatetimeIndex, *, futures: bool) -> pd.DatetimeIndex:
    """Calendar date for stocks. Next regular session for futures nights.

    Minute stamps are Taiwan wall time. A futures bar at 15:00 or later
    belongs to the next weekday: Friday night, including Saturday
    morning, is Monday. A weekday holiday is not moved.
    """
    dates = index.normalize()
    if not futures:
        return dates
    night = (index.hour >= _NIGHT_SESSION_HOUR).astype("int64")
    shifted = dates + pd.to_timedelta(night, unit="D")
    weekday = shifted.day_of_week
    extra = (weekday == 5).astype("int64") * 2 + (weekday == 6).astype("int64")
    return shifted + pd.to_timedelta(extra, unit="D")


def _kbar_mapping(kbars) -> dict:
    if kbars is None:
        return {}
    if hasattr(kbars, "dict") and callable(kbars.dict):
        mapped = kbars.dict()
    elif isinstance(kbars, dict):
        mapped = kbars
    else:
        mapped = {name: getattr(kbars, name, None) for name in ("ts", *_OHLCV)}
    if not isinstance(mapped, dict):
        raise ProviderError("Shioaji returned an unexpected kbars payload.")
    return mapped


def _minutes_from_kbars(kbars) -> pd.DataFrame:
    data = _kbar_mapping(kbars)
    ts = data.get("ts")
    if ts is None or len(ts) == 0:
        return pd.DataFrame(columns=list(_OHLCV))

    missing = [col for col in _OHLCV if col not in data or data[col] is None]
    if missing:
        raise ProviderError(f"Shioaji kbars missing {', '.join(missing)}")
    lengths = {len(data[col]) for col in ("ts", *_OHLCV)}
    if len(lengths) != 1:
        raise ProviderError("Shioaji kbars columns have different lengths.")

    # The integer is nanoseconds, and the clock is already Taiwan local
    # (09:01 prints as 09:01). Converting that instant to Asia/Taipei
    # would move the bar to 17:01.
    stamps = pd.to_datetime(list(ts), unit="ns", utc=True, errors="coerce")
    if stamps.isna().any():
        raise ProviderError("Shioaji returned a timestamp that could not be parsed.")
    index = pd.DatetimeIndex(stamps.strftime("%Y-%m-%d %H:%M:%S"))
    frame = pd.DataFrame(
        {col: pd.Series(list(data[col]), dtype="float64").to_numpy() for col in _OHLCV},
        index=index,
    )
    return frame


def _to_daily(minutes: pd.DataFrame, *, futures: bool) -> pd.DataFrame:
    minutes = minutes.sort_index()
    minutes = minutes[~minutes.index.duplicated(keep="last")]
    labels = _session_dates(minutes.index, futures=futures)
    # ndarray, not an Index: groupby must follow row order, not reindex.
    daily = minutes.groupby(labels.to_numpy(), sort=True).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    daily.index = drop_timezone(pd.DatetimeIndex(pd.to_datetime(daily.index), name="Date"))
    return daily


class ShioajiProvider(PriceProvider):
    """Taiwan daily bars from SinoPac Shioaji (SDK 1.7+).

    Reads ``SHIOAJI_API_KEY`` and ``SHIOAJI_SECRET_KEY``. Simulation is
    the default; ``SHIOAJI_SIMULATION=0`` passes ``simulation=False``.
    Login is kept for the life of this object. No certificate is
    activated and no order is sent.

    Prices are unadjusted exchange prints. Stock and ETF volume is in
    board lots (張), futures volume is in contracts.
    """

    name = "shioaji"

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        simulation: bool | None = None,
    ):
        raw_key = api_key if api_key is not None else os.environ.get("SHIOAJI_API_KEY", "")
        raw_secret = (
            secret_key if secret_key is not None else os.environ.get("SHIOAJI_SECRET_KEY", "")
        )
        self._api_key = raw_key.strip()
        self._secret_key = raw_secret.strip()
        missing = []
        if not self._api_key:
            missing.append("SHIOAJI_API_KEY")
        if not self._secret_key:
            missing.append("SHIOAJI_SECRET_KEY")
        if missing:
            joined = " and ".join(missing)
            verb = "is" if len(missing) == 1 else "are"
            raise MissingApiKeyError(
                f"Shioaji is selected but {joined} {verb} not set."
            )
        if simulation is None:
            self._simulation = _simulation_from_env(os.environ.get("SHIOAJI_SIMULATION"))
        else:
            self._simulation = bool(simulation)
        self._api = None

    def fetch_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        window = request_window(start, end)
        if window is None:
            return empty_ohlcv()

        code = contract_code(symbol)
        start_ts, end_ts = window
        first, last = _daily_bounds(start_ts, end_ts)
        if first > last:
            return empty_ohlcv()

        futures = _is_futures(code)
        fetch_first = first - pd.Timedelta(days=_FUTURES_LOOKBACK_DAYS) if futures else first
        minutes = self._download_minutes(code, fetch_first, last)
        if minutes.empty:
            return empty_ohlcv()
        return finalize_ohlcv(_to_daily(minutes, futures=futures), start, end)

    def _download_minutes(self, code: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
        api = self._connect()
        try:
            contract = api.contracts.get(code)
        except Exception as exc:
            raise ProviderError(
                f"Shioaji contract lookup failed: {self._scrub(exc)}"
            ) from None
        if contract is None:
            raise ProviderError(f"Shioaji has no contract for {code}.")

        frames = []
        for chunk_start, chunk_end in _kbar_windows(first, last):
            try:
                raw = api.kbars(
                    contract=contract,
                    start=chunk_start.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    timeout=_KBAR_TIMEOUT_MS,
                )
            except Exception as exc:
                raise ProviderError(f"Shioaji kbars failed: {self._scrub(exc)}") from None
            frame = _minutes_from_kbars(raw)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return empty_ohlcv()
        return pd.concat(frames)

    def _connect(self):
        if self._api is not None:
            return self._api
        sj = _load_shioaji()
        try:
            api = sj.Shioaji(simulation=self._simulation)
            # subscribe_trade defaults to True and would attach order
            # callbacks. This provider only reads kbars.
            api.login(
                api_key=self._api_key,
                secret_key=self._secret_key,
                subscribe_trade=False,
            )
        except Exception as exc:
            raise ProviderError(f"Shioaji login failed: {self._scrub(exc)}") from None
        self._api = api
        return api

    def _scrub(self, exc: BaseException) -> str:
        text = " ".join(str(exc).split())
        secrets = [value for value in (self._secret_key, self._api_key) if value]
        for secret in sorted(secrets, key=len, reverse=True):
            text = text.replace(secret, "***")
        return text[:200]
