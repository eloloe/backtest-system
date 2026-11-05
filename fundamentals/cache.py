"""Parquet cache for normalized statement panels.

Shares ``BACKTEST_DATA_CACHE`` and ``BACKTEST_DATA_CACHE_DIR`` with prices.
Files live under ``data_cache/fundamentals/``. Empty panels are not stored.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import pandas as pd

from fundamentals.statements import normalize_statements
from providers.base import DataProviderConfigError

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def cache_enabled() -> bool:
    raw = os.environ.get("BACKTEST_DATA_CACHE")
    if raw is None or raw.strip() == "":
        return True
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise DataProviderConfigError(
        f"BACKTEST_DATA_CACHE must be 0/1/true/false, got {raw!r}."
    )


def cache_ttl_days() -> int:
    raw = os.environ.get("BACKTEST_FUNDAMENTALS_CACHE_TTL_DAYS", "1").strip() or "1"
    try:
        days = int(raw)
    except ValueError as exc:
        raise DataProviderConfigError(
            f"BACKTEST_FUNDAMENTALS_CACHE_TTL_DAYS must be an integer, got {raw!r}."
        ) from exc
    if days < 0:
        raise DataProviderConfigError(
            "BACKTEST_FUNDAMENTALS_CACHE_TTL_DAYS must be >= 0."
        )
    return days


def cache_directory() -> Path:
    return Path(os.environ.get("BACKTEST_DATA_CACHE_DIR", "data_cache"))


def statement_cache_path(source: str, symbol: str, lag_days: int) -> Path:
    symbol_s = symbol.strip().upper()
    raw = f"v1|{source}|{symbol_s}|lag={int(lag_days)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", symbol_s)[:40] or "symbol"
    return (
        cache_directory()
        / "fundamentals"
        / source
        / f"{safe}_lag{int(lag_days)}_{digest}.parquet"
    )


def read_statement_cache(source: str, symbol: str, lag_days: int) -> pd.DataFrame | None:
    if not cache_enabled():
        return None
    path = statement_cache_path(source, symbol, lag_days)
    if not path.exists():
        return None
    ttl_days = cache_ttl_days()
    age = time.time() - path.stat().st_mtime
    if ttl_days == 0 or age > ttl_days * 86400:
        return None
    try:
        frame = pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return None
    frame = normalize_statements(frame)
    if frame.empty:
        return None
    return frame


def write_statement_cache(source: str, symbol: str, lag_days: int, frame: pd.DataFrame) -> None:
    if not cache_enabled() or frame is None or frame.empty:
        return
    path = statement_cache_path(source, symbol, lag_days)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        normalize_statements(frame).to_parquet(path, engine="pyarrow")
    except Exception:
        return
