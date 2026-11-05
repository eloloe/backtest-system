"""Local Parquet cache in front of a price provider."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd

from backtest_system.providers.base import PriceProvider, finalize_ohlcv, parse_bound


class CachingProvider(PriceProvider):
    """Disk cache keyed by provider, symbol, and the requested window.

    The hash includes a version prefix so a change in adjustment rules
    does not reuse old files. Empty results are not cached.
    """

    def __init__(self, inner: PriceProvider, cache_dir: str | os.PathLike[str]):
        self._inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir)

    def fetch_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self._path(symbol, start, end)
        cached = self._read(path, start, end)
        if cached is not None:
            return cached

        fresh = finalize_ohlcv(self._inner.fetch_ohlcv(symbol, start, end), start, end)
        if fresh.empty:
            return fresh
        self._write(path, fresh)
        return fresh

    def _path(self, symbol: str, start: str, end: str) -> Path:
        symbol_s = symbol.strip()
        start_s = parse_bound(start).strftime("%Y-%m-%d")
        end_s = parse_bound(end).strftime("%Y-%m-%d")
        raw = f"v1|{self.name}|{symbol_s}|{start_s}|{end_s}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", symbol_s)[:40] or "symbol"
        return self.cache_dir / self.name / f"{safe}_{start_s}_{end_s}_{digest}.parquet"

    def _read(self, path: Path, start: str, end: str) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
            return finalize_ohlcv(frame, start, end)
        except Exception:
            # Unreadable or the wrong shape: treat as a miss and refetch.
            return None

    def _write(self, path: Path, frame: pd.DataFrame) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, engine="pyarrow")
        except Exception:
            # A cache write must not fail a fetch that already succeeded.
            return
