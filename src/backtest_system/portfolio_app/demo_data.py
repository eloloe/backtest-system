"""Deterministic synthetic data for public, credential-free mini app demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoDataset:
    prices: dict[str, pd.DataFrame]
    benchmark_close: pd.Series


def build_demo_dataset(start: str = "2022-01-03", periods: int = 260) -> DemoDataset:
    index = pd.bdate_range(start=start, periods=periods)
    t = np.arange(periods, dtype="float64")
    close = 100 + np.sin(t / 8.0) * 6 + t * 0.18
    open_ = close * (1 + np.sin(t / 5.0) * 0.002)
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    volume = np.full(periods, 1_000_000.0)
    frame = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )
    bench = pd.Series(100 + t * 0.12, index=index, name="benchmark")
    return DemoDataset(prices={"DEMO": frame}, benchmark_close=bench)
