"""OHLCV entry point used by the app and the factor engine.

The provider comes from ``BACKTEST_DATA_PROVIDER`` (default ``yahoo``).
"""

from __future__ import annotations

import pandas as pd

from providers import get_provider
from providers.base import DataProviderConfigError, empty_ohlcv


def fetch_data(
    symbol: str,
    start_date: str,
    end_date: str,
    provider: str | None = None,
) -> pd.DataFrame:
    """取得 OHLCV。``provider`` 省略時用環境變數，預設 Yahoo。

    網路或解析錯誤回傳空表，跟原本的 Yahoo helper 一樣。
    來源設定錯誤（未知名稱、缺 API key、選了 Shioaji 卻沒裝套件）會往外丟。
    """
    client = get_provider(provider)
    try:
        return client.fetch_ohlcv(symbol, start_date, end_date)
    except DataProviderConfigError:
        raise
    except Exception as exc:
        print(f"Error fetching {symbol} from {client.name}: {exc}")
        return empty_ohlcv()
