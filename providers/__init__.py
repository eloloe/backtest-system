"""Compatibility import. The implementation is ``backtest_system.providers``.

This module stays a separate package object so ``providers.yahoo`` still
loads the shim below, which is the module tests patch.
"""

from _bootstrap import ensure_src

ensure_src()

from backtest_system.providers import (
    CachingProvider,
    DataProviderConfigError,
    MissingApiKeyError,
    PolygonProvider,
    PriceProvider,
    ProviderError,
    ShioajiProvider,
    TiingoProvider,
    YahooProvider,
    get_provider,
)

__all__ = [
    "CachingProvider",
    "DataProviderConfigError",
    "MissingApiKeyError",
    "PolygonProvider",
    "PriceProvider",
    "ProviderError",
    "ShioajiProvider",
    "TiingoProvider",
    "YahooProvider",
    "get_provider",
]
