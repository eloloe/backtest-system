"""Price provider selection.

``BACKTEST_DATA_PROVIDER`` picks the source (default ``yahoo``).
Tiingo and Polygon raise ``MissingApiKeyError`` when their key is unset.
"""

from __future__ import annotations

import os

from providers.base import (
    DataProviderConfigError,
    MissingApiKeyError,
    PriceProvider,
    ProviderError,
)
from providers.cache import CachingProvider
from providers.polygon import PolygonProvider
from providers.tiingo import TiingoProvider
from providers.yahoo import YahooProvider

_FACTORIES: dict[str, type[PriceProvider]] = {
    "yahoo": YahooProvider,
    "tiingo": TiingoProvider,
    "polygon": PolygonProvider,
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def get_provider(name: str | None = None) -> PriceProvider:
    """Build the provider named by ``name`` or ``BACKTEST_DATA_PROVIDER``."""
    selected = name if name is not None else os.environ.get("BACKTEST_DATA_PROVIDER", "yahoo")
    key = selected.strip().lower() or "yahoo"
    factory = _FACTORIES.get(key)
    if factory is None:
        known = ", ".join(_FACTORIES)
        raise DataProviderConfigError(
            f"Unknown data provider {selected!r}. Expected one of: {known}."
        )

    provider = factory()
    if not _cache_enabled():
        return provider

    cache_dir = os.environ.get("BACKTEST_DATA_CACHE_DIR", "data_cache")
    return CachingProvider(provider, cache_dir)


def _cache_enabled() -> bool:
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


__all__ = [
    "CachingProvider",
    "DataProviderConfigError",
    "MissingApiKeyError",
    "PolygonProvider",
    "PriceProvider",
    "ProviderError",
    "TiingoProvider",
    "YahooProvider",
    "get_provider",
]
