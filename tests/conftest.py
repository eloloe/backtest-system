import pytest


@pytest.fixture(autouse=True)
def _isolated_provider_env(monkeypatch):
    """Keep tests off the network and off the repo's data_cache/."""
    monkeypatch.delenv("BACKTEST_DATA_PROVIDER", raising=False)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setenv("BACKTEST_DATA_CACHE", "0")
