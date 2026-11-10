"""Read-only local Shioaji quote collector for execution-assumption evidence.

No trade API is called. Credentials are read from environment variables only.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtest_system.providers.base import DataProviderConfigError, MissingApiKeyError, ProviderError
from backtest_system.providers.shioaji import contract_code

DOC_SOURCES = (
    "https://sinotrade.github.io/tutor/market_data/streaming/stocks/",
    "https://sinotrade.github.io/tutor/market_data/streaming/futures/",
    "https://sinotrade.github.io/tutor/advanced/nonblock/",
)


@dataclass(frozen=True)
class CaptureConfig:
    symbols: tuple[str, ...]
    duration_seconds: float
    include_depth: bool = True
    simulation: bool | None = None


def capture_quote_fixture(
    *,
    symbols: list[str],
    duration_seconds: float,
    output_path: str | Path,
    include_depth: bool = True,
    simulation: bool | None = None,
) -> Path:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if not symbols:
        raise ValueError("at least one symbol is required")
    cleaned = tuple(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())
    if not cleaned:
        raise ValueError("symbols are empty")
    cfg = CaptureConfig(cleaned, float(duration_seconds), include_depth=bool(include_depth), simulation=simulation)
    payload = collect_quotes(cfg)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def collect_quotes(config: CaptureConfig) -> dict[str, Any]:
    sj = _load_shioaji()
    api_key = os.getenv("SHIOAJI_API_KEY")
    secret = os.getenv("SHIOAJI_SECRET_KEY")
    if not api_key or not secret:
        raise MissingApiKeyError("Shioaji quote capture needs SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY")

    use_sim = _simulation_mode(config.simulation)
    api = sj.Shioaji(simulation=use_sim)
    contract_map: dict[str, Any] = {}
    snapshots: list[dict[str, Any]] = []
    lock = threading.Lock()

    def on_quote(topic, quote):
        shot = _to_snapshot(topic, quote, include_depth=config.include_depth)
        if shot is None:
            return
        with lock:
            snapshots.append(shot)

    try:
        api.login(api_key=api_key, secret_key=secret)
        _register_callback(api, on_quote)
        for symbol in config.symbols:
            code = contract_code(symbol)
            contract = _resolve_contract(api, code)
            contract_map[symbol] = contract
            _subscribe(api, contract)
        time.sleep(config.duration_seconds)
    finally:
        for symbol, contract in contract_map.items():
            try:
                _unsubscribe(api, contract)
            except Exception:
                pass
        try:
            api.logout()
        except Exception:
            pass

    snapshots.sort(key=lambda row: row["captured_at"])
    return {
        "meta": {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "symbols": list(config.symbols),
            "duration_seconds": config.duration_seconds,
            "simulation": use_sim,
            "source": "shioaji-readonly-capture",
            "connectivity_verified": False,
            "doc_sources": list(DOC_SOURCES),
            "warning": "本檔只代表本機當下串流快照，不代表可成交價格，也不是下單回報。",
            "sdk_connection_smoke_test": "not_run_in_ci_requires_local_account",
        },
        "snapshots": snapshots,
    }


def _simulation_mode(explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = os.getenv("SHIOAJI_SIMULATION")
    if raw is None or raw.strip() == "":
        return True
    key = raw.strip().lower()
    if key in {"1", "true", "yes", "on"}:
        return True
    if key in {"0", "false", "no", "off"}:
        return False
    raise DataProviderConfigError(f"SHIOAJI_SIMULATION must be 0/1/true/false, got {raw!r}.")


def _load_shioaji():
    try:
        import shioaji as sj  # type: ignore
    except ImportError as exc:
        raise DataProviderConfigError(
            "Shioaji quote capture needs shioaji package. Install with: pip install -r requirements-shioaji.txt"
        ) from exc
    return sj


def _register_callback(api, callback):
    quote = api.quote
    if hasattr(quote, "set_on_bidask_stk_v1_callback"):
        quote.set_on_bidask_stk_v1_callback(callback)
        return
    if hasattr(quote, "set_quote_callback"):
        quote.set_quote_callback(callback)
        return
    raise ProviderError("Shioaji SDK does not expose a quote callback method supported by this collector")


def _resolve_contract(api, code: str):
    contracts = getattr(api, "Contracts", None)
    if contracts is None:
        raise ProviderError("Shioaji SDK missing Contracts")
    if code.endswith("R1") or code.endswith("R2") or (len(code) >= 4 and code[-2].isalpha() and code[-1].isdigit()):
        root = code[:3]
        try:
            return contracts.Futures[root][code]
        except Exception as exc:
            raise ProviderError(f"cannot resolve futures contract {code}") from exc
    try:
        return contracts.Stocks[code]
    except Exception as exc:
        raise ProviderError(f"cannot resolve stock contract {code}") from exc


def _subscribe(api, contract):
    quote = api.quote
    try:
        quote.subscribe(contract, quote_type="bidask", version="v1")
    except TypeError:
        quote.subscribe(contract)


def _unsubscribe(api, contract):
    quote = api.quote
    try:
        quote.unsubscribe(contract, quote_type="bidask", version="v1")
    except TypeError:
        quote.unsubscribe(contract)


def _to_snapshot(topic, quote, *, include_depth: bool):
    symbol = _get_symbol(topic, quote)
    if not symbol:
        return None
    bid = _first(_read_field(quote, "bid_price"), _read_field(quote, "bid"), _read_field(quote, "buy_price"))
    ask = _first(_read_field(quote, "ask_price"), _read_field(quote, "ask"), _read_field(quote, "sell_price"))
    if bid is None or ask is None:
        return None
    stamp = _timestamp_text(_read_field(quote, "datetime"), _read_field(quote, "ts"), _read_field(quote, "timestamp"))
    row = {
        "symbol": str(symbol).upper(),
        "timestamp": stamp,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "bid": float(bid),
        "ask": float(ask),
        "bid_size": _number(_first(_read_field(quote, "bid_volume"), _read_field(quote, "bid_size"))),
        "ask_size": _number(_first(_read_field(quote, "ask_volume"), _read_field(quote, "ask_size"))),
    }
    if include_depth:
        row["depth"] = {
            "bid_price": _list_numbers(_read_field(quote, "bid_price")),
            "ask_price": _list_numbers(_read_field(quote, "ask_price")),
            "bid_volume": _list_numbers(_read_field(quote, "bid_volume")),
            "ask_volume": _list_numbers(_read_field(quote, "ask_volume")),
        }
    return row


def _get_symbol(topic, quote):
    for candidate in (_read_field(quote, "code"), _read_field(quote, "symbol"), _topic_symbol(topic)):
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _topic_symbol(topic):
    text = str(topic or "")
    parts = [part for part in text.replace("/", ".").split(".") if part]
    if not parts:
        return None
    return parts[-1]


def _read_field(obj, name: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _first(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            item = value[0]
            if item is not None:
                return item
            continue
        return value
    return None


def _timestamp_text(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).strip()
            if text:
                return text
            continue
        if number > 10_000_000_000:
            number /= 1_000_000_000
        return datetime.fromtimestamp(number, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list_numbers(value):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    out = []
    for item in value:
        number = _number(item)
        if number is not None:
            out.append(number)
    return out
