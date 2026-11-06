"""Replay this engine's fills inside a Nautilus backtest.

Nautilus is imported only by this module. ``load_crosscheck("nautilus")``
imports it after it has checked that the distribution is installed.

What Nautilus actually does
--------------------------
A market order submitted from ``on_bar`` fills against the book that bar
just left behind. The matching engine walks a bar Open, High, Low, Close,
so a normal bar can print away from the open. This adapter does not try to
recreate that walk. It hands Nautilus a one-price bar at the session open,
submits the primary trade list as market orders, and lets Nautilus match,
charge fees, and move the cash account. ``use_message_queue`` is off so a
sell on that bar credits cash before a later buy on the same bar is checked.

The position is then marked at the primary close. The flat bar's own last
price is the open, and using it would make every curve disagree even when
the fills are the same.

Limits worth keeping next to the numbers
-----------------------------------------
* Slippage is not applied here. The primary fill already includes it, so a
  non-zero slippage shows up as a price difference.
* Cash is a Nautilus USD balance, rounded to the cent. The fee model below
  charges the US rate plus the flat fee, or Taiwan brokerage, its minimum,
  and the sell-side transaction tax. Dividend withholding is unused, same
  as the primary engine.
* The instrument is a cash CFD with fractional size. Nautilus equities are
  whole shares, and a CFD's multiplier is fixed at 1, so a contract
  multiplier is folded into the order quantity.
* Bar volume is raised until the top of book can absorb the order. The
  historical volume is not a fill constraint.
* The account does not short. A sell larger than the position is denied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from backtest_system.engine.costs import MarketCosts, resolve_costs

PRICE_DECIMALS = 4
SIZE_DECIMALS = 6
_VENUE_NAME = "SIM"
_PRICE_STEP = 10 ** (-PRICE_DECIMALS)
_SIZE_STEP = 10 ** (-SIZE_DECIMALS)


@dataclass
class NautilusExecution:
    """Equity is marked at the primary closes. Fills use primary share units."""

    equity: pd.Series
    fills: pd.DataFrame
    problems: tuple[str, ...]


def execute(
    trades: pd.DataFrame,
    bars: pd.DataFrame | Mapping[str, pd.DataFrame],
    equity_index: pd.Index,
    initial_capital: float,
    *,
    costs: MarketCosts | Mapping[str, MarketCosts] | None = None,
    contract_multiplier: float | Mapping[str, float] = 1.0,
    symbol: str | None = None,
    mode: str = "pooled",
    symbols: Sequence[str] | None = None,
) -> NautilusExecution:
    """Run ``trades`` through Nautilus and return that account's close marks.

    ``pooled`` is one cash account (a target-weight book). ``sleeves`` is
    one account per requested name, each funded with
    ``initial_capital / len(symbols)``, summed the way an equal-weight book
    drops a name that never printed.
    """
    kind = (mode or "pooled").strip().lower()
    if kind not in {"pooled", "sleeves"}:
        raise ValueError(f"unknown nautilus account mode {mode!r}; expected pooled or sleeves")
    frames = _frames(bars, symbol)
    if not frames:
        raise ValueError("nautilus cross-check needs at least one Open/Close series")
    requested = [str(name) for name in symbols] if symbols is not None else list(frames)
    if kind == "sleeves":
        return _sleeves(
            trades,
            frames,
            equity_index,
            float(initial_capital),
            requested,
            costs,
            contract_multiplier,
        )
    return _pooled(
        trades,
        frames,
        equity_index,
        float(initial_capital),
        costs,
        contract_multiplier,
        symbol,
    )


def _sleeves(trades, frames, equity_index, capital, requested, costs, contract_multiplier) -> NautilusExecution:
    if not requested:
        raise ValueError("sleeves need the requested symbol list so each account is funded the same way")
    per = float(capital) / len(requested)
    curves: list[pd.Series] = []
    fills: list[pd.DataFrame] = []
    problems: list[str] = []
    for name in requested:
        frame = frames.get(name)
        if frame is None:
            continue
        own = _rows_for(trades, name, requested)
        part = _one_book(
            own,
            {name: frame},
            _dates_with_close(frame, equity_index),
            per,
            {name: _cost_for(name, costs)},
            contract_multiplier,
            name,
        )
        curves.append(part.equity.rename(name))
        if not part.fills.empty:
            fills.append(part.fills)
        problems.extend(part.problems)
    equity = _sum_sleeves(curves, equity_index)
    stacked = pd.concat(fills, ignore_index=True) if fills else _empty_fills()
    return NautilusExecution(equity, stacked, tuple(problems))


def _pooled(trades, frames, equity_index, capital, costs, contract_multiplier, symbol) -> NautilusExecution:
    return _one_book(
        trades,
        frames,
        equity_index,
        capital,
        {name: _cost_for(name, costs) for name in frames},
        contract_multiplier,
        symbol,
    )


def _one_book(trades, frames, equity_index, capital, costs, contract_multiplier, symbol) -> NautilusExecution:
    _require_nautilus()
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.models import FillModel
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import Bar, BarSpecification, BarType
    from nautilus_trader.model.enums import (
        AccountType,
        AggregationSource,
        AssetClass,
        BarAggregation,
        BookType,
        OmsType,
        OrderSide,
        PriceType,
        TimeInForce,
    )
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
    from nautilus_trader.model.instruments import Cfd
    from nautilus_trader.model.objects import Money, Price, Quantity
    from nautilus_trader.trading.strategy import Strategy

    index = _calendar(equity_index)
    opens, closes = _open_close(frames, index)
    if opens.empty:
        raise ValueError("nautilus cross-check has no dates to mark")
    names = list(opens.columns)
    safe = _safe_ids(names)
    scales = {name: _scale(name, contract_multiplier) for name in names}
    orders = _scheduled_orders(trades, names, symbol, scales)
    venue = Venue(_VENUE_NAME)
    instruments = {}
    bar_types = {}
    ids = {}
    for name in names:
        instrument_id = InstrumentId(Symbol(safe[name]), venue)
        ids[name] = instrument_id
        instruments[name] = Cfd(
            instrument_id=instrument_id,
            raw_symbol=Symbol(safe[name]),
            asset_class=AssetClass.EQUITY,
            quote_currency=USD,
            price_precision=PRICE_DECIMALS,
            size_precision=SIZE_DECIMALS,
            price_increment=Price.from_str(f"{_PRICE_STEP:.{PRICE_DECIMALS}f}"),
            size_increment=Quantity(_SIZE_STEP, SIZE_DECIMALS),
            lot_size=Quantity(_SIZE_STEP, SIZE_DECIMALS),
            ts_event=0,
            ts_init=0,
        )
        bar_types[name] = BarType(
            instrument_id,
            BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
            AggregationSource.EXTERNAL,
        )

    max_qty = max((row["qty"] for day_rows in orders.values() for row in day_rows), default=1.0)
    volume = Quantity(max(1_000_000.0, float(max_qty) * 8.0), SIZE_DECIMALS)
    nautilus_bars: list = []
    remaining: dict[pd.Timestamp, int] = {}
    for offset, name in enumerate(names):
        for day in opens.index:
            price = _bar_price(opens, closes, name, day, orders)
            if price is None:
                continue
            px = Price(price, PRICE_DECIMALS)
            if float(px) <= 0:
                raise ValueError(f"open for {name} on {day.date()} rounds to a non-positive Nautilus price")
            ts = int(day.value) + offset
            nautilus_bars.append(Bar(bar_types[name], px, px, px, px, volume, ts, ts))
            remaining[day] = remaining.get(day, 0) + 1
    if not nautilus_bars:
        raise ValueError("no finite open or close to build a Nautilus bar")

    class _Config(StrategyConfig, frozen=True):
        venue_name: str = _VENUE_NAME

    class _Replay(Strategy):
        def __init__(self, config: _Config) -> None:
            super().__init__(config)
            self.problems: list[str] = []
            self.marks: list[tuple[pd.Timestamp, float]] = []

        def on_start(self) -> None:
            for bar_type in bar_types.values():
                self.subscribe_bars(bar_type)

        def on_bar(self, bar) -> None:
            day = _day_from_ns(bar.ts_event)
            left = remaining.get(day)
            if left is None:
                return
            left -= 1
            remaining[day] = left
            if left > 0:
                return
            for row in orders.get(day, ()):
                self._send(day, row)
            self.marks.append((day, self._marked(day)))

        def _send(self, day, row) -> None:
            qty = Quantity(float(row["qty"]), SIZE_DECIMALS)
            if float(qty) <= 0:
                self.problems.append(
                    f"{row['symbol']} {row['action']} on {day.date()} rounded to zero at {SIZE_DECIMALS} decimals"
                )
                return
            side = OrderSide.BUY if row["action"] == "BUY" else OrderSide.SELL
            self.submit_order(
                self.order_factory.market(
                    instrument_id=ids[row["symbol"]],
                    order_side=side,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC,
                )
            )

        def _marked(self, day) -> float:
            account = self.portfolio.account(venue)
            cash = float(account.balance_total(USD))
            marked = cash
            for name, instrument_id in ids.items():
                held = 0.0
                for position in self.cache.positions_open(instrument_id=instrument_id):
                    held += _signed_qty(position)
                if held == 0.0:
                    continue
                close = closes.at[day, name] if day in closes.index else float("nan")
                if not math.isfinite(float(close)):
                    continue
                marked += held * float(close)
            return marked

        def on_order_denied(self, event) -> None:
            self.problems.append(f"denied: {event.reason}")

        def on_order_rejected(self, event) -> None:
            self.problems.append(f"rejected: {event.reason}")

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId(_next_trader_id()),
            logging=LoggingConfig(log_level="ERROR"),
        )
    )
    try:
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=USD,
            starting_balances=[Money(float(capital), USD)],
            book_type=BookType.L1_MBP,
            fill_model=FillModel(prob_slippage=0.0, random_seed=1),
            fee_model=_fee_model({str(ids[name]): costs[name] for name in names}),
            use_message_queue=False,
            bar_adaptive_high_low_ordering=False,
        )
        for instrument in instruments.values():
            engine.add_instrument(instrument)
        engine.add_data(nautilus_bars, sort=True)
        strategy = _Replay(_Config())
        engine.add_strategy(strategy)
        engine.run()
        report = engine.trader.generate_order_fills_report()
        problems = tuple(strategy.problems)
        marks = strategy.marks
    finally:
        engine.dispose()

    equity = _equity_series(marks, index, float(capital))
    fills = _fills_from_report(report, {safe[name]: name for name in names}, scales)
    return NautilusExecution(equity, fills, problems)


_TRADER_SEQ = 0


def _next_trader_id() -> str:
    global _TRADER_SEQ
    _TRADER_SEQ += 1
    return f"CROSS-{_TRADER_SEQ:03d}"


def _require_nautilus() -> None:
    import importlib.util

    if importlib.util.find_spec("nautilus_trader") is None:
        raise ImportError("nautilus_trader is not installed")


def _fee_model(costs_by_instrument: dict[str, MarketCosts]):
    """A Nautilus ``FeeModel`` closed over the primary schedules.

    The class is created on first use so importing this module does not
    import Nautilus, and so ``get_commission`` is a real ``FeeModel`` method.
    The schedules live in the closure: a Cython caller does not have to see
    a Python attribute on the instance.
    """
    from nautilus_trader.backtest.models import FeeModel
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.objects import Money

    schedules = dict(costs_by_instrument)

    class _Model(FeeModel):
        def get_commission(self, order, fill_qty, fill_px, instrument):
            schedule = schedules.get(str(instrument.id))
            if schedule is None:
                return Money(0, USD)
            notional = float(fill_qty) * float(fill_px) * float(instrument.multiplier)
            selling = int(order.side) == int(OrderSide.SELL)
            if schedule.market == "TW":
                fee = max(notional * schedule.brokerage_rate, schedule.brokerage_min)
                if selling:
                    fee += notional * schedule.sell_tax_rate
            else:
                fee = notional * schedule.commission_rate + schedule.commission_flat
            if not math.isfinite(fee) or fee < 0:
                fee = 0.0
            return Money(fee, USD)

    return _Model()


def _signed_qty(position) -> float:
    signed = getattr(position, "signed_qty", None)
    if signed is not None:
        return float(signed)
    qty = float(position.quantity)
    side = str(getattr(position, "side", "")).upper()
    if "SHORT" in side:
        return -qty
    return qty


def _frames(bars, symbol: str | None) -> dict[str, pd.DataFrame]:
    if isinstance(bars, pd.DataFrame):
        name = symbol or "asset"
        frame = _one_frame(bars)
        if frame is None:
            return {}
        return {str(name): frame}
    frames = {}
    for name, frame in dict(bars).items():
        prepared = _one_frame(frame)
        if prepared is not None:
            frames[str(name)] = prepared
    return frames


def _one_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    if frame is None or len(frame) == 0:
        return None
    if "Open" not in frame.columns or "Close" not in frame.columns:
        raise ValueError("bars need Open and Close; Nautilus is shown the open and marked at the close")
    out = pd.DataFrame(
        {
            "Open": pd.to_numeric(frame["Open"], errors="coerce"),
            "Close": pd.to_numeric(frame["Close"], errors="coerce"),
        },
        index=pd.DatetimeIndex(_day(stamp) for stamp in frame.index),
    )
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _calendar(index: pd.Index) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(_day(stamp) for stamp in index)
    days = days[~days.duplicated(keep="last")].sort_values()
    if len(days) == 0:
        raise ValueError("equity index is empty")
    return days


def _dates_with_close(frame: pd.DataFrame, equity_index: pd.Index) -> pd.DatetimeIndex:
    calendar = _calendar(equity_index)
    closes = frame["Close"].reindex(calendar)
    kept = closes.index[pd.to_numeric(closes, errors="coerce").map(math.isfinite)]
    if len(kept) == 0:
        return calendar
    return pd.DatetimeIndex(kept)


def _open_close(frames: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens = {}
    closes = {}
    for name, frame in frames.items():
        opens[name] = frame["Open"].reindex(index)
        closes[name] = frame["Close"].reindex(index)
    close_frame = pd.DataFrame(closes, index=index).sort_index()
    for column in close_frame.columns:
        close_frame[column] = pd.to_numeric(close_frame[column], errors="coerce").ffill()
    open_frame = pd.DataFrame(opens, index=index).sort_index()
    for column in open_frame.columns:
        open_frame[column] = pd.to_numeric(open_frame[column], errors="coerce")
    return open_frame, close_frame


def _bar_price(opens, closes, name, day, orders) -> float | None:
    """Trade dates must print at the open. Other dates only need a book to exist."""
    traded = any(row["symbol"] == name for row in orders.get(day, ()))
    open_px = opens.at[day, name]
    close_px = closes.at[day, name]
    if traded:
        if not math.isfinite(float(open_px)) or float(open_px) <= 0:
            raise ValueError(f"{name} trades on {day.date()} but that open is missing or not positive")
        return float(open_px)
    for candidate in (open_px, close_px):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def _scheduled_orders(trades, names, symbol, scales) -> dict[pd.Timestamp, tuple[dict, ...]]:
    grouped: dict[pd.Timestamp, list[dict]] = {}
    rows = _normalize_trades(trades, {name: None for name in names}, symbol)
    known = set(names)
    for row in rows:
        if row["symbol"] not in known:
            raise ValueError(
                f"trade symbol {row['symbol']!r} has no bars; Nautilus only sees {', '.join(names)}"
            )
        scale = scales[row["symbol"]]
        qty = float(row["shares"]) * scale
        if not math.isfinite(qty) or qty < 0:
            raise ValueError(f"trade quantity for {row['symbol']} is not a finite number >= 0")
        grouped.setdefault(row["date"], []).append(
            {"symbol": row["symbol"], "action": row["action"], "qty": qty}
        )
    return {day: tuple(items) for day, items in grouped.items()}


def _normalize_trades(trades: pd.DataFrame, frames, symbol: str | None) -> list[dict]:
    if trades is None or trades.empty:
        return []
    default = symbol
    if default is None and len(frames) == 1:
        default = next(iter(frames))
    rows = []
    for record in trades.to_dict(orient="records"):
        name = record.get("symbol")
        if name is None or (isinstance(name, float) and math.isnan(name)) or str(name) == "":
            if default is None:
                raise ValueError("trade is missing a symbol and more than one name was passed to Nautilus")
            name = default
        action = str(record.get("action") or "").upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError(f"unknown trade action {record.get('action')!r}")
        rows.append(
            {
                "date": _day(record["date"]),
                "symbol": str(name),
                "action": action,
                "shares": float(record["shares"]),
                "price": float(record["price"]),
                "fee": float(record.get("fee") or 0.0),
            }
        )
    return rows


def _rows_for(trades: pd.DataFrame, symbol: str, requested: Sequence[str]) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return trades if trades is not None else pd.DataFrame()
    if "symbol" not in trades.columns:
        if len(requested) == 1:
            out = trades.copy()
            out["symbol"] = symbol
            return out
        raise ValueError("sleeves need a symbol column when more than one name was requested")
    return trades.loc[trades["symbol"].astype(str) == str(symbol)].copy()


def _fills_from_report(report: pd.DataFrame, safe_to_name: dict[str, str], scales: dict[str, float]) -> pd.DataFrame:
    columns = ["date", "symbol", "action", "price", "shares", "fee"]
    if report is None or report.empty:
        return _empty_fills()
    rows = []
    frame = report.reset_index()
    for record in frame.to_dict(orient="records"):
        instrument = str(record.get("instrument_id") or "")
        safe = instrument.split(".", 1)[0]
        name = safe_to_name.get(safe, safe)
        scale = scales.get(name, 1.0)
        qty = float(record.get("filled_qty") or 0.0)
        rows.append(
            {
                "date": _day(record.get("ts_last")),
                "symbol": name,
                "action": str(record.get("side") or "").upper(),
                "price": float(record.get("avg_px") or 0.0),
                "shares": qty / scale if scale else qty,
                "fee": _money_amount(record.get("commissions")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _empty_fills() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "symbol", "action", "price", "shares", "fee"])


def _money_amount(value) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    if isinstance(value, (list, tuple)):
        return sum(_money_amount(item) for item in value)
    text = str(value).strip()
    if text in {"", "[]", "None", "nan"}:
        return 0.0
    total = 0.0
    for part in text.replace("[", "").replace("]", "").split(","):
        piece = part.strip().split()
        if not piece:
            continue
        total += abs(float(piece[0]))
    return total


def _equity_series(marks: list[tuple[pd.Timestamp, float]], index: pd.DatetimeIndex, capital: float) -> pd.Series:
    if marks:
        series = pd.Series({day: value for day, value in marks}, dtype=float).sort_index()
    else:
        series = pd.Series(dtype=float)
    equity = series.reindex(index).ffill()
    equity = equity.fillna(float(capital))
    equity.name = "nautilus"
    return equity


def _sum_sleeves(curves: list[pd.Series], equity_index: pd.Index) -> pd.Series:
    index = _calendar(equity_index)
    if not curves:
        equity = pd.Series(0.0, index=index, name="nautilus")
        return equity
    wide = pd.concat(curves, axis=1).sort_index().reindex(index).ffill().fillna(0.0)
    equity = wide.sum(axis=1)
    equity.name = "nautilus"
    return equity


def _cost_for(symbol: str, costs) -> MarketCosts:
    if isinstance(costs, Mapping):
        if symbol in costs:
            return costs[symbol]
        raise ValueError(f"no cost schedule for {symbol}")
    if costs is None:
        return resolve_costs(symbol=symbol, commission=0.0, slippage=0.0)
    return costs


def _scale(symbol: str, contract_multiplier) -> float:
    if isinstance(contract_multiplier, Mapping):
        value = float(contract_multiplier.get(symbol, 1.0))
    else:
        value = float(contract_multiplier)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("contract_multiplier must be a finite number > 0")
    return value


def _safe_ids(names: Sequence[str]) -> dict[str, str]:
    used: set[str] = set()
    out = {}
    for name in names:
        base = "".join(ch for ch in str(name).upper() if ch.isalnum()) or "ASSET"
        base = base[:24]
        candidate = base
        suffix = 2
        while candidate in used:
            tail = str(suffix)
            candidate = f"{base[: 24 - len(tail)]}{tail}"
            suffix += 1
        used.add(candidate)
        out[str(name)] = candidate
    return out


def _day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"invalid date: {value!r}")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def _day_from_ns(ts_event: int) -> pd.Timestamp:
    return _day(pd.Timestamp(int(ts_event), unit="ns"))
