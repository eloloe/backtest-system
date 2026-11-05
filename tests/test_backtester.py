"""Next-open fills and US / TW cost math. No network."""

import math

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester, detect_market


def _ohlcv(opens, closes, start="2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(opens))
    open_s = pd.Series(opens, index=idx, dtype=float)
    close_s = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "Open": open_s,
            "High": np.maximum(open_s, close_s),
            "Low": np.minimum(open_s, close_s),
            "Close": close_s,
            "Volume": 1.0,
        },
        index=idx,
    )


def _signals(data: pd.DataFrame, values) -> pd.Series:
    return pd.Series(values, index=data.index, dtype=float)


def _zero() -> Backtester:
    return Backtester(initial_capital=100_000, commission=0.0, slippage=0.0)


def test_buy_signal_fills_at_next_open_not_same_close():
    data = _ohlcv([10.0, 20.0], [15.0, 20.0])
    # Day-0 close is 15. A same-bar fill would buy there and then earn 15 -> 20.
    bt = _zero()
    portfolio, trades = bt.run(data, _signals(data, [1, 1]), symbol="AAPL")

    assert list(portfolio.columns) == ["portfolio_value", "signal", "price", "returns"]
    assert portfolio["signal"].iloc[0] == 1
    assert portfolio["price"].iloc[0] == pytest.approx(15.0)
    assert portfolio["portfolio_value"].iloc[0] == pytest.approx(100_000)
    assert len(trades) == 1
    assert trades.iloc[0]["date"] == data.index[1]
    assert trades.iloc[0]["action"] == "BUY"
    assert trades.iloc[0]["price"] == pytest.approx(20.0)
    assert trades.iloc[0]["shares"] == pytest.approx(100_000 / 20.0)
    assert trades.iloc[0]["fee"] == pytest.approx(0.0)
    assert portfolio["portfolio_value"].iloc[1] == pytest.approx(100_000)


def test_exit_signal_sells_at_next_open_and_keeps_the_gap():
    data = _ohlcv([10.0, 10.0, 8.0], [10.0, 14.0, 8.0])
    portfolio, trades = _zero().run(data, _signals(data, [1, 0, 0]))

    assert list(trades["action"]) == ["BUY", "SELL"]
    assert trades.iloc[0]["date"] == data.index[1]
    assert trades.iloc[0]["price"] == pytest.approx(10.0)
    assert trades.iloc[1]["date"] == data.index[2]
    assert trades.iloc[1]["price"] == pytest.approx(8.0)
    # Signal is already flat on day 1, but the sell is still the day-2 open.
    assert portfolio["signal"].iloc[1] == 0
    assert portfolio["portfolio_value"].iloc[1] == pytest.approx(140_000)
    assert portfolio["portfolio_value"].iloc[2] == pytest.approx(80_000)


def test_last_bar_signal_is_not_closed_at_that_close():
    data = _ohlcv([10.0, 10.0, 10.0], [10.0, 10.0, 7.0])
    portfolio, trades = _zero().run(data, _signals(data, [1, 1, 0]))

    assert list(trades["action"]) == ["BUY"]
    assert trades.iloc[0]["date"] == data.index[1]
    assert portfolio["portfolio_value"].iloc[2] == pytest.approx(70_000)


def test_entry_earns_open_to_close_then_close_to_close():
    data = _ohlcv([9.0, 10.0, 11.0, 12.0], [9.0, 12.0, 11.0, 13.2])
    portfolio, trades = _zero().run(data, _signals(data, [1, 1, 1, 1]))

    shares = 100_000 / 10.0
    assert len(trades) == 1
    assert portfolio["portfolio_value"].iloc[0] == pytest.approx(100_000)
    assert portfolio["portfolio_value"].iloc[1] == pytest.approx(shares * 12.0)
    assert portfolio["portfolio_value"].iloc[2] == pytest.approx(shares * 11.0)
    assert portfolio["portfolio_value"].iloc[3] == pytest.approx(shares * 13.2)
    held = portfolio["returns"].iloc[2:]
    price_ret = data["Close"].pct_change().iloc[2:]
    assert held.tolist() == pytest.approx(price_ret.tolist())


def test_missing_open_defers_the_fill_until_the_signal_still_says_buy():
    data = _ohlcv([10.0, math.nan, 12.0], [10.0, 11.0, 12.0])
    portfolio, trades = _zero().run(data, _signals(data, [1, 1, 1]))

    assert trades.iloc[0]["date"] == data.index[2]
    assert trades.iloc[0]["price"] == pytest.approx(12.0)
    assert portfolio["portfolio_value"].iloc[1] == pytest.approx(100_000)


def test_signal_that_flips_before_a_valid_open_does_not_fill():
    data = _ohlcv([10.0, math.nan, 12.0], [10.0, 11.0, 12.0])
    portfolio, trades = _zero().run(data, _signals(data, [1, 0, 0]))

    assert trades.empty
    assert portfolio["portfolio_value"].tolist() == pytest.approx([100_000, 100_000, 100_000])


def test_nan_signal_is_forward_filled():
    data = _ohlcv([10.0, 10.0, 10.0, 9.0], [10.0, 10.0, 10.0, 9.0])
    signals = _signals(data, [1, np.nan, 0, 0])
    _, trades = _zero().run(data, signals)

    assert list(trades["action"]) == ["BUY", "SELL"]
    assert trades.iloc[1]["date"] == data.index[3]
    assert trades.iloc[1]["price"] == pytest.approx(9.0)


def test_us_cost_math_on_notional():
    data = _ohlcv([100.0, 100.0, 110.0, 120.0], [100.0, 105.0, 115.0, 125.0])
    capital = 100_000.0
    commission = 0.001
    slippage = 0.0005
    bt = Backtester(initial_capital=capital, commission=commission, slippage=slippage)
    portfolio, trades = bt.run(data, _signals(data, [1, 1, 0, 0]), symbol="MSFT")

    buy_fill = 100.0 * (1.0 + slippage)
    buy_notional = capital / (1.0 + commission)
    buy_fee = buy_notional * commission
    shares = buy_notional / buy_fill
    sell_fill = 120.0 * (1.0 - slippage)
    sell_notional = shares * sell_fill
    sell_fee = sell_notional * commission
    proceeds = sell_notional - sell_fee

    assert trades.iloc[0]["market"] == "US"
    assert trades.iloc[0]["price"] == pytest.approx(buy_fill)
    assert trades.iloc[0]["value"] == pytest.approx(buy_notional)
    assert trades.iloc[0]["fee"] == pytest.approx(buy_fee)
    assert trades.iloc[0]["shares"] == pytest.approx(shares)
    assert trades.iloc[1]["price"] == pytest.approx(sell_fill)
    assert trades.iloc[1]["fee"] == pytest.approx(sell_fee)
    assert trades.iloc[1]["value"] == pytest.approx(proceeds)
    assert portfolio["portfolio_value"].iloc[0] == pytest.approx(capital)
    assert portfolio["portfolio_value"].iloc[1] == pytest.approx(shares * 105.0)
    assert portfolio["portfolio_value"].iloc[2] == pytest.approx(shares * 115.0)
    assert portfolio["portfolio_value"].iloc[3] == pytest.approx(proceeds)


def test_bps_override_flat_fee_and_zero_bps():
    data = _ohlcv([10.0, 10.0], [10.0, 10.0])
    signals = _signals(data, [1, 1])

    flat = Backtester(initial_capital=1_000, commission_bps=0, slippage=0, commission_flat=10)
    _, trades = flat.run(data, signals)
    assert trades.iloc[0]["fee"] == pytest.approx(10.0)
    assert trades.iloc[0]["shares"] == pytest.approx(99.0)
    assert trades.iloc[0]["value"] == pytest.approx(990.0)

    bps = Backtester(initial_capital=100_000, commission=0.05, commission_bps=10, slippage_bps=50)
    _, bps_trades = bps.run(data, signals)
    fraction = Backtester(initial_capital=100_000, commission=0.001, slippage=0.005)
    _, fraction_trades = fraction.run(data, signals)
    assert bps_trades.iloc[0]["fee"] == pytest.approx(fraction_trades.iloc[0]["fee"])
    assert bps_trades.iloc[0]["price"] == pytest.approx(10.0 * 1.005)
    assert bps_trades.iloc[0]["price"] == pytest.approx(fraction_trades.iloc[0]["price"])

    too_small = Backtester(initial_capital=10, commission_bps=0, slippage=0, commission_flat=20)
    portfolio, no_trade = too_small.run(data, signals)
    assert no_trade.empty
    assert portfolio["portfolio_value"].iloc[-1] == pytest.approx(10.0)


def test_tw_stock_fees_ignore_us_commission():
    data = _ohlcv([500.0, 500.0, 500.0, 550.0], [500.0, 500.0, 520.0, 550.0])
    signals = _signals(data, [1, 1, 0, 0])
    capital = 1_000_000.0
    bt = Backtester(initial_capital=capital, commission=0.05, slippage=0.0)
    portfolio, trades = bt.run(data, signals, symbol="2330.TW")

    buy_notional = capital / (1.0 + 0.001425)
    buy_fee = buy_notional * 0.001425
    shares = buy_notional / 500.0
    sell_notional = shares * 550.0
    sell_fee = sell_notional * 0.001425 + sell_notional * 0.003
    proceeds = sell_notional - sell_fee

    assert trades.iloc[0]["market"] == "TW"
    assert trades.iloc[0]["fee"] == pytest.approx(buy_fee)
    assert trades.iloc[0]["fee"] > 20.0
    assert trades.iloc[0]["value"] == pytest.approx(buy_notional)
    assert trades.iloc[1]["price"] == pytest.approx(550.0)
    assert trades.iloc[1]["fee"] == pytest.approx(sell_fee)
    assert trades.iloc[1]["value"] == pytest.approx(proceeds)
    # 5% US commission would have been much larger than 14.25 bps.
    assert buy_fee < capital * 0.01
    assert portfolio["portfolio_value"].iloc[2] == pytest.approx(shares * 520.0)
    assert portfolio["portfolio_value"].iloc[3] == pytest.approx(proceeds)

    costs = bt.cost_model(symbol="2330.TW")
    assert costs.market == "TW"
    assert costs.sell_tax_rate == pytest.approx(0.003)
    assert costs.brokerage_rate == pytest.approx(0.001425)
    assert costs.brokerage_min == pytest.approx(20.0)
    assert costs.commission_rate == pytest.approx(0.0)
    assert costs.dividend_withholding == pytest.approx(0.0)


def test_tw_etf_sell_tax_and_override():
    data = _ohlcv([100.0, 100.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0])
    signals = _signals(data, [1, 1, 0, 0])
    capital = 1_000_000.0

    etf = Backtester(initial_capital=capital, commission=0.02, slippage=0.0)
    stock = Backtester(initial_capital=capital, commission=0.02, slippage=0.0)
    _, etf_trades = etf.run(data, signals, symbol="0050.TW")
    _, stock_trades = stock.run(data, signals, symbol="2330.TW")

    sell_notional = etf_trades.iloc[1]["shares"] * 100.0
    assert etf_trades.iloc[1]["fee"] == pytest.approx(
        sell_notional * 0.001425 + sell_notional * 0.001
    )
    assert stock_trades.iloc[1]["fee"] - etf_trades.iloc[1]["fee"] == pytest.approx(
        sell_notional * 0.002
    )
    assert etf.cost_model("00631L.TWO").sell_tax_rate == pytest.approx(0.001)

    forced = Backtester(initial_capital=capital, slippage=0.0, sell_tax_rate=0.003)
    _, forced_trades = forced.run(data, signals, symbol="0050.TW")
    assert forced_trades.iloc[1]["fee"] == pytest.approx(stock_trades.iloc[1]["fee"])


def test_tw_minimum_brokerage_binds_on_buy_and_sell():
    data = _ohlcv([50.0, 50.0, 50.0], [50.0, 50.0, 50.0])
    signals = _signals(data, [1, 0, 0])
    bt = Backtester(initial_capital=1_000, commission=0.0, slippage=0.0)
    _, trades = bt.run(data, signals, symbol="2330.TW")

    assert trades.iloc[0]["fee"] == pytest.approx(20.0)
    assert trades.iloc[0]["value"] == pytest.approx(980.0)
    assert trades.iloc[0]["shares"] == pytest.approx(19.6)
    sell_notional = 19.6 * 50.0
    assert trades.iloc[1]["fee"] == pytest.approx(20.0 + sell_notional * 0.003)
    assert trades.iloc[1]["value"] == pytest.approx(sell_notional - trades.iloc[1]["fee"])


def test_tw_slippage_moves_the_open_and_us_market_override_drops_the_tax():
    data = _ohlcv([100.0, 100.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0])
    signals = _signals(data, [1, 1, 0, 0])
    tw = Backtester(initial_capital=1_000_000, commission=0.0, slippage=0.0005)
    _, tw_trades = tw.run(data, signals, symbol="2330.tw")
    assert tw_trades.iloc[0]["price"] == pytest.approx(100.0 * 1.0005)
    assert tw_trades.iloc[1]["price"] == pytest.approx(100.0 * 0.9995)

    us = Backtester(initial_capital=1_000_000, commission=0.001, slippage=0.0)
    _, us_trades = us.run(data, signals, symbol="2330.TW", market="US")
    sell_notional = us_trades.iloc[1]["shares"] * 100.0
    assert us_trades.iloc[0]["market"] == "US"
    assert us_trades.iloc[1]["fee"] == pytest.approx(sell_notional * 0.001)
    assert us.cost_model(symbol="2330.TW", market="US").sell_tax_rate == pytest.approx(0.0)


def test_discounted_brokerage_rate():
    data = _ohlcv([100.0, 100.0], [100.0, 100.0])
    signals = _signals(data, [1, 1])
    rate = 0.001425 * 0.6
    bt = Backtester(
        initial_capital=1_000_000,
        slippage=0.0,
        brokerage_rate=rate,
        brokerage_min=1.0,
    )
    _, trades = bt.run(data, signals, symbol="2317.TW")
    notional = 1_000_000 / (1.0 + rate)
    assert trades.iloc[0]["fee"] == pytest.approx(notional * rate)


@pytest.mark.parametrize(
    ("symbol", "market", "expected"),
    [
        ("AAPL", None, "US"),
        ("BTC-USD", None, "US"),
        ("^TWII", None, "US"),
        ("ES=F", None, "US"),
        ("2330.TW", None, "TW"),
        (" 6488.two ", None, "TW"),
        (None, None, "US"),
        ("", None, "US"),
        ("2330.TW", "US", "US"),
        ("AAPL", "taiwan", "TW"),
        ("ES=F", "futures", "FUTURES"),
    ],
)
def test_detect_market(symbol, market, expected):
    assert detect_market(symbol, market) == expected


def test_unknown_market_and_negative_cost_raise():
    with pytest.raises(ValueError, match="unknown market"):
        detect_market("AAPL", "JP")
    data = _ohlcv([10.0, 10.0], [10.0, 10.0])
    signals = _signals(data, [1, 1])
    with pytest.raises(ValueError):
        Backtester(commission=-0.01).run(data, signals)
    with pytest.raises(ValueError, match="TW"):
        Backtester(sell_tax_rate=0.003).run(data, signals, symbol="AAPL")


def test_futures_multiplier_and_min_lot():
    data = _ohlcv([80.0, 80.0, 80.0, 100.0], [80.0, 90.0, 100.0, 100.0])
    signals = _signals(data, [1, 1, 0, 0])
    bt = Backtester(
        initial_capital=10_000,
        commission=0.0,
        slippage=0.0,
        contract_multiplier=50,
        min_lot=1,
    )
    portfolio, trades = bt.run(data, signals, market="FUTURES")

    assert list(trades["action"]) == ["BUY", "SELL"]
    assert trades.iloc[0]["shares"] == pytest.approx(2.0)
    assert trades.iloc[0]["value"] == pytest.approx(8_000.0)
    assert trades.iloc[0]["fee"] == pytest.approx(0.0)
    assert trades.iloc[1]["price"] == pytest.approx(100.0)
    assert trades.iloc[1]["value"] == pytest.approx(2 * 100 * 50)
    assert portfolio["portfolio_value"].iloc[0] == pytest.approx(10_000)
    assert portfolio["portfolio_value"].iloc[1] == pytest.approx(2_000 + 2 * 90 * 50)
    # Signal is flat on this bar; the sell still waits for the next open.
    assert portfolio["portfolio_value"].iloc[2] == pytest.approx(2_000 + 2 * 100 * 50)
    assert portfolio["portfolio_value"].iloc[3] == pytest.approx(12_000)

    blocked = Backtester(
        initial_capital=1_000,
        commission=0.0,
        slippage=0.0,
        contract_multiplier=50,
        min_lot=1,
    )
    blocked_pv, blocked_trades = blocked.run(data, signals, market="FUTURES")
    assert blocked_trades.empty
    assert blocked_pv["portfolio_value"].iloc[-1] == pytest.approx(1_000)


def test_futures_without_multiplier_still_runs_like_us():
    data = _ohlcv([10.0, 12.0, 11.0], [10.0, 13.0, 11.0])
    signals = _signals(data, [1, 1, 0])
    us = Backtester(initial_capital=50_000, commission=0.001, slippage=0.0005)
    fut = Backtester(initial_capital=50_000, commission=0.001, slippage=0.0005)
    us_pv, us_trades = us.run(data, signals, symbol="ES=F")
    fut_pv, fut_trades = fut.run(data, signals, symbol="ES=F", market="FUTURES")

    assert fut.cost_model(market="FUTURES").contract_multiplier == pytest.approx(1.0)
    assert fut.cost_model(market="FUTURES").min_lot is None
    assert fut.cost_model(market="FUTURES").dividend_withholding == pytest.approx(0.0)
    assert fut.cost_model(market="FUTURES").sell_tax_rate == pytest.approx(0.0)
    pd.testing.assert_series_equal(us_pv["portfolio_value"], fut_pv["portfolio_value"])
    assert us_trades["fee"].tolist() == pytest.approx(fut_trades["fee"].tolist())


def test_dividend_withholding_is_recorded_and_not_charged():
    opens = [10.0, 10.0, 12.0, 12.0]
    closes = [10.0, 11.0, 12.0, 12.0]
    data = _ohlcv(opens, closes)
    data["Dividends"] = [0.0, 5.0, 5.0, 0.0]
    signals = _signals(data, [1, 1, 0, 0])
    plain = _ohlcv(opens, closes)

    default = Backtester(initial_capital=20_000, commission=0.001, slippage=0.0005)
    other = Backtester(
        initial_capital=20_000,
        commission=0.001,
        slippage=0.0005,
        dividend_withholding=0.15,
    )
    assert default.cost_model(symbol="AAPL").dividend_withholding == pytest.approx(0.30)
    assert other.cost_model(symbol="AAPL").dividend_withholding == pytest.approx(0.15)

    pv_div, trades_div = default.run(data, signals, symbol="AAPL")
    pv_plain, _ = default.run(plain, signals, symbol="AAPL")
    pv_other, trades_other = other.run(data, signals, symbol="AAPL")
    pd.testing.assert_series_equal(pv_div["portfolio_value"], pv_plain["portfolio_value"])
    pd.testing.assert_series_equal(pv_div["portfolio_value"], pv_other["portfolio_value"])
    assert list(trades_div["action"]) == ["BUY", "SELL"]
    assert trades_div["fee"].tolist() == pytest.approx(trades_other["fee"].tolist())


def test_missing_open_column_raises():
    data = pd.DataFrame({"Close": [1.0, 2.0]})
    signals = pd.Series([1, 1])
    with pytest.raises(ValueError, match="Open"):
        Backtester().run(data, signals)


def test_legacy_run_signature_and_empty_frame():
    data = _ohlcv([10.0, 11.0], [10.0, 11.0])
    signals = pd.Series([1, 1], index=data.index)
    bt = Backtester(initial_capital=100_000, commission=0.001, slippage=0.0005)
    portfolio, trades = bt.run(data, signals)
    assert trades.iloc[0]["market"] == "US"
    assert portfolio["portfolio_value"].iloc[0] == pytest.approx(100_000)

    empty = pd.DataFrame(columns=["Open", "Close"])
    empty_pv, empty_trades = bt.run(empty, pd.Series(dtype=float))
    assert empty_pv.empty
    assert empty_trades.empty


def test_equal_weight_names_keep_their_own_next_bar_costs():
    """Same split the app uses: equal capital, one Backtester per symbol."""
    data = _ohlcv([100.0, 100.0, 100.0, 110.0], [100.0, 102.0, 104.0, 110.0])
    signals = _signals(data, [1, 1, 0, 0])
    initial = 200_000.0
    per_name = initial / 2
    curves = []
    fees = {}
    for sym in ("AAPL", "2330.TW"):
        bt = Backtester(initial_capital=per_name, commission=0.001, slippage=0.0005)
        portfolio, trades = bt.run(data, signals, symbol=sym)
        curves.append(portfolio["portfolio_value"])
        fees[sym] = trades["fee"].tolist()
        assert trades.iloc[0]["date"] == data.index[1]
        assert trades.iloc[1]["date"] == data.index[3]
        assert trades.iloc[0]["price"] == pytest.approx(data["Open"].iloc[1] * 1.0005)

    equity = pd.concat(curves, axis=1).ffill().fillna(0).sum(axis=1)
    assert equity.iloc[0] == pytest.approx(initial)
    assert fees["AAPL"][0] != pytest.approx(fees["2330.TW"][0])
    assert equity.iloc[-1] == pytest.approx(sum(curve.iloc[-1] for curve in curves))
