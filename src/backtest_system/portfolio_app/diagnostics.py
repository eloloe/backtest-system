"""Evidence-based diagnostics for the interactive portfolio mini app."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest_system.engine.costs import DEFAULT_TW_BROKERAGE_MIN, DEFAULT_TW_BROKERAGE_RATE, DEFAULT_TW_STOCK_SELL_TAX
from backtest_system.portfolio_app.replay import ReplayCosts, replay_fixed_trades


@dataclass(frozen=True)
class Finding:
    key: str
    question: str
    title: str
    status: str
    severity: str
    why: str
    evidence: dict
    next_step: str


def diagnose_result(result) -> dict:
    equity = pd.Series(result.equity, dtype="float64")
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    benchmark = pd.Series(result.benchmark_equity, dtype="float64") if result.benchmark_equity is not None else pd.Series(dtype="float64")
    trades = result.trades if result.trades is not None else pd.DataFrame()
    prices = _price_pack(result)
    tape = _trade_tape(trades, prices)

    findings: list[Finding] = []
    findings.append(_benchmark_finding(equity, benchmark))
    findings.append(_trade_concentration_finding(tape))
    findings.append(_cost_stress_finding(result, prices, tape))
    findings.append(_time_split_finding(result))
    findings.append(_data_quality_finding(result, prices, tape))
    findings.append(_multiple_testing_finding(result))
    findings.append(_nautilus_scope_finding(result))

    counts = {"pass": 0, "fail": 0, "insufficient": 0, "not_applicable": 0}
    for finding in findings:
        counts[finding.status] = counts.get(finding.status, 0) + 1
    return {
        "summary": counts,
        "findings": [finding.__dict__ for finding in findings],
        "trade_count": int(len(tape)),
        "period_days": int(len(equity)),
        "return_mean": _finite(float(returns.mean() if len(returns) else 0.0)),
    }


def default_replay_costs(result) -> ReplayCosts:
    return ReplayCosts(
        commission_rate=float(result.config.commission),
        commission_flat=0.0,
        slippage_rate=float(result.config.slippage),
        tw_brokerage_rate=DEFAULT_TW_BROKERAGE_RATE,
        tw_brokerage_min=DEFAULT_TW_BROKERAGE_MIN,
        tw_sell_tax_rate=DEFAULT_TW_STOCK_SELL_TAX,
    )


def replay_preview(result, *, commission: float, slippage: float) -> dict:
    prices = _price_pack(result)
    tape = _trade_tape(result.trades, prices)
    costs = ReplayCosts(
        commission_rate=float(max(0.0, commission)),
        commission_flat=0.0,
        slippage_rate=float(max(0.0, slippage)),
        tw_brokerage_rate=DEFAULT_TW_BROKERAGE_RATE,
        tw_brokerage_min=DEFAULT_TW_BROKERAGE_MIN,
        tw_sell_tax_rate=DEFAULT_TW_STOCK_SELL_TAX,
    )
    replay = replay_fixed_trades(
        trades=tape,
        prices=prices,
        calendar=result.equity.index,
        initial_capital=float(result.config.initial_capital),
        costs=costs,
    )
    return {
        "equity": _series_points(replay.equity),
        "drawdown": _series_points(replay.drawdown),
        "problems": list(replay.problems),
        "trades_applied": replay.trades_applied,
    }


def _benchmark_finding(equity: pd.Series, benchmark: pd.Series) -> Finding:
    if len(benchmark) == 0:
        return Finding(
            key="benchmark_relative",
            question="賺的是行情，還是策略？",
            title="沒有基準可比對",
            status="insufficient",
            severity="medium",
            why="沒有 benchmark 時，無法判斷超額報酬是不是來自策略。",
            evidence={"benchmark": None},
            next_step="補同期間基準（例如 SPY 或 0050.TW），再看策略與基準的同期報酬差。",
        )
    left, right = equity.align(benchmark, join="inner")
    if len(left) < 2:
        return Finding(
            key="benchmark_relative",
            question="賺的是行情，還是策略？",
            title="可比對樣本太少",
            status="insufficient",
            severity="medium",
            why="重疊日期太短，無法用來判斷穩定超額。",
            evidence={"overlap_days": int(len(left))},
            next_step="延長區間或補齊資料，再看同期累積報酬差與回撤差。",
        )
    s_ret = float(left.iloc[-1] / left.iloc[0] - 1.0)
    b_ret = float(right.iloc[-1] / right.iloc[0] - 1.0)
    gap = s_ret - b_ret
    status = "pass" if gap > 0 else "fail"
    severity = "high" if status == "fail" else "low"
    return Finding(
        key="benchmark_relative",
        question="賺的是行情，還是策略？",
        title="相對基準同期報酬",
        status=status,
        severity=severity,
        why="策略若長期只跟著大盤同方向但沒有超額，策略價值有限。",
        evidence={
            "strategy_total_return": _finite(s_ret),
            "benchmark_total_return": _finite(b_ret),
            "excess_return": _finite(gap),
            "overlap_days": int(len(left)),
        },
        next_step="把區間切成多段，觀察超額是否只集中在少數行情期。",
    )


def _trade_concentration_finding(tape: pd.DataFrame) -> Finding:
    if tape.empty:
        return Finding(
            key="trade_concentration",
            question="獲利是不是只靠幾筆交易？",
            title="沒有成交可分析",
            status="insufficient",
            severity="medium",
            why="零成交時無法判斷獲利來源集中度。",
            evidence={"trade_count": 0},
            next_step="確認策略是否真的有觸發；若是買進持有，改用期間貢獻日分析。",
        )
    pnl = _realized_pnl(tape)
    if len(pnl) == 0:
        return Finding(
            key="trade_concentration",
            question="獲利是不是只靠幾筆交易？",
            title="沒有已實現平倉損益",
            status="insufficient",
            severity="medium",
            why="沒有賣出平倉時，交易層級獲利集中度不足以判斷。",
            evidence={"trade_count": int(len(tape)), "closed_trade_count": 0},
            next_step="拉長回測或加入平倉規則，再看獲利是否集中在少數幾筆。",
        )
    profits = pnl[pnl > 0]
    total_profit = float(profits.sum())
    if total_profit <= 0:
        status = "fail"
        ratio = None
    else:
        top3 = float(profits.sort_values(ascending=False).head(3).sum())
        ratio = top3 / total_profit
        status = "fail" if ratio > 0.7 else "pass"
    return Finding(
        key="trade_concentration",
        question="獲利是不是只靠幾筆交易？",
        title="獲利集中度（前 3 筆）",
        status=status,
        severity="high" if status == "fail" else "medium",
        why="若獲利幾乎靠少數交易，未來重現機率通常偏低。",
        evidence={
            "closed_trade_count": int(len(pnl)),
            "profitable_trade_count": int((pnl > 0).sum()),
            "total_realized_profit": _finite(float(total_profit)),
            "top3_profit_ratio": _finite(float(ratio)) if ratio is not None else None,
        },
        next_step="檢查前幾筆大賺是否由單一事件驅動，並在新區間重跑驗證。",
    )


def _cost_stress_finding(result, prices: Mapping[str, pd.DataFrame], tape: pd.DataFrame) -> Finding:
    if tape.empty:
        return Finding(
            key="cost_stress",
            question="成本加重後還撐得住嗎？",
            title="沒有成交，成本壓力測不到",
            status="not_applicable",
            severity="low",
            why="零成交時，成本敏感度無意義。",
            evidence={"trade_count": 0},
            next_step="用有成交的區間再做壓力測試。",
        )
    base_costs = default_replay_costs(result)
    stressed = ReplayCosts(
        commission_rate=base_costs.commission_rate * 2.0,
        commission_flat=base_costs.commission_flat,
        slippage_rate=base_costs.slippage_rate * 2.0,
        tw_brokerage_rate=base_costs.tw_brokerage_rate,
        tw_brokerage_min=base_costs.tw_brokerage_min,
        tw_sell_tax_rate=base_costs.tw_sell_tax_rate,
    )
    base = replay_fixed_trades(
        trades=tape,
        prices=prices,
        calendar=result.equity.index,
        initial_capital=float(result.config.initial_capital),
        costs=base_costs,
    )
    stressed_curve = replay_fixed_trades(
        trades=tape,
        prices=prices,
        calendar=result.equity.index,
        initial_capital=float(result.config.initial_capital),
        costs=stressed,
    )
    base_ret = float(base.equity.iloc[-1] / base.equity.iloc[0] - 1.0) if len(base.equity) else 0.0
    stress_ret = (
        float(stressed_curve.equity.iloc[-1] / stressed_curve.equity.iloc[0] - 1.0)
        if len(stressed_curve.equity)
        else 0.0
    )
    drop = base_ret - stress_ret
    status = "fail" if stress_ret <= 0 else "pass"
    return Finding(
        key="cost_stress",
        question="成本加重後還撐得住嗎？",
        title="固定成交成本壓力測試（費率與滑價 x2）",
        status=status,
        severity="high" if status == "fail" else "medium",
        why="若成本稍微變差就翻負，實務上容易因交易摩擦失效。",
        evidence={
            "base_total_return": _finite(base_ret),
            "stressed_total_return": _finite(stress_ret),
            "return_drop": _finite(drop),
            "scope": "固定成交重播，不含重新選股/再優化",
            "problems": list(dict.fromkeys(base.problems + stressed_curve.problems)),
        },
        next_step="若要驗證策略韌性，請用新成本設定完整重跑引擎（含訊號）。",
    )


def _time_split_finding(result) -> Finding:
    walk = result.walk_forward
    if walk is None:
        return Finding(
            key="time_split",
            question="換一段時間還行嗎？",
            title="目前是單段回測",
            status="insufficient",
            severity="medium",
            why="單段區間無法驗證時間外推能力。",
            evidence={"walk_forward": False},
            next_step="改跑 walk-forward，至少包含 train/gap/test 的多折樣本外結果。",
        )
    folds = list(getattr(walk, "folds", []) or [])
    if not folds:
        return Finding(
            key="time_split",
            question="換一段時間還行嗎？",
            title="walk-forward 沒有有效折數",
            status="insufficient",
            severity="high",
            why="沒有測試折時，時間切分驗證不存在。",
            evidence={"folds": 0},
            next_step="檢查區間長度與 train/test 設定，確保至少一折完整測試。",
        )
    separated = True
    for fold in folds:
        if not (fold.train_end < fold.test_start):
            separated = False
            break
    status = "pass" if separated else "fail"
    return Finding(
        key="time_split",
        question="換一段時間還行嗎？",
        title="訓練/測試切分檢查",
        status=status,
        severity="high" if status == "fail" else "medium",
        why="時間切分若重疊，會把未來資訊帶進訓練。",
        evidence={
            "folds": len(folds),
            "separated": separated,
            "holdout_opened": bool(getattr(result.config, "open_holdout", False)),
        },
        next_step="檢查折窗設定與日期範圍；必要時拉長資料，避免折數過少。",
    )


def _data_quality_finding(result, prices: Mapping[str, pd.DataFrame], tape: pd.DataFrame) -> Finding:
    notes = [str(note) for note in (result.notes or [])]
    missing = [note for note in notes if "沒有報價" in note]
    anomalies = []
    for row in tape.itertuples(index=False):
        frame = prices.get(row.symbol)
        if frame is None or row.date not in frame.index:
            anomalies.append(f"{row.symbol} {row.date.date()} trade has no matching bar")
    if missing or anomalies:
        return Finding(
            key="data_quality",
            question="哪些地方還沒驗證？",
            title="資料覆蓋或成交對齊有缺口",
            status="fail",
            severity="high",
            why="缺價或成交對不到 K 線時，績效與風險都可能被低估或高估。",
            evidence={
                "missing_price_notes": missing,
                "trade_alignment_issues": anomalies[:20],
                "issue_count": len(missing) + len(anomalies),
            },
            next_step="先補資料完整性，再重跑同區間比較前後差異。",
        )
    return Finding(
        key="data_quality",
        question="哪些地方還沒驗證？",
        title="目前未發現明顯資料缺口",
        status="pass",
        severity="low",
        why="沒有觀察到缺價或成交日期對不上 K 線的直接證據。",
        evidence={"notes_checked": len(notes), "trade_count": int(len(tape))},
        next_step="仍需額外檢查存活偏差、除權息與供應商版本差異（此項未自動驗證）。",
    )


def _multiple_testing_finding(result) -> Finding:
    selection = getattr(result, "selection", None) or {}
    if not selection:
        return Finding(
            key="multiple_testing",
            question="哪些地方還沒驗證？",
            title="沒有試驗數資訊",
            status="insufficient",
            severity="medium",
            why="不知道嘗試了幾組參數，就無法評估資料探勘風險。",
            evidence={"multiple_testing": "unknown"},
            next_step="若有參數搜尋，請輸出 trial 數與多重比較結果（DSR/Holm）。",
        )
    mt = selection.get("multiple_testing")
    if not mt:
        return Finding(
            key="multiple_testing",
            question="哪些地方還沒驗證？",
            title="N=1，不適用多重比較",
            status="not_applicable",
            severity="low",
            why="本次沒有在多組參數中挑最佳，無需 DSR/Holm。",
            evidence={"n_label": selection.get("n_label")},
            next_step="若後續做參數搜尋，再補多重比較。",
        )
    dsr = selection.get("dsr")
    status = "pass" if isinstance(dsr, (float, int)) and float(dsr) > 0 else "fail"
    return Finding(
        key="multiple_testing",
        question="哪些地方還沒驗證？",
        title="多重比較風險（DSR/Holm）",
        status=status,
        severity="medium",
        why="參數搜尋後需要折減樂觀偏誤，避免只記得最好看的一條。",
        evidence={"dsr": _finite(float(dsr)) if dsr is not None else None, "n_label": selection.get("n_label")},
        next_step="增加獨立樣本外，並對未存活的變體做原因分析。",
    )


def _nautilus_scope_finding(result) -> Finding:
    cross = getattr(result, "crosscheck", None) or {}
    if not cross:
        return Finding(
            key="nautilus_scope",
            question="哪些地方還沒驗證？",
            title="未執行第二帳本對帳",
            status="insufficient",
            severity="medium",
            why="沒有第二帳本，就少一層成交與淨值核對證據。",
            evidence={"crosscheck": None},
            next_step="加上 --cross-check nautilus，至少確認成交重播與主帳一致。",
        )
    ok = bool(cross.get("ok"))
    status = "pass" if ok else "fail"
    return Finding(
        key="nautilus_scope",
        question="哪些地方還沒驗證？",
        title="Nautilus 對帳結果",
        status=status,
        severity="medium" if ok else "high",
        why="Nautilus 只驗證獨立帳本與成交重播，不能證明訊號邏輯本身正確。",
        evidence={
            "ok": ok,
            "max_abs_diff": cross.get("max_abs_diff"),
            "material_fills": cross.get("material_fills"),
            "note": "獨立帳本驗證 != 獨立訊號驗證",
        },
        next_step="若要驗證訊號因果，請加上資料可得性與樣本外設計，不要只看對帳。",
    )


def _trade_tape(trades: pd.DataFrame | None, prices: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["date", "symbol", "action", "shares", "market"])
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "symbol" not in frame.columns:
        frame["symbol"] = next(iter(prices)) if prices else ""
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["action"] = frame["action"].astype(str).str.upper()
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
    frame["price"] = pd.to_numeric(frame.get("price"), errors="coerce").fillna(0.0)
    frame["fee"] = pd.to_numeric(frame.get("fee"), errors="coerce").fillna(0.0)
    if "market" not in frame.columns:
        frame["market"] = ""
    frame["market"] = frame["market"].fillna("").astype(str)
    return (
        frame[["date", "symbol", "action", "shares", "market", "price", "fee"]]
        .query("shares > 0 and action in ['BUY', 'SELL']")
        .sort_values(["date", "symbol", "action"], kind="stable")
        .reset_index(drop=True)
    )


def _realized_pnl(tape: pd.DataFrame) -> pd.Series:
    lots: dict[str, list[tuple[float, float]]] = {}
    out: list[float] = []
    for row in tape.itertuples(index=False):
        symbol = row.symbol
        if symbol not in lots:
            lots[symbol] = []
        if row.action == "BUY":
            cost_per_share = (float(row.price) * float(row.shares) + float(row.fee)) / float(row.shares)
            lots[symbol].append((float(row.shares), cost_per_share))
            continue
        remaining = float(row.shares)
        realized = 0.0
        sell_net_per_share = (float(row.price) * float(row.shares) - float(row.fee)) / float(row.shares)
        while remaining > 1e-12 and lots[symbol]:
            qty, cost = lots[symbol][0]
            take = min(qty, remaining)
            realized += (sell_net_per_share - cost) * take
            qty -= take
            remaining -= take
            if qty <= 1e-12:
                lots[symbol].pop(0)
            else:
                lots[symbol][0] = (qty, cost)
        out.append(realized)
    return pd.Series(out, dtype="float64")


def _price_pack(result) -> dict[str, pd.DataFrame]:
    packed = {}
    raw = getattr(result, "prices", None) or {}
    for symbol, frame in raw.items():
        if frame is None or frame.empty:
            continue
        safe = frame.copy()
        safe.index = pd.DatetimeIndex(safe.index).normalize()
        if "Open" in safe.columns and "Close" in safe.columns:
            packed[str(symbol).upper()] = safe[["Open", "Close"]]
    return packed


def _series_points(series: pd.Series) -> list[dict]:
    return [{"date": str(day.date()), "value": _finite(float(value))} for day, value in series.items()]


def _finite(value: float | None):
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)
