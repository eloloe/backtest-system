from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def calculate_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """計算完整的績效指標。"""

    n_days = len(equity_curve)
    n_years = n_days / 252

    # 報酬指標
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # 清理報酬率
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()

    # 波動率
    annual_vol = r.std() * np.sqrt(252)

    # 夏普比率
    excess = r.mean() * 252 - risk_free_rate
    sharpe = excess / annual_vol if annual_vol > 0 else 0.0

    # Sortino 比率
    downside = r[r < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 1e-10
    sortino = excess / downside_vol if downside_vol > 0 else 0.0

    # 回撤指標
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar 比率
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # VaR 與 CVaR (95%)
    var_95 = float(np.percentile(r, 5))
    tail = r[r <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

    # 勝率與獲利虧損比
    pos = r[r > 0]
    neg = r[r < 0]
    win_rate = len(pos) / len(r) if len(r) > 0 else 0.0
    avg_win = float(pos.mean()) if len(pos) > 0 else 0.0
    avg_loss = float(neg.mean()) if len(neg) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    # 盈虧比率 (Profit Factor)
    gross_profit = pos.sum()
    gross_loss = abs(neg.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # 回復因子
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # 分布指標
    skewness = float(r.skew())
    kurtosis = float(r.kurt())

    # 最長回撤持續天數
    in_drawdown = drawdown < 0
    max_dd_duration = 0
    current_duration = 0
    for v in in_drawdown:
        if v:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0

    # Beta 與 Alpha
    beta, alpha = None, None
    if benchmark_returns is not None and len(benchmark_returns) > 10:
        aligned = pd.concat([r, benchmark_returns.rename("bench")], axis=1).dropna()
        if len(aligned) > 10:
            cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
            var = aligned.iloc[:, 1].var()
            beta = cov / var if var > 0 else 0.0
            strat_ann = aligned.iloc[:, 0].mean() * 252
            bench_ann = aligned.iloc[:, 1].mean() * 252
            alpha = strat_ann - risk_free_rate - beta * (bench_ann - risk_free_rate)

    # 格式化顯示
    metrics_display: Dict[str, str] = {
        "總報酬率": f"{total_return:.2%}",
        "年化報酬率 (CAGR)": f"{cagr:.2%}",
        "夏普比率": f"{sharpe:.3f}",
        "Sortino 比率": f"{sortino:.3f}",
        "Calmar 比率": f"{calmar:.3f}",
        "最大回撤": f"{max_drawdown:.2%}",
        "最長回撤天數": f"{max_dd_duration} 天",
        "年化波動率": f"{annual_vol:.2%}",
        "VaR (95%)": f"{var_95:.2%}",
        "CVaR (95%)": f"{cvar_95:.2%}",
        "勝率": f"{win_rate:.2%}",
        "平均獲利/虧損比": f"{win_loss_ratio:.3f}",
        "盈虧比率 (Profit Factor)": f"{profit_factor:.3f}",
        "回復因子": f"{recovery_factor:.3f}",
        "偏度 (Skewness)": f"{skewness:.3f}",
        "超額峰度 (Kurtosis)": f"{kurtosis:.3f}",
        "回測天數": f"{n_days:,} 天",
    }
    if beta is not None:
        metrics_display["Beta"] = f"{beta:.3f}"
        metrics_display["Alpha (年化)"] = f"{alpha:.2%}"

    metrics_raw: Dict[str, float] = {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "annual_vol": annual_vol,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "profit_factor": profit_factor,
        "recovery_factor": recovery_factor,
    }

    return metrics_display, metrics_raw


def calculate_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """將月度報酬率整理成 年份 x 月份 的樞紐表。"""
    try:
        monthly = equity_curve.resample("ME").last()
    except Exception:
        monthly = equity_curve.resample("M").last()

    monthly_rets = monthly.pct_change().dropna()
    if monthly_rets.empty:
        return pd.DataFrame()

    years = sorted(monthly_rets.index.year.unique())
    pivot = pd.DataFrame(np.nan, index=years, columns=range(1, 13))
    for date, ret in monthly_rets.items():
        pivot.loc[date.year, date.month] = ret

    return pivot


def calculate_annual_returns(equity_curve: pd.Series) -> pd.Series:
    """計算每年報酬率。"""
    try:
        annual = equity_curve.resample("YE").last()
    except Exception:
        annual = equity_curve.resample("Y").last()
    return annual.pct_change().dropna()
