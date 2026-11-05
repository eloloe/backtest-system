"""Deflated Sharpe ratio and Holm's step-down correction.

Bailey and López de Prado, "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality" (Journal of
Portfolio Management, 2014). DSR is the probabilistic Sharpe ratio
evaluated at the expected maximum Sharpe of ``N`` independent trials,
not at zero.

Sharpes inside the formula are per observation. Annualized numbers are
only for the report: multiply by ``sqrt(periods_per_year)``. The
risk-free rate stays zero unless the caller passes the same annual rate
already used in ``metrics``; it is then subtracted as ``rate / periods``.
Kurtosis is the raw fourth moment (a normal sample is 3). Pandas reports
the excess, so estimates taken from a series add 3 back.

A trial, for this codebase, is one parameter set scored on one
walk-forward training window. ``N=1`` means the run did not choose among
alternatives. The report then shows the probabilistic Sharpe against zero
and does not call that number a deflated Sharpe. Trials share prices, so
they are not independent; treating them as independent raises the hurdle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

# Euler–Mascheroni. The expected maximum of N standard normals uses it
# as the weight between two inverse-cdf points. See Bailey and López de Prado.
_EULER = 0.5772156649015328606
_DEFAULT_PERIODS = 252


@dataclass(frozen=True)
class SharpeAssessment:
    """One PSR / DSR calculation. Probabilities are None when the moments break."""

    n_obs: int
    n_trials: int
    multiple_testing: bool
    observed_sharpe: float
    observed_sharpe_annual: float
    skewness: float
    kurtosis: float
    sharpe_variance: float | None
    benchmark_sharpe: float
    benchmark_sharpe_annual: float
    psr: float | None
    dsr: float | None
    note: str

    def as_dict(self) -> dict:
        return {
            "n_obs": self.n_obs,
            "n_trials": self.n_trials,
            "multiple_testing": self.multiple_testing,
            "observed_sharpe": self.observed_sharpe,
            "observed_sharpe_annual": self.observed_sharpe_annual,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "sharpe_variance": self.sharpe_variance,
            "benchmark_sharpe": self.benchmark_sharpe,
            "benchmark_sharpe_annual": self.benchmark_sharpe_annual,
            "psr": self.psr,
            "dsr": self.dsr,
            "note": self.note,
        }


@dataclass(frozen=True)
class CorrectionRow:
    label: str
    statistic: float | None
    p_value: float
    rank: int
    threshold: float
    adjusted_p: float
    reject: bool


@dataclass(frozen=True)
class CorrectionResult:
    """Holm or Benjamini–Hochberg decisions at one family-wise (or FDR) alpha."""

    method: str
    alpha: float
    rows: tuple[CorrectionRow, ...]
    n_rejected: int
    survivors: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n": len(self.rows),
            "n_rejected": self.n_rejected,
            "survivors": list(self.survivors),
            "rows": [
                {
                    "label": row.label,
                    "statistic": row.statistic,
                    "p_value": row.p_value,
                    "rank": row.rank,
                    "threshold": row.threshold,
                    "adjusted_p": row.adjusted_p,
                    "reject": row.reject,
                }
                for row in self.rows
            ],
        }


def assess_sharpe(
    returns=None,
    *,
    n_trials: int = 1,
    sharpe: float | None = None,
    n_obs: int | None = None,
    skewness: float | None = None,
    kurtosis: float | None = None,
    kurtosis_excess: bool = False,
    sharpe_variance: float | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _DEFAULT_PERIODS,
    annualized: bool = False,
) -> SharpeAssessment:
    """Probability that the observed Sharpe clears the selection hurdle.

    Pass ``returns`` when the path is available. Moments then come from
    that series. ``sharpe_variance`` is optional and is only the variance
    used to place the expected-maximum hurdle. The PSR standard error
    still comes from the observed series (or from the moments passed in).
    Annualized inputs are converted with ``sqrt(periods_per_year)`` and,
    for a variance, with ``periods_per_year`` itself.
    """
    periods = _periods(periods_per_year)
    trials = _trials(n_trials)
    scale = math.sqrt(periods)

    if returns is not None:
        clean, mean, std, skew, kurt, count = _moments_from_returns(
            returns, risk_free_rate, periods
        )
        del clean
        if count < 2 or not math.isfinite(std) or std <= 0:
            per = 0.0
            skew = 0.0 if not math.isfinite(skew) else skew
            kurt = 3.0 if not math.isfinite(kurt) else kurt
            note = "報酬的波動為零或樣本短於兩點，夏普沒有估計誤差可談。"
            return _pack(
                count=max(count, 0),
                trials=trials,
                per=per,
                scale=scale,
                skew=skew,
                kurt=kurt,
                variance=None,
                hurdle=0.0,
                psr=None,
                dsr=None,
                note=note,
            )
        per = mean / std
    else:
        if sharpe is None or n_obs is None or skewness is None or kurtosis is None:
            raise ValueError("pass returns, or sharpe, n_obs, skewness, and kurtosis")
        count = _obs(n_obs)
        per = float(sharpe)
        if annualized:
            per = per / scale
        skew = float(skewness)
        kurt = float(kurtosis)
        if kurtosis_excess:
            kurt += 3.0
        if count < 2:
            return _pack(
                count=count,
                trials=trials,
                per=per,
                scale=scale,
                skew=skew,
                kurt=kurt,
                variance=None,
                hurdle=0.0,
                psr=None,
                dsr=None,
                note="樣本短於兩點，不計算 PSR。",
            )

    if not all(math.isfinite(x) for x in (per, skew, kurt)):
        return _pack(
            count=count,
            trials=trials,
            per=0.0 if not math.isfinite(per) else per,
            scale=scale,
            skew=0.0,
            kurt=3.0,
            variance=None,
            hurdle=0.0,
            psr=None,
            dsr=None,
            note="偏度或峰度不是有限數，不計算 PSR。",
        )

    var_term = _variance_term(per, skew, kurt)
    analytic = var_term / (count - 1) if var_term > 0 else None
    variance = _as_variance(sharpe_variance, annualized, periods)
    if variance is None:
        variance = analytic

    psr = _psr(per, 0.0, count, var_term)
    if trials <= 1:
        return _pack(
            count=count,
            trials=trials,
            per=per,
            scale=scale,
            skew=skew,
            kurt=kurt,
            variance=variance,
            hurdle=0.0,
            psr=psr,
            dsr=None,
            note=(
                "N=1，這次沒有在一組以上的參數裡做選擇，不計算 Deflated Sharpe。"
                "PSR 是這一段扣過成本的夏普高於零的機率。"
            ),
        )

    if variance is None or not math.isfinite(variance) or variance < 0:
        return _pack(
            count=count,
            trials=trials,
            per=per,
            scale=scale,
            skew=skew,
            kurt=kurt,
            variance=variance,
            hurdle=0.0,
            psr=psr,
            dsr=None,
            note="峰度與偏度使夏普變異估成負的，Deflated Sharpe 停在這裡。",
        )

    hurdle = expected_maximum_sharpe(trials, math.sqrt(variance), mean=0.0)
    dsr = _psr(per, hurdle, count, var_term)
    return _pack(
        count=count,
        trials=trials,
        per=per,
        scale=scale,
        skew=skew,
        kurt=kurt,
        variance=variance,
        hurdle=hurdle,
        psr=psr,
        dsr=dsr,
        note=(
            "Deflated Sharpe 是觀測夏普大於 N 個獨立試驗的期望最大夏普的機率"
            "（Bailey and López de Prado, 2014）。"
            "公式裡的夏普是單期的；年化只供對照，乘上 √期間數。"
            "無風險利率沿用績效表，並以 rate / 期間數從每期報酬扣除。"
        ),
    )


def expected_maximum_sharpe(n_trials: int, sharpe_std: float, mean: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe estimates.

    ``N=1`` returns ``mean`` (the null benchmark, usually zero). The
    extreme-value weights are only defined once there is something to maximize.
    """
    trials = _trials(n_trials)
    width = float(sharpe_std)
    center = float(mean)
    if trials <= 1 or not math.isfinite(width) or width <= 0:
        return center
    z_one = float(norm.ppf(1.0 - 1.0 / trials))
    z_two = float(norm.ppf(1.0 - 1.0 / (trials * math.e)))
    haircut = (1.0 - _EULER) * z_one + _EULER * z_two
    return center + width * haircut


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skewness: float,
    kurtosis: float,
) -> float | None:
    """PSR of Bailey and López de Prado. ``kurtosis`` is non-excess (normal = 3)."""
    term = _variance_term(float(observed_sharpe), float(skewness), float(kurtosis))
    return _psr(float(observed_sharpe), float(benchmark_sharpe), _obs(n_obs), term)


def sharpe_pvalue(returns, risk_free_rate: float = 0.0, periods_per_year: int = _DEFAULT_PERIODS) -> float | None:
    """One-sided p-value for the hypothesis that the true Sharpe is at most zero."""
    assessment = assess_sharpe(
        returns,
        n_trials=1,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    if assessment.psr is None or not math.isfinite(assessment.psr):
        return None
    return float(min(1.0, max(0.0, 1.0 - assessment.psr)))


def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> CorrectionResult:
    """Holm–Bonferroni step-down. Reject while ``p_(k) <= alpha / (m - k + 1)``."""
    names = list(labels) if labels is not None else [str(i) for i in range(len(p_values))]
    if len(names) != len(p_values):
        raise ValueError("labels and p_values must have the same length")
    items = [(name, None, p) for name, p in zip(names, p_values)]
    return holm_decisions(items, alpha=alpha)


def holm_decisions(items: Sequence, alpha: float = 0.05) -> CorrectionResult:
    """Holm on ``(label, statistic, p)`` rows. Also accepts mappings with those keys.

    ``statistic`` may be a Sharpe or any other test statistic kept for the report.
    The decision uses the p-value. The first p-value that misses its threshold
    stops the step-down: that row and every larger p-value stay.
    """
    return _step_correction(items, alpha, method="holm")


def benjamini_hochberg(items: Sequence, alpha: float = 0.05) -> CorrectionResult:
    """Benjamini–Hochberg step-up at false-discovery rate ``alpha``.

    Shipped beside Holm because it is the same loop. Holm remains the decision
    this research path reports.
    """
    return _step_correction(items, alpha, method="bh")


def summarize_selection(
    returns,
    trials: Sequence,
    *,
    risk_free_rate: float = 0.0,
    alpha: float = 0.05,
    n_param_sets: int | None = None,
    n_folds_searched: int | None = None,
    sample: str = "樣本外",
    sharpe_variance: float | None = None,
) -> dict:
    """DSR on ``returns`` plus Holm on the trial p-values.

    Each trial is ``(label, sharpe, p_value)`` or a mapping with those keys.
    Rows with a missing p-value still count toward ``N`` (the look happened)
    and stay out of the Holm family.
    """
    parsed = [_trial(item) for item in trials]
    n_trials = len(parsed)
    assessment = assess_sharpe(
        returns,
        n_trials=n_trials if n_trials > 1 else 1,
        risk_free_rate=risk_free_rate,
        sharpe_variance=sharpe_variance,
    )
    summary = assessment.as_dict()
    summary["sample"] = sample
    summary["alpha"] = float(alpha)
    summary["n_param_sets"] = n_param_sets
    summary["n_folds_searched"] = n_folds_searched
    summary["n_label"] = _n_label(n_trials, n_param_sets, n_folds_searched)
    summary["holm"] = None
    summary["bh"] = None
    if n_trials > 1:
        summary["note"] = (
            f"N={summary['n_label']}。一次試驗是一組參數在一折訓練窗上的分數；"
            "同一組參數換一折再算一次。"
            "這些試驗共用價格，並不獨立，獨立假設會把門檻抬高。"
            + summary["note"]
        )
    usable = [row for row in parsed if row[2] is not None]
    if n_trials > 1 and len(usable) >= 2:
        summary["holm"] = holm_decisions(usable, alpha=alpha).as_dict()
        summary["bh"] = benjamini_hochberg(usable, alpha=alpha).as_dict()
    elif n_trials > 1:
        summary["note"] = summary["note"] + " 有 p 值的試驗少於兩個，Holm 先不做。"
    if sample.startswith("鎖定") and n_trials > 1:
        summary["note"] = (
            f"{sample}沿用選參時的 N={summary['n_label']}，這一段沒有再搜尋。"
            + summary["note"]
        )
    summary["display"] = selection_display(summary)
    return summary


def selection_display(summary: Mapping) -> dict[str, str]:
    """Short lines for the metrics table and the terminal."""
    if not summary.get("multiple_testing"):
        lines = {"多重檢定": "不適用（N=1）"}
        psr = summary.get("psr")
        if isinstance(psr, float) and math.isfinite(psr):
            lines["PSR (SR>0)"] = f"{psr:.3f}"
        return lines
    lines = {"試驗次數 N": str(summary.get("n_label") or summary.get("n_trials"))}
    dsr = summary.get("dsr")
    lines["Deflated Sharpe (機率)"] = f"{dsr:.3f}" if isinstance(dsr, float) and math.isfinite(dsr) else "算不出"
    hurdle = summary.get("benchmark_sharpe_annual")
    if isinstance(hurdle, float) and math.isfinite(hurdle):
        lines["期望最大 Sharpe"] = f"{hurdle:.3f}"
    holm = summary.get("holm") or {}
    if holm:
        alpha = float(summary.get("alpha") or 0.05)
        lines["Holm 存活"] = f"{holm['n_rejected']} / {holm['n']}（α={alpha:.2f}）"
    return lines


def _pack(*, count, trials, per, scale, skew, kurt, variance, hurdle, psr, dsr, note) -> SharpeAssessment:
    return SharpeAssessment(
        n_obs=int(count),
        n_trials=int(trials),
        multiple_testing=int(trials) > 1,
        observed_sharpe=float(per),
        observed_sharpe_annual=float(per * scale),
        skewness=float(skew),
        kurtosis=float(kurt),
        sharpe_variance=None if variance is None else float(variance),
        benchmark_sharpe=float(hurdle),
        benchmark_sharpe_annual=float(hurdle * scale),
        psr=psr,
        dsr=dsr,
        note=note,
    )


def _psr(observed: float, benchmark: float, n_obs: int, var_term: float) -> float | None:
    if n_obs < 2 or not math.isfinite(var_term) or var_term <= 0:
        return None
    z = (observed - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(var_term)
    if not math.isfinite(z):
        return None
    return float(norm.cdf(z))


def _variance_term(sharpe: float, skewness: float, kurtosis: float) -> float:
    return 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2


def _moments_from_returns(returns, risk_free_rate: float, periods: int):
    series = pd.Series(returns, dtype="float64")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    rate = float(risk_free_rate or 0.0)
    excess = series - (rate / periods if rate else 0.0)
    count = int(len(excess))
    if count == 0:
        return excess, 0.0, 0.0, 0.0, 3.0, 0
    mean = float(excess.mean())
    std = float(excess.std(ddof=1)) if count > 1 else 0.0
    skew = float(excess.skew()) if count >= 3 else 0.0
    # pandas kurtosis is Fisher's excess. The paper's gamma_4 is 3 under normality.
    kurt = float(excess.kurt()) + 3.0 if count >= 4 else 3.0
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 3.0
    return excess, mean, std, skew, kurt, count


def _step_correction(items: Sequence, alpha: float, *, method: str) -> CorrectionResult:
    level = _alpha(alpha)
    parsed = [_item(item) for item in items]
    if len(parsed) < 1:
        raise ValueError("the family is empty")
    m = len(parsed)
    order = sorted(range(m), key=lambda i: (parsed[i][2], i))
    adjusted = [0.0] * m
    thresholds = [0.0] * m
    reject = [False] * m
    if method == "holm":
        running = 0.0
        still = True
        for rank, idx in enumerate(order, start=1):
            threshold = level / (m - rank + 1)
            thresholds[idx] = threshold
            raw = min(1.0, (m - rank + 1) * parsed[idx][2])
            running = max(running, raw)
            adjusted[idx] = running
            if still and parsed[idx][2] <= threshold:
                reject[idx] = True
            else:
                still = False
    elif method == "bh":
        raw_adj = [0.0] * m
        for rank, idx in enumerate(order, start=1):
            thresholds[idx] = level * rank / m
            raw_adj[idx] = min(1.0, m / rank * parsed[idx][2])
        running = 1.0
        for idx in reversed(order):
            running = min(running, raw_adj[idx])
            adjusted[idx] = running
        # Largest k with p_(k) <= alpha * k / m. Reject that prefix.
        cutoff = 0
        for rank, idx in enumerate(order, start=1):
            if parsed[idx][2] <= thresholds[idx]:
                cutoff = rank
        for rank, idx in enumerate(order, start=1):
            reject[idx] = rank <= cutoff
    else:
        raise ValueError(f"unknown correction: {method}")

    ranks = {idx: rank for rank, idx in enumerate(order, start=1)}
    rows = tuple(
        CorrectionRow(
            label=parsed[i][0],
            statistic=parsed[i][1],
            p_value=parsed[i][2],
            rank=ranks[i],
            threshold=thresholds[i],
            adjusted_p=adjusted[i],
            reject=reject[i],
        )
        for i in range(m)
    )
    survivors = tuple(row.label for row in sorted(rows, key=lambda row: row.rank) if row.reject)
    return CorrectionResult(
        method=method,
        alpha=level,
        rows=rows,
        n_rejected=len(survivors),
        survivors=survivors,
    )


def _item(item) -> tuple[str, float | None, float]:
    if isinstance(item, Mapping):
        label = str(item["label"])
        stat = item.get("statistic", item.get("sharpe"))
        p_value = item.get("p", item.get("p_value"))
    elif isinstance(item, (tuple, list)) and len(item) == 3:
        label, stat, p_value = item
        label = str(label)
    elif isinstance(item, (tuple, list)) and len(item) == 2:
        label, p_value = item
        label = str(label)
        stat = None
    else:
        raise ValueError("each row must be (label, statistic, p) or a mapping")
    p = _probability(p_value)
    statistic = None if stat is None else float(stat)
    if statistic is not None and not math.isfinite(statistic):
        statistic = None
    return label, statistic, p


def _trial(item) -> tuple[str, float | None, float | None]:
    if isinstance(item, Mapping):
        label = str(item["label"])
        sharpe = item.get("sharpe", item.get("statistic"))
        p_value = item.get("p_value", item.get("p"))
    elif isinstance(item, (tuple, list)) and len(item) >= 3:
        label, sharpe, p_value = item[0], item[1], item[2]
        label = str(label)
    else:
        raise ValueError("each trial must carry a label, a Sharpe, and a p-value")
    stat = None if sharpe is None else float(sharpe)
    if stat is not None and not math.isfinite(stat):
        stat = None
    if p_value is None:
        return label, stat, None
    return label, stat, _probability(p_value)


def _n_label(n_trials: int, n_param_sets: int | None, n_folds_searched: int | None) -> str:
    if (
        n_trials > 1
        and n_param_sets
        and n_folds_searched
        and int(n_param_sets) * int(n_folds_searched) == n_trials
    ):
        return f"{n_trials}（{int(n_param_sets)} 組 × {int(n_folds_searched)} 折）"
    return str(max(n_trials, 1))


def _as_variance(value, annualized: bool, periods: int) -> float | None:
    if value is None:
        return None
    variance = float(value)
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("sharpe_variance must be a finite number >= 0")
    if annualized:
        variance = variance / periods
    return variance


def _periods(value: int) -> int:
    if isinstance(value, bool) or int(value) < 2:
        raise ValueError("periods_per_year must be an integer >= 2")
    return int(value)


def _trials(value: int) -> int:
    if isinstance(value, bool) or int(value) < 1:
        raise ValueError("n_trials must be an integer >= 1")
    return int(value)


def _obs(value: int) -> int:
    if isinstance(value, bool) or int(value) < 0:
        raise ValueError("n_obs must be an integer >= 0")
    return int(value)


def _alpha(value: float) -> float:
    alpha = float(value)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return alpha


def _probability(value) -> float:
    p_value = float(value)
    if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
        raise ValueError("p-values must lie in [0, 1]")
    return p_value
