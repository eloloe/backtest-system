"""Deflated Sharpe and Holm on synthetic numbers. No network."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from metrics import calculate_metrics
from multiple_testing import (
    assess_sharpe,
    benjamini_hochberg,
    expected_maximum_sharpe,
    holm_decisions,
    sharpe_pvalue,
)


def _paper_dsr(sr, n_obs, skew, kurtosis, n_trials):
    """Bailey and López de Prado, written out again so the test is not a copy of the helper."""
    var_term = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    variance = var_term / (n_obs - 1)
    gamma = 0.5772156649015328606
    z_one = float(norm.ppf(1.0 - 1.0 / n_trials))
    z_two = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    hurdle = math.sqrt(variance) * ((1.0 - gamma) * z_one + gamma * z_two)
    z = (sr - hurdle) * math.sqrt(n_obs - 1) / math.sqrt(var_term)
    return float(norm.cdf(z)), hurdle, variance


def test_dsr_matches_the_bailey_formula():
    sr, n_obs, skew, kurtosis, n_trials = 0.05, 500, -0.4, 4.5, 20
    expected, hurdle, variance = _paper_dsr(sr, n_obs, skew, kurtosis, n_trials)
    got = assess_sharpe(
        n_trials=n_trials,
        sharpe=sr,
        n_obs=n_obs,
        skewness=skew,
        kurtosis=kurtosis,
        annualized=False,
    )
    assert got.multiple_testing is True
    assert got.dsr == pytest.approx(expected, rel=1e-10)
    assert got.benchmark_sharpe == pytest.approx(hurdle, rel=1e-10)
    assert got.sharpe_variance == pytest.approx(variance, rel=1e-10)
    assert got.observed_sharpe_annual == pytest.approx(sr * math.sqrt(252))


def test_dsr_falls_as_n_grows_and_as_the_variance_grows():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0008, 0.01, 756))
    one = assess_sharpe(returns, n_trials=1)
    ten = assess_sharpe(returns, n_trials=10)
    hundred = assess_sharpe(returns, n_trials=100)
    assert one.multiple_testing is False
    assert one.dsr is None
    assert one.psr is not None and one.psr > ten.dsr > hundred.dsr
    assert hundred.benchmark_sharpe > ten.benchmark_sharpe > 0

    wider = assess_sharpe(returns, n_trials=10, sharpe_variance=ten.sharpe_variance * 4)
    assert wider.dsr < ten.dsr
    assert wider.psr == pytest.approx(ten.psr)


def test_fat_tails_and_negative_skew_make_the_same_sharpe_less_convincing():
    returns = pd.Series(np.linspace(-0.01, 0.02, 400))
    base = assess_sharpe(
        n_trials=1,
        sharpe=0.08,
        n_obs=len(returns),
        skewness=0.0,
        kurtosis=3.0,
    )
    fat = assess_sharpe(
        n_trials=1,
        sharpe=0.08,
        n_obs=len(returns),
        skewness=0.0,
        kurtosis=8.0,
    )
    skewed = assess_sharpe(
        n_trials=1,
        sharpe=0.08,
        n_obs=len(returns),
        skewness=-1.5,
        kurtosis=3.0,
    )
    assert fat.psr < base.psr
    assert skewed.psr < base.psr


def test_moments_from_returns_match_the_reported_sharpe():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0005, 0.012, 500))
    equity = (1.0 + returns).cumprod() * 100_000
    path = equity.pct_change().dropna()
    _display, raw = calculate_metrics(equity, path, risk_free_rate=0.04)
    assessed = assess_sharpe(path, n_trials=1, risk_free_rate=0.04)
    assert assessed.observed_sharpe_annual == pytest.approx(raw["sharpe"], rel=1e-10)
    plain = assess_sharpe(path, n_trials=1, risk_free_rate=0.0)
    assert plain.kurtosis == pytest.approx(float(path.kurt()) + 3.0, rel=1e-8)
    assert plain.skewness == pytest.approx(float(path.skew()), rel=1e-8)
    assert sharpe_pvalue(path, risk_free_rate=0.04) == pytest.approx(1.0 - assessed.psr)


def test_n_equals_one_uses_a_zero_hurdle():
    assert expected_maximum_sharpe(1, 0.2) == 0.0
    single = assess_sharpe(pd.Series(np.full(30, 0.001)), n_trials=1, risk_free_rate=0.0)
    assert single.benchmark_sharpe == 0.0
    assert "N=1" in single.note
    assert single.dsr is None


def test_holm_hand_worked_example():
    """p = 0.005, 0.01, 0.03, 0.04 at alpha 0.05.

    Thresholds are 0.0125, 0.0167, 0.025, 0.05. The third p-value misses
    0.025, so the step-down stops even though 0.04 is still below 0.05.
    """
    rows = [
        ("a", 1.1, 0.01),
        ("b", 0.4, 0.04),
        ("c", 0.7, 0.03),
        ("d", 2.0, 0.005),
    ]
    result = holm_decisions(rows, alpha=0.05)
    assert result.survivors == ("d", "a")
    assert result.n_rejected == 2
    by_label = {row.label: row for row in result.rows}
    assert by_label["d"].rank == 1
    assert by_label["d"].threshold == pytest.approx(0.05 / 4)
    assert by_label["a"].threshold == pytest.approx(0.05 / 3)
    assert by_label["c"].reject is False
    assert by_label["b"].reject is False
    assert by_label["d"].adjusted_p == pytest.approx(0.02)
    assert by_label["a"].adjusted_p == pytest.approx(0.03)
    assert by_label["c"].adjusted_p == pytest.approx(0.06)
    assert by_label["b"].adjusted_p == pytest.approx(0.06)
    # Input order is preserved on `rows`; survivors follow the p-value order.
    assert [row.label for row in result.rows] == ["a", "b", "c", "d"]


def test_holm_accepts_mappings_and_rejects_a_p_value_outside_the_unit_interval():
    result = holm_decisions(
        [
            {"label": "keep", "sharpe": 1.5, "p": 0.01},
            {"label": "drop", "statistic": 0.2, "p_value": 0.80},
        ],
        alpha=0.05,
    )
    assert result.survivors == ("keep",)
    with pytest.raises(ValueError):
        holm_decisions([("bad", 1.0, 1.2)])


def test_benjamini_hochberg_is_available_beside_holm():
    rows = [("a", 1.0, 0.01), ("b", 1.0, 0.03), ("c", 1.0, 0.04), ("d", 1.0, 0.20)]
    bh = benjamini_hochberg(rows, alpha=0.05)
    # Sorted 0.01, 0.03, 0.04, 0.20 against 0.0125, 0.025, 0.0375, 0.05.
    # Only the first clears its step-up threshold.
    assert bh.survivors == ("a",)
    by_label = {row.label: row for row in bh.rows}
    assert by_label["a"].adjusted_p == pytest.approx(0.04)
    # From the right, min(4/2*0.03, 4/3*0.04) = min(0.06, 0.0533).
    assert by_label["b"].adjusted_p == pytest.approx(4.0 / 3.0 * 0.04)
    holm = holm_decisions(rows, alpha=0.05)
    assert holm.survivors == ("a",)
