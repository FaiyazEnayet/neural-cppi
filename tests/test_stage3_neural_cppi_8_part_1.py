"""Part 1 calibration fixtures for R8 (uncertainty-aware budgeting).

These tests pin the source contract before any implementation exists:
forecast a conditional downside loss, calibrate with time-respecting
validation, and report conditional coverage by regime, including whether
calibration collapses under shocks. All fixtures are deterministic
synthetic arrays; no real data, no training run, no performance claim.
"""

import math

import numpy as np
import pytest

from dhl.risk.calibration import (
    CoverageReport,
    budget_from_forecast,
    collapse_check,
    coverage_report,
    time_respecting_splits,
)
from dhl.risk.forecast import (
    adaptive_quantile_forecast,
    empirical_quantile,
    rolling_quantile_forecast,
    trailing_statistic,
)


def test_stage3_neural_cppi_8_part_1_rolling_quantile_hand_oracle():
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Median of [1, 2, 3] is 2; median of [2, 3, 4] is 3. Forecasts are
    # aligned to the period AFTER the window, so index 0 of the output
    # is the forecast for period 3.
    got = rolling_quantile_forecast(losses, window=3, alpha=0.5)
    assert got.tolist() == [2.0, 3.0]
    # 90% quantile of [1, 2, 3] by linear interpolation: 1 + 0.9 * 2 * 1
    # in sorted-order position terms: index 1.8 -> 2 + 0.8 * (3 - 2) = 2.8.
    got90 = rolling_quantile_forecast(losses, window=3, alpha=0.9)
    assert got90.tolist() == [2.8, 3.8]


def test_stage3_neural_cppi_8_part_1_empirical_quantile_oracle():
    values = np.array([3.0, 1.0, 2.0])
    # Sorted [1, 2, 3]; median is 2; 0% is the minimum, 100% the maximum.
    assert empirical_quantile(values, 0.5) == 2.0
    assert empirical_quantile(values, 0.0) == 1.0
    assert empirical_quantile(values, 1.0) == 3.0


def test_stage3_neural_cppi_8_part_1_no_lookahead():
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    base = rolling_quantile_forecast(losses, window=3, alpha=0.5)
    # Perturb only the last observation; every earlier forecast is made
    # from data strictly before its period, so the prefix must not move.
    perturbed = losses.copy()
    perturbed[-1] = 100.0
    got = rolling_quantile_forecast(perturbed, window=3, alpha=0.5)
    assert got[:-1].tolist() == base[:-1].tolist()


def test_stage3_neural_cppi_8_part_1_forecast_fail_closed():
    losses = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rolling_quantile_forecast(losses, window=0, alpha=0.5)
    with pytest.raises(ValueError):
        rolling_quantile_forecast(losses, window=3, alpha=0.5)
    with pytest.raises(ValueError):
        rolling_quantile_forecast(losses, window=2, alpha=0.0)
    with pytest.raises(ValueError):
        rolling_quantile_forecast(losses, window=2, alpha=1.5)
    with pytest.raises(ValueError):
        rolling_quantile_forecast(np.array([1.0, float("nan"), 3.0]), window=2, alpha=0.5)
    with pytest.raises(ValueError):
        rolling_quantile_forecast(np.array([[1.0, 2.0]]), window=1, alpha=0.5)
    with pytest.raises(ValueError):
        empirical_quantile(np.array([]), 0.5)


def test_stage3_neural_cppi_8_part_1_adaptive_update_hand_oracle():
    # Hand-stepped adaptive conformal update with gamma = 0.1 and
    # target coverage alpha = 0.8 (target miscoverage 0.2). The level in
    # force moves by gamma * (miscoverage - (1 - alpha)): a miss adds
    # 0.1 * 0.8 = 0.08, a cover subtracts 0.1 * 0.2 = 0.02.
    # Initial level: 80% quantile of the first window [1.0, 2.0] by
    # linear interpolation: 1.0 + 0.8 * (2.0 - 1.0) = 1.8.
    # Forecast for period 2: 1.8; loss 5.0 misses -> level 1.88.
    # Forecast for period 3: 1.88; loss 2.0 misses -> level 1.96.
    # Forecast for period 4: 1.96.
    losses = np.array([1.0, 2.0, 5.0, 2.0, 1.0])
    got = adaptive_quantile_forecast(losses, window=2, alpha=0.8, gamma=0.1)
    assert got.tolist() == pytest.approx([1.8, 1.88, 1.96])
    with pytest.raises(ValueError):
        adaptive_quantile_forecast(losses, window=2, alpha=0.8, gamma=0.0)
    with pytest.raises(ValueError):
        adaptive_quantile_forecast(losses, window=2, alpha=0.8, gamma=-0.1)


def test_stage3_neural_cppi_8_part_1_coverage_report_oracle():
    forecasts = np.array([2.0, 2.0, 2.0, 2.0])
    realized = np.array([1.0, 3.0, 2.0, 4.0])
    regimes = ["calm", "calm", "stress", "stress"]
    report = coverage_report(forecasts, realized, target_alpha=0.8, regimes=regimes)
    assert isinstance(report, CoverageReport)
    assert report.n == 4
    assert report.covered == 2
    assert report.coverage == pytest.approx(0.5)
    assert report.target_alpha == 0.8
    assert report.by_regime["calm"] == (2, 1, pytest.approx(0.5))
    assert report.by_regime["stress"] == (2, 1, pytest.approx(0.5))
    with pytest.raises(ValueError):
        coverage_report(forecasts[:3], realized, target_alpha=0.8)
    with pytest.raises(ValueError):
        coverage_report(forecasts, realized, target_alpha=0.8, regimes=["calm"])


def test_stage3_neural_cppi_8_part_1_collapse_check():
    report = CoverageReport(
        n=10,
        covered=6,
        coverage=0.6,
        target_alpha=0.8,
        by_regime={"calm": (5, 4, 0.8), "stress": (5, 2, 0.4)},
    )
    assert collapse_check(report, min_regime_n=5, tolerance=0.2) is True
    assert collapse_check(report, min_regime_n=6, tolerance=0.2) is False
    assert collapse_check(report, min_regime_n=5, tolerance=0.5) is False


def test_stage3_neural_cppi_8_part_1_time_respecting_splits():
    # 100 periods: train 0..60, calibration 60..80, test 80..100.
    train, cal, test = time_respecting_splits(100, 60, 20, 20)
    assert (train, cal, test) == ((0, 60), (60, 80), (80, 100))
    with pytest.raises(ValueError):
        time_respecting_splits(100, 60, 20, 21)
    with pytest.raises(ValueError):
        time_respecting_splits(100, 0, 20, 20)
    with pytest.raises(ValueError):
        time_respecting_splits(100, 60, -1, 20)


def test_stage3_neural_cppi_8_part_1_budget_scaling_oracle():
    # Higher forecast downside scales the budget down proportionally,
    # clipped to [floor, cap]; a nonpositive forecast holds the cap.
    assert budget_from_forecast(2.0, 1.0) == pytest.approx(0.5)
    assert budget_from_forecast(0.5, 1.0) == pytest.approx(1.0)
    assert budget_from_forecast(10.0, 1.0, floor=0.2) == pytest.approx(0.2)
    assert budget_from_forecast(0.0, 1.0) == pytest.approx(1.0)
    assert budget_from_forecast(2.0, 1.0, base_budget=0.8) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        budget_from_forecast(2.0, 0.0)
    with pytest.raises(ValueError):
        budget_from_forecast(-1.0, 1.0)
    with pytest.raises(ValueError):
        budget_from_forecast(2.0, 1.0, floor=0.8, cap=0.2)


def test_stage3_neural_cppi_8_part_1_trailing_statistic_no_lookahead():
    losses = np.array([1.0, 2.0, 3.0, 4.0])
    # Trailing 2-window standard deviation (population) at t=2 uses
    # [1, 2] -> 0.5; at t=3 uses [2, 3] -> 0.5.
    got = trailing_statistic(losses, window=2)
    assert got.tolist() == pytest.approx([0.5, 0.5])
    perturbed = losses.copy()
    perturbed[-1] = 100.0
    assert trailing_statistic(perturbed, window=2)[:-1].tolist() == got[:-1].tolist()


def test_stage3_neural_cppi_8_part_1_shock_collapse_report():
    # Acceptance fixture: a calm block (losses 1..2) followed by a shock
    # block (losses 8..10). Rolling forecasts calibrated on calm data
    # cannot cover the shock block; the report must say so explicitly,
    # per regime, instead of hiding it in an overall average.
    calm = np.array([1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0])
    shock = np.array([8.0, 9.0, 10.0, 8.0, 9.0, 10.0, 8.0, 9.0, 10.0, 8.0])
    losses = np.concatenate([calm, shock])
    forecasts = rolling_quantile_forecast(losses, window=5, alpha=0.8)
    realized = losses[5:]
    regimes = ["calm"] * 5 + ["stress"] * 10
    assert len(forecasts) == len(realized) == len(regimes)
    report = coverage_report(forecasts, realized, target_alpha=0.8, regimes=regimes)
    assert set(report.by_regime) == {"calm", "stress"}
    calm_n, calm_cov, stress_n, stress_cov = (
        report.by_regime["calm"][0],
        report.by_regime["calm"][2],
        report.by_regime["stress"][0],
        report.by_regime["stress"][2],
    )
    assert calm_n == 5 and stress_n == 10
    # The shock regime covers only half its periods (0.5 vs the 0.8
    # target): every shock-onset period misses, and only once shock data
    # enters the rolling window do later shock periods get covered. The
    # collapse flag fires on that 0.3 shortfall at tolerance 0.2.
    assert stress_cov == pytest.approx(0.5)
    assert collapse_check(report, min_regime_n=5, tolerance=0.2) is True
    # The calm regime stays near target; the collapse is regime-specific,
    # not a property of the whole series.
    assert calm_cov >= 0.6


def test_stage3_neural_cppi_8_part_1_baseline_and_adaptive_both_reported():
    # Baseline capture: both models produce well-formed forecasts and
    # coverage reports on the same series. No claim is made about which
    # is better; Part 1 captures baselines, it does not promote one.
    rng_losses = np.array(
        [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 4.0, 2.0, 1.5, 3.5, 2.0, 5.0]
    )
    base = rolling_quantile_forecast(rng_losses, window=4, alpha=0.8)
    adapt = adaptive_quantile_forecast(rng_losses, window=4, alpha=0.8, gamma=0.05)
    assert len(base) == len(adapt) == len(rng_losses) - 4
    assert not np.array_equal(base, adapt)
    realized = rng_losses[4:]
    base_report = coverage_report(base, realized, target_alpha=0.8)
    adapt_report = coverage_report(adapt, realized, target_alpha=0.8)
    assert 0.0 <= base_report.coverage <= 1.0
    assert 0.0 <= adapt_report.coverage <= 1.0
    assert math.isfinite(base_report.coverage) and math.isfinite(adapt_report.coverage)
