"""Part 2 experiment fixtures for R8 (uncertainty-aware budgeting).

Part 2 closes parent R8: it adds the lightweight probabilistic
comparator the contract names (a Gaussian trailing quantile forecast),
an experiment runner that compares rolling, adaptive, and Gaussian
forecasts with time-respecting splits and train-fixed regime
thresholds, a seeded synthetic shock series, and a real-data
experiment on the committed in-repo SPY daily bars.

No test here asserts a winning model. Comparisons are reported
honestly, including null and negative outcomes; these tests pin the
wiring (causal alignment, train-only thresholds, exact coverage
recomputation) and record the measured numbers. All synthetic fixtures
are labeled synthetic; the real-data test uses only committed CSVs.
"""

import math
import statistics

import numpy as np
import pytest

from dhl.risk.calibration import (
    budget_from_forecast,
    collapse_check,
    coverage_report,
)
from dhl.risk.experiments import (
    compare_downside_models,
    label_regimes,
    stress_threshold,
    synthetic_shock_losses,
)
from dhl.risk.forecast import (
    empirical_quantile,
    gaussian_quantile_forecast,
)


def test_stage3_neural_cppi_8_part_2_gaussian_hand_oracle():
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Median (alpha 0.5) of a Gaussian is its mean: trailing means.
    got = gaussian_quantile_forecast(losses, window=3, alpha=0.5)
    assert got.tolist() == [2.0, 3.0]
    # Independent oracle via the statistics module, not numpy: mean and
    # population sd of [1, 2, 3] plus the standard normal 0.9 quantile.
    mean = statistics.mean([1.0, 2.0, 3.0])
    sd = statistics.pstdev([1.0, 2.0, 3.0])
    z90 = statistics.NormalDist().inv_cdf(0.9)
    got90 = gaussian_quantile_forecast(losses, window=3, alpha=0.9)
    assert got90[0] == pytest.approx(mean + sd * z90)
    assert got90[1] == pytest.approx(
        statistics.mean([2.0, 3.0, 4.0])
        + statistics.pstdev([2.0, 3.0, 4.0]) * z90
    )


def test_stage3_neural_cppi_8_part_2_gaussian_no_lookahead():
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    base = gaussian_quantile_forecast(losses, window=3, alpha=0.8)
    perturbed = losses.copy()
    perturbed[-1] = 100.0
    got = gaussian_quantile_forecast(perturbed, window=3, alpha=0.8)
    assert got[:-1].tolist() == base[:-1].tolist()


def test_stage3_neural_cppi_8_part_2_gaussian_fail_closed():
    losses = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        gaussian_quantile_forecast(losses, window=0, alpha=0.5)
    with pytest.raises(ValueError):
        gaussian_quantile_forecast(losses, window=3, alpha=0.5)
    with pytest.raises(ValueError):
        gaussian_quantile_forecast(losses, window=2, alpha=0.0)
    with pytest.raises(ValueError):
        gaussian_quantile_forecast(losses, window=2, alpha=1.0)
    with pytest.raises(ValueError):
        gaussian_quantile_forecast(
            np.array([1.0, float("nan"), 3.0]), window=2, alpha=0.5
        )
    with pytest.raises(ValueError):
        gaussian_quantile_forecast(np.array([[1.0, 2.0]]), window=1, alpha=0.5)


def test_stage3_neural_cppi_8_part_2_threshold_and_labels_oracle():
    trailing = np.array([1.0, 2.0, 3.0, 4.0])
    # Median of [1, 2, 3, 4] by linear interpolation is 2.5.
    assert stress_threshold(trailing, 0.5) == pytest.approx(2.5)
    labels = label_regimes(np.array([1.0, 3.0, 2.5, 4.0]), 2.5)
    assert labels == ["calm", "stress", "calm", "stress"]
    with pytest.raises(ValueError):
        stress_threshold(np.array([]), 0.5)
    with pytest.raises(ValueError):
        label_regimes(np.array([1.0, float("nan")]), 2.5)


def test_stage3_neural_cppi_8_part_2_threshold_is_train_only():
    # Two series sharing the training prefix but differing afterwards
    # must get the same regime threshold: the threshold is fixed on the
    # training block, never on the evaluation window.
    train = np.linspace(1.0, 2.0, 100)
    series_a = np.concatenate([train, np.full(50, 5.0)])
    series_b = np.concatenate([train, np.full(50, 0.5)])
    res_a = compare_downside_models(
        series_a, window=20, alpha=0.8, gamma=0.05, train_end=100
    )
    res_b = compare_downside_models(
        series_b, window=20, alpha=0.8, gamma=0.05, train_end=100
    )
    assert res_a.threshold == pytest.approx(res_b.threshold)


def test_stage3_neural_cppi_8_part_2_compare_structure():
    rng = np.random.default_rng(7)
    losses = np.abs(rng.normal(1.0, 0.4, size=200)) + 0.5
    res = compare_downside_models(
        losses, window=20, alpha=0.8, gamma=0.05, train_end=150
    )
    assert set(res.models) == {"rolling", "adaptive", "gaussian"}
    assert res.n_eval == 50
    assert res.train_end == 150
    assert sum(res.eval_regime_counts.values()) == 50
    for result in res.models.values():
        assert result.report.n == 50
        assert 0.0 <= result.report.coverage <= 1.0
        assert isinstance(result.collapsed, bool)
        assert result.mean_budget is None or 0.0 <= result.mean_budget <= 1.0
        assert math.isfinite(result.report.coverage)


def test_stage3_neural_cppi_8_part_2_shock_experiment_reports_by_regime():
    # Acceptance experiment: a seeded synthetic calm block followed by
    # a shock block. The runner must report coverage per regime for
    # every model and a collapse flag consistent with an independent
    # recomputation from the raw forecasts. No winner is asserted.
    losses = synthetic_shock_losses(seed=0, n_calm=200, n_shock=100)
    assert len(losses) == 300
    assert np.all(np.isfinite(losses)) and np.all(losses > 0)
    res = compare_downside_models(
        losses, window=20, alpha=0.8, gamma=0.05, train_end=150
    )
    assert "stress" in res.eval_regime_counts
    assert res.eval_regime_counts["stress"] >= 10
    assert "calm" in res.eval_regime_counts
    from dhl.risk.forecast import (
        adaptive_quantile_forecast,
        gaussian_quantile_forecast as gauss,
        rolling_quantile_forecast,
        trailing_statistic,
    )

    forecasts = {
        "rolling": rolling_quantile_forecast(losses, 20, 0.8),
        "adaptive": adaptive_quantile_forecast(losses, 20, 0.8, 0.05),
        "gaussian": gauss(losses, 20, 0.8),
    }
    trailing = trailing_statistic(losses, 20)
    threshold = stress_threshold(trailing[: 150 - 20], 0.9)
    assert res.threshold == pytest.approx(threshold)
    labels = label_regimes(trailing[150 - 20 :], threshold)
    realized = losses[150:]
    for name, forecast in forecasts.items():
        recomputed = coverage_report(
            forecast[150 - 20 :], realized, 0.8, regimes=labels
        )
        got = res.models[name]
        assert got.report.coverage == pytest.approx(recomputed.coverage)
        assert got.report.by_regime == recomputed.by_regime
        assert got.collapsed == collapse_check(
            recomputed, min_regime_n=10, tolerance=0.2
        )


def test_stage3_neural_cppi_8_part_2_mean_budget_oracle():
    losses = synthetic_shock_losses(seed=1, n_calm=200, n_shock=100)
    res = compare_downside_models(
        losses, window=20, alpha=0.8, gamma=0.05, train_end=150
    )
    reference = empirical_quantile(losses[:150], 0.8)
    from dhl.risk.forecast import rolling_quantile_forecast

    forecast = rolling_quantile_forecast(losses, 20, 0.8)[150 - 20 :]
    expected = float(
        np.mean([budget_from_forecast(f, reference) for f in forecast])
    )
    assert res.models["rolling"].mean_budget == pytest.approx(expected)


def test_stage3_neural_cppi_8_part_2_negative_forecast_budget_absent():
    # A downside forecast can be negative on gain-dominated data; the
    # Part 1 budget map refuses negative forecasts, so the runner
    # reports the mean budget as absent (None), never as zero.
    losses = -np.abs(np.random.default_rng(3).normal(1.0, 0.4, size=200)) - 0.5
    res = compare_downside_models(
        losses, window=20, alpha=0.8, gamma=0.05, train_end=150
    )
    assert res.models["gaussian"].mean_budget is None


def test_stage3_neural_cppi_8_part_2_compare_fail_closed():
    losses = np.linspace(1.0, 2.0, 100)
    with pytest.raises(ValueError):
        compare_downside_models(
            losses, window=20, alpha=0.8, gamma=0.05, train_end=20
        )
    with pytest.raises(ValueError):
        compare_downside_models(
            losses, window=20, alpha=0.8, gamma=0.05, train_end=100
        )
    with pytest.raises(ValueError):
        compare_downside_models(
            losses, window=20, alpha=1.0, gamma=0.05, train_end=50
        )
    with pytest.raises(ValueError):
        compare_downside_models(
            losses, window=20, alpha=0.8, gamma=0.0, train_end=50
        )
    with pytest.raises(ValueError):
        compare_downside_models(
            np.array([]), window=20, alpha=0.8, gamma=0.05, train_end=50
        )


def test_stage3_neural_cppi_8_part_2_synthetic_series_deterministic():
    first = synthetic_shock_losses(seed=42, n_calm=50, n_shock=25)
    second = synthetic_shock_losses(seed=42, n_calm=50, n_shock=25)
    other = synthetic_shock_losses(seed=43, n_calm=50, n_shock=25)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert len(first) == 75


def test_stage3_neural_cppi_8_part_2_real_spy_experiment():
    # Real data, in-repo only: SPY daily adjusted closes, losses as
    # negative log returns. Structure only; the measured numbers are
    # recorded in the return packet, not asserted as a winner.
    from dhl.markets.real import load_spy_daily, log_returns

    data = load_spy_daily()
    losses = -log_returns(data["adjclose"])
    assert len(losses) > 2000
    n = len(losses)
    train_end = n // 2
    res = compare_downside_models(
        losses, window=21, alpha=0.9, gamma=0.05, train_end=train_end
    )
    assert res.n_eval == n - train_end
    assert set(res.models) == {"rolling", "adaptive", "gaussian"}
    for result in res.models.values():
        assert result.report.n == n - train_end
        assert math.isfinite(result.report.coverage)
    res2 = compare_downside_models(
        losses, window=21, alpha=0.9, gamma=0.05, train_end=train_end
    )
    for name in res.models:
        assert res.models[name].report.coverage == pytest.approx(
            res2.models[name].report.coverage
        )
