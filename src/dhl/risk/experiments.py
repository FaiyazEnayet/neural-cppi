"""Model-comparison experiments for downside-loss forecasts (R8 part 2).

Part 2 closes parent R8. Part 1 supplied the primitives (rolling and
adaptive forecasts, coverage reporting, collapse flag, time-respecting
splits, budget map) and captured baselines without promoting a model.
This module adds the remaining pieces the contract names: a Gaussian
comparator lives in ``dhl.risk.forecast``; here, an experiment runner
compares all three models on one series with one regime rule, plus a
seeded synthetic shock series for the shock experiment.

Time discipline, restated because the runner is where it could break:
the regime threshold is fixed on the training block only; every
forecast for period t uses data strictly before t; evaluation periods
are never used to fit anything. Regime labels for an evaluation period
use that period's own trailing statistic, which is causal by
construction, and are descriptive, not fitted.

No result here is a performance claim. Coverage is reported per
regime for every model, collapse flags are computed from those
reports, and a mean budget is reported as absent (None) whenever the
Part 1 budget map refuses a negative forecast, never as zero.
"""

from dataclasses import dataclass, field

import numpy as np

from dhl.risk.calibration import (
    budget_from_forecast,
    collapse_check,
    coverage_report,
    CoverageReport,
)
from dhl.risk.forecast import (
    adaptive_quantile_forecast,
    empirical_quantile,
    gaussian_quantile_forecast,
    rolling_quantile_forecast,
    trailing_statistic,
    _validate_series,
)


def stress_threshold(trailing_train, quantile):
    """Regime threshold fixed on the training block only.

    Returns the empirical ``quantile`` of the training-block trailing
    statistics. Callers must pass trailing statistics for training
    periods only; anything from the evaluation window would leak the
    answer into the regime labels.
    """
    arr = _validate_series(trailing_train, "trailing_train")
    return float(empirical_quantile(arr, quantile))


def label_regimes(trailing, threshold):
    """Label each period 'stress' or 'calm' against a fixed threshold.

    A period is 'stress' when its trailing statistic strictly exceeds
    ``threshold`` and 'calm' otherwise. The threshold is caller-fixed
    (see :func:`stress_threshold`); this function fits nothing.
    """
    arr = _validate_series(trailing, "trailing")
    if not np.isfinite(float(threshold)):
        raise ValueError(f"threshold must be finite, got {threshold}")
    return ["stress" if float(v) > float(threshold) else "calm" for v in arr]


def synthetic_shock_losses(seed=0, n_calm=200, n_shock=100,
                           calm_base=0.5, calm_scale=0.3,
                           shock_base=4.0, shock_scale=1.0):
    """Seeded synthetic calm block followed by a shock block.

    Losses are folded-normal magnitudes (absolute value of a standard
    normal draw, scaled and shifted), so every value is positive by
    construction. This is SYNTHETIC data for the shock experiment
    only: it tests whether calibration collapses when the loss scale
    jumps, and proves nothing about any real market. Same seed gives
    the same series.
    """
    for name, value in (
        ("n_calm", n_calm),
        ("n_shock", n_shock),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    for name, value in (
        ("calm_base", calm_base),
        ("calm_scale", calm_scale),
        ("shock_base", shock_base),
        ("shock_scale", shock_scale),
    ):
        if not float(value) >= 0.0:
            raise ValueError(f"{name} must be nonnegative, got {value}")
    rng = np.random.default_rng(seed)
    calm = np.abs(rng.standard_normal(n_calm)) * float(calm_scale) + float(calm_base)
    shock = np.abs(rng.standard_normal(n_shock)) * float(shock_scale) + float(shock_base)
    return np.concatenate([calm, shock])


@dataclass(frozen=True)
class ModelResult:
    """One model's experiment outcome: coverage report, collapse flag.

    ``mean_budget`` is the mean Part 1 budget multiplier over the
    evaluation periods, or None when the budget map refuses any
    evaluation forecast (a negative downside forecast) or the
    reference loss is nonpositive. Absent is reported as absent,
    never as zero.
    """

    report: CoverageReport
    collapsed: bool
    mean_budget: float | None


@dataclass(frozen=True)
class ModelComparison:
    """Comparison of rolling, adaptive, and Gaussian forecasts.

    All models share the series, window, target coverage, evaluation
    window, and regime labels, so their reports are directly
    comparable. ``eval_regime_counts`` maps each regime label to its
    evaluation-period count.
    """

    models: dict
    threshold: float
    train_end: int
    n_eval: int
    eval_regime_counts: dict = field(default_factory=dict)


def compare_downside_models(losses, window, alpha, gamma, train_end,
                            threshold_quantile=0.9, min_regime_n=10,
                            tolerance=0.2):
    """Compare the three downside models on one series, honestly.

    ``train_end`` splits the series: periods before it fix the regime
    threshold (from their trailing statistics) and the budget
    reference (their empirical ``alpha`` quantile); periods from it
    onward are the evaluation window. Every model is forecast over the
    full series with Part 1's causal functions, then sliced to the
    evaluation window, so no model sees evaluation data before its
    period and no threshold is fitted on evaluation data.

    Returns a :class:`ModelComparison` with per-model coverage
    reports, collapse flags, and mean budgets. No winner is selected
    here: callers report all three models, including null and
    negative outcomes.
    """
    arr = _validate_series(losses, "losses")
    if not isinstance(train_end, (int, np.integer)):
        raise ValueError(f"train_end must be an integer, got {train_end}")
    if not int(window) < int(train_end) < arr.size:
        raise ValueError(
            f"train_end ({train_end}) must lie strictly between window "
            f"({window}) and series length ({arr.size})"
        )
    train_end = int(train_end)
    forecasts = {
        "rolling": rolling_quantile_forecast(arr, window, alpha),
        "adaptive": adaptive_quantile_forecast(arr, window, alpha, gamma),
        "gaussian": gaussian_quantile_forecast(arr, window, alpha),
    }
    trailing = trailing_statistic(arr, window)
    # Trailing index j is period window + j. Training periods are
    # 0..train_end - 1, so their trailing statistics are the first
    # train_end - window entries. Evaluation periods start at train_end.
    threshold = stress_threshold(trailing[: train_end - window], threshold_quantile)
    labels = label_regimes(trailing[train_end - window :], threshold)
    realized = arr[train_end:]
    reference = empirical_quantile(arr[:train_end], alpha)
    eval_counts = {}
    for label in labels:
        eval_counts[label] = eval_counts.get(label, 0) + 1
    models = {}
    for name, forecast in forecasts.items():
        eval_forecast = forecast[train_end - window :]
        report = coverage_report(eval_forecast, realized, alpha, regimes=labels)
        collapsed = collapse_check(report, min_regime_n, tolerance)
        if reference > 0.0 and np.all(eval_forecast >= 0.0):
            mean_budget = float(
                np.mean(
                    [budget_from_forecast(f, reference) for f in eval_forecast]
                )
            )
        else:
            mean_budget = None
        models[name] = ModelResult(
            report=report, collapsed=collapsed, mean_budget=mean_budget
        )
    return ModelComparison(
        models=models,
        threshold=float(threshold),
        train_end=train_end,
        n_eval=int(realized.size),
        eval_regime_counts=eval_counts,
    )
