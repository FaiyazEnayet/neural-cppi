"""Calibration reporting for downside-loss forecasts.

Part 1 of R8: given forecasts and realized losses, report empirical
coverage overall and by regime, test whether calibration collapses
under shocks, and split a series into time-respecting train /
calibration / test blocks. A separate small function maps a downside
forecast to a risk budget: the forecast sets the budget, it is not
traded as its own signal.

Regime labels are caller-supplied (for example from a trailing
statistic against a threshold fixed on a training split). This module
never fits a regime threshold on the evaluation window itself, because
that would leak the answer into the coverage report.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CoverageReport:
    """Empirical coverage of downside forecasts against realized losses.

    A period is covered when ``realized <= forecast``. ``by_regime``
    maps each regime label to ``(n, covered, coverage)`` for that
    regime only. Coverage is a descriptive statistic, not a guarantee.
    """

    n: int
    covered: int
    coverage: float
    target_alpha: float
    by_regime: dict = field(default_factory=dict)


def _validate_pair(forecasts, realized, name_forecasts="forecasts"):
    f = np.asarray(forecasts, dtype=float)
    r = np.asarray(realized, dtype=float)
    if f.ndim != 1 or r.ndim != 1:
        raise ValueError("forecasts and realized must both be 1-D")
    if f.size != r.size:
        raise ValueError(
            f"forecasts ({f.size}) and realized ({r.size}) must match"
        )
    if f.size == 0:
        raise ValueError("forecasts and realized must not be empty")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(r)):
        raise ValueError("forecasts and realized must contain only finite values")
    return f, r


def coverage_report(forecasts, realized, target_alpha, regimes=None):
    """Empirical coverage overall and, when given, per regime.

    ``target_alpha`` is the coverage the forecasts aimed at, in (0, 1).
    ``regimes`` (optional) labels each period; its length must match
    the forecasts. Regimes with no periods never appear in ``by_regime``.
    """
    f, r = _validate_pair(forecasts, realized)
    if not 0.0 < float(target_alpha) < 1.0:
        raise ValueError(f"target_alpha must be in (0, 1), got {target_alpha}")
    if regimes is not None and len(regimes) != f.size:
        raise ValueError(
            f"regimes ({len(regimes)}) must match forecasts ({f.size})"
        )
    covered_mask = r <= f
    by_regime = {}
    if regimes is not None:
        for label, is_covered in zip(regimes, covered_mask):
            n, covered, _ = by_regime.get(label, (0, 0, 0.0))
            n += 1
            covered += int(is_covered)
            by_regime[label] = (n, covered, covered / n)
    return CoverageReport(
        n=int(f.size),
        covered=int(covered_mask.sum()),
        coverage=float(covered_mask.mean()),
        target_alpha=float(target_alpha),
        by_regime=by_regime,
    )


def collapse_check(report, min_regime_n, tolerance):
    """True when some well-populated regime misses its target badly.

    A regime counts only when it has at least ``min_regime_n`` periods,
    so a tiny regime cannot trigger the flag on noise. Collapse means
    ``abs(coverage - target_alpha) > tolerance`` for such a regime.
    The tolerance is caller-chosen; there is no universal collapse
    threshold, and this function asserts none.
    """
    if min_regime_n < 1:
        raise ValueError(f"min_regime_n must be at least 1, got {min_regime_n}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be nonnegative, got {tolerance}")
    for n, _, coverage in report.by_regime.values():
        if n >= min_regime_n and abs(coverage - report.target_alpha) > tolerance:
            return True
    return False


def time_respecting_splits(n, n_train, n_cal, n_test):
    """Chronological (train, calibration, test) index boundaries.

    Returns ``((0, n_train), (n_train, n_train + n_cal),
    (n_train + n_cal, n_train + n_cal + n_test))`` as half-open index
    ranges: the calibration block starts exactly where training ends,
    and the test block starts exactly where calibration ends, so no
    period is shared and no later data trains an earlier block. Raises
    when any block is nonpositive or the blocks exceed ``n``.
    """
    for name, value in (
        ("n", n),
        ("n_train", n_train),
        ("n_cal", n_cal),
        ("n_test", n_test),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    if n_train + n_cal + n_test > n:
        raise ValueError(
            f"blocks ({n_train} + {n_cal} + {n_test}) exceed series length ({n})"
        )
    train_end = n_train
    cal_end = n_train + n_cal
    test_end = cal_end + n_test
    return (0, train_end), (train_end, cal_end), (cal_end, test_end)


def budget_from_forecast(forecast_loss, reference_loss, base_budget=1.0,
                         floor=0.0, cap=1.0):
    """Map a downside forecast to a risk budget multiplier.

    Higher forecast downside scales the budget down proportionally:
    ``base_budget * reference_loss / forecast_loss``, clipped to
    ``[floor, cap]``. A nonpositive forecast holds the cap (no
    downside, no reason to cut). This is budget-setting only: the
    forecast is not traded as its own signal, and the scaled budget
    carries no performance claim.
    """
    if not float(reference_loss) > 0.0:
        raise ValueError(f"reference_loss must be positive, got {reference_loss}")
    if not float(forecast_loss) >= 0.0:
        raise ValueError(f"forecast_loss must be nonnegative, got {forecast_loss}")
    if not float(base_budget) >= 0.0:
        raise ValueError(f"base_budget must be nonnegative, got {base_budget}")
    if not 0.0 <= float(floor) <= float(cap):
        raise ValueError(f"require 0 <= floor <= cap, got floor={floor}, cap={cap}")
    if float(forecast_loss) <= 0.0:
        return float(cap)
    scaled = float(base_budget) * float(reference_loss) / float(forecast_loss)
    return float(min(max(scaled, float(floor)), float(cap)))
