"""Conditional downside-loss forecasting for uncertainty-aware budgeting.

Part 1 of R8: forecast a conditional downside loss level, calibrate it
with time-respecting validation, and report coverage by regime. The
forecasts here set a risk budget; they are not a separate alpha signal,
and nothing in this module promises floor protection, crash frequency,
or forecast correctness.

Time discipline: every forecast for period t is computed from data
strictly before t. A forecast that peeked at its own period would make
its coverage report meaningless, so the alignment is tested, not assumed.

The adaptive update follows the adaptive conformal idea of Gibbs and
Candes (2021): move the level up after a miss and down after a cover,
by an amount proportional to the gap from the target miscoverage. That
paper's long-run coverage statements are about its own setting; they
are NOT a guarantee for any CPPI floor, crash rate, or financial
forecast made with this code.
"""

import numpy as np


def _validate_series(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_window(window, n):
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window}")
    if window >= n:
        raise ValueError(
            f"window ({window}) must be smaller than the series length ({n})"
        )


def empirical_quantile(values, alpha):
    """Linear-interpolation empirical quantile of ``values`` at ``alpha``.

    ``alpha`` in [0, 1]: 0 returns the minimum, 1 the maximum. This is
    the numpy default (linear) rule, stated so an independent oracle
    can reproduce it exactly.
    """
    arr = _validate_series(values, "values")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return float(np.quantile(arr, float(alpha), method="linear"))


def rolling_quantile_forecast(losses, window, alpha):
    """Rolling empirical-quantile downside forecast, strictly causal.

    Returns one forecast per period from ``window`` onward. Output ``i``
    is the forecast for period ``window + i`` and equals the empirical
    ``alpha`` quantile of ``losses[i:i + window]``: only data strictly
    before the forecast period enters it. ``alpha`` is the target
    coverage level in (0, 1), exclusive: a 0 or 1 quantile is a
    degenerate forecast and is refused.
    """
    arr = _validate_series(losses, "losses")
    _validate_window(window, arr.size)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    out = np.empty(arr.size - window, dtype=float)
    for i in range(arr.size - window):
        out[i] = empirical_quantile(arr[i : i + window], alpha)
    return out


def adaptive_quantile_forecast(losses, window, alpha, gamma):
    """Adaptive conformal quantile forecast (Gibbs and Candes, 2021).

    Starts at the empirical ``alpha`` quantile of the first window, then
    after each realized period moves the level by
    ``gamma * (miss - (1 - alpha))`` where ``miss`` is 1 when the
    realized loss exceeded the level in force and 0 otherwise. A miss
    raises the level; a cover lowers it. Output ``i`` is the level in
    force for period ``window + i``. ``gamma`` must be positive.
    """
    arr = _validate_series(losses, "losses")
    _validate_window(window, arr.size)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not float(gamma) > 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    level = empirical_quantile(arr[:window], alpha)
    out = np.empty(arr.size - window, dtype=float)
    for i in range(arr.size - window):
        out[i] = level
        realized = arr[window + i]
        miss = 1.0 if realized > level else 0.0
        level = level + float(gamma) * (miss - (1.0 - float(alpha)))
    return out


def trailing_statistic(losses, window):
    """Trailing population standard deviation, strictly causal.

    Output ``i`` is the standard deviation of ``losses[i:i + window]``
    and is aligned to period ``window + i``, the same alignment as the
    forecasts above. Callers use it to label regimes (for example,
    stress when the trailing statistic exceeds a threshold fixed on a
    training split); the threshold itself is deliberately not fitted
    here, so regime labels stay time-respecting.
    """
    arr = _validate_series(losses, "losses")
    _validate_window(window, arr.size)
    out = np.empty(arr.size - window, dtype=float)
    for i in range(arr.size - window):
        out[i] = float(np.std(arr[i : i + window], ddof=0))
    return out
