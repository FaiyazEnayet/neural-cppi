"""Chronological evaluation: date-based splits, contiguous historical
test paths, and paired strategy comparison.

Conventions
-----------
``dates`` is a list of date strings (``YYYY-MM-DD``) aligned with a
price series: ``prices[i]`` is the close on ``dates[i]``. Log returns
``rets[i]`` therefore cover the interval ``dates[i] -> dates[i+1]``;
the "date" of a return is its end date, ``dates[i+1]``.

A split at ``train_end`` puts every return with end date strictly
before ``train_end`` in the training set. Nothing from the future
leaks in: no shuffling, no shared blocks across the boundary.
"""

import numpy as np
import torch


def split_returns_by_date(dates, returns, train_end, val_end,
                          val_start=None):
    """Split a return series into train/val/test by calendar date.

    ``dates``: date strings aligned with prices (len = len(returns)+1).
    Every return whose end date is < ``train_end`` goes to train;
    in ``[val_start, val_end)`` goes to validation; >= ``val_end``
    goes to test. ``val_start`` defaults to ``train_end`` (the old
    behavior); pass a later ``val_start`` when the span between the
    training end and the validation start belongs to no bucket, e.g.
    a reserved stress year that has its own named test set. Returns
    in that gap are dropped from all three buckets by design, so the
    bucket counts then sum to less than the input length.

    Returns ``{"train": ..., "val": ..., "test": ...}`` plus the
    matching end-date lists for auditing.
    """
    dates = list(dates)
    rets = np.asarray(returns, dtype=float)
    if len(dates) != len(rets) + 1:
        raise ValueError(
            f"dates must align with prices: got {len(dates)} dates for "
            f"{len(rets)} returns"
        )
    if val_start is not None and not train_end <= val_start <= val_end:
        raise ValueError(
            f"val_start {val_start} must lie between train_end "
            f"{train_end} and val_end {val_end}")
    end_dates = dates[1:]
    out, out_dates = {}, {}
    bounds = (("train", None, train_end),
              ("val", val_start or train_end, val_end),
              ("test", val_end, None))
    for name, lo, hi in bounds:
        mask = np.array([
            (lo is None or d >= lo) and (hi is None or d < hi)
            for d in end_dates
        ])
        out[name] = rets[mask]
        out_dates[name] = [d for d, m in zip(end_dates, mask) if m]
    if any(len(v) == 0 for v in out.values()):
        empty = [k for k, v in out.items() if len(v) == 0]
        raise ValueError(f"split left {empty} empty; check the cutoffs")
    return out, out_dates


def contiguous_paths(prices, path_len, stride=1, s0=100.0):
    """Real historical price paths: sliding windows, no resampling.

    Returns ``(paths, start_dates_idx)`` where ``paths`` has shape
    ``(n_windows, path_len + 1)`` and each row is the actual price
    history renormalized to start at ``s0``. Windows start every
    ``stride`` days. Pass ``dates`` separately if you need labels;
    ``start_dates_idx[i]`` is the index into the price/date arrays
    where window ``i`` begins.
    """
    p = np.asarray(prices, dtype=float)
    n = len(p)
    if n < path_len + 1:
        raise ValueError(
            f"need at least {path_len + 1} prices, got {n}")
    starts = np.arange(0, n - path_len, stride)
    paths = np.empty((len(starts), path_len + 1))
    for k, s in enumerate(starts):
        window = p[s:s + path_len + 1]
        paths[k] = s0 * window / window[0]
    return paths, starts


def window_paths_in_range(dates, prices, start_date, end_date, path_len,
                          stride=1, s0=100.0):
    """Contiguous paths drawn only from ``[start_date, end_date)``.

    Used for stress periods (e.g. the COVID crash): every window lies
    fully inside the date range. Returns ``(paths, window_start_dates)``.
    """
    dates = list(dates)
    p = np.asarray(prices, dtype=float)
    if len(dates) != len(p):
        raise ValueError("dates and prices must align")
    idx = [i for i, d in enumerate(dates)
           if start_date <= d < end_date]
    if len(idx) < path_len + 1:
        raise ValueError(
            f"range {start_date}..{end_date} holds {len(idx)} bars, "
            f"need {path_len + 1}")
    sub_p = p[idx]
    paths, starts = contiguous_paths(sub_p, path_len, stride=stride, s0=s0)
    return paths, [dates[idx[s]] for s in starts]


def max_drawdown(wealth_paths):
    """Max peak-to-trough drawdown per path, as a fraction of the peak.

    ``wealth_paths``: array (n_paths, n_steps+1). Returns (n_paths,).
    """
    w = np.asarray(wealth_paths, dtype=float)
    peak = np.maximum.accumulate(w, axis=1)
    dd = (peak - w) / np.maximum(peak, 1e-12)
    return dd.max(axis=1)


def paired_evaluate(strategies, S_np, rollout_fn, floor,
                    reference=None):
    """Evaluate every strategy on the IDENTICAL price paths ``S_np``.

    ``strategies``: dict mapping name -> weight_fn, as accepted by
    ``rollout_fn`` (the single-asset ``rollout`` in
    ``dhl.experiments.cppi_learned``). ``rollout_fn`` must accept
    ``(S, weight_fn, return_path=True)`` and return ``(V_T, path)``.

    Returns ``{name: metrics}`` where each metrics dict holds
    mean/std terminal wealth, terminal and anytime breach rates
    against ``floor``, mean max drawdown, and -- when ``reference``
    names one of the strategies -- the mean and std of the per-path
    terminal-wealth difference versus that reference (the paired
    comparison: same paths, so the gap is the policy, not the draw).
    """
    S = torch.as_tensor(np.asarray(S_np), dtype=torch.float32)
    names = list(strategies)
    V_T_all, paths_all = {}, {}
    with torch.no_grad():
        for name in names:
            V_T, path = rollout_fn(S, strategies[name], return_path=True)
            V_T_all[name] = V_T.numpy()
            paths_all[name] = path.numpy()

    if reference is not None and reference not in strategies:
        raise ValueError(f"reference {reference!r} not in strategies")
    ref_V = V_T_all[reference] if reference else None

    results = {}
    for name in names:
        V_T = V_T_all[name]
        W = paths_all[name]
        dd = max_drawdown(W)
        breached_any = (W.min(axis=1) < floor)
        m = {
            "n_paths": int(len(V_T)),
            "mean_wealth": float(V_T.mean()),
            "std_wealth": float(V_T.std()),
            "p_breach_terminal": float((V_T < floor).mean()),
            "p_breach_anytime": float(breached_any.mean()),
            "mean_max_drawdown": float(dd.mean()),
            "worst_drawdown": float(dd.max()),
        }
        if ref_V is not None and name != reference:
            gap = V_T - ref_V
            m["paired_gap_vs_" + reference] = {
                "mean": float(gap.mean()),
                "std": float(gap.std()),
                "frac_paths_better": float((gap > 0).mean()),
            }
        results[name] = m
    return results
