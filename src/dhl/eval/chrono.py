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

import hashlib
import json
import math

import numpy as np
import torch

from dhl.eval.statistics import aggregate_by_family


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
                    reference=None, window_keys=None, event_families=None):
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

    P1-C enrichment: when ``window_keys`` and ``event_families`` are
    both supplied (aligned with the paths, one entry per path), each
    non-reference strategy additionally gains
    ``paired_differences_per_window`` (the per-window gaps in key
    order, via ``dhl.eval.statistics``) and ``family_aggregation``
    (the family-equal mean, so one event cannot vote once per
    horizon). Supplying only one of the two, or a length mismatch,
    raises. With neither supplied the output is exactly as before.
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

    if (window_keys is None) != (event_families is None):
        raise ValueError(
            "window_keys and event_families must be given together")
    if window_keys is not None:
        n_paths = len(next(iter(V_T_all.values())))
        if len(window_keys) != n_paths or len(event_families) != n_paths:
            raise ValueError(
                f"window keys/families must align with {n_paths} paths")
        if len(set(window_keys)) != len(window_keys):
            raise ValueError("duplicate window key in window_keys")

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
            if window_keys is not None:
                rows = [{"window": k,
                         "a": float(V_T[i]),
                         "b": float(ref_V[i]),
                         "diff": float(gap[i])}
                        for i, k in enumerate(window_keys)]
                m["paired_differences_per_window"] = rows
                m["family_aggregation"] = aggregate_by_family(
                    [float(g) for g in gap], list(event_families))
        results[name] = m
    return results


# ---------------------------------------------------------------------------
# NC6: dependent-path uncertainty.
#
# The paired mean/std in ``paired_evaluate`` treats every window as an
# independent draw. It is not: sliding windows share bars, and a
# bootstrap over training paths is not a sample of out-of-time
# markets. The helpers below report uncertainty over what was actually
# observed: strictly nonoverlapping real windows, or whole calendar
# blocks resampled as units. This block bootstrap is an EVALUATION
# device for confidence intervals; it is not the training-time block
# bootstrap in ``dhl.markets.real`` that generates synthetic paths.
# ---------------------------------------------------------------------------

def nonoverlapping_windows(prices, path_len, s0=100.0):
    """Real historical price paths with no shared price bar.

    Delegates to :func:`contiguous_paths` with stride ``path_len + 1``,
    so consecutive windows are separated by one bar: window ``k``
    covers bars ``[s, s + path_len]`` and the next starts at
    ``s + path_len + 1``. The inter-window return is dropped by
    design; that is the cost of independence. This matches the
    disjoint-window count in ``dhl.eval.registry.summarize_windows``.
    Returns ``(paths, starts)`` as in :func:`contiguous_paths`.
    """
    if path_len < 1:
        raise ValueError(f"path_len must be >= 1, got {path_len}")
    return contiguous_paths(prices, path_len, stride=path_len + 1, s0=s0)


def calendar_block_ids(dates, freq="M"):
    """Calendar block label per date string (``YYYY-MM-DD``).

    ``freq`` is ``"M"`` (``YYYY-MM``), ``"Q"`` (``YYYY-Qn``) or ``"Y"``
    (``YYYY``). Labels are aligned with the input order. Raises on an
    unknown frequency or a malformed date instead of inventing a
    block.
    """
    if freq not in ("M", "Q", "Y"):
        raise ValueError(f"freq must be 'M', 'Q' or 'Y', got {freq!r}")
    out = []
    for d in dates:
        if (not isinstance(d, str) or len(d) < 10
                or d[4] != "-" or d[7] != "-"):
            raise ValueError(f"malformed date {d!r}, expected YYYY-MM-DD")
        year, month = d[0:4], d[5:7]
        try:
            m = int(month)
        except ValueError:
            raise ValueError(f"malformed month in date {d!r}") from None
        if not 1 <= m <= 12:
            raise ValueError(f"month out of range in date {d!r}")
        if freq == "M":
            out.append(f"{year}-{month}")
        elif freq == "Q":
            out.append(f"{year}-Q{(m - 1) // 3 + 1}")
        else:
            out.append(year)
    return out


def block_bootstrap_ci(values, block_ids, n_boot=2000, seed=0,
                       alpha=0.05):
    """Percentile CI for the mean, resampling whole blocks as units.

    ``values`` is a 1-D array of per-path statistics (e.g. paired
    terminal-wealth gaps) and ``block_ids`` labels the market block
    each path belongs to (see :func:`calendar_block_ids`). Each
    replicate draws ``n_blocks`` blocks with replacement using
    ``numpy.random.default_rng(seed).integers`` and averages the
    pooled values; the CI is the ``alpha/2`` and ``1 - alpha/2``
    percentiles of the replicate means. With a single block the CI
    collapses to the point estimate: one block carries no information
    about cross-block variation, and the function says so by
    returning a zero-width interval rather than a spurious spread.
    """
    vals = np.asarray(values, dtype=float)
    blocks = list(block_ids)
    if vals.ndim != 1 or len(vals) == 0:
        raise ValueError("values must be a nonempty 1-D array")
    if len(blocks) != len(vals):
        raise ValueError(
            f"block_ids length {len(blocks)} != values length {len(vals)}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    uniq, inverse = np.unique(blocks, return_inverse=True)
    groups = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boot_means = np.empty(n_boot)
    for i, row in enumerate(draws):
        pooled = np.concatenate([vals[groups[g]] for g in row])
        boot_means[i] = pooled.mean()
    pct = [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    lo, hi = np.percentile(boot_means, pct)
    return {
        "mean": float(vals.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(len(vals)),
        "n_blocks": int(len(groups)),
        "n_boot": int(n_boot),
        "seed": seed,
        "alpha": float(alpha),
        "method": "block-bootstrap percentile",
    }


def _binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p), log-sum-exp stable."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(p) + (n - i) * math.log(1 - p)
        for i in range(k + 1)
    ]
    m = max(logs)
    return sum(math.exp(x - m) for x in logs) * math.exp(m)


def clopper_pearson_upper(k, n, alpha=0.05):
    """Exact one-sided upper bound on a breach probability.

    Given ``k`` breaches in ``n`` paths, returns the smallest ``p``
    with P(Binomial(n, p) <= k) <= ``alpha``: the exact
    Clopper-Pearson upper limit at confidence ``1 - alpha``. For
    zero observed breaches this has the closed form
    ``1 - alpha ** (1 / n)`` (the exact finite-sample statement
    behind the rule of three). No normal approximation is used, so
    the bound stays valid for small ``n`` and for ``k = 0``.
    """
    if n < 1 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n with n >= 1, got k={k}, n={n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def frozen_config_sha256(config):
    """SHA-256 of a hyperparameter config in canonical JSON form.

    Key order does not affect the hash, so the same frozen config
    always records the same fingerprint. Non-JSON-serializable
    values raise instead of being coerced into a meaningless string.
    """
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


POST_SELECTION_LABEL = (
    "POST-SELECTION: these test episodes were inspected during "
    "selection, so this interval is not a fresh-holdout CI and must "
    "not be reported as unseen performance."
)


def paired_dependent_summary(wealth_a, wealth_b, block_ids, floor,
                             frozen_config=None, n_boot=2000, seed=0,
                             alpha=0.05, previously_viewed=False):
    """Point estimates plus dependent-data CIs for a paired gap.

    ``wealth_a`` / ``wealth_b`` are per-path terminal wealths on the
    SAME paths (the paired comparison: same markets, so the gap is
    the policy, not the draw). ``block_ids`` labels the market block
    of each path; the block bootstrap resamples market blocks only.
    Optimization seeds never enter the resampling: they belong in
    ``frozen_config``, whose SHA-256 is recorded so the evaluation
    is tied to hyperparameters frozen before it ran. When
    ``frozen_config`` is None the output says so explicitly instead
    of implying a frozen final evaluation. When ``previously_viewed``
    is True the output carries the post-selection label: the CI is
    reported, but it is not a fresh-holdout CI.

    Breach bounds are exact Clopper-Pearson upper limits on the
    terminal breach rate (wealth < floor) at confidence
    ``1 - alpha``; with zero observed breaches the bound is the
    closed form ``1 - alpha ** (1 / n)``, not zero.
    """
    a = np.asarray(wealth_a, dtype=float)
    b = np.asarray(wealth_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("wealth_a and wealth_b must be equal-length 1-D")
    if len(list(block_ids)) != len(a):
        raise ValueError("block_ids must align with the wealth arrays")
    gap = a - b
    gap_ci = block_bootstrap_ci(gap, block_ids, n_boot=n_boot, seed=seed,
                                alpha=alpha)
    a_ci = block_bootstrap_ci(a, block_ids, n_boot=n_boot, seed=seed,
                              alpha=alpha)
    b_ci = block_bootstrap_ci(b, block_ids, n_boot=n_boot, seed=seed,
                              alpha=alpha)
    k_a = int((a < floor).sum())
    k_b = int((b < floor).sum())
    out = {
        "n_paths": int(len(a)),
        "n_blocks": gap_ci["n_blocks"],
        "mean_wealth_a": float(a.mean()),
        "mean_wealth_b": float(b.mean()),
        "paired_gap": {
            "mean": float(gap.mean()),
            "ci_low": gap_ci["ci_low"],
            "ci_high": gap_ci["ci_high"],
        },
        "wealth_a_ci": {
            "mean": a_ci["mean"], "ci_low": a_ci["ci_low"],
            "ci_high": a_ci["ci_high"],
        },
        "wealth_b_ci": {
            "mean": b_ci["mean"], "ci_low": b_ci["ci_low"],
            "ci_high": b_ci["ci_high"],
        },
        "p_breach_terminal_a": float((a < floor).mean()),
        "p_breach_terminal_b": float((b < floor).mean()),
        "breach_upper_bound_a": clopper_pearson_upper(k_a, len(a), alpha),
        "breach_upper_bound_b": clopper_pearson_upper(k_b, len(b), alpha),
        "alpha": float(alpha),
        "hyperparameters_frozen": frozen_config is not None,
        "post_selection": bool(previously_viewed),
    }
    if frozen_config is not None:
        out["frozen_config_sha256"] = frozen_config_sha256(frozen_config)
    else:
        out["frozen_config_note"] = (
            "no frozen config supplied; this is not a frozen final "
            "evaluation")
    if previously_viewed:
        out["post_selection_label"] = POST_SELECTION_LABEL
    return out


_SENSITIVITY_KEYS = ("horizon", "floor", "block_length", "mean_paired_gap",
                     "ci_low", "ci_high", "n_paths", "n_blocks")


def sensitivity_report(evaluations):
    """Tabulate paired evaluations across horizon/floor/block length.

    Each input dict must carry ``horizon``, ``floor``,
    ``block_length``, ``mean_paired_gap``, ``ci_low``, ``ci_high``,
    ``n_paths`` and ``n_blocks`` (as produced by
    :func:`paired_dependent_summary`); ``breach_upper_bound`` is
    carried through when present. The output adds ``ci_width`` per
    row and preserves the inputs, so a reader can see how the point
    estimate, the interval width, the path count and the breach bound
    move when the horizon, the floor, or the block length changes.
    Missing keys raise instead of producing a silently partial row.
    """
    rows = []
    for ev in evaluations:
        missing = [k for k in _SENSITIVITY_KEYS if k not in ev]
        if missing:
            raise ValueError(
                f"sensitivity row missing keys {missing}: {ev!r}")
        row = {k: ev[k] for k in _SENSITIVITY_KEYS}
        row["ci_width"] = float(ev["ci_high"]) - float(ev["ci_low"])
        if "breach_upper_bound" in ev:
            row["breach_upper_bound"] = ev["breach_upper_bound"]
        rows.append(row)
    return rows
