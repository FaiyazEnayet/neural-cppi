"""Dependence-aware evaluation statistics (Stage 3 P1-C).

The defect this module addresses: a mean over historical windows is
not a mean over independent trials. Thirty-one US windows include
repeated horizons from the same event, so a 252-day window nested
inside a 504-day window from the same event must not count twice,
win fractions must not treat related episodes as independent trials,
and a constraint must never be summarized by a single terminal-only
metric.

Conventions
-----------
A "window" is one evaluated price path, identified by a key (usually
a start date plus horizon, or an explicit id). An "event family" is
the set of windows that belong to one market event: windows sharing
an explicit event label, or windows whose time intervals overlap
(share at least one bar), belong to one family. The effective sample
size is the number of families, not the number of windows.

This module builds on, and does not duplicate, the earlier units: it
consumes the episode registry in ``dhl.eval.registry`` (independence
groups are valid family labels) and the dependent-data machinery in
``dhl.eval.chrono`` (nonoverlapping windows, block bootstrap), and
it reports paired differences per exact matched window, which is
what ``paired_evaluate`` in ``dhl.eval.chrono`` enriches itself with
when window keys and families are supplied.

Every interval this module reports carries its assumption in the
open. Episode-aware bootstrap intervals are exploratory: families
are resampled as units, and families are not independent market
draws, so no interval here carries an independent-sample
significance claim.
"""

import json
import math

import numpy as np

TRADING_DAYS_PER_YEAR = 252

_BOOTSTRAP_ASSUMPTION = (
    "EXPLORATORY: resamples event families as exchangeable units; "
    "families are not independent market draws, so this interval "
    "carries no independent-sample significance claim."
)


def _kind(value):
    if isinstance(value, str):
        return "str"
    if isinstance(value, bool):
        raise ValueError(f"window bound {value!r} is not a date or bar index")
    if isinstance(value, (int, float)):
        return "num"
    raise ValueError(f"window bound {value!r} is not a date or bar index")


def assign_event_families(windows):
    """Assign each window to an event family id (ints, first-seen order).

    ``windows``: list of dicts with ``start`` and ``end`` (both ISO
    date strings ``YYYY-MM-DD`` or both bar indices) and an optional
    ``event`` label. Two windows share a family when they carry the
    same explicit event label, or when their intervals overlap
    (inclusive: sharing even one bar). Overlap is transitive, so a
    chain of overlapping windows forms one family, and a 252-day
    window nested inside a 504-day window shares that window's
    family by construction.

    Fail-closed: a missing bound, an end before its start, or mixed
    bound kinds raises ``ValueError``.
    """
    windows = list(windows)
    kinds = set()
    for w in windows:
        if "start" not in w or "end" not in w:
            raise ValueError(f"window {w!r} is missing start or end")
        start_kind, end_kind = _kind(w["start"]), _kind(w["end"])
        if start_kind != end_kind:
            raise ValueError(
                f"window {w!r} mixes date strings and bar indices")
        kinds.add(start_kind)
    if len(kinds) > 1:
        raise ValueError("window bounds mix date strings and bar indices")
    for w in windows:
        if w["end"] < w["start"]:
            raise ValueError(
                f"window end {w['end']!r} precedes start {w['start']!r}")

    parent = list(range(len(windows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            a, b = windows[i], windows[j]
            same_event = (a.get("event") is not None
                          and a.get("event") == b.get("event"))
            overlap = a["start"] <= b["end"] and b["start"] <= a["end"]
            if same_event or overlap:
                union(i, j)

    ids, out = {}, []
    for i in range(len(windows)):
        root = find(i)
        if root not in ids:
            ids[root] = len(ids)
        out.append(ids[root])
    return out


def effective_n(families):
    """Effective sample size: the number of distinct event families."""
    return len(set(families))


def paired_differences_per_window(values_a, values_b, window_keys):
    """Per-window paired differences for exact matched windows.

    ``values_a``/``values_b`` map window key to a scalar (for example
    terminal wealth under two strategies on the same path).
    ``window_keys`` fixes the order. Fail-closed: a duplicate key,
    or a key missing from either side, raises ``ValueError`` naming
    the offending key. Nothing is silently aligned by position.
    """
    keys = list(window_keys)
    if len(set(keys)) != len(keys):
        dup = next(k for k in keys if keys.count(k) > 1)
        raise ValueError(f"duplicate window key {dup!r}")
    rows = []
    for k in keys:
        if k not in values_a:
            raise ValueError(f"window key {k!r} missing from first series")
        if k not in values_b:
            raise ValueError(f"window key {k!r} missing from second series")
        a, b = float(values_a[k]), float(values_b[k])
        rows.append({"window": k, "a": a, "b": b, "diff": a - b})
    return rows


def aggregate_by_family(diffs, families):
    """Aggregate paired differences with each event family weighted once.

    Returns the per-family means, the family-equal mean, the plain
    window mean for contrast, and the effective sample size. When
    families hold different window counts the two means disagree,
    and the family mean is the honest one: it does not let one event
    vote once per horizon.
    """
    diffs = [float(d) for d in diffs]
    families = list(families)
    if len(diffs) != len(families):
        raise ValueError(
            f"{len(diffs)} differences for {len(families)} families")
    if not diffs:
        raise ValueError("cannot aggregate zero differences")
    per_family = {}
    for d, f in zip(diffs, families):
        per_family.setdefault(f, []).append(d)
    fam_stats = {f: {"n_windows": len(v),
                      "mean_diff": float(np.mean(v))}
                 for f, v in per_family.items()}
    return {
        "families": fam_stats,
        "family_mean": float(np.mean([s["mean_diff"]
                                      for s in fam_stats.values()])),
        "window_mean": float(np.mean(diffs)),
        "effective_n": len(per_family),
        "n_windows": len(diffs),
    }


def horizon_slices(diffs, horizons):
    """Group paired differences by horizon (for example 252 vs 504 days).

    Returns one entry per horizon with its count, mean difference,
    and worst (minimum) difference. Horizons are not pooled: a mean
    over mixed horizons is not horizon-normalized, so each horizon
    keeps its own row.
    """
    diffs = [float(d) for d in diffs]
    horizons = list(horizons)
    if len(diffs) != len(horizons):
        raise ValueError(
            f"{len(diffs)} differences for {len(horizons)} horizons")
    if not diffs:
        raise ValueError("cannot slice zero differences")
    grouped = {}
    for d, h in zip(diffs, horizons):
        grouped.setdefault(h, []).append(d)
    return {h: {"n": len(v),
                "mean_diff": float(np.mean(v)),
                "worst_diff": float(np.min(v)),
                "best_diff": float(np.max(v))}
            for h, v in grouped.items()}


def annualized_log_growth(v_t, v_0=100.0, horizon_days=None):
    """Annualized log growth: log(V_T / V_0) / (horizon_days / 252).

    Horizon normalization is what makes a 252-day path comparable to
    a 504-day path. Fail-closed on non-positive wealth or horizon.
    """
    if horizon_days is None or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive day count")
    if v_t <= 0 or v_0 <= 0:
        raise ValueError("wealth values must be positive for log growth")
    return math.log(v_t / v_0) / (horizon_days / TRADING_DAYS_PER_YEAR)


def summarize_window(path, v_0=100.0, horizon_days=None, fees_total=None,
                     notional_total=None, risky_weights=None, cap=None):
    """One window on every economically interpretable scale at once.

    ``path`` is the wealth series (n_steps + 1 values). The returned
    row always carries terminal wealth, worst intrawindow wealth,
    and max drawdown together: a constraint is never summarized by a
    terminal-only metric. Annualized log growth is computed only
    when ``horizon_days`` is given. Fees, trade notional, and
    risk-budget utilization (mean risky weight over the cap) pass
    through when supplied and are ``None`` otherwise, never an
    invented zero.
    """
    w = np.asarray(path, dtype=float)
    if w.size == 0:
        raise ValueError("cannot summarize an empty wealth path")
    if not np.all(np.isfinite(w)):
        raise ValueError("wealth path contains non-finite values")
    peak = np.maximum.accumulate(w)
    drawdown = float(np.max((peak - w) / np.maximum(peak, 1e-12)))
    utilization = None
    if risky_weights is not None and cap is not None:
        if cap <= 0:
            raise ValueError("cap must be positive for utilization")
        utilization = float(np.mean(np.asarray(risky_weights, dtype=float))
                            / cap)
    return {
        "terminal_V_T": float(w[-1]),
        "annualized_log_growth": (
            annualized_log_growth(float(w[-1]), v_0, horizon_days)
            if horizon_days is not None else None),
        "worst_intrawindow_wealth": float(np.min(w)),
        "max_drawdown": drawdown,
        "total_fees": (float(fees_total) if fees_total is not None
                       else None),
        "trade_notional": (float(notional_total)
                           if notional_total is not None else None),
        "risk_budget_utilization": utilization,
    }


def episode_aware_bootstrap_ci(diffs, families, n_boot=2000, seed=0,
                               alpha=0.05):
    """Percentile interval for the family-equal mean, resampling families.

    Protocol (documented so an independent reimplementation matches
    exactly): family means are taken in first-appearance order, and
    ``numpy.random.default_rng(seed).integers(0, n_families,
    size=(n_boot, n_families))`` draws whole families per resample;
    each resample's mean is the mean of its drawn family means. A
    single family collapses the interval to the point estimate by
    design: one family carries no cross-family information.

    The returned ``assumption`` string is part of the result: this
    interval is exploratory and carries no independent-sample
    significance claim.
    """
    agg = aggregate_by_family(diffs, families)
    fam_means = [s["mean_diff"]
                 for s in agg["families"].values()]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(fam_means), size=(n_boot, len(fam_means)))
    means = np.array([[fam_means[i] for i in row] for row in draws]).mean(
        axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2.0,
                                   100 * (1.0 - alpha / 2.0)])
    return {
        "lo": float(lo),
        "hi": float(hi),
        "point": agg["family_mean"],
        "n_families": agg["effective_n"],
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "assumption": _BOOTSTRAP_ASSUMPTION,
    }


def seed_x_episode_table(values):
    """Full seed x episode variation.

    ``values`` maps ``(seed, episode)`` to a scalar. Returns the
    per-episode mean and population std across seeds, the overall
    mean across all cells, and the per-seed means. No cell is
    dropped: every seed x episode combination supplied is reported.
    """
    if not values:
        raise ValueError("cannot tabulate zero seed x episode values")
    by_episode, by_seed = {}, {}
    for (seed, episode), value in values.items():
        by_episode.setdefault(episode, []).append(float(value))
        by_seed.setdefault(seed, []).append(float(value))
    episodes = {ep: {"mean_over_seeds": float(np.mean(v)),
                      "std_over_seeds": float(np.std(v)),
                      "n_seeds": len(v)}
                for ep, v in by_episode.items()}
    return {
        "episodes": episodes,
        "per_seed_mean": {s: float(np.mean(v))
                          for s, v in by_seed.items()},
        "overall_mean": float(np.mean([float(v)
                                       for v in values.values()])),
    }


class TrialLedger:
    """Policy-selection multiplicity counts, negative variants included.

    Every variant tried is recorded, whether selected or not and
    whether it regressed or not, so a reported winner cannot hide
    how many losers were tried first. ``summary`` always reports
    the counts; there is no view of this ledger without them.
    """

    def __init__(self):
        self._variants = []

    def record(self, name, selected, metrics=None):
        self._variants.append({
            "name": name,
            "selected": bool(selected),
            "regressed": bool((metrics or {}).get("regressed", False)),
            "metrics": metrics or {},
        })

    def summary(self):
        regressed = [v["name"] for v in self._variants if v["regressed"]]
        selected = [v["name"] for v in self._variants if v["selected"]]
        return {
            "n_tried": len(self._variants),
            "n_selected": len(selected),
            "n_regressed": len(regressed),
            "selected_names": selected,
            "regressed_names": regressed,
            "variants": [v["name"] for v in self._variants],
        }


def render_summary(paired, families, horizons, metrics,
                   ledger_summary=None, ci=None):
    """Assemble the full dependence-aware summary for one comparison.

    ``paired`` is the per-window list from
    ``paired_differences_per_window``; ``families`` and ``horizons``
    align with it; ``metrics`` is the aligned list of
    ``summarize_window`` rows. The ``worst_case`` section is
    unconditional: worst paired difference, worst drawdown, and
    worst intrawindow wealth are reported as prominently as the
    means, with the window each came from. Horizon slices are stored
    as a sorted list (not an int-keyed dict) so the JSON form below
    round-trips exactly.
    """
    paired = list(paired)
    families = list(families)
    horizons = list(horizons)
    metrics = list(metrics)
    n = len(paired)
    if not (len(families) == len(horizons) == len(metrics) == n):
        raise ValueError(
            f"misaligned summary inputs: {n} paired rows, "
            f"{len(families)} families, {len(horizons)} horizons, "
            f"{len(metrics)} metric rows")
    if n == 0:
        raise ValueError("cannot render a summary with zero windows")
    diffs = [float(p["diff"]) for p in paired]
    fam_agg = aggregate_by_family(diffs, families)
    slices = horizon_slices(diffs, horizons)
    slice_list = [{"horizon": h, **slices[h]} for h in sorted(slices)]
    worst_diff_i = int(np.argmin(diffs))
    worst_dd_i = int(np.argmax([m["max_drawdown"] for m in metrics]))
    worst_wealth_i = int(np.argmin([m["worst_intrawindow_wealth"]
                                    for m in metrics]))
    worst_case = {
        "min_paired_diff": float(diffs[worst_diff_i]),
        "min_paired_diff_window": paired[worst_diff_i]["window"],
        "mean_paired_diff": float(np.mean(diffs)),
        "worst_drawdown": float(metrics[worst_dd_i]["max_drawdown"]),
        "worst_drawdown_window": paired[worst_dd_i]["window"],
        "worst_intrawindow_wealth": float(
            metrics[worst_wealth_i]["worst_intrawindow_wealth"]),
        "worst_intrawindow_wealth_window": paired[worst_wealth_i]["window"],
    }
    return {
        "n_windows": n,
        "effective_n": fam_agg["effective_n"],
        "per_window": paired,
        "families": families,
        "horizons": horizons,
        "window_metrics": metrics,
        "family_aggregation": fam_agg,
        "horizon_slices": slice_list,
        "worst_case": worst_case,
        "trial_ledger": ledger_summary,
        "ci": ci,
        "assumptions": [
            "Paired differences are per exact matched window; windows "
            "sharing an event family are not independent trials.",
            "Family means weight each event family equally; the window "
            "mean is reported alongside for contrast.",
            "Bootstrap intervals, when present, are exploratory: "
            "families resampled as units, no independent-sample "
            "significance claim.",
            "Constraints are reported on every interpretable scale; "
            "no terminal-only summary is emitted.",
        ],
    }


def summarize_to_json(summary):
    """Canonical JSON form of a rendered summary (sorted keys)."""
    return json.dumps(summary, sort_keys=True, indent=2)


_REQUIRED_SECTIONS = ("per_window", "family_aggregation",
                      "horizon_slices", "worst_case")


def tables_from_json(text):
    """Parse a summary JSON, refusing any table without its worst case.

    A summary that lost its ``worst_case`` section is not a summary
    this module will hand back: worst-case regressions are reported
    as prominently as mean improvements, unconditionally.
    """
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("summary JSON must decode to an object")
    for section in _REQUIRED_SECTIONS:
        if section not in parsed:
            raise ValueError(
                f"summary JSON is missing required section: {section}")
    return parsed


def recompute_from_json(text):
    """Recompute every derived section from the JSON's own rows.

    The family aggregation, horizon slices, and worst case are
    recomputed from the stored per-window rows, families, horizons,
    and metric rows, and must equal the stored sections exactly;
    any mismatch raises. The returned dict is the parsed summary.
    """
    parsed = tables_from_json(text)
    paired = parsed["per_window"]
    diffs = [float(p["diff"]) for p in paired]
    if (aggregate_by_family(diffs, parsed["families"])
            != parsed["family_aggregation"]):
        raise ValueError("family aggregation does not match stored rows")
    slices = horizon_slices(diffs, parsed["horizons"])
    slice_list = [{"horizon": h, **slices[h]} for h in sorted(slices)]
    if slice_list != parsed["horizon_slices"]:
        raise ValueError("horizon slices do not match stored rows")
    metrics = parsed["window_metrics"]
    worst_diff_i = int(np.argmin(diffs))
    worst_dd_i = int(np.argmax([m["max_drawdown"] for m in metrics]))
    worst_wealth_i = int(np.argmin([m["worst_intrawindow_wealth"]
                                    for m in metrics]))
    worst_case = {
        "min_paired_diff": float(diffs[worst_diff_i]),
        "min_paired_diff_window": paired[worst_diff_i]["window"],
        "mean_paired_diff": float(np.mean(diffs)),
        "worst_drawdown": float(metrics[worst_dd_i]["max_drawdown"]),
        "worst_drawdown_window": paired[worst_dd_i]["window"],
        "worst_intrawindow_wealth": float(
            metrics[worst_wealth_i]["worst_intrawindow_wealth"]),
        "worst_intrawindow_wealth_window": paired[worst_wealth_i]["window"],
    }
    if worst_case != parsed["worst_case"]:
        raise ValueError("worst case does not match stored rows")
    return parsed
