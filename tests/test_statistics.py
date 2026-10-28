"""Stage 3 P1-C: dependence-aware evaluation and uncertainty.

The defect: 31 US windows include repeated horizons from the same
event, so a mean over windows is not a mean over independent trials.
A 252-day window nested inside a 504-day window from the same event
must not count twice, win fractions must not treat related episodes
as independent trials, and a terminal-only metric must never stand
alone for a constraint. This module tests the honest machinery in
``dhl.eval.statistics``: event-family clustering, per-window paired
differences with family aggregation, horizon slices, interpretable
scales, episode-aware bootstrap intervals with explicit assumptions,
seed x episode variation, a trial ledger that keeps negative
variants, and JSON summaries that recompute exactly.

Oracles here are independent of the implementation: hand-computed
means, a brute-force overlap check, closed-form log growth, and a
from-scratch family resampling loop.
"""

import json
import math

import numpy as np
import pytest
import torch

from dhl.eval.chrono import paired_evaluate
from dhl.eval.statistics import (
    TrialLedger,
    aggregate_by_family,
    annualized_log_growth,
    assign_event_families,
    effective_n,
    episode_aware_bootstrap_ci,
    horizon_slices,
    paired_differences_per_window,
    recompute_from_json,
    render_summary,
    seed_x_episode_table,
    summarize_to_json,
    summarize_window,
    tables_from_json,
)


# ---------------------------------------------------------------------------
# Event clustering
# ---------------------------------------------------------------------------

def test_stage3_neural_cppi_1_statistics_nested_windows_share_family():
    windows = [
        {"start": 0, "end": 504},
        {"start": 100, "end": 352},
        {"start": 600, "end": 852},
    ]
    fams = assign_event_families(windows)
    assert fams[0] == fams[1]
    assert fams[0] != fams[2]
    assert effective_n(fams) == 2


def test_stage3_neural_cppi_1_statistics_explicit_event_label_merges():
    windows = [
        {"start": "2020-02-19", "end": "2020-05-01", "event": "COVID"},
        {"start": "2020-03-02", "end": "2020-06-01", "event": "COVID"},
        {"start": "2008-09-15", "end": "2009-03-09", "event": "GFC"},
    ]
    fams = assign_event_families(windows)
    assert fams[0] == fams[1]
    assert fams[0] != fams[2]


def test_stage3_neural_cppi_1_statistics_overlap_is_transitive():
    windows = [
        {"start": 0, "end": 100},
        {"start": 50, "end": 150},
        {"start": 120, "end": 200},
    ]
    fams = assign_event_families(windows)
    assert len(set(fams)) == 1


def test_stage3_neural_cppi_1_statistics_malformed_windows_raise():
    with pytest.raises(ValueError):
        assign_event_families([{"start": 10, "end": 5}])
    with pytest.raises(ValueError):
        assign_event_families([{"start": 0}])
    with pytest.raises(ValueError):
        assign_event_families([{"start": "2020-01-01", "end": 5}])


# ---------------------------------------------------------------------------
# Paired differences and family aggregation
# ---------------------------------------------------------------------------

def test_stage3_neural_cppi_1_statistics_paired_diffs_exact_match():
    keys = ["w1", "w2"]
    a = {"w1": 110.0, "w2": 95.0}
    b = {"w1": 100.0, "w2": 100.0}
    rows = paired_differences_per_window(a, b, keys)
    assert [r["window"] for r in rows] == keys
    assert rows[0]["diff"] == pytest.approx(10.0)
    assert rows[1]["diff"] == pytest.approx(-5.0)


def test_stage3_neural_cppi_1_statistics_paired_diffs_fail_closed():
    with pytest.raises(ValueError, match="w2"):
        paired_differences_per_window({"w1": 1.0}, {"w1": 1.0, "w2": 2.0},
                                      ["w1", "w2"])
    with pytest.raises(ValueError, match="duplicate"):
        paired_differences_per_window({"w1": 1.0}, {"w1": 1.0},
                                      ["w1", "w1"])


def test_stage3_neural_cppi_1_statistics_family_mean_not_window_mean():
    diffs = [10.0, 10.0, -5.0]
    fams = ["X", "X", "Y"]
    agg = aggregate_by_family(diffs, fams)
    assert agg["window_mean"] == pytest.approx(5.0)
    assert agg["family_mean"] == pytest.approx(2.5)
    assert agg["effective_n"] == 2
    assert agg["families"]["X"]["mean_diff"] == pytest.approx(10.0)
    assert agg["families"]["Y"]["mean_diff"] == pytest.approx(-5.0)


def test_stage3_neural_cppi_1_statistics_horizon_slices():
    diffs = [4.0, 6.0, -2.0]
    horizons = [252, 252, 504]
    out = horizon_slices(diffs, horizons)
    assert out[252]["mean_diff"] == pytest.approx(5.0)
    assert out[252]["n"] == 2
    assert out[504]["mean_diff"] == pytest.approx(-2.0)
    assert out[504]["worst_diff"] == pytest.approx(-2.0)
    assert out[252]["worst_diff"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Interpretable scales
# ---------------------------------------------------------------------------

def test_stage3_neural_cppi_1_statistics_annualized_log_growth():
    assert annualized_log_growth(121.0, 100.0, 252) == pytest.approx(
        math.log(1.21))
    assert annualized_log_growth(121.0, 100.0, 504) == pytest.approx(
        math.log(1.21) / 2.0)
    with pytest.raises(ValueError):
        annualized_log_growth(0.0, 100.0, 252)
    with pytest.raises(ValueError):
        annualized_log_growth(110.0, 100.0, 0)


def test_stage3_neural_cppi_1_statistics_window_metrics_all_scales():
    path = [100.0, 110.0, 95.0, 105.0]
    m = summarize_window(path, v_0=100.0, horizon_days=252,
                         fees_total=1.5, notional_total=40.0,
                         risky_weights=[0.5, 0.6, 0.4], cap=1.0)
    assert m["terminal_V_T"] == pytest.approx(105.0)
    assert m["worst_intrawindow_wealth"] == pytest.approx(95.0)
    assert m["max_drawdown"] == pytest.approx((110.0 - 95.0) / 110.0)
    assert m["annualized_log_growth"] == pytest.approx(math.log(1.05))
    assert m["total_fees"] == pytest.approx(1.5)
    assert m["trade_notional"] == pytest.approx(40.0)
    assert m["risk_budget_utilization"] == pytest.approx(0.5)
    m2 = summarize_window(path)
    assert m2["total_fees"] is None
    assert m2["trade_notional"] is None
    assert m2["risk_budget_utilization"] is None
    assert m2["annualized_log_growth"] is None
    with pytest.raises(ValueError):
        summarize_window([])


# ---------------------------------------------------------------------------
# Episode-aware bootstrap, seed x episode, trial ledger
# ---------------------------------------------------------------------------

def test_stage3_neural_cppi_1_statistics_family_bootstrap_oracle():
    diffs = [10.0, 10.0, -5.0]
    fams = ["X", "X", "Y"]
    ci = episode_aware_bootstrap_ci(diffs, fams, n_boot=500, seed=7)
    assert ci["point"] == pytest.approx(2.5)
    assert ci["n_families"] == 2
    assert "exploratory" in ci["assumption"].lower()
    assert "not independent" in ci["assumption"].lower()
    fam_means = [10.0, -5.0]
    rng = np.random.default_rng(7)
    draws = rng.integers(0, 2, size=(500, 2))
    means = np.array([np.mean([fam_means[i] for i in row]) for row in draws])
    lo, hi = np.percentile(means, [2.5, 97.5])
    assert ci["lo"] == pytest.approx(float(lo))
    assert ci["hi"] == pytest.approx(float(hi))
    ci2 = episode_aware_bootstrap_ci(diffs, fams, n_boot=500, seed=7)
    assert ci2 == ci


def test_stage3_neural_cppi_1_statistics_single_family_ci_collapses():
    ci = episode_aware_bootstrap_ci([3.0, 4.0], ["A", "A"],
                                    n_boot=100, seed=0)
    assert ci["lo"] == ci["hi"] == ci["point"] == pytest.approx(3.5)


def test_stage3_neural_cppi_1_statistics_seed_x_episode_table():
    values = {(0, "COVID"): 10.0, (1, "COVID"): 14.0,
              (0, "GFC"): 2.0, (1, "GFC"): 6.0}
    tab = seed_x_episode_table(values)
    assert tab["episodes"]["COVID"]["mean_over_seeds"] == pytest.approx(12.0)
    assert tab["episodes"]["COVID"]["std_over_seeds"] == pytest.approx(2.0)
    assert tab["episodes"]["GFC"]["mean_over_seeds"] == pytest.approx(4.0)
    assert tab["overall_mean"] == pytest.approx(8.0)
    with pytest.raises(ValueError):
        seed_x_episode_table({})


def test_stage3_neural_cppi_1_statistics_trial_ledger_keeps_negatives():
    ledger = TrialLedger()
    ledger.record("m4_plain", selected=False, metrics={"regressed": False})
    ledger.record("learned_v2", selected=True, metrics={"regressed": False})
    ledger.record("learned_v3", selected=False, metrics={"regressed": True})
    s = ledger.summary()
    assert s["n_tried"] == 3
    assert s["n_selected"] == 1
    assert s["n_regressed"] == 1
    assert "learned_v3" in s["regressed_names"]


# ---------------------------------------------------------------------------
# JSON summary: exact recompute, worst case always present
# ---------------------------------------------------------------------------

def _example_summary():
    paired = [
        {"window": "w1", "a": 110.0, "b": 100.0, "diff": 10.0},
        {"window": "w2", "a": 105.0, "b": 100.0, "diff": 5.0},
        {"window": "w3", "a": 90.0, "b": 100.0, "diff": -10.0},
    ]
    fams = ["X", "X", "Y"]
    horizons = [252, 252, 504]
    metrics = [
        summarize_window([100.0, 110.0, 108.0], horizon_days=252,
                         fees_total=0.5, notional_total=10.0),
        summarize_window([100.0, 105.0, 103.0], horizon_days=252,
                         fees_total=0.5, notional_total=10.0),
        summarize_window([100.0, 92.0, 90.0], horizon_days=504,
                         fees_total=0.5, notional_total=10.0),
    ]
    ledger = TrialLedger()
    ledger.record("learned_v2", selected=True)
    return render_summary(paired, fams, horizons, metrics,
                          ledger_summary=ledger.summary())


def test_stage3_neural_cppi_1_statistics_json_roundtrip_exact():
    summary = _example_summary()
    assert "worst_case" in summary
    assert summary["worst_case"]["min_paired_diff"] == pytest.approx(-10.0)
    assert summary["worst_case"]["min_paired_diff_window"] == "w3"
    assert summary["assumptions"]
    text = summarize_to_json(summary)
    back = tables_from_json(text)
    assert back == summary
    assert recompute_from_json(text) == summary


def test_stage3_neural_cppi_1_statistics_json_without_worst_case_refused():
    summary = _example_summary()
    del summary["worst_case"]
    text = json.dumps(summary, sort_keys=True)
    with pytest.raises(ValueError, match="worst_case"):
        tables_from_json(text)
    with pytest.raises(ValueError, match="worst_case"):
        recompute_from_json(text)


# ---------------------------------------------------------------------------
# paired_evaluate enrichment (default behavior unchanged)
# ---------------------------------------------------------------------------

def test_stage3_neural_cppi_1_statistics_paired_evaluate_enrichment():
    ref_vt = [100.0, 100.0, 100.0]
    alt_vt = [110.0, 105.0, 90.0]
    ref_path = [[100.0, 100.0, 100.0]] * 3
    alt_path = [[100.0, 110.0, 110.0],
                [100.0, 105.0, 105.0],
                [100.0, 90.0, 90.0]]

    def rollout_fn(S, weight_fn, return_path=True):
        vt, path = {"ref": (ref_vt, ref_path),
                    "alt": (alt_vt, alt_path)}[weight_fn]
        return (torch.tensor(vt, dtype=torch.float32),
                torch.tensor(path, dtype=torch.float32))

    strategies = {"ref": "ref", "alt": "alt"}
    keys = ["w1", "w2", "w3"]
    fams = ["X", "X", "Y"]
    s_np = [[100.0, 100.0, 100.0]] * 3
    out = paired_evaluate(strategies, s_np, rollout_fn,
                          floor=90.0, reference="ref",
                          window_keys=keys, event_families=fams)
    alt = out["alt"]
    diffs = alt["paired_differences_per_window"]
    assert [d["window"] for d in diffs] == keys
    assert [d["diff"] for d in diffs] == pytest.approx([10.0, 5.0, -10.0])
    agg = alt["family_aggregation"]
    assert agg["family_mean"] == pytest.approx(-1.25)
    assert agg["effective_n"] == 2
    assert "paired_differences_per_window" not in out["ref"]
    plain = paired_evaluate(strategies, s_np, rollout_fn,
                            floor=90.0, reference="ref")
    assert "paired_differences_per_window" not in plain["alt"]
    assert "family_aggregation" not in plain["alt"]
    with pytest.raises(ValueError):
        paired_evaluate(strategies, s_np, rollout_fn,
                        floor=90.0, reference="ref",
                        window_keys=["w1"], event_families=fams)
