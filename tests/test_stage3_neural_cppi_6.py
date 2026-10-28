"""Stage 3 NC6: dependent-path uncertainty and frozen ablations.

The defect: contiguous historical windows overlap (stride 1), so a
paired mean/std over them is not an independent-sample statistic, and
bootstrap seeds are not out-of-time markets. This module adds the
honest machinery in ``dhl.eval.chrono``: strictly nonoverlapping
windows, calendar-block bootstrap CIs for paired differences, exact
finite-sample breach bounds, frozen-hyperparameter recording, a
clearly labeled post-selection path, and a sensitivity report.

Oracles used here are independent of the implementation: brute-force
bar-sharing checks, direct binomial sums, closed-form zero-breach
bounds, and a from-scratch bootstrap re-implementation.
"""

import math

import numpy as np
import pytest

from dhl.eval.chrono import (
    block_bootstrap_ci,
    calendar_block_ids,
    clopper_pearson_upper,
    frozen_config_sha256,
    nonoverlapping_windows,
    paired_dependent_summary,
    sensitivity_report,
)


def _prices(n=211, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, size=n - 1)
    return 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))


def _dates(n, start_year=2020):
    import datetime
    d0 = datetime.date(start_year, 1, 1)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n)]


# ---------------- nonoverlapping windows ----------------

def test_stage3_neural_cppi_6_windows_share_no_bar():
    p = _prices(211)
    paths, starts = nonoverlapping_windows(p, 30)
    width = 31
    seen = set()
    for s in starts:
        bars = set(range(int(s), int(s) + width))
        assert not (bars & seen), "windows share a price bar"
        seen |= bars
    assert len(paths) == len(starts) == 6
    # Renormalization is (s0 * window) / window[0], so the first point
    # is 100 up to one floating-point ulp, not bit-exact.
    assert np.allclose(paths[:, 0], 100.0)
    assert list(starts) == [0, 31, 62, 93, 124, 155]


def test_stage3_neural_cppi_6_windows_match_contiguous_values():
    p = _prices(100)
    paths, starts = nonoverlapping_windows(p, 10)
    for row, s in zip(paths, starts):
        assert np.allclose(row, 100.0 * p[s:s + 11] / p[s])
    with pytest.raises(ValueError):
        nonoverlapping_windows(p[:5], 10)


# ---------------- calendar blocks ----------------

def test_stage3_neural_cppi_6_calendar_block_ids():
    dates = ["2020-01-15", "2020-01-31", "2020-02-01", "2020-04-01",
             "2021-01-01"]
    assert calendar_block_ids(dates, "M") == [
        "2020-01", "2020-01", "2020-02", "2020-04", "2021-01"]
    assert calendar_block_ids(dates, "Q") == [
        "2020-Q1", "2020-Q1", "2020-Q1", "2020-Q2", "2021-Q1"]
    assert calendar_block_ids(dates, "Y") == [
        "2020", "2020", "2020", "2020", "2021"]
    with pytest.raises(ValueError):
        calendar_block_ids(dates, "W")
    with pytest.raises(ValueError):
        calendar_block_ids(["2020-13-01"], "M")


# ---------------- block bootstrap ----------------

def test_stage3_neural_cppi_6_bootstrap_is_deterministic():
    vals = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    blocks = ["a", "a", "a", "b", "b", "b"]
    r1 = block_bootstrap_ci(vals, blocks, n_boot=500, seed=7)
    r2 = block_bootstrap_ci(vals, blocks, n_boot=500, seed=7)
    assert r1 == r2
    assert r1["mean"] == pytest.approx(vals.mean())
    assert r1["ci_low"] <= r1["mean"] <= r1["ci_high"]
    assert r1["n_blocks"] == 2 and r1["n"] == 6


def test_stage3_neural_cppi_6_bootstrap_matches_independent_resample():
    vals = np.array([0.5, 1.5, 2.5, 8.0, 9.0, 10.0])
    blocks = ["a", "a", "b", "b", "c", "c"]
    out = block_bootstrap_ci(vals, blocks, n_boot=400, seed=3)
    # Independent re-implementation of the documented protocol:
    # numpy default_rng(seed), block indices drawn with integers.
    groups = [["a", "b", "c"].index(b) for b in blocks]
    by_block = [[vals[i] for i in range(len(vals)) if groups[i] == g]
                for g in range(3)]
    rng = np.random.default_rng(3)
    draws = rng.integers(0, 3, size=(400, 3))
    means = np.array([np.concatenate([by_block[g] for g in row]).mean()
                      for row in draws])
    lo, hi = np.percentile(means, [2.5, 97.5])
    assert out["ci_low"] == pytest.approx(lo)
    assert out["ci_high"] == pytest.approx(hi)


def test_stage3_neural_cppi_6_blocks_are_the_resampling_unit():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    # One block: every resample reproduces the same mean.
    one = block_bootstrap_ci(vals, ["a"] * 4, n_boot=200, seed=0)
    assert one["ci_low"] == one["ci_high"] == pytest.approx(vals.mean())
    # Four blocks: genuine spread.
    many = block_bootstrap_ci(vals, ["a", "b", "c", "d"], n_boot=200,
                              seed=0)
    assert many["ci_high"] > many["ci_low"]
    with pytest.raises(ValueError):
        block_bootstrap_ci(vals, ["a", "b"], n_boot=10, seed=0)
    with pytest.raises(ValueError):
        block_bootstrap_ci([], [], n_boot=10, seed=0)


# ---------------- breach bounds ----------------

def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k + 1))


def test_stage3_neural_cppi_6_zero_breach_bound_closed_form():
    for n in (10, 38, 100):
        bound = clopper_pearson_upper(0, n, alpha=0.05)
        assert bound == pytest.approx(1 - 0.05 ** (1.0 / n), rel=1e-9)


def test_stage3_neural_cppi_6_breach_bound_satisfies_definition():
    # The bound is the p whose Binomial CDF at k equals alpha.
    for k, n in ((2, 20), (5, 50), (1, 30)):
        bound = clopper_pearson_upper(k, n, alpha=0.05)
        assert abs(_binom_cdf(k, n, bound) - 0.05) < 1e-6
    assert clopper_pearson_upper(5, 5) == 1.0
    prev = 0.0
    for k in range(6):
        b = clopper_pearson_upper(k, 20)
        assert b >= prev
        prev = b
    with pytest.raises(ValueError):
        clopper_pearson_upper(6, 5)
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 0)


# ---------------- frozen config ----------------

def test_stage3_neural_cppi_6_frozen_config_hash():
    h1 = frozen_config_sha256({"seed": 0, "floor": 90, "cap": 2.0})
    h2 = frozen_config_sha256({"cap": 2.0, "floor": 90, "seed": 0})
    h3 = frozen_config_sha256({"seed": 1, "floor": 90, "cap": 2.0})
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)


# ---------------- paired dependent summary ----------------

def _summary_inputs():
    rng = np.random.default_rng(11)
    n = 60
    wealth_b = 100.0 + rng.normal(0, 3.0, size=n)
    wealth_a = wealth_b + rng.normal(0.8, 1.0, size=n)
    blocks = (["m%d" % (i // 10) for i in range(n)])
    return wealth_a, wealth_b, blocks


def test_stage3_neural_cppi_6_paired_summary_point_and_ci():
    a, b, blocks = _summary_inputs()
    out = paired_dependent_summary(a, b, blocks, floor=90.0,
                                   n_boot=500, seed=5)
    assert out["n_paths"] == 60 and out["n_blocks"] == 6
    assert out["mean_wealth_a"] == pytest.approx(a.mean())
    assert out["mean_wealth_b"] == pytest.approx(b.mean())
    gap = out["paired_gap"]
    assert gap["mean"] == pytest.approx((a - b).mean())
    assert gap["ci_low"] <= gap["mean"] <= gap["ci_high"]
    assert gap["ci_high"] > gap["ci_low"]
    assert out["hyperparameters_frozen"] is False
    assert out["post_selection"] is False


def test_stage3_neural_cppi_6_paired_summary_zero_breach_bound():
    a = np.full(38, 100.0)
    b = np.full(38, 99.0)
    blocks = ["m%d" % (i // 10) for i in range(38)]
    out = paired_dependent_summary(a, b, blocks, floor=90.0,
                                   n_boot=100, seed=0)
    assert out["p_breach_terminal_a"] == 0.0
    assert out["breach_upper_bound_a"] == pytest.approx(
        1 - 0.05 ** (1.0 / 38), rel=1e-9)
    out_all = paired_dependent_summary(np.full(10, 80.0),
                                       np.full(10, 80.0),
                                       ["x"] * 10, floor=90.0,
                                       n_boot=50, seed=0)
    assert out_all["breach_upper_bound_a"] == 1.0


def test_stage3_neural_cppi_6_seeds_do_not_enter_resampling():
    a, b, blocks = _summary_inputs()
    cfg1 = {"seed": 0, "floor": 90}
    cfg2 = {"seed": 1, "floor": 90}
    o1 = paired_dependent_summary(a, b, blocks, floor=90.0,
                                  frozen_config=cfg1, n_boot=300, seed=2)
    o2 = paired_dependent_summary(a, b, blocks, floor=90.0,
                                  frozen_config=cfg2, n_boot=300, seed=2)
    assert o1["hyperparameters_frozen"] is True
    assert o1["frozen_config_sha256"] != o2["frozen_config_sha256"]
    # Optimizer seeds change only the recorded hash, never the CI:
    # market blocks are the resampling unit.
    assert o1["paired_gap"] == o2["paired_gap"]


def test_stage3_neural_cppi_6_post_selection_label():
    a, b, blocks = _summary_inputs()
    out = paired_dependent_summary(a, b, blocks, floor=90.0,
                                   previously_viewed=True,
                                   n_boot=100, seed=0)
    assert out["post_selection"] is True
    assert "POST-SELECTION" in out["post_selection_label"]
    assert "fresh-holdout" in out["post_selection_label"]
    with pytest.raises(ValueError):
        paired_dependent_summary(a, b[:5], blocks, floor=90.0)


# ---------------- sensitivity ----------------

def _evals_for_sensitivity():
    p = _prices(421)
    dates = _dates(421)
    evals = []
    for horizon in (30, 60):
        paths, starts = nonoverlapping_windows(p, horizon)
        wealth_a = paths[:, -1]
        wealth_b = np.full(len(paths), 100.0)
        for floor in (90.0, 95.0):
            for block_freq in ("M", "Q"):
                b2 = calendar_block_ids(
                    [dates[int(s)] for s in starts], block_freq)
                s = paired_dependent_summary(
                    wealth_a, wealth_b, b2, floor=floor,
                    n_boot=200, seed=0)
                evals.append({
                    "horizon": horizon, "floor": floor,
                    "block_length": block_freq,
                    "mean_paired_gap": s["paired_gap"]["mean"],
                    "ci_low": s["paired_gap"]["ci_low"],
                    "ci_high": s["paired_gap"]["ci_high"],
                    "n_paths": s["n_paths"],
                    "n_blocks": s["n_blocks"],
                    "breach_upper_bound": s["breach_upper_bound_a"],
                })
    return evals


def test_stage3_neural_cppi_6_sensitivity_report():
    rows = sensitivity_report(_evals_for_sensitivity())
    assert len(rows) == 8
    for r in rows:
        assert r["ci_width"] == pytest.approx(r["ci_high"] - r["ci_low"])
        assert r["ci_width"] >= 0.0
    by_h = {}
    for r in rows:
        by_h.setdefault(r["horizon"], set()).add(r["n_paths"])
    assert by_h[60] < by_h[30] or list(by_h[60])[0] < list(by_h[30])[0]
    # A higher floor can only raise the breach count, hence the bound.
    for horizon in (30, 60):
        b90 = [r["breach_upper_bound"] for r in rows
               if r["horizon"] == horizon and r["floor"] == 90.0][0]
        b95 = [r["breach_upper_bound"] for r in rows
               if r["horizon"] == horizon and r["floor"] == 95.0][0]
        assert b95 >= b90
    # Finer blocks mean more blocks.
    n_m = [r["n_blocks"] for r in rows
           if r["block_length"] == "M"][0]
    n_q = [r["n_blocks"] for r in rows
           if r["block_length"] == "Q"][0]
    assert n_m >= n_q
    with pytest.raises(ValueError):
        sensitivity_report([{"horizon": 30}])
