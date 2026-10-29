"""E3: are simple controls as good as the network? (Stage 3, Nov 7).

The new pieces beyond the P1-B comparator ladder are: a
threshold/hysteresis utilization rule (new weight function), an
optimized volatility multiplier selected on tuning paths only, the
original fixed-floor convention pinned fail-closed, separate frontiers
per cap (a 1.5-capped rung is never pooled with a 1.0-capped rung),
and the falsifier procedure: do neural gains vanish when classic CPPI
gets the same emergency brake or a simple utilization cap?

Honesty labels used throughout: the real-data judge here is the
previously-viewed in-repo SPY benchmark, used as a regression only.
No in-repo episode family is genuinely unseen (every in-repo window
was reported by an earlier unit), so no freshness is claimed anywhere.
The falsifier test with a stub policy is mechanical only: it pins the
comparison arithmetic, not trained-network skill. No trained
checkpoint exists in the repo, so no trained-policy falsifier verdict
is asserted; the procedure is what this unit verifies, plus the
measured guard contribution on classic itself.
"""

import math

import numpy as np
import pytest
import torch

from dhl.eval.comparators import (
    ComparatorConfig,
    Conventions,
    run_ladder,
    validate_selection,
)
from dhl.experiments.cppi_learned import CAP, FLOOR, rollout


def _toy_prices(n_paths=4, steps=10, seed=5):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=(n_paths, steps))
    S = np.empty((n_paths, steps + 1))
    S[:, 0] = 100.0
    S[:, 1:] = 100.0 * np.exp(np.cumsum(rets, axis=1))
    return torch.as_tensor(S, dtype=torch.float32)


def _crash_prices(n_paths=4, steps=20, seed=5):
    """Calm then a sharp volatile drop: the brake and the cap have
    work to do, and realized vol rises past the brake threshold."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.005, size=(n_paths, steps))
    rets[:, 10:] = rng.normal(-0.01, 0.06, size=(n_paths, steps - 10))
    S = np.empty((n_paths, steps + 1))
    S[:, 0] = 100.0
    S[:, 1:] = 100.0 * np.exp(np.cumsum(rets, axis=1))
    return torch.as_tensor(S, dtype=torch.float32)


def _cfg(name, kind, **params):
    return ComparatorConfig(name=name, kind=kind, params=params)


# ---------------------------------------------------------------- hysteresis


def test_stage3_neural_cppi_3_hysteresis_steps():
    """Utilization drops on the high threshold and returns only below
    the low threshold; between the two it holds its previous state."""
    from dhl.benchmarks.cppi_rules import threshold_hysteresis_weight_fn

    fn = threshold_hysteresis_weight_fn(
        multiplier=4.0, floor=90.0, cap=1.5, signal="vol",
        high_threshold=0.35, low_threshold=0.20,
        high_fraction=1.0, low_fraction=0.25)
    V = torch.full((1,), 100.0)
    vols = [0.10, 0.40, 0.30, 0.15, 0.10]
    got = []
    for i, v in enumerate(vols):
        feats = {"vol": torch.full((1,), v)}
        got.append(float(fn(i, V, 0.5, 0.1, feats)[0]))
    # Classic weight at V=100, floor=90, m=4 is 0.4; low state is 0.1.
    assert got == pytest.approx([0.4, 0.1, 0.1, 0.4, 0.4], abs=1e-6)


def test_stage3_neural_cppi_3_hysteresis_per_path_state():
    """Two paths keep independent hysteresis state; a fresh rollout
    (i == 0) resets to the high fraction."""
    from dhl.benchmarks.cppi_rules import threshold_hysteresis_weight_fn

    fn = threshold_hysteresis_weight_fn(
        multiplier=4.0, floor=90.0, cap=1.5, signal="vol",
        high_threshold=0.35, low_threshold=0.20,
        high_fraction=1.0, low_fraction=0.25)
    V = torch.full((2,), 100.0)
    w0 = fn(0, V, 0.5, 0.1, {"vol": torch.tensor([0.10, 0.40])})
    assert w0.tolist() == pytest.approx([0.4, 0.1], abs=1e-6)
    w1 = fn(1, V, 0.5, 0.1, {"vol": torch.tensor([0.30, 0.30])})
    assert w1.tolist() == pytest.approx([0.4, 0.1], abs=1e-6)
    w2 = fn(0, V, 0.5, 0.1, {"vol": torch.tensor([0.10, 0.10])})
    assert w2.tolist() == pytest.approx([0.4, 0.4], abs=1e-6)


def test_stage3_neural_cppi_3_hysteresis_validation():
    from dhl.benchmarks.cppi_rules import threshold_hysteresis_weight_fn

    with pytest.raises(ValueError, match="low_threshold"):
        threshold_hysteresis_weight_fn(
            high_threshold=0.20, low_threshold=0.35)
    with pytest.raises(ValueError, match="fraction"):
        threshold_hysteresis_weight_fn(low_fraction=1.5)
    with pytest.raises(ValueError, match="fraction"):
        threshold_hysteresis_weight_fn(high_fraction=-0.1)
    fn = threshold_hysteresis_weight_fn(signal="panic_z")
    with pytest.raises(ValueError, match="signal"):
        fn(0, torch.full((1,), 100.0), 0.5, 0.1, {"vol": torch.tensor([0.5])})


def test_stage3_neural_cppi_3_hysteresis_ladder_kind():
    """The factory builds the hysteresis rung like any other rung."""
    from dhl.eval.comparators import make_weight_fn

    cfg = ComparatorConfig(
        name="threshold_hysteresis", kind="threshold_hysteresis",
        params={"multiplier": 4.0, "high_threshold": 0.35,
                "low_threshold": 0.20, "high_fraction": 1.0,
                "low_fraction": 0.25})
    fn = make_weight_fn(cfg, Conventions())
    w = fn(0, torch.full((1,), 100.0), 0.5, 0.1,
           {"vol": torch.tensor([0.40])})
    assert float(w[0]) == pytest.approx(0.1, abs=1e-6)


# ------------------------------------------------------- vol-target tuning


def test_stage3_neural_cppi_3_select_vol_target_oracle():
    """Optimized vol target is chosen on tuning paths only, with an
    independent recomputation as oracle and tuning-only provenance."""
    from dhl.benchmarks.classic_controls import select_vol_target
    from dhl.benchmarks.cppi_rules import dynamic_vol_weight_fn

    S = _toy_prices(n_paths=8, steps=20, seed=3)
    conv = Conventions()
    candidates = [0.10, 0.15, 0.20, 0.25, 0.30]
    best, prov = select_vol_target(candidates, S, conv)
    assert prov["selection_kind"] == "tuned"
    assert prov["candidates"] == candidates
    assert prov["n_tuning_paths"] == 8
    assert best in candidates
    # Independent oracle: same criterion, recomputed rung by rung.
    oracle = None
    for target in candidates:
        fn = dynamic_vol_weight_fn(multiplier=4.0, floor=conv.floor,
                                   cap=conv.cap, vol_target=target)
        with torch.no_grad():
            V_T = rollout(S, fn, floor=conv.floor, r=conv.r, dt=conv.dt,
                          cost_rate=conv.cost_rate, cap=conv.cap)
        key = (int((V_T < conv.floor).sum()), -float(V_T.mean()))
        if oracle is None or key < oracle[0]:
            oracle = (key, target)
    assert best == oracle[1]
    assert prov["best_breach_count"] == oracle[0][0]
    assert prov["best_mean_wealth"] == pytest.approx(-oracle[0][1], abs=1e-6)


def test_stage3_neural_cppi_3_select_vol_target_validation():
    from dhl.benchmarks.classic_controls import select_vol_target

    S = _toy_prices(n_paths=4, steps=10, seed=3)
    with pytest.raises(ValueError, match="candidates"):
        select_vol_target([], S, Conventions())
    with pytest.raises(ValueError, match="vol_target"):
        select_vol_target([0.0, 0.2], S, Conventions())
    with pytest.raises(ValueError, match="vol_target"):
        select_vol_target([-0.1], S, Conventions())


# ------------------------------------------------------------ original floor


def test_stage3_neural_cppi_3_original_floor_pinned():
    """Every E3 control runs under the original fixed floor (90, not
    grown at the cash rate); anything else is refused fail-closed."""
    from dhl.benchmarks.classic_controls import (
        ORIGINAL_FLOOR,
        validate_original_floor,
    )

    assert ORIGINAL_FLOOR == FLOOR == 90.0
    validate_original_floor(90.0)
    with pytest.raises(ValueError, match="original floor"):
        validate_original_floor(90.0 * math.exp(0.02 * 63 / 252))
    with pytest.raises(ValueError, match="original floor"):
        validate_original_floor(85.0)


def test_stage3_neural_cppi_3_runner_refuses_other_floors():
    from dhl.benchmarks.classic_controls import run_e3_controls

    S = _toy_prices(n_paths=4, steps=10, seed=3)
    with pytest.raises(ValueError, match="original floor"):
        run_e3_controls(S, S, Conventions(floor=85.0))


# ------------------------------------------------------------ cap frontiers


def test_stage3_neural_cppi_3_frontiers_refuse_mixed_caps():
    from dhl.benchmarks.classic_controls import frontiers_by_cap

    mixed = {
        "a": {"mean_wealth": 100.0, "std_wealth": 1.0,
              "mean_shortfall": 0.0, "applied_cap": 1.5,
              "n_paths": 4, "n_steps": 10},
        "b": {"mean_wealth": 101.0, "std_wealth": 2.0,
              "mean_shortfall": 0.0, "applied_cap": 1.0,
              "n_paths": 4, "n_steps": 10},
    }
    with pytest.raises(ValueError, match="mixes caps"):
        frontiers_by_cap(mixed)
    single = {"a": mixed["a"]}
    pts = frontiers_by_cap(single)
    assert pts == [{"name": "a", "risk": 1.0, "return": 100.0,
                    "shortfall": 0.0}]


# ---------------------------------------------------------------- falsifier


def test_stage3_neural_cppi_3_falsifier_mechanical():
    """Mechanical only: with a constant-weight stub policy, the
    falsifier gaps equal an independent recomputation from the table
    rows. No trained-network skill is claimed by this test."""
    from dhl.benchmarks.classic_controls import falsifier_report

    class StubPolicy(torch.nn.Module):
        def forward(self, t, cushion, feats):
            return torch.full_like(cushion, 1.4)

    S = _crash_prices(n_paths=8, steps=20, seed=5)
    conv = Conventions()
    configs = [
        _cfg("classic_m4_brake", "classic_m4_brake", multiplier=4.0),
        _cfg("budget_fraction_50", "budget_fraction", multiplier=4.0,
             fraction=0.5),
        _cfg("learned_constrained", "learned_constrained", multiplier=4.0),
        _cfg("learned_constrained_brake", "learned_constrained_brake",
             multiplier=4.0),
    ]
    table = run_ladder(S, configs, conv, policy=StubPolicy())
    rep = falsifier_report(table)
    brake = table["classic_m4_brake"]
    lbrake = table["learned_constrained_brake"]
    assert rep["brake_gap_mean_wealth"] == pytest.approx(
        lbrake["mean_wealth"] - brake["mean_wealth"], abs=1e-9)
    assert rep["brake_gap_breaches"] == (
        lbrake["breach_count_terminal"] - brake["breach_count_terminal"])
    assert rep["brake_vanishes"] == (
        brake["mean_wealth"] >= lbrake["mean_wealth"]
        and brake["breach_count_terminal"]
        <= lbrake["breach_count_terminal"])
    capped = table["budget_fraction_50"]
    lcap = table["learned_constrained"]
    assert rep["cap_gap_mean_wealth"] == pytest.approx(
        lcap["mean_wealth"] - capped["mean_wealth"], abs=1e-9)
    assert rep["cap_vanishes"] == (
        capped["mean_wealth"] >= lcap["mean_wealth"]
        and capped["breach_count_terminal"]
        <= lcap["breach_count_terminal"])
    assert rep["verdict"] in ("vanishes", "persists")


def test_stage3_neural_cppi_3_falsifier_missing_and_mismatch():
    from dhl.benchmarks.classic_controls import falsifier_report

    classic_only = {
        "classic_m4_brake": {"mean_wealth": 100.0, "std_wealth": 1.0,
                             "breach_count_terminal": 0, "applied_cap": 1.5,
                             "n_paths": 4, "n_steps": 10},
    }
    rep = falsifier_report(classic_only)
    assert rep["verdict"] == "not_computable"
    assert "learned" in rep["reason"]
    mismatched = {
        "classic_m4_brake": {"mean_wealth": 100.0, "std_wealth": 1.0,
                             "breach_count_terminal": 0, "applied_cap": 1.5,
                             "n_paths": 4, "n_steps": 10},
        "learned_constrained_brake": {
            "mean_wealth": 101.0, "std_wealth": 1.0,
            "breach_count_terminal": 0, "applied_cap": 1.5,
            "n_paths": 8, "n_steps": 10},
    }
    with pytest.raises(ValueError, match="identical paths"):
        falsifier_report(mismatched)


def test_stage3_neural_cppi_3_brake_never_raises_classic_weight():
    """Guard contribution on identical inputs: at every step where
    realized vol exceeds the brake threshold, the braked classic
    weight is at most the unbraked classic weight on the same wealth
    and features, and the two agree exactly where the brake is off.
    (The comparison is on identical inputs, not divergent rollouts:
    once the brake changes wealth, later cushions differ legitimately.)
    """
    from dhl.benchmarks.cppi_rules import (
        SHIPPED_EMERGENCY_VOL,
        brake_weight_fn,
        classic_weight_fn,
    )
    from dhl.experiments.cppi_learned import _crash_features

    S = _crash_prices(n_paths=8, steps=20, seed=5)
    classic_fn = classic_weight_fn(multiplier=4.0)
    brake_fn = brake_weight_fn(multiplier=4.0)
    feats_full = _crash_features(S)
    vols = feats_full["vol"]
    assert bool((vols > SHIPPED_EMERGENCY_VOL).any()), \
        "fixture must trigger the brake"
    V = torch.full((8,), 100.0)
    for i in range(20):
        feats = {"vol": vols[:, i]}
        cw = classic_fn(i, V, 0.5, 0.1, feats)
        bw = brake_fn(i, V, 0.5, 0.1, feats)
        triggered = vols[:, i] > SHIPPED_EMERGENCY_VOL
        assert bool((bw[triggered] <= cw[triggered] + 1e-6).all())
        assert bool((bw[~triggered] == cw[~triggered]).all())


# ------------------------------------------------------- real-data regression


def test_stage3_neural_cppi_3_real_spy_regression_judge():
    """Previously-viewed real SPY benchmark as regression only.
    m is tuned on the 2021-2022 validation slice and evaluated on the
    disjoint 2023+ slice; both slices are previously viewed, so no
    freshness is claimed. Structure and determinism are asserted;
    measured numbers are recorded in the return packet, not asserted
    as a winner."""
    from dhl.benchmarks.classic_controls import run_e3_controls
    from dhl.markets.real import block_bootstrap_paths, load_spy_daily, log_returns

    data = load_spy_daily()
    dates = data["dates"]
    rets = log_returns(data["adjclose"])
    ret_dates = dates[1:]
    assert len(rets) > 2000
    tuning = np.array([("2021-01-01" <= d < "2023-01-01") for d in ret_dates])
    eval_mask = np.array([(d >= "2023-01-01") for d in ret_dates])
    assert tuning.sum() > 200 and eval_mask.sum() > 200
    tuning_paths = torch.as_tensor(block_bootstrap_paths(
        rets[tuning], n_paths=100, path_len=63, block_len=21, seed=7),
        dtype=torch.float32)
    eval_paths = torch.as_tensor(block_bootstrap_paths(
        rets[eval_mask], n_paths=100, path_len=63, block_len=21, seed=11),
        dtype=torch.float32)
    res = run_e3_controls(tuning_paths, eval_paths, Conventions())
    assert res["judge"]["previously_viewed"] is True
    assert res["judge"]["tuning_slice"] == "2021-01-01..2023-01-01"
    assert res["judge"]["eval_slice"] == "2023-01-01.."
    assert res["m_opt"] in [2.0, 3.0, 4.0, 5.0, 6.0]
    assert res["vol_target_opt"] in [0.10, 0.15, 0.20, 0.25, 0.30]
    expected = {"classic_m4", "classic_m_opt", "classic_m4_brake",
                "budget_fraction_50", "budget_fraction_75",
                "threshold_hysteresis", "dynamic_vol", "dynamic_vol_opt",
                "fixed_cap_50"}
    for cap in (1.0, 1.5):
        table = res["tables"][cap]
        assert set(table) == expected
        for row in table.values():
            assert row["applied_cap"] == pytest.approx(cap)
            assert row["n_paths"] == 100 and row["n_steps"] == 63
        assert {p["name"] for p in res["frontiers"][cap]} == expected
    assert res["falsifier"]["verdict"] == "not_computable"
    assert "learned" in res["falsifier"]["reason"]
    res2 = run_e3_controls(tuning_paths, eval_paths, Conventions())
    for cap in (1.0, 1.5):
        for name in expected:
            assert res["tables"][cap][name]["mean_wealth"] == pytest.approx(
                res2["tables"][cap][name]["mean_wealth"], abs=1e-9)
    validate_selection(
        ComparatorConfig(name="classic_m_opt", kind="classic_m_opt",
                         params={"multiplier": res["m_opt"]},
                         selection_kind="tuned",
                         tuning_split="validation"),
        eval_split="test")
