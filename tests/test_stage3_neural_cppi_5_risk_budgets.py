"""NC5 fixtures: adaptive CPPI risk budgets.

These tests pin the NC5 contract before the study runs: the
volatility-scaled multiplier and the predeclared gap-loss budget have
known answers; the budget layer provably bounds the one-step loss
from a gap of the predeclared size; the frontier metrics reproduce
hand-computed values on fixed wealth paths; all four compared
policies share the repo leverage cap and cost rate; the rule-based
response curves are monotone in the direction their definitions
require; and the jump-sweep path generator is deterministic with the
gap it claims to inject. No training happens in these fixtures.
"""

import math

import numpy as np
import pytest
import torch

from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    FLOOR,
    V0,
    CPPIPolicy,
    cppi_vol_budget_weight_fn,
    gap_budget_multiplier,
    seeded_policy,
    vol_scaled_multiplier,
)
from dhl.experiments.risk_budgets import (
    CONFIG,
    DISCLAIMER,
    GAP_DROP,
    GAP_LOSS_BUDGET,
    frontier_metrics,
    headline_strategies,
    jump_paths,
    response_curve,
)


# --- volatility-scaled multiplier ---------------------------------------

def test_stage3_neural_cppi_5_vol_scaled_multiplier_known_answers():
    # At the reference vol the multiplier is the base multiplier.
    assert vol_scaled_multiplier(0.20) == pytest.approx(4.0)
    # Double the reference vol halves the multiplier.
    assert vol_scaled_multiplier(0.40) == pytest.approx(2.0)
    # Half the reference vol doubles it, but the predeclared max binds.
    assert vol_scaled_multiplier(0.10) == pytest.approx(6.0)
    # Extreme vol floors at the predeclared minimum.
    assert vol_scaled_multiplier(5.0) == pytest.approx(1.0)
    # Zero observed vol (no history yet) takes the max, never inf/nan.
    assert math.isfinite(vol_scaled_multiplier(0.0))
    assert vol_scaled_multiplier(0.0) == pytest.approx(6.0)
    # Tensor path matches the float path elementwise.
    out = vol_scaled_multiplier(torch.tensor([0.20, 0.40, 0.10]))
    assert torch.allclose(out, torch.tensor([4.0, 2.0, 6.0]))


def test_stage3_neural_cppi_5_gap_budget_multiplier_predeclared():
    # A 20% gap may eat at most 50% of the cushion: m <= 0.50 / 0.20.
    assert GAP_DROP == 0.20 and GAP_LOSS_BUDGET == 0.50
    assert gap_budget_multiplier() == pytest.approx(2.5)
    assert gap_budget_multiplier(gap_drop=0.10,
                                 gap_loss_budget=0.50) == pytest.approx(5.0)


# --- gap-budget layer: one-step loss bound ------------------------------

def _cap_proposal(i, V, tau_frac, cushion, feats):
    """A maximally aggressive proposal: always the leverage cap."""
    return torch.full_like(V, CAP)


def test_stage3_neural_cppi_5_gap_budget_bounds_one_step_gap_loss():
    fn = cppi_vol_budget_weight_fn(_cap_proposal)
    for V_val in (90.5, 95.0, 100.0, 110.0, 150.0, 300.0):
        for vol in (0.05, 0.20, 0.40, 0.80):
            V = torch.tensor([V_val])
            feats = {"vol": torch.tensor([vol]),
                     "vol_ratio": torch.ones(1),
                     "drawdown": torch.zeros(1),
                     "panic_z": torch.zeros(1)}
            cushion = torch.clamp((V - FLOOR) / V, -1.0, 1.0)
            w = float(fn(0, V, 0.5, cushion, feats))
            # Scale-down only: never above the proposal or the cap.
            assert 0.0 <= w <= CAP
            # One-step gap loss as a fraction of wealth, w * d, may
            # not exceed the budget share of the cushion fraction.
            cushion_frac = (V_val - FLOOR) / V_val
            assert w * GAP_DROP <= GAP_LOSS_BUDGET * cushion_frac + 1e-6


def test_stage3_neural_cppi_5_gap_budget_zero_cushion_means_zero_weight():
    fn = cppi_vol_budget_weight_fn(_cap_proposal)
    V = torch.tensor([FLOOR])  # exactly at the floor: no cushion
    feats = {"vol": torch.tensor([0.10]),
             "vol_ratio": torch.ones(1),
             "drawdown": torch.zeros(1),
             "panic_z": torch.zeros(1)}
    w = float(fn(0, V, 0.5, torch.zeros(1), feats))
    assert w == 0.0


# --- frontier metrics: hand-computed fixture ----------------------------

def _fixture_paths():
    # Four paths, floor 90. Hand-computed expectations below.
    return np.array([
        [100.0, 100.0, 100.0],   # never breaches
        [100.0, 80.0, 95.0],     # anytime breach only (min 80)
        [100.0, 85.0, 85.0],     # breaches, terminal shortfall 5
        [100.0, 60.0, 70.0],     # breaches, terminal shortfall 20
    ])


def test_stage3_neural_cppi_5_frontier_metrics_known_answers():
    m = frontier_metrics(_fixture_paths(), floor=90.0)
    assert m["mean_wealth"] == pytest.approx(87.5)
    assert m["p_breach_terminal"] == pytest.approx(0.5)
    assert m["p_breach_anytime"] == pytest.approx(0.75)
    # Anytime shortfalls: 0, 10, 5, 30 -> mean 11.25.
    assert m["mean_shortfall_anytime"] == pytest.approx(11.25)
    # Terminal shortfalls: 0, 0, 5, 20; CVaR95 takes the worst
    # ceil(0.05 * 4) = 1 path -> 20.
    assert m["cvar95_shortfall"] == pytest.approx(20.0)


def test_stage3_neural_cppi_5_frontier_metrics_cvar_tail_mean():
    # 40 paths, terminal shortfalls exactly 0..39 (wealth 90 - k
    # below a floor of 90 with a flat first step at 100 -> construct
    # directly: terminal wealth = 90 - k is below floor; use paths
    # [100, 90 - k] so anytime min equals terminal).
    paths = np.array([[100.0, 90.0 - k] for k in range(40)])
    m = frontier_metrics(paths, floor=90.0)
    # Worst ceil(0.05 * 40) = 2 shortfalls are 39 and 38 -> 38.5.
    assert m["cvar95_shortfall"] == pytest.approx(38.5)
    assert m["p_breach_terminal"] == pytest.approx(39 / 40)


# --- matched leverage and costs ------------------------------------------

def test_stage3_neural_cppi_5_matched_cap_and_cost_config():
    assert CONFIG["cap"] == CAP
    assert CONFIG["cost_rate"] == COST_RATE
    assert CONFIG["floor"] == FLOOR
    assert CONFIG["v0"] == V0


def test_stage3_neural_cppi_5_all_policies_respect_cap():
    policy = seeded_policy(CPPIPolicy, seed=7,
                           feature_names=("vol", "vol_ratio",
                                          "drawdown", "panic_z"))
    strategies = headline_strategies(policy)
    assert set(strategies) == {"static_cppi", "vol_target",
                               "neural_unconstrained", "neural_constrained"}
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.02, size=(8, 21))
    S = torch.tensor(100.0 * np.exp(np.cumsum(
        np.concatenate([np.zeros((8, 1)), rets], axis=1), axis=1)),
        dtype=torch.float32)
    n, steps = 8, 21
    for name, fn in strategies.items():
        V = torch.full((n,), V0)
        feats = {"vol": torch.full((n,), 0.25),
                 "vol_ratio": torch.ones(n),
                 "drawdown": torch.full((n,), 0.05),
                 "panic_z": torch.zeros(n)}
        for i in range(steps):
            cushion = torch.clamp((V - FLOOR) / V, -1.0, 1.0)
            with torch.no_grad():
                w = fn(i, V, 0.5, cushion, feats)
            assert float(w.min()) >= 0.0, name
            assert float(w.max()) <= CAP + 1e-6, name


# --- response curves ------------------------------------------------------

def test_stage3_neural_cppi_5_response_curve_monotonicity():
    policy = seeded_policy(CPPIPolicy, seed=7,
                           feature_names=("vol", "vol_ratio",
                                          "drawdown", "panic_z"))
    strategies = headline_strategies(policy)
    cushion_grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    vol_grid = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80]

    # Static CPPI: weight is nondecreasing in the cushion.
    w = response_curve(strategies["static_cppi"],
                       cushion_grid=cushion_grid, vol=0.20)
    assert all(b >= a - 1e-6 for a, b in zip(w, w[1:]))

    # Vol-targeting: weight is nonincreasing in realized vol.
    w = response_curve(strategies["vol_target"],
                       vol_grid=vol_grid, cushion_ratio=0.10)
    assert all(b <= a + 1e-6 for a, b in zip(w, w[1:]))

    # Constrained layer around a constant cap proposal: weight is
    # nonincreasing in vol at fixed cushion (the budget shrinks).
    fn = cppi_vol_budget_weight_fn(_cap_proposal)
    w = response_curve(fn, vol_grid=vol_grid, cushion_ratio=0.10)
    assert all(b <= a + 1e-6 for a, b in zip(w, w[1:]))

    # Every policy's curve stays inside [0, cap] on both grids.
    for name, strat in strategies.items():
        for curve in (response_curve(strat, cushion_grid=cushion_grid,
                                     vol=0.20),
                      response_curve(strat, vol_grid=vol_grid,
                                     cushion_ratio=0.10)):
            assert all(0.0 <= x <= CAP + 1e-6 for x in curve), name


# --- jump sweep generator --------------------------------------------------

def test_stage3_neural_cppi_5_jump_paths_deterministic_and_sized():
    a = jump_paths(50, 0.20, seed=123)
    b = jump_paths(50, 0.20, seed=123)
    assert np.array_equal(a, b)
    lr = np.log(a[:, 1:] / a[:, :-1])
    # Every path carries exactly one injected gap of log(0.8).
    hits = (np.abs(lr - math.log(0.8)) < 1e-9).sum(axis=1)
    assert (hits == 1).all()
    # The zero-jump set injects nothing: no single-day move anywhere
    # near gap size in calm noise (sigma 1%).
    z = jump_paths(50, 0.0, seed=123)
    lrz = np.log(z[:, 1:] / z[:, :-1])
    assert lrz.min() > math.log(0.95)


# --- honesty guard ---------------------------------------------------------

def test_stage3_neural_cppi_5_disclaimer_states_no_guarantee():
    assert "does not guarantee" in DISCLAIMER
    assert CONFIG["gap_drop"] == GAP_DROP
    assert CONFIG["gap_loss_budget"] == GAP_LOSS_BUDGET
