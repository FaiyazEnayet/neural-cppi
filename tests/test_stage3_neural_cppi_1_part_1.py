"""Part 1 fixtures for R1 (budget-utilization policy).

These tests pin the source contract before any promotion exists: the
policy chooses a fraction u in [0, 1] of the available CPPI risk
budget B_t = min(cap, m_t * max(V_t - F, 0) / V_t), and the executed
weight is w_t = B_t * u. With the same m_t, u = 1 must reproduce
classical CPPI exactly and u = 0 must be all cash. All fixtures are
deterministic seeded synthetic scenarios; no real data, no training
run, no performance claim. This file carries the targeted test key
and substitutes for the proposed tests/test_budget_policy.py name,
following the R8 Part 1 precedent.
"""

import math

import numpy as np
import pytest
import torch

from dhl.benchmarks.cppi_rules import brake_weight_fn, classic_weight_fn
from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    DT,
    FLOOR,
    R,
    V0,
    classic_cppi_weight,
    rollout,
)
from dhl.markets.gbm import simulate_gbm
from dhl.policies.budget_allocator import (
    BudgetFractionNet,
    baseline_report,
    budget_weight_fn,
    constant_fraction,
    cushion_ramp_fraction,
    ledger_self_financing_violations,
    net_fraction_fn,
    oracle_budget_rollout,
    risk_budget,
    validate_fraction,
    vol_threshold_fraction,
)

MULT = 4.0
TOL_EXACT = 1e-10


def _scenario(seed, n_paths=8, n_steps=40, sigma=0.35, dtype=torch.float64):
    S = simulate_gbm(n_paths, n_steps, n_steps * DT, 100.0, R, sigma,
                     seed=seed)
    return torch.tensor(S, dtype=dtype)


def _twenty_scenarios():
    return [(1000 + k, _scenario(1000 + k)) for k in range(20)]


def _recording_fraction(fn, sink):
    def wrapped(i, V, tau_frac, cushion, feats):
        u = fn(i, V, tau_frac, cushion, feats)
        sink.append((float(u.detach().min()), float(u.detach().max())))
        return u

    return wrapped


# --- the budget itself ---------------------------------------------------


def test_stage3_neural_cppi_1_part_1_risk_budget_hand_oracle():
    # V=100, F=90, m=4: cushion 10, budget 4 * 10 / 100 = 0.4.
    assert risk_budget(100.0, floor=90.0, multiplier=4.0) == pytest.approx(0.4)
    # V=95: 4 * 5 / 95.
    assert risk_budget(95.0, floor=90.0, multiplier=4.0) == pytest.approx(
        4.0 * 5.0 / 95.0)
    # At or below the floor the cushion is floored at zero: no budget.
    assert risk_budget(90.0, floor=90.0, multiplier=4.0) == 0.0
    assert risk_budget(85.0, floor=90.0, multiplier=4.0) == 0.0
    # Cap binds: 5 * 110 / 200 = 2.75 -> capped at 1.5.
    assert risk_budget(200.0, floor=90.0, multiplier=5.0,
                       cap=1.5) == pytest.approx(1.5)
    # Tensor form agrees elementwise, and an m_t override wins over the
    # scalar multiplier: halving m halves the budget.
    V = torch.tensor([100.0, 95.0], dtype=torch.float64)
    B = risk_budget(V, floor=90.0, multiplier=4.0)
    assert B.tolist() == pytest.approx([0.4, 4.0 * 5.0 / 95.0])
    B_half = risk_budget(V, floor=90.0, multiplier=4.0,
                         m_t=torch.tensor([2.0, 2.0], dtype=torch.float64))
    assert B_half.tolist() == pytest.approx([0.2, 2.0 * 5.0 / 95.0])


def test_stage3_neural_cppi_1_part_1_budget_fraction_of_classic():
    # For any state, the budget IS the classical CPPI weight at the
    # same multiplier, so B * u with u in [0, 1] never exceeds classic.
    g = torch.Generator().manual_seed(3)
    V = 80.0 + 60.0 * torch.rand(512, generator=g, dtype=torch.float64)
    B = risk_budget(V, floor=FLOOR, multiplier=MULT)
    classic = classic_cppi_weight(V, floor=FLOOR, multiplier=MULT)
    assert float((B - classic).abs().max()) == 0.0
    for u in (0.0, 0.25, 0.5, 1.0):
        assert bool((B * u <= classic + 0.0).all())


def test_stage3_neural_cppi_1_part_1_fraction_validation_fail_closed():
    with pytest.raises(ValueError):
        constant_fraction(-0.1)
    with pytest.raises(ValueError):
        constant_fraction(1.1)
    with pytest.raises(ValueError):
        validate_fraction(torch.tensor([0.5, 1.2]))
    with pytest.raises(ValueError):
        validate_fraction(torch.tensor([-0.2, 0.5]))
    ok = validate_fraction(torch.tensor([0.0, 0.5, 1.0]))
    assert ok.tolist() == [0.0, 0.5, 1.0]

    # A rogue fraction function is refused by the weight function
    # instead of being clamped into a different strategy.
    def rogue(i, V, tau_frac, cushion, feats):
        return torch.full_like(V, 1.5)

    S = _scenario(5, n_paths=2, n_steps=3)
    with pytest.raises(ValueError):
        rollout(S, budget_weight_fn(rogue, multiplier=MULT))


# --- fraction bounds in every seeded fixture -----------------------------


def test_stage3_neural_cppi_1_part_1_fraction_bounds_seeded_fixtures():
    controllers = {
        "cash": constant_fraction(0.0),
        "classic": constant_fraction(1.0),
        "constant_0.5": constant_fraction(0.5),
        "vol_threshold": vol_threshold_fraction(),
        "cushion_ramp": cushion_ramp_fraction(),
    }
    for seed, S in _twenty_scenarios():
        for name, frac_fn in controllers.items():
            sink = []
            rollout(S, budget_weight_fn(
                _recording_fraction(frac_fn, sink), multiplier=MULT))
            lo = min(a for a, _ in sink)
            hi = max(b for _, b in sink)
            assert 0.0 <= lo and hi <= 1.0, (seed, name, lo, hi)


def test_stage3_neural_cppi_1_part_1_fraction_bounds_net_forms():
    torch.manual_seed(0)
    for form in ("residual", "direct"):
        net = BudgetFractionNet(form=form)
        sink = []
        S = _scenario(7, dtype=torch.float32)
        rollout(S, budget_weight_fn(
            _recording_fraction(net_fraction_fn(net), sink),
            multiplier=MULT))
        lo = min(a for a, _ in sink)
        hi = max(b for _, b in sink)
        assert 0.0 <= lo and hi <= 1.0, (form, lo, hi)


# --- u = 1 is classical CPPI; u = 0 is all cash --------------------------


def test_stage3_neural_cppi_1_part_1_fraction_one_matches_classic_exactly():
    for seed, mult in ((11, 4.0), (12, 8.0), (13, 3.0)):
        S = _scenario(seed, n_paths=6, n_steps=30)
        _, _, led_budget = rollout(
            S, budget_weight_fn(constant_fraction(1.0), multiplier=mult),
            return_path=True, return_ledger=True)
        _, _, led_classic = rollout(
            S, classic_weight_fn(multiplier=mult),
            return_path=True, return_ledger=True)
        for key in ("holding_post", "fee", "weight", "cash_post"):
            diff = float((led_budget[key] - led_classic[key]).abs().max())
            assert diff <= TOL_EXACT, (seed, mult, key, diff)
        fee_diff = float((led_budget["cum_fees"]
                          - led_classic["cum_fees"]).abs().max())
        assert fee_diff <= TOL_EXACT


def test_stage3_neural_cppi_1_part_1_fraction_zero_is_all_cash():
    S = _scenario(21, n_steps=25)
    V_T, path, ledger = rollout(
        S, budget_weight_fn(constant_fraction(0.0), multiplier=MULT),
        return_path=True, return_ledger=True)
    steps = S.shape[1] - 1
    assert float(ledger["weight"].abs().max()) == 0.0
    assert float(ledger["cum_fees"].abs().max()) == 0.0
    expected = V0 * math.exp(R * steps * DT)
    assert float((V_T - expected).abs().max()) <= 1e-9
    assert float((path[:, -1] - V_T).abs().max()) == 0.0


def test_stage3_neural_cppi_1_part_1_tightening_lives_in_budget():
    # Emergency tightening expressed through m_t (halve the multiplier
    # while 21d vol exceeds 0.35) must reproduce the shipped brake rung
    # exactly: the tightening is part of the budget, not an override.
    def halving(i, V, feats):
        vol = feats.get("vol", torch.zeros_like(V))
        return torch.where(vol > 0.35, torch.full_like(V, MULT * 0.5),
                           torch.full_like(V, MULT))

    S = _scenario(31, n_paths=6, n_steps=40, sigma=0.6)
    _, _, led_budget = rollout(
        S, budget_weight_fn(constant_fraction(1.0), multiplier=MULT,
                            multiplier_fn=halving),
        return_path=True, return_ledger=True)
    _, _, led_brake = rollout(
        S, brake_weight_fn(multiplier=MULT),
        return_path=True, return_ledger=True)
    diff = float((led_budget["weight"] - led_brake["weight"]).abs().max())
    assert diff <= TOL_EXACT


# --- independent oracle ---------------------------------------------------


def test_stage3_neural_cppi_1_part_1_oracle_matches_rollout():
    S = _scenario(41, n_paths=4, n_steps=25)
    prices = S.numpy()
    for u in (0.0, 0.5, 1.0):
        V_T, _, ledger = rollout(
            S, budget_weight_fn(constant_fraction(u), multiplier=MULT),
            return_path=True, return_ledger=True)
        for p in range(S.shape[0]):
            oracle = oracle_budget_rollout(prices[p], u, multiplier=MULT)
            assert abs(oracle["terminal_wealth"]
                       - float(V_T[p])) <= 1e-9, (u, p)
            assert abs(oracle["total_fees"]
                       - float(ledger["cum_fees"][p])) <= 1e-9, (u, p)
            hold_diff = np.abs(
                oracle["holding_post"] - ledger["holding_post"][p].numpy())
            fee_diff = np.abs(
                oracle["fee"] - ledger["fee"][p].numpy())
            assert float(hold_diff.max()) <= 1e-9, (u, p)
            assert float(fee_diff.max()) <= 1e-9, (u, p)


def test_stage3_neural_cppi_1_part_1_oracle_hand_case():
    # One path, two steps, no price move: V stays 100 until fees.
    # u=1, m=4: w = 0.4 at both steps. Step 0 buys 0.4 * V' where
    # V' = 100 - 0.002 * H', so H' = 40 / 1.0008 = 39.968025...
    prices = np.array([100.0, 100.0, 100.0])
    oracle = oracle_budget_rollout(prices, 1.0, multiplier=4.0)
    h1 = 40.0 / 1.0008
    assert oracle["holding_post"][0] == pytest.approx(h1, abs=1e-9)
    assert oracle["fee"][0] == pytest.approx(0.002 * h1, abs=1e-11)
    # Cash check: fee-only loss plus cash growth at r over the window,
    # then liquidation of the same holding at the same price.
    assert oracle["breach_anytime"] is False
    assert oracle["min_wealth"] >= FLOOR


# --- self-financing over 20 seeded matched scenarios ----------------------


def test_stage3_neural_cppi_1_part_1_twenty_scenarios_self_financing():
    torch.manual_seed(0)
    net = BudgetFractionNet(form="residual")
    controllers = {
        "classic": constant_fraction(1.0),
        "constant_0.5": constant_fraction(0.5),
        "vol_threshold": vol_threshold_fraction(),
        "cushion_ramp": cushion_ramp_fraction(),
        "net_residual_untrained": net_fraction_fn(net),
    }
    total = 0
    for seed, S in _twenty_scenarios():
        for name, frac_fn in controllers.items():
            _, _, ledger = rollout(
                S, budget_weight_fn(frac_fn, multiplier=MULT),
                return_path=True, return_ledger=True)
            total += ledger_self_financing_violations(ledger)
    assert total == 0


# --- the bounded fraction network contract --------------------------------


def test_stage3_neural_cppi_1_part_1_net_init_close_to_cppi_unsaturated():
    for form in ("residual", "direct"):
        torch.manual_seed(1)
        net = BudgetFractionNet(form=form)
        # Varied ex-ante states: the initial fraction is the same for
        # all of them (zero final-layer weights by construction) and
        # sits close to CPPI (u = 1 - sigmoid(-2) = sigmoid(2)).
        tau = torch.tensor([0.1, 0.5, 1.0])
        cushion = torch.tensor([-0.05, 0.1, 0.3])
        feats = {"vol": torch.tensor([0.1, 0.4, 0.9])}
        u = net(tau, cushion, feats)
        expected = 1.0 / (1.0 + math.exp(-2.0))
        assert u.shape == (3,)
        assert float((u.detach() - expected).abs().max()) <= 1e-6
        assert bool(((u > 0.8) & (u < 0.95)).all())


def test_stage3_neural_cppi_1_part_1_net_bounded_and_state_dependent():
    torch.manual_seed(2)
    net = BudgetFractionNet(form="residual")
    # Perturb the final layer so the fraction can depend on state.
    with torch.no_grad():
        net.net[-1].weight.fill_(1.5)
    tau = torch.tensor([0.5, 0.5])
    cushion = torch.tensor([-0.5, 0.5])
    feats = {"vol": torch.tensor([0.05, 1.5])}
    u = net(tau, cushion, feats)
    assert bool(((u > 0.0) & (u < 1.0)).all())
    assert float(u.detach()[0]) != float(u.detach()[1])
    # Extreme states still cannot saturate past the logit clip: the
    # fraction stays within [sigmoid(-6), sigmoid(6)] and strictly
    # inside (0, 1).
    big = {"vol": torch.tensor([1e6, -1e6])}
    u_big = net(torch.tensor([1e6, -1e6]),
                torch.tensor([1e6, -1e6]), big)
    lo = 1.0 / (1.0 + math.exp(6.0))
    hi = 1.0 / (1.0 + math.exp(-6.0))
    assert bool(((u_big >= lo - 1e-9) & (u_big <= hi + 1e-9)).all())
    assert bool(((u_big > 0.0) & (u_big < 1.0)).all())
    with pytest.raises(ValueError):
        BudgetFractionNet(form="sideways")


# --- baseline capture report ----------------------------------------------


def test_stage3_neural_cppi_1_part_1_baseline_report():
    scenarios = [(2000 + k, _scenario(2000 + k, n_paths=6, n_steps=30))
                 for k in range(3)]
    report = baseline_report(scenarios, multiplier=MULT)
    rows = {row["controller"]: row for row in report["controllers"]}
    assert set(rows) == {"cash", "classic", "constant_0.5",
                         "constant_0.25", "vol_threshold", "cushion_ramp"}
    for row in rows.values():
        for key in ("mean_terminal_wealth", "mean_return", "mean_shortfall",
                    "breach_count_terminal", "breach_count_anytime",
                    "mean_fees", "fraction_min", "fraction_max"):
            assert key in row
        assert 0.0 <= row["fraction_min"] <= row["fraction_max"] <= 1.0
    # Cash is exact: no fees, terminal wealth is pure cash growth.
    steps = 30
    assert rows["cash"]["mean_fees"] == 0.0
    assert rows["cash"]["mean_terminal_wealth"] == pytest.approx(
        V0 * math.exp(R * steps * DT), abs=1e-9)
    # Parity and accounting checks are part of the report itself.
    assert report["parity"]["max_holding_diff"] <= TOL_EXACT
    assert report["parity"]["max_fee_diff"] <= TOL_EXACT
    assert report["self_financing_violations"] == 0
    assert report["n_scenarios"] == 3
