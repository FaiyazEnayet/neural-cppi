"""Stage 3, neural-cppi R1 part 2: training and experiment tests.

R1 (budget-utilization policy) part 2 trains the bounded budget-fraction
network from part 1 and runs the utilization experiments.  The parent
acceptance criteria are re-pinned here so part 2 cannot regress them:
the fraction stays in [0, 1] in every deterministic seed fixture,
fraction 1 exactly matches reference classical CPPI risky holdings and
costs to <= 1e-10 per step, and 20 seeded matched scenarios satisfy 0
self-financing violations, now including briefly trained policies among
the audited controllers.

Training tests use tiny fixtures (a few scenarios, paths and epochs) so
the suite stays fast; the shipped experiment scale lives in
scripts/experiments/budget_utilization.py --part2.
"""
import math

import numpy as np
import pytest
import torch

from dhl.benchmarks.cppi_rules import classic_cppi_weight
from dhl.experiments.cppi_learned import COST_RATE, DT, FLOOR, R, rollout
from dhl.markets.gbm import simulate_gbm
from dhl.policies.budget_allocator import (
    SEED_SETS,
    BudgetFractionNet,
    budget_experiment_scenarios,
    budget_weight_fn,
    constant_fraction,
    deterministic_controllers,
    evaluate_fraction_controllers,
    ledger_self_financing_violations,
    make_gbm_scenarios,
    net_fraction_fn,
    risk_budget,
    select_derisk_penalty,
    train_budget_fraction_net,
)


def _scenario(seed, n_paths=8, n_steps=40, sigma=0.35, drift=R,
              dtype=torch.float64):
    S = simulate_gbm(n_paths, n_steps, n_steps * DT, 100.0, drift, sigma,
                     seed=seed)
    return torch.tensor(S, dtype=dtype)


def _classic_fn(i, V, tau_frac, cushion, feats):
    # classic_cppi_weight is a plain weight function (V -> w); the
    # rollout weight-function shape wraps it like baseline_report does.
    return classic_cppi_weight(V)


def _twenty_scenarios():
    return [(1000 + k, _scenario(1000 + k)) for k in range(20)]


def _tiny_train_scenarios():
    # Stress-only fixture: downside pressure, so the floor penalty binds
    # and the untrained fraction near 0.88 is visibly suboptimal.
    return [(2000 + k, _scenario(2000 + k, n_paths=6, n_steps=30,
                                  sigma=0.55, drift=-0.25))
            for k in range(4)]


def _tiny_validation_scenarios():
    return [(3000 + k, _scenario(3000 + k, n_paths=6, n_steps=30,
                                  sigma=0.55, drift=-0.25))
            for k in range(4)]


def _trained(form="residual", epochs=10, derisk_penalty=0.0, seed=3):
    torch.manual_seed(seed)
    net = BudgetFractionNet(form=form)
    history = train_budget_fraction_net(
        net, _tiny_train_scenarios(), n_epochs=epochs,
        derisk_penalty=derisk_penalty, seed=seed)
    return net, history


# --- scenario protocol -----------------------------------------------------

def test_stage3_neural_cppi_1_part_2_seed_sets_disjoint():
    names = sorted(SEED_SETS)
    for a in range(len(names)):
        assert len(SEED_SETS[names[a]]) == 20
        for b in range(a + 1, len(names)):
            overlap = set(SEED_SETS[names[a]]) & set(SEED_SETS[names[b]])
            assert overlap == set(), (names[a], names[b], overlap)


def test_stage3_neural_cppi_1_part_2_scenario_builder_shapes_and_regimes():
    train = budget_experiment_scenarios("training", n_paths=3, n_steps=12)
    assert len(train) == 20
    assert all(S.shape == (3, 13) for _, S in train)
    # Regime mixture by seed modulo 3: seeds 2000 (calm), 2001 (stress)
    # and 2002 (gap) head the set, in seed order.
    assert [seed for seed, _ in train[:3]] == [2000, 2001, 2002]
    base = budget_experiment_scenarios("baseline", n_paths=3, n_steps=12)
    ref = make_gbm_scenarios([1000], n_paths=3, n_steps=12)
    assert torch.equal(base[0][1], ref[0][1])  # baseline reproduces part 1
    gap = budget_experiment_scenarios("gap", n_paths=3, n_steps=12)
    calm = make_gbm_scenarios([5000], n_paths=3, n_steps=12)
    # The gap family shares the calm GBM draws but not the prices: each
    # path has exactly one one-step return at or below -25%.
    assert not torch.equal(gap[0][1], calm[0][1])
    rets = gap[0][1][:, 1:] / gap[0][1][:, :-1] - 1.0
    assert bool(((rets <= -0.25).sum(dim=1) == 1).all())
    with pytest.raises(ValueError):
        budget_experiment_scenarios("nope")


# --- parent acceptance, re-pinned ------------------------------------------

def test_stage3_neural_cppi_1_part_2_fraction_one_parity_not_regressed():
    for seed in (11, 13):
        S = _scenario(seed)
        _, led_u = rollout(S, budget_weight_fn(constant_fraction(1.0)),
                           return_ledger=True)
        _, led_c = rollout(S, _classic_fn, return_ledger=True)
        assert float((led_u["holding_post"]
                      - led_c["holding_post"]).abs().max()) <= 1e-10
        assert float((led_u["fee"] - led_c["fee"]).abs().max()) <= 1e-10
        assert float((led_u["cum_fees"] - led_c["cum_fees"]).abs().max()) \
            <= 1e-10


def test_stage3_neural_cppi_1_part_2_trained_fractions_bounded_within_budget():
    scenarios = _twenty_scenarios()
    for form in ("residual", "direct"):
        net, _ = _trained(form=form)
        fn = net_fraction_fn(net)
        for _, S in scenarios:
            seen, seen_V = [], []

            def recording(i, V, tau, cushion, feats, _fn=fn):
                u = _fn(i, V, tau, cushion, feats)
                seen.append(u.detach().clone())
                seen_V.append(V.detach().clone())
                return u

            _, led_u = rollout(S, budget_weight_fn(recording),
                               return_ledger=True)
            U = torch.stack(seen, dim=1)
            assert float(U.min()) >= 0.0 and float(U.max()) <= 1.0
            # On the policy's own trajectory the executed weight is
            # exactly the budget times the fraction, so it never
            # exceeds the budget classical CPPI would have available
            # at the same wealth.
            budget = risk_budget(torch.stack(seen_V, dim=1))
            assert bool((led_u["weight"] <= budget + 1e-10).all())
            assert float((led_u["weight"] - budget * U).abs().max()) \
                <= 1e-9


def test_stage3_neural_cppi_1_part_2_twenty_scenarios_self_financing():
    scenarios = _twenty_scenarios()
    nets = {}
    for form in ("residual", "direct"):
        net, _ = _trained(form=form, epochs=6)
        nets[f"trained_{form}"] = net_fraction_fn(net)
    controllers = dict(deterministic_controllers())
    controllers.update(nets)
    for name, fn in controllers.items():
        weight = budget_weight_fn(fn)
        total = 0
        for _, S in scenarios:
            _, led = rollout(S, weight, return_ledger=True)
            total += ledger_self_financing_violations(led)
        assert total == 0, name


# --- trainer ---------------------------------------------------------------

def test_stage3_neural_cppi_1_part_2_training_deterministic():
    torch.manual_seed(3)
    net_a = BudgetFractionNet(form="residual")
    hist_a = train_budget_fraction_net(net_a, _tiny_train_scenarios(),
                                       n_epochs=6, seed=3)
    torch.manual_seed(3)
    net_b = BudgetFractionNet(form="residual")
    hist_b = train_budget_fraction_net(net_b, _tiny_train_scenarios(),
                                       n_epochs=6, seed=3)
    for pa, pb in zip(net_a.state_dict().values(),
                      net_b.state_dict().values()):
        assert torch.equal(pa, pb)
    assert hist_a == hist_b
    assert len(hist_a) == 7  # initial eval + one entry per epoch
    torch.manual_seed(3)
    fresh = BudgetFractionNet(form="residual")
    changed = any(not torch.equal(p, q) for p, q in
                  zip(net_a.state_dict().values(),
                      fresh.state_dict().values()))
    assert changed  # training actually moved the parameters


def test_stage3_neural_cppi_1_part_2_objective_decomposition():
    scenarios = _tiny_train_scenarios()
    torch.manual_seed(3)
    net0 = BudgetFractionNet(form="residual")
    h0 = train_budget_fraction_net(net0, scenarios, n_epochs=1,
                                   derisk_penalty=0.0, seed=3)[0]
    torch.manual_seed(3)
    net2 = BudgetFractionNet(form="residual")
    h2 = train_budget_fraction_net(net2, scenarios, n_epochs=1,
                                   derisk_penalty=2.0, seed=3)[0]
    # Epoch 0 is the pre-update evaluation: identical nets and scenarios
    # give identical components, and the loss differs by exactly
    # lambda * mean de-risking.
    assert h0["mean_wealth"] == h2["mean_wealth"]
    assert h0["mean_shortfall"] == h2["mean_shortfall"]
    assert h0["mean_derisk"] == h2["mean_derisk"]
    assert h0["mean_fraction"] == pytest.approx(1.0 - h0["mean_derisk"])
    delta = h2["loss"] - h0["loss"]
    assert delta == pytest.approx(2.0 * h0["mean_derisk"], abs=1e-9)
    expect = -h0["mean_wealth"] + 30.0 * h0["mean_shortfall"]
    assert h0["loss"] == pytest.approx(expect, abs=1e-9)


def test_stage3_neural_cppi_1_part_2_training_reduces_objective():
    _, history = _trained(epochs=40, seed=3)
    assert all(math.isfinite(v) for h in history
               for v in (h["loss"], h["mean_wealth"],
                         h["mean_shortfall"]))
    assert history[-1]["loss"] < history[0]["loss"]


def test_stage3_neural_cppi_1_part_2_derisk_penalty_direction():
    scenarios = _tiny_train_scenarios()
    net_free, _ = _trained(epochs=40, derisk_penalty=0.0, seed=3)
    net_pen, _ = _trained(epochs=40, derisk_penalty=4.0, seed=3)
    rows = evaluate_fraction_controllers(
        scenarios, {"free": net_fraction_fn(net_free),
                    "penalized": net_fraction_fn(net_pen)})
    by_name = {r["controller"]: r for r in rows["controllers"]}
    # The penalty prices unused budget, so the penalized policy keeps a
    # weakly higher mean fraction on the same training fixture.
    assert by_name["penalized"]["mean_fraction"] \
        >= by_name["free"]["mean_fraction"] - 0.02


def test_stage3_neural_cppi_1_part_2_fraction_causality():
    net, _ = _trained(epochs=10)
    fn = net_fraction_fn(net)
    base = _scenario(5, n_paths=2, n_steps=30, sigma=0.45)
    alt = base.clone()
    other = _scenario(6, n_paths=2, n_steps=30, sigma=0.45)
    alt[:, 16:] = other[:, 16:]  # identical prices through index 15

    def recorded(S):
        seen = []

        def rec(i, V, tau, cushion, feats):
            u = fn(i, V, tau, cushion, feats)
            seen.append(u.detach().clone())
            return u

        rollout(S, budget_weight_fn(rec))
        return torch.stack(seen)

    ua, ub = recorded(base), recorded(alt)
    # Decisions at steps 0..15 see only the shared price prefix.
    assert torch.equal(ua[:16], ub[:16])


def test_stage3_neural_cppi_1_part_2_training_with_emergency_tightening():
    def halving(i, V, feats):
        if feats is not None and "vol" in feats \
                and bool((feats["vol"] > 0.35).any()):
            return 2.0
        return 4.0

    torch.manual_seed(3)
    net = BudgetFractionNet(form="residual")
    hist = train_budget_fraction_net(net, _tiny_train_scenarios(),
                                     n_epochs=6, multiplier_fn=halving,
                                     seed=3)
    assert all(math.isfinite(h["loss"]) for h in hist)
    # On the trained policy's own trajectory, the executed weight at
    # every step stays within the tightened budget for that step:
    # w_t = B_t(m_t) * u with u <= 1.  The per-step wealth and
    # multiplier are captured from the rollout itself, since a
    # cross-rollout comparison would mix different wealth paths.
    seen_V, seen_m = [], []

    def recording(i, V, tau, cushion, feats):
        seen_V.append(V.detach().clone())
        return net_fraction_fn(net)(i, V, tau, cushion, feats)

    def recording_mult(i, V, feats):
        m = halving(i, V, feats)
        seen_m.append(torch.full_like(V, float(m)))
        return m

    S = _scenario(2000, n_paths=6, n_steps=30, sigma=0.55, drift=-0.25)
    _, led = rollout(
        S, budget_weight_fn(recording, multiplier_fn=recording_mult),
        return_ledger=True)
    V_pre = torch.stack(seen_V, dim=1)
    m_t = torch.stack(seen_m, dim=1)
    budget = risk_budget(V_pre, m_t=m_t)
    assert bool((led["weight"] <= budget + 1e-10).all())
    assert bool((m_t == 2.0).any())  # the brake actually engaged


# --- selection and evaluation ------------------------------------------------

def test_stage3_neural_cppi_1_part_2_penalty_selection_protocol():
    result = select_derisk_penalty(
        [1.0, 0.0], _tiny_train_scenarios(),
        _tiny_validation_scenarios(), n_epochs=6, seed=3)
    assert result["selected_derisk_penalty"] in (0.0, 1.0)
    table = result["table"]
    assert [row["derisk_penalty"] for row in table] == [0.0, 1.0]
    for row in table:
        expect = row["mean_terminal_wealth"] - 30.0 * row["mean_shortfall"]
        assert row["validation_objective"] == pytest.approx(expect)
        assert 0.0 <= row["mean_fraction"] <= 1.0
        assert row["fraction_min"] >= 0.0 and row["fraction_max"] <= 1.0


def test_stage3_neural_cppi_1_part_2_evaluator_matches_part1_baseline():
    from dhl.policies.budget_allocator import baseline_report

    scenarios = [(1000 + k, _scenario(1000 + k, n_paths=6, n_steps=30,
                                       sigma=0.30))
                 for k in range(3)]
    controllers = deterministic_controllers()
    mine = evaluate_fraction_controllers(scenarios, controllers)
    theirs = baseline_report(scenarios)
    by_mine = {r["controller"]: r for r in mine["controllers"]}
    by_theirs = {r["controller"]: r for r in theirs["controllers"]}
    for name in controllers:
        for key in ("mean_terminal_wealth", "mean_return", "mean_shortfall",
                    "breach_count_terminal", "breach_count_anytime",
                    "mean_fees", "fraction_min", "fraction_max", "n_paths"):
            assert by_mine[name][key] == \
                pytest.approx(by_theirs[name][key], abs=1e-12), (name, key)
    # Cash row is exact: no trades, wealth grows at the cash rate.
    steps = scenarios[0][1].shape[1] - 1
    assert by_mine["cash"]["mean_terminal_wealth"] == \
        pytest.approx(100.0 * float(np.exp(R * steps * DT)), abs=1e-9)
    assert mine["self_financing_violations"] == 0
    assert by_mine["classic"]["mean_fraction"] == 1.0
    assert by_mine["cash"]["mean_fraction"] == 0.0
    assert 0.0 < by_mine["constant_0.5"]["mean_fraction"] < 1.0
