"""Part 1 fixtures for E5 (budget-action candidate comparison).

These tests pin the source contract before any training exists: the
four predeclared candidates act through the R1 budget-fraction
contract on identical paths with matched features, decision history
and initialization; the comparison harness reproduces the
independent bisection oracle; the hierarchical selection rule and
the training-log schema are fixed as tested code. All fixtures are
deterministic seeded synthetic scenarios; no real data, no training
run, no performance claim. This file carries the targeted test key
and substitutes for the proposed tests/test_budget_comparison.py
name, following the R8/R1 Part 1 precedents.
"""

import math

import pytest
import torch

from dhl.experiments.cppi_learned import (
    FEATURE_NAMES,
    FLOOR,
    R,
    V0,
)
from dhl.policies.budget_allocator import (
    SEED_SETS as R1_SEED_SETS,
    constant_fraction,
    oracle_budget_rollout,
)
from dhl.policies.candidate_comparison import (
    CANDIDATE_ORDER,
    CANDIDATE_SEED,
    EPISODE_PROTOCOL,
    E5_FEATURE_NAMES,
    E5_SEED_SETS,
    LinearLogisticPolicy,
    candidate_fraction_fn,
    candidate_model,
    candidate_specs,
    e5_scenarios,
    evaluate_candidates,
    gradient_null_fraction,
    hierarchical_select,
    max_drawdown,
    reference_fraction_fn,
    validate_training_log,
)

SIGMOID2 = 1.0 / (1.0 + math.exp(-2.0))


def _small_family(name="training", n_scenarios=5):
    scenarios = e5_scenarios(name, n_paths=4, n_steps=25)
    return scenarios[:n_scenarios]


def _all_fraction_fns():
    fns = {name: candidate_fraction_fn(name)
           for name in CANDIDATE_ORDER}
    for name in ("cash", "classic"):
        fns[name] = reference_fraction_fn(name)
    return fns


# --- the declared protocol ---------------------------------------------------


def test_stage3_neural_cppi_5_part_1_seed_sets_disjoint():
    e5 = E5_SEED_SETS
    names = list(e5)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(e5[a]) & set(e5[b]), (a, b)
    # The baseline family is R1's, shared deliberately; every other
    # E5 family is disjoint from every R1 seed set.
    assert e5["baseline"] == R1_SEED_SETS["baseline"]
    r1_all = set().union(*(set(v) for v in R1_SEED_SETS.values()))
    for name in ("training", "validation", "stress", "gap",
                 "episodes"):
        assert not set(e5[name]) & r1_all, name
    # Episode seeds are declarations for Part 2's real-data run.
    assert list(e5["episodes"]) == EPISODE_PROTOCOL["seeds"]
    with pytest.raises(ValueError):
        e5_scenarios("episodes")


def test_stage3_neural_cppi_5_part_1_scenario_builders():
    train = e5_scenarios("training", n_paths=3, n_steps=12)
    assert len(train) == 20
    assert train[0][0] == 21000
    for _, S in train:
        assert S.shape == (3, 13)
    base = e5_scenarios("baseline", n_paths=3, n_steps=12)
    assert base[0][0] == 1000
    # The gap family is the calm family at the same seeds with one
    # seeded drop per path: the first price is untouched, the path
    # as a whole is not.
    calm = e5_scenarios("baseline", n_paths=3, n_steps=12)
    gap_seeds = E5_SEED_SETS["gap"]
    from dhl.policies.budget_allocator import make_gbm_scenarios
    calm_gap = make_gbm_scenarios(list(gap_seeds), 3, 12)
    gap = e5_scenarios("gap", n_paths=3, n_steps=12)
    for (seed_g, S_g), (seed_c, S_c) in zip(gap, calm_gap):
        assert seed_g == seed_c
        assert float(S_g[0, 0]) == float(S_c[0, 0])
        assert not torch.equal(S_g, S_c)
    assert len(base) == 20 and len(gap) == 20


def test_stage3_neural_cppi_5_part_1_candidate_specs():
    specs = {s["name"]: s for s in candidate_specs()}
    assert [s["name"] for s in candidate_specs()] == [
        "baseline_mlp", "linear_logistic", "residual_utilization",
        "dynamic_rule", "cash", "classic"]
    for name in CANDIDATE_ORDER[:3]:
        assert specs[name]["kind"] == "learned"
        assert specs[name]["trainable"] is True
    assert specs["dynamic_rule"]["trainable"] is False
    assert specs["cash"]["kind"] == "reference"
    assert E5_FEATURE_NAMES == FEATURE_NAMES
    with pytest.raises(ValueError):
        candidate_model("not_a_candidate")
    with pytest.raises(ValueError):
        reference_fraction_fn("not_a_reference")


# --- the linear/logistic candidate -------------------------------------------


def _feats(vol=0.3, vol_ratio=1.0, drawdown=0.0, panic_z=0.0, n=1):
    return {
        "vol": torch.full((n,), vol),
        "vol_ratio": torch.full((n,), vol_ratio),
        "drawdown": torch.full((n,), drawdown),
        "panic_z": torch.full((n,), panic_z),
    }


def test_stage3_neural_cppi_5_part_1_linear_logistic_hand_oracle():
    policy = LinearLogisticPolicy()
    # Zero weights: the fraction is sigmoid(2) for every state.
    u = policy(torch.tensor([0.2, 0.9]), torch.tensor([-0.4, 0.3]),
               _feats(n=2))
    assert u.shape == (2,)
    assert float((u.detach() - SIGMOID2).abs().max()) <= 1e-6
    # Known weights: logit = 0.5*tau - 1.0*cushion + 2.0*vol + 0.25.
    with torch.no_grad():
        policy.linear.weight.copy_(
            torch.tensor([[0.5, -1.0, 2.0, 0.0, 0.0, 0.0]]))
        policy.linear.bias.fill_(0.25)
    with torch.no_grad():
        u = policy(torch.tensor([0.5]), torch.tensor([0.1]),
                   _feats(vol=0.3))
    logit = 0.5 * 0.5 - 1.0 * 0.1 + 2.0 * 0.3 + 0.25
    assert float(u[0]) == pytest.approx(
        1.0 / (1.0 + math.exp(-logit)), abs=1e-6)
    # Extreme weights cannot saturate past the logit clip.
    with torch.no_grad():
        policy.linear.weight.fill_(1e6)
        u = policy(torch.tensor([1.0]), torch.tensor([1.0]),
                   _feats(vol=1.0, vol_ratio=1.0, drawdown=1.0,
                          panic_z=1.0))
    hi = 1.0 / (1.0 + math.exp(-6.0))
    assert 0.0 < float(u[0]) <= hi + 1e-6
    with pytest.raises(ValueError):
        LinearLogisticPolicy(feature_names=("nope",))


def test_stage3_neural_cppi_5_part_1_learned_candidates_matched_init():
    # Every learned candidate starts at the same state-independent
    # fraction sigmoid(2), and the dynamic rule has no model.
    assert candidate_model("dynamic_rule") is None
    tau = torch.tensor([0.1, 0.5, 1.0])
    cushion = torch.tensor([-0.05, 0.1, 0.3])
    feats = _feats(vol=0.4, n=3)
    for name in CANDIDATE_ORDER[:3]:
        model = candidate_model(name, seed=CANDIDATE_SEED)
        assert model.feature_names == E5_FEATURE_NAMES
        u = model(tau, cushion, feats)
        assert float((u.detach() - SIGMOID2).abs().max()) <= 1e-6, name
        assert float(u.detach().max() - u.detach().min()) <= 1e-7, name
    # Input width is the matched decision-history width for all three.
    linear = candidate_model("linear_logistic")
    assert linear.linear.in_features == 2 + len(E5_FEATURE_NAMES)
    for name in ("baseline_mlp", "residual_utilization"):
        net = candidate_model(name)
        assert net.net[0].in_features == 2 + len(E5_FEATURE_NAMES)


# --- matched features and decision history -------------------------------------


def test_stage3_neural_cppi_5_part_1_identical_decision_history():
    from dhl.experiments.cppi_learned import rollout
    from dhl.policies.budget_allocator import budget_weight_fn

    S = e5_scenarios("training", n_paths=3, n_steps=15)[0][1]
    histories = {}
    for name, frac_fn in _all_fraction_fns().items():
        seen = []

        def recording(i, V, tau_frac, cushion, feats, _fn=frac_fn,
                      _seen=seen):
            _seen.append((i, tau_frac,
                          {k: v.clone() for k, v in feats.items()}))
            return _fn(i, V, tau_frac, cushion, feats)

        rollout(S, budget_weight_fn(recording))
        histories[name] = seen
    ref = histories["classic"]
    for name, seen in histories.items():
        assert len(seen) == len(ref), name
        for (i, tau, feats), (i_r, tau_r, feats_r) in zip(seen, ref):
            assert i == i_r and tau == tau_r, name
            assert set(feats) == set(E5_FEATURE_NAMES), name
            for key in E5_FEATURE_NAMES:
                assert torch.equal(feats[key], feats_r[key]), (name, key)


def test_stage3_neural_cppi_5_part_1_fraction_bounds_and_accounting():
    report = evaluate_candidates(_small_family(), _all_fraction_fns())
    assert report["self_financing_violations"] == 0
    assert report["feature_names"] == list(E5_FEATURE_NAMES)
    for row in report["candidates"]:
        assert 0.0 <= row["fraction_min"] <= row["fraction_max"] <= 1.0, \
            row["candidate"]
        assert row["self_financing_violations"] == 0, row["candidate"]
        assert row["n_paths"] == 20


# --- independent oracles --------------------------------------------------------


def test_stage3_neural_cppi_5_part_1_harness_matches_oracle():
    S = e5_scenarios("baseline", n_paths=16, n_steps=63)[0][1][:1]
    prices = S[0].tolist()
    scenarios = [(1000, S)]
    # Constant fractions through the harness vs the bisection oracle.
    for u in (0.0, 0.5, 1.0):
        report = evaluate_candidates(
            scenarios, {"c": constant_fraction(u)})
        row = report["candidates"][0]
        oracle = oracle_budget_rollout(prices, u)
        assert abs(row["mean_terminal_wealth"]
                   - oracle["terminal_wealth"]) <= 1e-9, u
        assert abs(row["mean_fees"] - oracle["total_fees"]) <= 1e-9, u
        expected_anytime = max(FLOOR - oracle["min_wealth"], 0.0)
        assert abs(row["mean_anytime_shortfall"]
                   - expected_anytime) <= 1e-9, u
    # The linear policy at its matched init is the constant fraction
    # sigmoid(2); compare through the same kernel (its fraction is
    # computed in float32, hence the looser tolerance) and then the
    # constant against the oracle at full precision.
    report = evaluate_candidates(
        scenarios, {"linear": candidate_fraction_fn("linear_logistic"),
                    "const": constant_fraction(SIGMOID2)})
    rows = {r["candidate"]: r for r in report["candidates"]}
    assert abs(rows["linear"]["mean_terminal_wealth"]
               - rows["const"]["mean_terminal_wealth"]) <= 1e-4
    oracle = oracle_budget_rollout(prices, SIGMOID2)
    assert abs(rows["const"]["mean_terminal_wealth"]
               - oracle["terminal_wealth"]) <= 1e-9


def test_stage3_neural_cppi_5_part_1_metric_oracles():
    assert max_drawdown([100.0, 110.0, 99.0, 105.0, 90.0]) == 20.0
    assert max_drawdown([100.0, 101.0, 102.0]) == 0.0
    assert max_drawdown([100.0]) == 0.0
    assert max_drawdown([100.0, 80.0, 120.0, 60.0]) == 60.0


def test_stage3_neural_cppi_5_part_1_baseline_capture_references():
    scenarios = e5_scenarios("baseline")[:5]
    report = evaluate_candidates(scenarios, _all_fraction_fns())
    rows = {r["candidate"]: r for r in report["candidates"]}
    assert report["seeds"] == [1000 + k for k in range(5)]
    required = {
        "n_paths", "mean_terminal_wealth", "mean_return",
        "mean_terminal_shortfall", "breach_count_terminal",
        "breach_count_anytime", "mean_anytime_shortfall",
        "mean_max_drawdown", "mean_fees", "n_negative_terminal",
        "min_terminal_wealth", "mean_fraction", "fraction_min",
        "fraction_max", "cap_activation_share", "budget_zero_share",
        "self_financing_violations",
    }
    for row in rows.values():
        assert required <= set(row), row["candidate"]
    # Cash is exact on any scenario subset: no fees, pure cash growth.
    steps = 63
    assert rows["cash"]["mean_fees"] == 0.0
    assert rows["cash"]["mean_terminal_wealth"] == pytest.approx(
        V0 * math.exp(R * steps * (1.0 / 252.0)), abs=1e-9)
    assert rows["cash"]["mean_max_drawdown"] == 0.0
    assert rows["cash"]["n_negative_terminal"] == 0
    # Classic holds the full budget at every step.
    assert rows["classic"]["fraction_min"] == 1.0
    assert rows["classic"]["fraction_max"] == 1.0
    # The three untrained learned candidates are the same constant
    # policy at the matched init, so their rows agree to float32
    # forward-pass rounding (the architectures round differently in
    # float32 even with a zero final layer); the count fields, which
    # do not pass through the networks, agree exactly.
    learned = [rows[n] for n in CANDIDATE_ORDER[:3]]
    for key in ("mean_terminal_wealth", "mean_fees",
                "mean_max_drawdown"):
        assert learned[0][key] == pytest.approx(learned[1][key],
                                                abs=1e-5), key
        assert learned[0][key] == pytest.approx(learned[2][key],
                                                abs=1e-5), key
    assert (learned[0]["n_negative_terminal"]
            == learned[1]["n_negative_terminal"]
            == learned[2]["n_negative_terminal"])
    assert (learned[0]["breach_count_anytime"]
            == learned[1]["breach_count_anytime"]
            == learned[2]["breach_count_anytime"] == 0)
    assert learned[0]["mean_fraction"] == pytest.approx(SIGMOID2,
                                                        abs=1e-6)


# --- hierarchical selection -----------------------------------------------------


def _row(name, wealth, shortfall, breach_anytime, violations=0,
         fraction_min=0.0, fraction_max=1.0):
    return {
        "candidate": name,
        "mean_terminal_wealth": wealth,
        "mean_terminal_shortfall": shortfall,
        "breach_count_anytime": breach_anytime,
        "self_financing_violations": violations,
        "fraction_min": fraction_min,
        "fraction_max": fraction_max,
    }


def test_stage3_neural_cppi_5_part_1_hierarchical_selection_oracle():
    reference = _row("classic", 99.0, 0.5, 3)
    rows = [
        _row("baseline_mlp", 100.0, 0.1, 2),
        _row("linear_logistic", 105.0, 0.0, 9),
        _row("residual_utilization", 110.0, 0.0, 0, violations=1),
        _row("dynamic_rule", 101.0, 0.2, 3),
    ]
    out = hierarchical_select(rows, reference)
    # Tier 1 removes the accounting violator despite the best raw
    # wealth; tier 2 removes the frequent breacher despite the best
    # surviving wealth; tier 3 picks utility 100 - 30*0.1 = 97 over
    # 101 - 30*0.2 = 95.
    assert out["selected"] == "baseline_mlp"
    assert set(out["discarded"]) == {"linear_logistic",
                                     "residual_utilization"}
    assert out["survivors"] == ["baseline_mlp", "dynamic_rule"]
    assert out["utility"]["baseline_mlp"] == pytest.approx(97.0)
    assert out["utility"]["dynamic_rule"] == pytest.approx(95.0)
    # A fraction outside [0, 1] is a tier-1 discard.
    rows2 = [_row("baseline_mlp", 100.0, 0.1, 0, fraction_max=1.2),
             _row("dynamic_rule", 99.0, 0.4, 1)]
    out2 = hierarchical_select(rows2, reference)
    assert out2["selected"] == "dynamic_rule"
    assert "baseline_mlp" in out2["discarded"]
    # Ties break toward the earlier-declared row.
    rows3 = [_row("baseline_mlp", 100.0, 0.1, 0),
             _row("dynamic_rule", 100.0, 0.1, 0)]
    assert hierarchical_select(rows3, reference)["selected"] == \
        "baseline_mlp"


def test_stage3_neural_cppi_5_part_1_selection_fail_closed():
    reference = _row("classic", 99.0, 0.5, 0)
    with pytest.raises(ValueError):
        hierarchical_select([_row("baseline_mlp", 1.0, 0.0, 0)], None)
    out = hierarchical_select(
        [_row("baseline_mlp", 100.0, 0.0, 5),
         _row("dynamic_rule", 100.0, 0.0, 2)], reference)
    assert out["selected"] is None
    assert out["survivors"] == []


# --- training-log schema ---------------------------------------------------------


def test_stage3_neural_cppi_5_part_1_gradient_null_fraction_oracle():
    grads = [torch.tensor([0.0, 1e-13, 0.5, -2.0]), None,
             torch.tensor([0.0])]
    assert gradient_null_fraction(grads) == pytest.approx(0.6)
    assert gradient_null_fraction([torch.tensor([1.0])]) == 0.0
    with pytest.raises(ValueError):
        gradient_null_fraction([None])


def _valid_log():
    epoch = {
        "epoch": 0, "loss": -95.0,
        "neg_expected_terminal_wealth": -100.0,
        "terminal_shortfall": 0.1, "anytime_shortfall": 0.2,
        "turnover": 0.05, "gradient_null_fraction": 0.25,
        "cap_activation_share": 0.0, "budget_zero_share": 0.01,
    }
    return {
        "candidate": "baseline_mlp", "seed": 7,
        "policy_init_hash": "0123456789abcdef",
        "loss": "multihorizon_loss", "epochs": [epoch],
    }


def test_stage3_neural_cppi_5_part_1_training_log_schema():
    assert validate_training_log(_valid_log()) is True
    bad = _valid_log()
    del bad["seed"]
    with pytest.raises(ValueError):
        validate_training_log(bad)
    bad = _valid_log()
    del bad["epochs"][0]["gradient_null_fraction"]
    with pytest.raises(ValueError):
        validate_training_log(bad)
    bad = _valid_log()
    bad["epochs"][0]["cap_activation_share"] = 1.5
    with pytest.raises(ValueError):
        validate_training_log(bad)
    bad = _valid_log()
    bad["epochs"] = []
    with pytest.raises(ValueError):
        validate_training_log(bad)
