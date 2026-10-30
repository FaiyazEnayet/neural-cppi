"""Part 2 fixtures for E5 (budget-action candidate comparison).

Part 1 fixed the contract, the candidates, the comparison harness,
the hierarchical selection rule and the training-log schema, and
captured the untrained baseline. Part 2 trains the three learned
candidates on the E5 training family with the corrected
multi-horizon path loss, selects on the validation family only
under ``hierarchical_select``, and evaluates the frozen selection
on the held-out stress and gap families and on SPY block-bootstrap
episodes (declared seeds 31000-31019, generated here for the first
time). No holdout family influences any training or selection
decision, and no performance promotion is made beyond the reported
tables: a candidate that loses on a family is reported as losing.

All fixtures use small synthetic families and a small episode
subset so the suite stays fast; the full run (20 scenarios per
family, 50 epochs) is exercised by the run script, not by these
tests, and its numbers are reported in the CHANGELOG entry.
"""

import math

import pytest
import torch

from dhl.experiments.cppi_learned import FLOOR, FLOOR_PENALTY
from dhl.policies.budget_allocator import budget_weight_fn
from dhl.policies.candidate_comparison import (
    CANDIDATE_ORDER,
    E5_SEED_SETS,
    candidate_fraction_fn,
    candidate_model,
    e5_scenarios,
    evaluate_candidates,
    hierarchical_select,
    reference_fraction_fn,
    validate_training_log,
)
from dhl.policies.candidate_training import (
    PART2_TRAINING,
    episode_scenarios,
    fraction_fn_for_model,
    run_part2,
    train_candidate,
)


def _tiny_family(name, n_scenarios=2):
    scenarios = e5_scenarios(name, n_paths=4, n_steps=10)
    return scenarios[:n_scenarios]


def _trained_fns(n_epochs=2):
    train = _tiny_family("training")
    fns = {}
    for name in CANDIDATE_ORDER:
        model, _log = train_candidate(name, train, n_epochs=n_epochs)
        if model is not None:
            fns[name] = fraction_fn_for_model(model)
        else:
            fns[name] = candidate_fraction_fn(name)
    return fns


# --- training ----------------------------------------------------------------


def test_stage3_neural_cppi_5_part_2_training_log_schema():
    train = _tiny_family("training")
    model, log = train_candidate("baseline_mlp", train, n_epochs=2)
    assert model is not None
    assert validate_training_log(log) is True
    assert log["candidate"] == "baseline_mlp"
    assert log["loss"] == "multihorizon_loss"
    assert log["epochs"][0]["epoch"] == 0
    assert log["epochs"][-1]["epoch"] == 2
    for entry in log["epochs"]:
        assert 0.0 <= entry["gradient_null_fraction"] <= 1.0
        assert 0.0 <= entry["cap_activation_share"] <= 1.0
        assert 0.0 <= entry["budget_zero_share"] <= 1.0


def test_stage3_neural_cppi_5_part_2_training_changes_model():
    train = _tiny_family("training")
    before = candidate_model("linear_logistic")
    before_params = [p.detach().clone() for p in before.parameters()]
    model, _log = train_candidate("linear_logistic", train,
                                  n_epochs=2)
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.equal(a, b)
               for a, b in zip(before_params, after))


def test_stage3_neural_cppi_5_part_2_dynamic_rule_not_trained():
    train = _tiny_family("training")
    model, log = train_candidate("dynamic_rule", train, n_epochs=2)
    assert model is None
    assert log["candidate"] == "dynamic_rule"
    assert log["trained"] is False


def test_stage3_neural_cppi_5_part_2_loss_decomposition_oracle():
    """The logged loss equals the hand recomputation from its parts.

    total = -E[V_T] + floor_penalty * E[terminal shortfall]
            + anytime_penalty * E[anytime shortfall]
            + turnover_penalty * E[turnover], with the Part 2
    predeclared penalties (floor 30, anytime 30, turnover 0).
    """
    train = _tiny_family("training")
    _model, log = train_candidate("baseline_mlp", train, n_epochs=1)
    for entry in log["epochs"]:
        recomputed = (
            entry["neg_expected_terminal_wealth"]
            + PART2_TRAINING["floor_penalty"]
            * entry["terminal_shortfall"]
            + PART2_TRAINING["anytime_penalty"]
            * entry["anytime_shortfall"]
            + PART2_TRAINING["turnover_penalty"]
            * entry["turnover"]
        )
        assert entry["loss"] == pytest.approx(recomputed, rel=1e-9)


def test_stage3_neural_cppi_5_part_2_unknown_candidate_refused():
    with pytest.raises(ValueError):
        train_candidate("not_a_candidate", _tiny_family("training"))


def test_stage3_neural_cppi_5_part_2_empty_training_refused():
    with pytest.raises(ValueError):
        train_candidate("baseline_mlp", [])


# --- episodes ------------------------------------------------------------------


def test_stage3_neural_cppi_5_part_2_episodes_deterministic():
    first = episode_scenarios(n_paths=4, n_steps=10, seeds=(31000, 31001))
    second = episode_scenarios(n_paths=4, n_steps=10, seeds=(31000, 31001))
    assert [s for s, _ in first] == [31000, 31001]
    for (_, a), (_, b) in zip(first, second):
        assert torch.equal(a, b)
    assert first[0][1].shape == (4, 11)
    assert float(first[0][1][:, 0].min()) == 100.0


def test_stage3_neural_cppi_5_part_2_episodes_differ_from_synthetic():
    episodes = episode_scenarios(n_paths=4, n_steps=10,
                                 seeds=(31000,))
    synthetic = e5_scenarios("training", n_paths=4, n_steps=10)[:1]
    assert not torch.equal(episodes[0][1], synthetic[0][1])


def test_stage3_neural_cppi_5_part_2_episode_seed_set_declared():
    assert E5_SEED_SETS["episodes"] == tuple(31000 + k for k in range(20))


# --- selection and the full Part 2 run ----------------------------------------


def test_stage3_neural_cppi_5_part_2_selection_uses_validation_only():
    """run_part2 selects exactly as hierarchical_select on validation.

    The selection is recomputed independently from the returned
    validation rows; the stress, gap and episode rows are never
    passed to the selector, so they cannot change the winner.
    """
    result = run_part2(
        n_epochs=1,
        families={
            "training": _tiny_family("training"),
            "validation": _tiny_family("validation"),
            "stress": _tiny_family("stress"),
            "gap": _tiny_family("gap"),
            "episodes": episode_scenarios(n_paths=4, n_steps=10,
                                          seeds=(31000,)),
        },
    )
    validation = result["families"]["validation"]
    rows = [r for r in validation["candidates"]
            if r["candidate"] in CANDIDATE_ORDER]
    classic = next(r for r in validation["candidates"]
                   if r["candidate"] == "classic")
    expected = hierarchical_select(rows, classic)
    assert result["selection"] == expected
    assert result["selected"] == expected["selected"]
    for family in ("stress", "gap", "episodes"):
        assert family in result["families"]


def test_stage3_neural_cppi_5_part_2_report_carries_acceptance_fields():
    result = run_part2(
        n_epochs=1,
        families={
            "training": _tiny_family("training"),
            "validation": _tiny_family("validation"),
            "stress": _tiny_family("stress"),
            "gap": _tiny_family("gap"),
            "episodes": episode_scenarios(n_paths=4, n_steps=10,
                                          seeds=(31000,)),
        },
    )
    required = (
        "candidate", "n_paths", "mean_terminal_wealth",
        "mean_terminal_shortfall", "breach_count_terminal",
        "breach_count_anytime", "mean_anytime_shortfall",
        "mean_max_drawdown", "mean_fees", "n_negative_terminal",
        "min_terminal_wealth", "mean_fraction",
        "self_financing_violations",
    )
    for family in result["families"].values():
        assert family["seeds"]
        for row in family["candidates"]:
            for field in required:
                assert field in row, (family, row, field)


def test_stage3_neural_cppi_5_part_2_no_holdout_in_training_log():
    """Training logs record only the training family seed set."""
    result = run_part2(
        n_epochs=1,
        families={
            "training": _tiny_family("training"),
            "validation": _tiny_family("validation"),
            "stress": _tiny_family("stress"),
            "gap": _tiny_family("gap"),
            "episodes": episode_scenarios(n_paths=4, n_steps=10,
                                          seeds=(31000,)),
        },
    )
    for name, log in result["training_logs"].items():
        if log.get("trained"):
            assert log["training_seeds"] == [
                int(s) for s, _ in _tiny_family("training")]
            assert 31000 not in log["training_seeds"]
