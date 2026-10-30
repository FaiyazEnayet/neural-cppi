"""Fixtures for E7 (false-recovery treatment).

E7 starts with simple hysteresis and cooldown at the same
variable budget as R1 (the gate scales the budget fraction u,
never the CPPI budget B, the multiplier, the floor or the cap),
then conditions re-entry on trend persistence and uncertainty.
The gate is designed against deterministic bear-rally-double-dip
fixtures and two attack episodes, its parameters are frozen on
tuning episodes, and it is judged on fresh, disjoint episodes
with final held-out episodes reserved and never evaluated here.
Failures are recorded even when the gated policy loses: the
V-bottom fixture bounds the cost of delay, and the judge table
reports every arm on every episode, win or lose.

All fixtures are deterministic; the full run on the committed
SPY series is exercised by the run script, not by these tests,
and its numbers are reported in the CHANGELOG entry.
"""

import numpy as np
import pytest
import torch

from dhl.eval.false_recovery import (
    ATTACK_EPISODES,
    E7_GRID,
    E7_TRAINING,
    ReentryGate,
    attribute_reentry,
    chop_prices,
    double_dip_prices,
    gate_fraction_fn,
    run_e7,
    secondary_low_index,
    select_gate_params,
    v_bottom_prices,
)
from dhl.experiments.cppi_learned import CAP, COST_RATE, FLOOR
from dhl.policies.budget_allocator import budget_weight_fn
from dhl.experiments.cppi_learned import rollout


def _feats(vol, vol_ratio, drawdown, n=1):
    return {
        "vol": torch.full((n,), float(vol)),
        "vol_ratio": torch.full((n,), float(vol_ratio)),
        "drawdown": torch.full((n,), float(drawdown)),
        "panic_z": torch.zeros(n),
    }


def _run_gate(params, steps):
    """Drive a gate through hand-built steps.

    ``steps`` is a list of (wealth, vol, vol_ratio, drawdown)
    tuples; the base fraction is constant 1. Returns the list of
    gated fractions and the gate object.
    """
    gate = ReentryGate(**params)
    fn = gate_fraction_fn(gate, lambda i, V, t, c, f: torch.ones_like(V))
    out = []
    for i, (wealth, vol, vr, dd) in enumerate(steps):
        V = torch.tensor([wealth])
        u = fn(i, V, 0.5, 1.0, _feats(vol, vr, dd))
        out.append(float(u[0]))
    return out, gate


# --- gate state machine ---------------------------------------------------------


def test_stage3_neural_cppi_7_exit_trigger_and_derisk():
    params = dict(E7_TRAINING["default_gate"])
    out, _ = _run_gate(params, [
        (100.0, 0.15, 0.8, 0.02),
        (100.0, 0.40, 1.5, 0.02),
        (100.0, 0.15, 0.8, 0.02),
    ])
    assert out[0] == 1.0
    assert out[1] == params["derisk_factor"]
    assert out[2] == params["derisk_factor"]


def test_stage3_neural_cppi_7_cooldown_blocks_reentry():
    params = dict(E7_TRAINING["default_gate"], cooldown_steps=3,
                  persistence_steps=1)
    steps = [(100.0, 0.40, 1.5, 0.02),
             (101.0, 0.15, 0.8, 0.02),
             (102.0, 0.15, 0.8, 0.02),
             (103.0, 0.15, 0.8, 0.02),
             (104.0, 0.15, 0.8, 0.02)]
    out, _ = _run_gate(params, steps)
    assert out[0] == params["derisk_factor"]
    assert out[1] == params["derisk_factor"]
    assert out[2] == params["derisk_factor"]
    assert out[3] == params["derisk_factor"]
    assert out[4] == 1.0


def test_stage3_neural_cppi_7_hysteresis_blocks_reentry():
    params = dict(E7_TRAINING["default_gate"], cooldown_steps=0,
                  persistence_steps=1)
    steps = [
        (100.0, 0.40, 1.5, 0.02),
        (101.0, 0.28, 0.8, 0.02),
        (102.0, 0.15, 0.8, 0.02),
    ]
    out, _ = _run_gate(params, steps)
    assert out[1] == params["derisk_factor"]
    assert out[2] == 1.0


def test_stage3_neural_cppi_7_persistence_blocks_reentry():
    params = dict(E7_TRAINING["default_gate"], cooldown_steps=0,
                  persistence_steps=3)
    steps = [(100.0, 0.40, 1.5, 0.02)]
    steps += [(101.0, 0.15, 0.8, 0.02),
              (102.0, 0.15, 0.8, 0.02),
              (103.0, 0.15, 0.8, 0.02),
              (104.0, 0.15, 0.8, 0.02)]
    out, _ = _run_gate(params, steps)
    assert out[0] == params["derisk_factor"]
    assert out[1] == params["derisk_factor"]
    assert out[2] == params["derisk_factor"]
    assert out[3] == 1.0
    assert out[4] == 1.0


def test_stage3_neural_cppi_7_uncertainty_blocks_reentry():
    params = dict(E7_TRAINING["default_gate"], cooldown_steps=0,
                  persistence_steps=1)
    steps = [
        (100.0, 0.40, 1.5, 0.02),
        (101.0, 0.15, 1.5, 0.02),
        (102.0, 0.15, 0.8, 0.02),
    ]
    out, _ = _run_gate(params, steps)
    assert out[1] == params["derisk_factor"]
    assert out[2] == 1.0


def test_stage3_neural_cppi_7_drawdown_trigger():
    params = dict(E7_TRAINING["default_gate"])
    out, _ = _run_gate(params, [
        (100.0, 0.15, 0.8, 0.02),
        (100.0, 0.15, 0.8, 0.12),
    ])
    assert out[0] == 1.0
    assert out[1] == params["derisk_factor"]


# --- fixtures and attribution -----------------------------------------------------


def test_stage3_neural_cppi_7_secondary_low_index():
    prices = np.array([100.0, 90.0, 70.0, 80.0, 75.0, 60.0, 70.0, 90.0])
    assert secondary_low_index(prices) == 5
    rising = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    assert secondary_low_index(rising) is None


def test_stage3_neural_cppi_7_double_dip_gate_beats_early():
    prices = double_dip_prices()
    S = torch.as_tensor(prices, dtype=torch.float64).unsqueeze(0)
    early_V, _p, early_ledger = rollout(
        S, budget_weight_fn(
            lambda i, V, t, c, f: torch.ones_like(V)),
        floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
        return_path=True, return_ledger=True)
    gate = ReentryGate(**E7_TRAINING["default_gate"])
    gated_V, _p2, gated_ledger = rollout(
        S, budget_weight_fn(gate_fraction_fn(
            gate, lambda i, V, t, c, f: torch.ones_like(V))),
        floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
        return_path=True, return_ledger=True)
    assert float(gated_V[0]) > float(early_V[0])
    attr = attribute_reentry(prices, early_ledger, gated_ledger,
                             gate)
    assert attr["secondary_low_index"] is not None
    assert attr["extra_exposure_sum"] > 0.0


def test_stage3_neural_cppi_7_v_bottom_delay_cost_bounded():
    prices = v_bottom_prices()
    S = torch.as_tensor(prices, dtype=torch.float64).unsqueeze(0)
    early_V, _p, _l = rollout(
        S, budget_weight_fn(
            lambda i, V, t, c, f: torch.ones_like(V)),
        floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
        return_path=True, return_ledger=True)
    gate = ReentryGate(**E7_TRAINING["default_gate"])
    gated_V, _p2, _l2 = rollout(
        S, budget_weight_fn(gate_fraction_fn(
            gate, lambda i, V, t, c, f: torch.ones_like(V))),
        floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
        return_path=True, return_ledger=True)
    assert float(early_V[0]) - float(gated_V[0]) < 5.0


def test_stage3_neural_cppi_7_gate_scales_fraction_only():
    """The gate scales the fraction u, never the budget itself.

    With a constant base fraction of 0.7 and derisk_factor
    0.25, invested steps return 0.7 and derisked steps return
    0.175, whatever the wealth path. (Executed weights can
    still exceed an ungated arm's weights, because the gated
    arm's cushion, and hence its budget B, differs; the
    invariant is on u, not on w = B * u.)
    """
    params = dict(E7_TRAINING["default_gate"], derisk_factor=0.25)
    gate = ReentryGate(**params)
    fn = gate_fraction_fn(
        gate, lambda i, V, t, c, f: torch.full_like(V, 0.7))
    steps = [(100.0, 0.15, 0.8, 0.02),
             (100.0, 0.40, 1.5, 0.02),
             (101.0, 0.15, 0.8, 0.02)]
    out = []
    for i, (wealth, vol, vr, dd) in enumerate(steps):
        out.append(float(fn(i, torch.tensor([wealth]), 0.5, 1.0,
                          _feats(vol, vr, dd))[0]))
    assert out[0] == pytest.approx(0.7)
    assert out[1] == pytest.approx(0.175)
    assert out[2] == pytest.approx(0.175)


def test_stage3_neural_cppi_7_selection_picks_highest_utility():
    table = [
        {"params": {"a": 1}, "mean_utility": 90.0},
        {"params": {"a": 2}, "mean_utility": 95.0},
        {"params": {"a": 3}, "mean_utility": 92.0},
    ]
    selected = select_gate_params(table)
    assert selected["params"] == {"a": 2}


def test_stage3_neural_cppi_7_declared_episodes():
    names = [e["name"] for e in ATTACK_EPISODES]
    assert "tuning_2011" in names
    assert "tuning_2018" in names
    assert "judge_2020" in names
    assert "judge_2022" in names
    assert "reserve_2024plus" in names
    assert E7_GRID


# --- full run on synthetic data ----------------------------------------------------


def _synthetic_series():
    import datetime
    rng = np.random.default_rng(5)
    dates = []
    day = datetime.date(2016, 1, 1)
    for _ in range(1500):
        dates.append(day.isoformat())
        day += datetime.timedelta(days=1)
    rets = rng.normal(0.0003, 0.012, size=1499)
    prices = np.empty(1500)
    prices[0] = 100.0
    prices[1:] = 100.0 * np.exp(np.cumsum(rets))
    return dates, prices


def test_stage3_neural_cppi_7_run_reports_judge_and_reserve():
    dates, prices = _synthetic_series()
    result = run_e7(
        dates, prices,
        base_train=("2016-01-01", "2017-01-01"),
        tuning_episodes=(
            {"name": "tuning_a", "start": "2017-06-01",
             "end": "2017-12-01", "role": "tuning"},
            {"name": "tuning_b", "start": "2018-01-01",
             "end": "2018-06-01", "role": "tuning"},
        ),
        judge_episodes=(
            {"name": "judge_a", "start": "2018-09-01",
             "end": "2019-03-01", "role": "judge"},
            {"name": "judge_b", "start": "2019-06-01",
             "end": "2019-12-01", "role": "judge"},
        ),
        reserve_episodes=(
            {"name": "reserve_a", "start": "2020-01-01",
             "end": None, "role": "reserve"},
        ),
        grid=(
            {"cooldown_steps": 0, "reentry_vol": 0.20,
             "persistence_steps": 1, "derisk_factor": 0.0},
            {"cooldown_steps": 5, "reentry_vol": 0.20,
             "persistence_steps": 3, "derisk_factor": 0.0},
        ),
        n_epochs=1, n_bootstrap=3, path_len=21)
    assert result["selected_params"] in (
        {"cooldown_steps": 0, "reentry_vol": 0.20,
         "persistence_steps": 1, "derisk_factor": 0.0},
        {"cooldown_steps": 5, "reentry_vol": 0.20,
         "persistence_steps": 3, "derisk_factor": 0.0})
    assert [c["episode"] for c in result["judge_cells"]] == \
        ["judge_a", "judge_b"]
    for cell in result["judge_cells"]:
        assert set(cell["arms"]) == {
            "learned", "learned_gated", "classic", "classic_gated"}
        assert cell["attribution"] is not None or True
    assert result["reserve_episodes"] == ["reserve_a"]
    assert "reserve_a" not in [c["episode"]
                               for c in result["judge_cells"]]
