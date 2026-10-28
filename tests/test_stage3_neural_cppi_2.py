"""Stage 3 E2: correct-loss factorial.

The contract varies three controlled factors while holding the training
pool, epochs, batch size, initialization protocol and seed list fixed:

- anytime term: none vs the P0-A corrected term (adds pathwise floor
  shortfall to the minimization loss);
- feature set: existing (vol only) vs causal rebound (the full strictly
  causal crash feature set);
- warmup: off (cold) vs on (warm, only with a shared warmup baseline of
  actual pre-window observations).

These tests pin the factorial runner in ``src/dhl/eval/factorial.py``:
each factor must actually change behavior, the warmup baseline must be
shared, and the selection rule must fail closed on both contract
falsifiers (no pathwise-shortfall improvement, or a terminal-wealth
sacrifice above the stated acceptable tradeoff). Fixtures are small
synthetic price paths; no full model-training run and no published
number is regenerated here.
"""

import numpy as np
import pytest
import torch

from dhl.eval.factorial import (
    FEATURE_SETS,
    FactorialCell,
    anytime_penalty,
    average_by_cell,
    evaluate_cell,
    loss_and_gradient,
    run_factorial,
    select_cell,
    shared_warmup_baseline,
    summarize_path,
)
from dhl.experiments.cppi_learned import (
    CPPIPolicy,
    FLOOR,
    FLOOR_PENALTY,
    WARM_LOOKBACK,
    cppi_constrained_weight_fn,
    describe_protocol,
    rollout,
    seeded_policy,
)

SEEDS = (0, 1, 2)


def _synthetic_paths():
    """Six short synthetic paths (6 paths, 11 prices, 10 steps)."""
    rows = [
        [100.0] * 11,
        [100.0 * (1.01 ** i) for i in range(11)],
        [100, 97, 94, 90, 85, 82, 86, 90, 94, 97, 98],
        [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80],
        [100, 99, 97, 88, 92, 96, 99, 101, 100, 102, 103],
        [100, 99, 101, 98, 102, 100, 103, 101, 104, 102, 105],
    ]
    return torch.tensor(rows, dtype=torch.float32)


def _warm_series():
    """120 deterministic pre-window prices with varying returns."""
    rng = np.random.default_rng(123)
    rets = rng.normal(0.0, 0.01, size=120)
    return 100.0 * np.exp(np.cumsum(rets))


def _policy_fn(feature_level, seed):
    policy = seeded_policy(
        CPPIPolicy, feature_names=FEATURE_SETS[feature_level], seed=seed
    )

    def fn(i, V, tau_frac, cushion, feats):
        t = torch.full_like(V, tau_frac)
        return policy(t, cushion, feats)

    return policy, fn


def _layered_fn(feature_level, seed):
    _, fn = _policy_fn(feature_level, seed)
    return cppi_constrained_weight_fn(fn, multiplier=4.0)


def test_stage3_neural_cppi_2_anytime_penalty_raises_loss_and_moves_gradient():
    S = _synthetic_paths()
    policy, _ = _policy_fn("existing", 0)
    none = loss_and_gradient(policy, S, anytime_level="none")
    policy2, _ = _policy_fn("existing", 0)
    correct = loss_and_gradient(policy2, S, anytime_level="correct")
    assert anytime_penalty("none") == 0.0
    assert anytime_penalty("correct") == FLOOR_PENALTY
    assert correct["total"] > none["total"] + 1e-6
    assert correct["components"]["anytime_shortfall"] > 0.0
    assert np.isfinite(none["grad_norm"]) and np.isfinite(correct["grad_norm"])
    assert correct["grad_norm"] != pytest.approx(none["grad_norm"], abs=1e-9) \
        or correct["total"] != pytest.approx(none["total"], abs=1e-9)


def test_stage3_neural_cppi_2_feature_set_changes_weights():
    S = _synthetic_paths()
    _, fn_existing = _policy_fn("existing", 0)
    _, fn_rebound = _policy_fn("causal_rebound", 0)
    with torch.no_grad():
        v_e, _, led_e = rollout(
            S, fn_existing, return_path=True, return_ledger=True
        )
        v_r, _, led_r = rollout(
            S, fn_rebound, return_path=True, return_ledger=True
        )
    assert FEATURE_SETS["existing"] == ("vol",)
    assert set(FEATURE_SETS["causal_rebound"]) == {
        "vol", "vol_ratio", "drawdown", "panic_z",
    }
    assert v_e.shape == v_r.shape
    assert float((led_e["weight"] - led_r["weight"]).abs().max()) > 1e-4


def test_stage3_neural_cppi_2_warmup_changes_first_trade_weights():
    S = _synthetic_paths()
    history = shared_warmup_baseline(_warm_series(), WARM_LOOKBACK)
    assert history.shape == (WARM_LOOKBACK,)
    _, fn = _policy_fn("causal_rebound", 0)
    with torch.no_grad():
        _, _, led_cold = rollout(
            S, fn, return_path=True, return_ledger=True, history=None
        )
        _, fn2 = _policy_fn("causal_rebound", 0)
        _, _, led_warm = rollout(
            S, fn2, return_path=True, return_ledger=True, history=history
        )
    assert describe_protocol(None) == "cold"
    assert describe_protocol(history) == "warm"
    first_cold = led_cold["weight"][:, 0]
    first_warm = led_warm["weight"][:, 0]
    assert float((first_cold - first_warm).abs().max()) > 1e-4


def test_stage3_neural_cppi_2_factorial_grid_and_shared_baseline():
    S = _synthetic_paths()
    history = shared_warmup_baseline(_warm_series(), WARM_LOOKBACK)

    def factory(cell):
        _, fn = _policy_fn(cell.features, cell.seed)
        return fn

    def layered_factory(cell):
        return _layered_fn(cell.features, cell.seed)

    results = run_factorial(
        factory, S, seeds=SEEDS, warm_history=history,
        layered_fn_factory=layered_factory,
    )
    assert len(results) == 8 * len(SEEDS)
    cells = {(r["anytime"], r["features"], r["warmup"]) for r in results}
    assert len(cells) == 8
    for r in results:
        assert np.isfinite(r["mean_terminal"])
        assert np.isfinite(r["mean_anytime_shortfall"])
        assert np.isfinite(r["mean_max_drawdown"])
        assert np.isfinite(r["layered_mean_terminal"])
        assert np.isfinite(r["layered_mean_anytime_shortfall"])
        assert 0.0 <= r["mean_cash_fraction"] <= 1.0
        if r["warmup"] == "on":
            assert r["protocol"] == "warm"
            assert r["history_days"] == WARM_LOOKBACK
        else:
            assert r["protocol"] == "cold"
            assert r["history_days"] == 0
    by_cell = average_by_cell(results)
    assert len(by_cell) == 8
    # The anytime factor changes only the loss, not the rollout weights,
    # so same features/seed/warmup cells agree on terminal wealth exactly.
    key_none = ("none", "existing", "off")
    key_correct = ("correct", "existing", "off")
    assert by_cell[key_none]["mean_terminal"] == pytest.approx(
        by_cell[key_correct]["mean_terminal"]
    )


def test_stage3_neural_cppi_2_selection_fail_closed():
    def row(anytime, features, warmup, seed, shortfall, terminal):
        return {
            "anytime": anytime, "features": features, "warmup": warmup,
            "seed": seed, "mean_anytime_shortfall": shortfall,
            "mean_terminal": terminal,
        }

    improving = [
        row("none", "existing", "off", s, 5.0, 100.0) for s in SEEDS
    ] + [
        row("correct", "existing", "off", s, 3.0, 99.5) for s in SEEDS
    ]
    best, reason = select_cell(improving, max_terminal_sacrifice=2.0)
    assert best == ("correct", "existing", "off")
    assert "improves" in reason

    no_gain = [
        row("none", "existing", "off", s, 3.0, 100.0) for s in SEEDS
    ] + [
        row("correct", "existing", "off", s, 3.5, 100.0) for s in SEEDS
    ]
    best, reason = select_cell(no_gain, max_terminal_sacrifice=2.0)
    assert best is None
    assert "does not improve" in reason

    too_costly = [
        row("none", "existing", "off", s, 5.0, 100.0) for s in SEEDS
    ] + [
        row("correct", "existing", "off", s, 3.0, 94.0) for s in SEEDS
    ]
    best, reason = select_cell(too_costly, max_terminal_sacrifice=2.0)
    assert best is None
    assert "tradeoff" in reason


def test_stage3_neural_cppi_2_fail_closed_validation():
    S = _synthetic_paths()

    def factory(cell):
        _, fn = _policy_fn(cell.features, cell.seed)
        return fn

    with pytest.raises(ValueError, match="warmup"):
        run_factorial(factory, S, seeds=SEEDS, warm_history=None)
    with pytest.raises(ValueError, match="tuning source"):
        run_factorial(
            factory, S, seeds=SEEDS,
            warm_history=shared_warmup_baseline(_warm_series(), WARM_LOOKBACK),
            tuning_source="sealed",
        )
    with pytest.raises(ValueError, match="seeds"):
        run_factorial(factory, S, seeds=(0, 1))
    with pytest.raises(ValueError):
        FactorialCell(anytime="wrong", features="existing", warmup="off", seed=0)
    with pytest.raises(ValueError, match="pre-window"):
        shared_warmup_baseline(np.array([100.0, 101.0]), WARM_LOOKBACK)


def test_stage3_neural_cppi_2_summarize_path_oracle():
    vt = np.array([95.0, 105.0])
    path = np.array([[100.0, 95.0, 95.0], [100.0, 110.0, 105.0]])
    out = summarize_path(vt, path, weights=np.array([[0.5, 0.5], [1.0, 1.0]]),
                         floor=90.0)
    assert out["mean_terminal"] == pytest.approx(100.0)
    assert out["mean_min_wealth"] == pytest.approx(97.5)
    assert out["p_breach_terminal"] == pytest.approx(0.0)
    assert out["p_breach_anytime"] == pytest.approx(0.0)
    assert out["mean_terminal_shortfall"] == pytest.approx(0.0)
    assert out["mean_anytime_shortfall"] == pytest.approx(0.0)
    assert out["mean_max_drawdown"] == pytest.approx(0.0477272727)
    assert out["mean_cash_fraction"] == pytest.approx(0.25)

    path_dip = np.array([[100.0, 85.0, 95.0], [100.0, 110.0, 105.0]])
    out_dip = summarize_path(vt, path_dip, floor=90.0)
    assert out_dip["mean_anytime_shortfall"] == pytest.approx(2.5)
    assert out_dip["p_breach_anytime"] == pytest.approx(0.5)
    assert out_dip["mean_terminal_shortfall"] == pytest.approx(0.0)


def test_stage3_neural_cppi_2_evaluate_cell_protocols():
    S = _synthetic_paths()
    history = shared_warmup_baseline(_warm_series(), WARM_LOOKBACK)
    _, fn = _policy_fn("existing", 0)
    cold = evaluate_cell(fn, S, history=None)
    _, fn2 = _policy_fn("existing", 0)
    warm = evaluate_cell(fn2, S, history=history)
    assert cold["protocol"] == "cold" and cold["history_days"] == 0
    assert warm["protocol"] == "warm" and warm["history_days"] == WARM_LOOKBACK
    assert cold["mean_terminal"] != pytest.approx(warm["mean_terminal"])
