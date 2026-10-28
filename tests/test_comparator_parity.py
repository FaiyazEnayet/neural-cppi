"""Stage 3 P1-B: benchmark fairness and attribution.

The comparator ladder runs every strategy through the one execution
kernel (``dhl.experiments.cppi_learned.rollout``) with matched
conventions, so a learned policy can only beat a classic rule by
adding signal, never by facing a weaker rule, a different cap, or
different costs.

Independent oracles: buy/hold and all-cash terminal wealth are
recomputed by hand from the price ratio and the cost rate, never
through the function under test. Parity checks compare two weight
functions on identical states, not two runs of the same code path.
All paths are tiny synthetic series; no training run lives here.
"""

import numpy as np
import pytest
import torch

from dhl.eval.comparators import (
    LADDER_KINDS,
    ComparatorConfig,
    Conventions,
    make_weight_fn,
    run_ladder,
    select_multiplier,
    validate_selection,
)
from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    DT,
    FLOOR,
    R,
    V0,
    _classic_weight_fn,
    classic_cppi_weight,
    cppi_constrained_weight_fn,
    rollout,
)
from dhl.eval.registry import Episode, EpisodeRegistry

CONV = Conventions()


def _toy_prices(n_paths=8, steps=30, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, size=(n_paths, steps))
    S = np.empty((n_paths, steps + 1))
    S[:, 0] = V0
    S[:, 1:] = V0 * np.exp(np.cumsum(rets, axis=1))
    return torch.as_tensor(S, dtype=torch.float32)


def _feats(vol=0.20, drawdown=0.0, vol_ratio=1.0, n=4):
    return {
        "vol": torch.full((n,), vol),
        "vol_ratio": torch.full((n,), vol_ratio),
        "drawdown": torch.full((n,), drawdown),
        "panic_z": torch.zeros(n),
    }


def _cfg(kind, **params):
    return ComparatorConfig(name=kind, kind=kind, params=params)


def test_stage3_neural_cppi_1_comparator_factory_common_conventions():
    """One factory supplies every rung with the same conventions."""
    class StubPolicy(torch.nn.Module):
        def forward(self, t, cushion, feats):
            return torch.full_like(cushion, 0.7)

    stub = StubPolicy()
    for kind in LADDER_KINDS:
        cfg = _cfg(kind, multiplier=4.0)
        fn = make_weight_fn(cfg, CONV, policy=stub)
        V = torch.full((4,), 100.0)
        w = fn(0, V, 1.0, torch.full((4,), 0.1), _feats())
        assert w.shape == (4,)
        assert bool(((w >= 0.0) & (w <= CAP)).all())
    assert "classic_m4" in LADDER_KINDS
    assert "classic_m4_brake" in LADDER_KINDS
    assert "learned_constrained_brake" in LADDER_KINDS


def test_stage3_neural_cppi_1_comparator_budget_parity():
    """100% budget utilization reproduces classic CPPI exactly."""
    V = torch.full((4,), 100.0)
    cushion = torch.full((4,), 0.1)
    feats = _feats()
    w_budget = make_weight_fn(
        _cfg("budget_fraction", multiplier=4.0, fraction=1.0), CONV)(
        0, V, 1.0, cushion, feats)
    w_classic = classic_cppi_weight(V, multiplier=4.0)
    assert torch.allclose(w_budget, w_classic, atol=0.0, rtol=0.0)
    S = _toy_prices()
    V_b, _ = rollout(S, make_weight_fn(
        _cfg("budget_fraction", multiplier=4.0, fraction=1.0), CONV),
        return_path=True)
    V_c, _ = rollout(S, _classic_weight_fn(4.0), return_path=True)
    diff = float((V_b - V_c).abs().max())
    assert diff < 1e-5, f"budget/classic wealth diff {diff}"


def test_stage3_neural_cppi_1_comparator_brake_identical():
    """The ladder brake rung is the shipped emergency rule, literally."""
    V = torch.tensor([100.0, 100.0, 95.0, 80.0])
    cushion = torch.clamp((V - FLOOR) / V, -1.0, 1.0)
    feats = _feats(vol=0.20)
    feats["vol"] = torch.tensor([0.10, 0.40, 0.40, 0.10])
    w_ladder = make_weight_fn(
        _cfg("classic_m4_brake", multiplier=4.0, emergency_vol=0.35,
             emergency_multiplier=None), CONV)(0, V, 1.0, cushion, feats)
    classic_fn = _classic_weight_fn(4.0)
    w_shipped = cppi_constrained_weight_fn(
        classic_fn, multiplier=4.0, emergency_vol=0.35)(
        0, V, 1.0, cushion, feats)
    assert torch.allclose(w_ladder, w_shipped, atol=1e-6, rtol=0.0)
    # Brake active rows (vol 0.40) sit below the plain classic weight.
    w_plain = classic_cppi_weight(V, multiplier=4.0)
    assert bool((w_ladder[1] < w_plain[1]) and (w_ladder[2] < w_plain[2]))
    assert torch.allclose(w_ladder[0], w_plain[0], atol=1e-6, rtol=0.0)


def test_stage3_neural_cppi_1_comparator_buy_hold_costs():
    """Buy/hold pays entry and liquidation costs, nothing between."""
    S = _toy_prices(n_paths=4, steps=30, seed=3)
    V_T, path, ledger = rollout(
        S, make_weight_fn(_cfg("buy_hold"), CONV),
        return_path=True, return_ledger=True)
    ratio = (S[:, -1] / S[:, 0]).numpy()
    oracle = V0 * ratio * (1.0 - COST_RATE) / (1.0 + COST_RATE)
    rel = np.abs(V_T.numpy() - oracle) / oracle
    assert float(rel.max()) < 1e-5, f"buy/hold oracle rel diff {rel.max()}"
    # Mid-horizon fees are float32 rounding residue only: after the
    # entry trade cash is zero up to rounding, so later rebalances
    # trade dust, not positions.
    mid_fees = ledger["fee"][:, 1:].sum(dim=1)
    assert float(mid_fees.abs().max()) < 1e-4


def test_stage3_neural_cppi_1_comparator_cash():
    """All cash grows at r with no trades and no fees."""
    S = _toy_prices(n_paths=4, steps=30, seed=3)
    V_T, _, ledger = rollout(
        S, make_weight_fn(_cfg("all_cash"), CONV),
        return_path=True, return_ledger=True)
    oracle = V0 * np.exp(R * 30 * DT)
    assert float(np.abs(V_T.numpy() - oracle).max()) < 1e-4
    assert float(ledger["cum_fees"].abs().max()) == 0.0


def test_stage3_neural_cppi_1_comparator_table_columns():
    """Every table row carries every required metric."""
    S = _toy_prices()
    configs = [
        _cfg("all_cash"),
        _cfg("buy_hold"),
        _cfg("classic_m4", multiplier=4.0),
        _cfg("classic_m4_brake", multiplier=4.0, emergency_vol=0.35,
             emergency_multiplier=None),
        _cfg("dynamic_vol", multiplier=4.0, vol_target=0.20),
        _cfg("budget_fraction", multiplier=4.0, fraction=0.5),
        _cfg("fixed_cap", weight=0.6),
        _cfg("trend_vol", multiplier=4.0),
    ]
    table = run_ladder(S, configs, CONV)
    required = {
        "mean_wealth", "min_wealth", "mean_min_wealth",
        "breach_count_terminal", "breach_count_anytime",
        "drawdown_mean", "drawdown_max", "turnover", "fees",
        "avg_leverage", "max_weight", "risk_adjusted",
        "mean_shortfall", "selection", "applied_cap",
        "n_paths", "n_steps",
    }
    assert set(table) == {c.name for c in configs}
    for name, row in table.items():
        assert required <= set(row), f"{name} missing {required - set(row)}"
        assert row["breach_count_terminal"] <= row["n_paths"]
        assert row["breach_count_anytime"] >= row["breach_count_terminal"]


def test_stage3_neural_cppi_1_comparator_selection_legal_only():
    """Optimized m is chosen on tuning data only, and validated."""
    tuning = _toy_prices(n_paths=8, steps=30, seed=11)
    best_m, prov = select_multiplier((2.0, 4.0, 6.0), tuning, CONV)
    assert best_m in (2.0, 4.0, 6.0)
    assert prov["selection_kind"] == "tuned"
    assert prov["n_tuning_paths"] == 8
    # Selection never sees test paths: same tuning input, same answer.
    best_m2, _ = select_multiplier((2.0, 4.0, 6.0), tuning, CONV)
    assert best_m2 == best_m
    # Registry validation: tuning inside a held-out episode fails.
    dates = [f"2023-01-{d:02d}" for d in range(1, 32)]
    reg = EpisodeRegistry([
        Episode(name="test_2023", start="2023-01-01", end=None,
                role="test", independence_group="t"),
    ])
    bad = ComparatorConfig(
        name="classic_m_opt", kind="classic_m_opt",
        params={"multiplier": best_m}, selection_kind="tuned",
        tuning_start="2023-01-10", tuning_end="2023-01-20",
        selection="tuned on 2023-01-10..2023-01-20")
    with pytest.raises(ValueError, match="test_2023"):
        validate_selection(bad, registry=reg, dates=dates)
    good = ComparatorConfig(
        name="classic_m_opt", kind="classic_m_opt",
        params={"multiplier": best_m}, selection_kind="tuned",
        tuning_split="validation",
        selection="tuned on validation")
    validate_selection(good, eval_split="test")
    with pytest.raises(ValueError, match="eval split"):
        validate_selection(good, eval_split="validation")
    fixed = _cfg("classic_m4", multiplier=4.0)
    validate_selection(fixed, eval_split="test")


def test_stage3_neural_cppi_1_comparator_clip_disclosure():
    """A 1.0 clip on a trained 1.5 policy is recorded, not hidden."""
    class StubPolicy(torch.nn.Module):
        def forward(self, t, cushion, feats):
            return torch.full_like(cushion, 1.4)

    stub = StubPolicy()
    S = _toy_prices(n_paths=4, steps=10, seed=5)
    raw = run_ladder(S, [_cfg("learned")], CONV, policy=stub)
    capped = run_ladder(
        S, [ComparatorConfig(name="learned", kind="learned",
                              params={"cap": 1.0})], CONV, policy=stub)
    assert raw["learned"]["max_weight"] == pytest.approx(1.4, abs=1e-6)
    assert raw["learned"]["applied_cap"] == pytest.approx(CAP)
    assert capped["learned"]["max_weight"] == pytest.approx(1.0, abs=1e-6)
    assert capped["learned"]["applied_cap"] == pytest.approx(1.0)
    assert capped["learned"]["avg_leverage"] < raw["learned"]["avg_leverage"]


def test_stage3_neural_cppi_1_comparator_dynamic_vol():
    """Dynamic-vol weight falls as realized vol rises."""
    V = torch.full((4,), 100.0)
    cushion = torch.full((4,), 0.1)
    fn = make_weight_fn(
        _cfg("dynamic_vol", multiplier=4.0, vol_target=0.20,
             m_min=1.0, m_max=8.0), CONV)
    w_low = fn(0, V, 1.0, cushion, _feats(vol=0.10))
    w_high = fn(0, V, 1.0, cushion, _feats(vol=0.50))
    assert bool((w_high < w_low).all())
    assert bool(((w_low >= 0.0) & (w_low <= CAP)).all())


def test_stage3_neural_cppi_1_comparator_trend_vol():
    """Trend-vol equals classic when calm, scales down when falling."""
    V = torch.full((4,), 100.0)
    cushion = torch.full((4,), 0.1)
    fn = make_weight_fn(_cfg("trend_vol", multiplier=4.0), CONV)
    w_calm = fn(0, V, 1.0, cushion,
                _feats(vol=0.20, drawdown=0.0, vol_ratio=1.0))
    w_fall = fn(0, V, 1.0, cushion,
                _feats(vol=0.20, drawdown=0.5, vol_ratio=2.0))
    w_classic = classic_cppi_weight(V, multiplier=4.0)
    assert torch.allclose(w_calm, w_classic, atol=1e-6, rtol=0.0)
    assert torch.allclose(w_fall, 0.25 * w_classic, atol=1e-6, rtol=0.0)


def test_stage3_neural_cppi_1_comparator_fixed_cap():
    """Fixed-cap exposure ignores cushion and floor state."""
    V = torch.tensor([100.0, 95.0, 120.0, 80.0])
    cushion = torch.clamp((V - FLOOR) / V, -1.0, 1.0)
    fn = make_weight_fn(_cfg("fixed_cap", weight=0.6), CONV)
    w = fn(0, V, 1.0, cushion, _feats())
    assert torch.allclose(w, torch.full((4,), 0.6), atol=0.0, rtol=0.0)


def test_stage3_neural_cppi_1_comparator_frontier():
    """Frontier points carry risk, return, and floor shortfall."""
    from dhl.eval.comparators import frontier_points

    S = _toy_prices()
    table = run_ladder(
        S, [_cfg("all_cash"), _cfg("classic_m4", multiplier=4.0),
            _cfg("buy_hold")], CONV)
    pts = frontier_points(table)
    assert {p["name"] for p in pts} == set(table)
    for p in pts:
        assert p["risk"] >= 0.0
        assert p["return"] == pytest.approx(table[p["name"]]["mean_wealth"])
        assert p["shortfall"] == pytest.approx(
            table[p["name"]]["mean_shortfall"])
    # Shortfall recomputed independently from a direct rollout.
    V_T = rollout(S, _classic_weight_fn(4.0))
    shortfall = float(torch.relu(torch.full_like(V_T, FLOOR) - V_T).mean())
    assert table["classic_m4"]["mean_shortfall"] == pytest.approx(shortfall)


def test_stage3_neural_cppi_1_comparator_learned_rungs():
    """Learned rungs run through the factory and respect the budget."""
    from dhl.experiments.cppi_learned import CPPIPolicy, seeded_policy

    policy = seeded_policy(CPPIPolicy, seed=7)
    S = _toy_prices(n_paths=4, steps=10, seed=6)
    table = run_ladder(
        S,
        [_cfg("learned"),
         _cfg("learned_constrained", multiplier=4.0),
         _cfg("learned_constrained_brake", multiplier=4.0,
              emergency_vol=0.35, emergency_multiplier=None)],
        CONV, policy=policy)
    assert set(table) == {"learned", "learned_constrained",
                          "learned_constrained_brake"}
    for name, row in table.items():
        assert 0.0 <= row["max_weight"] <= CAP
    # Constrained never exceeds the classic budget on the same states.
    V = torch.full((4,), 100.0)
    cushion = torch.full((4,), 0.1)
    feats = _feats()
    w_con = make_weight_fn(
        _cfg("learned_constrained", multiplier=4.0), CONV,
        policy=policy)(0, V, 1.0, cushion, feats)
    w_classic = classic_cppi_weight(V, multiplier=4.0)
    assert bool((w_con <= w_classic + 1e-6).all())


def test_stage3_neural_cppi_1_comparator_unknown_kind_fails():
    """The factory fails closed on unknown rungs and bad params."""
    with pytest.raises(ValueError, match="unknown comparator kind"):
        make_weight_fn(_cfg("no_such_rung"), CONV)
    with pytest.raises(ValueError, match="fraction"):
        make_weight_fn(
            _cfg("budget_fraction", multiplier=4.0, fraction=1.5), CONV)
    with pytest.raises(ValueError, match="policy"):
        make_weight_fn(_cfg("learned"), CONV, policy=None)
