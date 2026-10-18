"""Tests for the multi-asset CPPI experiment.

No network: synthetic price paths only, except the aligned-returns test
which reads the committed CSVs. Fast by design.
"""

import math

import numpy as np
import torch

from dhl.experiments.cppi_learned import V0
from dhl.experiments.cppi_multi import (
    MAX_DD,
    MultiAssetPolicy,
    _classic_multi_weight_fn,
    _learned_multi_weight_fn,
    _trail_features,
    block_bootstrap_paths_multi,
    evaluate_multi,
    rollout_multi,
    train_multi_policy,
)
from dhl.markets.real import load_aligned_returns


def _const_multi(w):
    w_t = torch.as_tensor(w, dtype=torch.float32)

    def fn(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        return w_t.expand(V.shape[0], -1)

    return fn


def test_joint_bootstrap_preserves_correlation():
    # Two strongly correlated synthetic return series: bootstrapped
    # paths must show similar correlation (joint blocks, not iid).
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, size=3000)
    y = 0.8 * x + rng.normal(0, 0.006, size=3000)
    rets = {"A": x, "B": y}
    P = block_bootstrap_paths_multi(rets, ("A", "B"), 400, 60, 21, seed=5)
    assert P.shape == (400, 61, 2)
    pr = np.stack([np.log(P[:, 1:, k] / P[:, :-1, k]) for k in range(2)])
    corr = np.corrcoef(pr[0].ravel(), pr[1].ravel())[0, 1]
    assert corr > 0.6, f"correlation not preserved: {corr:.3f}"


def test_softmax_weights_valid():
    torch.manual_seed(0)
    pol = MultiAssetPolicy(3)
    w = pol(torch.rand(32), torch.rand(32) * 2 - 1,
            torch.rand(32, 3) * 0.3, torch.randn(32, 3) * 0.02)
    assert w.shape == (32, 4)
    assert bool(((w >= 0) & (w <= 1)).all())
    assert torch.allclose(w.sum(dim=1), torch.ones(32), atol=1e-5)


def test_multi_all_cash_earns_risk_free():
    # All weight on cash: wealth compounds at r, no costs.
    P = torch.full((4, 31, 2), 100.0)
    V = rollout_multi(P, _const_multi([0.0, 0.0, 1.0]), cost_rate=0.0)
    dt = 1.0 / 252.0
    assert abs(float(V[0]) - V0 * math.exp(0.02 * 30 * dt)) < 1e-3


def test_multi_single_risky_matches_spot():
    # All weight on asset 0, no costs: wealth tracks asset 0 exactly.
    rets = np.array([0.01] * 30)
    P = np.empty((1, 31, 2))
    P[0, :, 0] = 100.0 * np.exp(np.concatenate([[0], np.cumsum(rets)]))
    P[0, :, 1] = 100.0
    V = rollout_multi(torch.as_tensor(P, dtype=torch.float32),
                      _const_multi([1.0, 0.0, 0.0]), cost_rate=0.0)
    assert abs(float(V[0]) - P[0, -1, 0]) < 1e-2


def test_drawdown_floor_ratchets_and_breaches():
    # Rising then dipping path: floor must ratchet up with the peak,
    # and a dip of more than MAX_DD from the peak counts as a breach.
    prices = [100.0]
    for r in [0.02] * 10 + [-0.015] * 8:  # peak 122, then -11.4% dip
        prices.append(prices[-1] * math.exp(r))
    P = torch.as_tensor(np.array(prices, dtype=np.float32)).reshape(1, -1, 1)
    P = P.repeat(1, 1, 2)
    V, W, F = rollout_multi(P, _const_multi([1.0, 0.0, 0.0]),
                            floor_mode="drawdown", cost_rate=0.0,
                            return_path=True)
    assert bool((F[:, 1:] >= F[:, :-1] - 1e-6).all())  # ratchets only up
    assert float(F[0].max()) > 90.0  # floor rose above the fixed 90
    assert bool((W < F).any())  # the dip breached the ratcheted floor


def test_fixed_floor_matches_single_asset_rollout():
    # One asset, fixed floor, no costs: multi rollout ~= single rollout.
    from dhl.experiments.cppi_learned import rollout as rollout_1d
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0005, 0.012, size=30)
    p = [100.0]
    for r in rets:
        p.append(p[-1] * math.exp(r))
    S = torch.as_tensor(np.array([p], dtype=np.float32))
    P = S.unsqueeze(-1)

    def w1(i, V, tau_frac, cushion, vol):
        return torch.full_like(V, 0.6)

    V1 = rollout_1d(S, w1, cost_rate=0.0)
    V2 = rollout_multi(P, _const_multi([0.6, 0.4]), cost_rate=0.0)
    assert abs(float(V1[0]) - float(V2[0])) < 1e-2


def test_classic_multi_no_leverage():
    # Classic multi-asset weights never exceed 1.0 anywhere.
    torch.manual_seed(1)
    V = torch.linspace(50, 150, 64)
    fn = _classic_multi_weight_fn()
    w = fn(0, V, 0.5, torch.zeros(64), torch.rand(64, 3),
           torch.rand(64, 3), torch.full((64,), 90.0))
    assert bool(((w >= 0) & (w <= 1)).all())
    assert torch.allclose(w.sum(dim=1), torch.ones(64), atol=1e-5)


def test_training_improves_objective_multi():
    # Few epochs on synthetic correlated returns must improve the
    # penalized objective on a fixed batch.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.012, size=2000)
    y = 0.5 * x + rng.normal(0, 0.008, size=2000)
    rets = {"A": x, "B": y}
    horizon = 10
    P_np = block_bootstrap_paths_multi(rets, ("A", "B"), 256, horizon,
                                       21, seed=123)
    P = torch.as_tensor(P_np, dtype=torch.float32)
    policy = MultiAssetPolicy(2, hidden_dims=(8, 8))

    def objective():
        with torch.no_grad():
            V_T, W, F = rollout_multi(
                P, _learned_multi_weight_fn(policy), return_path=True)
            shortfall = torch.relu(F - W).amax(dim=1)
            return float(-(V_T.mean() - 30.0 * shortfall.mean()))

    before = objective()
    sampler = lambda n, s: block_bootstrap_paths_multi(
        rets, ("A", "B"), n, horizon, 21, seed=s)
    train_multi_policy(policy, sampler, horizon, n_epochs=6,
                       batch_size=256, lr=5e-3, log_every=0, seed=0)
    after = objective()
    assert after < before, f"objective did not improve: {before} -> {after}"


def test_aligned_returns_common_dates():
    dates, rets = load_aligned_returns(("SPY", "TLT", "GLD"))
    assert len(dates) > 2000
    for t, r in rets.items():
        assert len(r) == len(dates) - 1
        assert bool(np.isfinite(r).all())


def test_cross_decade_split_covers_covid_crash():
    # The 2020+ slice must contain the COVID crash drawdown in SPY.
    dates, rets = load_aligned_returns(("SPY",))
    d = np.array(dates[1:])
    post = rets["SPY"][d >= "2020-01-01"]
    pre = rets["SPY"][d < "2020-01-01"]
    assert len(pre) > 1000 and len(post) > 1000
    assert post.min() < -0.10  # COVID crash days are in the test slice


def test_trail_features_are_causal():
    # Shifting prices at i+1 must not move any feature at times <= i.
    torch.manual_seed(2)
    P = torch.exp(torch.randn(4, 31, 3).cumsum(dim=1) * 0.02)
    base = _trail_features(P)
    for i in [0, 5, 15, 29]:
        bumped = P.clone()
        bumped[:, i + 1, :] *= 1.1
        got = _trail_features(bumped)
        assert torch.equal(got[:, : i + 1, :], base[:, : i + 1, :]), f"leak at i={i}"


def test_trail_features_first_step_is_zero():
    # No history at i=0, so the feature is exactly 0, not a nan cleanup.
    torch.manual_seed(3)
    P = torch.exp(torch.randn(3, 21, 2).cumsum(dim=1) * 0.02)
    trail = _trail_features(P)
    assert torch.all(trail[:, 0, :] == 0.0)
    assert torch.isfinite(trail).all()


def test_multi_terminal_liquidation_in_path():
    # Same terminal convention as rollout: W[:, -1] == V_T exactly,
    # with nonzero liquidation costs on the risky legs.
    torch.manual_seed(0)
    rets = torch.randn(3, 20, 2) * 0.02
    P = torch.as_tensor(
        100.0 * np.exp(np.concatenate(
            [np.zeros((3, 1, 2)), np.cumsum(rets.numpy(), axis=1)], axis=1)),
        dtype=torch.float32)
    V, W, F = rollout_multi(P, _const_multi([0.6, 0.4, 0.0]),
                            cost_rate=0.02, return_path=True)
    assert torch.equal(W[:, -1], V)
