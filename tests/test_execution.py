"""Tests for the shared trade-accounting kernel (Stage 3 P0-B).

Every test pins the one convention the kernel implements: a rebalance
moves the portfolio to the desired post-fee weights of post-fee
wealth, and the fee is charged on the actual traded notional. The
single-asset closed form is the one-asset case of the same equation,
so the kernel and the legacy closed form must agree.
"""

import math

import numpy as np
import torch

from dhl.backtest.execution import (
    rebalance_multi,
    rebalance_single,
    solve_post_fee_wealth,
)
from dhl.experiments.cppi_learned import V0, rollout
from dhl.experiments.cppi_multi import rollout_multi

COST = 0.002


def _random_long_only_weights(rng, batch, n_assets):
    raw = rng.random((batch, n_assets))
    # Scale so weights sum to at most one (the rest is cash).
    return raw / (raw.sum(axis=1, keepdims=True) + 0.2)


def test_stage3_neural_cppi_0_kernel_identity_random():
    rng = np.random.default_rng(11)
    batch, na = 64, 3
    V = torch.tensor(rng.uniform(50.0, 150.0, batch), dtype=torch.float64)
    H = torch.tensor(rng.uniform(0.0, 80.0, (batch, na)), dtype=torch.float64)
    w = torch.tensor(_random_long_only_weights(rng, batch, na),
                     dtype=torch.float64)
    D, cash, fee, Vp = rebalance_multi(V, H, w, COST)
    assert torch.allclose(D, w * Vp.unsqueeze(-1), rtol=0, atol=1e-10)
    assert torch.allclose(cash, (1.0 - w.sum(dim=1)) * Vp, atol=1e-10)
    assert torch.allclose(fee, COST * (D - H).abs().sum(dim=1), atol=1e-10)
    assert torch.allclose(Vp, V - fee, atol=1e-10)
    assert bool((cash >= 0).all())
    assert bool((Vp > 0).all())


def test_stage3_neural_cppi_0_kernel_matches_single_asset_closed_form():
    torch.manual_seed(3)
    V = torch.rand(128, dtype=torch.float32) * 100.0 + 50.0
    H = torch.rand(128, dtype=torch.float32) * 80.0
    w = torch.rand(128, dtype=torch.float32)
    post, fee = rebalance_single(V, H, w, COST)
    Vp = solve_post_fee_wealth(V, H.unsqueeze(-1), w.unsqueeze(-1), COST)
    assert torch.allclose(post, w * Vp, rtol=1e-5, atol=1e-4)
    assert torch.allclose(fee, COST * (post - H).abs(), rtol=1e-5, atol=1e-4)


def test_stage3_neural_cppi_0_one_buy_hold_sell_analytic_cost():
    # Buy: V=100, H=0, w=0.5, c=0.002 -> H' = 50 / 1.001.
    # Float64: the analytic values deserve more than float32 precision.
    post, fee = rebalance_single(torch.tensor(100.0, dtype=torch.float64),
                                 torch.tensor(0.0, dtype=torch.float64),
                                 torch.tensor(0.5, dtype=torch.float64), COST)
    assert abs(float(post) - 50.0 / 1.001) < 1e-6
    assert abs(float(fee) - COST * 50.0 / 1.001) < 1e-8
    # Sell the whole holding back to cash: fee is charged on the full
    # holding, and the round trip on flat prices costs exactly the two
    # fees, nothing else.
    post2, fee2 = rebalance_single(torch.tensor(100.0 - float(fee),
                                               dtype=torch.float64),
                                   post,
                                   torch.tensor(0.0, dtype=torch.float64), COST)
    assert float(post2) == 0.0
    assert abs(float(fee2) - COST * float(post)) < 1e-8
    assert abs(float(100.0 - float(fee) - float(fee2))
               - (100.0 - 2 * COST * 50.0 / 1.001)) < 1e-6


def test_stage3_neural_cppi_0_all_cash_rollout_multi_charges_no_fees():
    # All weight on cash with a nonzero cost rate: no trades ever
    # happen, so no fee is charged and wealth compounds at r exactly.
    P = torch.full((4, 31, 2), 100.0)
    V = rollout_multi(P, _const_multi([0.0, 0.0, 1.0]), cost_rate=COST)
    dt = 1.0 / 252.0
    assert abs(float(V[0]) - V0 * math.exp(0.02 * 30 * dt)) < 1e-3


def test_stage3_neural_cppi_0_rollout_multi_matches_numpy_reference():
    # Independent float64 reference for the same convention, written
    # from the equation rather than imported from the kernel.
    P = torch.tensor([[[100.0, 50.0], [110.0, 45.0], [99.0, 55.0],
                       [108.0, 52.0]]], dtype=torch.float64)
    schedule = [[0.4, 0.3, 0.3], [0.2, 0.6, 0.2], [0.7, 0.1, 0.2]]

    def ref():
        Pn = P.numpy()[0]
        cash, D = 100.0, np.zeros(2)
        for i, w in enumerate(schedule):
            w = np.array(w[:2])
            H = D if i == 0 else D * Pn[i] / Pn[i - 1]
            V = cash + H.sum()
            # Breakpoint scan for V' = V - c * sum|w_j V' - H_j|.
            bps = sorted([(H[j] / w[j] if w[j] > 0 else np.inf, j)
                          for j in range(2)])
            Vp = None
            for k in range(3):
                le = [j for _, j in bps[:k]]
                gt = [j for _, j in bps[k:]]
                a = sum(w[j] for j in le) - sum(w[j] for j in gt)
                b = sum(H[j] for j in gt) - sum(H[j] for j in le)
                cand = (V - COST * b) / (1.0 + COST * a)
                lo = bps[k - 1][0] if k > 0 else 0.0
                hi = bps[k][0] if k < 2 else np.inf
                if lo <= cand <= hi:
                    Vp = cand
                    break
            assert Vp is not None
            D = w * Vp
            cash = (1.0 - w.sum()) * Vp * math.exp(0.02 / 252.0)
            D = D  # dollar holdings at price i; grown next date
        HT = D * Pn[3] / Pn[2]
        return cash + HT.sum() - COST * np.abs(HT).sum()

    def fn(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        w = torch.tensor(schedule[i], dtype=torch.float64)
        return w.expand(V.shape[0], -1)

    V, W, _F = rollout_multi(P, fn, cost_rate=COST, return_path=True)
    assert abs(float(V[0]) - ref()) < 1e-8 * max(1.0, abs(ref()))
    assert abs(float(W[0, -1]) - float(V[0])) < 1e-9


def test_stage3_neural_cppi_0_float64_float32_within_tolerance():
    rng = np.random.default_rng(5)
    batch, na = 32, 3
    V64 = torch.tensor(rng.uniform(50.0, 150.0, batch), dtype=torch.float64)
    H64 = torch.tensor(rng.uniform(0.0, 80.0, (batch, na)), dtype=torch.float64)
    w64 = torch.tensor(_random_long_only_weights(rng, batch, na),
                       dtype=torch.float64)
    D64, _, _, Vp64 = rebalance_multi(V64, H64, w64, COST)
    D32, _, _, Vp32 = rebalance_multi(V64.float(), H64.float(),
                                      w64.float(), COST)
    rel = ((Vp32.double() - Vp64).abs() / Vp64).max()
    assert float(rel) < 1e-4
    rel_d = ((D32.double() - D64).abs() / D64.clamp_min(1.0)).max()
    assert float(rel_d) < 1e-4


def test_stage3_neural_cppi_0_rollout_multi_jump_property():
    # Large price jumps, random long-only weights, nonzero cost:
    # wealth stays finite and nonnegative, and the final path point
    # is exactly the post-liquidation terminal wealth.
    torch.manual_seed(7)
    P = torch.tensor(100.0) * torch.exp(
        torch.randn(16, 21, 2) * 0.35)
    V, W, _F = rollout_multi(P, _const_multi([0.5, 0.3, 0.2]),
                             cost_rate=COST, return_path=True)
    assert bool(torch.isfinite(V).all())
    assert bool((V >= 0).all())
    assert torch.allclose(W[:, -1], V, rtol=0, atol=1e-5)


def test_stage3_neural_cppi_0_kernel_gradient_finite():
    V = torch.tensor([100.0, 80.0], dtype=torch.float64)
    H = torch.tensor([[20.0, 30.0], [0.0, 50.0]], dtype=torch.float64)
    w = torch.tensor([[0.4, 0.3], [0.6, 0.2]], dtype=torch.float64,
                     requires_grad=True)
    D, cash, fee, Vp = rebalance_multi(V, H, w, COST)
    (Vp.sum() + D.sum() + cash.sum() + fee.sum()).backward()
    assert w.grad is not None
    assert bool(torch.isfinite(w.grad).all())


def test_stage3_neural_cppi_0_single_asset_rollout_unchanged():
    # Golden fixture computed from the pre-refactor rollout: moving
    # the closed form into the kernel must not change any number.
    torch.manual_seed(0)
    S = 100 * torch.exp(torch.randn(8, 64).cumsum(1) * 0.01)

    def wf(i, V, tau, cush, feats):
        return torch.full_like(V, 0.5)

    V = rollout(S, wf)
    expected = [103.706795, 100.78756, 98.326256, 102.546921,
                97.556892, 107.782867, 111.878876, 97.489014]
    for got, want in zip(V.tolist(), expected):
        assert abs(got - want) < 1e-4, (got, want)


def _const_multi(w):
    w_t = torch.as_tensor(w, dtype=torch.float32)

    def fn(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        return w_t.expand(V.shape[0], -1)

    return fn
