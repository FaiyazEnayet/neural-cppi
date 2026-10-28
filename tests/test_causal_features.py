"""Stage 3 neural-cppi P0-D: valid historical prehistory for every feature.

A rollout supplied only the trade-window price path starts its features
cold: the first trade sees all-zero crash features even if the market
just crashed before the window. The fix separates feature-history context
from the capital/return/trading interval: 60 actual pre-window
observations are prepended to the feature computation only, while wealth
starts at 100, holdings start at zero, and charges start at t0.
"""

import math

import numpy as np
import pytest
import torch

from dhl.experiments.cppi_learned import (
    _crash_features,
    describe_protocol,
    evaluate,
    rollout,
)

WARM_LOOKBACK = 60


def _crash_history(n=60, dtype=torch.float64):
    """60 pre-window prices: a falling, choppy market, not a flat line."""
    rets = np.array([-0.03 if i % 2 == 0 else 0.01 for i in range(n - 1)])
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    return torch.as_tensor(prices, dtype=dtype)


def _window(steps=30, start=100.0, dtype=torch.float64):
    """A calm window: steady small gains, so history drives the features.

    ``start`` should be the last history price for a contiguous window;
    a jump between history end and window start would itself be a fake
    market move.
    """
    rets = np.full(steps, 0.002)
    prices = start * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    return torch.as_tensor(prices, dtype=dtype).unsqueeze(0)


class _Recorder:
    """Weight function that records every decision input, returns fixed w."""

    def __init__(self, weight=0.5):
        self.weight = weight
        self.seen = []

    def __call__(self, i, V, tau_frac, cushion, feats):
        self.seen.append(
            (i, V.detach().clone(),
             {k: v.detach().clone() for k, v in feats.items()}))
        return torch.full_like(V, self.weight)


def test_stage3_neural_cppi_0_future_perturbation_leaves_earlier_decisions_identical():
    hist = _crash_history()
    S = _window(steps=30)
    k = 10
    S2 = S.clone()
    S2[:, k] = S2[:, k] * 1.5
    for history in (None, hist):
        rec1, rec2 = _Recorder(), _Recorder()
        rollout(S, rec1, history=history)
        rollout(S2, rec2, history=history)
        for (i1, V1, f1), (i2, V2, f2) in zip(rec1.seen, rec2.seen):
            assert i1 == i2
            if i1 < k:
                assert torch.equal(V1, V2)
                for name in f1:
                    assert torch.equal(f1[name], f2[name]), (
                        f"step {i1} feature {name} changed by a future price")


def test_stage3_neural_cppi_0_first_trade_nonzero_with_real_history():
    hist = _crash_history()
    # Window opens one normal step after history ends: no jump, no
    # duplicated price at the boundary.
    S = _window(steps=30, start=hist[-1].item() * math.exp(0.002))
    rec_warm, rec_cold = _Recorder(), _Recorder()
    rollout(S, rec_warm, history=hist)
    rollout(S, rec_cold)
    feats0 = rec_warm.seen[0][2]
    assert feats0["vol"][0].item() > 0.0
    assert feats0["drawdown"][0].item() > 0.0
    # Independent numpy oracle for the step-0 vol: the 21 log returns
    # ending at the first window price, annualized.
    full = torch.cat([hist, S[0]]).numpy()
    lr = np.diff(np.log(full))
    vol0 = lr[WARM_LOOKBACK - 21:WARM_LOOKBACK].std(ddof=0) * math.sqrt(252.0)
    assert feats0["vol"][0].item() == pytest.approx(vol0, rel=1e-9)
    # Cold start is a reset, not a claim of calm: all-zero first features.
    feats0_cold = rec_cold.seen[0][2]
    for name in feats0_cold:
        assert feats0_cold[name][0].item() == 0.0


def test_stage3_neural_cppi_0_prehistory_change_never_creates_prewindow_pnl():
    hist_a = _crash_history()
    rets = np.array([0.001 if i % 2 == 0 else -0.001
                     for i in range(WARM_LOOKBACK - 1)])
    hist_b = torch.as_tensor(
        100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)])),
        dtype=torch.float64)
    S = _window(steps=30)
    # Constant weights: history must not move wealth at all.
    V_a, path_a = rollout(S, _Recorder(0.5), history=hist_a,
                          return_path=True)
    V_b, path_b = rollout(S, _Recorder(0.5), history=hist_b,
                          return_path=True)
    assert torch.equal(V_a, V_b)
    assert torch.equal(path_a, path_b)
    assert path_a[0, 0].item() == 100.0
    # History-sensitive weights: only decisions move, never pre-window P&L.
    def sensitive(i, V, tau_frac, cushion, feats):
        return torch.clamp(0.5 + 0.5 * feats["drawdown"], max=1.5)

    _, path_c = rollout(S, sensitive, history=hist_a, return_path=True)
    _, ledger_c = rollout(S, sensitive, history=hist_a, return_ledger=True)
    assert path_c[0, 0].item() == 100.0
    assert torch.equal(ledger_c["cash_pre"][:, 0],
                       torch.full_like(V_a, 100.0))
    assert torch.equal(ledger_c["units_pre"][:, 0],
                       torch.zeros_like(V_a))


def test_stage3_neural_cppi_0_index_positions_and_horizon_match_reference():
    hist = _crash_history()
    S = _window(steps=30)
    rec = _Recorder()
    _, path_warm = rollout(S, rec, history=hist, return_path=True)
    _, path_cold = rollout(S, _Recorder(), return_path=True)
    assert path_warm.shape == path_cold.shape == (1, 31)
    full = torch.cat([hist, S[0]])
    ref = _crash_features(full.unsqueeze(0))
    for i, _, feats in rec.seen:
        for name in feats:
            assert torch.equal(feats[name], ref[name][:, WARM_LOOKBACK + i]), (
                f"step {i} feature {name} misaligned with the full path")
    _, ledger = rollout(S, _Recorder(), history=hist, return_ledger=True)
    assert ledger["price"].shape[1] == 30


def test_stage3_neural_cppi_0_warm_cold_are_named_protocols():
    assert describe_protocol(None) == "cold"
    assert describe_protocol(_crash_history()) == "warm"
    S_np = _window(steps=30).numpy()
    hist_np = _crash_history().numpy()
    m_warm = evaluate(_Recorder(0.5), S_np, history=hist_np)
    m_cold = evaluate(_Recorder(0.5), S_np)
    assert m_warm["protocol"] == "warm"
    assert m_warm["history_days"] == 60
    assert m_cold["protocol"] == "cold"
    assert m_cold["history_days"] == 0
    m_flag = evaluate(_Recorder(0.5), S_np, history=hist_np,
                      context_reserved=True)
    assert m_flag["context_reserved"] is True


def test_stage3_neural_cppi_0_history_shape_validation():
    S = torch.cat([_window(steps=10), _window(steps=10)])
    assert S.shape == (2, 11)
    with pytest.raises(ValueError, match="history"):
        rollout(S, _Recorder(), history=torch.zeros(3, 60, dtype=torch.float64))
    with pytest.raises(ValueError, match="history"):
        rollout(S, _Recorder(), history=torch.zeros(2, 2, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="history"):
        rollout(S, _Recorder(), history=torch.zeros(0, dtype=torch.float64))
    # A shared 1-D history broadcasts across paths; a per-path history
    # must match the path count.
    V1 = rollout(S, _Recorder(), history=_crash_history())
    V2 = rollout(S, _Recorder(),
                 history=torch.stack([_crash_history(), _crash_history()]))
    assert torch.equal(V1, V2)
