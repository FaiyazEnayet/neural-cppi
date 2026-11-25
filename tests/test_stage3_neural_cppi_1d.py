"""P1-D: multi-asset portfolio insurance, economically credible.

Stage 3 V3 unit neural-cppi P1-D (handoff section 121). Tests are
named test_stage3_neural_cppi_1d_* so the handoff selector
``-k stage3_neural_cppi_1`` still matches them.

No network except the committed CSVs (aligned-returns tests read
them from disk). Training tests use tiny epochs and small path sets.
"""

import csv
import math
import os

import numpy as np
import pytest
import torch

from dhl.experiments.cppi_learned import (
    CPPIPolicy,
    V0,
    seeded_policy,
)
from dhl.experiments.cppi_multi import (
    MultiAssetPolicy,
    _classic_multi_weight_fn,
    _learned_multi_weight_fn,
    _learned_single_weight_fn,
    _risk_budget_multi_weight_fn,
    _split_joint_by_date,
    block_bootstrap_paths_multi,
    experiment_cross_decade_multi,
    floor_timestamps,
    risk_contributions,
    rollout_multi,
)
from dhl.markets.real import (
    load_aligned_returns,
    load_ticker_daily,
)


def _const_multi(w):
    w_t = torch.as_tensor(w, dtype=torch.float32)

    def fn(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        return w_t.expand(V.shape[0], -1)

    return fn


def _scalar_reference_multi(P_np, w_risky, cost_rate=0.002, r=0.02,
                            dt=1.0 / 252.0):
    """Independent per-path, per-step ledger in float64.

    Mirrors rollout_multi's accounting from first principles: cash
    plus holdings carried by the day's returns, a post-fee rebalance
    solved by bisection on the monotone fee equation (not by the
    interval scan in dhl.backtest.execution), cash growing at the
    risk-free rate, terminal liquidation charged on its true value.
    Returns (V_T, wealth_path, ledger) with wealth_path[0] == V0 and
    wealth_path[-1] == V_T.
    """
    P = np.asarray(P_np, dtype=float)
    n, steps_p1, na = P.shape
    steps = steps_p1 - 1
    w = np.asarray(w_risky, dtype=float)
    rets = P[:, 1:, :] / P[:, :-1, :]
    growth = math.exp(r * dt)
    V_T = np.empty(n)
    W = np.empty((n, steps + 1))
    ledger = {"wealth_before": [], "fee": [], "wealth_post": [],
              "cash_post": []}
    for k in range(n):
        cash = float(V0)
        D = np.zeros(na)
        W[k, 0] = cash
        for i in range(steps):
            holdings = D if i == 0 else D * rets[k, i - 1, :]
            V = cash + float(holdings.sum())
            # Solve x + c*sum|w*x - H| = V by bisection. f is strictly
            # increasing (slope >= 1 - c > 0); the root lies in
            # [V*(1-c)/(1+c), V] per the execution-kernel guarantee.
            lo = V * (1.0 - cost_rate) / (1.0 + cost_rate)
            hi = V
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                f = mid + cost_rate * float(
                    np.abs(w * mid - holdings).sum()) - V
                if f < 0:
                    lo = mid
                else:
                    hi = mid
            Vp = 0.5 * (lo + hi)
            D = w * Vp
            fee = cost_rate * float(np.abs(D - holdings).sum())
            cash_post = (1.0 - float(w.sum())) * Vp
            ledger["wealth_before"].append(V)
            ledger["fee"].append(fee)
            ledger["wealth_post"].append(Vp)
            ledger["cash_post"].append(cash_post)
            cash = cash_post * growth
            V = cash + float((D * rets[k, i, :]).sum())
            W[k, i + 1] = V
        holdings_T = D * rets[k, steps - 1, :]
        V_T[k] = (cash + float(holdings_T.sum())
                  - cost_rate * float(np.abs(holdings_T).sum()))
        W[k, steps] = V_T[k]
    return V_T, W, ledger


def _synthetic_paths(seed=0, n_paths=3, steps=30, n_assets=3):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, size=(n_paths, steps, n_assets))
    P = np.empty((n_paths, steps + 1, n_assets))
    P[:, 0, :] = 100.0
    P[:, 1:, :] = 100.0 * np.exp(np.cumsum(rets, axis=1))
    return P


def test_stage3_neural_cppi_1d_ledger_reference_matches_rollout():
    # Independent scalar ledger must match the vectorized rollout to
    # float32 noise on wealth path and terminal wealth, for a fully
    # invested and a cash-holding allocation, with costs on.
    for w_risky in ([0.5, 0.3, 0.2], [0.4, 0.3, 0.1]):
        P_np = _synthetic_paths()
        w_full = list(w_risky) + [1.0 - sum(w_risky)]
        P = torch.as_tensor(P_np, dtype=torch.float32)
        V, W, F = rollout_multi(P, _const_multi(w_full), cost_rate=0.002,
                                return_path=True)
        V_ref, W_ref, _ = _scalar_reference_multi(
            P_np, w_risky, cost_rate=0.002)
        assert np.max(np.abs(V.numpy() - V_ref)) < 5e-4
        assert np.max(np.abs(W.numpy() - W_ref)) < 5e-4


def test_stage3_neural_cppi_1d_ledger_per_trade_identity():
    # Every rebalance must satisfy wealth_post + fee == wealth_before,
    # with cash and holdings never negative (long-only, no borrowing).
    P_np = _synthetic_paths()
    P = torch.as_tensor(P_np, dtype=torch.float32)
    V, W, F, ledger = rollout_multi(
        P, _const_multi([0.5, 0.3, 0.1, 0.1]), cost_rate=0.002,
        return_path=True, return_ledger=True)
    wb = np.asarray(ledger["wealth_before"])
    fee = np.asarray(ledger["fee"])
    wp = np.asarray(ledger["wealth_post"])
    assert np.max(np.abs(wp + fee - wb)) < 1e-4
    assert bool((np.asarray(ledger["cash_post"]) >= -1e-6).all())
    assert bool((np.asarray(ledger["holdings_post"]) >= -1e-6).all())
    # Ledger has one row per rebalance step (30 steps, 3 paths).
    assert wb.shape == (30, 3)


def test_stage3_neural_cppi_1d_floor_timestamps_defined():
    dates = [f"2020-01-{d:02d}" for d in range(2, 21)]
    assert floor_timestamps(dates, 18) == dates
    with pytest.raises(ValueError):
        floor_timestamps(dates, 17)
    with pytest.raises(ValueError):
        floor_timestamps(dates[:5], 18)


def test_stage3_neural_cppi_1d_floor_t0_t1_exact():
    # Dated fixture: 19 consecutive common trading dates from the
    # committed data; 10 up days then 8 down days. F[0] is the floor
    # at the start date, F[j] the floor governing wealth on dates[j].
    dates, _ = load_aligned_returns(("SPY", "TLT", "GLD"))
    dates = dates[:19]
    prices = [100.0]
    for r in [0.02] * 10 + [-0.02] * 8:
        prices.append(prices[-1] * math.exp(r))
    P = torch.as_tensor(np.array(prices, dtype=np.float32)).reshape(1, -1, 1)
    P = P.repeat(1, 1, 2)
    _, W, F = rollout_multi(P, _const_multi([1.0, 0.0, 0.0]),
                            floor_mode="drawdown", cost_rate=0.0,
                            return_path=True)
    assert floor_timestamps(dates, 18) == dates
    w = W[0].numpy()
    f = F[0].numpy()
    # t0: the date-0 floor is (1 - max_dd) * V0, not the floor param.
    assert f[0] == pytest.approx(0.90 * 100.0, abs=1e-4)
    # t1: F[j] is set by the peak through the previous date only.
    for j in range(1, len(w)):
        assert f[j] == pytest.approx(0.90 * max(w[:j]), rel=1e-5)
    # The breach is stamped to the right calendar date: the dip
    # crosses 10% below the peak on the 6th down day, dates[16], and
    # stays breached through the end of the path.
    breached = [dates[j] for j in range(len(w)) if w[j] < f[j]]
    assert breached == dates[16:]
    assert breached[0] == dates[16]
    # Fixed mode: every entry is the floor parameter itself.
    _, _, F_fixed = rollout_multi(P, _const_multi([1.0, 0.0, 0.0]),
                                  floor_mode="fixed", floor=80.0,
                                  cost_rate=0.0, return_path=True)
    assert np.allclose(F_fixed[0].numpy(), 80.0, atol=1e-5)


def test_stage3_neural_cppi_1d_floor_ratchets_on_peak():
    # After a new peak, the floor steps up to 0.9 times that peak at
    # the next date and never falls while wealth slides.
    prices = [100.0]
    for r in [0.02] * 10 + [-0.02] * 8:
        prices.append(prices[-1] * math.exp(r))
    P = torch.as_tensor(np.array(prices, dtype=np.float32)).reshape(1, -1, 1)
    P = P.repeat(1, 1, 2)
    _, W, F = rollout_multi(P, _const_multi([1.0, 0.0, 0.0]),
                            floor_mode="drawdown", cost_rate=0.0,
                            return_path=True)
    peak = 100.0 * math.exp(0.20)
    assert float(F[0, 11]) == pytest.approx(0.90 * peak, rel=1e-5)
    assert np.allclose(F[:, 11:], 0.90 * peak, rtol=1e-5)
    assert bool((F[:, 1:] >= F[:, :-1] - 1e-6).all())


def test_stage3_neural_cppi_1d_min_data_guard():
    rng = np.random.default_rng(0)
    rets = {"A": rng.normal(0, 0.01, size=100),
            "B": rng.normal(0, 0.01, size=100)}
    # A path longer than the aligned history would repeat history
    # end to end: fail closed instead of silently wrapping.
    with pytest.raises(ValueError):
        block_bootstrap_paths_multi(rets, ("A", "B"), 4, 101, 21, seed=1)
    with pytest.raises(ValueError):
        block_bootstrap_paths_multi(rets, ("A", "B"), 4, 60, 101, seed=1)
    P = block_bootstrap_paths_multi(rets, ("A", "B"), 4, 60, 21, seed=1)
    assert P.shape == (4, 61, 2)


def test_stage3_neural_cppi_1d_aligned_returns_inner_join(tmp_path):
    # A date present for one ticker only must not enter the aligned
    # series: no window may cross a date with missing data.
    rows_a = [("2020-01-02", 100.0), ("2020-01-03", 101.0),
              ("2020-01-06", 102.0), ("2020-01-07", 103.0),
              ("2020-01-08", 104.0)]
    rows_b = [("2020-01-02", 50.0), ("2020-01-03", 51.0),
              ("2020-01-07", 52.0), ("2020-01-08", 53.0)]
    for ticker, rows in (("a", rows_a), ("b", rows_b)):
        with open(tmp_path / f"{ticker}_daily_2015.csv", "w",
                  newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Open", "High", "Low", "Close",
                             "AdjClose", "Volume"])
            for d, p in rows:
                writer.writerow([d, p, p, p, p, p, 1000])
    dates, rets = load_aligned_returns(("A", "B"), data_dir=str(tmp_path))
    assert dates == ["2020-01-02", "2020-01-03", "2020-01-07",
                     "2020-01-08"]
    assert len(rets["A"]) == len(dates) - 1 == 3
    assert rets["A"] == pytest.approx(np.log([101 / 100, 103 / 101,
                                              104 / 103]))
    assert rets["B"] == pytest.approx(np.log([51 / 50, 52 / 51,
                                              53 / 52]))


def test_stage3_neural_cppi_1d_aligned_returns_are_adjusted():
    # Returns on the committed data are log ratios of adjusted
    # closes (total returns, dividends reinvested), not raw closes.
    dates, rets = load_aligned_returns(("SPY", "TLT", "GLD"))
    assert len(dates) > 2000
    for t in ("SPY", "TLT", "GLD"):
        d = load_ticker_daily(t)
        by_date = dict(zip(d["dates"], d["adjclose"]))
        closes = dict(zip(d["dates"], d["close"]))
        adj = np.array([by_date[x] for x in dates])
        raw = np.array([closes[x] for x in dates])
        assert rets[t] == pytest.approx(np.log(adj[1:] / adj[:-1]),
                                        rel=1e-12)
    # SPY pays dividends, so adjusted closes must differ from raw
    # closes somewhere: the loader really uses total returns.
    d = load_ticker_daily("SPY")
    by_date = dict(zip(d["dates"], d["adjclose"]))
    closes = dict(zip(d["dates"], d["close"]))
    adj = np.array([by_date[x] for x in dates])
    raw = np.array([closes[x] for x in dates])
    assert float(np.max(np.abs(np.log(raw[1:] / raw[:-1])
                               - rets["SPY"]))) > 0.0


def test_stage3_neural_cppi_1d_caps_match_across_candidates():
    # All three experiment_multi_asset candidates share cap 1.0: the
    # total risky weight never exceeds one, and the classic candidate
    # sits exactly at the cap at full cushion.
    V = torch.linspace(50, 150, 64)
    zeros = torch.zeros(64)
    vols = torch.rand(64, 3) * 0.3
    trail = torch.randn(64, 3) * 0.02
    floor_t = torch.full((64,), 90.0)

    classic = _classic_multi_weight_fn()
    w = classic(0, V, 0.5, torch.ones(64), vols, trail, floor_t)
    assert bool(((w >= 0) & (w <= 1)).all())
    assert torch.allclose(w.sum(dim=1), torch.ones(64), atol=1e-5)
    # Classic risky weight follows the cushion formula exactly:
    # min(4 * max(V - floor, 0) / V, 1), never above the cap.
    expected = torch.clamp(
        4.0 * torch.clamp(V - 90.0, min=0.0) / V, max=1.0)
    assert torch.allclose(w[:, :3].sum(dim=1), expected, atol=1e-5)
    assert bool((w[:, :3].sum(dim=1) <= 1.0 + 1e-6).all())
    # Full cushion (V far above the floor): classic is at the cap.
    w_full = classic(0, torch.full((8,), 1e6), 0.5, torch.ones(8),
                     torch.rand(8, 3), torch.randn(8, 3),
                     torch.zeros(8))
    assert torch.allclose(w_full[:, :3].sum(dim=1), torch.ones(8),
                          atol=1e-6)

    policy_s = seeded_policy(CPPIPolicy, seed=0)
    single = _learned_single_weight_fn(policy_s)
    w_s = single(0, V, 0.5, torch.ones(64), vols, trail, floor_t)
    assert bool(((w_s >= 0) & (w_s <= 1)).all())
    assert bool((w_s[:, :3].sum(dim=1) <= 1.0 + 1e-6).all())

    # A policy proposing twice the cap must still come back capped.
    class _Greedy:
        def __call__(self, t, cushion, feats):
            return torch.full_like(t, 2.0)

    greedy = _learned_single_weight_fn(_Greedy())
    w_g = greedy(0, V, 0.5, torch.ones(64), vols, trail, floor_t)
    assert torch.allclose(w_g[:, 0], torch.ones(64), atol=1e-6)
    assert torch.allclose(w_g[:, -1], torch.zeros(64), atol=1e-6)

    torch.manual_seed(0)
    policy_m = seeded_policy(MultiAssetPolicy, 3, seed=0)
    multi = _learned_multi_weight_fn(policy_m)
    w_m = multi(0, V, 0.5, torch.ones(64), vols, trail, floor_t)
    assert bool(((w_m >= 0) & (w_m <= 1)).all())
    assert torch.allclose(w_m.sum(dim=1), torch.ones(64), atol=1e-5)
    assert bool((w_m[:, :3].sum(dim=1) <= 1.0 + 1e-6).all())


def test_stage3_neural_cppi_1d_shorting_and_borrow_fail_closed():
    P = torch.full((2, 11, 2), 100.0)

    def _short(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        w = torch.zeros(V.shape[0], 3)
        w[:, 0] = -0.5
        w[:, 1] = 1.0
        w[:, 2] = 0.5
        return w

    with pytest.raises(ValueError):
        rollout_multi(P, _short)

    def _borrow(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        w = torch.zeros(V.shape[0], 3)
        w[:, 0] = 1.5
        w[:, 1] = 0.0
        w[:, 2] = -0.5
        return w

    with pytest.raises(ValueError):
        rollout_multi(P, _borrow)


def test_stage3_neural_cppi_1d_risk_contributions_sum_to_one():
    rng = np.random.default_rng(0)
    # Equal weights on an identity covariance: equal contributions.
    rc = risk_contributions(np.array([1 / 3, 1 / 3, 1 / 3]),
                            np.eye(3))
    assert rc == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    # Random SPD covariance: contributions still sum to one. (A
    # contribution may be negative when an asset hedges the others;
    # only the sum is pinned here.)
    A = rng.normal(size=(3, 3))
    cov = A @ A.T + np.eye(3) * 0.1
    for w in (np.array([0.5, 0.3, 0.2]), np.array([0.7, 0.2, 0.1]),
              np.array([0.0, 0.6, 0.4])):
        rc = risk_contributions(w, cov)
        assert float(rc.sum()) == pytest.approx(1.0, abs=1e-12)
    # Zero weight means zero contribution; one asset means all of it.
    assert risk_contributions(np.array([0.0, 1.0]), np.eye(2))[0] == 0.0
    assert risk_contributions(np.array([1.0]), np.array([[0.04]]))[0] == 1.0
    with pytest.raises(ValueError):
        risk_contributions(np.array([0.5, 0.5]), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        risk_contributions(np.array([np.nan, 0.5]), np.eye(2))


def test_stage3_neural_cppi_1d_risk_budget_equalizes_contributions():
    # The total-risk-budget split hands the riskier asset the smaller
    # slice, so on a diagonal covariance the contributions equalize.
    torch.manual_seed(0)
    fn = _risk_budget_multi_weight_fn(multiplier=4.0, cap=1.0)
    V = torch.full((16,), 120.0)
    floor_t = torch.full((16,), 90.0)
    vols = torch.tensor([[0.10, 0.20, 0.40]] * 16)
    w = fn(0, V, 0.5, torch.ones(16), vols, torch.zeros(16, 3), floor_t)
    assert bool(((w >= 0) & (w <= 1)).all())
    assert bool((w[:, :3].sum(dim=1) <= 1.0 + 1e-6).all())
    risky = w[0, :3].numpy()
    cov = np.diag([0.10 ** 2, 0.20 ** 2, 0.40 ** 2])
    rc = risk_contributions(risky, cov)
    assert rc == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)
    # Zero vols fall back to the equal mix instead of dividing by zero.
    w_flat = fn(0, V, 0.5, torch.ones(16), torch.zeros(16, 3),
                torch.zeros(16, 3), floor_t)
    assert torch.allclose(w_flat[:, :3],
                          w_flat[:, 0:1].expand(-1, 3), atol=1e-6)


def test_stage3_neural_cppi_1d_cross_decade_split():
    dates, rets = load_aligned_returns(("SPY", "TLT", "GLD"))
    pre, post = _split_joint_by_date(dates, rets)
    d = np.array(dates[1:])
    assert len(pre["SPY"]) == int((d < "2020-01-01").sum())
    assert len(post["SPY"]) == int((d >= "2020-01-01").sum())
    assert len(pre["SPY"]) > 1000 and len(post["SPY"]) > 1000
    # The 2020+ slice holds the COVID crash in SPY.
    assert post["SPY"].min() < -0.10
    # Every ticker is split on the same dates: same lengths both sides.
    for t in ("TLT", "GLD"):
        assert len(pre[t]) == len(pre["SPY"])
        assert len(post[t]) == len(post["SPY"])


def test_stage3_neural_cppi_1d_cross_decade_multi_runs():
    dates, rets = load_aligned_returns(("SPY", "TLT", "GLD"))
    out = experiment_cross_decade_multi(
        dates, rets, horizon=10, n_epochs=1, batch_size=64, n_test=50,
        seed=0)
    assert set(out) == {"learned_multi (train pre-2020)",
                        "learned_multi (train all years)",
                        "classic m=4 (mix)"}
    for name, m in out.items():
        for k in ("mean_wealth", "std_wealth", "p_breach_terminal",
                  "p_breach_anytime", "mean_min_wealth"):
            assert k in m, f"{name} missing {k}"
            assert math.isfinite(m[k]), f"{name}.{k} not finite"
        assert 0.0 <= m["p_breach_terminal"] <= 1.0
        assert 0.0 <= m["p_breach_anytime"] <= 1.0
        assert m["mean_wealth"] > 0.0
