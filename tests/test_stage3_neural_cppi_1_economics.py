"""Stage 3 E1: economics kernel and genuine floor tests.

Covers the shared wealth-accounting invariants the audit item E1
requires, plus the three fail-open gaps found at HEAD by measurement
(not from audit prose):

1. ``rollout_multi`` accepted ANY weight matrix silently: negative
   weights (implicit shorting) and risky sums above one (implicit
   borrowing at the cash rate, with no borrow-rate parameter) both
   ran without complaint, even though the function documents a
   long-only, sum-to-one contract.
2. In drawdown mode the returned floor series ``F`` started with the
   fixed-floor parameter instead of the drawdown formula, so
   ``F[0]`` disagreed with the floor at every other date whenever
   ``floor != (1 - max_dd) * V0`` (the defaults coincide, which hid
   it).
3. Single-asset ``rollout`` kept trading a path whose wealth had gone
   nonpositive: with a levered weight the kernel invariant
   ``holdings = w * V'`` then produces NEGATIVE holdings (implicit
   shorting from a long-only weight), so post-bankruptcy numbers were
   meaningless yet flowed into breach statistics as if real.

Also pins the legacy diff table: the pre-NC3 accounting (parent of
merge 6962e353) charged fees on the pre-fee target and liquidated on
the stale pre-move holding. The oracle below is that exact old code,
copied from git history, so old-vs-new differences on identical
paths are measured, not asserted from prose.
"""

import math

import numpy as np
import pytest
import torch

from dhl.backtest.execution import validate_portfolio_weights
from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    DT,
    FLOOR,
    R,
    V0,
    _classic_weight_fn,
    classic_cppi_weight,
    rollout,
)
from dhl.experiments.cppi_multi import rollout_multi
from dhl.markets.real import (
    block_bootstrap_paths,
    load_spy_daily,
    log_returns,
)


def _const_weight(w):
    def fn(i, V, tau_frac, cushion, feats):
        return torch.full_like(V, w)

    return fn


def _const_multi(weights):
    w = torch.as_tensor(weights, dtype=torch.float32)

    def fn(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        return w.expand(V.shape[0], -1).clone()

    return fn


def _legacy_rollout_old_approximation(S, weight_fn, floor=FLOOR, r=R,
                                      dt=DT, cost_rate=COST_RATE):
    """Pre-NC3 single-asset accounting, verbatim from git history.

    Source: ``src/dhl/experiments/cppi_learned.py::rollout`` at the
    first parent of merge 6962e353 (the NC3 merge). Differences from
    the current kernel, all in this oracle: the fee is charged on the
    PRE-fee target ``w * V`` while the holding is set to ``w`` times
    post-fee wealth; cash always grows at ``r`` (no borrow-rate
    parameter, so levered paths borrow at the cash rate); the
    terminal liquidation charges on the STALE pre-move holding ``D``
    instead of the terminal holding value. ``weight_fn`` here takes
    ``(i, V)`` only; the strategies used with this oracle (classic,
    constant) ignore the other rollout arguments, so the comparison
    is exact for them.
    """
    n, steps_p1 = S.shape
    steps = steps_p1 - 1
    V = torch.full((n,), V0, dtype=S.dtype)
    D = torch.zeros(n, dtype=S.dtype)
    growth = math.exp(r * dt)
    rets = S[:, 1:] / S[:, :-1]
    for i in range(steps):
        w = weight_fn(i, V)
        prev_grown = D if i == 0 else D * rets[:, i - 1]
        target = w * V
        V = V - cost_rate * (target - prev_grown).abs()
        D = w * V
        V = (V - D) * growth + D * rets[:, i]
    V = V - cost_rate * D.abs()
    return V


# --- Gap 1: multi-asset weight validation -------------------------------

def test_stage3_neural_cppi_1_economics_multi_rejects_negative_weights():
    P = torch.full((2, 4, 1), 100.0)
    P[:, 1:, 0] = 101.0
    with pytest.raises(ValueError, match="negative"):
        rollout_multi(P, _const_multi([-0.5, 1.5]), cost_rate=0.0)


def test_stage3_neural_cppi_1_economics_multi_rejects_risky_sum_above_one():
    P = torch.full((2, 4, 1), 100.0)
    P[:, 1:, 0] = 101.0
    with pytest.raises(ValueError, match="sum"):
        rollout_multi(P, _const_multi([1.2, 0.0]), cost_rate=0.0)


def test_stage3_neural_cppi_1_economics_multi_rejects_weights_not_summing():
    P = torch.full((2, 4, 1), 100.0)
    P[:, 1:, 0] = 101.0
    with pytest.raises(ValueError, match="sum"):
        rollout_multi(P, _const_multi([0.5, 0.4]), cost_rate=0.0)


def test_stage3_neural_cppi_1_economics_validate_helper_accepts_valid():
    w = torch.tensor([[0.3, 0.2, 0.5], [0.0, 0.0, 1.0]])
    validate_portfolio_weights(w)  # must not raise
    with pytest.raises(ValueError):
        validate_portfolio_weights(torch.tensor([[0.5, 0.6]]))
    with pytest.raises(ValueError):
        validate_portfolio_weights(torch.tensor([[-0.1, 1.1]]))


def test_stage3_neural_cppi_1_economics_multi_valid_weights_unchanged():
    # A valid fixed-mix run must still work after validation is added.
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0005, 0.012, size=30)
    p = [100.0]
    for r in rets:
        p.append(p[-1] * math.exp(r))
    P = torch.as_tensor(np.array([p], dtype=np.float32)).unsqueeze(-1)
    V = rollout_multi(P, _const_multi([0.6, 0.4]), cost_rate=0.0)
    assert float(V[0]) > 0.0


# --- Gap 2: drawdown floor series indexing -------------------------------

def test_stage3_neural_cppi_1_economics_drawdown_floor_starts_at_formula():
    # floor param 80 differs from (1 - max_dd) * V0 = 90: the date-0
    # floor in drawdown mode must be the formula value, not the param.
    P = torch.full((1, 4, 1), 100.0)
    _, _, F = rollout_multi(P, _const_multi([0.0, 1.0]),
                            floor_mode="drawdown", floor=80.0,
                            max_dd=0.10, cost_rate=0.0, return_path=True)
    assert float(F[0, 0]) == pytest.approx(90.0, abs=1e-5)
    _, _, F_fixed = rollout_multi(
        P, _const_multi([0.0, 1.0]), floor_mode="fixed", floor=80.0,
        max_dd=0.10, cost_rate=0.0, return_path=True)
    assert float(F_fixed[0, 0]) == pytest.approx(80.0, abs=1e-5)


def test_stage3_neural_cppi_1_economics_drawdown_floor_indexing():
    # F[j] must equal (1 - max_dd) times the peak through W[j - 1]:
    # the floor governing date-j wealth is set by the peak BEFORE
    # date j (a new peak cannot breach its own floor, so both
    # formulations agree, but the lag-one form is the checkable one).
    prices = [100.0]
    for r in [0.02] * 10 + [-0.015] * 8:
        prices.append(prices[-1] * math.exp(r))
    P = torch.as_tensor(np.array(prices, dtype=np.float32)).reshape(1, -1, 1)
    P = P.repeat(1, 1, 2)
    _, W, F = rollout_multi(P, _const_multi([1.0, 0.0, 0.0]),
                            floor_mode="drawdown", cost_rate=0.0,
                            return_path=True)
    w = W[0].numpy()
    f = F[0].numpy()
    assert bool((F[:, 1:] >= F[:, :-1] - 1e-6).all())  # ratchets only up
    for j in range(1, len(w)):
        peak_before = max(w[:j])
        assert f[j] == pytest.approx(0.90 * peak_before, rel=1e-5)
    assert f[0] == pytest.approx(0.90 * w[0], rel=1e-5)


def test_stage3_neural_cppi_1_economics_terminal_floor_new_peak_no_breach():
    # Final move sets a new peak: the horizon floor is 0.9 times that
    # peak, and post-liquidation wealth (peak minus a 20bp fee) sits
    # above it, so no terminal breach. The returned F[steps] (floor
    # at the last trade date) gives the same verdict here because a
    # fee below 10% of wealth can never push wealth under 0.9*peak.
    P = torch.as_tensor([[[100.0], [110.0], [121.0]]],
                        dtype=torch.float32)
    V_T, W, F = rollout_multi(P, _const_multi([1.0, 0.0]),
                              floor_mode="drawdown", cost_rate=0.002,
                              return_path=True)
    assert float(V_T[0]) > 0.90 * float(W[0, -2])
    assert float(V_T[0]) >= float(F[0, -1])


def test_stage3_neural_cppi_1_economics_fixed_floor_series_unchanged():
    P = torch.full((1, 5, 1), 100.0)
    P[:, 1:, 0] = torch.tensor([101.0, 102.0, 101.5, 103.0])
    _, _, F = rollout_multi(P, _const_multi([0.5, 0.5]),
                            floor_mode="fixed", floor=90.0,
                            cost_rate=0.0, return_path=True)
    assert bool((F == 90.0).all())


# --- Gap 3: nonpositive wealth fails closed ------------------------------

def test_stage3_neural_cppi_1_economics_bankrupt_path_fails_closed():
    # Levered weight into an 80% one-day crash: wealth goes negative
    # at date 2, and the old code kept trading it (negative holdings
    # from a long-only weight). Now the rollout must refuse.
    S = torch.tensor([[100.0, 100.0, 20.0, 20.0, 25.0]])
    with pytest.raises(ValueError, match="nonpositive pre-trade wealth"):
        rollout(S, _const_weight(1.5), floor=FLOOR, cost_rate=0.002,
                return_path=True)


def test_stage3_neural_cppi_1_economics_terminal_bankruptcy_is_honest():
    # The check is on PRE-TRADE wealth at trade dates only: a path
    # that is solvent at its last trade but crashes on the final move
    # returns its negative terminal wealth honestly (bankruptcy at
    # the horizon), rather than raising or hiding it.
    S = torch.tensor([[100.0, 110.0, 22.0]])
    V_T = rollout(S, _const_weight(1.5), floor=FLOOR, cost_rate=0.0)
    assert float(V_T[0]) < 0.0
    # Hand check: step 0 invests 150 (cash -50, grows at r); step 1
    # pre-trade wealth = -50*exp(r/252) + 165 > 0, rebalances to 1.5x;
    # final move to 22/110 = 0.2 crashes it below zero.
    cash1 = -50.0 * math.exp(R * DT)
    v1 = cash1 + 165.0
    post1 = 1.5 * v1
    cash_post1 = v1 - post1
    v_terminal = cash_post1 * math.exp(R * DT) + post1 * 0.2
    assert float(V_T[0]) == pytest.approx(v_terminal, rel=1e-5)


# --- Exact fees and levered borrowing ------------------------------------

def test_stage3_neural_cppi_1_economics_borrow_accrues_at_borrow_rate():
    # Levered, no crash, cost 0: negative cash must accrue at the
    # borrow rate (10%), not the cash rate (2%). Independent oracle
    # in plain floats, mirroring the documented convention only.
    S = torch.tensor([[100.0, 110.0, 121.0]])
    V_T = rollout(S, _const_weight(1.5), floor=FLOOR, r=0.02,
                  borrow_rate=0.10, cost_rate=0.0)
    # Step 0: invest 150, cash -50, grows at 10%: -50*exp(0.10/252).
    # Step 1: V = -50*exp(0.10/252) + 165; reinvest 1.5x; cash negative
    # again, grows at 10%; terminal: cash + 1.5*V*121/110... computed
    # stepwise below in the same convention.
    cash = -50.0 * math.exp(0.10 * DT)
    v1 = cash + 165.0
    post = 1.5 * v1
    cash = (v1 - post) * math.exp(0.10 * DT)
    v_terminal = cash + post * (121.0 / 110.0)
    assert float(V_T[0]) == pytest.approx(v_terminal, rel=1e-5)
    # And at the default (borrow_rate=None) it borrows at r instead.
    V_T_r = rollout(S, _const_weight(1.5), floor=FLOOR, r=0.02,
                    borrow_rate=None, cost_rate=0.0)
    cash = -50.0 * math.exp(0.02 * DT)
    v1 = cash + 165.0
    post = 1.5 * v1
    cash = (v1 - post) * math.exp(0.02 * DT)
    v_terminal = cash + post * (121.0 / 110.0)
    assert float(V_T_r[0]) == pytest.approx(v_terminal, rel=1e-5)


# --- Property-style seeded fixtures --------------------------------------

def test_stage3_neural_cppi_1_economics_ledger_identity_seeded():
    # Self-financing identity at every trade, over seeded random
    # paths and weights: cash_post + holding_post == wealth_pre - fee,
    # and fee == cost * notional. No new dependency: the "property"
    # loop is a fixed-seed deterministic sweep.
    rng = np.random.default_rng(20261028)
    torch.manual_seed(0)
    for trial in range(12):
        n_paths, steps = 6, 10
        rets = rng.normal(0.0005, 0.015, size=(n_paths, steps))
        S = torch.as_tensor(
            100.0 * np.exp(np.concatenate(
                [np.zeros((n_paths, 1)), np.cumsum(rets, axis=1)],
                axis=1)), dtype=torch.float32)
        w_const = float(rng.uniform(0.0, CAP))
        _, _, ledger = rollout(S, _const_weight(w_const), cost_rate=0.002,
                               return_path=True, return_ledger=True)
        lhs = ledger["cash_post"] + ledger["holding_post"]
        rhs = ledger["wealth_pre"] - ledger["fee"]
        assert torch.allclose(lhs, rhs, atol=1e-4)
        assert torch.allclose(
            ledger["fee"],
            0.002 * ledger["notional"], atol=1e-5)
        # Terminal identity: cash + terminal holding - liq fee.
        assert torch.allclose(
            ledger["cash_terminal"] + ledger["holding_terminal"]
            - ledger["liquidation_fee"],
            ledger["terminal_wealth"], atol=1e-4)
        # Cumulative fees reconcile step by step.
        assert torch.allclose(
            ledger["fee"].sum(dim=1) + ledger["liquidation_fee"],
            ledger["cum_fees"], atol=1e-4)


def test_stage3_neural_cppi_1_economics_multi_oracle_short_path():
    # Independent short-path oracle for the multi loop wiring (rets
    # indexing, cash growth, liquidation) using the shared kernel for
    # the trade solve only, not re-implementing it.
    from dhl.backtest.execution import rebalance_multi
    torch.manual_seed(0)
    P = torch.tensor([[[100.0, 100.0], [110.0, 99.0], [121.0, 108.9]]])
    w = torch.tensor([[0.6, 0.4, 0.0]])
    V = rollout_multi(P, _const_multi([0.6, 0.4, 0.0]), cost_rate=0.002)
    # Oracle: replicate the loop by hand in floats.
    cash, D = 100.0, [0.0, 0.0]
    prices = [[100.0, 100.0], [110.0, 99.0], [121.0, 108.9]]
    growth = math.exp(R * DT)
    for i in range(2):
        holdings = D if i == 0 else [D[k] * prices[i][k] / prices[i - 1][k]
                                     for k in range(2)]
        v = cash + sum(holdings)
        hp, cp, fee, vp = rebalance_multi(
            torch.tensor([v]), torch.tensor([holdings]),
            torch.tensor([[0.6, 0.4]]), 0.002)
        cash = float(cp[0]) * growth
        D = [float(hp[0][k]) for k in range(2)]
    holdings_t = [D[k] * prices[2][k] / prices[1][k] for k in range(2)]
    v_expected = cash + sum(holdings_t) - 0.002 * sum(abs(h)
                                                     for h in holdings_t)
    assert float(V[0]) == pytest.approx(v_expected, rel=1e-5)


# --- Legacy diff table ----------------------------------------------------

def test_stage3_neural_cppi_1_economics_legacy_pinned_case():
    # One step, fully invested, 100 -> 92.5 at 2% cost. Old code:
    # entry fee on the pre-fee target (2.00 on 100), holding 98.00,
    # terminal pre-liquidation 90.65, liquidation fee on the STALE
    # holding (1.96 on 98) -> 88.69. New kernel: exact solve (entry
    # fee 1.9608, holding 98.0392), terminal 90.6863, liquidation fee
    # on the true terminal value (1.8137) -> 88.8725.
    S = torch.tensor([[100.0, 92.5]])
    old = _legacy_rollout_old_approximation(S, lambda i, V: torch.ones_like(V),
                                            cost_rate=0.02)
    new = rollout(S, _const_weight(1.0), cost_rate=0.02)
    assert float(old[0]) == pytest.approx(88.69, abs=1e-3)
    assert float(new[0]) == pytest.approx(88.8725, abs=1e-3)
    assert float(new[0]) - float(old[0]) == pytest.approx(0.1825, abs=1e-3)


def test_stage3_neural_cppi_1_economics_legacy_historical_regression():
    # Same real SPY returns, same bootstrap seeds, same strategies:
    # old formula vs current kernel. Funding is identical on both
    # sides (borrow at r by default), so every difference is the
    # exact fee solve. All-cash must be bit-identical (no trades).
    # N=2000 keeps the test fast; the full table is in the CHANGELOG.
    data = load_spy_daily()
    returns = log_returns(data["adjclose"])
    table = {}
    for horizon in (30, 63):
        S_np = block_bootstrap_paths(returns, 2000, horizon, 21, S0=V0,
                                     seed=7 + 999)
        S = torch.as_tensor(S_np, dtype=torch.float32)
        for name, wfun, old_wfun in (
            ("classic_m3", lambda i, V, t, c, f: classic_cppi_weight(
                V, multiplier=3.0), lambda i, V: classic_cppi_weight(
                V, multiplier=3.0)),
            ("classic_m4", lambda i, V, t, c, f: classic_cppi_weight(
                V, multiplier=4.0), lambda i, V: classic_cppi_weight(
                V, multiplier=4.0)),
            ("classic_m5", lambda i, V, t, c, f: classic_cppi_weight(
                V, multiplier=5.0), lambda i, V: classic_cppi_weight(
                V, multiplier=5.0)),
            ("buy_hold", _const_weight(1.0),
             lambda i, V: torch.ones_like(V)),
            ("all_cash", _const_weight(0.0),
             lambda i, V: torch.zeros_like(V)),
        ):
            new = rollout(S, wfun)
            old = _legacy_rollout_old_approximation(S, old_wfun)
            table[(horizon, name)] = (float(old.mean()),
                                      float(new.mean()))
    # All-cash identical on both sides (no trades, no fees).
    for horizon in (30, 63):
        old_mean, new_mean = table[(horizon, "all_cash")]
        assert old_mean == pytest.approx(new_mean, abs=1e-6)
        assert old_mean == pytest.approx(100.0 * math.exp(R * horizon * DT),
                                         rel=1e-4)
    # Every traded strategy differs: the exact fee solve moves mean
    # wealth (direction varies by strategy; the table pins it).
    for horizon in (30, 63):
        for name in ("classic_m3", "classic_m4", "classic_m5", "buy_hold"):
            old_mean, new_mean = table[(horizon, name)]
            assert abs(new_mean - old_mean) > 1e-6, (horizon, name)
    # Pinned table values (mean terminal wealth; tolerance absorbs
    # float32/BLAS wobble but not a change of accounting). At the
    # repo's 20bp cost the exact fee solve moves means by at most
    # 3e-4; the material legacy gap shows at the 2% cost of the
    # pinned analytic case above (+0.1825).
    pinned = {
        (30, "classic_m3"): (100.4112, 100.4112),
        (30, "classic_m4"): (100.4355, 100.4355),
        (30, "classic_m5"): (100.4465, 100.4466),
        (30, "buy_hold"): (100.9946, 100.9949),
        (63, "classic_m3"): (101.0017, 101.0017),
        (63, "classic_m4"): (101.0891, 101.0892),
        (63, "classic_m5"): (101.1441, 101.1442),
        (63, "buy_hold"): (102.5691, 102.5694),
    }
    for key, (old_pin, new_pin) in pinned.items():
        old_mean, new_mean = table[key]
        assert old_mean == pytest.approx(old_pin, abs=2e-3)
        assert new_mean == pytest.approx(new_pin, abs=2e-3)
    # Record the measured table on stdout for the CHANGELOG packet.
    for (horizon, name), (old_mean, new_mean) in sorted(table.items()):
        print(f"E1-legacy {horizon}d {name}: old={old_mean:.4f} "
              f"new={new_mean:.4f} diff={new_mean - old_mean:+.4f}")
