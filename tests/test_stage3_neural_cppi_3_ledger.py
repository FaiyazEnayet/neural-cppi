"""Stage 3 NC3 fixture: explicit holdings/cost ledger for the rollout.

Scoped to the NC3 defect: the old rollout charged each rebalance on a
pre-cost target and then held a post-cost risky amount, an approximation
to an exact proportional-cost rebalance, and it tracked no explicit
cash/units/fees ledger. These tests pin the exact behavior:

(a) the self-financing identity holds at every recorded trade;
(b) an independent scalar (non-vectorized) reference implementation,
    written here from the rebalance equation, matches the vectorized
    rollout to tight tolerance;
(c) property cases: zero return, buy/hold, cap=1, cap>1 (leverage and
    funding cost), and fee-only losses.
"""

import inspect
import math

import torch

from dhl.experiments.cppi_learned import CAP, COST_RATE, R, V0, rollout

DT = 1.0 / 252.0
TOL = 1e-9


def _const_weight(w):
    def fn(i, V, tau_frac, cushion, feats):
        return torch.full_like(V, w)

    return fn


def _seq_weight(weights):
    def fn(i, V, tau_frac, cushion, feats):
        return torch.full_like(V, weights[i])

    return fn


def _flat(steps, price=100.0):
    return torch.full((1, steps + 1), price, dtype=torch.float64)


def _close(a, b, tol=TOL):
    return bool((a - b).abs().max() <= tol)


# --- (b) independent scalar reference ------------------------------------
def _scalar_reference(prices, weights, v0, r, dt, cost, borrow_rate):
    """One path, plain floats, solving H' = w * (V - cost*|H' - H|).

    Buy case (target weight at least the current weight):
        H' = w * (V + cost*H) / (1 + w*cost)
    Sell case:
        H' = w * (V - cost*H) / (1 - w*cost)
    Cash pays for the trade and the fee; cash then earns ``r`` when
    nonnegative and ``borrow_rate`` when negative; the risky leg is
    liquidated at the terminal price for ``cost * |H_T|``.
    """
    cash = v0
    units = 0.0
    cum_fees = 0.0
    path = [v0]
    for i, w in enumerate(weights):
        price = prices[i]
        holding = units * price
        wealth = cash + holding
        omega = holding / wealth if wealth > 0 else 0.0
        if w >= omega:
            new_holding = w * (wealth + cost * holding) / (1.0 + w * cost)
        else:
            new_holding = w * (wealth - cost * holding) / (1.0 - w * cost)
        fee = cost * abs(new_holding - holding)
        cum_fees += fee
        cash = cash - (new_holding - holding) - fee
        units = new_holding / price
        rate = r if cash >= 0 else borrow_rate
        cash = cash * math.exp(rate * dt)
        path.append(cash + units * prices[i + 1])
    holding_t = units * prices[-1]
    liq_fee = cost * abs(holding_t)
    cum_fees += liq_fee
    v_t = cash + holding_t - liq_fee
    path[-1] = v_t
    return v_t, path, cum_fees


# --- (a) self-financing identity at every trade ---------------------------
def test_stage3_neural_cppi_3_self_financing_identity_every_trade():
    torch.manual_seed(0)
    steps = 15
    rets = torch.randn(3, steps, dtype=torch.float64) * 0.02
    S = 100.0 * torch.exp(
        torch.cat([torch.zeros(3, 1, dtype=torch.float64),
                   rets.cumsum(dim=1)], dim=1))
    cost = 0.01
    V_T, path, led = rollout(S, _const_weight(0.65), cost_rate=cost,
                             return_path=True, return_ledger=True)
    g = led["cash_pre"] * 0 + 1.0  # shape helper only
    assert g.shape == led["fee"].shape
    # Mark-to-market: wealth is exactly cash plus units at the price.
    assert _close(led["wealth_pre"],
                  led["cash_pre"] + led["units_pre"] * led["price"])
    # Fees are the proportional charge on the actual traded notional.
    assert _close(led["fee"], cost * led["notional"])
    assert _close(led["notional"],
                  (led["holding_post"] - led["holding_pre"]).abs())
    # The trade itself: cash pays for the holding change plus the fee.
    assert _close(led["cash_post"],
                  led["cash_pre"]
                  - (led["holding_post"] - led["holding_pre"])
                  - led["fee"])
    assert _close(led["units_post"] * led["price"], led["holding_post"])
    # The solved equation: post-trade holding is the target weight of
    # post-trade (post-fee) wealth.
    assert _close(led["holding_post"],
                  led["weight"] * (led["wealth_pre"] - led["fee"]))
    # Step linkage: units persist; cash grows at r (no borrowing here,
    # weights are below 1 and wealth stays positive).
    growth = math.exp(R * DT)
    assert _close(led["units_pre"][:, 1:], led["units_post"][:, :-1])
    assert _close(led["cash_pre"][:, 1:], led["cash_post"][:, :-1] * growth)
    assert _close(led["wealth_pre"][:, 1:],
                  led["cash_pre"][:, 1:]
                  + led["units_pre"][:, 1:] * led["price"][:, 1:])
    # Terminal liquidation is a trade too, on the true terminal holding.
    assert _close(led["liquidation_fee"],
                  cost * led["holding_terminal"].abs())
    assert _close(led["cum_fees"],
                  led["fee"].sum(dim=1) + led["liquidation_fee"])
    assert _close(led["terminal_wealth"],
                  led["wealth_pre_liquidation"] - led["liquidation_fee"])
    assert _close(led["terminal_wealth"], V_T)
    assert _close(path[:, -1], V_T)
    # Conservation: terminal wealth = initial wealth + ledger gains
    # (interest plus price moves on the recorded post-trade holdings)
    # minus every fee charged, all recomputed from ledger fields only.
    prices = torch.cat([led["price"], S[:, -1:].double()], dim=1)
    gains = (led["cash_post"] * (growth - 1.0)
             + led["units_post"] * (prices[:, 1:] - prices[:, :-1])
             ).sum(dim=1)
    assert _close(V_T, V0 + gains - led["cum_fees"])


# --- (b) scalar reference vs vectorized rollout ---------------------------
def test_stage3_neural_cppi_3_scalar_reference_matches_vectorized():
    steps = 20
    prices = [100.0]
    for i in range(steps):
        prices.append(prices[-1] * math.exp(0.01 * math.sin(i * 1.7)
                                           + (0.004 if i % 3 else -0.006)))
    weights = [0.3 + 0.07 * (i % 5) for i in range(steps)]  # buys and sells
    S = torch.tensor([prices], dtype=torch.float64)
    cost = 0.004
    V_T, path, led = rollout(S, _seq_weight(weights), cost_rate=cost,
                             return_path=True, return_ledger=True)
    v_ref, path_ref, fees_ref = _scalar_reference(
        prices, weights, V0, R, DT, cost, R)
    assert abs(float(V_T[0]) - v_ref) < 1e-9
    assert max(abs(float(p) - q) for p, q in zip(path[0], path_ref)) < 1e-9
    assert abs(float(led["cum_fees"][0]) - fees_ref) < 1e-9


# --- (c) property cases ----------------------------------------------------
def test_stage3_neural_cppi_3_zero_return_single_trade():
    # Flat prices, no interest: after the opening trade the holding is
    # already at target, so no further trade or fee may occur.
    steps = 10
    V_T, led = rollout(_flat(steps), _const_weight(0.6), cost_rate=0.01,
                       r=0.0, return_ledger=True)
    assert _close(led["notional"][:, 1:], led["notional"][:, 1:] * 0.0)
    assert _close(led["fee"][:, 1:], led["fee"][:, 1:] * 0.0)
    assert _close(led["cum_fees"],
                  led["fee"][:, 0] + led["liquidation_fee"])
    assert _close(V_T, V0 - led["cum_fees"])


def test_stage3_neural_cppi_3_buy_and_hold_no_rebalance():
    # w = 1 with a steep 5% cost: wealth is fully invested at step 0,
    # cash is exactly zero, the weight never drifts, so steps 1+ trade
    # nothing. Terminal wealth is the closed form of the two real
    # trades (initial buy, terminal liquidation).
    steps = 12
    S = torch.tensor([[100.0 * math.exp(0.01 * i) for i in range(steps + 1)]],
                     dtype=torch.float64)
    V_T, led = rollout(S, _const_weight(1.0), cost_rate=0.05, r=0.0,
                       return_ledger=True)
    assert float(led["notional"][0, 0]) > 0.0
    assert _close(led["notional"][:, 1:], led["notional"][:, 1:] * 0.0)
    assert _close(led["cash_post"], led["cash_post"] * 0.0)
    expected = (V0 / 1.05) * float(S[0, -1] / S[0, 0]) * 0.95
    assert abs(float(V_T[0]) - expected) < 1e-9 * expected


def test_stage3_neural_cppi_3_cap_one_forbids_borrowing():
    # The weight function asks for 1.5; cap=1 must clamp it: the ledger
    # never borrows (cash never negative) and the result matches the
    # scalar reference run at weight exactly 1.
    steps = 8
    S = torch.tensor([[100.0 * math.exp(0.008 * i - 0.002 * i * i / steps)
                       for i in range(steps + 1)]], dtype=torch.float64)
    V_T, led = rollout(S, _const_weight(1.5), cost_rate=COST_RATE, cap=1.0,
                       return_ledger=True)
    assert bool((led["weight"] <= 1.0 + 1e-12).all())
    assert bool((led["cash_post"] >= -1e-9).all())
    assert bool((led["cash_pre"] >= -1e-9).all())
    v_ref, _, _ = _scalar_reference(list(map(float, S[0])), [1.0] * steps,
                                    V0, R, DT, COST_RATE, R)
    assert abs(float(V_T[0]) - v_ref) < 1e-9


def test_stage3_neural_cppi_3_cap_above_one_borrows_and_funding_costs():
    # Default cap (1.5): a 1.5 weight borrows (negative cash), the
    # holding exceeds wealth, and a higher borrowing rate strictly
    # lowers terminal wealth on a flat path (cash stays negative).
    steps = 10
    S = _flat(steps)
    V_T, led = rollout(S, _const_weight(1.5), cost_rate=COST_RATE,
                       return_ledger=True)
    assert bool((led["cash_post"] < 0).all())
    assert bool((led["holding_post"]
                 > led["wealth_pre"] - led["fee"]).all())
    V_hi, _ = rollout(S, _const_weight(1.5), cost_rate=COST_RATE,
                      borrow_rate=0.20, return_ledger=True)
    assert float(V_hi[0]) < float(V_T[0])


def test_stage3_neural_cppi_3_fee_only_losses_drain_exactly_fees():
    # Flat prices and r = 0: no market gain and no interest, so wealth
    # may only fall, by exactly the cumulative fees, while an
    # alternating weight forces a real trade at every step.
    steps = 12
    weights = [0.4 if i % 2 == 0 else 0.6 for i in range(steps)]
    V_T, path, led = rollout(_flat(steps), _seq_weight(weights),
                             cost_rate=0.01, r=0.0, return_path=True,
                             return_ledger=True)
    assert bool((led["notional"] > 0).all())
    assert bool((torch.diff(path[0]) < 0).all())  # strictly draining
    assert _close(V_T, V0 - led["cum_fees"])
    assert float(led["cum_fees"][0]) > 0.0


def test_stage3_neural_cppi_3_configurable_defaults_preserve_signature():
    # The new knobs are opt-in: cap defaults to the module CAP and
    # borrow_rate defaults to None (borrow at the cash rate r), so
    # every existing positional/keyword caller keeps working.
    sig = inspect.signature(rollout)
    assert sig.parameters["cap"].default == CAP
    assert sig.parameters["borrow_rate"].default is None
    assert sig.parameters["cost_rate"].default == COST_RATE
