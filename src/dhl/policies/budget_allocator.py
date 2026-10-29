"""Budget-utilization policy: the source contract (R1).

The current learned allocator requests an absolute risky weight and a
protective layer may then discard part of the request, which both
underuses safe available budget and wastes gradient above the cap.
This module states the replacement contract: the policy chooses the
fraction of the available risk budget to deploy, so the constraint is
part of the action itself.

Definitions (per rebalance step, per path):

    B_t = min(cap, m_t * max(V_t - F, 0) / V_t)   (available budget)
    w_t = B_t * u                                  (executed weight)

``u`` is the budget fraction, in [0, 1]. With the same ``m_t``,
``u = 1`` is exactly classical CPPI (the budget is the classical
weight) and ``u = 0`` is all cash. Emergency-volatility tightening
belongs in ``B_t`` through ``m_t``: it shrinks the budget the policy
sees, and is never a silent override applied after the policy acts.

Part 1 scope: this contract, deterministic fraction controllers, the
bounded fraction network definition, an independent scalar oracle,
and baseline capture over seeded scenarios. No training run, no
policy promotion, no performance claim; training and experiments are
Part 2. The network defined here is the untrained contract object:
sigmoid-bounded fraction, residual form ``u = 1 - sigmoid(h)`` (CPPI
is the default and learning is the de-risking residual) or direct
form ``u = sigmoid(h)``, logits clamped against saturation, and a
zero final layer so the initial fraction is state-independent and
close to CPPI without a saturated derivative.

Part 2 scope: the declared scenario protocol, the deterministic
trainer for the fraction network, validation-based selection of the
de-risking penalty, and the general controller evaluator at the end
of this module (``train_budget_fraction_net``,
``select_derisk_penalty``, ``evaluate_fraction_controllers``,
``budget_experiment_scenarios``), with the experiment runs in
scripts/experiments/budget_utilization.py.

Execution goes through the existing single-asset ``rollout`` in
``dhl.experiments.cppi_learned``, which accounts every trade with the
shared kernel in ``dhl.backtest.execution`` (the proposed
``src/dhl/portfolio/execution.py`` path does not exist; the kernel
lives in the backtest package). The oracle below shares no code with
that kernel: it solves the self-financing rebalance equation by
bisection in plain float64 Python, so agreement between the two is an
independent check, not the same formula twice.
"""

import math

import numpy as np
import torch
from torch import nn

from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    DT,
    FEATURE_NAMES,
    FLOOR,
    FLOOR_PENALTY,
    R,
    V0,
    classic_cppi_weight,
    rollout,
)
from dhl.markets.gbm import simulate_gbm

LOGIT_CLIP = 6.0
FRACTION_TOL = 1e-12


# --- the budget ------------------------------------------------------------


def risk_budget(V, floor=FLOOR, multiplier=4.0, cap=CAP, m_t=None):
    """Available risk budget B_t = min(cap, m_t * max(V - F, 0) / V).

    ``m_t`` overrides ``multiplier`` when given (scalar or tensor
    broadcastable to ``V``); that is where any state-dependent
    tightening lives. With a constant multiplier this is exactly
    ``classic_cppi_weight`` at the same multiplier, computed with the
    same operations, so fraction 1 reproduces classical CPPI bit for
    bit. Works on floats or torch tensors.
    """
    m = m_t if m_t is not None else multiplier
    if torch.is_tensor(V):
        cushion = torch.clamp(V - floor, min=0.0)
        w = m * cushion / V.clamp_min(1e-8)
        return torch.clamp(w, max=cap)
    cushion = max(float(V) - floor, 0.0)
    return min(float(m) * cushion / max(float(V), 1e-8), cap)


def validate_fraction(u, tol=FRACTION_TOL):
    """Fail closed on a budget fraction outside [0, 1].

    A fraction outside [0, 1] is a different strategy, not a fraction
    of the budget, so it raises instead of being clamped into one.
    Values within ``tol`` of the boundary are clamped to it.
    """
    if torch.is_tensor(u):
        if bool((u < -tol).any()) or bool((u > 1.0 + tol).any()):
            raise ValueError(
                "budget fraction must lie in [0, 1], got range "
                f"[{float(u.min()):.6g}, {float(u.max()):.6g}]")
        return torch.clamp(u, min=0.0, max=1.0)
    u = float(u)
    if u < -tol or u > 1.0 + tol:
        raise ValueError(f"budget fraction must lie in [0, 1], got {u}")
    return min(max(u, 0.0), 1.0)


# --- deterministic fraction controllers ------------------------------------
#
# A fraction function has the rollout weight-function signature
# ``fn(i, V, tau_frac, cushion, feats)`` and returns the fraction u.
# ``cushion`` is the rollout cushion ratio, clamped to [-1, 1].


def constant_fraction(u0):
    """Constant budget fraction (u = 1 is classic CPPI, u = 0 cash)."""
    if not 0.0 <= u0 <= 1.0:
        raise ValueError(f"budget fraction must lie in [0, 1], got {u0}")

    def fn(i, V, tau_frac, cushion, feats):
        return torch.full_like(V, float(u0))

    return fn


def vol_threshold_fraction(threshold=0.35, high=1.0, low=0.25):
    """Full budget while 21d vol is calm, ``low`` fraction above it.

    Deterministic controller: the fraction steps down once the causal
    vol feature crosses ``threshold``. A missing vol feature raises,
    because a vol-gated fraction with no vol reading is not a rule.
    """
    if threshold <= 0.0:
        raise ValueError(f"vol threshold must be positive, got {threshold}")
    for name, frac in (("high", high), ("low", low)):
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"{name} fraction must lie in [0, 1], "
                             f"got {frac}")

    def fn(i, V, tau_frac, cushion, feats):
        if "vol" not in feats:
            raise ValueError("vol_threshold_fraction requires the "
                             "'vol' feature")
        vol = feats["vol"]
        return torch.where(vol > threshold,
                           torch.full_like(V, float(low)),
                           torch.full_like(V, float(high)))

    return fn


def cushion_ramp_fraction(full_at=0.10):
    """Fraction ramps linearly with the cushion ratio.

    u = clamp(cushion_ratio / full_at, 0, 1): full budget once the
    cushion ratio reaches ``full_at``, no budget at or below the
    floor. A deterministic de-risking schedule driven only by the
    ex-ante cushion state.
    """
    if full_at <= 0.0:
        raise ValueError(f"full_at must be positive, got {full_at}")

    def fn(i, V, tau_frac, cushion, feats):
        return torch.clamp(cushion / full_at, min=0.0, max=1.0)

    return fn


# --- the bounded fraction network (contract definition) ---------------------


class BudgetFractionNet(nn.Module):
    """Bounded budget-fraction policy: state -> u in (0, 1).

    Inputs mirror ``CPPIPolicy``: (time_left_frac, cushion_ratio,
    *feature_names) with strictly causal features. The head is a
    sigmoid, so the fraction is in (0, 1) by construction. Two forms:

    - ``form="residual"``: u = 1 - sigmoid(h). CPPI (u near 1) is the
      default and the network learns only the de-risking residual.
    - ``form="direct"``: u = sigmoid(h).

    Logits are clamped to +/- ``logit_clip`` before the sigmoid, so no
    state can saturate the action. The final layer starts at zero
    weights with bias set so the initial fraction is
    sigmoid(``init_bias``) (0.8808 at the default 2.0) for every
    state: close to CPPI, with a healthy sigmoid derivative, and
    state-independent until training makes it otherwise.
    """

    def __init__(self, hidden_dims=(16, 16), form="residual",
                 logit_clip=LOGIT_CLIP, init_bias=2.0,
                 feature_names=("vol",)):
        super().__init__()
        if form not in ("residual", "direct"):
            raise ValueError(f"unknown fraction form {form!r}")
        if logit_clip <= 0.0:
            raise ValueError(f"logit_clip must be positive, "
                             f"got {logit_clip}")
        self.form = form
        self.logit_clip = float(logit_clip)
        self.feature_names = tuple(feature_names)
        for name in self.feature_names:
            if name not in FEATURE_NAMES:
                raise ValueError(f"unknown feature {name!r}")
        layers = []
        dim = 2 + len(self.feature_names)
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.Tanh()]
            dim = h
        final = nn.Linear(dim, 1)
        with torch.no_grad():
            final.weight.zero_()
            final.bias.fill_(float(init_bias) if form == "direct"
                             else -float(init_bias))
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, tau_frac, cushion_ratio, feats):
        cushion_ratio = torch.as_tensor(cushion_ratio,
                                        dtype=torch.float32)
        cols = [
            torch.as_tensor(tau_frac, dtype=torch.float32)
            .expand_as(cushion_ratio),
            cushion_ratio,
        ]
        for name in self.feature_names:
            cols.append(torch.as_tensor(feats[name], dtype=torch.float32))
        x = torch.stack(cols, dim=1)
        h = self.net(x).squeeze(1)
        h = torch.clamp(h, min=-self.logit_clip, max=self.logit_clip)
        s = torch.sigmoid(h)
        return s if self.form == "direct" else 1.0 - s


def net_fraction_fn(net):
    """Adapt a ``BudgetFractionNet`` to the fraction-function shape."""

    def fn(i, V, tau_frac, cushion, feats):
        u = net(tau_frac, cushion, feats)
        return u.to(dtype=V.dtype) if torch.is_tensor(u) else u

    return fn


# --- the executed weight -----------------------------------------------------


def budget_weight_fn(fraction_fn, multiplier=4.0, floor=FLOOR, cap=CAP,
                     multiplier_fn=None):
    """Rollout weight function executing w_t = B_t * u.

    ``fraction_fn`` supplies u (validated fail-closed into [0, 1]).
    ``multiplier_fn(i, V, feats)``, when given, supplies the per-step
    multiplier m_t, so state-dependent tightening (for example the
    shipped emergency brake, halving m while vol exceeds 0.35) is
    expressed inside the budget and reproduces the corresponding
    classic rung exactly at u = 1.
    """

    def fn(i, V, tau_frac, cushion, feats):
        u = validate_fraction(fraction_fn(i, V, tau_frac, cushion, feats))
        if multiplier_fn is not None:
            B = risk_budget(V, floor=floor, cap=cap,
                            m_t=multiplier_fn(i, V, feats))
        else:
            B = risk_budget(V, floor=floor, multiplier=multiplier,
                            cap=cap)
        return B * u

    return fn


# --- independent scalar oracle -----------------------------------------------


def _solve_holding_bisect(wealth, holding, weight, cost_rate):
    """Solve H' = w * (V - c * |H' - H|) for H' by bisection.

    The right side minus H' is strictly decreasing in H' (its slope
    lies in [-1 - w c, -1 + w c] with w c < 1), positive at H' = 0 and
    nonpositive at H' = w V, so 200 bisection steps pin the root far
    below float64 resolution. This shares no code with the execution
    kernel's closed form, by design.
    """
    if weight == 0.0:
        return 0.0
    lo, hi = 0.0, weight * wealth
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if weight * (wealth - cost_rate * abs(mid - holding)) > mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def oracle_budget_rollout(prices, fraction, multiplier=4.0, floor=FLOOR,
                          cap=CAP, r=R, dt=DT, cost_rate=COST_RATE,
                          v0=V0):
    """Independent float64 reference for one path under w = B * u.

    ``prices`` is a 1-D sequence of length steps + 1. ``fraction`` is
    a constant in [0, 1] or a callable ``fraction(i, wealth)``. The
    accounting mirrors the rollout conventions exactly (target weight
    on post-fee wealth, fee on the actual trade, cash accrues at
    exp(r dt), terminal liquidation charged on the true terminal
    holding) but is computed step by step in plain Python floats with
    a bisection rebalance solve, independently of the torch kernel.

    Returns a dict with terminal wealth, total fees, the minimum
    wealth mark (post-move marks, the last one post-liquidation), an
    anytime-breach flag, and per-step holding_post / fee arrays.
    """
    prices = [float(p) for p in prices]
    if len(prices) < 2:
        raise ValueError("oracle needs at least two prices")
    steps = len(prices) - 1
    cash = float(v0)
    units = 0.0
    holding_post = []
    fees = []
    marks = [float(v0)]
    total_fees = 0.0
    for i in range(steps):
        price = prices[i]
        holding = units * price
        wealth = cash + holding
        if wealth <= 0.0:
            raise ValueError(
                f"oracle step {i}: nonpositive pre-trade wealth "
                f"{wealth:.6g}")
        u = fraction(i, wealth) if callable(fraction) else float(fraction)
        u = validate_fraction(u)
        w = risk_budget(wealth, floor=floor, multiplier=multiplier,
                        cap=cap) * u
        post = _solve_holding_bisect(wealth, holding, w, cost_rate)
        fee = cost_rate * abs(post - holding)
        cash = cash - (post - holding) - fee
        units = post / price
        holding_post.append(post)
        fees.append(fee)
        total_fees += fee
        cash *= math.exp(r * dt)
        marks.append(cash + units * prices[i + 1])
    holding_T = units * prices[steps]
    liq_fee = cost_rate * abs(holding_T)
    total_fees += liq_fee
    terminal = cash + holding_T - liq_fee
    marks[-1] = terminal
    return {
        "terminal_wealth": terminal,
        "total_fees": total_fees,
        "min_wealth": min(marks),
        "breach_anytime": min(marks) < floor,
        "holding_post": holding_post,
        "fee": fees,
    }


# --- ledger accounting audit ---------------------------------------------------


def ledger_self_financing_violations(ledger, r=R, dt=DT, tol=1e-9):
    """Count self-financing identity violations in a rollout ledger.

    Four identities, checked on every path and step: (1) the fee
    identity cash_post + holding_post + fee == wealth_pre; (2) the
    inter-step growth identity wealth_pre[i + 1] == cash_post[i]
    grown at exp(r dt) plus units_post[i] marked at price[i + 1]
    (default funding: borrowing at the cash rate, as in the
    rollout); (3) the liquidation identity terminal_wealth +
    liquidation_fee == wealth_pre_liquidation; (4) cum_fees == sum of
    step fees + liquidation fee. Returns the total violation count;
    a correct rollout scores 0.
    """
    fee = ledger["fee"]
    if fee.shape[1] == 0:
        return 0
    violations = 0
    ident = (ledger["cash_post"] + ledger["holding_post"] + fee
             - ledger["wealth_pre"]).abs()
    violations += int((ident > tol).sum())
    growth = (ledger["cash_post"][:, :-1] * math.exp(r * dt)
              + ledger["units_post"][:, :-1] * ledger["price"][:, 1:])
    violations += int(((growth - ledger["wealth_pre"][:, 1:]).abs()
                       > tol).sum())
    liq = (ledger["terminal_wealth"] + ledger["liquidation_fee"]
           - ledger["wealth_pre_liquidation"]).abs()
    violations += int((liq > tol).sum())
    cum = (ledger["cum_fees"] - fee.sum(dim=1)
           - ledger["liquidation_fee"]).abs()
    violations += int((cum > tol).sum())
    return violations


# --- baseline capture -----------------------------------------------------------


def deterministic_controllers():
    """The deterministic risk-budget controllers of the baseline."""
    return {
        "cash": constant_fraction(0.0),
        "classic": constant_fraction(1.0),
        "constant_0.5": constant_fraction(0.5),
        "constant_0.25": constant_fraction(0.25),
        "vol_threshold": vol_threshold_fraction(),
        "cushion_ramp": cushion_ramp_fraction(),
    }


def baseline_report(scenarios, multiplier=4.0, floor=FLOOR, cap=CAP,
                    cost_rate=COST_RATE):
    """Baseline capture: deterministic controllers on seeded scenarios.

    ``scenarios`` is a list of (seed, S) pairs with S a float64 price
    tensor of shape (n_paths, steps + 1). Every controller runs
    through the shared rollout; the report gives, per controller,
    mean terminal wealth, mean return (terminal / V0 - 1), mean
    terminal shortfall below the floor, terminal and anytime breach
    counts, mean fees, and the observed fraction range. The report
    also carries the u = 1 parity check against the classic CPPI
    reference (per-step holdings and fees on the first scenario), the
    independent-oracle agreement for constant fractions on the first
    scenario's first path, and the total self-financing violation
    count across every run. Numbers are reported as measured; no
    controller is promoted over another here.
    """
    controllers = deterministic_controllers()
    agg = {name: {"wealth": [], "shortfall": [], "fees": [],
                  "breach_terminal": 0, "breach_anytime": 0,
                  "fraction_min": 1.0, "fraction_max": 0.0}
           for name in controllers}
    violations = 0
    for seed, S in scenarios:
        for name, frac_fn in controllers.items():
            seen = []

            def recording(i, V, tau_frac, cushion, feats,
                          _fn=frac_fn, _seen=seen):
                u = _fn(i, V, tau_frac, cushion, feats)
                _seen.append((float(u.min()), float(u.max())))
                return u

            V_T, path, ledger = rollout(
                S, budget_weight_fn(recording, multiplier=multiplier,
                                    floor=floor, cap=cap),
                floor=floor, cost_rate=cost_rate, cap=cap,
                return_path=True, return_ledger=True)
            a = agg[name]
            a["wealth"].append(V_T)
            a["shortfall"].append(
                torch.clamp(floor - V_T, min=0.0))
            a["fees"].append(ledger["cum_fees"])
            a["breach_terminal"] += int((V_T < floor).sum())
            a["breach_anytime"] += int(
                (path.min(dim=1).values < floor).sum())
            a["fraction_min"] = min(a["fraction_min"],
                                    min(x for x, _ in seen))
            a["fraction_max"] = max(a["fraction_max"],
                                    max(y for _, y in seen))
            violations += ledger_self_financing_violations(ledger)

    rows = []
    for name in controllers:
        a = agg[name]
        wealth = torch.cat(a["wealth"])
        shortfall = torch.cat(a["shortfall"])
        fees = torch.cat(a["fees"])
        rows.append({
            "controller": name,
            "mean_terminal_wealth": float(wealth.mean()),
            "mean_return": float(wealth.mean()) / V0 - 1.0,
            "mean_shortfall": float(shortfall.mean()),
            "breach_count_terminal": a["breach_terminal"],
            "breach_count_anytime": a["breach_anytime"],
            "mean_fees": float(fees.mean()),
            "fraction_min": a["fraction_min"],
            "fraction_max": a["fraction_max"],
            "n_paths": int(wealth.shape[0]),
        })

    # Parity: u = 1 through the budget weight function vs the classic
    # reference weight on the first scenario.
    _, S0 = scenarios[0]
    _, _, led_budget = rollout(
        S0, budget_weight_fn(constant_fraction(1.0),
                             multiplier=multiplier, floor=floor,
                             cap=cap),
        floor=floor, cost_rate=cost_rate, cap=cap,
        return_path=True, return_ledger=True)
    classic_w = classic_cppi_weight  # reference formula, applied below

    def classic_fn(i, V, tau_frac, cushion, feats):
        return classic_w(V, floor=floor, multiplier=multiplier, cap=cap)

    _, _, led_classic = rollout(
        S0, classic_fn, floor=floor, cost_rate=cost_rate, cap=cap,
        return_path=True, return_ledger=True)
    parity = {
        "max_holding_diff": float(
            (led_budget["holding_post"]
             - led_classic["holding_post"]).abs().max()),
        "max_fee_diff": float(
            (led_budget["fee"] - led_classic["fee"]).abs().max()),
        "max_cum_fee_diff": float(
            (led_budget["cum_fees"]
             - led_classic["cum_fees"]).abs().max()),
    }

    # Oracle agreement on the first scenario, first path.
    prices = S0[0].tolist()
    oracle_diffs = []
    for u in (0.0, 0.5, 1.0):
        V_T, _, _ = rollout(
            S0[:1], budget_weight_fn(constant_fraction(u),
                                     multiplier=multiplier, floor=floor,
                                     cap=cap),
            floor=floor, cost_rate=cost_rate, cap=cap,
            return_path=True, return_ledger=True)
        oracle = oracle_budget_rollout(
            prices, u, multiplier=multiplier, floor=floor, cap=cap,
            cost_rate=cost_rate)
        oracle_diffs.append(abs(oracle["terminal_wealth"]
                                - float(V_T[0])))

    return {
        "n_scenarios": len(scenarios),
        "seeds": [int(seed) for seed, _ in scenarios],
        "multiplier": multiplier,
        "floor": floor,
        "cap": cap,
        "cost_rate": cost_rate,
        "controllers": rows,
        "parity": parity,
        "oracle": {"max_terminal_diff": max(oracle_diffs)},
        "self_financing_violations": violations,
    }


# --- scenario protocol (Part 2) ----------------------------------------------
#
# Declared scenario families for the utilization experiments.  Every
# family is GBM (dhl.markets.gbm.simulate_gbm), 20 scenarios of
# 16 paths x 63 steps unless a caller overrides the shape, evaluated
# at multiplier 4, floor 90, cap 1.5, 20bp costs.  The five seed sets
# are pairwise disjoint: the baseline reproduces the Part 1 capture
# for comparability, training and validation carry the
# penalty-selection protocol, and the stress and gap families are
# held out from both training and selection.
#
# Training and validation mix three regimes by position in the seed
# list (position modulo 3): calm, stress, gap.  The gap regime is a
# calm GBM path with one multiplicative jump of GAP_JUMP per path at
# a seeded step.  Rationale: under daily rebalancing a GBM-only
# stress family (sigma 0.55, drift -0.25) produces zero shortfall
# and zero breaches for every controller, classic CPPI included,
# because the cushion-proportional rule de-risks smoothly as wealth
# falls, and a breach needs a one-step drop larger than 1 / m (25%
# at m = 4).  The jump step is seeded per path and is not knowable
# from the ex-ante state, so the gap family measures the exposure
# tradeoff rather than forecasting skill.

CALM_SIGMA = 0.30
STRESS_SIGMA = 0.55
STRESS_DRIFT = -0.25
GAP_JUMP = -0.30

SEED_SETS = {
    "baseline": tuple(1000 + k for k in range(20)),
    "training": tuple(2000 + k for k in range(20)),
    "validation": tuple(3000 + k for k in range(20)),
    "stress": tuple(4000 + k for k in range(20)),
    "gap": tuple(5000 + k for k in range(20)),
}


def make_gbm_scenarios(seeds, n_paths=16, n_steps=63, sigma=CALM_SIGMA,
                       drift=R, dtype=torch.float64):
    """Seeded GBM scenarios as (seed, price tensor) pairs."""
    scenarios = []
    for seed in seeds:
        S = simulate_gbm(n_paths, n_steps, n_steps * DT, 100.0, drift,
                         sigma, seed=int(seed))
        scenarios.append((int(seed), torch.tensor(S, dtype=dtype)))
    return scenarios


def gap_augment(S, jump=GAP_JUMP, seed=0):
    """Copy of price tensor S with one multiplicative jump per path.

    Each path p gets one jump step j_p drawn uniformly from the
    middle two thirds of the path by a generator seeded with
    ``seed``; prices from j_p + 1 onward are scaled by ``1 + jump``,
    so the one-step return into j_p + 1 carries the jump.
    Deterministic in ``seed``.
    """
    if not -1.0 < jump < 0.0:
        raise ValueError(f"jump must lie in (-1, 0), got {jump}")
    out = S.clone()
    n_paths, n_prices = out.shape
    steps = n_prices - 1
    lo = max(1, steps // 6)
    hi = max(lo, (5 * steps) // 6)
    rng = np.random.default_rng(int(seed))
    for p in range(n_paths):
        j = int(rng.integers(lo, hi + 1))
        out[p, j + 1:] *= (1.0 + jump)
    return out


def _regime_scenario(seed, regime, n_paths, n_steps, dtype):
    if regime == 1:
        return make_gbm_scenarios([seed], n_paths, n_steps,
                                  sigma=STRESS_SIGMA,
                                  drift=STRESS_DRIFT, dtype=dtype)
    calm = make_gbm_scenarios([seed], n_paths, n_steps,
                              sigma=CALM_SIGMA, drift=R, dtype=dtype)
    if regime == 2:
        return [(seed, gap_augment(calm[0][1], seed=seed))]
    return calm


def budget_experiment_scenarios(name, n_paths=16, n_steps=63,
                                dtype=torch.float64):
    """One declared scenario family by name (see ``SEED_SETS``)."""
    if name not in SEED_SETS:
        raise ValueError(f"unknown scenario set {name!r}; expected "
                         f"one of {sorted(SEED_SETS)}")
    seeds = SEED_SETS[name]
    if name == "baseline":
        return make_gbm_scenarios(seeds, n_paths, n_steps,
                                  sigma=CALM_SIGMA, drift=R,
                                  dtype=dtype)
    if name == "stress":
        return make_gbm_scenarios(seeds, n_paths, n_steps,
                                  sigma=STRESS_SIGMA,
                                  drift=STRESS_DRIFT, dtype=dtype)
    if name == "gap":
        return [(seed, gap_augment(S, seed=seed))
                for seed, S in make_gbm_scenarios(
                    seeds, n_paths, n_steps, sigma=CALM_SIGMA,
                    drift=R, dtype=dtype)]
    scenarios = []
    for pos, seed in enumerate(seeds):
        scenarios += _regime_scenario(seed, pos % 3, n_paths,
                                      n_steps, dtype)
    return scenarios


# --- general controller evaluation (Part 2) -----------------------------------


def evaluate_fraction_controllers(scenarios, controllers,
                                  multiplier=4.0, floor=FLOOR,
                                  cap=CAP, cost_rate=COST_RATE):
    """Evaluate arbitrary fraction controllers on seeded scenarios.

    The metric family of ``baseline_report`` (mean terminal wealth,
    return, shortfall, terminal and anytime breach counts, fees,
    fraction range) plus the mean fraction, the utilization measure
    of this item.  Runs under no_grad and audits every ledger with
    ``ledger_self_financing_violations``; the total violation count
    is returned alongside the rows.  ``controllers`` maps names to
    fraction functions.
    """
    agg = {}
    violations = 0
    with torch.no_grad():
        for name, fn in controllers.items():
            a = agg[name] = {
                "wealth": [], "shortfall": [], "fees": [],
                "breach_terminal": 0, "breach_anytime": 0,
                "frac_sum": 0.0, "frac_count": 0,
                "fraction_min": float("inf"),
                "fraction_max": float("-inf"),
            }

            def recording(i, V, tau_frac, cushion, feats, _fn=fn,
                          _a=a):
                u = _fn(i, V, tau_frac, cushion, feats)
                _a["frac_sum"] += float(u.sum())
                _a["frac_count"] += int(u.numel())
                _a["fraction_min"] = min(_a["fraction_min"],
                                         float(u.min()))
                _a["fraction_max"] = max(_a["fraction_max"],
                                         float(u.max()))
                return u

            for _, S in scenarios:
                V_T, path, ledger = rollout(
                    S, budget_weight_fn(recording,
                                        multiplier=multiplier,
                                        floor=floor, cap=cap),
                    floor=floor, cost_rate=cost_rate, cap=cap,
                    return_path=True, return_ledger=True)
                a["wealth"].append(V_T)
                a["shortfall"].append(
                    torch.clamp(floor - V_T, min=0.0))
                a["fees"].append(ledger["cum_fees"])
                a["breach_terminal"] += int((V_T < floor).sum())
                a["breach_anytime"] += int(
                    (path.min(dim=1).values < floor).sum())
                violations += ledger_self_financing_violations(ledger)

    rows = []
    for name in controllers:
        a = agg[name]
        wealth = torch.cat(a["wealth"])
        shortfall = torch.cat(a["shortfall"])
        fees = torch.cat(a["fees"])
        rows.append({
            "controller": name,
            "mean_terminal_wealth": float(wealth.mean()),
            "mean_return": float(wealth.mean()) / V0 - 1.0,
            "mean_shortfall": float(shortfall.mean()),
            "breach_count_terminal": a["breach_terminal"],
            "breach_count_anytime": a["breach_anytime"],
            "mean_fees": float(fees.mean()),
            "mean_fraction": a["frac_sum"] / a["frac_count"],
            "fraction_min": a["fraction_min"],
            "fraction_max": a["fraction_max"],
            "n_paths": int(wealth.shape[0]),
        })
    return {
        "n_scenarios": len(scenarios),
        "seeds": [int(seed) for seed, _ in scenarios],
        "multiplier": multiplier,
        "floor": floor,
        "cap": cap,
        "cost_rate": cost_rate,
        "controllers": rows,
        "self_financing_violations": violations,
    }


# --- training (Part 2) ---------------------------------------------------------


def train_budget_fraction_net(net, scenarios, n_epochs=100, lr=3e-3,
                              floor_penalty=FLOOR_PENALTY,
                              derisk_penalty=0.0, multiplier=4.0,
                              floor=FLOOR, cap=CAP,
                              cost_rate=COST_RATE,
                              multiplier_fn=None, seed=0):
    """Train a ``BudgetFractionNet`` through the shared rollout.

    Full-batch Adam over the concatenation of ``scenarios`` (all
    scenarios must share a step count; their combined path population
    is the training batch).  The minimized objective is

        loss = -E[V_T] + floor_penalty * E[relu(floor - V_T)]
               + derisk_penalty * E[1 - u],

    where E[1 - u] averages the unused budget fraction over every
    step and path of the batch.  The first two terms are the repo
    objective family (``train_cppi_policy`` with FLOOR_PENALTY); the
    third is the optional de-risking penalty, zero unless selected by
    ``select_derisk_penalty``.  ``multiplier_fn`` threads emergency
    tightening through ``budget_weight_fn`` exactly as at evaluation
    time, so a tightened budget is inside the training rollout, never
    silently ignored by it.  Gradients are clipped at norm 1.0, the
    repo training convention.  ``torch.manual_seed(seed)`` is set on
    entry, so a fixed seed and fixed initial parameters reproduce the
    run exactly.

    Returns the per-epoch history: entry 0 is the pre-update
    evaluation of the initial network and entry k the evaluation
    after k optimizer steps.  Each entry records the loss
    decomposition (loss, mean_wealth, mean_shortfall, mean_derisk,
    mean_fraction), so the penalty's contribution is exactly
    recoverable as derisk_penalty * mean_derisk.
    """
    if not scenarios:
        raise ValueError("training needs at least one scenario")
    if len({S.shape[1] for _, S in scenarios}) != 1:
        raise ValueError("training scenarios must share a step count")
    if n_epochs < 0:
        raise ValueError("n_epochs must be non-negative")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if floor_penalty < 0.0 or derisk_penalty < 0.0:
        raise ValueError("penalties must be non-negative")
    torch.manual_seed(seed)
    S_all = torch.cat([S for _, S in scenarios], dim=0)
    base_fn = net_fraction_fn(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def evaluate():
        sums, counts = [], []

        def tracked(i, V, tau_frac, cushion, feats):
            u = base_fn(i, V, tau_frac, cushion, feats)
            sums.append(u.sum())
            counts.append(int(u.numel()))
            return u

        V_T = rollout(S_all, budget_weight_fn(
            tracked, multiplier=multiplier, floor=floor, cap=cap,
            multiplier_fn=multiplier_fn),
            floor=floor, cost_rate=cost_rate, cap=cap)
        mean_derisk = 1.0 - sum(sums) / float(sum(counts))
        mean_shortfall = torch.clamp(floor - V_T, min=0.0).mean()
        loss = (-V_T.mean() + floor_penalty * mean_shortfall
                + derisk_penalty * mean_derisk)
        metrics = {
            "loss": float(loss.detach()),
            "mean_wealth": float(V_T.detach().mean()),
            "mean_shortfall": float(mean_shortfall.detach()),
            "mean_derisk": float(mean_derisk.detach()),
            "mean_fraction": 1.0 - float(mean_derisk.detach()),
        }
        return loss, metrics

    history = []
    loss, metrics = evaluate()
    history.append({"epoch": 0, **metrics})
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        loss, metrics = evaluate()
        history.append({"epoch": epoch, **metrics})
    return history


def select_derisk_penalty(candidates, train_scenarios,
                          validation_scenarios, form="residual",
                          n_epochs=100, lr=3e-3,
                          floor_penalty=FLOOR_PENALTY, multiplier=4.0,
                          floor=FLOOR, cap=CAP, cost_rate=COST_RATE,
                          seed=0, net_kwargs=None):
    """Select the de-risking penalty on validation scenarios.

    Each candidate penalty trains a fresh ``BudgetFractionNet`` from
    the same seeded initialization on ``train_scenarios``; the
    trained network is then scored on ``validation_scenarios`` by
    the validation objective E[V_T] - floor_penalty * E[shortfall]:
    the repo objective family without the de-risking term, which is a
    training prior, not part of the score.  The highest objective
    wins; ties break toward the smaller penalty, since candidates
    are visited in ascending order and only a strict improvement
    replaces the incumbent.  The evaluation families (baseline,
    stress, gap) play no role in selection.

    Returns a dict with the selected penalty and one table row per
    candidate carrying its validation metrics.
    """
    candidates = sorted(float(c) for c in candidates)
    if not candidates:
        raise ValueError("at least one candidate penalty is required")
    if any(c < 0.0 for c in candidates):
        raise ValueError("candidate penalties must be non-negative")
    table = []
    best = None
    for lam in candidates:
        torch.manual_seed(seed)
        net = BudgetFractionNet(form=form, **(net_kwargs or {}))
        history = train_budget_fraction_net(
            net, train_scenarios, n_epochs=n_epochs, lr=lr,
            floor_penalty=floor_penalty, derisk_penalty=lam,
            multiplier=multiplier, floor=floor, cap=cap,
            cost_rate=cost_rate, seed=seed)
        report = evaluate_fraction_controllers(
            validation_scenarios,
            {"learned": net_fraction_fn(net)},
            multiplier=multiplier, floor=floor, cap=cap,
            cost_rate=cost_rate)
        row = dict(report["controllers"][0])
        row.pop("controller")
        objective = (row["mean_terminal_wealth"]
                     - floor_penalty * row["mean_shortfall"])
        table.append({
            "derisk_penalty": lam,
            "validation_objective": objective,
            "final_train_loss": history[-1]["loss"],
            "final_train_mean_fraction": history[-1]["mean_fraction"],
            **row,
        })
        if best is None or objective > best[1]:
            best = (lam, objective)
    return {
        "selected_derisk_penalty": best[0],
        "form": form,
        "floor_penalty": floor_penalty,
        "table": table,
    }
