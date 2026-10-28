"""Rule-based comparator weight functions for the CPPI ladder.

Every builder returns a weight function with the signature the
single-asset ``rollout`` calls: ``fn(i, V, tau_frac, cushion, feats)``
with ``feats`` the strictly causal crash-feature dict. All rules take
their floor, cap, and costs from the caller's conventions object, so
no rung can quietly run under a different cap or fee than the learned
policies it is compared against.

The shipped emergency brake is ``emergency_vol=0.35`` with the
multiplier halved while triggered, matching ``crash_ablation.py``.
The brake rung here applies that exact rule to a classic CPPI
proposal, so a learned policy running under the brake is compared
against a classic rule running under the identical brake, not against
an unbraked classic.
"""

import torch

from dhl.experiments.cppi_learned import (
    CAP,
    FLOOR,
    _learned_weight_fn,
    classic_cppi_weight,
    cppi_constrained_weight_fn,
)

SHIPPED_EMERGENCY_VOL = 0.35
SHIPPED_EMERGENCY_MULTIPLIER = None  # None means half the base multiplier


def _cushion_amount(V, floor):
    return torch.clamp(V - floor, min=0.0)


def cash_weight_fn():
    """All cash: weight zero at every step, no trades, no fees."""

    def fn(i, V, tau_frac, cushion, feats):
        return torch.zeros_like(V)

    return fn


def buy_hold_weight_fn():
    """Buy and hold with exact costs: fully invested, then no trades.

    The weight function returns 1.0 at every step. That is a true
    buy/hold, not daily rebalancing to 100%: after the entry trade
    cash is exactly zero, so the risky weight stays 1.0 by itself and
    every later rebalance trades zero notional at zero fee. Entry and
    terminal-liquidation costs are still charged exactly by the kernel.
    """

    def fn(i, V, tau_frac, cushion, feats):
        return torch.ones_like(V)

    return fn


def classic_weight_fn(multiplier=4.0, floor=FLOOR, cap=CAP):
    """Plain classic CPPI: multiplier * cushion / wealth, capped."""

    def fn(i, V, tau_frac, cushion, feats):
        return classic_cppi_weight(V, floor=floor, multiplier=multiplier,
                                   cap=cap)

    return fn


def brake_weight_fn(multiplier=4.0, floor=FLOOR, cap=CAP,
                    emergency_vol=SHIPPED_EMERGENCY_VOL,
                    emergency_multiplier=SHIPPED_EMERGENCY_MULTIPLIER):
    """Classic CPPI under the identical shipped emergency brake.

    When the 21d realized vol in ``feats`` exceeds ``emergency_vol``,
    the multiplier tightens to ``emergency_multiplier`` (half the base
    multiplier when None), exactly as ``cppi_constrained_weight_fn``
    does for a learned proposal. A missing vol feature never triggers
    the brake, which degrades this rung to plain classic CPPI.
    """

    def fn(i, V, tau_frac, cushion, feats):
        vol = feats.get("vol", torch.zeros_like(V))
        m_eff = torch.where(
            vol > emergency_vol,
            torch.full_like(
                V, emergency_multiplier
                if emergency_multiplier is not None
                else multiplier * 0.5),
            torch.full_like(V, multiplier),
        )
        w = m_eff * _cushion_amount(V, floor) / V.clamp_min(1e-8)
        return torch.clamp(w, max=cap)

    return fn


def dynamic_vol_weight_fn(multiplier=4.0, floor=FLOOR, cap=CAP,
                          vol_target=0.20, m_min=1.0, m_max=8.0):
    """Dynamic-volatility CPPI: multiplier scales with target vol.

    ``m_eff = clip(multiplier * vol_target / vol, m_min, m_max)``.
    Calm markets run a higher multiplier, violent markets a lower
    one. A zero vol reading clips to ``m_max`` rather than dividing
    by zero. A missing vol feature degrades to the base multiplier.
    """

    def fn(i, V, tau_frac, cushion, feats):
        vol = feats.get("vol", torch.full_like(V, vol_target))
        m_eff = torch.clamp(multiplier * vol_target / vol.clamp_min(1e-8),
                            min=m_min, max=m_max)
        w = m_eff * _cushion_amount(V, floor) / V.clamp_min(1e-8)
        return torch.clamp(w, max=cap)

    return fn


def budget_fraction_weight_fn(multiplier=4.0, floor=FLOOR, cap=CAP,
                              fraction=1.0):
    """Constant fraction of the allowable CPPI budget.

    ``fraction=1.0`` reproduces classic CPPI exactly (the parity
    rung); smaller fractions hold a fixed share of what the budget
    allows. ``fraction`` must lie in [0, 1]: holding more than the
    budget allows is a different strategy, not a fraction of it.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"budget fraction must be in [0, 1], got {fraction}")

    def fn(i, V, tau_frac, cushion, feats):
        w = classic_cppi_weight(V, floor=floor, multiplier=multiplier,
                                cap=cap)
        return torch.clamp(fraction * w, max=cap)

    return fn


def fixed_cap_weight_fn(weight=0.5, cap=CAP):
    """Fixed-cap exposure: a constant risky weight, cushion aside."""
    if not 0.0 <= weight <= cap:
        raise ValueError(
            f"fixed weight {weight} must lie in [0, {cap}]")

    def fn(i, V, tau_frac, cushion, feats):
        return torch.full_like(V, weight)

    return fn


def trend_vol_weight_fn(multiplier=4.0, floor=FLOOR, cap=CAP):
    """Simple trend-plus-vol CPPI: classic budget times a regime gate.

    gate = clip(1 - drawdown, 0.25, 1.0) * clip(1 / vol_ratio, 0.5, 1.0).
    In a calm market (no drawdown, vol_ratio 1) the gate is 1 and the
    rule equals classic CPPI. In a falling market with accelerating
    vol the gate shrinks toward 0.125, cutting exposure before the
    cushion alone would. Missing features degrade to classic CPPI.
    """

    def fn(i, V, tau_frac, cushion, feats):
        drawdown = feats.get("drawdown", torch.zeros_like(V))
        vol_ratio = feats.get("vol_ratio", torch.ones_like(V))
        gate = (torch.clamp(1.0 - drawdown, min=0.25, max=1.0)
                * torch.clamp(1.0 / vol_ratio.clamp_min(1e-8),
                              min=0.5, max=1.0))
        w = classic_cppi_weight(V, floor=floor, multiplier=multiplier,
                                cap=cap)
        return torch.clamp(gate * w, max=cap)

    return fn


def learned_weight_fns(policy, multiplier=4.0, floor=FLOOR, cap=CAP,
                       emergency_vol=SHIPPED_EMERGENCY_VOL,
                       emergency_multiplier=SHIPPED_EMERGENCY_MULTIPLIER):
    """All learned-policy rungs: raw, CPPI-constrained, constrained+brake.

    The raw rung is the network's own proposal. The constrained rung
    scales the proposal down to the CPPI budget (never up). The brake
    rung adds the identical shipped emergency brake on top. Comparing
    the three separates what the network adds from what the risk layer
    and the brake add.
    """
    raw = _learned_weight_fn(policy)
    constrained = cppi_constrained_weight_fn(
        raw, multiplier=multiplier, cap=cap, floor=floor)
    constrained_brake = cppi_constrained_weight_fn(
        raw, multiplier=multiplier, cap=cap, floor=floor,
        emergency_vol=emergency_vol,
        emergency_multiplier=emergency_multiplier)
    return {
        "learned": raw,
        "learned_constrained": constrained,
        "learned_constrained_brake": constrained_brake,
    }
