"""Deterministic correct-loss factorial for Stage 3 E2.

Three controlled factors vary while the training pool, epochs, batch
size, initialization protocol and seed list stay fixed:

- anytime term: ``none`` (anytime penalty 0, the terminal-only
  objective) vs ``correct`` (the P0-A corrected term, which ADDS
  pathwise floor shortfall to the minimization loss);
- feature set: ``existing`` (vol only, the original policy input) vs
  ``causal_rebound`` (the full strictly causal crash feature set from
  ``cppi_learned._crash_features``: vol, vol_ratio, drawdown, panic_z);
- warmup: ``off`` (cold, no pre-window history) vs ``on`` (warm, only
  with a shared warmup baseline of actual pre-window observations, the
  P0-D protocol; warm and cold are separate benchmark versions and
  their results must never be mixed under one name).

Penalty coefficients and the selection rule are chosen on legal real
training/validation only. The selection entry points refuse any other
tuning source, so a sealed previously viewed tier can never silently
become the selection set: pass ``tuning_source="training"`` or
``"validation"`` and nothing else.

Outcomes per cell and seed, unlayered and (when a layered weight
function is supplied) layered through the shipped CPPI risk layer:
mean terminal wealth, mean minimum wealth, terminal and anytime
breach rates, mean terminal shortfall, mean raw pathwise (anytime)
shortfall, mean maximum drawdown, mean cash fraction, plus the
training gradient norm for the cell's loss. The layered run reuses the
same rollout, ledger and costs, so layered and unlayered numbers are
directly comparable.

This is a methods and harness module: it proves each factor changes
behavior on small synthetic or tiny legal-validation runs with several
fixed seeds, and it states the fail-closed selection rule. It does not
train a full model, does not touch sealed tiers, and does not
regenerate any published number.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from dhl.eval.chrono import frozen_config_sha256
from dhl.experiments.cppi_learned import (
    FLOOR,
    FLOOR_PENALTY,
    WARM_LOOKBACK,
    describe_protocol,
    rollout,
)
from dhl.experiments.cppi_learned import FEATURE_NAMES
from dhl.experiments.losses import multihorizon_loss

ANYTIME_LEVELS = ("none", "correct")
FEATURE_LEVELS = ("existing", "causal_rebound")
WARMUP_LEVELS = ("off", "on")

FEATURE_SETS = {
    "existing": ("vol",),
    "causal_rebound": tuple(FEATURE_NAMES),
}

LEGAL_TUNING_SOURCES = ("training", "validation")
MIN_SEEDS = 3


def anytime_penalty(level, floor_penalty=FLOOR_PENALTY):
    """Anytime penalty for a level: 0 for none, the floor penalty for correct."""
    if level == "none":
        return 0.0
    if level == "correct":
        return float(floor_penalty)
    raise ValueError(f"unknown anytime level {level!r}")


def _validate_tuning_source(tuning_source):
    if tuning_source not in LEGAL_TUNING_SOURCES:
        raise ValueError(
            f"refusing tuning source {tuning_source!r}: selection must use "
            "legal training or validation only, never a sealed tier"
        )


def shared_warmup_baseline(series, n_hist=WARM_LOOKBACK):
    """Shared warmup baseline: the last ``n_hist`` actual pre-window prices.

    Every warm cell in a factorial run uses this one baseline, so
    warmup differences between cells cannot come from different
    histories. A series shorter than the lookback raises instead of
    padding with invented prices.
    """
    arr = np.asarray(series, dtype=float).ravel()
    if arr.shape[0] < n_hist:
        raise ValueError(
            f"pre-window series has {arr.shape[0]} prices, fewer than the "
            f"{n_hist} required for a shared warmup baseline"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("pre-window series must be finite")
    return arr[-n_hist:].copy()


@dataclass(frozen=True)
class FactorialCell:
    """One factorial cell: one level per factor, one seed."""

    anytime: str
    features: str
    warmup: str
    seed: int

    def __post_init__(self):
        if self.anytime not in ANYTIME_LEVELS:
            raise ValueError(f"unknown anytime level {self.anytime!r}")
        if self.features not in FEATURE_LEVELS:
            raise ValueError(f"unknown feature level {self.features!r}")
        if self.warmup not in WARMUP_LEVELS:
            raise ValueError(f"unknown warmup level {self.warmup!r}")

    def key(self):
        return (self.anytime, self.features, self.warmup)

    def config(self, floor=FLOOR, tuning_source="validation"):
        return {
            "anytime": self.anytime,
            "features": self.features,
            "warmup": self.warmup,
            "seed": self.seed,
            "floor": float(floor),
            "tuning_source": tuning_source,
        }


def summarize_path(v_terminal, path, weights=None, floor=FLOOR):
    """Outcome metrics for one rollout, from terminal wealth and the path.

    ``v_terminal`` is per-path terminal wealth, ``path`` the full wealth
    path (last point equals terminal wealth by the rollout convention),
    ``weights`` the per-step risky weights when a ledger is available.
    Shortfalls are raw means of relu(floor - wealth): terminal shortfall
    at the horizon, anytime shortfall on the path minimum. Drawdown is
    the per-path maximum peak-to-trough fall, averaged. Cash fraction
    is mean(1 - weight); it is None when no weights are supplied, never
    silently zero.
    """
    vt = np.asarray(v_terminal, dtype=float).ravel()
    p = np.asarray(path, dtype=float)
    path_min = p.min(axis=1)
    peak = np.maximum.accumulate(p, axis=1)
    drawdown = ((peak - p) / np.clip(peak, 1e-12, None)).max(axis=1)
    out = {
        "mean_terminal": float(vt.mean()),
        "mean_min_wealth": float(path_min.mean()),
        "p_breach_terminal": float((vt < floor).mean()),
        "p_breach_anytime": float((path_min < floor).mean()),
        "mean_terminal_shortfall": float(np.maximum(floor - vt, 0.0).mean()),
        "mean_anytime_shortfall": float(
            np.maximum(floor - path_min, 0.0).mean()
        ),
        "mean_max_drawdown": float(drawdown.mean()),
    }
    if weights is None:
        out["mean_cash_fraction"] = None
    else:
        w = np.asarray(weights, dtype=float)
        out["mean_cash_fraction"] = float((1.0 - w).mean())
    return out


def evaluate_cell(weight_fn, S, *, floor=FLOOR, history=None,
                  layered_fn=None):
    """Run one cell's weight functions on paths ``S`` and summarize.

    Returns the unlayered outcomes plus, when ``layered_fn`` is given,
    the same outcomes under the layered weight function with a
    ``layered_`` prefix, plus the P0-D protocol fields. ``S`` is a torch
    price tensor of shape (n_paths, n_steps + 1); ``history`` is the
    shared warmup baseline or None for the cold protocol.
    """
    S_t = torch.as_tensor(S, dtype=torch.float32)
    hist = None
    if history is not None:
        hist = torch.as_tensor(np.asarray(history), dtype=torch.float32)
    with torch.no_grad():
        v_t, path, ledger = rollout(
            S_t, weight_fn, floor=floor, return_path=True,
            return_ledger=True, history=hist,
        )
    out = summarize_path(
        v_t.numpy(), path.numpy(), weights=ledger["weight"].numpy(),
        floor=floor,
    )
    if layered_fn is not None:
        with torch.no_grad():
            lv_t, lpath, lledger = rollout(
                S_t, layered_fn, floor=floor, return_path=True,
                return_ledger=True, history=hist,
            )
        layered = summarize_path(
            lv_t.numpy(), lpath.numpy(), weights=lledger["weight"].numpy(),
            floor=floor,
        )
        for key, value in layered.items():
            out[f"layered_{key}"] = value
    out["protocol"] = describe_protocol(history)
    out["history_days"] = 0 if hist is None else int(hist.shape[-1])
    return out


def loss_and_gradient(policy, S, *, anytime_level, floor=FLOOR,
                      floor_penalty=FLOOR_PENALTY, history=None):
    """Training loss and gradient norm for one policy on paths ``S``.

    Runs the policy through the rollout with gradients enabled, scores
    it with the corrected multi-horizon loss at the cell's anytime
    level, and returns the total, the named components, and the L2 norm
    of the parameter gradient. The anytime factor is the only difference
    between the ``none`` and ``correct`` calls on the same policy.
    """
    S_t = torch.as_tensor(S, dtype=torch.float32)
    hist = None
    if history is not None:
        hist = torch.as_tensor(np.asarray(history), dtype=torch.float32)

    def weight_fn(i, V, tau_frac, cushion, feats):
        t = torch.full_like(V, tau_frac)
        return policy(t, cushion, feats)

    policy.zero_grad()
    v_t, path, ledger = rollout(
        S_t, weight_fn, floor=floor, return_path=True,
        return_ledger=True, history=hist,
    )
    total, components = multihorizon_loss(
        v_t, path, weight_path=ledger["weight"], floor=floor,
        floor_penalty=floor_penalty,
        anytime_penalty=anytime_penalty(anytime_level, floor_penalty),
    )
    total.backward()
    sq = 0.0
    finite = True
    for p in policy.parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            finite = False
        else:
            sq += float((p.grad.detach() ** 2).sum())
    return {
        "total": float(total.detach()),
        "components": {k: float(v.detach()) for k, v in components.items()},
        "grad_norm": float(sq ** 0.5),
        "grad_finite": finite,
    }


def run_factorial(weight_fn_factory, S, *, seeds,
                  anytime_levels=ANYTIME_LEVELS,
                  feature_levels=FEATURE_LEVELS,
                  warmup_levels=WARMUP_LEVELS, warm_history=None,
                  layered_fn_factory=None, floor=FLOOR,
                  tuning_source="validation"):
    """Run the full factorial grid: every level combination, every seed.

    ``weight_fn_factory(cell)`` builds the unlayered weight function for
    a cell (it must seed its policy from ``cell.seed`` through the repo
    initialization protocol, so equal seeds give equal initializations
    across cells); ``layered_fn_factory(cell)`` optionally builds the
    layered counterpart. Warm cells all share ``warm_history``; a warm
    level with no baseline raises instead of inventing one. At least
    three distinct seeds are required: fewer cannot separate a factor
    effect from seed noise.
    """
    _validate_tuning_source(tuning_source)
    seed_list = tuple(dict.fromkeys(seeds))
    if len(seed_list) < MIN_SEEDS:
        raise ValueError(
            f"factorial needs at least {MIN_SEEDS} distinct seeds, "
            f"got {len(seed_list)}"
        )
    if "on" in warmup_levels and warm_history is None:
        raise ValueError(
            "warmup 'on' requires a shared warmup baseline; "
            "no baseline was supplied"
        )
    results = []
    for anytime, features, warmup in itertools.product(
        anytime_levels, feature_levels, warmup_levels
    ):
        for seed in seed_list:
            cell = FactorialCell(anytime, features, warmup, seed)
            history = warm_history if warmup == "on" else None
            layered_fn = None
            if layered_fn_factory is not None:
                layered_fn = layered_fn_factory(cell)
            out = evaluate_cell(
                weight_fn_factory(cell), S, floor=floor, history=history,
                layered_fn=layered_fn,
            )
            out.update({
                "anytime": anytime,
                "features": features,
                "warmup": warmup,
                "seed": seed,
                "config_hash": frozen_config_sha256(
                    cell.config(floor=floor, tuning_source=tuning_source)
                ),
            })
            results.append(out)
    return results


def average_by_cell(results):
    """Mean outcomes per cell across seeds (cash fraction averaged too)."""
    grouped = {}
    for r in results:
        grouped.setdefault((r["anytime"], r["features"], r["warmup"]), []).append(r)
    averaged = {}
    for key, rows in grouped.items():
        means = {}
        for col in rows[0]:
            if col in ("anytime", "features", "warmup", "seed", "protocol",
                       "config_hash"):
                continue
            vals = [row[col] for row in rows if row[col] is not None]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                means[col] = float(np.mean(vals))
            elif vals:
                means[col] = rows[0][col]
        averaged[key] = means
    return averaged


def select_cell(results, *, max_terminal_sacrifice,
                tuning_source="validation"):
    """Fail-closed selection over factorial results.

    Baseline is the (none, existing, off) cell. The best corrected cell
    is the anytime-correct cell with the lowest mean raw pathwise
    shortfall across seeds. Returns ``(cell_key, reason)`` when it both
    improves on the baseline shortfall and sacrifices no more terminal
    wealth than ``max_terminal_sacrifice``; otherwise returns
    ``(None, reason)`` naming the fired falsifier, and no cell is
    selected. Selection inputs must come from legal training or
    validation; anything else raises.
    """
    _validate_tuning_source(tuning_source)
    averaged = average_by_cell(results)
    baseline_key = ("none", "existing", "off")
    if baseline_key not in averaged:
        raise ValueError("baseline cell (none, existing, off) is missing")
    baseline = averaged[baseline_key]
    correct = {
        k: v for k, v in averaged.items() if k[0] == "correct"
    }
    if not correct:
        raise ValueError("no anytime-correct cell to select from")
    best_key = min(correct, key=lambda k: correct[k]["mean_anytime_shortfall"])
    best = correct[best_key]
    if best["mean_anytime_shortfall"] >= baseline["mean_anytime_shortfall"]:
        return None, (
            "falsifier: the corrected penalty does not improve raw "
            "pathwise shortfall across seeds, so no cell is selected"
        )
    sacrifice = baseline["mean_terminal"] - best["mean_terminal"]
    if sacrifice > max_terminal_sacrifice:
        return None, (
            f"falsifier: the shortfall improvement costs {sacrifice:.4f} "
            "terminal wealth, above the stated acceptable tradeoff "
            f"{max_terminal_sacrifice:.4f}, so no cell is selected"
        )
    return best_key, (
        f"selected {best_key}: it improves raw pathwise shortfall from "
        f"{baseline['mean_anytime_shortfall']:.4f} to "
        f"{best['mean_anytime_shortfall']:.4f} at a terminal sacrifice "
        f"of {sacrifice:.4f}, within the stated tradeoff "
        f"{max_terminal_sacrifice:.4f}"
    )
