"""NC5: adaptive CPPI risk budgets, compared head to head.

The constraint wrapper (``cppi_constrained_weight_fn``) and the
crash-weighted loss already exist; re-adding them would not be a
contribution. What has been missing is an isolation of their value
from cap, cost and training-history effects, and a budget rule whose
size is set by a predeclared gap-loss limit instead of a step change
at one vol threshold. This study supplies both.

Four policies, all run through the same ``rollout`` at the same
leverage cap (1.5) and the same cost rate (20 bps):

- ``static_cppi``: classic CPPI with a fixed multiplier m = 4.
- ``vol_target``: classic vol-targeting, weight = target vol /
  realized 21d vol, clipped to the cap. No floor logic at all.
- ``neural_unconstrained``: a CPPIPolicy trained with the existing
  crash-weighted (tail-sensitive) objective on pre-2020 data.
- ``neural_constrained``: the *same* trained checkpoint behind
  ``cppi_vol_budget_weight_fn`` -- a CPPI budget whose multiplier is
  volatility-scaled and hard-capped by the gap-loss budget below.
  One checkpoint per seed serves both neural policies, so the
  constraint layer is the only difference between them.

Predeclared gap-loss budget (locked here, before any run): a
single-day gap of 20% may consume at most 50% of the remaining
cushion, which caps the effective multiplier at 2.5
(``gap_budget_multiplier``). The budget bounds one step from one
state; it says nothing about whole paths, and the jump sweep below
is designed to show exactly where it stops helping.

Published outputs (``runs/risk_budgets/metrics.json``): a config
lock (dataset hashes, full configuration, seed schedule, checkpoint
identifiers) written before comparisons; headline metrics on the
2023+ holdout, the COVID window and a -12% flash-crash set; return
versus CVaR / shortfall / anytime-breach frontiers traced by a risk
dial per policy family; a jump-size sweep from 0% to 30%; and
state-dependent response curves (weight versus cushion ratio and
versus realized vol). Nothing is tuned on any evaluation set; the
2023+ set is the final holdout and is only ever read.

No result in this study should be read as a promise that the floor
holds: discrete rebalancing leaves gap risk that no budget rule
removes, and adverse or null comparisons are reported as measured.

Run: ``python -m dhl.experiments.risk_budgets [--epochs 100]``
"""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from dhl.eval.chrono import (
    contiguous_paths,
    split_returns_by_date,
    window_paths_in_range,
)
from dhl.eval.registry import standard_chrono_registry
from dhl.experiments.cppi_learned import (
    BLOCK_LEN,
    CAP,
    COST_RATE,
    FLOOR,
    R,
    SEED,
    V0,
    WARM_LOOKBACK,
    CPPIPolicy,
    _classic_weight_fn,
    _learned_weight_fn,
    cppi_vol_budget_weight_fn,
    gap_budget_multiplier,
    policy_init_hash,
    rollout,
    seeded_policy,
)
from dhl.experiments.crash_ablation import (
    BATCH_SIZE as ABLATION_BATCH_SIZE,
    COVID_END,
    COVID_START,
    FEATS_FULL,
    HORIZON,
    N_EPOCHS as ABLATION_EPOCHS,
    TRAIN_END,
    VAL_END,
    _flash_crash_paths,
    _train_variant,
    seed_schedule,
)
from dhl.markets.real import DATA_DIR, load_spy_daily, log_returns

# --- predeclared study constants (locked before any run) -----------------
BASE_MULTIPLIER = 4.0
REF_VOL = 0.20
MIN_MULTIPLIER = 1.0
MAX_MULTIPLIER = 6.0
GAP_DROP = 0.20           # predeclared stress gap: a -20% single day
GAP_LOSS_BUDGET = 0.50    # such a gap may eat at most half the cushion
TARGET_VOL = 0.16
VOL_FLOOR = 0.02
CRASH_MULT = 3.0          # the existing tail-sensitive objective's weight
N_SEEDS = 3

STATIC_MULTIPLIERS = (2.0, 3.0, 4.0, 5.0, 6.0)
TARGET_VOLS = (0.08, 0.12, 0.16, 0.20, 0.24)
EXPOSURE_SCALES = (0.50, 0.75, 1.00, 1.25)
GAP_BUDGETS = (0.25, 0.50, 0.75, 1.00)
JUMP_SIZES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
JUMP_N_PATHS = 1000
JUMP_SEED = 20261127

CUSHION_GRID = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
VOL_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80)
FAMILY_ORDER = ("static_cppi", "vol_target", "neural_unconstrained",
                "neural_constrained")

OUTDIR = os.path.join("runs", "risk_budgets")

DISCLAIMER = (
    "This study does not guarantee that the floor will hold on any "
    "path. The gap-loss budget bounds the loss from a single gap of "
    "the predeclared size, measured from the state where the weight "
    "was set; larger gaps, repeated gaps and diffusive losses "
    "between rebalances can still breach the floor, and the reported "
    "breach probabilities are the honest measure of that."
)

CONFIG = {
    "cap": CAP,
    "cost_rate": COST_RATE,
    "floor": FLOOR,
    "v0": V0,
    "cash_rate": R,
    "block_len": BLOCK_LEN,
    "base_multiplier": BASE_MULTIPLIER,
    "ref_vol": REF_VOL,
    "min_multiplier": MIN_MULTIPLIER,
    "max_multiplier": MAX_MULTIPLIER,
    "gap_drop": GAP_DROP,
    "gap_loss_budget": GAP_LOSS_BUDGET,
    "gap_multiplier_cap": gap_budget_multiplier(GAP_DROP, GAP_LOSS_BUDGET),
    "target_vol": TARGET_VOL,
    "vol_floor": VOL_FLOOR,
    "crash_mult": CRASH_MULT,
    "feature_names": list(FEATS_FULL),
    "horizon_days": HORIZON,
    "epochs": ABLATION_EPOCHS,
    "batch_size": ABLATION_BATCH_SIZE,
    "train_end": TRAIN_END,
    "val_end": VAL_END,
    "jump_seed": JUMP_SEED,
    "jump_n_paths": JUMP_N_PATHS,
}


# --- policy builders -------------------------------------------------------

def static_cppi_fn(multiplier=BASE_MULTIPLIER):
    """Classic CPPI with a fixed multiplier."""
    return _classic_weight_fn(multiplier)


def vol_target_fn(target_vol=TARGET_VOL, vol_floor=VOL_FLOOR, cap=CAP):
    """Classic vol-targeting: weight inversely proportional to the
    realized 21d vol feature, clipped to [0, cap]. No floor logic:
    the cushion is ignored by construction."""

    def fn(i, V, tau_frac, cushion, feats):
        vol = torch.as_tensor(feats["vol"]).clamp_min(vol_floor)
        return torch.clamp(target_vol / vol, min=0.0, max=cap)

    return fn


def exposure_scaled_fn(base_fn, scale, cap=CAP):
    """Inference-time risk dial for the neural family: scale the
    proposed weight, clipped to the shared cap."""

    def fn(i, V, tau_frac, cushion, feats):
        w = base_fn(i, V, tau_frac, cushion, feats)
        return torch.clamp(w * scale, min=0.0, max=cap)

    return fn


def constrained_fn(policy, gap_loss_budget=GAP_LOSS_BUDGET):
    """The trained policy behind the vol-scaled, gap-capped budget."""
    return cppi_vol_budget_weight_fn(
        _learned_weight_fn(policy),
        base_multiplier=BASE_MULTIPLIER, ref_vol=REF_VOL,
        min_multiplier=MIN_MULTIPLIER, max_multiplier=MAX_MULTIPLIER,
        gap_drop=GAP_DROP, gap_loss_budget=gap_loss_budget)


def headline_strategies(policy):
    """The four compared policies for one trained checkpoint."""
    return {
        "static_cppi": static_cppi_fn(),
        "vol_target": vol_target_fn(),
        "neural_unconstrained": _learned_weight_fn(policy),
        "neural_constrained": constrained_fn(policy),
    }


def frontier_families(policy):
    """Risk-dial sweeps per policy family, for the frontier plots."""
    learned = _learned_weight_fn(policy)
    return {
        "static_cppi": {f"m={m:g}": static_cppi_fn(m)
                        for m in STATIC_MULTIPLIERS},
        "vol_target": {f"target={t:g}": vol_target_fn(t)
                       for t in TARGET_VOLS},
        "neural_unconstrained": {
            f"scale={s:g}": exposure_scaled_fn(learned, s)
            for s in EXPOSURE_SCALES},
        "neural_constrained": {
            f"budget={b:g}": constrained_fn(policy, gap_loss_budget=b)
            for b in GAP_BUDGETS},
    }


# --- metrics -----------------------------------------------------------------

def frontier_metrics(wealth_paths, floor=FLOOR):
    """Return-versus-risk point for one policy on one path set.

    ``wealth_paths``: array (n_paths, n_steps + 1) of wealth over
    time, final column post-liquidation. CVaR95 of the terminal
    shortfall is the mean of the worst ``ceil(0.05 * n)`` terminal
    shortfalls (at least one path), the tail convention used
    throughout this study.
    """
    W = np.asarray(wealth_paths, dtype=float)
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] < 2:
        raise ValueError(
            f"wealth_paths must be (n_paths >= 1, n_steps + 1 >= 2), "
            f"got shape {W.shape}")
    terminal = W[:, -1]
    min_wealth = W.min(axis=1)
    shortfall_T = np.maximum(floor - terminal, 0.0)
    shortfall_any = np.maximum(floor - min_wealth, 0.0)
    k = max(1, math.ceil(0.05 * len(shortfall_T)))
    cvar = float(np.sort(shortfall_T)[::-1][:k].mean())
    return {
        "mean_wealth": float(terminal.mean()),
        "p_breach_terminal": float((terminal < floor).mean()),
        "p_breach_anytime": float((min_wealth < floor).mean()),
        "mean_shortfall_anytime": float(shortfall_any.mean()),
        "cvar95_shortfall": cvar,
        "worst_shortfall_anytime": float(shortfall_any.max()),
        "n_paths": int(len(terminal)),
    }


def evaluate_point(weight_fn, S_np, floor=FLOOR):
    """Run one policy on one path set and return its frontier point."""
    S = torch.as_tensor(np.asarray(S_np), dtype=torch.float32)
    with torch.no_grad():
        _, path = rollout(S, weight_fn, return_path=True, floor=floor)
    return frontier_metrics(path.numpy(), floor=floor)


def _aggregate(per_seed_metrics):
    """Across-seed mean/sd per metric (n_paths must agree; checked)."""
    keys = [k for k in per_seed_metrics[0] if k != "n_paths"]
    out = {k: {"mean": float(np.mean([d[k] for d in per_seed_metrics])),
               "sd": float(np.std([d[k] for d in per_seed_metrics]))}
           for k in keys}
    out["n_paths"] = int(per_seed_metrics[0]["n_paths"])
    return out


# --- jump sweep ----------------------------------------------------------------

def jump_paths(n_paths, jump, seed=JUMP_SEED, horizon=HORIZON, s0=V0):
    """Calm paths with one injected single-day gap of size ``jump``.

    Same calm-noise recipe as ``crash_ablation._flash_crash_paths``
    (drift 0.0004, sigma 1%, gap day uniform in [20, 50)) with the
    gap size, path count and seed parameterized for the sweep. A
    ``jump`` of 0 injects nothing.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.010, size=(n_paths, horizon))
    if jump > 0:
        gap_day = rng.integers(20, 50, size=n_paths)
        for k, g in enumerate(gap_day):
            rets[k, g] = math.log(1.0 - jump)
    S = np.empty((n_paths, horizon + 1))
    S[:, 0] = s0
    S[:, 1:] = s0 * np.exp(np.cumsum(rets, axis=1))
    return S


# --- response curves ---------------------------------------------------------

def _state(cushion_ratio, vol):
    """One synthetic state: wealth implied by the cushion ratio at
    the study floor, the other crash features at neutral values."""
    cr = float(cushion_ratio)
    if not (0.0 <= cr < 1.0):
        raise ValueError(f"cushion_ratio must lie in [0, 1), got {cr}")
    V_val = FLOOR / (1.0 - cr) if cr > 0 else FLOOR
    V = torch.tensor([V_val])
    cushion = torch.tensor([cr])
    feats = {"vol": torch.tensor([float(vol)]),
             "vol_ratio": torch.ones(1),
             "drawdown": torch.zeros(1),
             "panic_z": torch.zeros(1)}
    return V, cushion, feats


def response_curve(weight_fn, cushion_grid=None, vol_grid=None, *,
                   cushion_ratio=0.10, vol=0.20, tau_frac=0.5):
    """A policy's risky weight along one state axis, the other
    pinned. Pass exactly one of ``cushion_grid`` / ``vol_grid``."""
    if (cushion_grid is None) == (vol_grid is None):
        raise ValueError("pass exactly one of cushion_grid / vol_grid")
    out = []
    if cushion_grid is not None:
        states = [_state(cr, vol) for cr in cushion_grid]
    else:
        states = [_state(cushion_ratio, v) for v in vol_grid]
    with torch.no_grad():
        for V, cushion, feats in states:
            w = weight_fn(0, V, tau_frac, cushion, feats)
            out.append(float(torch.as_tensor(w).reshape(-1)[0]))
    return out


# --- config lock ---------------------------------------------------------------

def policy_param_hash(policy):
    """Full SHA-256 fingerprint of a policy's trained parameters."""
    h = hashlib.sha256()
    for p in policy.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def dataset_lock(dates):
    """Identifiers for the exact data the study ran on."""
    files = {}
    for name in ("spy_daily_2015.csv", "spy_daily_2021.csv"):
        path = os.path.join(DATA_DIR, name)
        with open(path, "rb") as f:
            blob = f.read()
        files[name] = {"sha256": hashlib.sha256(blob).hexdigest(),
                       "bytes": len(blob)}
    return {"files": files, "date_range": [dates[0], dates[-1]],
            "n_prices": len(dates)}


# --- the study -------------------------------------------------------------------

def run_study(n_epochs=ABLATION_EPOCHS, batch_size=ABLATION_BATCH_SIZE,
              seed=SEED, n_seeds=N_SEEDS, outdir=OUTDIR):
    data = load_spy_daily()
    dates = list(data["dates"])
    prices = np.asarray(data["adjclose"])
    rets_all = log_returns(prices)
    print(f"loaded {len(rets_all)} SPY daily log returns "
          f"({dates[0]} -> {dates[-1]})", flush=True)

    splits, _ = split_returns_by_date(dates, rets_all, TRAIN_END, VAL_END)
    rets_train = splits["train"]
    standard_chrono_registry().validate_training_period(
        dates, dates[0], TRAIN_END, lookback_bars=WARM_LOOKBACK)
    seeds = seed_schedule(seed, n_seeds)

    # The lock is assembled before any training or comparison: data
    # identity, the full predeclared configuration and the shared
    # seed schedule. Checkpoint identifiers join it as each seed's
    # policy is trained, before that seed is evaluated.
    lock = {
        "dataset": dataset_lock(dates),
        "config": dict(CONFIG),
        "seed_schedule": list(seeds),
        "checkpoints": {},
    }
    print(f"seed schedule: {seeds}", flush=True)

    test_mask = np.array(dates) >= VAL_END
    S_2023, _ = contiguous_paths(prices[test_mask], HORIZON, stride=5,
                                 s0=V0)
    S_cov, _ = window_paths_in_range(dates, prices, COVID_START,
                                     COVID_END, 30, stride=1, s0=V0)
    S_flash = _flash_crash_paths()
    eval_sets = {"2023plus_63d": S_2023, "covid_30d": S_cov,
                 "flashcrash_63d": S_flash}
    frontier_sets = {"2023plus_63d": S_2023, "covid_30d": S_cov}
    print("test sets: " + ", ".join(
        f"{k} {v.shape}" for k, v in eval_sets.items()), flush=True)

    headline = {name: {"per_seed": {}, "across_seeds": {}}
                for name in eval_sets}
    frontiers = {name: {} for name in frontier_sets}
    jump = {f"{j:.2f}": {} for j in JUMP_SIZES}
    curves = None

    for s in seeds:
        t0 = time.time()
        init_hash = policy_init_hash(seeded_policy(
            CPPIPolicy, seed=s, feature_names=FEATS_FULL))
        policy = _train_variant(rets_train, FEATS_FULL, CRASH_MULT, s,
                                n_epochs, batch_size,
                                label=f"NC5 seed={s}")
        lock["checkpoints"][str(s)] = {
            "init_hash": init_hash,
            "trained_param_sha256": policy_param_hash(policy),
        }
        print(f"seed {s}: trained in {time.time() - t0:.0f}s, "
              f"checkpoint {lock['checkpoints'][str(s)]['trained_param_sha256'][:16]}",
              flush=True)
        strategies = headline_strategies(policy)

        for set_name, S in eval_sets.items():
            headline[set_name]["per_seed"][str(s)] = {
                name: evaluate_point(fn, S)
                for name, fn in strategies.items()}

        families = frontier_families(policy)
        for set_name, S in frontier_sets.items():
            fam_out = frontiers[set_name].setdefault(
                str(s), {})
            for fam, dials in families.items():
                fam_out[fam] = {dial: evaluate_point(fn, S)
                                for dial, fn in dials.items()}

        for j in JUMP_SIZES:
            S_j = jump_paths(JUMP_N_PATHS, j)
            jump[f"{j:.2f}"].setdefault("per_seed", {})[str(s)] = {
                name: evaluate_point(fn, S_j)
                for name, fn in strategies.items()}

        if s == seeds[0]:
            curves = {
                "checkpoint_seed": int(s),
                "vs_cushion": {
                    "grid": list(CUSHION_GRID), "vol": 0.20,
                    "weights": {name: response_curve(
                        fn, cushion_grid=CUSHION_GRID, vol=0.20)
                        for name, fn in strategies.items()}},
                "vs_vol": {
                    "grid": list(VOL_GRID), "cushion_ratio": 0.10,
                    "weights": {name: response_curve(
                        fn, vol_grid=VOL_GRID, cushion_ratio=0.10)
                        for name, fn in strategies.items()}},
            }

    for set_name in eval_sets:
        per = headline[set_name]["per_seed"]
        headline[set_name]["across_seeds"] = {
            name: _aggregate([per[str(s)][name] for s in seeds])
            for name in per[str(seeds[0])]}

    frontiers_out = {}
    for set_name in frontier_sets:
        frontiers_out[set_name] = {}
        for fam in FAMILY_ORDER:
            frontiers_out[set_name][fam] = {}
            dials = frontiers[set_name][str(seeds[0])][fam]
            for dial in dials:
                frontiers_out[set_name][fam][dial] = _aggregate(
                    [frontiers[set_name][str(s)][fam][dial]
                     for s in seeds])

    jump_out = {}
    for j in JUMP_SIZES:
        key = f"{j:.2f}"
        per = jump[key]["per_seed"]
        jump_out[key] = {
            name: _aggregate([per[str(s)][name] for s in seeds])
            for name in per[str(seeds[0])]}

    results = {
        "study": "NC5 adaptive CPPI risk budgets",
        "disclaimer": DISCLAIMER,
        "config_lock": lock,
        "headline": headline,
        "frontiers": frontiers_out,
        "jump_sweep": jump_out,
        "response_curves": curves,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {outdir}/metrics.json", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="NC5 adaptive CPPI risk budgets")
    parser.add_argument("--epochs", type=int, default=ABLATION_EPOCHS)
    parser.add_argument("--batch-size", type=int,
                        default=ABLATION_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    run_study(n_epochs=args.epochs, batch_size=args.batch_size,
              seed=args.seed, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
