"""Ablation study: which crash-robustness interventions actually help?

Controlled comparison of learned portfolio-insurance policies that
differ in exactly one intervention each, all trained with identical
budgets (epochs, batch, seed, architecture) on strictly pre-2020
data. COVID crash and 2023+ sets are held-out -- nothing is tuned
on them.

Variants:
  A = baseline: 2015-2020 training, (tau, cushion, vol) features,
      unweighted breach penalty. Replicates the pre-existing policy.
  B = A + 2008 GFC in the training window (2007-2020).
  C = B + crash-indicator features (vol_ratio, drawdown, panic_z).
  D = B + CPPI safety layer at inference (same trained net as B).
  E = B + crash-weighted breach penalty (crash_mult swept).
  F = all combined (2008 data, crash features, weighted loss,
      CPPI safety layer at inference).
  G = classic CPPI m=4 (no learning; benchmark).

Leave-one-out on F drops each intervention in turn.

Metrics per test set: expected terminal wealth, terminal and
anytime breach probability, worst shortfall below the floor, mean
max drawdown, and average turnover (proxied by mean absolute
weight change per step).

Stocks and cash only. No derivatives, no shorting, no leverage.

Run: ``python -m dhl.experiments.crash_ablation [--epochs 100]``

Writes ``runs/crash_ablation/metrics.json``.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from dhl.eval.chrono import (
    contiguous_paths,
    max_drawdown,
    paired_evaluate,
    split_returns_by_date,
    window_paths_in_range,
)
from dhl.experiments.cppi_learned import (
    BLOCK_LEN,
    CAP,
    FLOOR,
    SEED,
    V0,
    CPPIPolicy,
    _crash_weights,
    _learned_weight_fn,
    block_bootstrap_paths,
    classic_cppi_weight,
    cppi_constrained_weight_fn,
    rollout,
    policy_init_hash,
    seeded_policy,
    train_cppi_policy,
)
from dhl.markets.real import load_spy_daily, log_returns

TRAIN_END = "2020-01-01"
VAL_END = "2023-01-01"
COVID_START = "2020-02-19"
COVID_END = "2020-05-01"
N_EPOCHS = 100
BATCH_SIZE = 1024
HORIZON = 63
FEATS_FULL = ("vol", "vol_ratio", "drawdown", "panic_z")
FEATS_BASE = ("vol",)
CRASH_MULTS = (0.0, 1.0, 3.0, 5.0)

OUTDIR = os.path.join("runs", "crash_ablation")


def _train_variant(train_rets, feature_names, crash_mult, seed,
                   n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                   label=""):
    """Train one policy variant; returns the trained CPPIPolicy."""
    print(f"\n--- training {label} "
          f"(feats={len(feature_names)}, crash_mult={crash_mult}) ---",
          flush=True)
    t0 = time.time()

    def sampler(n, s):
        return block_bootstrap_paths(
            train_rets, n, HORIZON, BLOCK_LEN, S0=V0, seed=s)

    policy = seeded_policy(CPPIPolicy, seed=seed,
                           feature_names=feature_names)
    print(f"    init hash: {policy_init_hash(policy)}", flush=True)
    train_cppi_policy(policy, sampler, HORIZON, n_epochs=n_epochs,
                      batch_size=batch_size, seed=seed, log_every=0,
                      crash_mult=crash_mult)
    print(f"    done in {time.time() - t0:.0f}s", flush=True)
    return policy


def _turnover(weight_fn, S_np, floor=FLOOR):
    """Mean absolute weight change per step, averaged over paths.

    Records the weight sequence from a real rollout, so turnover
    reflects actual trading behavior including cost-driven smoothing.
    """
    S = torch.as_tensor(np.asarray(S_np), dtype=torch.float32)
    weights = []
    orig_fn = weight_fn

    def recording_fn(i, V, tau_frac, cushion, feats):
        w = orig_fn(i, V, tau_frac, cushion, feats)
        weights.append(w.detach().clone())
        return w

    with torch.no_grad():
        rollout(S, recording_fn, floor=floor)
    if len(weights) < 2:
        return 0.0
    diffs = [(b - a).abs().mean().item()
             for a, b in zip(weights[:-1], weights[1:])]
    return sum(diffs) / len(diffs)


def _flash_crash_paths(n_paths=200, seed=0, s0=100.0):
    """Sudden-deterioration stress: calm paths with an unannounced
    -12% single-day gap at a random point. Vol features cannot see it
    coming; only the CPPI cushion mechanism can react in time."""
    rng = np.random.default_rng(seed)
    horizon = 63
    rets = rng.normal(0.0004, 0.010, size=(n_paths, horizon))
    gap_day = rng.integers(20, 50, size=n_paths)
    for k, g in enumerate(gap_day):
        rets[k, g] = math.log(0.88)  # -12% overnight gap
    S = np.empty((n_paths, horizon + 1))
    S[:, 0] = s0
    S[:, 1:] = s0 * np.exp(np.cumsum(rets, axis=1))
    return S


def run_ablation(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, seed=SEED):
    data = load_spy_daily()
    dates = list(data["dates"])
    prices = np.asarray(data["adjclose"])
    rets_all = log_returns(prices)
    print(f"loaded {len(rets_all)} returns ({dates[0]} -> {dates[-1]})",
          flush=True)

    splits, _ = split_returns_by_date(dates, rets_all, TRAIN_END, VAL_END)
    # B+ training uses the full 2007-2020 window (includes 2008);
    # A training uses 2015-2020 only.
    rets_2007 = splits["train"]
    mask_2015 = np.array(dates[1:]) >= "2015-01-01"
    mask_pre2020 = np.array(dates[1:]) < TRAIN_END
    rets_2015 = rets_all[mask_2015 & mask_pre2020]
    print(f"  train 2007+: {len(rets_2007)} returns "
          f"({dates[1]} -> ...)", flush=True)
    print(f"  train 2015+: {len(rets_2015)} returns", flush=True)

    # ---------------- train variants ----------------
    policies = {}
    policies["A"] = _train_variant(rets_2015, FEATS_BASE, 0.0, seed,
                                   n_epochs, batch_size, label="A baseline")
    policies["B"] = _train_variant(rets_2007, FEATS_BASE, 0.0, seed + 1,
                                   n_epochs, batch_size, label="B +2008 data")
    policies["C"] = _train_variant(rets_2007, FEATS_FULL, 0.0, seed + 2,
                                   n_epochs, batch_size,
                                   label="C +crash features")
    # E sweep over crash_mult
    e_policies = {}
    for cm in CRASH_MULTS:
        e_policies[cm] = _train_variant(
            rets_2007, FEATS_BASE, cm, seed + 3, n_epochs, batch_size,
            label=f"E mult={cm}")
    policies["F"] = _train_variant(rets_2007, FEATS_FULL, 3.0, seed + 4,
                                   n_epochs, batch_size, label="F all")
    # Leave-one-out on F
    policies["F_no2008"] = _train_variant(
        rets_2015, FEATS_FULL, 3.0, seed + 5, n_epochs, batch_size,
        label="F minus 2008 data")
    policies["F_nofeats"] = _train_variant(
        rets_2007, FEATS_BASE, 3.0, seed + 6, n_epochs, batch_size,
        label="F minus crash features")
    policies["F_noloss"] = _train_variant(
        rets_2007, FEATS_FULL, 0.0, seed + 7, n_epochs, batch_size,
        label="F minus crash loss")

    # ---------------- build strategies ----------------
    def classic(m=4.0):
        def fn(i, V, tau_frac, cushion, feats):
            return classic_cppi_weight(V, multiplier=m, cap=CAP)
        return fn

    strategies = {
        "A_base": _learned_weight_fn(policies["A"]),
        "B_+2008": _learned_weight_fn(policies["B"]),
        "C_+feats": _learned_weight_fn(policies["C"]),
        "D_+cppi_layer": cppi_constrained_weight_fn(
            _learned_weight_fn(policies["B"]),
            multiplier=4.0, emergency_vol=0.35,
            emergency_multiplier=2.0),
        "F_all": cppi_constrained_weight_fn(
            _learned_weight_fn(policies["F"]),
            multiplier=4.0, emergency_vol=0.35,
            emergency_multiplier=2.0),
        "F_no2008": cppi_constrained_weight_fn(
            _learned_weight_fn(policies["F_no2008"]),
            multiplier=4.0, emergency_vol=0.35,
            emergency_multiplier=2.0),
        "F_nofeats": cppi_constrained_weight_fn(
            _learned_weight_fn(policies["F_nofeats"]),
            multiplier=4.0, emergency_vol=0.35,
            emergency_multiplier=2.0),
        "F_noloss": cppi_constrained_weight_fn(
            _learned_weight_fn(policies["F_noloss"]),
            multiplier=4.0, emergency_vol=0.35,
            emergency_multiplier=2.0),
        "G_classic_m4": classic(4.0),
    }
    # E sweep strategies (raw, no CPPI layer, to isolate the loss effect)
    for cm, pol in e_policies.items():
        strategies[f"E_mult{cm}"] = _learned_weight_fn(pol)

    # ---------------- test sets ----------------
    test_mask = np.array(dates) >= VAL_END
    S_2023, _ = contiguous_paths(prices[test_mask], 63, stride=5, s0=V0)
    S_cov, _ = window_paths_in_range(
        dates, prices, COVID_START, COVID_END, 30, stride=1, s0=V0)
    S_flash = _flash_crash_paths()
    print(f"\ntest sets: 2023+ {S_2023.shape}, "
          f"covid {S_cov.shape}, flash {S_flash.shape}", flush=True)

    results = {}
    for set_name, S in (("2023plus_63d", S_2023),
                        ("covid_30d", S_cov),
                        ("flashcrash_63d", S_flash)):
        print(f"\n===== {set_name} =====", flush=True)
        res = paired_evaluate(strategies, S, rollout, FLOOR,
                              reference="G_classic_m4")
        # worst shortfall + turnover per strategy
        for name, fn in strategies.items():
            St = torch.as_tensor(np.asarray(S), dtype=torch.float32)
            with torch.no_grad():
                V_T, path = rollout(St, fn, return_path=True)
            W = path.numpy()
            shortfall = np.maximum(FLOOR - W.min(axis=1), 0.0)
            res[name]["worst_shortfall"] = float(shortfall.max())
            res[name]["mean_shortfall"] = float(shortfall.mean())
        # turnover on a subset (expensive) -- 2023+ only
        if set_name == "2023plus_63d":
            for name, fn in strategies.items():
                res[name]["turnover"] = _turnover(fn, S[:200])
        _print_table(set_name, res)
        results[set_name] = res

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUTDIR}/metrics.json", flush=True)
    return results


def _print_table(title, res):
    print(f"\n--- {title} ---", flush=True)
    print(f"{'strategy':16s}  {'E[V]':>7s}  {'P(brT)':>6s}  "
          f"{'P(any)':>6s}  {'worstSF':>7s}  {'meanDD':>6s}", flush=True)
    for name, m in res.items():
        print(f"{name:16s}  {m['mean_wealth']:7.2f}  "
              f"{m['p_breach_terminal']:6.3f}  {m['p_breach_anytime']:6.3f}  "
              f"{m['worst_shortfall']:7.2f}  {m['mean_max_drawdown']:6.3f}",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description="Crash-robustness ablation")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_ablation(n_epochs=args.epochs, batch_size=args.batch_size,
                 seed=args.seed)


if __name__ == "__main__":
    main()
