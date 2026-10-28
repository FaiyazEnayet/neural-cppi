"""Ablation study: which crash-robustness interventions actually help?

Controlled comparison of learned portfolio-insurance policies that
differ in exactly one intervention each, all trained with matched
budgets (epochs, batch, architecture) under one shared seed schedule
on strictly pre-2020 data. Every variant is trained once per
schedule seed, so a variant gap can be read per seed instead of
confounding the intervention with a per-variant seed offset (the
original run used seed, seed+1, ..., seed+7 across variants).
Variants with different feature counts have different parameter
shapes by construction; the shared schedule compares their
replicated training outcomes, it does not pretend their tensors
match. COVID crash and 2023+ sets are held-out -- nothing is tuned
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
from dhl.eval.registry import standard_chrono_registry
from dhl.experiments.cppi_learned import (
    BLOCK_LEN,
    CAP,
    FLOOR,
    SEED,
    V0,
    WARM_LOOKBACK,
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
N_SEEDS = 3
HORIZON = 63
FEATS_FULL = ("vol", "vol_ratio", "drawdown", "panic_z")
FEATS_BASE = ("vol",)
CRASH_MULTS = (0.0, 1.0, 3.0, 5.0)

# (variant name, training window, feature names, crash_mult, CPPI layer)
# The E sweep shares the base spec and varies crash_mult only.
VARIANT_SPECS = [
    ("A", "2015", FEATS_BASE, 0.0, False),
    ("B", "2007", FEATS_BASE, 0.0, False),
    ("C", "2007", FEATS_FULL, 0.0, False),
    ("F", "2007", FEATS_FULL, 3.0, True),
    ("F_no2008", "2015", FEATS_FULL, 3.0, True),
    ("F_nofeats", "2007", FEATS_BASE, 3.0, True),
    ("F_noloss", "2007", FEATS_FULL, 0.0, True),
]

OUTDIR = os.path.join("runs", "crash_ablation")


def seed_schedule(base_seed, n_seeds):
    """One shared seed list applied to every variant.

    Every variant trains once per schedule seed with matched budgets,
    so per-seed gaps separate seed noise from design differences.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    return tuple(base_seed + i for i in range(n_seeds))


def variant_training_plan(base_seed, n_seeds):
    """Machine-checkable training plan: (variant, seed) pairs.

    Every spec, including each E-sweep crash_mult, appears once per
    schedule seed and never with a per-variant seed offset.
    """
    plan = []
    for seed in seed_schedule(base_seed, n_seeds):
        for name, _, _, _, _ in VARIANT_SPECS:
            plan.append((name, seed))
        for cm in CRASH_MULTS:
            plan.append((f"E_mult{cm}", seed))
    return plan


def summarize_across_seeds(per_seed, reference="G"):
    """Across-seed summary that preserves per-seed gaps.

    ``per_seed`` maps seed -> {strategy: metrics} where each metrics
    dict holds ``mean_wealth`` and, for non-reference strategies, a
    ``paired_gap_vs_<reference>`` dict with a ``mean``. Returns
    per-strategy across-seed mean/sd plus the per-seed gap list, so a
    mean improvement that flips sign across seeds stays visible.
    """
    strategies = sorted({name for seed_res in per_seed.values()
                         for name in seed_res})
    out = {"n_seeds": len(per_seed)}
    for name in strategies:
        means = [per_seed[s][name]["mean_wealth"] for s in per_seed
                 if name in per_seed[s]]
        entry = {
            "mean_wealth_mean": float(np.mean(means)),
            "mean_wealth_sd": float(np.std(means)),
        }
        gap_key = "paired_gap_vs_" + reference
        gaps = [per_seed[s][name][gap_key]["mean"] for s in per_seed
                if name in per_seed[s] and gap_key in per_seed[s][name]]
        if gaps:
            entry["paired_gap_mean"] = float(np.mean(gaps))
            entry["paired_gap_sd"] = float(np.std(gaps))
            entry["per_seed_gap"] = [float(g) for g in gaps]
        out[name] = entry
    return out


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


def _train_seed_policies(rets_2007, rets_2015, seed, n_epochs, batch_size):
    """Train every variant spec once at one shared schedule seed."""
    windows = {"2007": rets_2007, "2015": rets_2015}
    policies = {}
    for name, window, feats, cm, _layer in VARIANT_SPECS:
        policies[name] = _train_variant(
            windows[window], feats, cm, seed, n_epochs, batch_size,
            label=f"{name} seed={seed}")
    e_policies = {}
    for cm in CRASH_MULTS:
        e_policies[cm] = _train_variant(
            rets_2007, FEATS_BASE, cm, seed, n_epochs, batch_size,
            label=f"E mult={cm} seed={seed}")
    return policies, e_policies


def _strategies_for_seed(policies, e_policies):
    """Strategy map for one seed's trained policies."""

    def classic(m=4.0):
        def fn(i, V, tau_frac, cushion, feats):
            return classic_cppi_weight(V, multiplier=m, cap=CAP)
        return fn

    def layered(policy):
        return cppi_constrained_weight_fn(
            _learned_weight_fn(policy),
            multiplier=4.0, emergency_vol=0.35,
            emergency_multiplier=2.0)

    strategies = {
        "A_base": _learned_weight_fn(policies["A"]),
        "B_+2008": _learned_weight_fn(policies["B"]),
        "C_+feats": _learned_weight_fn(policies["C"]),
        "D_+cppi_layer": layered(policies["B"]),
        "F_all": layered(policies["F"]),
        "F_no2008": layered(policies["F_no2008"]),
        "F_nofeats": layered(policies["F_nofeats"]),
        "F_noloss": layered(policies["F_noloss"]),
        "G_classic_m4": classic(4.0),
    }
    # E sweep strategies (raw, no CPPI layer, to isolate the loss effect)
    for cm, pol in e_policies.items():
        strategies[f"E_mult{cm}"] = _learned_weight_fn(pol)
    return strategies


def run_ablation(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, seed=SEED,
                 n_seeds=N_SEEDS):
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

    # Fail-closed: training windows must touch no held-out episode bar,
    # even through the feature lookback.
    standard_chrono_registry().validate_training_period(
        dates, dates[0], TRAIN_END, lookback_bars=WARM_LOOKBACK)
    seeds = seed_schedule(seed, n_seeds)
    print(f"  seed schedule: {seeds}", flush=True)

    # ---------------- test sets (built once, shared by all seeds) ----
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
        per_seed = {}
        for s in seeds:
            policies, e_policies = _train_seed_policies(
                rets_2007, rets_2015, s, n_epochs, batch_size)
            strategies = _strategies_for_seed(policies, e_policies)
            res = paired_evaluate(strategies, S, rollout, FLOOR,
                                  reference="G_classic_m4")
            # worst shortfall per strategy
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
            per_seed[s] = res
        across = summarize_across_seeds(per_seed,
                                        reference="G_classic_m4")
        _print_across_table(set_name, across)
        results[set_name] = {"per_seed": per_seed,
                             "across_seeds": across}

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUTDIR}/metrics.json", flush=True)
    return results


def _print_across_table(title, across):
    print(f"\n--- {title} (across {across['n_seeds']} seeds) ---", flush=True)
    print(f"{'strategy':16s}  {'E[V]':>12s}  {'gap_vs_G':>14s}  "
          f"{'per-seed gap':>24s}", flush=True)
    for name, m in sorted(across.items()):
        if name == "n_seeds":
            continue
        gap = m.get("paired_gap_mean")
        gap_s = (f"{gap:+6.2f} (sd {m['paired_gap_sd']:.2f})"
                 if gap is not None else " " * 14)
        per_seed = ", ".join(f"{g:+.2f}"
                             for g in m.get("per_seed_gap", []))
        print(f"{name:16s}  "
              f"{m['mean_wealth_mean']:6.2f} (sd {m['mean_wealth_sd']:4.2f})  "
              f"{gap_s:>14s}  {per_seed:>24s}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Crash-robustness ablation")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="replications per variant on the shared "
                             "seed schedule (default %(default)s)")
    args = parser.parse_args()
    run_ablation(n_epochs=args.epochs, batch_size=args.batch_size,
                 seed=args.seed, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
