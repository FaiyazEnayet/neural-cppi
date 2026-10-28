"""Chronological out-of-time evaluation for the learned CPPI policy.

The honest test the bootstrap-only evaluation can't give: train the
policy on data strictly before 2020, then evaluate it on contiguous
windows of what actually happened afterwards -- the 2021-2022
validation years, 2023-2026 as the test set, and the Feb-Apr 2020
COVID crash as a stress case. No resampling in the test paths; every
window is real history. All strategies see the identical paths, so
gaps are policy gaps, not luck of the draw. 2020 bars outside the
COVID stress window are reserved and belong to no train/val/test
bucket; the episode registry records that explicitly.

Every results table carries a metadata block (frequency, training
cutoff, proxy, dividends, cash convention, seen flag) built by
``chrono_table_metadata`` and validated by
``dhl.markets.history_ext``. The 2008 crisis window is an in-sample
diagnostic for a pre-2020-trained model, never an unseen crash; a
model trained strictly before 2008 is what earns the unseen label.
The monthly 1929-1933 proxy tier is a metadata definition only: no
Depression series of any frequency is loaded, and daily Depression
claims are refused outright.

Stocks and cash only. No derivatives, no shorting, no leverage.

Run: ``python -m dhl.experiments.chrono_eval [--epochs 150]``

Writes ``runs/chrono_eval/metrics.json`` and prints paired tables.
"""

import argparse
import json
import os

import numpy as np

from dhl.eval.chrono import (
    contiguous_paths,
    paired_evaluate,
    split_returns_by_date,
    window_paths_in_range,
)
from dhl.eval.registry import (
    standard_chrono_registry,
    summarize_window_count,
)
from dhl.experiments.cppi_learned import (
    BLOCK_LEN,
    CAP,
    FLOOR,
    MULTIPLIERS,
    SEED,
    V0,
    WARM_LOOKBACK,
    CPPIPolicy,
    _learned_weight_fn,
    block_bootstrap_paths,
    policy_init_hash,
    rollout,
    seeded_policy,
    train_cppi_policy,
)
from dhl.markets.history_ext import (
    CASH_CONVENTION,
    SPY_DAILY_PROXY,
    SPY_DIVIDENDS,
    stress_tiers,
    table_metadata,
    validate_table_metadata,
)
from dhl.markets.real import load_spy_daily, log_returns

TRAIN_END = "2020-01-01"   # train strictly before 2020: COVID unseen
VAL_START = "2021-01-01"   # validation is 2021-2022 only; 2020 reserved
VAL_END = "2023-01-01"     # validate on 2021-2022, test on 2023+
COVID_START = "2020-02-19"  # SPY pre-crash peak
COVID_END = "2020-05-01"    # through the rebound's first leg
N_EPOCHS = 150
BATCH_SIZE = 1024
SEED = SEED

OUTDIR = os.path.join("runs", "chrono_eval")


def _print_table(title, results, reference):
    print(f"\n=== {title} (n_paths={results[reference]['n_paths']}) ===",
          flush=True)
    print(f"{'strategy':22s}  {'E[V_T]':>7s}  {'sd':>6s}  "
          f"{'P(brch)':>7s}  {'P(any)':>6s}  {'meanDD':>6s}  "
          f"{'gap_vs_' + reference[:12]:>22s}",
          flush=True)
    for name, m in results.items():
        gap = m.get("paired_gap_vs_" + reference)
        gap_s = (f"{gap['mean']:+6.2f} (sd {gap['std']:.2f}, "
                 f"{gap['frac_paths_better']:.0%} better)"
                 if gap else " " * 22)
        print(
            f"{name:22s}  {m['mean_wealth']:7.2f}  {m['std_wealth']:6.2f}  "
            f"{m['p_breach_terminal']:7.3f}  {m['p_breach_anytime']:6.3f}  "
            f"{m['mean_max_drawdown']:6.3f}  {gap_s}",
            flush=True,
        )


def chrono_table_metadata():
    """Validated metadata blocks for every table run_chrono_eval writes.

    Each block states frequency, training cutoff, proxy, dividends,
    and cash convention, plus the seen flag. Validation is
    fail-closed: a bad block raises here, before any table is written.
    The in-sample bootstrap table is resampled from training data, so
    its seen flag is True by construction.
    """
    tiers = stress_tiers()
    tables = {
        "test_2023plus": table_metadata(tiers["post_freeze_reserved"]),
        "val_2021_2022": table_metadata(
            tiers["validation_2021_2022"]),
        "covid_crash": table_metadata(tiers["covid_held_out"]),
        "insample_bootstrap": {
            "tier": "insample_bootstrap",
            "frequency": "block-bootstrap resample of daily",
            "training_start": "2007-01-01",
            "training_cutoff": TRAIN_END,
            "eval_start": "2007-01-01",
            "eval_end": TRAIN_END,
            "proxy": SPY_DAILY_PROXY,
            "dividends": SPY_DIVIDENDS,
            "cash_convention": CASH_CONVENTION,
            "seen": True,
            "previously_viewed": True,
            "instruments": ["SPY"],
            "freeze_date": None,
        },
    }
    for meta in tables.values():
        validate_table_metadata(meta)
    return tables


def run_chrono_eval(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, seed=SEED):
    data = load_spy_daily()
    dates, prices = data["dates"], data["adjclose"]
    rets = log_returns(prices)
    print(f"loaded {len(rets)} SPY daily log returns "
          f"({dates[0]} -> {dates[-1]})", flush=True)

    # --- strict chronological split: nothing from 2020+ in training ---
    # Validation starts at VAL_START, not TRAIN_END: 2020 bars belong
    # to no bucket (reserved; the COVID window has its own test set).
    splits, split_dates = split_returns_by_date(
        dates, rets, TRAIN_END, VAL_END, val_start=VAL_START)
    for k in ("train", "val", "test"):
        ds = split_dates[k]
        print(f"  {k:5s}: {len(splits[k])} returns, "
              f"{ds[0]} -> {ds[-1]}", flush=True)

    # --- fail-closed registry check: the training period must touch
    # no held-out episode bar, even through the feature lookback ---
    registry = standard_chrono_registry()
    registry.validate_training_period(
        dates, dates[0], TRAIN_END, lookback_bars=WARM_LOOKBACK)

    # --- every results table carries a validated metadata block ---
    tables = chrono_table_metadata()

    results = {
        "splits": {k: {"n_returns": len(v),
                       "first": split_dates[k][0],
                       "last": split_dates[k][-1]}
                   for k, v in splits.items()},
        "train_end": TRAIN_END, "val_start": VAL_START,
        "val_end": VAL_END,
        "registry": {
            "episodes": [
                {"name": e.name, "start": e.start, "end": e.end,
                 "role": e.role,
                 "independence_group": e.independence_group,
                 "previously_viewed": e.previously_viewed}
                for e in registry.episodes],
        },
    }

    # --- train once, on pre-2020 bootstraps only ---
    horizon = 63
    train_rets = splits["train"]

    def train_sampler(n, s):
        return block_bootstrap_paths(
            train_rets, n, horizon, BLOCK_LEN, S0=V0, seed=s)

    print("\ntraining policy on pre-2020 data...", flush=True)
    policy = seeded_policy(CPPIPolicy, seed=seed)
    print(f"policy init hash: {policy_init_hash(policy)}", flush=True)
    train_cppi_policy(policy, train_sampler, horizon, n_epochs=n_epochs,
                      batch_size=batch_size, seed=seed, log_every=50)

    strategies = {
        "learned (train pre-2020)": _learned_weight_fn(policy),
    }
    for m in MULTIPLIERS:
        from dhl.experiments.cppi_learned import classic_cppi_weight

        def _fn(i, V, tau_frac, cushion, feats, _m=m):
            return classic_cppi_weight(V, multiplier=_m, cap=CAP)

        strategies[f"classic_m{m}"] = _fn
    reference = "learned (train pre-2020)"

    # --- test set 1: contiguous 2023+ history, 30d and 63d ---
    test_mask = np.array(dates) >= VAL_END
    test_prices = prices[test_mask]
    test_dates = [d for d, m in zip(dates, test_mask) if m]
    print(f"\ntest window: {test_dates[0]} -> {test_dates[-1]} "
          f"({len(test_prices)} bars)", flush=True)

    for hlen, stride in ((30, 5), (63, 5)):
        S, starts = contiguous_paths(test_prices, hlen, stride=stride,
                                     s0=V0)
        res = paired_evaluate(strategies, S, rollout, FLOOR,
                              reference=reference)
        _print_table(f"2023+ contiguous, {hlen}d", res, reference)
        results[f"test_2023plus_{hlen}d"] = {
            "window_starts": [test_dates[s] for s in starts],
            "windows": summarize_window_count(len(S), hlen, stride),
            "metadata": tables["test_2023plus"],
            "strategies": res,
        }

    # --- test set 2: validation years 2021-2022, contiguous ---
    # Previously the mask started at TRAIN_END, so the "2021-2022"
    # table actually covered 2020-2022 including the COVID year.
    val_mask = (np.array(dates) >= VAL_START) & (np.array(dates) < VAL_END)
    val_prices = prices[val_mask]
    S_val, starts_val = contiguous_paths(val_prices, 63, stride=10, s0=V0)
    res_val = paired_evaluate(strategies, S_val, rollout, FLOOR,
                              reference=reference)
    _print_table("2021-2022 contiguous (validation), 63d", res_val,
                 reference)
    results["val_2021_2022_63d"] = {
        "windows": summarize_window_count(len(S_val), 63, 10),
        "metadata": tables["val_2021_2022"],
        "strategies": res_val}

    # --- test set 3: COVID crash stress, model never saw it ---
    S_cov, cov_starts = window_paths_in_range(
        dates, prices, COVID_START, COVID_END, 30, stride=1, s0=V0)
    print(f"\nCOVID stress: {len(S_cov)} contiguous 30d paths from "
          f"{cov_starts[0]} -> {cov_starts[-1]}", flush=True)
    res_cov = paired_evaluate(strategies, S_cov, rollout, FLOOR,
                              reference=reference)
    _print_table("COVID crash Feb-Apr 2020, 30d", res_cov, reference)
    results["covid_crash_30d"] = {
        "window_starts": cov_starts,
        "windows": summarize_window_count(len(S_cov), 30, 1),
        "metadata": tables["covid_crash"],
        "strategies": res_cov}

    # --- reference: same strategies on in-sample bootstrap (old method) ---
    S_boot = train_sampler(5000, seed + 999)
    res_boot = paired_evaluate(strategies, S_boot, rollout, FLOOR,
                               reference=reference)
    _print_table("in-sample bootstrap (old method), 63d", res_boot,
                 reference)
    results["insample_bootstrap_63d"] = {
        "metadata": tables["insample_bootstrap"],
        "strategies": res_boot}

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUTDIR}/metrics.json", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Chronological out-of-time CPPI evaluation")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_chrono_eval(n_epochs=args.epochs, batch_size=args.batch_size,
                    seed=args.seed)


if __name__ == "__main__":
    main()
