"""Experiments for the budget-action candidate comparison (E5).

Default mode is the Part 1 baseline capture: the four predeclared
candidates (baseline MLP, linear/logistic policy, residual CPPI
utilization network, dynamic-rule utilization baseline), all
untrained, plus the cash and classic CPPI references, over the
declared baseline family (the R1 baseline seeds, GBM sigma 0.30,
16 paths x 63 steps), reporting the full acceptance field set per
candidate: terminal wealth and return, terminal and anytime
shortfall, terminal and anytime breach, drawdown, costs, negative
outcomes, fraction statistics, regulatory activation shares, and
the self-financing audit. The deterministic reference rows must
reproduce the R1 baseline capture on the same family. No training,
no selection, no performance claim; training and selection arrive
with Part 2.

Usage:
    python scripts/experiments/budget_candidates.py [--out report.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "src"))

from dhl.policies.candidate_comparison import (  # noqa: E402
    CANDIDATE_ORDER,
    CANDIDATE_SEED,
    EPISODE_PROTOCOL,
    E5_FEATURE_NAMES,
    E5_SEED_SETS,
    MULTIPLIER,
    REFERENCE_ORDER,
    candidate_fraction_fn,
    candidate_specs,
    e5_scenarios,
    evaluate_candidates,
    reference_fraction_fn,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="optional path for the JSON report")
    args = parser.parse_args()

    scenarios = e5_scenarios("baseline")
    candidates = {}
    for name in CANDIDATE_ORDER:
        candidates[name] = candidate_fraction_fn(name)
    for name in REFERENCE_ORDER:
        candidates[name] = reference_fraction_fn(name)

    report = evaluate_candidates(scenarios, candidates,
                                 multiplier=MULTIPLIER)
    report["experiment"] = "budget_action_candidate_comparison_part1"
    report["protocol"] = {
        "candidates": candidate_specs(),
        "candidate_seed": CANDIDATE_SEED,
        "feature_names": list(E5_FEATURE_NAMES),
        "feature_protocol": "cold (causal within each path)",
        "seed_sets": {k: [int(s) for s in v]
                      for k, v in E5_SEED_SETS.items()},
        "episode_protocol": EPISODE_PROTOCOL,
        "scenario_config": {
            "generator": "dhl.markets.gbm.simulate_gbm",
            "n_paths": 16,
            "n_steps": 63,
            "sigma": 0.30,
            "drift": "R (cash rate 0.02)",
            "s0": 100.0,
            "comparability": "baseline family shared with the R1 "
                             "capture; deterministic rows must "
                             "reproduce it",
        },
    }

    print(f"{'candidate':<22} {'mean V_T':>9} {'return':>8} "
          f"{'short T':>8} {'breachT':>8} {'breachA':>8} "
          f"{'short A':>8} {'drawdn':>7} {'fees':>7} {'neg':>4} "
          f"{'mean u':>7} {'capAct':>7}")
    for row in report["candidates"]:
        print(f"{row['candidate']:<22} "
              f"{row['mean_terminal_wealth']:>9.4f} "
              f"{row['mean_return']:>8.4f} "
              f"{row['mean_terminal_shortfall']:>8.4f} "
              f"{row['breach_count_terminal']:>8d} "
              f"{row['breach_count_anytime']:>8d} "
              f"{row['mean_anytime_shortfall']:>8.4f} "
              f"{row['mean_max_drawdown']:>7.4f} "
              f"{row['mean_fees']:>7.4f} "
              f"{row['n_negative_terminal']:>4d} "
              f"{row['mean_fraction']:>7.4f} "
              f"{row['cap_activation_share']:>7.4f}")
    print(f"self-financing violations: "
          f"{report['self_financing_violations']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")
    return report


if __name__ == "__main__":
    main()
