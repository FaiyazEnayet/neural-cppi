"""Baseline capture for the budget-utilization policy (R1, Part 1).

Runs the deterministic risk-budget controllers (cash, classic CPPI,
constant fractions, vol threshold, cushion ramp) over 20 seeded
synthetic GBM scenarios and reports return, shortfall, breach and
fees per controller, plus the u = 1 parity check against classical
CPPI, the independent-oracle agreement, and the self-financing
violation count. Baseline capture only: no training run, no policy
promotion, no performance claim.

Usage:
    python scripts/experiments/budget_utilization.py [--out report.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "src"))

import torch  # noqa: E402

from dhl.experiments.cppi_learned import DT, R  # noqa: E402
from dhl.markets.gbm import simulate_gbm  # noqa: E402
from dhl.policies.budget_allocator import baseline_report  # noqa: E402

SEEDS = [1000 + k for k in range(20)]
N_PATHS = 16
N_STEPS = 63
SIGMA = 0.30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="optional path for the JSON report")
    args = parser.parse_args()

    scenarios = []
    for seed in SEEDS:
        S = simulate_gbm(N_PATHS, N_STEPS, N_STEPS * DT, 100.0, R,
                         SIGMA, seed=seed)
        scenarios.append((seed, torch.tensor(S, dtype=torch.float64)))

    report = baseline_report(scenarios, multiplier=4.0)
    report["scenario_config"] = {
        "generator": "dhl.markets.gbm.simulate_gbm",
        "n_paths": N_PATHS,
        "n_steps": N_STEPS,
        "sigma": SIGMA,
        "drift": R,
        "s0": 100.0,
    }

    print(f"{'controller':<16} {'mean V_T':>9} {'return':>8} "
          f"{'shortfall':>10} {'breachT':>8} {'breachA':>8} {'fees':>7}")
    for row in report["controllers"]:
        print(f"{row['controller']:<16} "
              f"{row['mean_terminal_wealth']:>9.4f} "
              f"{row['mean_return']:>8.4f} "
              f"{row['mean_shortfall']:>10.4f} "
              f"{row['breach_count_terminal']:>8d} "
              f"{row['breach_count_anytime']:>8d} "
              f"{row['mean_fees']:>7.4f}")
    print(f"parity max holding diff: "
          f"{report['parity']['max_holding_diff']:.3g}")
    print(f"parity max fee diff: {report['parity']['max_fee_diff']:.3g}")
    print(f"oracle max terminal diff: "
          f"{report['oracle']['max_terminal_diff']:.3g}")
    print(f"self-financing violations: "
          f"{report['self_financing_violations']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
