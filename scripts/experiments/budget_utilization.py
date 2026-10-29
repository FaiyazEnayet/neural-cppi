"""Experiments for the budget-utilization policy (R1).

Default mode is the Part 1 baseline capture: the deterministic
risk-budget controllers (cash, classic CPPI, constant fractions, vol
threshold, cushion ramp) over 20 seeded synthetic GBM scenarios,
reporting return, shortfall, breach and fees per controller, plus
the u = 1 parity check against classical CPPI, the
independent-oracle agreement, and the self-financing violation
count.

With --part2, the Part 2 experiment: the de-risking penalty is
selected on the declared training/validation protocol, residual and
direct budget-fraction networks are trained at the selected penalty,
and the learned policies are evaluated against the deterministic
controllers on three held-out or comparability families (baseline,
stress, gap; see dhl.policies.budget_allocator.SEED_SETS). No
required gain: outcomes are reported as measured.

Usage:
    python scripts/experiments/budget_utilization.py [--out report.json]
    python scripts/experiments/budget_utilization.py --part2 \
        [--epochs 150] [--out report.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "src"))

import torch  # noqa: E402

from dhl.experiments.cppi_learned import (  # noqa: E402
    COST_RATE,
    DT,
    FLOOR_PENALTY,
    R,
)
from dhl.markets.gbm import simulate_gbm  # noqa: E402
from dhl.policies.budget_allocator import (  # noqa: E402
    GAP_JUMP,
    SEED_SETS,
    STRESS_DRIFT,
    STRESS_SIGMA,
    BudgetFractionNet,
    baseline_report,
    budget_experiment_scenarios,
    deterministic_controllers,
    evaluate_fraction_controllers,
    net_fraction_fn,
    select_derisk_penalty,
    train_budget_fraction_net,
)

SEEDS = [1000 + k for k in range(20)]
N_PATHS = 16
N_STEPS = 63
SIGMA = 0.30

# Part 2 protocol (fixed before any run; the evaluation families are
# never used for training or selection).
PART2_CANDIDATES = [0.0, 1.0, 4.0]
PART2_LR = 3e-3
PART2_SEED = 7
PART2_MULTIPLIER = 4.0


def _print_table(title, report):
    print(f"--- {title} ---")
    print(f"{'controller':<18} {'mean V_T':>9} {'return':>8} "
          f"{'shortfall':>10} {'breachT':>8} {'breachA':>8} "
          f"{'fees':>7} {'mean u':>7}")
    for row in report["controllers"]:
        print(f"{row['controller']:<18} "
              f"{row['mean_terminal_wealth']:>9.4f} "
              f"{row['mean_return']:>8.4f} "
              f"{row['mean_shortfall']:>10.4f} "
              f"{row['breach_count_terminal']:>8d} "
              f"{row['breach_count_anytime']:>8d} "
              f"{row['mean_fees']:>7.4f} "
              f"{row['mean_fraction']:>7.4f}")
    print(f"self-financing violations: "
          f"{report['self_financing_violations']}")


def run_part2(epochs, out):
    sets = {name: budget_experiment_scenarios(name)
            for name in ("training", "validation", "baseline",
                         "stress", "gap")}
    print(f"selecting the de-risking penalty from {PART2_CANDIDATES} "
          f"({epochs} epochs per candidate, residual form)",
          flush=True)
    selection = select_derisk_penalty(
        PART2_CANDIDATES, sets["training"], sets["validation"],
        form="residual", n_epochs=epochs, lr=PART2_LR, seed=PART2_SEED)
    lam = selection["selected_derisk_penalty"]
    print(f"{'lambda':>7} {'val objective':>14} {'val V_T':>9} "
          f"{'val short':>10} {'val breachT':>12} {'val mean u':>11}")
    for row in selection["table"]:
        print(f"{row['derisk_penalty']:>7.1f} "
              f"{row['validation_objective']:>14.4f} "
              f"{row['mean_terminal_wealth']:>9.4f} "
              f"{row['mean_shortfall']:>10.4f} "
              f"{row['breach_count_terminal']:>12d} "
              f"{row['mean_fraction']:>11.4f}")
    print(f"selected derisk penalty: {lam}", flush=True)

    nets, histories = {}, {}
    torch.manual_seed(PART2_SEED)
    nets["untrained_residual"] = BudgetFractionNet(form="residual")
    for form in ("residual", "direct"):
        torch.manual_seed(PART2_SEED)
        net = BudgetFractionNet(form=form)
        histories[form] = train_budget_fraction_net(
            net, sets["training"], n_epochs=epochs, lr=PART2_LR,
            derisk_penalty=lam, seed=PART2_SEED)
        nets[f"learned_{form}"] = net
        first, last = histories[form][0], histories[form][-1]
        print(f"trained {form}: train loss {first['loss']:.4f} -> "
              f"{last['loss']:.4f}, mean fraction "
              f"{first['mean_fraction']:.4f} -> "
              f"{last['mean_fraction']:.4f}", flush=True)

    controllers = dict(deterministic_controllers())
    for name, net in nets.items():
        controllers[name] = net_fraction_fn(net)

    report = {
        "experiment": "budget_utilization_part2",
        "config": {
            "multiplier": PART2_MULTIPLIER,
            "floor_penalty": FLOOR_PENALTY,
            "cost_rate": COST_RATE,
            "epochs": epochs,
            "lr": PART2_LR,
            "seed": PART2_SEED,
            "derisk_candidates": PART2_CANDIDATES,
            "selected_derisk_penalty": lam,
            "n_paths_per_scenario": 16,
            "n_steps": 63,
            "seed_sets": {k: [int(s) for s in v]
                          for k, v in SEED_SETS.items()},
            "stress_sigma": STRESS_SIGMA,
            "stress_drift": STRESS_DRIFT,
            "gap_jump": GAP_JUMP,
            "training_regimes": "position modulo 3: calm GBM "
                                "(sigma 0.30, drift R), stress GBM "
                                "(sigma 0.55, drift -0.25), gap "
                                "(calm GBM, one -30% jump per path)",
        },
        "selection": selection,
        "training_history": histories,
        "families": {},
    }
    for name in ("baseline", "stress", "gap"):
        fam = evaluate_fraction_controllers(
            sets[name], controllers, multiplier=PART2_MULTIPLIER)
        report["families"][name] = fam
        _print_table(f"{name} family", fam)

    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="optional path for the JSON report")
    parser.add_argument("--part2", action="store_true",
                        help="run the Part 2 training experiment")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Part 2 training epochs per run")
    args = parser.parse_args()
    if args.part2:
        run_part2(args.epochs, args.out)
        return

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
