# Changelog

## [Unreleased]
### Fixed
- `rollout_multi` in `src/dhl/experiments/cppi_multi.py` now routes every rebalance through the shared trade-accounting kernel in `src/dhl/backtest/execution.py`: holdings move to the target weights of post-fee wealth and the fee is charged on the actual traded notional, so a fully invested path can no longer end a date with negative cash from fees alone. The old code charged the fee on the pre-fee target minus the grown old holding and then set the holding to the pre-fee target, so executed and charged notionals disagreed. The single-asset `rollout` now imports the same kernel's closed form with unchanged numbers (a golden fixture pins the previous terminal wealths exactly). The kernel solves the post-fee wealth equation exactly by scanning trade-sign breakpoints, stays differentiable for training, and keeps cash nonnegative for long-only weights. New tests: `tests/test_execution.py` (exact accounting identity on random fixtures, analytic buy/sell costs, all-cash charging no fees, an independent NumPy reference for a multi-asset rollout, float64/float32 agreement, jump properties, finite gradients, and the single-asset golden fixture).

### Docs
- Centered the README header: uniform dark badge labels and three action buttons in the dashboard palette (Open Dashboard, Explore Method, Run Locally). The XERJ Corpus Hub contributor badge stays in the badge row.

### Changed
- `rollout` in `src/dhl/experiments/cppi_learned.py` now keeps an explicit holdings ledger (cash, risky units, traded notional, fees) instead of a wealth shortcut: each rebalance solves the exact proportional-cost equation so the target weight holds on post-trade wealth and every fee is charged on the actual trade, and the terminal liquidation is charged on the true terminal holding value. The leverage cap and the funding assumption are now parameters (`cap=CAP`, `borrow_rate=None`, which borrows at the cash rate as before); existing callers and signatures are unchanged. Exact self-financing moves recorded experiment numbers slightly (the old approximation overcharged buys and the stale-value liquidation, undercharged sells); committed results were not regenerated. New fixture: `tests/test_stage3_neural_cppi_3_ledger.py` (self-financing identity at every trade, an independent scalar reference, and zero-return / buy-and-hold / cap=1 / cap>1 / fee-only property cases).

### Fixed
- The multi-horizon training loss now lives in `src/dhl/experiments/losses.py` as explicit named components (negative expected terminal wealth, terminal floor shortfall, anytime floor shortfall, turnover), each reported alongside the total. The anytime (pathwise) floor-shortfall term now ADDS to the minimization loss; a private-experiment variant of this objective previously subtracted it, so the two candidates trained with a positive anytime weight were corrupted tests of the anytime-shortfall hypothesis. Zero anytime weight reproduces the previous terminal-only objective exactly. New tests: `tests/test_losses.py` (shortfall raises the loss, finite-difference gradient matches autograd, zero-weight equivalence, finite gradients). The affected candidates are being re-evaluated privately with the corrected sign under identical training budgets; no results are published from this change.

## [v1.2.2] - 2026-10-18
### Fixed
- Policy construction is now seeded *before* initialization everywhere it is trained: new `seeded_policy` helper (`torch.manual_seed(seed)` runs before the network is built) replaces bare `CPPIPolicy()` / `MultiAssetPolicy()` construction in the chronological, crash-ablation and multi-asset runners. Equal seeds now guarantee equal initializations regardless of construction order or preceding RNG state; the training functions still re-seed the torch stream at train start, and sampler/evaluation draws keep their separate `seed + epoch` / `seed + 999` numpy seeds. New `policy_init_hash` helper records a short SHA-256 fingerprint of the initial weights (logged by the runners). Regression tests: init hash depends only on the seed across polluted RNG states, and two fresh processes produce identical init and post-training hashes under deterministic CPU settings.

## [v1.2.1] - 2026-10-18
### Fixed
- `rollout` and `rollout_multi` now write the post-liquidation wealth into the final path point, so `path[:, -1] == V_T` for every path. Previously the liquidation cost was subtracted from `V_T` after the path was closed, so terminal and anytime breach statistics could disagree at the horizon. Regression tests: terminal-point equality on random paths, a hand-computed fixture where wealth sits just above the floor before liquidation but below it after (the anytime breach now registers), and a zero-cost fixture confirming the economic path is unchanged.

## [v1.2.0] - 2026-10-17
### Added
- Chronological out-of-time evaluation: strict train (pre-2020) / validate (2021-2022) / test (2023+) splits with contiguous real-history paths, COVID crash stress set, and paired strategy comparison on identical paths.
- Crash-robustness upgrade: 2008 crisis training data, crash-regime features (volatility ratio, drawdown, panic indicator, lagged VIX), crash-weighted breach penalty, and a CPPI-constrained allocator where the network picks assets and a classical risk layer caps total exposure.
- Full ablation study (A-G plus leave-one-out) proving each component's contribution.

### Results
- COVID crash (held out of training): breach rate 24% → 0%, terminal wealth 97.03 → 98.37, beating classic CPPI (96.59).
- 2008 training data alone was the biggest single improvement.

### Tests
- 78 passing (was 52), including causality regression tests for all new features.

## [v1.1.0] - 2026-10-16

### Fixed
- Look-ahead leakage in the decision-time features (`_vol_features` in `src/dhl/experiments/cppi_learned.py`, `_trail_features` in `src/dhl/experiments/cppi_multi.py`). Both builders included the return from price `i` to price `i+1` in the feature vector used to choose the allocation at time `i`. They now use strictly past returns only, and the empty first-step window yields exactly `0.0`. Added regression tests pinning temporal causality (perturbing price `i+1` leaves features at times `<= i` unchanged) and the exact-zero first step.
- Reran all experiments on the corrected features and updated the README tables. Corrected single-asset 63d learned numbers: E[wealth] 102.00 to 101.70, std 4.28 to 3.63, P(breach) 0.004 to 0.003. Multi-asset and cross-decade numbers moved only in the third decimal. The conclusions are unchanged.
