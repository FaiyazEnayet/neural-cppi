# Changelog

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
