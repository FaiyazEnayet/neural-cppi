"""neural-cppi: neural networks for portfolio insurance.

A small research lab for a single question: can a neural network learn
to protect a portfolio better than the textbook rule? The lab has three
parts: price scenario generators (GBM, Heston stochastic volatility,
Merton jumps, and a block bootstrap over real SPY history), convex risk
measures usable as training objectives, and a learned dynamic allocator
that guards a wealth floor under transaction costs. No options, no
derivatives of any kind. Just a portfolio, a floor, and scenarios.
"""

__version__ = "1.3.0"

from dhl.markets import (
    block_bootstrap_paths,
    fetch_spy_daily_yahoo,
    load_spy_daily,
    log_returns,
    simulate_gbm,
    simulate_heston,
    simulate_merton,
)
from dhl.risk import cvar, entropic_risk, variance
from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    FLOOR,
    V0,
    CPPIPolicy,
    classic_cppi_weight,
    evaluate as evaluate_cppi,
    rollout as cppi_rollout,
    run_experiment as run_cppi_experiment,
    train_cppi_policy,
)
from dhl.experiments.cppi_multi import (
    MAX_DD,
    TICKERS,
    MultiAssetPolicy,
    block_bootstrap_paths_multi,
    evaluate_multi,
    rollout_multi,
    run_all as run_cppi_multi_experiment,
    train_multi_policy,
)

__all__ = [
    "simulate_gbm",
    "simulate_heston",
    "simulate_merton",
    "block_bootstrap_paths",
    "fetch_spy_daily_yahoo",
    "load_spy_daily",
    "log_returns",
    "cvar",
    "entropic_risk",
    "variance",
    "CAP",
    "COST_RATE",
    "FLOOR",
    "V0",
    "CPPIPolicy",
    "classic_cppi_weight",
    "evaluate_cppi",
    "cppi_rollout",
    "run_cppi_experiment",
    "train_cppi_policy",
    "MAX_DD",
    "TICKERS",
    "MultiAssetPolicy",
    "block_bootstrap_paths_multi",
    "evaluate_multi",
    "rollout_multi",
    "run_cppi_multi_experiment",
    "train_multi_policy",
]
