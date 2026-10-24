"""Exact trade accounting shared by every backtest rollout.

One convention, used by the single-asset and multi-asset rollouts
alike: a rebalance moves the portfolio to the desired post-fee
weights of post-fee wealth. Post-fee wealth ``V'`` solves

    V' = V - cost_rate * sum_j |w_j * V' - H_j|

where ``H_j`` is the pre-trade dollar holding of asset ``j`` and
``w_j`` its desired post-fee weight. The fee is charged on the
actual traded notional, never on a pre-fee target, and a trade that
does not happen costs nothing. Cash takes the remaining weight, so
a long-only portfolio (weights nonnegative, summing to at most one)
never borrows: cash after the trade is nonnegative by construction.

The single-asset closed form (buy/sell branches) is the one-asset
case of the same equation; the multi-asset solver finds the same
root by scanning the breakpoints where each asset's trade changes
sign. Both are differentiable in wealth, holdings and weights, so
learned policies can still train through the rollout.
"""

from dhl.backtest.execution import (
    rebalance_multi,
    rebalance_single,
    solve_post_fee_wealth,
)

__all__ = ["rebalance_multi", "rebalance_single", "solve_post_fee_wealth"]
