"""Pure trade-accounting kernel for backtest rollouts.

The convention (see ``dhl.backtest``): after a rebalance, risky
holding ``j`` equals ``w_j * V'`` and cash equals ``(1 - sum w) * V'``,
where post-fee wealth ``V'`` solves ``V' = V - c * sum_j |w_j V' - H_j|``.
The right-hand side is a monotone piecewise-linear function of ``V'``
with one breakpoint per asset (``H_j / w_j``), so the root is found
exactly by scanning intervals: on the interval after ``k`` breakpoints
the absolute values resolve to ``A * V' + B`` and the root is the
linear solve ``(V - c * B) / (1 + c * A)``. Exactly one interval
contains its own root, because the function is strictly increasing
when ``c < 1`` and the weights sum to at most one.

All functions are pure: no market data, no policy, no randomness.
They work on floats or torch tensors and stay differentiable, since
the interval choice is a mask and the selected root is a rational
function of the inputs.
"""

import torch


def solve_post_fee_wealth(wealth, holdings, weights, cost_rate):
    """Post-fee wealth for a rebalance to the given weights.

    ``wealth`` has shape ``(...)``; ``holdings`` and ``weights`` have
    shape ``(..., n_assets)`` with nonnegative holdings and weights
    summing to at most one. Returns post-fee wealth with the shape of
    ``wealth``. Requires ``wealth > 0``; rollouts maintain positive
    wealth because every trade leaves ``V' >= V * (1 - c) / (1 + c)``.
    """
    if torch.is_tensor(wealth):
        return _solve_post_fee_wealth_torch(wealth, holdings, weights,
                                            cost_rate)
    raise TypeError("solve_post_fee_wealth expects torch tensors")


def _solve_post_fee_wealth_torch(wealth, holdings, weights, cost_rate):
    # Breakpoints b_j = H_j / w_j (infinity when w_j == 0: that asset
    # is fully sold, so its term is the constant H_j on every interval).
    inf = torch.full_like(weights, float("inf"))
    breakpoints = torch.where(weights > 0, holdings / weights.clamp_min(1e-12),
                              inf)
    order = torch.argsort(breakpoints, dim=-1)
    w_sorted = torch.gather(weights, -1, order)
    h_sorted = torch.gather(holdings, -1, order)
    b_sorted = torch.gather(breakpoints, -1, order)
    total_w = weights.sum(dim=-1)
    total_h = holdings.sum(dim=-1)
    n = weights.shape[-1]

    # Interval k (0..n): the first k sorted breakpoints are at or below
    # V'. On it, sum|w_j V' - H_j| = A_k * V' + B_k with
    # A_k = sum_{<=k} w - sum_{>k} w and B_k = total_H - 2 * sum_{<=k} H.
    zeros = torch.zeros_like(total_w)
    cum_w = torch.cat([zeros.unsqueeze(-1),
                       torch.cumsum(w_sorted, dim=-1)], dim=-1)
    cum_h = torch.cat([zeros.unsqueeze(-1),
                       torch.cumsum(h_sorted, dim=-1)], dim=-1)
    a = 2.0 * cum_w - total_w.unsqueeze(-1)
    b_term = total_h.unsqueeze(-1) - 2.0 * cum_h
    candidates = (wealth.unsqueeze(-1) - cost_rate * b_term) \
        / (1.0 + cost_rate * a)
    lo = torch.cat([zeros.unsqueeze(-1), b_sorted], dim=-1)
    hi = torch.cat([b_sorted, inf[..., :1]], dim=-1)
    # A root sitting exactly on a breakpoint is valid in both adjacent
    # intervals, and float rounding can push either side's candidate a
    # few ulps outside its interval, so widen by a few dozen ulps and
    # keep only the FIRST valid interval. The widening affects
    # selection only: the selected candidate is still a formula output
    # within a few ulps of the true root.
    tol = 64.0 * torch.finfo(candidates.dtype).eps \
        * (1.0 + wealth.abs()).unsqueeze(-1)
    valid = (candidates >= lo - tol) & (candidates <= hi + tol)
    shifted = torch.cat([torch.zeros_like(valid[..., :1]), valid[..., :-1]],
                        dim=-1)
    first = valid & ~shifted
    return (candidates * first.to(candidates.dtype)).sum(dim=-1)


def rebalance_single(wealth, holding, weight, cost_rate):
    """Exact self-financing rebalance to a target risky weight.

    The target weight applies to post-trade wealth, so the new risky
    holding ``H'`` solves ``H' = w * (V - c * |H' - H|)`` with one
    branch per trade direction (decided by comparing ``w`` with the
    current weight ``H / V``; both branches agree at ``w == H / V``,
    where the trade is zero):

        buy  (w >= H/V):  H' = w (V + c H) / (1 + w c)
        sell (w <  H/V):  H' = w (V - c H) / (1 - w c)

    Returns ``(holding_post, fee)`` with ``fee = c * |H' - H|``
    charged on the actual trade. Differentiable in wealth, holding
    and weight. This is the one-asset case of ``solve_post_fee_wealth``.
    """
    current = holding / wealth.clamp_min(1e-8)
    buy = weight * (wealth + cost_rate * holding) \
        / (1.0 + weight * cost_rate)
    sell = weight * (wealth - cost_rate * holding) \
        / (1.0 - weight * cost_rate).clamp_min(1e-9)
    post = torch.where(weight >= current, buy, sell)
    fee = cost_rate * (post - holding).abs()
    return post, fee


def rebalance_multi(wealth, holdings, weights, cost_rate):
    """Exact self-financing rebalance to target risky weights.

    ``holdings`` and ``weights`` have shape ``(..., n_assets)``;
    weights are the desired post-fee risky weights (nonnegative,
    summing to at most one). Returns ``(holdings_post, cash_post, fee,
    wealth_post)`` where ``holdings_post = weights * wealth_post``,
    ``cash_post = (1 - sum weights) * wealth_post``, ``fee`` is
    charged on the actual traded notional ``sum |holdings_post -
    holdings|``, and ``wealth_post = wealth - fee``. Cash is
    nonnegative whenever the weights satisfy the long-only condition,
    so a long-only rebalance never borrows implicitly.
    """
    wealth_post = solve_post_fee_wealth(wealth, holdings, weights,
                                        cost_rate)
    holdings_post = weights * wealth_post.unsqueeze(-1)
    fee = cost_rate * (holdings_post - holdings).abs().sum(dim=-1)
    cash_post = (1.0 - weights.sum(dim=-1)) * wealth_post
    return holdings_post, cash_post, fee, wealth_post
