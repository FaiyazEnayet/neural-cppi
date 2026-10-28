"""Multi-asset learned portfolio insurance.

Three extensions over the single-asset CPPI experiment in
``cppi_learned.py``:

1. Multi-asset allocation. The policy outputs softmax weights over N
   risky assets plus cash. Softmax means long-only and at most fully
   invested: no leverage, no shorting, by construction.
2. Drawdown floors. Instead of a fixed floor, the floor ratchets with
   the running wealth peak: floor_t = (1 - max_dd) * peak_t. Stricter
   and more realistic than a fixed floor.
3. Cross-decade stress test. Train on pre-2020 bootstrapped paths,
   evaluate on 2020+ paths that include the COVID crash: an honest
   out-of-distribution test.

Stocks and cash only. No derivatives, no shorting, no leverage
anywhere. The new comparisons below all use a max risky allocation
of 1.0 (no leverage); the original single-asset README numbers used
a 1.5 cap and are kept separately.

Run: ``python -m dhl.experiments.cppi_multi [--epochs 100]``

Writes ``runs/cppi_multi/metrics.json`` and prints comparison tables.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from dhl.backtest.execution import rebalance_multi, validate_portfolio_weights
from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    DT,
    FLOOR,
    R,
    V0,
    _vol_features,
    classic_cppi_weight,
    policy_init_hash,
    seeded_policy,
)
from dhl.markets.real import block_bootstrap_paths, load_aligned_returns

TICKERS = ("SPY", "TLT", "GLD")
MIX = (1 / 3, 1 / 3, 1 / 3)  # fixed-mix risky leg for the classic baseline
BLOCK_LEN = 21
N_EPOCHS = 100
BATCH_SIZE = 1024
N_TEST = 10000
FLOOR_PENALTY = 30.0
MAX_DD = 0.10  # drawdown floor: floor_t = 0.90 * running peak
SEED = 11

OUTDIR = os.path.join("runs", "cppi_multi")


def block_bootstrap_paths_multi(rets, tickers, n_paths, path_len,
                                block_len=BLOCK_LEN, S0=V0, seed=None):
    """Joint block bootstrap of several assets' log returns.

    The same block start indices are used for every asset, so the
    cross-asset correlation structure of each block is preserved.
    Returns prices of shape ``(n_paths, path_len + 1, n_assets)``,
    each asset normalized to ``S0``.
    """
    rs = [np.asarray(rets[t], dtype=float) for t in tickers]
    n = min(len(r) for r in rs)
    R = np.stack([r[:n] for r in rs], axis=1)  # (n, n_assets)
    if n < block_len:
        raise ValueError(f"need at least block_len={block_len} returns")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(path_len / block_len))
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    offsets = np.arange(block_len)[None, None, :]
    idx = (starts[:, :, None] + offsets) % n
    blocks = R[idx]  # (n_paths, n_blocks, block_len, n_assets)
    sampled = blocks.reshape(n_paths, n_blocks * block_len, -1)[:, :path_len, :]
    P = np.empty((n_paths, path_len + 1, len(tickers)))
    P[:, 0, :] = S0
    P[:, 1:, :] = S0 * np.exp(np.cumsum(sampled, axis=1))
    return P


def _trail_features(P, window=21):
    """Trailing mean log return per asset, shape (n_paths, steps, n_assets).

    Strictly causal: the feature at time ``i`` uses only returns
    ``lr[:, :i, :]``. The first step has no history yet, so its
    feature is 0.0.
    """
    n, steps_p1, na = P.shape
    steps = steps_p1 - 1
    lr = torch.log(P[:, 1:, :] / P[:, :-1, :])
    out = torch.empty_like(lr)
    for i in range(steps):
        j = max(0, i - window)
        w = lr[:, j:i, :]
        if w.shape[1] == 0:
            out[:, i, :] = 0.0
        else:
            out[:, i, :] = w.mean(dim=1)
    return out


class MultiAssetPolicy(nn.Module):
    """MLP: (tau_frac, cushion_ratio, vols[N], trail_rets[N]) -> softmax(N+1).

    The last output is the cash weight. Softmax keeps every weight in
    [0, 1] and summing to one: long-only, no leverage, by construction.
    """

    def __init__(self, n_assets, hidden_dims=(32, 32)):
        super().__init__()
        self.n_assets = n_assets
        dim = 2 + 2 * n_assets
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.Tanh()]
            dim = h
        layers.append(nn.Linear(dim, n_assets + 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tau_frac, cushion_ratio, vols, trail):
        x = torch.cat(
            [
                torch.as_tensor(tau_frac, dtype=torch.float32).reshape(-1, 1),
                cushion_ratio.reshape(-1, 1),
                vols,
                trail,
            ],
            dim=1,
        )
        return torch.softmax(self.net(x), dim=1)


def _floor_series(V_path, floor_mode, floor, max_dd):
    """Floor value at each date given the wealth path so far.

    ``fixed``: constant floor. ``drawdown``: (1 - max_dd) times the
    running peak, ratcheting up and never down.
    """
    if floor_mode == "fixed":
        return torch.full_like(V_path, floor)
    peak = torch.cummax(V_path, dim=1).values
    return (1.0 - max_dd) * peak


def rollout_multi(P, weight_fn, floor_mode="fixed", floor=FLOOR,
                  max_dd=MAX_DD, r=R, dt=DT, cost_rate=COST_RATE,
                  return_path=False):
    """Self-financing multi-asset portfolio under a dynamic allocation.

    P has shape (n_paths, n_steps + 1, n_assets), each asset starting
    at the same price. weight_fn(i, V, tau_frac, cushion, vols_i,
    trail_i, floor_t) returns (n_paths, n_assets + 1) weights, last
    column cash, nonnegative and summing to one. Returns terminal
    wealth per path; with return_path=True also the wealth and floor
    paths. The final wealth path point is post-liquidation wealth,
    so W[:, -1] == V_T exactly.

    Accounting is an explicit (cash, holdings) ledger, not a wealth
    shortcut. Each rebalance goes through the shared kernel
    ``dhl.backtest.execution.rebalance_multi``: holdings move to the
    target weights of post-fee wealth and the fee is charged on the
    actual traded notional, so a fully invested path never ends a
    date with negative cash from fees alone. Weights are validated
    fail-closed every step (nonnegative, summing to one): the
    long-only contract is enforced, not assumed, so no weight
    function can silently short or borrow here; there is deliberately
    no borrow-rate parameter, because an enforced long-only rollout
    never borrows. Between dates cash earns ``r`` (cash is always
    nonnegative for validated long-only weights) and holdings ride
    the price moves. At the horizon the risky leg is liquidated at
    the terminal prices and charged on its true terminal value.

    Floor series indexing (return_path=True): ``F[j]`` is the floor
    governing date-``j`` wealth, i.e. the floor set by the peak
    through date ``j - 1``; ``F[0]`` is the date-0 floor itself
    (the fixed floor in fixed mode, ``(1 - max_dd) * V0`` in
    drawdown mode). The terminal comparison uses ``F[steps]``, the
    floor at the last trade date: it agrees with the horizon floor
    in every case, because a final move to a new peak leaves
    post-liquidation wealth above 0.9 times that peak whenever the
    fee is below 10% of wealth, and otherwise the peak is unchanged.
    """
    n, steps_p1, na = P.shape
    steps = steps_p1 - 1
    T = steps * dt
    device, dtype = P.device, P.dtype
    cash = torch.full((n,), V0, device=device, dtype=dtype)
    D = torch.zeros((n, na), device=device, dtype=dtype)
    growth = math.exp(r * dt)
    rets = P[:, 1:, :] / P[:, :-1, :]
    vols = torch.stack(
        [_vol_features(P[:, :, k]) for k in range(na)], dim=2
    )
    trail = _trail_features(P)
    V = cash.clone()
    wealth_path = [V.clone()] if return_path else None
    floor_vals = []

    for i in range(steps):
        peak = V.clone() if i == 0 else torch.maximum(peak, V)
        if floor_mode == "fixed":
            floor_t = torch.full_like(V, floor)
        else:
            floor_t = (1.0 - max_dd) * peak
        floor_vals.append(floor_t.clone())
        tau_frac = (T - i * dt) / T
        cushion = torch.clamp((V - floor_t) / V.clamp_min(1e-8), -1.0, 1.0)
        w = weight_fn(i, V, tau_frac, cushion, vols[:, i, :],
                      trail[:, i, :], floor_t)
        validate_portfolio_weights(w)
        w_risky = w[:, :na]
        holdings = D if i == 0 else D * rets[:, i - 1, :]
        V = cash + holdings.sum(dim=1)
        D, cash_post, _fee, _Vp = rebalance_multi(V, holdings, w_risky,
                                                  cost_rate)
        cash = cash_post * growth
        V = cash + (D * rets[:, i, :]).sum(dim=1)
        if return_path:
            wealth_path.append(V.clone())

    holdings_T = D * rets[:, steps - 1, :]
    V = cash + holdings_T.sum(dim=1) - cost_rate * holdings_T.abs().sum(dim=1)
    if return_path:
        # Same terminal convention as rollout: the final path point is
        # post-liquidation wealth, so W[:, -1] == V_T exactly.
        wealth_path[-1] = V.clone()
        W = torch.stack(wealth_path, dim=1)
        # F[0] is the date-0 floor itself (floor_vals[0]), not the
        # fixed-floor parameter: in drawdown mode those differ
        # whenever floor != (1 - max_dd) * V0.
        F = torch.stack([floor_vals[0]] + floor_vals, dim=1)
        return V, W, F
    return V


def _learned_multi_weight_fn(policy):
    def fn(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        t = torch.full_like(V, tau_frac)
        return policy(t, cushion, vols_i, trail_i)

    return fn


def _classic_multi_weight_fn(mix=MIX, multiplier=4.0, cap=1.0):
    """Classic CPPI where the risky leg is a fixed-mix portfolio.

    Cushion is invested at ``multiplier * cushion / V`` (capped at 1.0:
    no leverage) and split across assets by ``mix``; the rest is cash.
    """
    mix_t = torch.as_tensor(mix, dtype=torch.float32)

    def fn(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        w_risky = classic_cppi_weight(
            V, floor=floor_t, multiplier=multiplier, cap=cap)
        if not torch.is_tensor(w_risky):
            w_risky = torch.full_like(V, w_risky)
        w = torch.zeros(V.shape[0], len(mix) + 1,
                        device=V.device, dtype=V.dtype)
        w[:, : len(mix)] = w_risky.unsqueeze(1) * mix_t.to(V.device)
        w[:, -1] = 1.0 - w_risky
        return w

    return fn


def _learned_single_weight_fn(policy, cap=1.0):
    """Single-asset learned policy wrapped for the multi-asset rollout.

    The policy sees only the first asset (SPY); everything goes to
    asset 0 or cash. Lets the single-asset learner compete on the
    same joint test paths.
    """
    from dhl.experiments.cppi_learned import _learned_weight_fn as _single

    inner = _single(policy)

    def fn(i, V, tau_frac, cushion, vols_i, trail_i, floor_t):
        # Single-asset policy sees only SPY (asset 0). The feats dict
        # carries vol; crash-feature policies need the multi rollout
        # to supply the full feature set (not yet wired here).
        feats = {"vol": vols_i[:, 0]}
        w0 = torch.clamp(inner(i, V, tau_frac, cushion, feats), max=cap)
        na = vols_i.shape[1]
        w = torch.zeros(V.shape[0], na + 1, device=V.device, dtype=V.dtype)
        w[:, 0] = w0
        w[:, -1] = 1.0 - w0
        return w

    return fn


def train_multi_policy(policy, sampler, horizon_days, floor_mode="fixed",
                       n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=1e-3,
                       floor_penalty=FLOOR_PENALTY, seed=SEED, log_every=25):
    """Train on fresh joint-bootstrapped paths every epoch.

    Objective: maximize E[V_T] - penalty * E[worst shortfall vs the
    (possibly ratcheting) floor along the path].
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for epoch in range(n_epochs):
        P_np = sampler(batch_size, seed + epoch)
        P = torch.as_tensor(P_np, dtype=torch.float32)
        V_T, W, F = rollout_multi(
            P, _learned_multi_weight_fn(policy),
            floor_mode=floor_mode, return_path=True)
        shortfall = torch.relu(F - W).amax(dim=1)
        objective = -(V_T.mean() - floor_penalty * shortfall.mean())
        optimizer.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        history.append(float(objective.detach()))
        if log_every and (epoch + 1) % log_every == 0:
            with torch.no_grad():
                pb = float((shortfall > 0).float().mean())
            print(
                f"epoch {epoch + 1:4d}/{n_epochs}  objective={history[-1]:.4f}  "
                f"E[V_T]={float(V_T.mean()):.2f}  P(breach)={pb:.3f}",
                flush=True,
            )
    return history


def evaluate_multi(weight_fn, P_np, floor_mode="fixed", floor=FLOOR,
                   max_dd=MAX_DD):
    """Metrics on held-out paths: wealth stats and breach rates.

    Breach is measured against the (possibly ratcheting) floor, both
    at the terminal date and at any point along the path.
    """
    P = torch.as_tensor(np.asarray(P_np), dtype=torch.float32)
    with torch.no_grad():
        V_T, W, F = rollout_multi(
            P, weight_fn, floor_mode=floor_mode, floor=floor,
            max_dd=max_dd, return_path=True)
    V_T = V_T.numpy()
    W, F = W.numpy(), F.numpy()
    breached_any = (W < F).any(axis=1)
    return {
        "mean_wealth": float(V_T.mean()),
        "std_wealth": float(V_T.std()),
        "p_breach_terminal": float((V_T < F[:, -1]).mean()),
        "p_breach_anytime": float(breached_any.mean()),
        "mean_min_wealth": float(W.min(axis=1).mean()),
    }


def _print_row(name, m):
    print(
        f"{name:22s}  E[V_T]={m['mean_wealth']:7.2f}  "
        f"sd={m['std_wealth']:6.2f}  "
        f"P(breach)={m['p_breach_terminal']:.3f}  "
        f"P(any)={m['p_breach_anytime']:.3f}",
        flush=True,
    )


def experiment_multi_asset(dates, rets, horizons=(30, 63),
                           n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                           n_test=N_TEST, seed=SEED):
    """Learned 3-asset vs learned single-asset vs classic CPPI, fixed floor.

    The classic baseline holds the cushion in an equal-weight
    SPY/TLT/GLD mix. The single-asset learner only sees SPY.
    """
    from dhl.experiments.cppi_learned import (
        CPPIPolicy,
        _learned_weight_fn,
        block_bootstrap_paths as _bb1,
        policy_init_hash,
        seeded_policy,
        train_cppi_policy,
    )

    out = {}
    for horizon in horizons:
        print(f"\n=== multi-asset, horizon {horizon}d, fixed floor ===",
              flush=True)

        def sampler(n, s, h=horizon):
            return block_bootstrap_paths_multi(
                rets, TICKERS, n, h, BLOCK_LEN, S0=V0, seed=s)

        P_test = sampler(n_test, seed + 999)

        # Learned multi-asset.
        policy_m = seeded_policy(MultiAssetPolicy, len(TICKERS), seed=seed)
        train_multi_policy(policy_m, sampler, horizon, n_epochs=n_epochs,
                           batch_size=batch_size, seed=seed, log_every=50)
        m_multi = evaluate_multi(_learned_multi_weight_fn(policy_m), P_test)

        # Learned single-asset (SPY only), no-leverage cap for fairness.
        policy_s = seeded_policy(CPPIPolicy, seed=seed)
        train_cppi_policy(
            policy_s,
            lambda n, s: _bb1(rets["SPY"], n, horizon, BLOCK_LEN,
                              S0=V0, seed=s),
            horizon, n_epochs=n_epochs, batch_size=batch_size,
            seed=seed, log_every=0)
        m_single = evaluate_multi(
            _learned_single_weight_fn(policy_s), P_test)

        # Classic CPPI on the equal-weight mix.
        m_classic = evaluate_multi(_classic_multi_weight_fn(), P_test)

        _print_row("learned multi (3 assets)", m_multi)
        _print_row("learned single (SPY)", m_single)
        _print_row("classic m=4 (mix)", m_classic)
        out[f"{horizon}d"] = {
            "learned_multi": m_multi,
            "learned_single": m_single,
            "classic_m4_mix": m_classic,
        }
    return out


def experiment_drawdown(dates, rets, horizon=30, n_epochs=N_EPOCHS,
                        batch_size=BATCH_SIZE, n_test=N_TEST, seed=SEED):
    """Fixed floor vs 10% drawdown floor, learned multi vs classic."""
    out = {}

    def sampler(n, s):
        return block_bootstrap_paths_multi(
            rets, TICKERS, n, horizon, BLOCK_LEN, S0=V0, seed=s)

    P_test = sampler(n_test, seed + 999)
    for floor_mode in ("fixed", "drawdown"):
        print(f"\n=== drawdown experiment, {horizon}d, floor={floor_mode} ===",
              flush=True)
        policy = seeded_policy(MultiAssetPolicy, len(TICKERS), seed=seed)
        train_multi_policy(policy, sampler, horizon, floor_mode=floor_mode,
                           n_epochs=n_epochs, batch_size=batch_size,
                           seed=seed, log_every=50)
        m_learned = evaluate_multi(
            _learned_multi_weight_fn(policy), P_test,
            floor_mode=floor_mode)
        m_classic = evaluate_multi(
            _classic_multi_weight_fn(), P_test, floor_mode=floor_mode)
        _print_row("learned multi", m_learned)
        _print_row("classic m=4 (mix)", m_classic)
        out[floor_mode] = {"learned_multi": m_learned,
                           "classic_m4_mix": m_classic}
    return out


def experiment_cross_decade(dates, rets, horizon=30, n_epochs=N_EPOCHS,
                            batch_size=BATCH_SIZE, n_test=N_TEST, seed=SEED):
    """Train before 2020, evaluate on 2020+ paths (includes COVID crash).

    The honest out-of-distribution test: the learned policy meets a
    regime it never trained on. A same-data policy (trained on all
    years) is included to show the gap.
    """
    from dhl.experiments.cppi_learned import (
        CPPIPolicy,
        _learned_weight_fn,
        block_bootstrap_paths as _bb1,
        policy_init_hash,
        seeded_policy,
        train_cppi_policy,
    )

    d = np.array(dates[1:])  # dates[i] is the end date of rets[i]
    pre = rets["SPY"][d < "2020-01-01"]
    post = rets["SPY"][d >= "2020-01-01"]
    print(f"\ncross-decade: {len(pre)} pre-2020 returns, "
          f"{len(post)} 2020+ returns", flush=True)

    def pre_sampler(n, s):
        return _bb1(pre, n, horizon, BLOCK_LEN, S0=V0, seed=s)

    def post_sampler(n, s):
        return _bb1(post, n, horizon, BLOCK_LEN, S0=V0, seed=s)

    S_test = post_sampler(n_test, seed + 999)

    # Learned policy trained only on pre-2020 data.
    policy_pre = seeded_policy(CPPIPolicy, seed=seed)
    train_cppi_policy(policy_pre, pre_sampler, horizon, n_epochs=n_epochs,
                      batch_size=batch_size, seed=seed, log_every=0)

    # Same architecture trained on all data (upper reference).
    policy_all = seeded_policy(CPPIPolicy, seed=seed)
    train_cppi_policy(
        policy_all,
        lambda n, s: _bb1(rets["SPY"], n, horizon, BLOCK_LEN, S0=V0, seed=s),
        horizon, n_epochs=n_epochs, batch_size=batch_size,
        seed=seed, log_every=0)

    from dhl.experiments.cppi_learned import evaluate as evaluate_1d
    print(f"\n=== cross-decade, horizon {horizon}d, test on 2020+ ===",
          flush=True)
    out = {}
    for name, pol in (("learned (train pre-2020)", policy_pre),
                      ("learned (train all years)", policy_all)):
        m = evaluate_1d(_learned_weight_fn(pol), S_test)
        _print_row(name, m)
        out[name] = m
    m_classic = evaluate_1d(
        lambda i, V, t, c, v: classic_cppi_weight(V, multiplier=4.0, cap=1.0),
        S_test)
    _print_row("classic m=4", m_classic)
    out["classic m=4"] = m_classic
    return out


def run_all(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, n_test=N_TEST,
            seed=SEED):
    dates, rets = load_aligned_returns(TICKERS)
    print(f"loaded {len(dates)} common trading days for {TICKERS}", flush=True)
    results = {
        "multi_asset": experiment_multi_asset(
            dates, rets, n_epochs=n_epochs, batch_size=batch_size,
            n_test=n_test, seed=seed),
        "drawdown_floor": experiment_drawdown(
            dates, rets, n_epochs=n_epochs, batch_size=batch_size,
            n_test=n_test, seed=seed),
        "cross_decade": experiment_cross_decade(
            dates, rets, n_epochs=n_epochs, batch_size=batch_size,
            n_test=n_test, seed=seed),
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUTDIR}/metrics.json", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-asset / drawdown / cross-decade CPPI experiments")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_all(n_epochs=args.epochs, batch_size=args.batch_size,
            n_test=args.n_test, seed=args.seed)


if __name__ == "__main__":
    main()
