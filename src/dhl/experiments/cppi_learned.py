"""Portfolio insurance: classic CPPI vs a learned dynamic allocator.

CPPI (constant proportion portfolio insurance) protects a floor under
wealth by holding ``multiplier * cushion`` in the risky asset, where
``cushion = wealth - floor``. Here a small neural policy learns the
allocation directly: it maps (time left, cushion ratio, recent
volatility) to a risky weight, trained to maximize expected terminal
wealth with a penalty for breaching the floor. No options anywhere.
Just a portfolio, a floor, and a network that learns not to fall
through it.

Training uses block-bootstrapped real SPY paths (21-day blocks, fresh
every epoch). Evaluation is on held-out bootstrap seeds.

Run: ``python -m dhl.experiments.cppi_learned [--epochs 200]``

Writes ``runs/cppi/metrics.json`` and prints the comparison table.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from dhl.markets.real import block_bootstrap_paths, load_spy_daily, log_returns

# --- experiment settings -------------------------------------------------
V0 = 100.0
FLOOR = 90.0  # 90% of initial capital, fixed (not grown at r)
R = 0.02  # cash earns 2% annually
DT = 1.0 / 252.0  # daily rebalancing
COST_RATE = 0.002  # 20 bps proportional cost on each rebalance
CAP = 1.5  # max risky weight for every strategy (apples to apples)
BLOCK_LEN = 21
HORIZONS = (30, 63)  # trading days
MULTIPLIERS = (3, 4, 5)
N_EPOCHS = 200
BATCH_SIZE = 2048
N_TEST = 20000
FLOOR_PENALTY = 30.0  # weight on expected shortfall below the floor
SEED = 7

OUTDIR = os.path.join("runs", "cppi")


def classic_cppi_weight(V, floor=FLOOR, multiplier=4.0, cap=CAP):
    """Classic CPPI risky weight: multiplier * cushion / wealth, capped.

    Works on floats or torch tensors; cushion is floored at zero so the
    strategy fully de-risks once the floor is breached.
    """
    if torch.is_tensor(V):
        cushion = torch.clamp(V - floor, min=0.0)
        w = multiplier * cushion / V.clamp_min(1e-8)
        return torch.clamp(w, max=cap)
    cushion = max(float(V) - floor, 0.0)
    return min(multiplier * cushion / max(float(V), 1e-8), cap)


class CPPIPolicy(nn.Module):
    """Small MLP: (time_left_frac, cushion_ratio, *FEATURE_NAMES) -> [0, cap].

    cushion_ratio = (V - floor) / V, clipped to [-1, 1]; it goes
    negative if the floor is breached, which the network is allowed
    to see. The remaining inputs are the crash-regime features from
    ``_crash_features`` (all strictly causal).

    ``feature_names`` selects which features the policy consumes, in
    order; it must be a subset of FEATURE_NAMES. A 3-feature policy
    ("vol" only plus time/cushion) reproduces the original behavior.
    """

    def __init__(self, hidden_dims=(16, 16), cap=CAP,
                 feature_names=("vol",)):
        super().__init__()
        self.cap = cap
        self.feature_names = tuple(feature_names)
        for name in self.feature_names:
            if name not in FEATURE_NAMES:
                raise ValueError(f"unknown feature {name!r}")
        layers = []
        dim = 2 + len(self.feature_names)
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.Tanh()]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tau_frac, cushion_ratio, feats):
        cols = [
            torch.as_tensor(tau_frac, dtype=torch.float32).expand_as(cushion_ratio),
            cushion_ratio,
        ]
        for name in self.feature_names:
            cols.append(feats[name])
        x = torch.stack(cols, dim=1)
        return (self.cap * torch.sigmoid(self.net(x))).squeeze(1)


def _vol_features(S, window=21):
    """Annualized rolling std of log returns, shape (n_paths, n_steps).

    Strictly causal: the feature at time ``i`` uses only returns
    ``lr[:, :i]``, i.e. price moves that already happened. The first
    step has no history yet, so its feature is 0.0.
    """
    n, steps_p1 = S.shape
    steps = steps_p1 - 1
    lr = torch.log(S[:, 1:] / S[:, :-1])
    vols = torch.empty_like(lr)
    for i in range(steps):
        j = max(0, i - window)
        w = lr[:, j:i]
        if w.shape[1] == 0:
            vols[:, i] = 0.0
        else:
            vols[:, i] = w.std(dim=1, unbiased=False) * math.sqrt(252.0)
    return vols


# Feature names in the order the policy consumes them. "vol" is the
# original 21d annualized vol; the rest are crash detectors, all
# strictly causal (no look-ahead -- see _crash_features).
FEATURE_NAMES = ("vol", "vol_ratio", "drawdown", "panic_z")


def _crash_features(S):
    """Crash-regime indicators from price paths, shape (n_paths, n_steps).

    Returns a dict of tensors keyed by feature name. Every feature at
    time ``i`` uses only prices ``S[:, :i+1]`` (history up to and
    including the current bar). Strictly causal -- a regression test
    pins this.

    - ``vol``: 21d annualized realized vol (the original feature).
    - ``vol_ratio``: 5d vol / 60d vol. Spikes above 1 when volatility
      is accelerating -- the classic crash signature.
    - ``drawdown``: trailing 60d peak-to-trough drawdown of the price
      level, in [0, 1). Already falling before vol fully reprices.
    - ``panic_z``: 5d cumulative log return divided by 60d vol scaled
      to 5d. Negative tail z-score of the recent move; large negative
      values mean "this drop is extreme even for current vol".
    """
    n, steps_p1 = S.shape
    steps = steps_p1 - 1
    lr = torch.log(S[:, 1:] / S[:, :-1])  # (n, steps)
    logS = torch.log(S)

    vols = _vol_features(S, window=21)
    vol_ratio = torch.zeros_like(lr)
    drawdown = torch.zeros_like(lr)
    panic_z = torch.zeros_like(lr)

    sqrt252 = math.sqrt(252.0)
    for i in range(steps):
        # --- vol_ratio: 5d vol / 60d vol ---
        j5 = max(0, i - 5)
        j60 = max(0, i - 60)
        w5 = lr[:, j5:i]
        w60 = lr[:, j60:i]
        v5 = w5.std(dim=1, unbiased=False) * sqrt252 if w5.shape[1] > 1 else torch.zeros(n, device=S.device, dtype=S.dtype)
        v60 = w60.std(dim=1, unbiased=False) * sqrt252 if w60.shape[1] > 1 else torch.zeros(n, device=S.device, dtype=S.dtype)
        vol_ratio[:, i] = v5 / v60.clamp_min(1e-8)
        # clamp explosive ratios from tiny denominators on flat history
        vol_ratio[:, i] = torch.clamp(vol_ratio[:, i], max=10.0)

        # --- drawdown: trailing 60d peak-to-trough on price level ---
        k0 = max(0, i + 1 - 60)
        window = S[:, k0:i + 1]
        peak = window.max(dim=1).values
        drawdown[:, i] = (peak - S[:, i]) / peak.clamp_min(1e-8)

        # --- panic_z: 5d return z-scored by 60d vol ---
        r5 = lr[:, j5:i].sum(dim=1) if w5.shape[1] > 0 else torch.zeros(n, device=S.device, dtype=S.dtype)
        denom = v60 * math.sqrt(5.0 / 252.0)
        panic_z[:, i] = r5 / denom.clamp_min(1e-8)
        panic_z[:, i] = torch.clamp(panic_z[:, i], min=-10.0, max=10.0)

    return {
        "vol": vols,
        "vol_ratio": vol_ratio,
        "drawdown": drawdown,
        "panic_z": panic_z,
    }


def rollout(S, weight_fn, floor=FLOOR, r=R, dt=DT, cost_rate=COST_RATE,
            return_path=False):
    """Self-financing portfolio wealth under a dynamic allocation.

    S has shape (n_paths, n_steps + 1). weight_fn(i, V, tau_frac,
    cushion_ratio, vol) picks the risky weight in [0, cap] at each
    rebalancing date. Returns terminal wealth per path (n_paths,);
    with return_path=True also the full wealth path. The final path
    point is post-liquidation wealth, so path[:, -1] == V_T exactly.
    """
    n, steps_p1 = S.shape
    steps = steps_p1 - 1
    T = steps * dt
    device, dtype = S.device, S.dtype
    V = torch.full((n,), V0, device=device, dtype=dtype)
    D = torch.zeros(n, device=device, dtype=dtype)  # risky dollars held
    growth = math.exp(r * dt)
    rets = S[:, 1:] / S[:, :-1]
    feats_full = _crash_features(S)
    path = [V.clone()] if return_path else None

    for i in range(steps):
        tau_frac = (T - i * dt) / T
        cushion = torch.clamp((V - floor) / V.clamp_min(1e-8), -1.0, 1.0)
        feats = {k: v[:, i] for k, v in feats_full.items()}
        w = weight_fn(i, V, tau_frac, cushion, feats)
        prev_grown = D if i == 0 else D * rets[:, i - 1]
        target = w * V
        V = V - cost_rate * (target - prev_grown).abs()
        D = w * V
        V = (V - D) * growth + D * rets[:, i]
        if return_path:
            path.append(V.clone())

    V = V - cost_rate * D.abs()  # liquidate the risky leg at T
    if return_path:
        # Terminal accounting convention: the final path point is the
        # post-liquidation wealth, so path[:, -1] == V_T exactly and
        # anytime-breach statistics see the liquidation cost too.
        path[-1] = V.clone()
        return V, torch.stack(path, dim=1)
    return V


def _learned_weight_fn(policy):
    def fn(i, V, tau_frac, cushion, feats):
        t = torch.full_like(V, tau_frac)
        return policy(t, cushion, feats)

    return fn


def _classic_weight_fn(multiplier):
    def fn(i, V, tau_frac, cushion, feats):
        return classic_cppi_weight(V, multiplier=multiplier)

    return fn


def cppi_constrained_weight_fn(learned_fn, multiplier=4.0, cap=CAP,
                               floor=FLOOR, emergency_vol=None,
                               emergency_multiplier=None):
    """CPPI risk layer wrapping a learned allocator.

    Division of labor: the network decides WHERE to invest (its proposed
    risky weight); this layer decides HOW MUCH total risk the CPPI
    budget allows. If the network proposes 70% risky but the CPPI
    budget ``m * max(V - F, 0) / V`` allows only 35%, the allocation
    is scaled down to 35% and the rest stays in cash.

    This is strictly more protective than the raw network: the output
    never exceeds what classic CPPI with the same multiplier would
    hold. It does not make breaches impossible -- gap risk from
    discrete rebalancing remains (see README) -- but it gives
    improved downside protection, especially when the cushion is thin.

    ``emergency_vol``: if set, when the 21d realized vol in ``feats``
    exceeds it, the multiplier tightens to ``emergency_multiplier``
    (default: half the base multiplier) for that step. A volatility
    spike therefore shrinks the risk budget before the network's
    slower features fully repricing the regime.
    """

    def fn(i, V, tau_frac, cushion, feats):
        w_proposed = learned_fn(i, V, tau_frac, cushion, feats)
        m = multiplier
        if emergency_vol is not None:
            trig = feats["vol"] > emergency_vol
            m_eff = torch.where(
                trig,
                torch.full_like(V, emergency_multiplier
                                if emergency_multiplier is not None
                                else multiplier * 0.5),
                torch.full_like(V, multiplier),
            )
        else:
            m_eff = torch.full_like(V, m)
        cushion_amt = torch.clamp(V - floor, min=0.0)
        budget = torch.clamp(
            m_eff * cushion_amt / V.clamp_min(1e-8), max=cap)
        # Scale down only; never scale up beyond what the network wants.
        scale = torch.clamp(budget / w_proposed.clamp_min(1e-8), max=1.0)
        # If the network proposes ~0, keep ~0 (avoid 0/0 artifacts).
        scale = torch.where(w_proposed < 1e-8,
                            torch.zeros_like(scale), scale)
        return w_proposed * scale

    return fn


def _crash_weights(S, crash_mult=3.0, vol_cap=0.30):
    """Per-path breach-penalty multipliers, emphasizing crash regimes.

    ``path_vol`` is the annualized realized vol of each training path.
    Paths calmer than ``vol_cap`` keep weight 1. Above the cap, the
    weight grows linearly: a path at 2x the cap gets weight
    ``1 + crash_mult``. This makes the optimizer care more about not
    breaching exactly when markets are violent -- the failure mode
    that matters -- without changing the objective on calm paths.

    Set ``crash_mult=0`` to recover the original unweighted penalty.
    """
    lr = torch.log(S[:, 1:] / S[:, :-1])
    path_vol = lr.std(dim=1, unbiased=False) * math.sqrt(252.0)
    w = 1.0 + crash_mult * torch.relu(path_vol - vol_cap) / vol_cap
    return w


def train_cppi_policy(policy, sampler, horizon_days, n_epochs=N_EPOCHS,
                       batch_size=BATCH_SIZE, lr=1e-3,
                       floor_penalty=FLOOR_PENALTY, seed=SEED, log_every=20,
                       crash_mult=0.0, vol_cap=0.30):
    """Train on fresh bootstrapped paths every epoch (infinite data).

    Objective: maximize E[V_T] - floor_penalty * E[w * max(floor - V_T, 0)],
    where ``w`` are per-path crash weights (see ``_crash_weights``).
    With ``crash_mult=0`` (default) this is the original objective.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    dt = DT
    history = []
    for epoch in range(n_epochs):
        S_np = sampler(batch_size, seed + epoch)
        S = torch.as_tensor(S_np, dtype=torch.float32)
        V_T = rollout(S, _learned_weight_fn(policy), dt=dt)
        shortfall = torch.relu(FLOOR - V_T)
        if crash_mult > 0:
            w = _crash_weights(S, crash_mult=crash_mult, vol_cap=vol_cap)
            penalty = (w * shortfall).mean()
        else:
            penalty = shortfall.mean()
        objective = -(V_T.mean() - floor_penalty * penalty)
        optimizer.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        history.append(float(objective.detach()))
        if log_every and (epoch + 1) % log_every == 0:
            with torch.no_grad():
                ev = float(V_T.mean())
                pb = float((V_T < FLOOR).float().mean())
            print(
                f"epoch {epoch + 1:4d}/{n_epochs}  objective={history[-1]:.4f}  "
                f"E[V_T]={ev:.2f}  P(breach)={pb:.3f}",
                flush=True,
            )
    return history


def evaluate(weight_fn, S_np, floor=FLOOR):
    """Metrics on held-out paths: mean/std wealth, breach probabilities."""
    S = torch.as_tensor(np.asarray(S_np), dtype=torch.float32)
    with torch.no_grad():
        V_T, path = rollout(S, weight_fn, return_path=True)
    V_T = V_T.numpy()
    wealth_min = path.numpy().min(axis=1)
    return {
        "mean_wealth": float(V_T.mean()),
        "std_wealth": float(V_T.std()),
        "p_breach_terminal": float((V_T < floor).mean()),
        "p_breach_anytime": float((wealth_min < floor).mean()),
        "mean_min_wealth": float(wealth_min.mean()),
    }


def run_experiment(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, n_test=N_TEST,
                   seed=SEED):
    data = load_spy_daily()
    returns = log_returns(data["adjclose"])
    print(f"loaded {len(returns)} SPY daily log returns", flush=True)

    results = {}
    for horizon in HORIZONS:
        print(f"\n=== horizon {horizon} days ===", flush=True)

        def make_sampler(h=horizon):
            return lambda n, s: block_bootstrap_paths(
                returns, n, h, BLOCK_LEN, S0=V0, seed=s
            )

        sampler_h = make_sampler()
        S_test = sampler_h(n_test, seed + 999)

        strategies = {}
        for m in MULTIPLIERS:
            strategies[f"classic_m{m}"] = _classic_weight_fn(m)

        policy = CPPIPolicy()
        train_cppi_policy(policy, sampler_h, horizon, n_epochs=n_epochs,
                          batch_size=batch_size, seed=seed, log_every=50)
        strategies["learned"] = _learned_weight_fn(policy)

        horizon_results = {}
        for name, fn in strategies.items():
            metrics = evaluate(fn, S_test)
            horizon_results[name] = metrics
            print(
                f"{name:12s}  E[V_T]={metrics['mean_wealth']:7.2f}  "
                f"sd={metrics['std_wealth']:6.2f}  "
                f"P(breach)={metrics['p_breach_terminal']:.3f}  "
                f"P(any)={metrics['p_breach_anytime']:.3f}",
                flush=True,
            )
        results[f"{horizon}d"] = horizon_results

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUTDIR}/metrics.json", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="CPPI vs learned allocator")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_experiment(n_epochs=args.epochs, batch_size=args.batch_size,
                   n_test=args.n_test, seed=args.seed)


if __name__ == "__main__":
    main()
