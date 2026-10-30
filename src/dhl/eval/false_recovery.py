"""E7: false-recovery treatment.

A bear-market rally that fails into a secondary low is the
attack this item defends against. The treatment is a re-entry
gate on the budget fraction, at the SAME variable budget as R1:
the gate scales the fraction u that the base policy proposes
and never touches the CPPI budget B, the multiplier, the floor
or the cap, so gated and ungated arms differ only in timing.

The gate has three stages, applied in order:

1. Simple hysteresis and cooldown. A de-risking trigger fires
   when the causal 21d vol exceeds ``exit_vol`` (0.35, the
   shipped emergency-brake threshold) or the trailing 60d price
   drawdown exceeds ``exit_drawdown`` (0.10). On the trigger
   step itself the fraction is already cut to
   ``derisk_factor`` times the base fraction (the shipped brake
   acts on the same step, so the gate does too). Re-entry is
   then blocked for ``cooldown_steps`` full steps, and after
   the cooldown it requires vol below ``reentry_vol`` (0.20),
   strictly below the exit threshold, and the price drawdown
   below ``reentry_drawdown`` (0.05), strictly below the exit
   drawdown: those gaps are the hysteresis, and they stop
   chatter between the thresholds and stop re-entry while the
   market is still deeply below its peak (a constant-rate fall
   has near-zero measured vol, so vol alone cannot be trusted
   to block re-entry mid-fall).
2. Trend persistence. Even after the cooldown and the vol
   level, re-entry requires ``persistence_steps`` consecutive
   rises in portfolio wealth: one up day is not a recovery.
3. Uncertainty. Re-entry also requires the vol ratio (5d vol
   over 60d vol) at or below 1.0: volatility must be falling
   relative to its own history, not merely below a level.

The gate is stateful per path (mode, cooldown counter, rise
streak, previous wealth) and uses only causal inputs: the
rollout features at the current step and the wealth marks
already realized. It never looks ahead.

Attack examination and the fresh judge. The four named attack
patterns (the 1932 bear rally, 1990, 2011, 2018) are examined
as deterministic fixture shapes, not as loaded prices: the
in-repo SPY series starts in 2007, so 1932 and 1990 prices are
not available in this repo and are not fabricated. The
in-repo attack episodes used for tuning are the 2011 debt
ceiling crash and the 2018 fourth-quarter crash, both present
in the committed series. Gate parameters are selected on a
predeclared grid over the synthetic fixtures plus those two
tuning episodes, then frozen. The judge episodes (COVID
2020 recovery, 2022 bear-market recovery) are disjoint from
the tuning episodes and from the base model's training eras,
and the gate parameters are not re-tuned on them. The
2024+ episodes are declared as a reserve and are NOT
evaluated in this unit: they are the final held-out set for
later work. The base learned policy trains on 2007-2010
windows only, disjoint from tuning and judge episodes alike.

Attribution. For each realized judge episode the report
records the secondary low (the minimum price after the first
rebound peak, defined in ``secondary_low_index``: the first
interior local maximum at least 5% above the episode minimum;
absent when no such rebound exists), the first gate trigger
step, and the extra dollar exposure bought before the
secondary low: the sum over steps from the trigger to the
secondary low of (early holding minus gated holding), plus
its per-step mean, the fee difference and the terminal
wealth difference. Extra exposure is reported for the early
arm relative to the gated arm, so a positive number means
the early arm bought more before the secondary low. When
the gate never triggers or no secondary low exists, the
attribution fields are absent, never zero-filled.
"""

import numpy as np
import torch

from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    FLOOR,
    FLOOR_PENALTY,
    rollout,
)
from dhl.policies.budget_allocator import (
    BudgetFractionNet,
    budget_weight_fn,
)
from dhl.policies.candidate_comparison import (
    E5_FEATURE_NAMES,
    MULTIPLIER,
)
from dhl.policies.candidate_training import fraction_fn_for_model

#: Attack episodes: tuning (gate design), judge (frozen
#: parameters only), reserve (declared, never evaluated here).
ATTACK_EPISODES = (
    {"name": "tuning_2011", "start": "2011-08-01",
     "end": "2012-01-01", "role": "tuning"},
    {"name": "tuning_2018", "start": "2018-10-01",
     "end": "2019-04-01", "role": "tuning"},
    {"name": "judge_2020", "start": "2020-02-19",
     "end": "2020-09-01", "role": "judge"},
    {"name": "judge_2022", "start": "2022-06-01",
     "end": "2023-07-01", "role": "judge"},
    {"name": "reserve_2024plus", "start": "2024-01-01",
     "end": None, "role": "reserve"},
)

#: Predeclared gate-parameter grid (tuning only).
E7_GRID = tuple(
    {"cooldown_steps": cd, "reentry_vol": rv,
     "persistence_steps": ps, "derisk_factor": df}
    for cd in (0, 5, 10)
    for rv in (0.20, 0.25)
    for ps in (1, 3)
    for df in (0.0, 0.25)
)

#: E7 experiment choices, predeclared here.
E7_TRAINING = {
    "base_train": ("2007-01-01", "2011-01-01"),
    "base_path_len": 63,
    "n_epochs": 30,
    "lr": 3e-3,
    "seed": 7,
    "n_bootstrap": 20,
    "bootstrap_block_len": 21,
    "bootstrap_seed_base": 8000,
    "floor_penalty": float(FLOOR_PENALTY),
    "anytime_penalty": float(FLOOR_PENALTY),
    "default_gate": {
        "exit_vol": 0.35,
        "exit_drawdown": 0.10,
        "reentry_vol": 0.20,
        "reentry_drawdown": 0.05,
        "cooldown_steps": 5,
        "persistence_steps": 3,
        "derisk_factor": 0.0,
    },
}


# --- the gate ---------------------------------------------------------------------


class ReentryGate:
    """Stateful re-entry gate over a base fraction function.

    ``mode`` is True while invested and False while derisked,
    per path. ``history_modes`` records the invested flag at
    every step the gate is called, so attribution can find the
    first trigger step. State is reset on the first call
    (step index 0) for the calling batch size.
    """

    def __init__(self, exit_vol=0.35, exit_drawdown=0.10,
                 reentry_vol=0.20, reentry_drawdown=0.05,
                 cooldown_steps=5,
                 persistence_steps=3, derisk_factor=0.0):
        if not 0.0 < reentry_vol < exit_vol:
            raise ValueError(
                "hysteresis needs 0 < reentry_vol < exit_vol, got "
                f"{reentry_vol} vs {exit_vol}")
        if not 0.0 <= reentry_drawdown < exit_drawdown:
            raise ValueError(
                "hysteresis needs 0 <= reentry_drawdown < "
                f"exit_drawdown, got {reentry_drawdown} vs "
                f"{exit_drawdown}")
        if cooldown_steps < 0 or persistence_steps < 1:
            raise ValueError("cooldown_steps must be >= 0 and "
                             "persistence_steps >= 1")
        if not 0.0 <= derisk_factor <= 1.0:
            raise ValueError("derisk_factor must lie in [0, 1]")
        self.exit_vol = float(exit_vol)
        self.exit_drawdown = float(exit_drawdown)
        self.reentry_vol = float(reentry_vol)
        self.reentry_drawdown = float(reentry_drawdown)
        self.cooldown_steps = int(cooldown_steps)
        self.persistence_steps = int(persistence_steps)
        self.derisk_factor = float(derisk_factor)
        self._state = None
        self.history_modes = []

    def params(self):
        return {
            "exit_vol": self.exit_vol,
            "exit_drawdown": self.exit_drawdown,
            "reentry_vol": self.reentry_vol,
            "reentry_drawdown": self.reentry_drawdown,
            "cooldown_steps": self.cooldown_steps,
            "persistence_steps": self.persistence_steps,
            "derisk_factor": self.derisk_factor,
        }

    def _reset(self, n):
        self._state = {
            "invested": np.ones(n, dtype=bool),
            "cooldown": np.zeros(n, dtype=int),
            "streak": np.zeros(n, dtype=int),
            "prev_wealth": None,
        }
        self.history_modes = []

    def __call__(self, i, V, tau_frac, cushion, feats, u_base):
        Vd = V.detach().cpu().numpy().astype(float)
        n = len(Vd)
        if i == 0 or self._state is None \
                or len(self._state["invested"]) != n:
            self._reset(n)
        st = self._state
        vol = feats["vol"].detach().cpu().numpy().astype(float)
        vol_ratio = feats["vol_ratio"].detach().cpu().numpy() \
            .astype(float)
        drawdown = feats["drawdown"].detach().cpu().numpy() \
            .astype(float)
        if st["prev_wealth"] is not None:
            rose = Vd > st["prev_wealth"]
            st["streak"] = np.where(rose, st["streak"] + 1, 0)
        st["prev_wealth"] = Vd.copy()
        trigger = (vol > self.exit_vol) \
            | (drawdown > self.exit_drawdown)
        newly_derisked = st["invested"] & trigger
        st["invested"][newly_derisked] = False
        st["cooldown"][newly_derisked] = self.cooldown_steps
        derisked = ~st["invested"]
        if np.any(derisked):
            idx = np.where(derisked)[0]
            # The trigger step itself is not a cooldown step:
            # paths derisked this step skip the decrement, so
            # cooldown_steps counts full blocked steps after it.
            already = idx[~newly_derisked[idx]]
            still_cooling = st["cooldown"][already] > 0
            st["cooldown"][already[still_cooling]] -= 1
            cand = already[~still_cooling]
            can_reenter = (vol[cand] < self.reentry_vol) \
                & (drawdown[cand] < self.reentry_drawdown) \
                & (vol_ratio[cand] <= 1.0) \
                & (st["streak"][cand] >= self.persistence_steps)
            st["invested"][cand[can_reenter]] = True
            st["cooldown"][cand[can_reenter]] = 0
        self.history_modes.append(st["invested"].copy())
        factor = np.where(st["invested"], 1.0, self.derisk_factor)
        return u_base * torch.as_tensor(
            factor, dtype=u_base.dtype, device=u_base.device)


def gate_fraction_fn(gate, base_fraction_fn):
    """Wrap a base fraction function with a re-entry gate."""

    def fn(i, V, tau_frac, cushion, feats):
        u_base = base_fraction_fn(i, V, tau_frac, cushion, feats)
        return gate(i, V, tau_frac, cushion, feats, u_base)

    return fn


# --- deterministic fixtures ----------------------------------------------------------


def _ramp(start, end, n_steps):
    if n_steps <= 0:
        return np.array([])
    return np.linspace(start, end, n_steps + 1)[1:]


def double_dip_prices():
    """Bear rally then secondary low, then recovery.

    Falls 100 to 70, rallies strongly to 95 (a 36% bear rally
    that rebuilds the early arm's cushion), gaps down to 60 in
    three steps (the secondary low CPPI cannot de-risk through
    fast enough), then recovers to 100. The early arm buys the
    rally and rides the gap fall; the gated arm must sit the
    rally out.
    """
    return np.concatenate([
        [100.0],
        _ramp(100.0, 70.0, 15),
        _ramp(70.0, 95.0, 15),
        np.array([78.0, 60.0]),
        _ramp(60.0, 100.0, 25),
    ])


def v_bottom_prices():
    """Single V-bottom control: falls 100 to 70, recovers to 105."""
    return np.concatenate([
        [100.0],
        _ramp(100.0, 70.0, 15),
        _ramp(70.0, 105.0, 30),
    ])


def chop_prices():
    """Choppy recovery: falls to 70, then three small false
    rallies of 5-8% each separated by dips, then recovers."""
    return np.concatenate([
        [100.0],
        _ramp(100.0, 70.0, 15),
        _ramp(70.0, 76.0, 5),
        _ramp(76.0, 71.0, 5),
        _ramp(71.0, 78.0, 5),
        _ramp(78.0, 72.0, 5),
        _ramp(72.0, 80.0, 5),
        _ramp(80.0, 100.0, 20),
    ])


# --- secondary low and attribution -----------------------------------------------------


def secondary_low_index(prices):
    """Index of the minimum price after the first rebound peak.

    The episode minimum is the first trough; the rebound peak
    is the first interior local maximum at least 5% above that
    minimum. The secondary low is the minimum after that peak.
    Returns None when no qualifying rebound peak exists.
    """
    p = np.asarray(prices, dtype=float)
    if len(p) < 4:
        return None
    trough = None
    for i in range(1, len(p) - 1):
        if p[i] <= p[i - 1] and p[i] <= p[i + 1]:
            trough = i
            break
    if trough is None:
        return None
    peak = None
    for i in range(trough + 1, len(p) - 1):
        if p[i] >= 1.05 * p[trough] and p[i] >= p[i - 1] \
                and p[i] >= p[i + 1]:
            peak = i
            break
    if peak is None:
        return None
    return int(peak + 1 + np.argmin(p[peak + 1:]))


def attribute_reentry(prices, early_ledger, gated_ledger, gate):
    """Cost/return attribution for one realized episode path.

    Compares the early (ungated) arm with the gated arm on the
    same path. Returns None fields when the gate never triggers
    or no secondary low exists. Ledger holdings are per-step
    post-trade dollar holdings for the single path.
    """
    p = np.asarray(prices, dtype=float)
    secondary = secondary_low_index(p)
    modes = gate.history_modes
    trigger = None
    if modes:
        arr = np.array(modes, dtype=bool)
        if arr.shape[1] >= 1:
            derisked = ~arr[:, 0]
            hit = np.where(derisked)[0]
            trigger = int(hit[0]) if len(hit) else None
    out = {
        "secondary_low_index": secondary,
        "secondary_low_price": (None if secondary is None
                                else float(p[secondary])),
        "trigger_step": trigger,
        "extra_exposure_sum": None,
        "extra_exposure_mean": None,
        "fee_difference": None,
        "terminal_wealth_difference": None,
    }
    if secondary is None or trigger is None:
        return out
    early_hold = early_ledger["holding_post"][0].detach() \
        .cpu().numpy().astype(float)
    gated_hold = gated_ledger["holding_post"][0].detach() \
        .cpu().numpy().astype(float)
    lo = max(0, trigger)
    hi = min(secondary, len(early_hold) - 1, len(gated_hold) - 1)
    if hi < lo:
        return out
    diff = early_hold[lo:hi + 1] - gated_hold[lo:hi + 1]
    out["extra_exposure_sum"] = float(diff.sum())
    out["extra_exposure_mean"] = float(diff.mean())
    out["fee_difference"] = float(
        early_ledger["cum_fees"][0] - gated_ledger["cum_fees"][0])
    out["terminal_wealth_difference"] = float(
        early_ledger["terminal_wealth"][0]
        - gated_ledger["terminal_wealth"][0])
    return out


# --- selection ---------------------------------------------------------------------------


def select_gate_params(table):
    """Select the grid row with the highest mean utility.

    ``table`` rows carry ``params`` and ``mean_utility``; ties
    break toward the earlier row (callers pass grid order).
    Raises on an empty table instead of inventing a winner.
    """
    if not table:
        raise ValueError("selection needs a non-empty table")
    best = None
    for row in table:
        if best is None or row["mean_utility"] > best["mean_utility"]:
            best = row
    return best


# --- helpers for the run --------------------------------------------------------------------


def _date_slice(dates, start, end):
    start_idx = None
    end_idx = len(dates)
    for i, d in enumerate(dates):
        if start_idx is None and d >= start:
            start_idx = i
        if end is not None and d >= end:
            end_idx = i
            break
    if start_idx is None:
        raise ValueError(f"no dates on or after {start}")
    return start_idx, end_idx


def _slice_prices(dates, prices, start, end):
    lo, hi = _date_slice(dates, start, end)
    return np.asarray(prices, dtype=float)[lo:hi], dates[lo:hi]


def _rollout_arm(S, fraction_fn):
    V_T, path, ledger = rollout(
        S, budget_weight_fn(fraction_fn, multiplier=MULTIPLIER,
                            floor=FLOOR, cap=CAP),
        floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
        return_path=True, return_ledger=True)
    return V_T, path, ledger


def _arm_metrics(V_T, path, ledger):
    V_T = V_T.detach()
    path = path.detach()
    shortfall = torch.clamp(FLOOR - V_T, min=0.0)
    dd = []
    for row in path.detach().cpu().numpy():
        peak = row[0]
        worst = 0.0
        for v in row:
            peak = max(peak, v)
            worst = max(worst, peak - v)
        dd.append(worst)
    return {
        "mean_terminal_wealth": float(V_T.mean()),
        "mean_terminal_shortfall": float(shortfall.mean()),
        "breach_count_terminal": int((V_T < FLOOR).sum()),
        "mean_max_drawdown": float(np.mean(dd)),
        "mean_fees": float(ledger["cum_fees"].mean()),
        "n_paths": int(len(V_T)),
    }


def _train_base_model(dates, prices, base_train, path_len,
                      n_epochs, lr, seed):
    from dhl.eval.chrono import nonoverlapping_windows
    lo, hi = _date_slice(dates, base_train[0], base_train[1])
    slice_prices = np.asarray(prices, dtype=float)[lo:hi]
    paths, _starts = nonoverlapping_windows(slice_prices, path_len,
                                            s0=100.0)
    S = torch.as_tensor(paths, dtype=torch.float64)
    torch.manual_seed(seed)
    model = BudgetFractionNet(hidden_dims=(16, 16), form="direct",
                              feature_names=E5_FEATURE_NAMES)
    fn = fraction_fn_for_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    from dhl.experiments.losses import multihorizon_loss
    for _epoch in range(n_epochs):
        optimizer.zero_grad()
        V_T, path, ledger = rollout(
            S, budget_weight_fn(fn, multiplier=MULTIPLIER,
                                floor=FLOOR, cap=CAP),
            floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
            return_path=True, return_ledger=True)
        total, _comp = multihorizon_loss(
            V_T, path, weight_path=ledger["weight"],
            floor=FLOOR, floor_penalty=FLOOR_PENALTY,
            anytime_penalty=FLOOR_PENALTY, turnover_penalty=0.0)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _episode_bootstrap_paths(episode_prices, path_len, block_len,
                             n_paths, seed_base, episode_index):
    from dhl.markets.real import block_bootstrap_paths, log_returns
    rets = log_returns(np.asarray(episode_prices, dtype=float))
    paths = block_bootstrap_paths(
        rets, n_paths, path_len, block_len=block_len, S0=100.0,
        seed=seed_base + episode_index * 100)
    return torch.as_tensor(paths, dtype=torch.float64)


def run_e7(dates, prices, base_train=None, tuning_episodes=None,
           judge_episodes=None, reserve_episodes=None, grid=None,
           n_epochs=None, n_bootstrap=None, path_len=None):
    """Execute the full E7 experiment and return its report."""
    base_train = E7_TRAINING["base_train"] if base_train is None \
        else tuple(base_train)
    tuning_episodes = tuple(e for e in ATTACK_EPISODES
                            if e["role"] == "tuning") \
        if tuning_episodes is None else tuple(tuning_episodes)
    judge_episodes = tuple(e for e in ATTACK_EPISODES
                           if e["role"] == "judge") \
        if judge_episodes is None else tuple(judge_episodes)
    reserve_episodes = tuple(e for e in ATTACK_EPISODES
                             if e["role"] == "reserve") \
        if reserve_episodes is None else tuple(reserve_episodes)
    grid = E7_GRID if grid is None else tuple(grid)
    n_epochs = E7_TRAINING["n_epochs"] if n_epochs is None \
        else int(n_epochs)
    n_bootstrap = E7_TRAINING["n_bootstrap"] if n_bootstrap is None \
        else int(n_bootstrap)
    path_len = E7_TRAINING["base_path_len"] if path_len is None \
        else int(path_len)
    dates = list(dates)
    prices = np.asarray(prices, dtype=float)
    base_model = _train_base_model(
        dates, prices, base_train, path_len, n_epochs,
        E7_TRAINING["lr"], E7_TRAINING["seed"])
    base_fn = fraction_fn_for_model(base_model)
    classic_fn = lambda i, V, t, c, f: torch.ones_like(V)
    tuning_paths = []
    for fixture_fn in (double_dip_prices, v_bottom_prices,
                       chop_prices):
        tuning_paths.append(torch.as_tensor(
            fixture_fn(), dtype=torch.float64).unsqueeze(0))
    for ep in tuning_episodes:
        ep_prices, _d = _slice_prices(dates, prices, ep["start"],
                                      ep["end"])
        tuning_paths.append(torch.as_tensor(
            ep_prices / ep_prices[0] * 100.0,
            dtype=torch.float64).unsqueeze(0))
    table = []
    for combo in grid:
        gate = ReentryGate(
            exit_vol=E7_TRAINING["default_gate"]["exit_vol"],
            exit_drawdown=E7_TRAINING["default_gate"]["exit_drawdown"],
            reentry_vol=combo["reentry_vol"],
            cooldown_steps=combo["cooldown_steps"],
            persistence_steps=combo["persistence_steps"],
            derisk_factor=combo["derisk_factor"])
        gated_fn = gate_fraction_fn(gate, base_fn)
        utils = []
        for S in tuning_paths:
            V_T, path, ledger = _rollout_arm(S, gated_fn)
            shortfall = torch.clamp(FLOOR - V_T, min=0.0)
            utils.append(float(V_T.detach().mean())
                         - FLOOR_PENALTY * float(shortfall.detach().mean()))
        table.append({"params": dict(combo),
                      "mean_utility": float(np.mean(utils))})
    selected = select_gate_params(table)
    gate_params = dict(E7_TRAINING["default_gate"])
    gate_params.update(selected["params"])
    judge_cells = []
    for j, ep in enumerate(judge_episodes):
        ep_prices, _d = _slice_prices(dates, prices, ep["start"],
                                      ep["end"])
        realized = torch.as_tensor(
            ep_prices / ep_prices[0] * 100.0,
            dtype=torch.float64).unsqueeze(0)
        boot = _episode_bootstrap_paths(
            ep_prices, len(ep_prices) - 1,
            E7_TRAINING["bootstrap_block_len"], n_bootstrap,
            E7_TRAINING["bootstrap_seed_base"], j)
        arms = {}
        ledgers = {}
        for name, fn in (
                ("learned", base_fn),
                ("learned_gated", gate_fraction_fn(
                    ReentryGate(**gate_params), base_fn)),
                ("classic", classic_fn),
                ("classic_gated", gate_fraction_fn(
                    ReentryGate(**gate_params), classic_fn))):
            V_T, path, ledger = _rollout_arm(boot, fn)
            arms[name] = _arm_metrics(V_T, path, ledger)
        gate_real = ReentryGate(**gate_params)
        V_e, _pe, led_e = _rollout_arm(
            realized, base_fn)
        V_g, _pg, led_g = _rollout_arm(
            realized, gate_fraction_fn(gate_real, base_fn))
        attribution = attribute_reentry(
            ep_prices / ep_prices[0] * 100.0, led_e, led_g,
            gate_real)
        attribution["realized_learned_terminal"] = float(V_e[0])
        attribution["realized_gated_terminal"] = float(V_g[0])
        judge_cells.append({
            "episode": ep["name"],
            "start": ep["start"], "end": ep["end"],
            "n_bootstrap_paths": n_bootstrap,
            "realized_length": int(len(ep_prices)),
            "arms": arms,
            "attribution": attribution,
        })
    return {
        "training": dict(E7_TRAINING),
        "tuning_table": table,
        "selected_params": selected["params"],
        "selected_gate_params": gate_params,
        "judge_cells": judge_cells,
        "reserve_episodes": [e["name"] for e in reserve_episodes],
    }
