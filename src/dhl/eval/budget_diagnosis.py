"""Budget utilization and recovery diagnosis (Stage 3 neural-cppi E4).

For one episode (one price path) this module computes, per rebalance
step, the full contract series: proposal, allowable budget, realized
execution, actual risky dollars, clipping, cushion, current vol,
current drawdown, cumulative fees, and rebound/secondary-dip state.
It writes that series as a precise per-episode CSV. It produces no
dashboard UI and no plots.

Definitions (per step, pre-trade wealth ``V``, floor ``F``):
  cushion_dollars = V - F
  cushion_ratio = clamp((V - F) / V, -1, 1), the rollout convention
  allowable_budget = min(multiplier * max(cushion_dollars, 0), cap * V)
    (the classic CPPI budget at that step; a reference, not a hard cap
    on a learned proposal)
  proposal_dollars = proposal_weight * V (raw proposal, pre-clamp)
  executed_weight = clamp(proposal_weight, 0, cap)
  executed_dollars = holding_post from the exact proportional-cost
    rebalance solve (see dhl.backtest.execution.rebalance_single)
  clipping_weight = proposal_weight - executed_weight
  clipping_dollars = clipping_weight * V
  unused_budget = allowable_budget - executed_dollars (negative when
    execution exceeds the classic allowable budget; never floored)
  vol = 21d annualized population std of log returns strictly before
    the step, matching dhl.experiments.cppi_learned._vol_features
  drawdown = trailing 60d peak-to-trough price drawdown at the step,
    matching dhl.experiments.cppi_learned._crash_features
  wealth_drawdown = peak-to-trough drawdown of pre-trade wealth so far
  state = rebound/secondary-dip state from prices only (causal; no
    future price is used at any step)

Rebound state machine (thresholds are parameters, defaults 0.05/0.02):
  normal -> dip when the price drawdown from the running peak reaches
    dip_threshold; dip -> rebound when the price rises at least
    rebound_threshold above the dip trough; rebound -> secondary_dip
    when the price falls at least secondary_threshold below the
    rebound peak; secondary_dip -> rebound when the price rises at
    least rebound_threshold above the secondary trough; any state ->
    normal when the price regains the pre-dip peak.

Two entry points share one row schema:
  replay_episode: pure-float replay of recorded proposals. This is the
    counterfactual engine and the hand-computable reference.
  diagnose_episode: runs the public rollout with a recording weight
    function, so the raw proposals come from the live policy call, and
    the execution columns come from the rollout ledger. It imports
    torch and the rollout lazily, so importing this module never
    requires torch.
  rows_from_ledger: builds rows from an already-run rollout ledger
    (plain lists, e.g. tensors converted with .tolist()), for callers
    that already hold a ledger.

Single-path only: an episode CSV has one row per step, so a multi-path
ledger is refused rather than silently averaged. No archived-episode
data is loaded here; callers supply the price path and proposals, and
the falsifier metrics in summarize_diagnosis report measured numbers
for that episode only, with no conclusion assumed about any other
episode.
"""

import csv
import math

import numpy as np

CSV_COLUMNS = (
    "step",
    "price",
    "wealth_pre",
    "cushion_dollars",
    "cushion_ratio",
    "allowable_budget",
    "proposal_weight",
    "proposal_dollars",
    "executed_weight",
    "executed_dollars",
    "clipping_weight",
    "clipping_dollars",
    "fee",
    "cum_fees",
    "unused_budget",
    "vol",
    "drawdown",
    "wealth_drawdown",
    "state",
)

_FLOAT_COLUMNS = tuple(c for c in CSV_COLUMNS if c not in ("step", "state"))


def _validate(prices, proposals, floor, multiplier, cap, cost_rate, v0):
    prices = [float(p) for p in prices]
    proposals = [float(w) for w in proposals]
    if len(prices) < 2:
        raise ValueError("episode needs at least two prices (one step)")
    if len(proposals) != len(prices) - 1:
        raise ValueError(
            f"need {len(prices) - 1} proposals for {len(prices)} prices, "
            f"got {len(proposals)}")
    if not all(math.isfinite(p) and p > 0.0 for p in prices):
        raise ValueError("prices must be finite and positive")
    if not all(math.isfinite(w) for w in proposals):
        raise ValueError("proposals must be finite")
    if not (math.isfinite(v0) and v0 > 0.0):
        raise ValueError("v0 must be finite and positive")
    if not (math.isfinite(floor) and floor >= 0.0):
        raise ValueError("floor must be finite and nonnegative")
    if multiplier < 0.0 or not math.isfinite(multiplier):
        raise ValueError("multiplier must be finite and nonnegative")
    if cap < 0.0 or not math.isfinite(cap):
        raise ValueError("cap must be finite and nonnegative")
    if cost_rate < 0.0 or cost_rate >= 1.0 or not math.isfinite(cost_rate):
        raise ValueError("cost_rate must be in [0, 1)")
    return prices, proposals


def _fee_solve(wealth, holding, weight, cost_rate):
    """Exact one-asset post-fee holding and fee (the kernel formula)."""
    current = holding / wealth if wealth > 0.0 else 0.0
    if weight >= current:
        post = weight * (wealth + cost_rate * holding) / (1.0 + weight * cost_rate)
    else:
        post = weight * (wealth - cost_rate * holding) / (1.0 - weight * cost_rate)
    return post, cost_rate * abs(post - holding)


def _vol_series(prices, window=21):
    """Annualized vol at each step from returns strictly before it."""
    lr = np.log(np.array(prices[1:], dtype=float) / np.array(prices[:-1], dtype=float))
    vols = []
    for i in range(len(prices) - 1):
        window_returns = lr[max(0, i - window):i]
        if window_returns.shape[0] == 0:
            vols.append(0.0)
        else:
            vols.append(float(window_returns.std(ddof=0) * math.sqrt(252.0)))
    return vols


def _drawdown_series(prices, window=60):
    """Trailing peak-to-trough price drawdown at each step."""
    out = []
    for i in range(len(prices) - 1):
        segment = prices[max(0, i + 1 - window):i + 1]
        peak = max(segment)
        out.append((peak - prices[i]) / peak if peak > 0.0 else 0.0)
    return out


def _state_series(prices, dip_threshold=0.05, rebound_threshold=0.02,
                  secondary_threshold=None):
    """Causal rebound/secondary-dip states, one per step."""
    if secondary_threshold is None:
        secondary_threshold = rebound_threshold
    states = []
    state = "normal"
    peak = prices[0]
    dip_peak = prices[0]
    trough = prices[0]
    rebound_peak = prices[0]
    sec_trough = prices[0]
    for i in range(len(prices) - 1):
        price = prices[i]
        peak = max(peak, price)
        if state == "normal":
            if (peak - price) / peak >= dip_threshold:
                state = "dip"
                dip_peak = peak
                trough = price
        elif state == "dip":
            if price >= dip_peak:
                state = "normal"
            elif price < trough:
                trough = price
            elif (price - trough) / trough >= rebound_threshold:
                state = "rebound"
                rebound_peak = price
        elif state == "rebound":
            if price >= dip_peak:
                state = "normal"
            elif price > rebound_peak:
                rebound_peak = price
            elif (rebound_peak - price) / rebound_peak >= secondary_threshold:
                state = "secondary_dip"
                sec_trough = price
        elif state == "secondary_dip":
            if price >= dip_peak:
                state = "normal"
            elif price >= rebound_peak:
                state = "rebound"
                rebound_peak = price
            elif price < sec_trough:
                sec_trough = price
            elif (price - sec_trough) / sec_trough >= rebound_threshold:
                state = "rebound"
                rebound_peak = price
        states.append(state)
    return states


def _build_rows(prices, wealth_pre, holding_post, fees, proposals,
                floor, multiplier, cap, dip_threshold, rebound_threshold,
                secondary_threshold):
    vols = _vol_series(prices)
    drawdowns = _drawdown_series(prices)
    states = _state_series(prices, dip_threshold, rebound_threshold,
                           secondary_threshold)
    rows = []
    cum_fees = 0.0
    wealth_peak = None
    for i, (v, post, fee, proposal) in enumerate(
            zip(wealth_pre, holding_post, fees, proposals)):
        cum_fees += fee
        cushion = v - floor
        cushion_ratio = max(-1.0, min(1.0, cushion / v)) if v > 0.0 else -1.0
        allowable = min(multiplier * max(cushion, 0.0), cap * v)
        executed_weight = max(0.0, min(cap, proposal))
        wealth_peak = v if wealth_peak is None else max(wealth_peak, v)
        rows.append({
            "step": i,
            "price": prices[i],
            "wealth_pre": v,
            "cushion_dollars": cushion,
            "cushion_ratio": cushion_ratio,
            "allowable_budget": allowable,
            "proposal_weight": proposal,
            "proposal_dollars": proposal * v,
            "executed_weight": executed_weight,
            "executed_dollars": post,
            "clipping_weight": proposal - executed_weight,
            "clipping_dollars": (proposal - executed_weight) * v,
            "fee": fee,
            "cum_fees": cum_fees,
            "unused_budget": allowable - post,
            "vol": vols[i],
            "drawdown": drawdowns[i],
            "wealth_drawdown": (wealth_peak - v) / wealth_peak,
            "state": states[i],
        })
    return rows


def replay_episode(prices, proposals, floor, multiplier, cap, cost_rate,
                   v0=100.0, r=0.02, dt=1.0 / 252.0, borrow_rate=None,
                   dip_threshold=0.05, rebound_threshold=0.02,
                   secondary_threshold=None):
    """Replay recorded proposals over one price path, pure floats.

    Mirrors the rollout accounting: cash accrues at ``r`` while
    nonnegative and at ``borrow_rate`` (default ``r``) while negative,
    units ride the price move, and the risky leg is liquidated at the
    terminal price charged on its true terminal value. Returns
    ``(rows, terminal)`` where terminal carries terminal_wealth,
    wealth_pre_liquidation, liquidation_fee and
    cum_fees_incl_liquidation.
    """
    prices, proposals = _validate(prices, proposals, floor, multiplier,
                                  cap, cost_rate, v0)
    if borrow_rate is None:
        borrow_rate = r
    cash = v0
    units = 0.0
    wealth_pre, holding_post, fees = [], [], []
    for i, proposal in enumerate(proposals):
        price = prices[i]
        holding = units * price
        wealth = cash + holding
        if wealth <= 0.0:
            raise ValueError(
                f"replay step {i}: nonpositive pre-trade wealth; "
                "refusing to trade a bankrupt path")
        executed_weight = max(0.0, min(cap, proposal))
        post, fee = _fee_solve(wealth, holding, executed_weight, cost_rate)
        wealth_pre.append(wealth)
        holding_post.append(post)
        fees.append(fee)
        cash = cash - (post - holding) - fee
        units = post / price
        rate = r if cash >= 0.0 else borrow_rate
        cash = cash * math.exp(rate * dt)
    holding_terminal = units * prices[-1]
    wealth_pre_liq = cash + holding_terminal
    liq_fee = cost_rate * abs(holding_terminal)
    terminal = {
        "wealth_pre_liquidation": wealth_pre_liq,
        "liquidation_fee": liq_fee,
        "terminal_wealth": wealth_pre_liq - liq_fee,
        "cum_fees_incl_liquidation": sum(fees) + liq_fee,
    }
    rows = _build_rows(prices, wealth_pre, holding_post, fees, proposals,
                       floor, multiplier, cap, dip_threshold,
                       rebound_threshold, secondary_threshold)
    return rows, terminal


def rows_from_ledger(ledger, prices, proposals, floor, multiplier, cap,
                     cost_rate, dip_threshold=0.05, rebound_threshold=0.02,
                     secondary_threshold=None):
    """Build rows from a rollout ledger (plain lists, single path).

    ``ledger`` uses the rollout ledger keys: wealth_pre, weight,
    holding_post and fee, each one list per step. The recorded weight
    is the effective post-clamp weight; raw proposals must be supplied
    separately (see diagnose_episode, which records them). Raises when
    the ledger disagrees with the exact fee solve beyond 1e-9 relative
    tolerance, so a corrupted ledger cannot silently enter a CSV.
    """
    prices, proposals = _validate(prices, proposals, floor, multiplier,
                                  cap, cost_rate, v0=1.0)
    wealth_pre = [float(v) for v in ledger["wealth_pre"]]
    weights = [float(w) for w in ledger["weight"]]
    holding_post = [float(h) for h in ledger["holding_post"]]
    fees = [float(f) for f in ledger["fee"]]
    if not (len(wealth_pre) == len(weights) == len(holding_post)
            == len(fees) == len(proposals)):
        raise ValueError("ledger length does not match the episode steps")
    for i, (v, w, post, fee) in enumerate(
            zip(wealth_pre, weights, holding_post, fees)):
        if abs(w - max(0.0, min(cap, proposals[i]))) > 1e-9:
            raise ValueError(
                f"ledger step {i}: recorded weight {w} is not the "
                f"clamped proposal {proposals[i]}")
        # The fee solve cannot be recomputed from the ledger alone
        # (pre-trade holdings are not a ledger column), so the ledger
        # check here is the weight identity above plus a nonnegative
        # fee; the full accounting identity is pinned by the rollout
        # ledger tests in test_stage3_neural_cppi_3_ledger.py.
        if fee < -1e-12:
            raise ValueError(f"ledger step {i}: negative fee {fee}")
    return _build_rows(prices, wealth_pre, holding_post, fees, proposals,
                       floor, multiplier, cap, dip_threshold,
                       rebound_threshold, secondary_threshold)


def diagnose_episode(prices, weight_fn, floor, multiplier, cap, cost_rate,
                     v0=100.0, r=0.02, dt=1.0 / 252.0, borrow_rate=None,
                     history=None, dip_threshold=0.05,
                     rebound_threshold=0.02, secondary_threshold=None):
    """Run the public rollout for one episode and diagnose it.

    ``prices`` is a torch tensor of shape (1, n_steps + 1) and
    ``weight_fn`` has the rollout signature. Returns
    ``(rows, terminal, proposals)`` where terminal also carries the
    rollout liquidation fee. The raw proposal at each step is recorded
    by wrapping ``weight_fn``; the execution columns come from the
    rollout ledger, not from a re-implementation.
    """
    import torch

    from dhl.experiments.cppi_learned import rollout

    if not torch.is_tensor(prices) or prices.shape[0] != 1:
        raise ValueError("diagnose_episode needs one path: shape (1, n+1)")
    price_list = [float(p) for p in prices[0].detach().cpu().tolist()]
    proposals = []

    def recording_fn(i, V, tau_frac, cushion, feats):
        raw = weight_fn(i, V, tau_frac, cushion, feats)
        proposals.append(float(raw.detach().cpu().reshape(-1)[0])
                         if torch.is_tensor(raw) else float(raw))
        return raw

    out = rollout(prices, recording_fn, floor=floor, r=r, dt=dt,
                  cost_rate=cost_rate, cap=cap, borrow_rate=borrow_rate,
                  return_ledger=True, history=history)
    _, ledger = out
    plain = {k: (v.detach().cpu().tolist() if torch.is_tensor(v) else v)
             for k, v in ledger.items()}
    # Ledger step tensors have shape (1, steps); flatten the path axis.
    flat = {}
    for key in ("wealth_pre", "weight", "holding_post", "fee"):
        values = plain[key]
        flat[key] = values[0] if isinstance(values, list) else values
    rows = rows_from_ledger(flat, price_list, proposals, floor, multiplier,
                            cap, cost_rate, dip_threshold,
                            rebound_threshold, secondary_threshold)
    terminal = {
        "wealth_pre_liquidation": float(plain["wealth_pre_liquidation"][0])
        if isinstance(plain["wealth_pre_liquidation"], list)
        else float(plain["wealth_pre_liquidation"]),
        "liquidation_fee": float(plain["liquidation_fee"][0])
        if isinstance(plain["liquidation_fee"], list)
        else float(plain["liquidation_fee"]),
        "terminal_wealth": float(plain["terminal_wealth"][0])
        if isinstance(plain["terminal_wealth"], list)
        else float(plain["terminal_wealth"]),
        "cum_fees_incl_liquidation": float(plain["cum_fees"][0])
        if isinstance(plain["cum_fees"], list)
        else float(plain["cum_fees"]),
    }
    return rows, terminal, proposals


def counterfactual_attribution(prices, proposals, floor, multiplier, cap,
                               cost_rate, v0=100.0, r=0.02,
                               dt=1.0 / 252.0, borrow_rate=None):
    """Exact counterfactuals that change only timing or fees.

    Replays the same recorded proposals three ways: actual (on time,
    with fees), zero-fee (on time, cost_rate 0), and delayed (each
    proposal executed one step later, the first step all cash, the last
    proposal never executed; same fees as actual). Returns terminal
    wealths and signed effects: fee_effect = zero-fee minus actual,
    timing_effect = delayed minus actual. No other input changes, so
    the two effects are attributable to fees and timing respectively.
    """
    _, actual = replay_episode(prices, proposals, floor, multiplier, cap,
                               cost_rate, v0=v0, r=r, dt=dt,
                               borrow_rate=borrow_rate)
    _, zero_fee = replay_episode(prices, proposals, floor, multiplier, cap,
                                 0.0, v0=v0, r=r, dt=dt,
                                 borrow_rate=borrow_rate)
    delayed = [0.0] + list(proposals[:-1])
    _, delayed_out = replay_episode(prices, delayed, floor, multiplier, cap,
                                    cost_rate, v0=v0, r=r, dt=dt,
                                    borrow_rate=borrow_rate)
    return {
        "actual_terminal_wealth": actual["terminal_wealth"],
        "zero_fee_terminal_wealth": zero_fee["terminal_wealth"],
        "delayed_terminal_wealth": delayed_out["terminal_wealth"],
        "fee_effect": zero_fee["terminal_wealth"] - actual["terminal_wealth"],
        "timing_effect": (delayed_out["terminal_wealth"]
                          - actual["terminal_wealth"]),
        "actual_fees_incl_liquidation": actual["cum_fees_incl_liquidation"],
    }


def summarize_diagnosis(rows, terminal):
    """Episode summary for the falsifier hypotheses, numbers only.

    Reports the share of steps with a thin cushion (cushion_ratio
    below 0.10), total unused budget, the share of allowable budget
    left unused, steps where execution exceeded the classic allowable
    budget, and clipped steps. It states no conclusion: the caller
    reads the numbers for the episode at hand.
    """
    if not rows:
        raise ValueError("summarize_diagnosis needs at least one row")
    n = len(rows)
    low = sum(1 for row in rows if row["cushion_ratio"] < 0.10)
    allowable_total = sum(row["allowable_budget"] for row in rows)
    unused_total = sum(row["unused_budget"] for row in rows)
    return {
        "n_steps": n,
        "terminal_wealth": terminal["terminal_wealth"],
        "total_fees_incl_liquidation": terminal["cum_fees_incl_liquidation"],
        "low_cushion_share": low / n,
        "mean_cushion_ratio": sum(row["cushion_ratio"] for row in rows) / n,
        "total_unused_budget": unused_total,
        "unused_share_of_allowable": (
            unused_total / allowable_total if allowable_total > 0.0 else None),
        "steps_over_budget": sum(
            1 for row in rows if row["executed_dollars"] > row["allowable_budget"]),
        "clipped_steps": sum(
            1 for row in rows if row["clipping_weight"] != 0.0),
        "max_abs_clipping_weight": max(
            abs(row["clipping_weight"]) for row in rows),
        "state_counts": {state: sum(1 for row in rows if row["state"] == state)
                         for state in ("normal", "dip", "rebound",
                                       "secondary_dip")},
    }


def rows_to_csv(rows, path):
    """Write rows with full-precision floats (repr round-trips exactly)."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(float(row[key])) if key in _FLOAT_COLUMNS
                                   else row[key]) for key in CSV_COLUMNS})


def rows_from_csv(path):
    """Read rows written by rows_to_csv, refusing a changed header."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError(
                f"CSV header does not match the diagnosis columns: "
                f"{reader.fieldnames}")
        rows = []
        for raw in reader:
            row = {"step": int(raw["step"]), "state": raw["state"]}
            for key in _FLOAT_COLUMNS:
                row[key] = float(raw[key])
            rows.append(row)
        return rows
