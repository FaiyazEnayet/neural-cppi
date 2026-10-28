"""Comparator ladder: one factory, one kernel, one table.

The factory ``make_weight_fn`` is the only place a comparator
strategy is built, and every rung it builds runs through the same
``rollout`` in ``dhl.experiments.cppi_learned`` with the same
accounting kernel, so floors, caps, financing rates, rebalance
frequency, costs, and initial state match by construction. A rung
whose cap differs (for example a learned policy wrapped at 1.0 while
the raw policy allows 1.5) records the cap actually applied in its
table row, so the clip is disclosed, not hidden.

Selection honesty: rungs whose parameters were tuned carry
``selection_kind="tuned"`` plus their tuning evidence. A tuned rung
with no tuning evidence, or tuning data that overlaps the evaluation
episode, is refused fail-closed by ``validate_selection``. Fixed
rungs (no tuned parameter) need no tuning evidence. The optimized-m
rung is produced by ``select_multiplier``, which only ever sees
tuning paths: changing the test paths cannot change the selection.
"""

import math
from dataclasses import dataclass, field

import torch

from dhl.benchmarks import cppi_rules as rules
from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    DT,
    FLOOR,
    R,
    V0,
    rollout,
)

LADDER_KINDS = (
    "all_cash",
    "buy_hold",
    "classic_m4",
    "classic_m_opt",
    "classic_m4_brake",
    "dynamic_vol",
    "budget_fraction",
    "fixed_cap",
    "trend_vol",
    "learned",
    "learned_constrained",
    "learned_constrained_brake",
)

LEGAL_TUNING_SPLITS = ("train", "validation")

TABLE_COLUMNS = (
    "mean_wealth",
    "std_wealth",
    "min_wealth",
    "mean_min_wealth",
    "breach_count_terminal",
    "breach_count_anytime",
    "drawdown_mean",
    "drawdown_max",
    "turnover",
    "fees",
    "avg_leverage",
    "max_weight",
    "risk_adjusted",
    "mean_shortfall",
    "selection",
    "applied_cap",
    "n_paths",
    "n_steps",
)


@dataclass(frozen=True)
class Conventions:
    """Shared conventions every rung runs under."""

    floor: float = FLOOR
    cap: float = CAP
    cost_rate: float = COST_RATE
    r: float = R
    dt: float = DT
    borrow_rate: float | None = None


@dataclass
class ComparatorConfig:
    """One ladder rung: kind, parameters, and selection provenance."""

    name: str
    kind: str
    params: dict = field(default_factory=dict)
    selection_kind: str = "fixed"
    tuning_split: str | None = None
    tuning_start: str | None = None
    tuning_end: str | None = None
    selection: str = ""


def effective_cap(config, conventions):
    """The cap a rung actually runs under (per-rung override wins)."""
    return float(config.params.get("cap", conventions.cap))


def make_weight_fn(config, conventions, policy=None):
    """Build one rung's weight function. The only factory for rungs.

    ``policy`` is required for learned kinds and ignored otherwise.
    Unknown kinds and invalid parameters raise instead of building a
    silently different strategy.
    """
    kind = config.kind
    p = config.params
    cap = effective_cap(config, conventions)
    floor = conventions.floor
    m = float(p.get("multiplier", 4.0))
    if kind == "all_cash":
        return rules.cash_weight_fn()
    if kind == "buy_hold":
        return rules.buy_hold_weight_fn()
    if kind in ("classic_m4", "classic_m_opt"):
        return rules.classic_weight_fn(multiplier=m, floor=floor, cap=cap)
    if kind == "classic_m4_brake":
        return rules.brake_weight_fn(
            multiplier=m, floor=floor, cap=cap,
            emergency_vol=float(p.get("emergency_vol",
                                      rules.SHIPPED_EMERGENCY_VOL)),
            emergency_multiplier=p.get(
                "emergency_multiplier", rules.SHIPPED_EMERGENCY_MULTIPLIER))
    if kind == "dynamic_vol":
        return rules.dynamic_vol_weight_fn(
            multiplier=m, floor=floor, cap=cap,
            vol_target=float(p.get("vol_target", 0.20)),
            m_min=float(p.get("m_min", 1.0)),
            m_max=float(p.get("m_max", 8.0)))
    if kind == "budget_fraction":
        return rules.budget_fraction_weight_fn(
            multiplier=m, floor=floor, cap=cap,
            fraction=float(p.get("fraction", 1.0)))
    if kind == "fixed_cap":
        return rules.fixed_cap_weight_fn(
            weight=float(p.get("weight", 0.5)), cap=cap)
    if kind == "trend_vol":
        return rules.trend_vol_weight_fn(multiplier=m, floor=floor, cap=cap)
    if kind in ("learned", "learned_constrained", "learned_constrained_brake"):
        if policy is None:
            raise ValueError(
                f"comparator kind {kind!r} requires a policy")
        fns = rules.learned_weight_fns(
            policy, multiplier=m, floor=floor, cap=cap,
            emergency_vol=float(p.get("emergency_vol",
                                      rules.SHIPPED_EMERGENCY_VOL)),
            emergency_multiplier=p.get(
                "emergency_multiplier", rules.SHIPPED_EMERGENCY_MULTIPLIER))
        return fns[kind]
    raise ValueError(f"unknown comparator kind {kind!r}")


def validate_selection(config, eval_split=None, registry=None, dates=None):
    """Fail-closed check that a rung's tuning data is legal.

    Fixed rungs (no tuned parameter) always pass. Tuned rungs pass
    only with tuning evidence: dated tuning windows validated against
    the episode registry (60-bar feature lookback embargo), or a
    tuning split in the legal set that is not the evaluation split.
    """
    if config.selection_kind == "fixed":
        return None
    if config.selection_kind != "tuned":
        raise ValueError(
            f"comparator {config.name!r} has unknown selection_kind "
            f"{config.selection_kind!r}")
    if (config.tuning_start is not None
            and config.tuning_end is not None):
        if registry is None or dates is None:
            raise ValueError(
                f"tuned comparator {config.name!r} declares dated tuning "
                f"data but no registry/dates were supplied to validate it")
        registry.validate_training_period(
            dates, config.tuning_start, config.tuning_end,
            lookback_bars=60)
        return None
    if config.tuning_split is not None:
        if config.tuning_split not in LEGAL_TUNING_SPLITS:
            raise ValueError(
                f"tuned comparator {config.name!r} tunes on split "
                f"{config.tuning_split!r}, which is not a legal tuning "
                f"split {LEGAL_TUNING_SPLITS}")
        if eval_split is not None and config.tuning_split == eval_split:
            raise ValueError(
                f"tuned comparator {config.name!r} tunes on the eval "
                f"split {eval_split!r}; tuning data must be legal "
                f"train/validation only")
        return None
    raise ValueError(
        f"tuned comparator {config.name!r} declares no tuning data: "
        f"give tuning_start/tuning_end with a registry, or a tuning_split")


def select_multiplier(candidates, tuning_S, conventions):
    """Pick a fixed CPPI multiplier on tuning paths only.

    Each candidate runs the classic rule on the tuning paths. The
    winner minimizes terminal breaches, then maximizes mean terminal
    wealth; ties go to the earliest candidate. The function accepts
    tuning paths and nothing else, so test paths cannot influence the
    choice. Returns ``(best_m, provenance)``.
    """
    S = tuning_S if torch.is_tensor(tuning_S) else torch.as_tensor(
        tuning_S, dtype=torch.float32)
    best = None
    for m in candidates:
        fn = rules.classic_weight_fn(
            multiplier=float(m), floor=conventions.floor,
            cap=conventions.cap)
        with torch.no_grad():
            V_T = rollout(S, fn, floor=conventions.floor, r=conventions.r,
                          dt=conventions.dt,
                          cost_rate=conventions.cost_rate,
                          borrow_rate=conventions.borrow_rate,
                          cap=conventions.cap)
        breaches = int((V_T < conventions.floor).sum())
        mean_wealth = float(V_T.mean())
        key = (breaches, -mean_wealth)
        if best is None or key < best[0]:
            best = (key, float(m), breaches, mean_wealth)
    _, best_m, breaches, mean_wealth = best
    provenance = {
        "selection_kind": "tuned",
        "criterion": "min terminal breaches, then max mean wealth",
        "candidates": [float(c) for c in candidates],
        "n_tuning_paths": int(S.shape[0]),
        "n_steps": int(S.shape[1] - 1),
        "best_mean_wealth": mean_wealth,
        "best_breach_count": breaches,
    }
    return best_m, provenance


def _row_metrics(V_T, path, ledger, floor, conventions, config):
    """Table row from one rollout's path and ledger."""
    steps = path.shape[1] - 1
    horizon = steps * conventions.dt
    peak = torch.cummax(path, dim=1).values
    dd = (peak - path) / peak.clamp_min(1e-8)
    per_path_max_dd = dd.max(dim=1).values
    mean_wealth = float(V_T.mean())
    std_wealth = float(V_T.std())
    cash_terminal = V0 * math.exp(conventions.r * horizon)
    risk_adjusted = 0.0 if std_wealth == 0.0 else (
        (mean_wealth - cash_terminal) / std_wealth)
    return {
        "mean_wealth": mean_wealth,
        "std_wealth": std_wealth,
        "min_wealth": float(path.min()),
        "mean_min_wealth": float(path.min(dim=1).values.mean()),
        "breach_count_terminal": int((V_T < floor).sum()),
        "breach_count_anytime": int(
            (path.min(dim=1).values < floor).sum()),
        "drawdown_mean": float(per_path_max_dd.mean()),
        "drawdown_max": float(per_path_max_dd.max()),
        "turnover": float(
            ((ledger["notional"].sum(dim=1)
              + ledger["liquidation_notional"]) / V0).mean()),
        "fees": float(ledger["cum_fees"].mean()),
        "avg_leverage": float(ledger["weight"].mean()),
        "max_weight": float(ledger["weight"].max()),
        "risk_adjusted": risk_adjusted,
        "mean_shortfall": float(
            torch.relu(torch.full_like(V_T, floor) - V_T).mean()),
        "selection": config.selection or (
            "fixed a priori" if config.selection_kind == "fixed"
            else "tuned"),
        "applied_cap": effective_cap(config, conventions),
        "n_paths": int(V_T.shape[0]),
        "n_steps": steps,
    }


def run_ladder(S, configs, conventions, policy=None, eval_split=None,
               registry=None, dates=None):
    """Run every rung through the one kernel; return the table.

    Each rung's selection is validated before it runs, and every row
    is checked to carry all table columns. Wealth paths come from the
    same ``rollout`` with the same ledger, so cross-rung differences
    are strategy differences, never accounting differences.
    """
    S_t = S if torch.is_tensor(S) else torch.as_tensor(S, dtype=torch.float32)
    table = {}
    for config in configs:
        validate_selection(config, eval_split=eval_split,
                           registry=registry, dates=dates)
        fn = make_weight_fn(config, conventions, policy=policy)
        with torch.no_grad():
            V_T, path, ledger = rollout(
                S_t, fn, floor=conventions.floor, r=conventions.r,
                dt=conventions.dt, cost_rate=conventions.cost_rate,
                borrow_rate=conventions.borrow_rate,
                cap=effective_cap(config, conventions),
                return_path=True, return_ledger=True)
        row = _row_metrics(V_T, path, ledger, conventions.floor,
                           conventions, config)
        missing = set(TABLE_COLUMNS) - set(row)
        if missing:
            raise ValueError(
                f"comparator {config.name!r} row is missing {missing}")
        table[config.name] = row
    return table


def frontier_points(table):
    """Risk-return frontier points: one per rung, with floor shortfall.

    Risk is terminal-wealth std, return is mean terminal wealth, and
    shortfall is mean wealth below the floor. Same multiplier does not
    mean same risk, so rungs are compared on this frontier, never on
    mean wealth alone.
    """
    return [{
        "name": name,
        "risk": row["std_wealth"],
        "return": row["mean_wealth"],
        "shortfall": row["mean_shortfall"],
    } for name, row in table.items()]
