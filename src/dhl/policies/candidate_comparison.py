"""Budget-action candidate comparison: the source contract (E5).

E5 compares budget-action policy candidates for the CPPI allocator.
Every candidate acts through the R1 budget-fraction contract
(``dhl.policies.budget_allocator``): the policy chooses the fraction
u in [0, 1] of the available risk budget, and the executed weight is
w_t = B_t * u. Four candidates are predeclared, and no other learned
candidate may join the comparison later:

- ``baseline_mlp``: a small MLP with a sigmoid fraction head (the
  direct form of ``BudgetFractionNet``).
- ``linear_logistic``: a linear model over the same inputs with a
  logistic (sigmoid) fraction (``LinearLogisticPolicy`` below).
- ``residual_utilization``: the residual CPPI utilization network
  (the residual form of ``BudgetFractionNet``, u = 1 - sigmoid(h);
  CPPI is the default and learning is the de-risking residual).
- ``dynamic_rule``: the deterministic vol-threshold utilization rule
  of R1 (full budget while 21d vol is calm, a reduced fraction above
  the threshold). It is never trained.

Matched conditions, fixed here before anything runs:

- Features and decision history. Every learned candidate consumes
  the identical input vector at every step: (tau_frac,
  cushion_ratio, vol, vol_ratio, drawdown, panic_z) — the full
  strictly causal feature set of ``dhl.experiments.cppi_learned``,
  computed once per step by the shared rollout and handed to every
  candidate unchanged. The feature protocol is cold for synthetic
  scenarios and real episodes alike: features are computed causally
  within each path from its own start, so the warm-up state is the
  same defined cold state for every candidate.
- Initialization. Every learned candidate starts from the same
  state-independent fraction sigmoid(2) = 0.8808: the networks via
  their zero final layer, the linear policy via zero weights. No
  candidate begins with a state-dependent advantage.
- Accounting and loss. Execution goes through the shared rollout
  and its ledger accounting for every candidate. Part 2 trains the
  learned candidates with the corrected multi-horizon path loss
  (``dhl.experiments.losses.multihorizon_loss``; every component
  adds, including the anytime-shortfall term).
- Selection budget. Part 2 selects on the validation family only,
  under ``hierarchical_select`` below. The evaluation families and
  the real episodes play no role in selection: 0 holdout-driven
  hyperparameter decisions.

Scenario protocol (``E5_SEED_SETS``): the baseline capture reuses
the R1 baseline seed family (GBM seeds 1000-1019, 16 paths x 63
steps, sigma 0.30, drift R), so the deterministic reference rows
(cash, classic CPPI) must reproduce the R1 capture exactly. The
training, validation, held-out stress and held-out gap families use
E5-only seeds (21000+, 22000+, 23000+, 24000+), each 20 scenarios of
16 paths x 63 steps mixing calm, stress and gap regimes by position
modulo 3, the R1 Part 2 regime construction. The final evaluation
episodes of Part 2 are SPY block-bootstrap episodes
(``EPISODE_PROTOCOL``) with declared seeds 31000-31019; they are
generated only in Part 2. Apart from the intentionally shared
baseline family, every E5 seed set is disjoint from every other E5
set and from every R1 seed set.

Part 1 scope: this contract, the candidate definitions, the
comparison harness, the hierarchical selection rule, the
training-log schema, the baseline capture over the baseline family,
and independent oracles (R1's bisection oracle for the accounting,
hand-computed cases for the new policy, the metrics, the selection
rule and the gradient statistic). No training run, no policy
promotion, no performance claim; training, selection and the
episode evaluation are Part 2.
"""

import torch
from torch import nn

from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    FEATURE_NAMES,
    FLOOR,
    FLOOR_PENALTY,
    R,
    V0,
    rollout,
)
from dhl.policies.budget_allocator import (
    CALM_SIGMA,
    LOGIT_CLIP,
    SEED_SETS as R1_SEED_SETS,
    STRESS_DRIFT,
    STRESS_SIGMA,
    BudgetFractionNet,
    budget_weight_fn,
    constant_fraction,
    gap_augment,
    ledger_self_financing_violations,
    make_gbm_scenarios,
    net_fraction_fn,
    risk_budget,
    validate_fraction,
    vol_threshold_fraction,
)

# --- the predeclared comparison ---------------------------------------------

#: Every candidate consumes exactly these features, in this order,
#: after (tau_frac, cushion_ratio). Matched by construction: the
#: rollout computes the feature dict once per step.
E5_FEATURE_NAMES = FEATURE_NAMES

#: The four predeclared candidates, in declaration order. Selection
#: ties break toward the earlier-declared candidate.
CANDIDATE_ORDER = (
    "baseline_mlp",
    "linear_logistic",
    "residual_utilization",
    "dynamic_rule",
)

#: Deterministic reference rows captured alongside the candidates.
#: They are not candidates and never enter selection.
REFERENCE_ORDER = ("cash", "classic")

#: Construction seed for the learned candidates' initialization.
CANDIDATE_SEED = 7

#: Hidden width of the two network candidates (the shipped default).
HIDDEN_DIMS = (16, 16)

MULTIPLIER = 4.0

#: Scenario families for E5. The baseline family is R1's, shared
#: deliberately for capture comparability; every other family is
#: E5-only. Episodes are seed declarations for Part 2's real-data
#: evaluation, not synthetic scenarios.
E5_SEED_SETS = {
    "baseline": R1_SEED_SETS["baseline"],
    "training": tuple(21000 + k for k in range(20)),
    "validation": tuple(22000 + k for k in range(20)),
    "stress": tuple(23000 + k for k in range(20)),
    "gap": tuple(24000 + k for k in range(20)),
    "episodes": tuple(31000 + k for k in range(20)),
}

#: Declared protocol for Part 2's final evaluation episodes: SPY
#: daily block bootstrap, the repo's real-episode generator.
EPISODE_PROTOCOL = {
    "generator": "dhl.markets.real.block_bootstrap_paths",
    "source": "dhl.markets.real.load_spy_daily",
    "block_len": 21,
    "horizon": 63,
    "seeds": [int(s) for s in E5_SEED_SETS["episodes"]],
}


def candidate_specs():
    """Metadata for the predeclared candidates and references."""
    return [
        {"name": "baseline_mlp", "kind": "learned",
         "trainable": True,
         "form": "BudgetFractionNet, direct (u = sigmoid(h))"},
        {"name": "linear_logistic", "kind": "learned",
         "trainable": True,
         "form": "LinearLogisticPolicy (u = sigmoid(w . x + b))"},
        {"name": "residual_utilization", "kind": "learned",
         "trainable": True,
         "form": "BudgetFractionNet, residual (u = 1 - sigmoid(h))"},
        {"name": "dynamic_rule", "kind": "deterministic",
         "trainable": False,
         "form": "R1 vol_threshold_fraction (never trained)"},
        {"name": "cash", "kind": "reference", "trainable": False,
         "form": "constant fraction 0"},
        {"name": "classic", "kind": "reference", "trainable": False,
         "form": "constant fraction 1 (classical CPPI)"},
    ]


# --- the linear/logistic candidate ------------------------------------------


class LinearLogisticPolicy(nn.Module):
    """Linear/logistic budget-fraction policy: state -> u in (0, 1).

    u = sigmoid(clamp(w . x + b)) over the same input vector the
    network candidates consume: (tau_frac, cushion_ratio,
    *feature_names), with the logit clamped to +/- ``logit_clip``
    before the sigmoid, exactly as in ``BudgetFractionNet``. The
    weights start at zero with bias ``init_bias``, so the initial
    fraction is sigmoid(``init_bias``) (0.8808 at the default 2.0)
    for every state: the matched, state-independent start shared by
    every learned E5 candidate.
    """

    def __init__(self, logit_clip=LOGIT_CLIP, init_bias=2.0,
                 feature_names=E5_FEATURE_NAMES):
        super().__init__()
        if logit_clip <= 0.0:
            raise ValueError(f"logit_clip must be positive, "
                             f"got {logit_clip}")
        self.logit_clip = float(logit_clip)
        self.feature_names = tuple(feature_names)
        for name in self.feature_names:
            if name not in FEATURE_NAMES:
                raise ValueError(f"unknown feature {name!r}")
        self.linear = nn.Linear(2 + len(self.feature_names), 1)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(float(init_bias))

    def forward(self, tau_frac, cushion_ratio, feats):
        cushion_ratio = torch.as_tensor(cushion_ratio,
                                        dtype=torch.float32)
        cols = [
            torch.as_tensor(tau_frac, dtype=torch.float32)
            .expand_as(cushion_ratio),
            cushion_ratio,
        ]
        for name in self.feature_names:
            cols.append(torch.as_tensor(feats[name],
                                        dtype=torch.float32))
        x = torch.stack(cols, dim=1)
        h = self.linear(x).squeeze(1)
        h = torch.clamp(h, min=-self.logit_clip, max=self.logit_clip)
        return torch.sigmoid(h)


# --- candidate construction ---------------------------------------------------


def candidate_model(name, seed=CANDIDATE_SEED):
    """The seeded contract object for a learned candidate.

    Returns None for the deterministic dynamic rule, which has no
    parameters. Construction seeds the torch stream first (the
    ``seeded_policy`` convention), so the initialization depends
    only on ``seed``.
    """
    if name == "baseline_mlp":
        torch.manual_seed(seed)
        return BudgetFractionNet(hidden_dims=HIDDEN_DIMS,
                                 form="direct",
                                 feature_names=E5_FEATURE_NAMES)
    if name == "linear_logistic":
        torch.manual_seed(seed)
        return LinearLogisticPolicy(feature_names=E5_FEATURE_NAMES)
    if name == "residual_utilization":
        torch.manual_seed(seed)
        return BudgetFractionNet(hidden_dims=HIDDEN_DIMS,
                                 form="residual",
                                 feature_names=E5_FEATURE_NAMES)
    if name == "dynamic_rule":
        return None
    raise ValueError(f"unknown candidate {name!r}; expected one of "
                     f"{list(CANDIDATE_ORDER)}")


def candidate_fraction_fn(name, seed=CANDIDATE_SEED):
    """The rollout fraction function for a predeclared candidate."""
    if name == "dynamic_rule":
        return vol_threshold_fraction()
    model = candidate_model(name, seed=seed)
    if model is None:  # pragma: no cover - dynamic_rule handled above
        raise ValueError(f"candidate {name!r} has no model")
    return net_fraction_fn(model)


def reference_fraction_fn(name):
    """The rollout fraction function for a deterministic reference."""
    if name == "cash":
        return constant_fraction(0.0)
    if name == "classic":
        return constant_fraction(1.0)
    raise ValueError(f"unknown reference {name!r}; expected one of "
                     f"{list(REFERENCE_ORDER)}")


# --- scenario protocol ---------------------------------------------------------


def _regime_scenario(seed, regime, n_paths, n_steps, dtype):
    """One training/validation scenario: calm, stress or gap.

    Mirrors the R1 Part 2 regime construction (position modulo 3 in
    the caller's seed list picks the regime) on E5's own seeds.
    """
    if regime == 1:
        return make_gbm_scenarios([seed], n_paths, n_steps,
                                  sigma=STRESS_SIGMA,
                                  drift=STRESS_DRIFT, dtype=dtype)
    calm = make_gbm_scenarios([seed], n_paths, n_steps,
                              sigma=CALM_SIGMA, drift=R, dtype=dtype)
    if regime == 2:
        return [(seed, gap_augment(calm[0][1], seed=seed))]
    return calm


def e5_scenarios(name, n_paths=16, n_steps=63,
                 dtype=torch.float64):
    """One declared synthetic scenario family by name.

    The baseline family is R1's baseline construction (calm GBM,
    sigma 0.30) on the shared seeds. Stress is the high-vol
    negative-drift GBM family; gap is calm GBM with one seeded
    GAP_JUMP drop per path; training and validation mix the three
    regimes by position modulo 3. The episode family is real data
    and is generated only in Part 2, so requesting it here raises.
    """
    if name == "episodes":
        raise ValueError(
            "episode scenarios are SPY block-bootstrap paths, "
            "generated in Part 2 under EPISODE_PROTOCOL; Part 1 "
            "declares their seeds only")
    if name not in E5_SEED_SETS:
        raise ValueError(f"unknown scenario set {name!r}; expected "
                         f"one of {sorted(E5_SEED_SETS)}")
    seeds = E5_SEED_SETS[name]
    if name == "baseline":
        return make_gbm_scenarios(seeds, n_paths, n_steps,
                                  sigma=CALM_SIGMA, drift=R,
                                  dtype=dtype)
    if name == "stress":
        return make_gbm_scenarios(seeds, n_paths, n_steps,
                                  sigma=STRESS_SIGMA,
                                  drift=STRESS_DRIFT, dtype=dtype)
    if name == "gap":
        return [(seed, gap_augment(S, seed=seed))
                for seed, S in make_gbm_scenarios(
                    seeds, n_paths, n_steps, sigma=CALM_SIGMA,
                    drift=R, dtype=dtype)]
    scenarios = []
    for pos, seed in enumerate(seeds):
        scenarios += _regime_scenario(seed, pos % 3, n_paths,
                                      n_steps, dtype)
    return scenarios


# --- metrics -------------------------------------------------------------------


def max_drawdown(marks):
    """Maximum peak-to-trough decline of one wealth path, in
    currency units: max over t of (max_{s <= t} V_s - V_t)."""
    peak = None
    worst = 0.0
    for v in marks:
        v = float(v)
        peak = v if peak is None else max(peak, v)
        worst = max(worst, peak - v)
    return worst


# --- the comparison harness -----------------------------------------------------


def evaluate_candidates(scenarios, candidates, multiplier=MULTIPLIER,
                        floor=FLOOR, cap=CAP, cost_rate=COST_RATE):
    """Evaluate fraction candidates on identical scenarios.

    ``candidates`` maps names to fraction functions; every candidate
    runs through the shared rollout on the same price tensors, so
    paths, cash-rate convention, leverage cap and fee schedule are
    identical by construction. Each row carries the full acceptance
    field set: n, mean terminal wealth and return, terminal
    shortfall, terminal and anytime breach counts, anytime
    shortfall (mean relu(floor - worst mark)), mean maximum
    drawdown, costs (mean fees), negative outcomes (count of paths
    ending below V0, and the worst terminal wealth), the fraction
    range and mean, the regulatory activation shares (steps at the
    leverage cap; steps with an exhausted budget), and the
    candidate's own self-financing violation count from the ledger
    audit. ``seeds`` are the scenario seeds, shared by every row.
    """
    rows = []
    total_violations = 0
    for name, frac_fn in candidates.items():
        agg = {
            "wealth": [], "shortfall": [], "anytime_shortfall": [],
            "drawdown": [], "fees": [],
            "breach_terminal": 0, "breach_anytime": 0,
            "n_negative": 0, "min_terminal": float("inf"),
            "frac_sum": 0.0, "frac_count": 0,
            "fraction_min": float("inf"),
            "fraction_max": float("-inf"),
            "cap_steps": 0, "zero_budget_steps": 0, "steps": 0,
            "violations": 0,
        }

        def recording(i, V, tau_frac, cushion, feats, _fn=frac_fn,
                      _agg=agg):
            u = validate_fraction(
                _fn(i, V, tau_frac, cushion, feats))
            B = risk_budget(V, floor=floor, multiplier=multiplier,
                            cap=cap)
            _agg["frac_sum"] += float(u.sum())
            _agg["frac_count"] += int(u.numel())
            _agg["fraction_min"] = min(_agg["fraction_min"],
                                       float(u.min()))
            _agg["fraction_max"] = max(_agg["fraction_max"],
                                       float(u.max()))
            _agg["zero_budget_steps"] += int((B <= 0.0).sum())
            _agg["steps"] += int(u.numel())
            return u

        with torch.no_grad():
            for _, S in scenarios:
                V_T, path, ledger = rollout(
                    S, budget_weight_fn(recording,
                                        multiplier=multiplier,
                                        floor=floor, cap=cap),
                    floor=floor, cost_rate=cost_rate, cap=cap,
                    return_path=True, return_ledger=True)
                agg["wealth"].append(V_T)
                agg["shortfall"].append(
                    torch.clamp(floor - V_T, min=0.0))
                worst_mark = path.min(dim=1).values
                agg["anytime_shortfall"].append(
                    torch.clamp(floor - worst_mark, min=0.0))
                agg["drawdown"].append(torch.tensor(
                    [max_drawdown(row) for row in path.tolist()],
                    dtype=torch.float64))
                agg["fees"].append(ledger["cum_fees"])
                agg["breach_terminal"] += int((V_T < floor).sum())
                agg["breach_anytime"] += int(
                    (worst_mark < floor).sum())
                agg["n_negative"] += int((V_T < V0).sum())
                agg["min_terminal"] = min(agg["min_terminal"],
                                          float(V_T.min()))
                agg["cap_steps"] += int(
                    (ledger["weight"] >= cap - 1e-12).sum())
                agg["violations"] += \
                    ledger_self_financing_violations(ledger)
        wealth = torch.cat(agg["wealth"])
        shortfall = torch.cat(agg["shortfall"])
        anytime_shortfall = torch.cat(agg["anytime_shortfall"])
        drawdown = torch.cat(agg["drawdown"])
        fees = torch.cat(agg["fees"])
        rows.append({
            "candidate": name,
            "n_paths": int(wealth.shape[0]),
            "mean_terminal_wealth": float(wealth.mean()),
            "mean_return": float(wealth.mean()) / V0 - 1.0,
            "mean_terminal_shortfall": float(shortfall.mean()),
            "breach_count_terminal": agg["breach_terminal"],
            "breach_count_anytime": agg["breach_anytime"],
            "mean_anytime_shortfall": float(anytime_shortfall.mean()),
            "mean_max_drawdown": float(drawdown.mean()),
            "mean_fees": float(fees.mean()),
            "n_negative_terminal": agg["n_negative"],
            "min_terminal_wealth": agg["min_terminal"],
            "mean_fraction": agg["frac_sum"] / agg["frac_count"],
            "fraction_min": agg["fraction_min"],
            "fraction_max": agg["fraction_max"],
            "cap_activation_share": agg["cap_steps"] / agg["steps"],
            "budget_zero_share": agg["zero_budget_steps"]
            / agg["steps"],
            "self_financing_violations": agg["violations"],
        })
        total_violations += agg["violations"]
    return {
        "n_scenarios": len(scenarios),
        "seeds": [int(seed) for seed, _ in scenarios],
        "multiplier": multiplier,
        "floor": floor,
        "cap": cap,
        "cost_rate": cost_rate,
        "feature_names": list(E5_FEATURE_NAMES),
        "candidates": rows,
        "self_financing_violations": total_violations,
    }


# --- hierarchical selection (executed in Part 2, fixed here) -------------------


def hierarchical_select(rows, reference_row):
    """The predeclared selection hierarchy over validation rows.

    Tier 1 (accounting): discard any candidate with a self-financing
    violation or an observed fraction outside [0, 1]. Tier 2
    (shortfall): discard any candidate whose anytime-breach count
    exceeds the classic CPPI reference (``reference_row``, the
    ``classic`` row on the same validation paths) — a candidate that
    breaches more often than classic CPPI has failed the insurance
    purpose. Tier 3 (utility): among the survivors select the
    highest validation utility E[V_T] - FLOOR_PENALTY *
    E[terminal shortfall], the repo objective family; ties break
    toward the earlier row in ``rows`` (callers pass declaration
    order). Only validation rows are ever passed here: evaluation
    families and episodes play no role in selection.

    Returns {"selected", "discarded", "survivors", "utility"} with
    ``selected`` None when every candidate is discarded.
    """
    if reference_row is None:
        raise ValueError("selection requires the classic CPPI "
                         "reference row on the same paths")
    discarded = {}
    survivors = []
    for row in rows:
        name = row["candidate"]
        if row["self_financing_violations"] > 0:
            discarded[name] = "self-financing accounting violations"
            continue
        if row["fraction_min"] < -1e-12 or row["fraction_max"] > 1.0 + 1e-12:
            discarded[name] = "fraction outside [0, 1]"
            continue
        if (row["breach_count_anytime"]
                > reference_row["breach_count_anytime"]):
            discarded[name] = ("anytime breaches exceed the classic "
                               "CPPI reference")
            continue
        survivors.append(row)
    utility = {
        row["candidate"]: row["mean_terminal_wealth"]
        - FLOOR_PENALTY * row["mean_terminal_shortfall"]
        for row in survivors
    }
    selected = None
    best = None
    for row in survivors:
        u = utility[row["candidate"]]
        if best is None or u > best:
            best = u
            selected = row["candidate"]
    return {
        "selected": selected,
        "discarded": discarded,
        "survivors": [row["candidate"] for row in survivors],
        "utility": utility,
    }


# --- the Part 2 training-log schema (declared and validated here) ---------------


#: A gradient entry counts as null when its magnitude is at most
#: this epsilon (the gradient null fraction is the share of null
#: entries over all parameter gradients at a logged epoch).
GRADIENT_NULL_EPS = 1e-12

REQUIRED_TRAINING_LOG_FIELDS = (
    "candidate", "seed", "policy_init_hash", "loss", "epochs",
)

REQUIRED_EPOCH_FIELDS = (
    "epoch", "loss",
    "neg_expected_terminal_wealth", "terminal_shortfall",
    "anytime_shortfall", "turnover",
    "gradient_null_fraction", "cap_activation_share",
    "budget_zero_share",
)


def gradient_null_fraction(grads):
    """Share of gradient entries with |g| <= GRADIENT_NULL_EPS.

    ``grads`` is an iterable of gradient tensors (or None entries,
    which are skipped: a parameter the loss never touches carries
    no gradient evidence either way). Raises on an empty gradient
    set, because a null fraction over no gradients is undefined,
    not zero.
    """
    total = 0
    null = 0
    for g in grads:
        if g is None:
            continue
        total += int(g.numel())
        null += int((g.abs() <= GRADIENT_NULL_EPS).sum())
    if total == 0:
        raise ValueError("gradient null fraction needs at least "
                         "one gradient entry")
    return null / total


def validate_training_log(record):
    """Fail closed on a Part 2 training log missing declared fields.

    A training log records one candidate's run: the candidate name,
    the run seed, the policy init hash, the loss identifier (the
    corrected multi-horizon path loss), and the per-epoch learning
    curve. Every epoch entry carries the loss decomposition (the
    four named components of ``multihorizon_loss``), the gradient
    null fraction, and the regulatory activation shares (steps at
    the leverage cap, steps with an exhausted budget). Returns True
    for a conforming record; raises ValueError otherwise.
    """
    if not isinstance(record, dict):
        raise ValueError("training log must be a dict")
    for field in REQUIRED_TRAINING_LOG_FIELDS:
        if field not in record:
            raise ValueError(f"training log missing field {field!r}")
    epochs = record["epochs"]
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("training log needs a non-empty epoch list")
    for entry in epochs:
        for field in REQUIRED_EPOCH_FIELDS:
            if field not in entry:
                raise ValueError(
                    f"training log epoch missing field {field!r}")
        for share in ("gradient_null_fraction",
                      "cap_activation_share", "budget_zero_share"):
            if not 0.0 <= float(entry[share]) <= 1.0:
                raise ValueError(
                    f"training log epoch field {share!r} must lie "
                    f"in [0, 1], got {entry[share]}")
    return True
