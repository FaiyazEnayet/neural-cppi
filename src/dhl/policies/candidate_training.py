"""Part 2 of E5: train, select and evaluate the candidate comparison.

Part 1 (``dhl.policies.candidate_comparison``) fixed the contract:
four predeclared candidates through the R1 budget-fraction contract,
matched features and initialization, identical scenarios, and the
hierarchical selection rule on validation rows only. This module
executes that contract, with the training choices predeclared here
so nothing is tuned after seeing a holdout family:

- Training family: the E5 training scenarios (seeds 21000-21019),
  mixed calm/stress/gap regimes by position modulo 3.
- Objective: the corrected multi-horizon path loss
  (``dhl.experiments.losses.multihorizon_loss``), every component
  adding: floor penalty 30 on terminal shortfall, anytime penalty
  30 on anytime shortfall (the factorial "correct" level), turnover
  penalty 0. The loss is a training objective only; selection uses
  the validation utility E[V_T] - 30 * E[terminal shortfall] from
  ``hierarchical_select``, never the training loss.
- Optimizer: full-batch Adam, lr 3e-3, gradient norm clipped at
  1.0 (the repo training convention), 50 epochs for the full run.
  Every learned candidate starts from its Part 1 matched
  initialization (seed 7) and trains on the identical batch, so
  differences after training are architecture and learning, not
  data, budget or starting point.
- The dynamic rule is never trained: it is evaluated untrained in
  every family, exactly as declared.
- Selection: ``hierarchical_select`` on the validation family
  only. The stress, gap and SPY episode families are evaluation
  only and cannot change the winner: 0 holdout-driven decisions.
- Episodes: SPY block-bootstrap paths (block length 21, horizon
  63) from the committed in-repo SPY daily series, one 16-path
  scenario per declared seed 31000-31019, generated here for the
  first time. The episodes are an evaluation family, not a
  training source, and no episode seed is used for training.
"""

import torch

from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    FLOOR,
    FLOOR_PENALTY,
    policy_init_hash,
    rollout,
)
from dhl.experiments.losses import multihorizon_loss
from dhl.markets.real import block_bootstrap_paths, load_spy_daily, log_returns
from dhl.policies.budget_allocator import (
    budget_weight_fn,
    risk_budget,
)
from dhl.policies.candidate_comparison import (
    CANDIDATE_ORDER,
    CANDIDATE_SEED,
    EPISODE_PROTOCOL,
    E5_SEED_SETS,
    MULTIPLIER,
    candidate_fraction_fn,
    candidate_model,
    e5_scenarios,
    evaluate_candidates,
    gradient_null_fraction,
    hierarchical_select,
    reference_fraction_fn,
)

#: Predeclared Part 2 training choices (see the module docstring).
PART2_TRAINING = {
    "n_epochs": 50,
    "lr": 3e-3,
    "train_seed": 0,
    "floor_penalty": float(FLOOR_PENALTY),
    "anytime_penalty": float(FLOOR_PENALTY),
    "turnover_penalty": 0.0,
    "multiplier": MULTIPLIER,
    "floor": FLOOR,
    "cap": CAP,
    "cost_rate": COST_RATE,
}


def fraction_fn_for_model(model):
    """Adapt a trained fraction model to the rollout fraction shape."""

    def fn(i, V, tau_frac, cushion, feats):
        u = model(tau_frac, cushion, feats)
        return u.to(dtype=V.dtype) if torch.is_tensor(u) else u

    return fn


def _validate_scenarios(scenarios):
    if not scenarios:
        raise ValueError("training needs at least one scenario")
    steps = {S.shape[1] for _, S in scenarios}
    if len(steps) != 1:
        raise ValueError("training scenarios must share a step count")


def train_candidate(name, scenarios, n_epochs=None, lr=None,
                    train_seed=None):
    """Train one predeclared candidate on the given scenarios.

    Returns ``(model, log)`` where ``model`` is the trained network
    (None for the dynamic rule, which is never trained) and ``log``
    conforms to the Part 1 training-log schema
    (``validate_training_log``) for learned candidates. Epoch 0 is
    the pre-update evaluation; its gradient null fraction comes
    from a probe backward on the initial model, which changes no
    parameter. Each epoch entry records the loss decomposition, the
    gradient null fraction, and the regulatory activation shares
    (steps at the leverage cap from the ledger weight, steps with
    an exhausted budget from the recorded budget).
    """
    if name not in CANDIDATE_ORDER:
        raise ValueError(f"unknown candidate {name!r}; expected one "
                         f"of {list(CANDIDATE_ORDER)}")
    _validate_scenarios(scenarios)
    n_epochs = PART2_TRAINING["n_epochs"] if n_epochs is None \
        else int(n_epochs)
    lr = PART2_TRAINING["lr"] if lr is None else float(lr)
    train_seed = PART2_TRAINING["train_seed"] if train_seed is None \
        else int(train_seed)
    if n_epochs < 0:
        raise ValueError("n_epochs must be non-negative")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if name == "dynamic_rule":
        return None, {
            "candidate": name,
            "trained": False,
            "seed": CANDIDATE_SEED,
            "training_seeds": [],
            "loss": "multihorizon_loss",
            "epochs": [],
        }
    torch.manual_seed(train_seed)
    model = candidate_model(name, seed=CANDIDATE_SEED)
    init_hash = policy_init_hash(model)
    base_fn = fraction_fn_for_model(model)
    S_all = torch.cat([S for _, S in scenarios], dim=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def evaluate():
        counts = {"zero_budget": 0, "steps": 0}

        def tracked(i, V, tau_frac, cushion, feats):
            u = base_fn(i, V, tau_frac, cushion, feats)
            B = risk_budget(V, floor=FLOOR,
                            multiplier=MULTIPLIER, cap=CAP)
            counts["zero_budget"] += int((B <= 0.0).sum())
            counts["steps"] += int(u.numel())
            return u

        with torch.no_grad():
            V_T, path, ledger = rollout(
                S_all, budget_weight_fn(tracked,
                                        multiplier=MULTIPLIER,
                                        floor=FLOOR, cap=CAP),
                floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
                return_path=True, return_ledger=True)
            total, components = multihorizon_loss(
                V_T, path, weight_path=ledger["weight"],
                floor=FLOOR,
                floor_penalty=PART2_TRAINING["floor_penalty"],
                anytime_penalty=PART2_TRAINING["anytime_penalty"],
                turnover_penalty=PART2_TRAINING["turnover_penalty"])
        cap_share = float(
            (ledger["weight"] >= CAP - 1e-12).float().mean())
        return {
            "loss": float(total.detach()),
            "neg_expected_terminal_wealth": float(
                components["neg_expected_terminal_wealth"].detach()),
            "terminal_shortfall": float(
                components["terminal_shortfall"].detach()),
            "anytime_shortfall": float(
                components["anytime_shortfall"].detach()),
            "turnover": float(components["turnover"].detach()),
            "cap_activation_share": cap_share,
            "budget_zero_share": counts["zero_budget"]
            / counts["steps"] if counts["steps"] else 0.0,
        }

    def forward_with_grad():
        V_T, path, ledger = rollout(
            S_all, budget_weight_fn(base_fn,
                                    multiplier=MULTIPLIER,
                                    floor=FLOOR, cap=CAP),
            floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
            return_path=True, return_ledger=True)
        total, _components = multihorizon_loss(
            V_T, path, weight_path=ledger["weight"],
            floor=FLOOR,
            floor_penalty=PART2_TRAINING["floor_penalty"],
            anytime_penalty=PART2_TRAINING["anytime_penalty"],
            turnover_penalty=PART2_TRAINING["turnover_penalty"])
        return total

    epochs = []
    metrics = evaluate()
    probe = forward_with_grad()
    probe.backward()
    null0 = gradient_null_fraction(
        [p.grad for p in model.parameters()])
    model.zero_grad()
    epochs.append({"epoch": 0, **metrics,
                   "gradient_null_fraction": null0})
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss = forward_with_grad()
        loss.backward()
        null = gradient_null_fraction(
            [p.grad for p in model.parameters()])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics = evaluate()
        epochs.append({"epoch": epoch, **metrics,
                       "gradient_null_fraction": null})
    log = {
        "candidate": name,
        "seed": CANDIDATE_SEED,
        "policy_init_hash": init_hash,
        "loss": "multihorizon_loss",
        "epochs": epochs,
        "training_seeds": [int(s) for s, _ in scenarios],
        "trained": True,
    }
    return model, log


def spy_episode_source():
    """Provenance of the committed SPY series behind the episodes."""
    spy = load_spy_daily()
    returns = log_returns(spy["adjclose"])
    return {
        "source": EPISODE_PROTOCOL["source"],
        "n_bars": len(spy["dates"]),
        "first_date": spy["dates"][0],
        "last_date": spy["dates"][-1],
        "n_log_returns": int(len(returns)),
    }


def episode_scenarios(n_paths=16, n_steps=63, seeds=None,
                      block_len=None):
    """SPY block-bootstrap episode scenarios, one per declared seed.

    Each seed generates ``n_paths`` price paths of ``n_steps``
    steps from the committed SPY daily log-return series with
    circular block resampling (block length 21 trading days).
    Deterministic in the seed: the same seed always yields the
    same paths, and no episode path is used for training.
    """
    seeds = E5_SEED_SETS["episodes"] if seeds is None \
        else tuple(int(s) for s in seeds)
    block_len = EPISODE_PROTOCOL["block_len"] if block_len is None \
        else int(block_len)
    spy = load_spy_daily()
    returns = log_returns(spy["adjclose"])
    scenarios = []
    for seed in seeds:
        paths = block_bootstrap_paths(
            returns, n_paths, n_steps, block_len=block_len,
            S0=100.0, seed=seed)
        scenarios.append((seed, torch.as_tensor(
            paths, dtype=torch.float64)))
    return scenarios


def run_part2(n_epochs=None, families=None, n_paths=16, n_steps=63):
    """Execute the full Part 2 comparison and return its report.

    Trains the three learned candidates on the training family,
    evaluates all four candidates plus the cash and classic
    references on every family, selects on validation rows only,
    and returns the training logs, the per-family reports, the
    selection dict and the selected name. ``families`` may supply
    a subset of scenario families (used by the tests); when
    omitted, the full declared families are built.
    """
    n_epochs = PART2_TRAINING["n_epochs"] if n_epochs is None \
        else int(n_epochs)
    if families is None:
        families = {
            "training": e5_scenarios("training", n_paths=n_paths,
                                     n_steps=n_steps),
            "validation": e5_scenarios("validation", n_paths=n_paths,
                                       n_steps=n_steps),
            "stress": e5_scenarios("stress", n_paths=n_paths,
                                   n_steps=n_steps),
            "gap": e5_scenarios("gap", n_paths=n_paths,
                                n_steps=n_steps),
            "episodes": episode_scenarios(n_paths=n_paths,
                                          n_steps=n_steps),
        }
    training_logs = {}
    fraction_fns = {}
    for name in CANDIDATE_ORDER:
        model, log = train_candidate(
            name, families["training"], n_epochs=n_epochs)
        training_logs[name] = log
        if model is not None:
            fraction_fns[name] = fraction_fn_for_model(model)
        else:
            fraction_fns[name] = candidate_fraction_fn(name)
    candidates = dict(fraction_fns)
    candidates["cash"] = reference_fraction_fn("cash")
    candidates["classic"] = reference_fraction_fn("classic")
    reports = {}
    for family, scenarios in families.items():
        if family == "training":
            continue
        reports[family] = evaluate_candidates(scenarios, candidates)
    validation = reports["validation"]
    rows = [r for r in validation["candidates"]
            if r["candidate"] in CANDIDATE_ORDER]
    classic = next(r for r in validation["candidates"]
                   if r["candidate"] == "classic")
    selection = hierarchical_select(rows, classic)
    return {
        "training": dict(PART2_TRAINING),
        "training_logs": training_logs,
        "families": reports,
        "selection": selection,
        "selected": selection["selected"],
        "episode_protocol": dict(EPISODE_PROTOCOL),
    }
