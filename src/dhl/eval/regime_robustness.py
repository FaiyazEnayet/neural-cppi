"""E6: regime robustness with real evaluation.

Every training-regime x evaluation-regime cell is predeclared
below and frozen before any fitting: the frozen record carries
the training and evaluation market start/end dates, the bar
indices, the SHA-256 hashes of the two market slices, the
nonoverlapping window counts and the measured realized volatility
of each slice. Freezing happens in ``freeze_cells``; ``verify_frozen``
recomputes the record from the dates and prices and refuses a
mismatch, and ``check_no_leakage`` refuses any cell whose
evaluation bars touch or precede the end of its training bars.
Evaluation uses the cold feature protocol (features computed
causally within each window, no pre-window history), so
bar-disjointness plus chronological order is the leakage test:
no raw return or date is shared between training and a declared
untouched evaluation.

Training regimes (all legal under the chronological registry,
which admits pre-2020 training):

- ``small_2017_2018``: two chronological years.
- ``large_2007_2019``: the full pre-2020 span, crash-inclusive
  (it contains 2008, 2011, 2015-2016 and 2018).
- ``calm_2017``: the single calendar year 2017, predeclared by
  date, with its realized volatility reported in the frozen
  table rather than used to move the window.

Evaluation regimes are the registry's previously viewed judges
(validation 2021-2022, test 2023+, COVID stress Feb-Apr 2020).
They are held out from every training regime, but they are NOT
new sealed evidence: their outcomes were reported in earlier
results, so every report carries the previously-viewed label
and no freshness is claimed. No old T/H/I episode is used here
at all, so nothing is turned into sealed evidence.

The model is the same for every regime: the R1/E5 budget
fraction network (direct form, full causal feature set, hidden
(16, 16), matched initialization), trained through the shared
rollout with the corrected multi-horizon path loss (floor
penalty 30 on terminal shortfall, anytime penalty 30 on
anytime shortfall, turnover penalty 0). Matched optimizer
updates means exactly the same epoch count for every regime;
comparable sample weighting means the loss is a mean over
paths, so every training path carries equal weight whatever
the regime size. Windows are strictly nonoverlapping
(``dhl.eval.chrono.nonoverlapping_windows``), one bar dropped
between windows by design, so paths share no price bar.

Leave-one-era-out grouped cross-validation is the development
procedure: for each calendar year in a training regime, a model
trains on the other years and is scored on the held-out year.
Group-DRO is implemented (``dro_update`` and the ``dro``
weighting in ``train_regime_model``) but is NOT activated
unless the predeclared criterion fires: after equal-weight
training, if the worst training era's mean terminal shortfall
exceeds the best era's by more than 1.0 currency unit, the
regime with the largest gap is retrained with group-DRO and
reported alongside the simple run. Otherwise the report records
the measured gap and that DRO was not activated, because
simple equal weighting did not fail.
"""

import hashlib
import json

import numpy as np
import torch

from dhl.eval.chrono import (
    calendar_block_ids,
    frozen_config_sha256,
    nonoverlapping_windows,
    paired_dependent_summary,
)
from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    FLOOR,
    FLOOR_PENALTY,
    rollout,
)
from dhl.experiments.losses import multihorizon_loss
from dhl.policies.budget_allocator import (
    BudgetFractionNet,
    budget_weight_fn,
)
from dhl.policies.candidate_comparison import (
    E5_FEATURE_NAMES,
    MULTIPLIER,
    evaluate_candidates,
    reference_fraction_fn,
)
from dhl.policies.candidate_training import fraction_fn_for_model

#: Predeclared training regimes (see the module docstring).
TRAINING_REGIMES = (
    {"name": "small_2017_2018", "train_start": "2017-01-01",
     "train_end": "2019-01-01"},
    {"name": "large_2007_2019", "train_start": "2007-01-01",
     "train_end": "2020-01-01"},
    {"name": "calm_2017", "train_start": "2017-01-01",
     "train_end": "2018-01-01"},
)

#: Predeclared evaluation regimes. All are previously viewed.
EVAL_REGIMES = (
    {"name": "covid_stress", "eval_start": "2020-02-19",
     "eval_end": "2020-05-01", "previously_viewed": True},
    {"name": "validation_2021_2022", "eval_start": "2021-01-01",
     "eval_end": "2023-01-01", "previously_viewed": True},
    {"name": "test_2023plus", "eval_start": "2023-01-01",
     "eval_end": None, "previously_viewed": True},
)

#: E6 experiment choices, predeclared here.
E6_PATH_LEN = 21
E6_TRAINING = {
    "n_epochs": 30,
    "cv_epochs": 10,
    "lr": 3e-3,
    "seed": 7,
    "floor_penalty": float(FLOOR_PENALTY),
    "anytime_penalty": float(FLOOR_PENALTY),
    "turnover_penalty": 0.0,
    "multiplier": MULTIPLIER,
    "floor": FLOOR,
    "cap": CAP,
    "cost_rate": COST_RATE,
    "path_len": E6_PATH_LEN,
    "n_boot": 1000,
}
DRO_GAP_THRESHOLD = 1.0
DRO_ETA = 0.1


# --- freezing -------------------------------------------------------------------


def _slice_indices(dates, start, end):
    """[start_idx, end_idx) for dates in [start, end)."""
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


def _slice_hash(prices_slice):
    p = np.asarray(prices_slice, dtype=float)
    rets = np.log(p[1:] / p[:-1])
    return hashlib.sha256(rets.tobytes()).hexdigest()


def _realized_vol(prices_slice):
    p = np.asarray(prices_slice, dtype=float)
    rets = np.log(p[1:] / p[:-1])
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets, ddof=1) * np.sqrt(252.0))


def _n_windows(n_prices, path_len):
    if n_prices < path_len + 1:
        return 0
    return len(np.arange(0, n_prices - path_len, path_len + 1))


def freeze_cells(dates, prices, training_regimes, eval_regimes,
                 path_len=E6_PATH_LEN):
    """Freeze every training x evaluation cell before fitting."""
    dates = list(dates)
    prices = np.asarray(prices, dtype=float)
    if len(dates) != len(prices):
        raise ValueError("dates and prices must align")
    cells = []
    for train in training_regimes:
        ts, te = _slice_indices(dates, train["train_start"],
                                train["train_end"])
        train_slice = prices[ts:te]
        n_train = _n_windows(len(train_slice), path_len)
        if n_train == 0:
            raise ValueError(
                f"training regime {train['name']} has no full "
                f"{path_len}-day window")
        train_hash = _slice_hash(train_slice)
        for ev in eval_regimes:
            es, ee = _slice_indices(dates, ev["eval_start"],
                                    ev["eval_end"])
            eval_slice = prices[es:ee]
            n_eval = _n_windows(len(eval_slice), path_len)
            if n_eval == 0:
                raise ValueError(
                    f"evaluation regime {ev['name']} has no full "
                    f"{path_len}-day window")
            eval_hash = _slice_hash(eval_slice)
            cell = {
                "training_regime": train["name"],
                "train_start": train["train_start"],
                "train_end": train["train_end"],
                "eval_regime": ev["name"],
                "eval_start": ev["eval_start"],
                "eval_end": ev["eval_end"],
                "previously_viewed": bool(ev["previously_viewed"]),
                "train_start_idx": ts, "train_end_idx": te,
                "eval_start_idx": es, "eval_end_idx": ee,
                "n_train_windows": n_train,
                "n_eval_windows": n_eval,
                "train_hash": train_hash,
                "eval_hash": eval_hash,
                "train_realized_vol": _realized_vol(train_slice),
                "eval_realized_vol": _realized_vol(eval_slice),
                "path_len": path_len,
            }
            check_no_leakage(cell)
            cell["cell_hash"] = hashlib.sha256(
                json.dumps(
                    {k: cell[k] for k in (
                        "training_regime", "train_start", "train_end",
                        "eval_regime", "eval_start", "eval_end",
                        "train_hash", "eval_hash", "path_len")},
                    sort_keys=True).encode()).hexdigest()
            cells.append(cell)
    return cells


def check_no_leakage(cell):
    """Fail closed on any training/evaluation bar overlap."""
    ts, te = cell["train_start_idx"], cell["train_end_idx"]
    es, ee = cell["eval_start_idx"], cell["eval_end_idx"]
    if es < te:
        raise ValueError(
            "evaluation starts before training ends: leakage "
            f"(train [{ts}, {te}), eval [{es}, {ee}))")
    if not (ee <= ts or es >= te):
        raise ValueError("training and evaluation bars overlap")
    return True


def verify_frozen(cell, dates, prices):
    """Recompute one frozen cell and compare hash and indices."""
    training = ({"name": cell["training_regime"],
                 "train_start": cell["train_start"],
                 "train_end": cell["train_end"]},)
    evals = ({"name": cell["eval_regime"],
              "eval_start": cell["eval_start"],
              "eval_end": cell["eval_end"],
              "previously_viewed": cell["previously_viewed"]},)
    recomputed = freeze_cells(dates, prices, training, evals,
                              path_len=cell["path_len"])[0]
    return (recomputed["cell_hash"] == cell["cell_hash"]
            and recomputed["train_start_idx"]
            == cell["train_start_idx"]
            and recomputed["eval_start_idx"]
            == cell["eval_start_idx"])


# --- windows and weights ---------------------------------------------------------


def windows_for_slice(prices_slice, path_len=E6_PATH_LEN):
    """Nonoverlapping renormalized windows and their start indices."""
    paths, starts = nonoverlapping_windows(
        np.asarray(prices_slice, dtype=float), path_len, s0=100.0)
    return paths, [int(s) for s in starts]


def era_labels_for_starts(start_dates):
    """Calendar year of each window start date."""
    return [d[:4] for d in start_dates]


def balanced_weights(era_labels):
    """Per-path weights inversely proportional to era size.

    Weights have mean 1, so the weighted loss stays on the same
    scale as the unweighted mean; the smallest era's paths carry
    the largest weight.
    """
    labels = np.asarray(era_labels)
    out = np.empty(len(labels), dtype=float)
    for era in sorted(set(labels)):
        mask = labels == era
        out[mask] = 1.0 / float(mask.sum())
    out = out * (len(labels) / float(out.sum()))
    return out


def dro_update(group_weights, group_names, group_losses, eta=DRO_ETA):
    """Exponentiated-gradient group-weight update (group DRO step).

    Groups with higher loss gain weight; the result renormalizes
    to sum 1. ``group_weights`` and ``group_losses`` are aligned
    with ``group_names`` (losses as floats).
    """
    w = np.asarray(group_weights, dtype=float)
    losses = np.array([float(group_losses[g]) for g in group_names])
    new = w * np.exp(eta * (losses - losses.mean()))
    return new / new.sum()


def dro_should_activate(per_era_shortfall,
                        threshold=DRO_GAP_THRESHOLD):
    """The predeclared DRO activation criterion.

    Activates only when at least two eras are present and the
    worst era's mean terminal shortfall exceeds the best era's
    by more than ``threshold`` currency units.
    """
    values = list(per_era_shortfall.values())
    if len(values) < 2:
        return False
    return (max(values) - min(values)) > threshold


# --- training ---------------------------------------------------------------------


def _rollout_wealths(model, paths_tensor):
    fn = fraction_fn_for_model(model)
    with torch.no_grad():
        V_T, path, ledger = rollout(
            paths_tensor,
            budget_weight_fn(fn, multiplier=MULTIPLIER,
                             floor=FLOOR, cap=CAP),
            floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
            return_path=True, return_ledger=True)
    return V_T, path, ledger


def train_regime_model(paths, era_labels, n_epochs=None, lr=None,
                       seed=None, weighting="equal", dro_eta=DRO_ETA):
    """Train the shared model on one regime's windows.

    ``weighting`` is "equal" (every path equal), "era_balanced"
    (``balanced_weights``) or "dro" (group weights updated each
    epoch by ``dro_update``). Returns ``(model, history)`` where
    history entries carry epoch, loss, mean wealth, mean
    shortfall, mean fraction and the group weights in force
    (None unless DRO).
    """
    n_epochs = E6_TRAINING["n_epochs"] if n_epochs is None \
        else int(n_epochs)
    lr = E6_TRAINING["lr"] if lr is None else float(lr)
    seed = E6_TRAINING["seed"] if seed is None else int(seed)
    if weighting not in ("equal", "era_balanced", "dro"):
        raise ValueError(f"unknown weighting {weighting!r}")
    paths_np = np.asarray(paths, dtype=float)
    if len(paths_np) == 0:
        raise ValueError("training needs at least one window")
    labels = list(era_labels)
    if len(labels) != len(paths_np):
        raise ValueError("era labels must align with windows")
    groups = sorted(set(labels))
    group_index = {g: i for i, g in enumerate(groups)}
    path_group = np.array([group_index[g] for g in labels])
    counts = np.array([float((path_group == i).sum())
                       for i in range(len(groups))])
    torch.manual_seed(seed)
    model = BudgetFractionNet(hidden_dims=(16, 16), form="direct",
                              feature_names=E5_FEATURE_NAMES)
    base_fn = fraction_fn_for_model(model)
    S = torch.as_tensor(paths_np, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if weighting == "era_balanced":
        path_w = torch.as_tensor(balanced_weights(labels),
                                 dtype=torch.float64)
    else:
        path_w = torch.ones(len(paths_np), dtype=torch.float64)
    group_w = np.full(len(groups), 1.0 / len(groups))

    def evaluate():
        sums, counts_u = [], []

        def tracked(i, V, tau_frac, cushion, feats):
            u = base_fn(i, V, tau_frac, cushion, feats)
            sums.append(u.detach().sum())
            counts_u.append(int(u.numel()))
            return u

        with torch.no_grad():
            V_T, path, ledger = rollout(
                S, budget_weight_fn(tracked, multiplier=MULTIPLIER,
                                    floor=FLOOR, cap=CAP),
                floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
                return_path=True, return_ledger=True)
        shortfall = torch.clamp(FLOOR - V_T, min=0.0)
        return {
            "mean_wealth": float(V_T.mean()),
            "mean_shortfall": float(shortfall.mean()),
            "mean_fraction": float(sum(float(s) for s in sums)
                                   / float(sum(counts_u))),
        }

    def forward_with_grad():
        sums, counts_u = [], []

        def tracked(i, V, tau_frac, cushion, feats):
            u = base_fn(i, V, tau_frac, cushion, feats)
            sums.append(u.detach().sum())
            counts_u.append(int(u.numel()))
            return u

        V_T, path, ledger = rollout(
            S, budget_weight_fn(tracked, multiplier=MULTIPLIER,
                                floor=FLOOR, cap=CAP),
            floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
            return_path=True, return_ledger=True)
        path_min = path.min(dim=1).values
        weight_path = ledger["weight"]
        turnover = torch.zeros_like(V_T)
        if weight_path.shape[1] > 1:
            turnover = weight_path.diff(dim=1).abs().mean(dim=1)
        per_path = (
            -V_T
            + FLOOR_PENALTY * torch.clamp(FLOOR - V_T, min=0.0)
            + FLOOR_PENALTY * torch.clamp(FLOOR - path_min, min=0.0)
            + 0.0 * turnover
        )
        mean_fraction = float(
            sum(float(s) for s in sums) / float(sum(counts_u)))
        return per_path, mean_fraction, V_T

    history = []
    metrics = evaluate()
    history.append({"epoch": 0, "loss": None,
                    "mean_wealth": metrics["mean_wealth"],
                    "mean_shortfall": metrics["mean_shortfall"],
                    "mean_fraction": metrics["mean_fraction"],
                    "group_weights": None})
    # Epoch 0 loss from a no-grad forward is filled by the first
    # training epoch's pre-step value; instead record the exact
    # initial loss with a probe forward.
    with torch.no_grad():
        V_T0, path0, ledger0 = _rollout_wealths(model, S)
        _t0, comp0 = multihorizon_loss(
            V_T0, path0, weight_path=ledger0["weight"],
            floor=FLOOR, floor_penalty=FLOOR_PENALTY,
            anytime_penalty=FLOOR_PENALTY, turnover_penalty=0.0)
        history[0]["loss"] = float(_t0.detach())
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        per_path, mean_fraction, _V = forward_with_grad()
        if weighting == "dro":
            group_losses = {}
            for i, g in enumerate(groups):
                mask = path_group == i
                group_losses[g] = float(
                    per_path.detach()[mask].mean())
            group_w = dro_update(group_w, groups, group_losses,
                                 eta=dro_eta)
            w_tensor = torch.as_tensor(
                np.array([group_w[path_group[k]] / counts[path_group[k]]
                          for k in range(len(paths_np))]),
                dtype=torch.float64)
            total = (w_tensor * per_path).sum()
        elif weighting == "era_balanced":
            total = (path_w * per_path).sum() / path_w.sum()
        else:
            total = per_path.mean()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics = evaluate()
        history.append({
            "epoch": epoch, "loss": float(total.detach()),
            "mean_wealth": metrics["mean_wealth"],
            "mean_shortfall": metrics["mean_shortfall"],
            "mean_fraction": metrics["mean_fraction"],
            "group_weights": (dict(zip(groups, group_w.tolist()))
                              if weighting == "dro" else None),
        })
    return model, history


def evaluate_windows(model, eval_paths, eval_start_dates,
                     frozen_config=None, n_boot=1000):
    """Score a trained model on evaluation windows.

    Returns the full acceptance rows (model, classic, cash via
    the shared harness), the per-path wealths summarized by the
    NC6 paired dependent summary against classic CPPI on the
    same paths (block bootstrap over calendar months, exact
    Clopper-Pearson breach bounds, previously-viewed label),
    and the frozen-config hash tying the evaluation to its
    predeclared hyperparameters.
    """
    S = torch.as_tensor(np.asarray(eval_paths, dtype=float),
                        dtype=torch.float64)
    scenarios = [(0, S)]
    candidates = {
        "model": fraction_fn_for_model(model),
        "classic": reference_fraction_fn("classic"),
        "cash": reference_fraction_fn("cash"),
    }
    report = evaluate_candidates(scenarios, candidates)
    rows = {r["candidate"]: r for r in report["candidates"]}
    V_model, _p, _l = _rollout_wealths(model, S)
    V_classic, _p2, _l2 = _rollout_wealths_classic(S)
    block_ids = calendar_block_ids(list(eval_start_dates), freq="M")
    paired = paired_dependent_summary(
        V_model.detach().numpy(), V_classic.detach().numpy(),
        block_ids, FLOOR, frozen_config=frozen_config,
        n_boot=n_boot, previously_viewed=True)
    return {
        "rows": rows,
        "paired_model_vs_classic": paired,
        "n_eval_paths": int(len(eval_paths)),
        "mean_wealth_model": float(V_model.mean()),
        "mean_wealth_classic": float(V_classic.mean()),
        "frozen_config_sha256": frozen_config_sha256(frozen_config)
        if frozen_config is not None else None,
    }


def _rollout_wealths_classic(S):
    fn = reference_fraction_fn("classic")
    with torch.no_grad():
        V_T, path, ledger = rollout(
            S, budget_weight_fn(fn, multiplier=MULTIPLIER,
                                floor=FLOOR, cap=CAP),
            floor=FLOOR, cost_rate=COST_RATE, cap=CAP,
            return_path=True, return_ledger=True)
    return V_T, path, ledger


def grouped_cv(paths, era_labels, n_epochs=None, lr=None, seed=None):
    """Leave-one-era-out grouped CV: one fold per calendar year.

    A single-era regime has no leave-one-out fold (holding out
    its only era would leave nothing to train on), so it returns
    an empty fold list, recorded as such, rather than training
    on nothing.
    """
    n_epochs = E6_TRAINING["cv_epochs"] if n_epochs is None \
        else int(n_epochs)
    paths_np = np.asarray(paths, dtype=float)
    labels = list(era_labels)
    if len(set(labels)) < 2:
        return []
    folds = []
    for era in sorted(set(labels)):
        test_idx = [i for i, g in enumerate(labels) if g == era]
        train_idx = [i for i, g in enumerate(labels) if g != era]
        model, _h = train_regime_model(
            paths_np[train_idx], [labels[i] for i in train_idx],
            n_epochs=n_epochs, lr=lr, seed=seed)
        S_test = torch.as_tensor(paths_np[test_idx],
                                 dtype=torch.float64)
        V_T, _p, _l = _rollout_wealths(model, S_test)
        shortfall = torch.clamp(FLOOR - V_T, min=0.0)
        folds.append({
            "held_era": era,
            "n_train": len(train_idx),
            "n_eval": len(test_idx),
            "mean_wealth": float(V_T.mean()),
            "mean_shortfall": float(shortfall.mean()),
            "breach_count_terminal": int((V_T < FLOOR).sum()),
        })
    return folds


def _per_era_shortfall(model, paths, era_labels):
    S = torch.as_tensor(np.asarray(paths, dtype=float),
                        dtype=torch.float64)
    V_T, _p, _l = _rollout_wealths(model, S)
    shortfall = torch.clamp(FLOOR - V_T, min=0.0).numpy()
    out = {}
    for era in sorted(set(era_labels)):
        mask = np.array([g == era for g in era_labels])
        out[era] = float(shortfall[mask].mean())
    return out


def run_e6(dates, prices, training_regimes=None,
           eval_regimes=None, n_epochs=None, cv_epochs=None,
           path_len=E6_PATH_LEN, n_boot=None, seed=None):
    """Execute the full E6 experiment and return its report."""
    training_regimes = TRAINING_REGIMES if training_regimes is None \
        else tuple(training_regimes)
    eval_regimes = EVAL_REGIMES if eval_regimes is None \
        else tuple(eval_regimes)
    n_epochs = E6_TRAINING["n_epochs"] if n_epochs is None \
        else int(n_epochs)
    cv_epochs = E6_TRAINING["cv_epochs"] if cv_epochs is None \
        else int(cv_epochs)
    n_boot = E6_TRAINING["n_boot"] if n_boot is None else int(n_boot)
    seed = E6_TRAINING["seed"] if seed is None else int(seed)
    dates = list(dates)
    prices = np.asarray(prices, dtype=float)
    frozen = freeze_cells(dates, prices, training_regimes,
                          eval_regimes, path_len=path_len)
    regime_models = {}
    regime_data = {}
    grouped = {}
    for regime in training_regimes:
        cell = next(c for c in frozen
                    if c["training_regime"] == regime["name"])
        ts, te = cell["train_start_idx"], cell["train_end_idx"]
        slice_dates = dates[ts:te]
        slice_prices = prices[ts:te]
        paths, starts = windows_for_slice(slice_prices,
                                          path_len=path_len)
        labels = era_labels_for_starts(
            [slice_dates[s] for s in starts])
        model, history = train_regime_model(
            paths, labels, n_epochs=n_epochs, seed=seed)
        per_era = _per_era_shortfall(model, paths, labels)
        regime_models[regime["name"]] = {
            "history": history,
            "per_era_shortfall": per_era,
            "gap": (max(per_era.values()) - min(per_era.values())
                    if len(per_era) >= 2 else 0.0),
        }
        regime_data[regime["name"]] = (model, paths, labels)
        grouped[regime["name"]] = grouped_cv(
            paths, labels, n_epochs=cv_epochs, seed=seed)
    dro_gap = max((v["gap"] for v in regime_models.values()),
                  default=0.0)
    dro_activated = any(
        dro_should_activate(v["per_era_shortfall"])
        for v in regime_models.values())
    dro_cells = []
    dro_model_name = None
    if dro_activated:
        dro_model_name = max(regime_models,
                             key=lambda k: regime_models[k]["gap"])
        model, paths, labels = regime_data[dro_model_name]
        dro_model, _h = train_regime_model(
            paths, labels, n_epochs=n_epochs, seed=seed,
            weighting="dro")
        regime_data[dro_model_name + "_dro"] = (dro_model, paths,
                                                labels)
        for cell in frozen:
            if cell["training_regime"] != dro_model_name:
                continue
            es, ee = cell["eval_start_idx"], cell["eval_end_idx"]
            slice_dates = dates[es:ee]
            slice_prices = prices[es:ee]
            paths_e, starts_e = windows_for_slice(
                slice_prices, path_len=path_len)
            start_dates = [slice_dates[s] for s in starts_e]
            config = {
                "training_regime": dro_model_name + "_dro",
                "eval_regime": cell["eval_regime"],
                "cell_hash": cell["cell_hash"],
                "path_len": path_len, "n_epochs": n_epochs,
                "seed": seed, "weighting": "dro",
            }
            ev = evaluate_windows(dro_model, paths_e, start_dates,
                                  frozen_config=config,
                                  n_boot=n_boot)
            dro_cells.append({"frozen": cell, **ev,
                              "previously_viewed": True})
    cells = []
    for cell in frozen:
        model, _p, _l = regime_data[cell["training_regime"]]
        es, ee = cell["eval_start_idx"], cell["eval_end_idx"]
        slice_dates = dates[es:ee]
        slice_prices = prices[es:ee]
        paths_e, starts_e = windows_for_slice(
            slice_prices, path_len=path_len)
        start_dates = [slice_dates[s] for s in starts_e]
        config = {
            "training_regime": cell["training_regime"],
            "eval_regime": cell["eval_regime"],
            "cell_hash": cell["cell_hash"],
            "path_len": path_len, "n_epochs": n_epochs,
            "seed": seed, "weighting": "equal",
        }
        ev = evaluate_windows(model, paths_e, start_dates,
                              frozen_config=config, n_boot=n_boot)
        cells.append({"frozen": cell, **ev,
                      "previously_viewed":
                          cell["previously_viewed"]})
    return {
        "training": dict(E6_TRAINING),
        "frozen": frozen,
        "cells": cells,
        "regime_models": regime_models,
        "grouped_cv": grouped,
        "dro_activated": dro_activated,
        "dro_criterion_gap": dro_gap,
        "dro_model": dro_model_name,
        "dro_cells": dro_cells,
    }
