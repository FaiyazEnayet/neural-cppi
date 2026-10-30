"""Fixtures for E6 (regime robustness with real evaluation).

E6 predeclares every training-regime x evaluation-regime cell and
freezes its market start/end/hash before any fitting, trains the
same model under each training regime with matched optimizer
updates and equal per-path weighting, checks grouped episode
leakage fail-closed, and reports per-cell outcomes with the NC6
dependent-data machinery. Evaluation regimes are previously
viewed judges (registry flags), never new sealed evidence, and
no old T/H/I episode is turned into sealed evidence here.

Fixtures use synthetic price series with calendar dates so the
suite stays fast; the full run on the committed SPY series is
exercised by the run script, not by these tests, and its numbers
are reported in the CHANGELOG entry.
"""

import numpy as np
import pytest
import torch

from dhl.eval.regime_robustness import (
    E6_PATH_LEN,
    EVAL_REGIMES,
    TRAINING_REGIMES,
    balanced_weights,
    check_no_leakage,
    dro_should_activate,
    dro_update,
    era_labels_for_starts,
    freeze_cells,
    grouped_cv,
    run_e6,
    train_regime_model,
    verify_frozen,
    windows_for_slice,
)


def _synthetic_dates_prices(start="2016-01-01", n_days=1400, seed=3):
    rng = np.random.default_rng(seed)
    dates = []
    import datetime
    day = datetime.date.fromisoformat(start)
    for _ in range(n_days):
        dates.append(day.isoformat())
        day += datetime.timedelta(days=1)
    rets = rng.normal(0.0002, 0.01, size=n_days - 1)
    prices = np.empty(n_days)
    prices[0] = 100.0
    prices[1:] = 100.0 * np.exp(np.cumsum(rets))
    return dates, prices


def _small_regimes():
    training = (
        {"name": "small", "train_start": "2016-06-01",
         "train_end": "2017-06-01"},
        {"name": "large", "train_start": "2016-01-01",
         "train_end": "2018-01-01"},
    )
    evals = (
        {"name": "eval_a", "eval_start": "2018-06-01",
         "eval_end": "2018-12-01", "previously_viewed": True},
        {"name": "eval_b", "eval_start": "2018-12-01",
         "eval_end": None, "previously_viewed": True},
    )
    return training, evals


# --- freezing and leakage ------------------------------------------------------


def test_stage3_neural_cppi_6_freeze_disjoint_and_hash_stable():
    dates, prices = _synthetic_dates_prices()
    training, evals = _small_regimes()
    first = freeze_cells(dates, prices, training, evals,
                         path_len=21)
    second = freeze_cells(dates, prices, training, evals,
                          path_len=21)
    assert len(first) == 4
    for cell in first:
        check_no_leakage(cell)
        assert verify_frozen(cell, dates, prices) is True
        assert cell["train_end_idx"] < cell["eval_start_idx"]
        assert cell["n_train_windows"] >= 1
        assert cell["n_eval_windows"] >= 1
    assert [c["cell_hash"] for c in first] == \
        [c["cell_hash"] for c in second]


def test_stage3_neural_cppi_6_freeze_tamper_changes_hash():
    dates, prices = _synthetic_dates_prices()
    training, evals = _small_regimes()
    cells = freeze_cells(dates, prices, training, evals,
                         path_len=21)
    tampered = prices.copy()
    tampered[400] *= 1.5
    assert verify_frozen(cells[0], dates, tampered) is False


def test_stage3_neural_cppi_6_overlap_refused():
    cell = {
        "train_start_idx": 0, "train_end_idx": 100,
        "eval_start_idx": 50, "eval_end_idx": 200,
    }
    with pytest.raises(ValueError):
        check_no_leakage(cell)
    cell2 = {
        "train_start_idx": 100, "train_end_idx": 200,
        "eval_start_idx": 100, "eval_end_idx": 300,
    }
    with pytest.raises(ValueError):
        check_no_leakage(cell2)


def test_stage3_neural_cppi_6_declared_regimes_cover_contract():
    names = [r["name"] for r in TRAINING_REGIMES]
    assert "small_2017_2018" in names
    assert "large_2007_2019" in names
    assert "calm_2017" in names
    eval_names = [r["name"] for r in EVAL_REGIMES]
    assert "validation_2021_2022" in eval_names
    assert "test_2023plus" in eval_names
    assert "covid_stress" in eval_names
    assert all(r["previously_viewed"] for r in EVAL_REGIMES)


def test_stage3_neural_cppi_6_windows_nonoverlapping():
    _dates, prices = _synthetic_dates_prices(n_days=100)
    paths, starts = windows_for_slice(prices, path_len=21)
    assert paths.shape == (4, 22)
    assert list(starts) == [0, 22, 44, 66]
    assert float(paths[:, 0].min()) == pytest.approx(100.0)


# --- weighting and DRO ----------------------------------------------------------


def test_stage3_neural_cppi_6_balanced_weights():
    labels = ["2017", "2017", "2018"]
    w = balanced_weights(labels)
    assert float(np.mean(w)) == pytest.approx(1.0)
    assert w[2] > w[0]
    assert w[0] == w[1]


def test_stage3_neural_cppi_6_dro_update_shifts_to_worse_group():
    groups = np.array(["a", "b"])
    weights = np.array([0.5, 0.5])
    losses = {"a": 1.0, "b": 5.0}
    new = dro_update(weights, groups, losses, eta=0.5)
    assert new[1] > new[0]
    assert float(new.sum()) == pytest.approx(1.0)


def test_stage3_neural_cppi_6_dro_activation_criterion():
    assert dro_should_activate({"2017": 0.1, "2018": 1.5}) is True
    assert dro_should_activate({"2017": 0.1, "2018": 0.5}) is False
    assert dro_should_activate({"2017": 0.1}) is False


def test_stage3_neural_cppi_6_era_labels():
    assert era_labels_for_starts(
        ["2017-03-01", "2018-01-15"]) == ["2017", "2018"]


# --- training and grouped CV ----------------------------------------------------


def test_stage3_neural_cppi_6_training_deterministic_and_bounded():
    dates, prices = _synthetic_dates_prices()
    paths, starts = windows_for_slice(prices[:400], path_len=21)
    labels = era_labels_for_starts([dates[s] for s in starts])
    model_a, hist_a = train_regime_model(paths, labels, n_epochs=2)
    model_b, hist_b = train_regime_model(paths, labels, n_epochs=2)
    assert len(hist_a) == 3
    assert hist_a[-1]["loss"] == pytest.approx(hist_b[-1]["loss"])
    assert all(0.0 <= h["mean_fraction"] <= 1.0 for h in hist_a)
    assert model_a is not None


def test_stage3_neural_cppi_6_grouped_cv_partitions_eras():
    dates, prices = _synthetic_dates_prices()
    paths, starts = windows_for_slice(prices[:500], path_len=21)
    labels = era_labels_for_starts([dates[s] for s in starts])
    folds = grouped_cv(paths, labels, n_epochs=1)
    eras = sorted(set(labels))
    assert [f["held_era"] for f in folds] == eras
    assert sum(f["n_eval"] for f in folds) == len(paths)
    for fold in folds:
        assert fold["n_train"] == len(paths) - fold["n_eval"]


# --- full run on synthetic data --------------------------------------------------


def test_stage3_neural_cppi_6_run_reports_every_cell():
    dates, prices = _synthetic_dates_prices()
    training, evals = _small_regimes()
    result = run_e6(dates, prices, training_regimes=training,
                    eval_regimes=evals, n_epochs=1, cv_epochs=1,
                    path_len=21)
    assert len(result["cells"]) == 4
    for cell in result["cells"]:
        assert cell["frozen"]["cell_hash"]
        assert cell["n_eval_paths"] >= 1
        assert "paired_model_vs_classic" in cell
        assert "mean_wealth_model" in cell
        assert cell["previously_viewed"] is True
    assert len(result["regime_models"]) == 2
    assert "dro_activated" in result
    assert result["dro_criterion_gap"] >= 0.0
    for regime, folds in result["grouped_cv"].items():
        assert folds
