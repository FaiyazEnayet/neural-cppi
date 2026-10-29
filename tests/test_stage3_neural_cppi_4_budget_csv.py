"""Stage 3 neural-cppi E4: per-episode CSV output, exact round-trip.

The CSV is the precise per-episode record the contract asks for: one
row per rebalance step, full-precision floats, no dashboard UI. The
round-trip test requires bit-exact float equality after a write and a
read, not approximate agreement, because a diagnostic that silently
rounds its own record cannot be used for attribution.
"""

import csv
import os

import pytest

from dhl.eval.budget_diagnosis import (
    CSV_COLUMNS,
    replay_episode,
    rows_from_csv,
    rows_to_csv,
    summarize_diagnosis,
)


def _episode():
    rows, terminal = replay_episode(
        prices=[100.0, 94.0, 97.0, 94.5, 98.0],
        proposals=[0.5, 0.25, 0.75, 0.0],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.002,
        v0=100.0,
        r=0.02,
    )
    return rows, terminal


def test_stage3_neural_cppi_4_csv_round_trip_is_exact(tmp_path):
    rows, _ = _episode()
    path = os.path.join(str(tmp_path), "episode.csv")
    rows_to_csv(rows, path)
    back = rows_from_csv(path)
    assert len(back) == len(rows)
    for original, restored in zip(rows, back):
        assert set(restored.keys()) == set(original.keys())
        for key, value in original.items():
            if isinstance(value, str):
                assert restored[key] == value
            else:
                assert restored[key] == value, key


def test_stage3_neural_cppi_4_csv_header_and_columns(tmp_path):
    rows, _ = _episode()
    path = os.path.join(str(tmp_path), "episode.csv")
    rows_to_csv(rows, path)
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        data_rows = list(reader)
    assert header == list(CSV_COLUMNS)
    assert len(data_rows) == len(rows)
    # Every contract field is present as a column, in a fixed order.
    for name in ("proposal_weight", "allowable_budget", "executed_dollars",
                 "clipping_weight", "cushion_ratio", "vol", "drawdown",
                 "cum_fees", "state"):
        assert name in header


def test_stage3_neural_cppi_4_csv_rejects_mismatched_columns(tmp_path):
    path = os.path.join(str(tmp_path), "bad.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "price"])
        writer.writerow([0, 100.0])
    with pytest.raises(ValueError):
        rows_from_csv(path)


def test_stage3_neural_cppi_4_summary_survives_csv_round_trip(tmp_path):
    rows, terminal = _episode()
    before = summarize_diagnosis(rows, terminal)
    path = os.path.join(str(tmp_path), "episode.csv")
    rows_to_csv(rows, path)
    after = summarize_diagnosis(rows_from_csv(path), terminal)
    assert before == after
