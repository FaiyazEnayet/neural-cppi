"""Stage 3 P1-A: honest chronological and embargoed evaluation.

Fast by design: synthetic series and tiny real-data slices only, plus
cheap policy-construction checks. No full training run lives here.

Independent oracles: date counts are recomputed in-test with plain
Python string comparisons, never through the function under test.
"""

import datetime

import numpy as np
import pytest
import torch

from dhl.eval.chrono import split_returns_by_date
from dhl.eval.registry import (
    Episode,
    EpisodeRegistry,
    predeclare_episode,
    standard_chrono_registry,
    summarize_window_count,
    summarize_windows,
)
from dhl.experiments.cppi_learned import (
    CPPIPolicy,
    policy_init_hash,
    seeded_policy,
)
from dhl.experiments.crash_ablation import (
    seed_schedule,
    summarize_across_seeds,
    variant_training_plan,
)


def _toy_series(n=900, start="2018-01-01"):
    base = datetime.date.fromisoformat(start)
    dates = [(base + datetime.timedelta(days=i)).isoformat()
             for i in range(n)]
    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    rets = np.log(prices[1:] / prices[:-1])
    return dates, prices, rets


def _independent_counts(end_dates, lo, hi):
    """Plain-Python count oracle: end dates in [lo, hi)."""
    return sum(1 for d in end_dates
               if (lo is None or d >= lo) and (hi is None or d < hi))


# ---------------- validation labels agree with date counts ----------------

def test_val_bucket_is_exactly_2021_2022():
    dates, _, rets = _toy_series(2000, "2018-01-01")
    splits, sdates = split_returns_by_date(
        dates, rets, "2020-01-01", "2023-01-01", val_start="2021-01-01")
    end_dates = dates[1:]
    assert len(splits["val"]) == _independent_counts(
        end_dates, "2021-01-01", "2023-01-01")
    assert min(sdates["val"]) >= "2021-01-01"
    assert max(sdates["val"]) < "2023-01-01"
    # No 2020 bar may sit in the validation bucket: the old mask
    # ([TRAIN_END, VAL_END)) admitted the whole COVID year.
    assert not any(d < "2021-01-01" for d in sdates["val"])
    assert len(sdates["val"]) == len(splits["val"])


def test_split_buckets_disjoint_at_bar_granularity():
    dates, _, rets = _toy_series(2000, "2018-01-01")
    splits, sdates = split_returns_by_date(
        dates, rets, "2020-01-01", "2023-01-01", val_start="2021-01-01")
    seen = set()
    for k in ("train", "val", "test"):
        assert not (set(sdates[k]) & seen), f"{k} overlaps another bucket"
        seen |= set(sdates[k])
    # The reserved 2020 gap belongs to no bucket by design.
    reserved = [d for d in dates[1:]
                if "2020-01-01" <= d < "2021-01-01"]
    assert not (set(reserved) & seen)
    assert len(reserved) == _independent_counts(
        dates[1:], "2020-01-01", "2021-01-01")


def test_split_default_val_start_preserves_old_behavior():
    dates, _, rets = _toy_series(800, "2018-01-01")
    splits, sdates = split_returns_by_date(
        dates, rets, "2019-01-01", "2019-07-01")
    assert max(sdates["train"]) < "2019-01-01"
    assert min(sdates["val"]) >= "2019-01-01"
    assert max(sdates["val"]) < "2019-07-01"
    assert min(sdates["test"]) >= "2019-07-01"
    assert sum(len(v) for v in splits.values()) == len(rets)


# ---------------- registry: disjointness and embargo ----------------

def test_registry_rejects_full_span_training_pool():
    # The original bootstrap baseline trained on the whole series, so
    # its pool overlaps every window later called held-out.
    dates, _, _ = _toy_series(1500, "2018-01-01")
    reg = standard_chrono_registry()
    with pytest.raises(ValueError, match="held-out"):
        reg.validate_training_period(dates, "2018-01-01", "2022-01-01")


def test_registry_accepts_pre2020_training_pool():
    dates, _, _ = _toy_series(1500, "2018-01-01")
    reg = standard_chrono_registry()
    reg.validate_training_period(dates, "2018-01-01", "2020-01-01",
                                 lookback_bars=60)


def test_registry_blocks_lookback_crossing_into_held_out():
    dates, _, _ = _toy_series(2200, "2018-01-01")
    reg = standard_chrono_registry()
    # A training block starting 10 bars after the validation episode
    # ends, with a 60-bar feature lookback, consumes held-out bars.
    val_start = dates.index("2021-01-01")
    val_end = dates.index("2023-01-01")
    block_start = val_end + 10
    with pytest.raises(ValueError, match="lookback"):
        reg.validate_blocks(dates, [(block_start, block_start + 63)],
                            lookback_bars=60)
    # A block fully inside the training period passes with or without
    # lookback: its extended span reaches only training bars.
    train_start = dates.index("2018-06-01")
    reg.validate_blocks(dates, [(train_start, train_start + 63)],
                        lookback_bars=60)
    reg.validate_blocks(dates, [(train_start, train_start + 63)],
                        lookback_bars=0)
    assert val_start < val_end  # episode endpoints are ordered


def test_registry_judge_windows_marked_previously_viewed():
    reg = standard_chrono_registry()
    by_name = {e.name: e for e in reg.episodes}
    for name in ("covid_stress", "validation_2021_2022", "test_2023plus"):
        assert by_name[name].previously_viewed is True
    assert by_name["train_pre2020"].previously_viewed is False


def test_predeclared_episode_records_geography_family_dates():
    ep = predeclare_episode(
        "future_test", "2027-01-01", "2027-07-01",
        geography="US", asset_family="single-stock",
        note="declared before inspecting outcomes")
    assert ep.predeclared is True
    assert ep.geography == "US"
    assert ep.asset_family == "single-stock"
    assert ep.start == "2027-01-01" and ep.end == "2027-07-01"
    assert ep.previously_viewed is False


def test_registry_rejects_inverted_episode():
    with pytest.raises(ValueError):
        Episode(name="bad", start="2023-01-01", end="2021-01-01",
                role="test", independence_group="bad")
    with pytest.raises(ValueError):
        Episode(name="bad", start="2023-01-01", end="2023-01-01",
                role="test", independence_group="bad")


# ---------------- window counts: unique vs overlapping ----------------

def test_window_counts_separate_unique_from_overlapping():
    starts = list(range(0, 500 - 63, 5))
    counts = summarize_windows(starts, 63)
    assert counts["n_windows"] == len(starts)
    assert counts["n_non_overlapping"] < counts["n_windows"]
    # Greedy non-overlapping count for stride 5, span 64 bars.
    expect = 0
    last_end = -1
    for s in starts:
        if s >= last_end:
            expect += 1
            last_end = s + 64
    assert counts["n_non_overlapping"] == expect
    assert counts["span_bars"] == starts[-1] + 64 - starts[0]


def test_window_count_from_stride_arithmetic():
    counts = summarize_window_count(n_windows=40, path_len=63, stride=5)
    assert counts["n_windows"] == 40
    assert counts["n_non_overlapping"] == 4  # starts 0, 65, 130, 195
    counts1 = summarize_window_count(n_windows=22, path_len=30, stride=1)
    assert counts1["n_windows"] == 22
    assert counts1["n_non_overlapping"] == 1  # 31-bar windows, stride 1


# ---------------- seed schedule: no seed confound ----------------

def test_seed_schedule_shared_across_variants():
    plan = variant_training_plan(base_seed=123, n_seeds=3)
    by_variant = {}
    for name, seed in plan:
        by_variant.setdefault(name, []).append(seed)
    assert len(by_variant) >= 7  # every ablation variant, incl. E sweep
    for name, seeds in by_variant.items():
        assert seeds == [123, 124, 125], name
    assert seed_schedule(123, 3) == (123, 124, 125)


def test_same_seed_same_dims_same_init_different_dims_differ():
    a = seeded_policy(CPPIPolicy, seed=7, feature_names=("vol",))
    b = seeded_policy(CPPIPolicy, seed=7, feature_names=("vol",))
    assert policy_init_hash(a) == policy_init_hash(b)
    c = seeded_policy(
        CPPIPolicy, seed=7,
        feature_names=("vol", "vol_ratio", "drawdown", "panic_z"))
    n_a = sum(p.numel() for p in a.parameters())
    n_c = sum(p.numel() for p in c.parameters())
    # Differing input dimensions change parameter shapes by
    # construction; the schedule must not pretend the tensors match.
    assert n_c > n_a
    assert policy_init_hash(a) != policy_init_hash(c)


def test_across_seed_summary_separates_seed_from_design():
    per_seed = {
        10: {"A": {"mean_wealth": 100.0,
                   "paired_gap_vs_G": {"mean": 1.0}},
             "B": {"mean_wealth": 102.0,
                   "paired_gap_vs_G": {"mean": 3.0}}},
        11: {"A": {"mean_wealth": 104.0,
                   "paired_gap_vs_G": {"mean": 2.0}},
             "B": {"mean_wealth": 106.0,
                   "paired_gap_vs_G": {"mean": 4.0}}},
    }
    summary = summarize_across_seeds(per_seed, reference="G")
    assert summary["A"]["mean_wealth_mean"] == pytest.approx(102.0)
    assert summary["A"]["mean_wealth_sd"] == pytest.approx(2.0)
    assert summary["B"]["paired_gap_mean"] == pytest.approx(3.5)
    # Per-seed gaps are preserved, not averaged away silently.
    assert summary["A"]["per_seed_gap"] == [1.0, 2.0]
    assert summary["n_seeds"] == 2


# ---------------- chrono_eval constants ----------------

def test_chrono_eval_validation_window_is_2021_2022():
    from dhl.experiments.chrono_eval import TRAIN_END, VAL_END, VAL_START
    assert TRAIN_END == "2020-01-01"
    assert VAL_START == "2021-01-01"
    assert VAL_END == "2023-01-01"
    dates, _, _ = _toy_series(1500, "2018-01-01")
    val_mask = [(d >= VAL_START) and (d < VAL_END) for d in dates]
    val_dates = [d for d, m in zip(dates, val_mask) if m]
    assert min(val_dates) >= "2021-01-01"
    assert max(val_dates) < "2023-01-01"
    assert not any(d < "2021-01-01" for d in val_dates)
