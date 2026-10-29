"""Stage 3 neural-cppi E4: budget utilization and recovery diagnosis.

Fixture-first tests. They import the diagnostic module that E4 adds.
Before the implementation they fail at collection (ImportError); after
it, they must all pass.

No private tail-risk data, results, or windows are used anywhere here.
Every path below is a synthetic fixture with hand-computed values, and
the source falsifiers are tested as hypotheses on those fixtures, not
as measured claims about any archived episode: on a synthetic path the
diagnostic must report the cushion and counterfactual numbers exactly,
whatever their sign, and must not assume the falsifier conclusion.

Definitions pinned here (all per rebalance step, pre-trade wealth V):
  cushion_dollars = V - floor
  cushion_ratio = clamp((V - floor) / V, -1, 1), matching rollout
  allowable_budget = min(multiplier * max(cushion_dollars, 0), cap * V)
  proposal_dollars = proposal_weight * V (raw proposal, pre-clamp)
  executed_weight = clamp(proposal_weight, 0, cap)
  executed_dollars = holding_post from the exact fee solve
  clipping_weight = proposal_weight - executed_weight
  unused_budget = allowable_budget - executed_dollars (may be negative
    when execution exceeds the classic allowable budget)
  current vol = 21d annualized population std of log returns strictly
    before the step, matching _vol_features
  current drawdown = trailing 60d peak-to-trough price drawdown at the
    step, matching _crash_features
"""

import math

import numpy as np
import pytest

from dhl.eval.budget_diagnosis import (
    counterfactual_attribution,
    diagnose_episode,
    replay_episode,
    summarize_diagnosis,
)

TOL = 1e-9


def _flat_fee_buy(v, w, cost):
    """Exact fee solve for a buy from zero holdings, plain floats."""
    holding_post = w * v / (1.0 + w * cost)
    fee = cost * holding_post
    return holding_post, fee


def test_stage3_neural_cppi_4_hand_computed_budget_series():
    # One step, flat-then-rising path, zero cash rate so the hand
    # computation has no accrual term to hide behind.
    rows, terminal = replay_episode(
        prices=[100.0, 110.0],
        proposals=[0.5],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.002,
        v0=100.0,
        r=0.0,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["step"] == 0
    assert row["price"] == 100.0
    assert row["wealth_pre"] == pytest.approx(100.0, abs=TOL)
    assert row["cushion_dollars"] == pytest.approx(10.0, abs=TOL)
    assert row["cushion_ratio"] == pytest.approx(0.10, abs=TOL)
    assert row["allowable_budget"] == pytest.approx(40.0, abs=TOL)
    assert row["proposal_weight"] == pytest.approx(0.5, abs=TOL)
    assert row["proposal_dollars"] == pytest.approx(50.0, abs=TOL)
    assert row["executed_weight"] == pytest.approx(0.5, abs=TOL)
    holding_post, fee = _flat_fee_buy(100.0, 0.5, 0.002)
    assert row["executed_dollars"] == pytest.approx(holding_post, abs=TOL)
    assert row["clipping_weight"] == pytest.approx(0.0, abs=TOL)
    assert row["fee"] == pytest.approx(fee, abs=TOL)
    assert row["unused_budget"] == pytest.approx(40.0 - holding_post, abs=TOL)
    assert row["cum_fees"] == pytest.approx(fee, abs=TOL)
    assert row["vol"] == pytest.approx(0.0, abs=TOL)
    assert row["drawdown"] == pytest.approx(0.0, abs=TOL)
    assert row["state"] == "normal"
    # Terminal: holding rides to 110, cash is unchanged at r = 0, then
    # liquidation is charged on the true terminal holding value.
    cash_post = 100.0 - holding_post - fee
    holding_terminal = (holding_post / 100.0) * 110.0
    liq_fee = 0.002 * holding_terminal
    assert terminal["terminal_wealth"] == pytest.approx(
        cash_post + holding_terminal - liq_fee, abs=TOL)
    assert terminal["liquidation_fee"] == pytest.approx(liq_fee, abs=TOL)
    assert terminal["cum_fees_incl_liquidation"] == pytest.approx(
        fee + liq_fee, abs=TOL)
    # Self-financing identity at the trade: cash + holding + fee = V.
    assert cash_post + holding_post + fee == pytest.approx(100.0, abs=TOL)


def test_stage3_neural_cppi_4_clipping_and_cushion_edges():
    # Proposal above the cap clips down; proposal below zero clips up.
    rows, _ = replay_episode(
        prices=[100.0, 100.0, 100.0],
        proposals=[2.0, -0.5],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    assert rows[0]["executed_weight"] == pytest.approx(1.5, abs=TOL)
    assert rows[0]["clipping_weight"] == pytest.approx(0.5, abs=TOL)
    assert rows[0]["clipping_dollars"] == pytest.approx(50.0, abs=TOL)
    assert rows[0]["executed_dollars"] == pytest.approx(150.0, abs=TOL)
    assert rows[1]["executed_weight"] == pytest.approx(0.0, abs=TOL)
    assert rows[1]["clipping_weight"] == pytest.approx(-0.5, abs=TOL)
    assert rows[1]["clipping_dollars"] == pytest.approx(-50.0, abs=TOL)
    assert rows[1]["executed_dollars"] == pytest.approx(0.0, abs=TOL)

    # Wealth exactly at the floor: no cushion, no allowable budget.
    rows, _ = replay_episode(
        prices=[100.0, 100.0],
        proposals=[0.5],
        floor=100.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    assert rows[0]["cushion_dollars"] == pytest.approx(0.0, abs=TOL)
    assert rows[0]["allowable_budget"] == pytest.approx(0.0, abs=TOL)
    # Execution still follows the proposal; unused budget goes negative
    # rather than being silently floored at zero.
    assert rows[0]["unused_budget"] == pytest.approx(-50.0, abs=TOL)

    # Wealth below the floor: cushion and allowable are negative/zero,
    # never a positive budget invented from a breach.
    rows, _ = replay_episode(
        prices=[100.0, 100.0],
        proposals=[0.5],
        floor=110.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    assert rows[0]["cushion_dollars"] == pytest.approx(-10.0, abs=TOL)
    assert rows[0]["cushion_ratio"] == pytest.approx(-0.10, abs=TOL)
    assert rows[0]["allowable_budget"] == pytest.approx(0.0, abs=TOL)


def test_stage3_neural_cppi_4_vol_and_drawdown_match_features():
    # Returns before step 2 are log(110/100) and log(99/110); the
    # population std of exactly those two returns, annualized, is the
    # vol at step 2. Step 0 and step 1 see fewer than two returns.
    prices = [100.0, 110.0, 99.0, 99.0]
    rows, _ = replay_episode(
        prices=prices,
        proposals=[0.0, 0.0, 0.0],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    assert rows[0]["vol"] == pytest.approx(0.0, abs=TOL)
    assert rows[1]["vol"] == pytest.approx(0.0, abs=TOL)
    lr = np.log(np.array([110.0 / 100.0, 99.0 / 110.0]))
    expected_vol = float(lr.std(ddof=0) * math.sqrt(252.0))
    assert rows[2]["vol"] == pytest.approx(expected_vol, abs=1e-12)
    assert rows[2]["drawdown"] == pytest.approx(0.10, abs=TOL)
    # A later step after the trough has recovered sees a smaller drawdown.
    assert rows[0]["drawdown"] == pytest.approx(0.0, abs=TOL)


def test_stage3_neural_cppi_4_rebound_and_secondary_dip_transitions():
    # 100 -> 94 (6% dip) -> 97 (3.19% off the trough: rebound)
    # -> 94.5 (2.58% off the rebound peak: secondary dip).
    prices = [100.0, 94.0, 97.0, 94.5, 94.5]
    rows, _ = replay_episode(
        prices=prices,
        proposals=[0.0, 0.0, 0.0, 0.0],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    assert [r["state"] for r in rows] == [
        "normal", "dip", "rebound", "secondary_dip"]
    # Regaining the pre-dip peak resets to normal.
    rows, _ = replay_episode(
        prices=[100.0, 94.0, 97.0, 101.0, 101.0],
        proposals=[0.0, 0.0, 0.0, 0.0],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    assert [r["state"] for r in rows] == ["normal", "dip", "rebound", "normal"]
    # Rising off a secondary-dip trough without regaining the peak
    # returns to rebound, not straight to normal.
    rows, _ = replay_episode(
        prices=[100.0, 94.0, 97.0, 94.5, 98.0, 98.0],
        proposals=[0.0, 0.0, 0.0, 0.0, 0.0],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    assert [r["state"] for r in rows] == [
        "normal", "dip", "rebound", "secondary_dip", "rebound"]
    # A shallow 2% fall never enters the dip state at all.
    rows, _ = replay_episode(
        prices=[100.0, 98.0, 98.0],
        proposals=[0.0, 0.0],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    assert [r["state"] for r in rows] == ["normal", "normal"]


def test_stage3_neural_cppi_4_counterfactual_timing_and_fees():
    # Rising path, constant half weight: removing fees must raise
    # terminal wealth, and delaying every proposal one step must lower
    # it (the position misses the first rise). On a flat path the
    # one-step delay must change nothing at zero cash rate.
    report = counterfactual_attribution(
        prices=[100.0, 110.0, 121.0, 133.1],
        proposals=[0.5, 0.5, 0.5],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.002,
        v0=100.0,
        r=0.0,
    )
    assert report["fee_effect"] > 0.0
    assert report["timing_effect"] < 0.0
    assert report["delayed_terminal_wealth"] < report["actual_terminal_wealth"]
    assert report["zero_fee_terminal_wealth"] > report["actual_terminal_wealth"]

    flat = counterfactual_attribution(
        prices=[100.0, 100.0, 100.0, 100.0],
        proposals=[0.5, 0.5, 0.5],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    assert flat["timing_effect"] == pytest.approx(0.0, abs=TOL)
    assert flat["fee_effect"] == pytest.approx(0.0, abs=TOL)


def test_stage3_neural_cppi_4_validation_is_fail_closed():
    with pytest.raises(ValueError):
        replay_episode(prices=[100.0], proposals=[],
                       floor=90.0, multiplier=4.0, cap=1.5,
                       cost_rate=0.002, v0=100.0, r=0.0)
    with pytest.raises(ValueError):
        replay_episode(prices=[100.0, 0.0], proposals=[0.5],
                       floor=90.0, multiplier=4.0, cap=1.5,
                       cost_rate=0.002, v0=100.0, r=0.0)
    with pytest.raises(ValueError):
        replay_episode(prices=[100.0, 110.0], proposals=[0.5],
                       floor=90.0, multiplier=4.0, cap=1.5,
                       cost_rate=1.0, v0=100.0, r=0.0)
    with pytest.raises(ValueError):
        replay_episode(prices=[100.0, 110.0], proposals=[0.5, 0.5],
                       floor=90.0, multiplier=4.0, cap=1.5,
                       cost_rate=0.002, v0=100.0, r=0.0)


def test_stage3_neural_cppi_4_summary_reports_cushion_honestly():
    # Two steps at a thin cushion and one at a fat cushion: the summary
    # must report the low-cushion share as 2/3, not hide it in a mean.
    rows, terminal = replay_episode(
        prices=[100.0, 91.0, 91.0, 200.0],
        proposals=[1.0, 0.0, 0.0],
        floor=90.0,
        multiplier=4.0,
        cap=1.5,
        cost_rate=0.0,
        v0=100.0,
        r=0.0,
    )
    summary = summarize_diagnosis(rows, terminal)
    assert summary["n_steps"] == 3
    # Step wealths are 100, 91, 91 (fully invested into the fall, then
    # all cash): cushion ratios 0.10, 0.01099, 0.01099, so two of three
    # are below 0.10.
    assert summary["low_cushion_share"] == pytest.approx(2.0 / 3.0, abs=TOL)
    assert summary["total_fees_incl_liquidation"] == pytest.approx(0.0, abs=TOL)


def test_stage3_neural_cppi_4_diagnose_episode_matches_replay():
    pytest.importorskip("torch")
    import torch

    from dhl.experiments.cppi_learned import rollout  # noqa: F401

    def const_weight(w):
        def fn(i, V, tau_frac, cushion, feats):
            return torch.full_like(V, w)
        return fn

    prices = torch.tensor([[100.0, 110.0, 121.0]], dtype=torch.float64)
    rows, terminal, proposals = diagnose_episode(
        prices, const_weight(0.5), floor=90.0, multiplier=4.0, cap=1.5,
        cost_rate=0.002, r=0.0)
    assert proposals == pytest.approx([0.5, 0.5], abs=TOL)
    assert [r["proposal_weight"] for r in rows] == pytest.approx(
        [0.5, 0.5], abs=TOL)
    # The rollout ledger terminal wealth and the replay terminal wealth
    # for the same recorded proposals must agree.
    _, replay_terminal = replay_episode(
        prices=[100.0, 110.0, 121.0], proposals=[0.5, 0.5],
        floor=90.0, multiplier=4.0, cap=1.5, cost_rate=0.002,
        v0=100.0, r=0.0)
    assert terminal["terminal_wealth"] == pytest.approx(
        replay_terminal["terminal_wealth"], abs=1e-9)
