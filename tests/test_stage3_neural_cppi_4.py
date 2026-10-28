"""Stage 3 NC4: honest historical stress tiers.

Fixture-first tests. They import the tier catalog, the window
classifier, and the table-metadata gate that NC4 adds. Before the
implementation they fail at collection (ImportError); after it, they
must all pass.

No private tail-risk data, results, or windows are used anywhere
here. Every date window below is either a public calendar fact
(2008, COVID 2020, 2023+) or a synthetic fixture. The 1929-1933 tier
is a metadata definition with stated conventions only; no Depression
series, daily or monthly, is loaded or asserted.
"""

import pytest

from dhl.experiments.chrono_eval import chrono_table_metadata
from dhl.markets.history_ext import (
    StressTier,
    classify_window,
    stress_tiers,
    table_metadata,
    validate_table_metadata,
    validate_tier,
)


def test_stage3_neural_cppi_4_gfc_pre2020_model_is_in_sample():
    tiers = stress_tiers()
    tier = tiers["gfc_in_sample_diagnostic"]
    assert tier.seen is True
    assert classify_window(
        tier.training_start, tier.training_cutoff,
        tier.eval_start, tier.eval_end) == "in_sample"


def test_stage3_neural_cppi_4_gfc_strictly_pre2008_model_is_unseen():
    tiers = stress_tiers()
    tier = tiers["gfc_unseen"]
    assert tier.seen is False
    assert tier.training_cutoff <= "2008-01-01"
    assert classify_window(
        tier.training_start, tier.training_cutoff,
        tier.eval_start, tier.eval_end) == "unseen"


def test_stage3_neural_cppi_4_covid_held_out_is_unseen():
    tiers = stress_tiers()
    tier = tiers["covid_held_out"]
    assert tier.seen is False
    assert tier.eval_start == "2020-02-19"
    assert classify_window(
        tier.training_start, tier.training_cutoff,
        tier.eval_start, tier.eval_end) == "unseen"


def test_stage3_neural_cppi_4_post_freeze_reserved():
    tiers = stress_tiers()
    tier = tiers["post_freeze_reserved"]
    assert tier.freeze_date is not None
    assert tier.eval_start >= tier.freeze_date
    validate_tier(tier, tuning_cutoff="2022-12-31")
    with pytest.raises(ValueError):
        validate_tier(tier, tuning_cutoff="2023-06-01")


def test_stage3_neural_cppi_4_depression_monthly_proxy_complete():
    tiers = stress_tiers()
    tier = tiers["depression_monthly_proxy"]
    assert tier.frequency == "monthly"
    assert tier.eval_start == "1929-01-01"
    assert tier.proxy and tier.dividends and tier.cash_convention
    assert "SPY" not in tier.instruments
    assert "TLT" not in tier.instruments
    assert "GLD" not in tier.instruments
    validate_tier(tier)


def test_stage3_neural_cppi_4_daily_depression_claim_rejected():
    tier = StressTier(
        name="bogus_daily_depression", frequency="daily",
        training_start="2007-01-01", training_cutoff="2020-01-01",
        eval_start="1931-01-01", eval_end="1932-01-01",
        proxy="daily proxy", dividends="price only",
        cash_convention="zero", seen=False)
    with pytest.raises(ValueError):
        validate_tier(tier)


def test_stage3_neural_cppi_4_etf_backcast_rejected():
    for etf in ("SPY", "TLT", "GLD"):
        tier = StressTier(
            name=f"bogus_{etf}_1930s", frequency="monthly",
            training_start="2007-01-01", training_cutoff="2020-01-01",
            eval_start="1932-01-01", eval_end="1933-01-01",
            proxy=f"{etf} monthly", dividends="adjusted",
            cash_convention="zero", seen=False, instruments=(etf,))
        with pytest.raises(ValueError):
            validate_tier(tier)


def test_stage3_neural_cppi_4_missing_metadata_field_rejected():
    meta = table_metadata(stress_tiers()["gfc_unseen"])
    for field in ("frequency", "training_cutoff", "proxy",
                  "dividends", "cash_convention"):
        broken = {k: v for k, v in meta.items() if k != field}
        with pytest.raises(ValueError):
            validate_table_metadata(broken)
    empty = dict(meta)
    empty["cash_convention"] = ""
    with pytest.raises(ValueError):
        validate_table_metadata(empty)


def test_stage3_neural_cppi_4_seen_flag_must_match_overlap():
    tier = StressTier(
        name="lying_tier", frequency="daily",
        training_start="2007-01-01", training_cutoff="2020-01-01",
        eval_start="2008-01-01", eval_end="2009-01-01",
        proxy="SPY daily", dividends="adjusted",
        cash_convention="2%", seen=False, instruments=("SPY",))
    with pytest.raises(ValueError):
        validate_tier(tier)


def test_stage3_neural_cppi_4_chrono_tables_carry_metadata():
    tables = chrono_table_metadata()
    assert {"test_2023plus", "val_2021_2022", "covid_crash",
            "insample_bootstrap"} <= set(tables)
    for name, meta in tables.items():
        validate_table_metadata(meta)
        assert meta["training_cutoff"] == "2020-01-01"
        assert "2%" in meta["cash_convention"]
        assert meta["dividends"] != ""
    assert tables["covid_crash"]["seen"] is False
    assert tables["insample_bootstrap"]["seen"] is True


def test_stage3_neural_cppi_4_catalog_all_tiers_validate():
    tiers = stress_tiers()
    assert {"gfc_in_sample_diagnostic", "gfc_unseen", "covid_held_out",
            "post_freeze_reserved",
            "depression_monthly_proxy"} <= set(tiers)
    for tier in tiers.values():
        validate_tier(tier)
        validate_table_metadata(table_metadata(tier))
