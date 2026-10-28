"""Extended market history: pre-2015 SPY (2008 crisis) and VIX.

The committed ``data/spy_daily_2015.csv`` / ``spy_daily_2021.csv`` start
in 2015, so the 2008 financial crisis -- the cleanest pre-2020 crash
regime for training -- is missing. This module fetches it (plus VIX,
the market's crash gauge) and stores it alongside the existing parts.

Files written (all under ``data/``):
- ``spy_daily_2007.csv``: SPY daily bars 2007-01-01 onward. The existing
  ``_part_files`` glob (``spy_daily_*.csv``) picks it up automatically
  and sorts it before the 2015 part, so ``load_spy_daily`` returns one
  continuous 2007+ series with no code changes elsewhere.
- ``vix_daily.csv``: Cboe VIX daily closes, same date range.

VIX timing discipline (no look-ahead): a decision made at the close of
day ``t`` after seeing that day's price action may NOT use the VIX
close of day ``t`` (published after the decision point in a live
setting). Features must use VIX from ``t-1`` or earlier. The
``aligned_vix_lagged`` helper enforces this by shifting one day.
"""

import csv
import datetime
import json
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

from dhl.markets.real import DATA_DIR, YAHOO_PERIOD1_2007

_VIX_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
    f"?period1={YAHOO_PERIOD1_2007}&period2=9999999999&interval=1d"
)
_UA = {"User-Agent": "Mozilla/5.0 (compatible; neural-cppi)"}


def _download_chart(url):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    return result


def fetch_spy_2007(data_dir=DATA_DIR):
    """Download SPY daily bars from 2007-01-01, save as ``spy_daily_2007.csv``.

    Returns the path written. Idempotent: re-running refreshes the file.
    """
    result = _download_chart(
        "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
        f"?period1={YAHOO_PERIOD1_2007}&period2=9999999999&interval=1d"
    )
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]

    rows = []
    for i, ts in enumerate(stamps):
        if quote["close"][i] is None or adj[i] is None:
            continue
        day = datetime.datetime.fromtimestamp(
            ts, datetime.timezone.utc).date().isoformat()
        # Keep only pre-2015 bars; the committed parts cover 2015+.
        if day >= "2015-01-01":
            continue
        rows.append([
            day,
            f"{quote['open'][i]:.2f}",
            f"{quote['high'][i]:.2f}",
            f"{quote['low'][i]:.2f}",
            f"{quote['close'][i]:.2f}",
            f"{adj[i]:.2f}",
            quote["volume"][i],
        ])
    path = os.path.join(data_dir, "spy_daily_2007.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"])
        writer.writerows(rows)
    return path


def fetch_vix_daily(data_dir=DATA_DIR):
    """Download Cboe VIX daily closes, save as ``vix_daily.csv``.

    Returns the path written. Idempotent.
    """
    result = _download_chart(_VIX_URL)
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    rows = []
    for i, ts in enumerate(stamps):
        if quote["close"][i] is None:
            continue
        day = datetime.datetime.fromtimestamp(
            ts, datetime.timezone.utc).date().isoformat()
        rows.append([day, f"{quote['close'][i]:.2f}"])
    path = os.path.join(data_dir, "vix_daily.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Close"])
        writer.writerows(rows)
    return path


def load_vix_daily(data_dir=DATA_DIR):
    """Load ``vix_daily.csv``. Returns dict with ``dates`` and ``close``.

    Raises FileNotFoundError if not yet fetched (see
    :func:`fetch_vix_daily`).
    """
    path = os.path.join(data_dir, "vix_daily.csv")
    dates, closes = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dates.append(row["Date"])
            closes.append(float(row["Close"]))
    return {"dates": dates, "close": np.asarray(closes)}


def aligned_vix_lagged(spy_dates, vix_data, lag_days=1):
    """VIX closes aligned to SPY dates, lagged to prevent look-ahead.

    For each SPY date ``d``, returns the VIX close from ``lag_days``
    trading days before ``d`` (default 1: yesterday's VIX, the latest
    value knowable before today's close). Dates with no available
    lagged VIX get NaN.

    This is the ONLY sanctioned way to feed VIX into features: the raw
    same-day close must never be used at decision time.
    """
    spy_dates = list(spy_dates)
    vix_dates = list(vix_data["dates"])
    vix_close = np.asarray(vix_data["close"], dtype=float)
    vix_pos = {d: j for j, d in enumerate(vix_dates)}

    out = np.full(len(spy_dates), np.nan)
    for i, d in enumerate(spy_dates):
        j = vix_pos.get(d)
        if j is not None and j >= lag_days:
            out[i] = vix_close[j - lag_days]
    return out


# --- Stage 3 NC4: honest historical stress tiers ---
#
# A pre-2020-trained model has seen 2008 (the registry admits the
# 2008 crisis window as a legal training window), so it cannot call
# 2008 an unseen crash. These tiers make that distinction mechanical:
# every evaluation table must carry frequency, training cutoff,
# proxy, dividends, and cash convention, and the seen flag must match
# the actual overlap between the training window and the eval window.
# No private tail-risk data, results, or windows are referenced here;
# the 1929-1933 tier is a metadata definition with stated conventions
# only, and no Depression series of any frequency is loaded.

ETF_INCEPTION = {
    "SPY": "1993-01-29",
    "TLT": "2002-07-22",
    "GLD": "2004-11-18",
}
BACKCAST_INSTRUMENTS = ("SPY", "TLT", "GLD")
DAILY_HISTORY_FLOOR = "1940-01-01"
CASH_CONVENTION = ("cash 2% per year; negative cash at the same 2% "
                    "rate (borrow_rate=None default)")
SPY_DAILY_PROXY = "SPY daily adjusted close"
SPY_DIVIDENDS = "adjusted close (dividends reinvested)"


@dataclass(frozen=True)
class StressTier:
    """One honestly labeled evaluation tier.

    ``training_cutoff`` is exclusive: training uses data strictly
    before it, starting at ``training_start``. ``seen`` must equal
    whether the training window overlaps the eval window; a tier
    that claims otherwise fails validation. ``freeze_date``, when
    set, is the last date any tuning decision may use; the eval
    window of a frozen tier must not feed tuning.
    """

    name: str
    frequency: str
    training_start: str
    training_cutoff: str
    eval_start: str
    eval_end: str | None
    proxy: str
    dividends: str
    cash_convention: str
    seen: bool
    instruments: tuple = ()
    freeze_date: str | None = None
    previously_viewed: bool = False
    note: str = ""


def _windows_overlap(training_start, training_cutoff,
                     eval_start, eval_end):
    """True when [training_start, training_cutoff) meets the eval window."""
    if eval_end is not None and eval_end <= training_start:
        return False
    if eval_start >= training_cutoff:
        return False
    return True


def classify_window(training_start, training_cutoff,
                    eval_start, eval_end):
    """Label an eval window "in_sample" or "unseen" for a training window."""
    if _windows_overlap(training_start, training_cutoff,
                        eval_start, eval_end):
        return "in_sample"
    return "unseen"


def validate_tier(tier, tuning_cutoff=None):
    """Fail-closed validation of one tier. Raises ValueError."""
    if tier.frequency not in ("daily", "monthly"):
        raise ValueError(
            f"tier {tier.name}: unknown frequency {tier.frequency!r}")
    for field_name in ("proxy", "dividends", "cash_convention"):
        if not getattr(tier, field_name):
            raise ValueError(
                f"tier {tier.name}: {field_name} must be stated")
    if not tier.training_start < tier.training_cutoff:
        raise ValueError(
            f"tier {tier.name}: training window is empty or inverted")
    if tier.eval_end is not None and not tier.eval_start < tier.eval_end:
        raise ValueError(f"tier {tier.name}: eval window is empty")
    expected_seen = _windows_overlap(
        tier.training_start, tier.training_cutoff,
        tier.eval_start, tier.eval_end)
    if tier.seen != expected_seen:
        raise ValueError(
            f"tier {tier.name}: seen={tier.seen} contradicts the "
            f"training window [{tier.training_start}, "
            f"{tier.training_cutoff}) vs eval window "
            f"[{tier.eval_start}, {tier.eval_end})")
    if tier.frequency == "daily" and tier.eval_start < DAILY_HISTORY_FLOOR:
        raise ValueError(
            f"tier {tier.name}: daily claims before "
            f"{DAILY_HISTORY_FLOOR} are refused; daily Depression "
            f"data/rights are not established, omit the claim")
    for inst in tier.instruments:
        if (inst in ETF_INCEPTION
                and tier.eval_start < ETF_INCEPTION[inst]):
            raise ValueError(
                f"tier {tier.name}: {inst} did not exist before "
                f"{ETF_INCEPTION[inst]}; backcasting it into "
                f"{tier.eval_start} is refused")
        if (inst in BACKCAST_INSTRUMENTS
                and tier.eval_start < DAILY_HISTORY_FLOOR):
            raise ValueError(
                f"tier {tier.name}: {inst} backcast into the 1930s "
                f"is refused")
    if (tier.freeze_date is not None and tuning_cutoff is not None
            and tuning_cutoff > tier.freeze_date):
        raise ValueError(
            f"tier {tier.name}: tuning cutoff {tuning_cutoff} is "
            f"after freeze date {tier.freeze_date}; post-freeze "
            f"data must not feed tuning")
    return True


def stress_tiers():
    """The catalog of honestly labeled stress tiers."""
    common_spy = dict(proxy=SPY_DAILY_PROXY, dividends=SPY_DIVIDENDS,
                      cash_convention=CASH_CONVENTION,
                      instruments=("SPY",))
    return {
        "gfc_in_sample_diagnostic": StressTier(
            name="gfc_in_sample_diagnostic", frequency="daily",
            training_start="2007-01-01", training_cutoff="2020-01-01",
            eval_start="2008-01-01", eval_end="2009-01-01",
            seen=True, **common_spy,
            note="A pre-2020-trained model has seen 2008; this is an "
                 "in-sample diagnostic, never an unseen crash."),
        "gfc_unseen": StressTier(
            name="gfc_unseen", frequency="daily",
            training_start="2007-01-01", training_cutoff="2008-01-01",
            eval_start="2008-01-01", eval_end="2009-01-01",
            seen=False, **common_spy,
            note="Model trained strictly before 2008; 2008 is "
                 "genuinely unseen."),
        "covid_held_out": StressTier(
            name="covid_held_out", frequency="daily",
            training_start="2007-01-01", training_cutoff="2020-01-01",
            eval_start="2020-02-19", eval_end="2020-05-01",
            seen=False, previously_viewed=True, **common_spy,
            note="COVID stress held out from training; outcomes "
                 "previously reported."),
        "validation_2021_2022": StressTier(
            name="validation_2021_2022", frequency="daily",
            training_start="2007-01-01", training_cutoff="2020-01-01",
            eval_start="2021-01-01", eval_end="2023-01-01",
            seen=False, previously_viewed=True, **common_spy,
            note="Validation years; previously reported."),
        "post_freeze_reserved": StressTier(
            name="post_freeze_reserved", frequency="daily",
            training_start="2007-01-01", training_cutoff="2020-01-01",
            eval_start="2023-01-01", eval_end=None,
            freeze_date="2023-01-01",
            seen=False, previously_viewed=True, **common_spy,
            note="Reserved post-freeze evaluation; tuning cutoffs "
                 "after the freeze date are refused."),
        "depression_monthly_proxy": StressTier(
            name="depression_monthly_proxy", frequency="monthly",
            training_start="2007-01-01", training_cutoff="2020-01-01",
            eval_start="1929-01-01", eval_end="1934-01-01",
            proxy="broad US equity monthly proxy (no ETF backcast)",
            dividends="total return (dividends reinvested) as stated "
                      "by the proxy series; a price-only series must "
                      "restate this field",
            cash_convention="period US Treasury bill monthly return "
                            "proxy",
            seen=False, instruments=(),
            note="Monthly proxy stress only. No daily Depression "
                 "claims and no SPY/TLT/GLD backcasting. A real run "
                 "must verify its series matches these stated "
                 "conventions or restate them."),
    }


def table_metadata(tier):
    """The metadata block every results table must carry."""
    return {
        "tier": tier.name,
        "frequency": tier.frequency,
        "training_start": tier.training_start,
        "training_cutoff": tier.training_cutoff,
        "eval_start": tier.eval_start,
        "eval_end": tier.eval_end,
        "proxy": tier.proxy,
        "dividends": tier.dividends,
        "cash_convention": tier.cash_convention,
        "seen": tier.seen,
        "previously_viewed": tier.previously_viewed,
        "instruments": list(tier.instruments),
        "freeze_date": tier.freeze_date,
    }


def validate_table_metadata(meta):
    """Fail-closed validation of a table metadata block."""
    for field_name in ("frequency", "training_cutoff", "proxy",
                       "dividends", "cash_convention", "eval_start"):
        if not meta.get(field_name):
            raise ValueError(
                f"table metadata: {field_name} must be stated")
    if meta["frequency"] not in ("daily", "monthly",
                                 "block-bootstrap resample of daily"):
        raise ValueError(
            f"table metadata: unknown frequency "
            f"{meta['frequency']!r}")
    if (meta["frequency"] == "daily"
            and meta["eval_start"] < DAILY_HISTORY_FLOOR):
        raise ValueError(
            "table metadata: daily claims before "
            f"{DAILY_HISTORY_FLOOR} are refused")
    for inst in meta.get("instruments", ()):
        if (inst in ETF_INCEPTION
                and meta["eval_start"] < ETF_INCEPTION[inst]):
            raise ValueError(
                f"table metadata: {inst} backcast into "
                f"{meta['eval_start']} is refused")
        if (inst in BACKCAST_INSTRUMENTS
                and meta["eval_start"] < DAILY_HISTORY_FLOOR):
            raise ValueError(
                f"table metadata: {inst} backcast into the 1930s "
                f"is refused")
    if all(k in meta for k in ("training_start", "training_cutoff",
                               "eval_start", "eval_end", "seen")):
        expected = _windows_overlap(
            meta["training_start"], meta["training_cutoff"],
            meta["eval_start"], meta["eval_end"])
        if meta["seen"] != expected:
            raise ValueError(
                "table metadata: seen flag contradicts the training "
                "and eval windows")
    return True
