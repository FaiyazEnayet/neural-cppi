"""Episode registry for honest chronological evaluation.

A test episode is a named, exactly-dated span of market history with a
role (train, validation, test, stress, reserved), an independence
group, and provenance flags. The registry is the machine-readable
record that training periods, feature lookbacks, and reported window
counts are checked against, so a claim like "held out" can be verified
instead of trusted.

Conventions
-----------
Dates are ``YYYY-MM-DD`` strings. An episode ``[start, end)`` covers
price indices ``i`` with ``start <= dates[i] < end``; ``end=None``
means open-ended (through the last available date). String comparison
works for this format, so endpoints are exact and auditable.

Disjointness is asserted at two granularities:

* Price-bar granularity: no training return end-date may fall inside
  a held-out episode's span.
* Feature-lookback granularity: a feature at bar ``j`` consumes bars
  back to ``j - lookback``. A training block ``[s, e)`` therefore
  effectively consumes ``[s - lookback, e)``. If that extended span
  intersects a held-out episode, the block crosses an embargo and is
  rejected, even when the block itself sits outside the episode.

Roles ``validation``, ``test``, ``stress`` and ``reserved`` are all
held out for training: a training period may not touch them. The
``previously_viewed`` flag marks judge windows that researchers have
already inspected (reported in earlier results), so they are never
again described as unseen. ``predeclared`` episodes record new test
geography, asset family, and dates *before* their outcomes are
inspected.

Window counts: sliding windows overlap by construction, so every
reported test set carries both ``n_windows`` (all windows, possibly
overlapping) and ``n_non_overlapping`` (a greedy maximal set of
pairwise-disjoint windows). Significance-style claims must use the
second number, never the first.
"""

import bisect
from dataclasses import dataclass, field

HELD_OUT_ROLES = ("validation", "test", "stress", "reserved")


@dataclass(frozen=True)
class Episode:
    """One named span of market history."""

    name: str
    start: str
    end: str | None
    role: str
    independence_group: str
    previously_viewed: bool = False
    predeclared: bool = False
    geography: str = ""
    asset_family: str = ""
    note: str = ""

    def __post_init__(self):
        if self.role not in ("train",) + HELD_OUT_ROLES:
            raise ValueError(f"unknown episode role {self.role!r}")
        if self.end is not None and not self.start < self.end:
            raise ValueError(
                f"episode {self.name!r} has inverted dates "
                f"{self.start}..{self.end}")

    def index_span(self, dates):
        """Half-open price-index span [lo, hi) for this episode."""
        lo = bisect.bisect_left(dates, self.start)
        hi = len(dates) if self.end is None else bisect.bisect_left(
            dates, self.end)
        return lo, hi

    @property
    def is_held_out(self):
        return self.role in HELD_OUT_ROLES


def predeclare_episode(name, start, end, geography, asset_family,
                       role="test", independence_group=None, note=""):
    """Record a new test episode before inspecting its outcomes.

    The geography, asset family, and exact dates are fixed at
    declaration time. Nothing about the episode may be adjusted after
    outcomes are seen; a new declaration is required instead.
    """
    return Episode(
        name=name, start=start, end=end, role=role,
        independence_group=independence_group or name,
        previously_viewed=False, predeclared=True,
        geography=geography, asset_family=asset_family, note=note)


class EpisodeRegistry:
    """A set of episodes with fail-closed training validation."""

    def __init__(self, episodes):
        self._episodes = list(episodes)

    @property
    def episodes(self):
        return list(self._episodes)

    def held_out(self):
        return [e for e in self._episodes if e.is_held_out]

    def _check_span(self, dates, lo, hi, what, lookback_bars=0):
        ext_lo = max(0, lo - lookback_bars)
        for ep in self.held_out():
            elo, ehi = ep.index_span(dates)
            if ext_lo < ehi and elo < hi:
                # Extension-only crossing: the span itself starts at or
                # after the episode ends, but its feature lookback
                # reaches back into the episode's bars.
                via_lookback = lo >= ehi and ext_lo < ehi
                raise ValueError(
                    f"{what} crosses held-out episode "
                    f"{ep.name!r} [{ep.start}..{ep.end or 'open'})"
                    + (f" via {lookback_bars}-bar feature lookback"
                       if via_lookback else "")
                    + f" (role={ep.role}, group={ep.independence_group})")

    def validate_training_period(self, dates, train_start, train_end,
                                 lookback_bars=0):
        """Fail unless [train_start, train_end) touches no held-out bar.

        ``dates`` is the full price-date list. The training span is
        extended backwards by ``lookback_bars`` for feature
        consumption. Raises ``ValueError`` naming the offending
        episode; returns ``None`` on success.
        """
        dates = list(dates)
        lo = bisect.bisect_left(dates, train_start)
        hi = bisect.bisect_left(dates, train_end)
        if not lo < hi:
            raise ValueError(
                f"empty training period {train_start}..{train_end}")
        self._check_span(dates, lo, hi, "training period",
                         lookback_bars=lookback_bars)

    def validate_blocks(self, dates, blocks, lookback_bars=0):
        """Fail unless no block's lookback-extended span crosses an embargo.

        ``blocks`` is an iterable of ``(start_idx, end_idx)`` half-open
        price-index ranges. Raises ``ValueError`` naming the first
        offending block and episode.
        """
        dates = list(dates)
        for k, (s, e) in enumerate(blocks):
            if not 0 <= s < e <= len(dates):
                raise ValueError(f"block {k} has invalid span {(s, e)}")
            self._check_span(dates, s, e, f"training block {k}",
                             lookback_bars=lookback_bars)


def summarize_windows(starts, path_len):
    """Count windows and non-overlapping windows for explicit starts.

    Each window starting at ``s`` covers ``path_len + 1`` prices
    (``path_len`` steps). Returns ``n_windows`` (all starts, possibly
    overlapping), ``n_non_overlapping`` (greedy disjoint subset), and
    ``span_bars`` (first start to last window end).
    """
    starts = sorted(starts)
    n_windows = len(starts)
    if not starts:
        return {"n_windows": 0, "n_non_overlapping": 0, "span_bars": 0}
    width = path_len + 1
    count, last_end = 0, -1
    for s in starts:
        if s >= last_end:
            count += 1
            last_end = s + width
    span = starts[-1] + width - starts[0]
    return {"n_windows": n_windows, "n_non_overlapping": count,
            "span_bars": span}


def summarize_window_count(n_windows, path_len, stride):
    """Same counts as :func:`summarize_windows` for a regular stride.

    Starts are ``0, stride, 2 * stride, ...`` (``n_windows`` of them).
    """
    if n_windows <= 0:
        return {"n_windows": 0, "n_non_overlapping": 0, "span_bars": 0}
    starts = [k * stride for k in range(n_windows)]
    return summarize_windows(starts, path_len)


def standard_chrono_registry():
    """The episode set behind the chronological evaluation.

    Train is strictly pre-2020. 2020 bars outside the named COVID
    stress window are reserved, not silently assigned to any bucket.
    Validation is 2021-2022 and test is 2023+, both previously viewed
    judge windows (their outcomes were reported in earlier results),
    so they must never be described as unseen. The 2008 crisis window
    is a second legal training window for the ablation study.
    """
    return EpisodeRegistry([
        Episode(name="train_pre2020", start="0001-01-01",
                end="2020-01-01", role="train",
                independence_group="pre2020",
                note="legal training period for chronological runs"),
        Episode(name="gfc_training_window", start="2007-01-01",
                end="2020-01-01", role="train",
                independence_group="gfc-2008",
                note="2008 crisis data admitted to ablation training"),
        Episode(name="reserved_2020", start="2020-01-01",
                end="2021-01-01", role="reserved",
                independence_group="2020",
                note="2020 bars outside the named stress window; "
                     "no train/val/test bucket claims them"),
        Episode(name="covid_stress", start="2020-02-19",
                end="2020-05-01", role="stress",
                independence_group="covid-2020",
                previously_viewed=True,
                note="Feb-Apr 2020 crash stress case; outcomes "
                     "reported in earlier results"),
        Episode(name="validation_2021_2022", start="2021-01-01",
                end="2023-01-01", role="validation",
                independence_group="val-2021-2022",
                previously_viewed=True,
                note="validation years; previously reported"),
        Episode(name="test_2023plus", start="2023-01-01", end=None,
                role="test", independence_group="test-2023plus",
                previously_viewed=True,
                note="out-of-time test set; previously reported"),
    ])
