"""The leakage tests. If these pass and the model still scores 0.85, the leak
is somewhere else — but this is where it usually is.

The strategy is invariance, not inspection. Rather than reading the code and
hoping, we build a synthetic dataset, plant an absurd result in it, and check
that features which must not see it are bit-for-bit unchanged.
"""

import datetime as dt
from typing import Any, cast

import polars as pl
import pytest

from src.features.build import FEATURE_COLUMNS, MAX_REST_DAYS, build_features

TEAMS = ("A", "B", "C", "D")
START = dt.date(2020, 8, 1)


def _fixture(
    day: int, home: str, away: str, hg: int, ag: int, season: str = "2020-21"
) -> dict[str, Any]:
    return {
        "date": START + dt.timedelta(days=7 * day),
        "season": season,
        "league": "Test League",
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "home_shots_target": 3 + hg,
        "away_shots_target": 3 + ag,
        "home_np_xg": 0.4 + 0.3 * hg,
        "away_np_xg": 0.4 + 0.3 * ag,
        "result": "H" if hg > ag else ("D" if hg == ag else "A"),
    }


def _synthetic(n_rounds: int = 12) -> pl.DataFrame:
    """A tidy double round-robin, two matches per matchday."""
    rows = []
    pairs = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
    day = 0
    for _ in range(n_rounds):
        for home, away in pairs:
            rows.append(_fixture(day, home, away, day % 3, (day + 1) % 3))
            day += 1
    return pl.DataFrame(rows).sort("date")


LAST_DAY = dt.date(2099, 1, 1)


def test_features_do_not_change_when_later_matches_are_added() -> None:
    """The test of prompt 3.2.

    A match played after D cannot influence the features of a match played on
    D. We plant four 9-0 blowouts at the end — a result so far out of
    distribution that any leak would move every rolling mean it touches.
    """
    base = _synthetic()
    cutoff = cast(dt.date, base["date"].max())

    later = pl.DataFrame(
        [_fixture(100 + i, "A", "B", 9, 0, season="2020-21") for i in range(4)]
    )
    extended = pl.concat([base, later]).sort("date")

    from_base = build_features(base, cutoff)
    from_extended = build_features(extended, cutoff)

    assert from_base.height == from_extended.height
    for column in FEATURE_COLUMNS:
        assert from_base[column].to_list() == from_extended[column].to_list(), column


def test_features_of_a_match_do_not_use_that_match_own_statistics() -> None:
    """A 0-0 and a 9-0 of the same fixture must produce the same features."""
    quiet = _synthetic()
    cutoff_date = cast(dt.date, quiet["date"].max())
    loud = quiet.with_columns(
        pl.when(pl.col("date") == cutoff_date)
        .then(9)
        .otherwise(pl.col("home_goals"))
        .alias("home_goals"),
        pl.when(pl.col("date") == cutoff_date)
        .then(20)
        .otherwise(pl.col("home_shots_target"))
        .alias("home_shots_target"),
        pl.when(pl.col("date") == cutoff_date)
        .then(5.0)
        .otherwise(pl.col("home_np_xg"))
        .alias("home_np_xg"),
    )
    cutoff = cutoff_date

    a = build_features(quiet, cutoff).filter(pl.col("date") == cutoff)
    b = build_features(loud, cutoff).filter(pl.col("date") == cutoff)

    assert a.height > 0
    for column in FEATURE_COLUMNS:
        assert a[column].to_list() == b[column].to_list(), column


def test_the_cutoff_is_a_hard_wall() -> None:
    """Features built with a cutoff must equal those built on truncated data.

    This is what makes build_features safe in production: passing today's date
    cannot pull in anything from the future, because there is nothing to pull.
    """
    full = _synthetic()
    cutoff = cast(dt.date, full["date"][40])

    with_cutoff = build_features(full, cutoff)
    on_truncated = build_features(full.filter(pl.col("date") <= cutoff), cutoff)

    assert with_cutoff.height == on_truncated.height
    for column in FEATURE_COLUMNS:
        assert with_cutoff[column].to_list() == on_truncated[column].to_list(), column


def test_no_row_after_the_cutoff_is_returned() -> None:
    full = _synthetic()
    cutoff = cast(dt.date, full["date"][20])
    latest = cast(dt.date, build_features(full, cutoff)["date"].max())
    assert latest <= cutoff


def test_the_first_matches_of_a_team_have_null_rolling_features() -> None:
    """A one-match average is not a five-match average; it must be null."""
    features = build_features(_synthetic(), LAST_DAY).sort("date")
    early = features.head(2)
    assert early["goals_scored_diff_5"].null_count() == 2


def test_rolling_windows_use_exactly_the_previous_n_matches() -> None:
    """Hand-computable case: A scores 1, 2, 3, 4, 5 then plays a sixth match."""
    rows = [
        _fixture(0, "A", "B", 1, 0),
        _fixture(1, "A", "C", 2, 0),
        _fixture(2, "A", "D", 3, 0),
        _fixture(3, "A", "B", 4, 0),
        _fixture(4, "A", "C", 5, 0),
        _fixture(5, "A", "D", 0, 0),
    ]
    features = build_features(pl.DataFrame(rows).sort("date"), LAST_DAY)
    sixth = features.filter(pl.col("date") == rows[-1]["date"])
    # (1 + 2 + 3 + 4 + 5) / 5 = 3.0 for A, still null for D.
    assert sixth["goals_scored_5_home"][0] == pytest.approx(3.0)


def test_rest_days_measure_the_gap_to_the_previous_match() -> None:
    rows = [_fixture(0, "A", "B", 1, 0), _fixture(1, "A", "C", 0, 0)]
    features = build_features(pl.DataFrame(rows).sort("date"), LAST_DAY)
    second = features.filter(pl.col("date") == rows[-1]["date"])
    assert second["rest_days_home"][0] == 7.0


def test_build_features_refuses_a_cutoff_before_the_first_match() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        build_features(_synthetic(), dt.date(2000, 1, 1))


def test_build_features_refuses_unsorted_input() -> None:
    with pytest.raises(ValueError, match="chronological"):
        build_features(_synthetic().sort("date", descending=True), LAST_DAY)


def test_rest_days_are_capped_so_a_long_absence_is_not_a_feature() -> None:
    """A team back after two years is not "well rested", it is a different team."""
    rows = [
        _fixture(0, "A", "B", 1, 0),
        _fixture(100, "A", "C", 0, 0),  # 700 days later
    ]
    features = build_features(pl.DataFrame(rows).sort("date"), LAST_DAY)
    late = features.filter(pl.col("date") == rows[-1]["date"])
    assert late["rest_days_home"][0] == MAX_REST_DAYS
