"""Elo tests. The invariance test is the one that matters.

An Elo implementation that leaks looks perfectly normal: it produces plausible
ratings and a suspiciously good model. The only way to catch it is to check
that a rating read at date D is unchanged by matches played after D.
"""

import datetime as dt

import polars as pl
import pytest

from src.features.elo import EloRatings


def _matches(rows: list[tuple[str, str, str, int, int]]) -> pl.DataFrame:
    """rows: (season, home, away, home_goals, away_goals), one per matchday."""
    return pl.DataFrame(
        {
            "date": [
                dt.date(2020, 8, 1) + dt.timedelta(days=7 * i) for i in range(len(rows))
            ],
            "season": [r[0] for r in rows],
            "home_team": [r[1] for r in rows],
            "away_team": [r[2] for r in rows],
            "home_goals": [r[3] for r in rows],
            "away_goals": [r[4] for r in rows],
        }
    )


BASE_ROWS = [
    ("2020-21", "A", "B", 2, 0),
    ("2020-21", "B", "A", 1, 1),
    ("2020-21", "A", "B", 0, 3),
    ("2020-21", "B", "A", 2, 2),
]


def test_an_unseen_team_sits_at_the_base_rating() -> None:
    elo = EloRatings().fit(_matches(BASE_ROWS))
    assert elo.get_rating("C", dt.date(2020, 9, 1)) == pytest.approx(1500.0)


def test_a_rating_read_before_any_match_is_the_base_rating() -> None:
    elo = EloRatings().fit(_matches(BASE_ROWS))
    assert elo.get_rating("A", dt.date(2020, 7, 31)) == pytest.approx(1500.0)


def test_elo_is_zero_sum_within_a_season() -> None:
    elo = EloRatings().fit(_matches(BASE_ROWS))
    after = dt.date(2021, 1, 1)
    total = elo.get_rating("A", after) + elo.get_rating("B", after)
    assert total == pytest.approx(3000.0)


def test_a_win_raises_the_winner_and_lowers_the_loser_by_the_same_amount() -> None:
    elo = EloRatings().fit(_matches([("2020-21", "A", "B", 2, 0)]))
    after = dt.date(2020, 8, 2)
    gain = elo.get_rating("A", after) - 1500.0
    loss = 1500.0 - elo.get_rating("B", after)
    assert gain > 0
    assert gain == pytest.approx(loss)


def test_the_home_side_gains_less_from_an_expected_win() -> None:
    # Home advantage is priced in, so beating an equal side at home is worth
    # less than beating it away.
    at_home = EloRatings().fit(_matches([("2020-21", "A", "B", 2, 0)]))
    away = EloRatings().fit(_matches([("2020-21", "B", "A", 0, 2)]))
    after = dt.date(2020, 8, 2)
    assert at_home.get_rating("A", after) < away.get_rating("A", after)


def test_get_rating_ignores_matches_played_later() -> None:
    """The critical test of prompt 3.1.

    Two fits on the same prefix of the calendar, one of which also ingests a
    wildly aberrant later match. Ratings read at a date before that match must
    be bit-for-bit identical.
    """
    prefix = BASE_ROWS[:2]
    aberrant = prefix + [("2020-21", "A", "B", 9, 0), ("2020-21", "A", "B", 9, 0)]

    short = EloRatings().fit(_matches(prefix))
    long = EloRatings().fit(_matches(aberrant))

    cutoff = dt.date(2020, 8, 9)  # after match 2, before match 3
    for team in ("A", "B"):
        assert short.get_rating(team, cutoff) == long.get_rating(team, cutoff)


def test_get_rating_is_strictly_before_the_query_date() -> None:
    """A rating read on a matchday must not contain that day's result."""
    rows = [("2020-21", "A", "B", 5, 0)]
    elo = EloRatings().fit(_matches(rows))
    match_day = dt.date(2020, 8, 1)
    assert elo.get_rating("A", match_day) == pytest.approx(1500.0)
    assert elo.get_rating("A", match_day + dt.timedelta(days=1)) > 1500.0


def test_ratings_regress_towards_the_mean_between_seasons() -> None:
    rows = [("2020-21", "A", "B", 3, 0), ("2021-22", "A", "B", 0, 0)]
    elo = EloRatings(regression=0.25).fit(_matches(rows))
    end_of_season = dt.date(2020, 8, 2)
    peak = elo.get_rating("A", end_of_season)
    start_of_next = elo.get_rating("A", dt.date(2020, 8, 8))
    assert 1500.0 < start_of_next < peak
    assert start_of_next == pytest.approx(1500.0 + 0.75 * (peak - 1500.0))


def test_no_regression_when_the_parameter_is_zero() -> None:
    rows = [("2020-21", "A", "B", 3, 0), ("2021-22", "A", "B", 0, 0)]
    elo = EloRatings(regression=0.0).fit(_matches(rows))
    assert elo.get_rating("A", dt.date(2020, 8, 8)) == pytest.approx(
        elo.get_rating("A", dt.date(2020, 8, 2))
    )


def test_pre_match_ratings_are_attached_to_every_row() -> None:
    frame = _matches(BASE_ROWS)
    annotated = EloRatings().fit(frame).with_pre_match_ratings(frame)
    assert annotated.height == frame.height
    assert annotated["elo_home_before"][0] == pytest.approx(1500.0)
    assert annotated["elo_away_before"][0] == pytest.approx(1500.0)
    # Row 2 must already reflect row 1 but nothing after it.
    assert annotated["elo_home_before"][1] != pytest.approx(1500.0)


def test_fit_refuses_matches_that_are_not_chronological() -> None:
    frame = _matches(BASE_ROWS).sort("date", descending=True)
    with pytest.raises(ValueError, match="chronological"):
        EloRatings().fit(frame)


def test_unplayed_fixtures_do_not_move_the_ratings() -> None:
    """A fixture with no score rides in the same frame and updates nothing."""
    played = _matches(BASE_ROWS)
    with_fixture = pl.concat(
        [
            played,
            pl.DataFrame(
                {
                    "date": [dt.date(2020, 9, 30)],
                    "season": ["2020-21"],
                    "home_team": ["A"],
                    "away_team": ["B"],
                    "home_goals": [None],
                    "away_goals": [None],
                },
                schema=played.schema,
            ),
        ]
    )
    after = dt.date(2020, 10, 31)
    played_only = EloRatings().fit(played)
    both = EloRatings().fit(with_fixture)
    for team in ("A", "B"):
        assert played_only.get_rating(team, after) == both.get_rating(team, after)
