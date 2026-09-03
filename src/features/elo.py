"""Elo ratings adapted to football.

The whole point of this module is the guarantee that `get_rating(team, date)`
returns the state of the world strictly before `date`. A rating that quietly
includes the match it is used to predict is the most damaging leak in this
project: it produces plausible ratings and a beautiful log-loss.
"""

from __future__ import annotations

import bisect
import datetime as dt
from typing import Any

import polars as pl

REQUIRED_COLUMNS = (
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)


class EloRatings:
    """Classic Elo, zero-sum, with a home advantage and a between-season reset.

    Parameters
    ----------
    k:
        Update step. 20 is the usual football value: high enough to track form,
        low enough not to overreact to one result.
    home_advantage:
        Added to the home rating when computing the expected score, in Elo
        points. 65 points is roughly the measured home edge in the big five.
    regression:
        Share of the gap to the mean given back between seasons. Handles
        promoted and relegated sides, whose true strength changes abruptly.
    base:
        Rating of a team never seen before.
    """

    def __init__(
        self,
        k: float = 20.0,
        home_advantage: float = 65.0,
        regression: float = 0.25,
        base: float = 1500.0,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not 0.0 <= regression <= 1.0:
            raise ValueError(f"regression must be in [0, 1], got {regression}")
        self.k = k
        self.home_advantage = home_advantage
        self.regression = regression
        self.base = base

        self._ratings: dict[str, float] = {}
        # Per team, two parallel sorted lists: when a rating took effect, and
        # what it became. Queried by bisection, never scanned.
        self._dates: dict[str, list[dt.date]] = {}
        self._values: dict[str, list[float]] = {}

    def fit(self, matches: pl.DataFrame) -> EloRatings:
        """Consume the matches in chronological order, once."""
        missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
        if missing:
            raise ValueError(f"missing columns {missing}")
        if not matches["date"].is_sorted():
            raise ValueError("matches must be in chronological order before fitting")

        season: str | None = None
        for row in matches.iter_rows(named=True):
            # The season change is handled before anything else, so that the
            # first fixture of a new season gets regressed ratings even though
            # it has not been played. Getting this order wrong means the first
            # matchday of a season is predicted with last season's ratings.
            if season is not None and row["season"] != season:
                # Effective the day before the new season opens, so that a
                # rating read on the first matchday already includes it.
                self._regress(row["date"] - dt.timedelta(days=1))
            season = row["season"]

            # Unplayed fixtures ride along in the same frame so that features
            # for upcoming matches go through exactly the same code path as
            # training ones. They carry no result, so they update nothing.
            if row["home_goals"] is None or row["away_goals"] is None:
                continue
            self._play(row)
        return self

    def get_rating(self, team: str, date: dt.date) -> float:
        """Rating of `team` using only what happened strictly before `date`."""
        dates = self._dates.get(team)
        if not dates:
            return self.base
        index = bisect.bisect_left(dates, date)
        if index == 0:
            return self.base
        return self._values[team][index - 1]

    def with_pre_match_ratings(self, matches: pl.DataFrame) -> pl.DataFrame:
        """Attach each side's pre-match rating to every row."""
        home = [
            self.get_rating(team, date)
            for team, date in zip(matches["home_team"], matches["date"], strict=True)
        ]
        away = [
            self.get_rating(team, date)
            for team, date in zip(matches["away_team"], matches["date"], strict=True)
        ]
        return matches.with_columns(
            pl.Series("elo_home_before", home),
            pl.Series("elo_away_before", away),
        )

    def expected_home_score(self, home_rating: float, away_rating: float) -> float:
        """Expected points share of the home side, draws counting as a half."""
        gap = home_rating + self.home_advantage - away_rating
        return float(1.0 / (1.0 + 10.0 ** (-gap / 400.0)))

    def _play(self, row: dict[str, Any]) -> None:
        home, away, date = row["home_team"], row["away_team"], row["date"]
        home_rating = self._ratings.get(home, self.base)
        away_rating = self._ratings.get(away, self.base)

        if row["home_goals"] > row["away_goals"]:
            actual = 1.0
        elif row["home_goals"] == row["away_goals"]:
            actual = 0.5
        else:
            actual = 0.0

        delta = self.k * (actual - self.expected_home_score(home_rating, away_rating))
        self._record(home, home_rating + delta, date)
        self._record(away, away_rating - delta, date)

    def _regress(self, effective: dt.date) -> None:
        if self.regression == 0.0:
            return
        for team, rating in list(self._ratings.items()):
            self._record(
                team,
                self.base + (1.0 - self.regression) * (rating - self.base),
                effective,
            )

    def _record(self, team: str, rating: float, date: dt.date) -> None:
        self._ratings[team] = rating
        dates = self._dates.setdefault(team, [])
        if dates and date < dates[-1]:
            raise ValueError(
                f"out-of-order rating for {team}: {date} after {dates[-1]}"
            )
        dates.append(date)
        self._values.setdefault(team, []).append(rating)
