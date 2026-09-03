"""Probable scorers and goal markets for an upcoming fixture.

Two layers, both deliberately simple:

1. **The team.** Non-penalty expected goals of each side, the geometric mean
   of what the attack creates and what the opposing defence concedes over the
   rolling window build_features already computes, split by the dataset's
   home factor. Penalties come back as a flat league rate. Goals are Poisson.
2. **The player.** His non-penalty xG per 90 minutes, pooled over the seasons
   available and shrunk toward the league rate for thin samples, then scaled
   by how favourable this match is relative to his team's average. The chance
   he scores at least once is 1 - exp(-that).

The estimate assumes the player starts and plays 90 minutes, and knows
nothing of injuries, suspensions or rotation. It is frozen at publication in
an append-only file so it can be checked afterwards, and it is *not* validated
the way the 1X2 probabilities are. The page says so.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

LOG = logging.getLogger(__name__)

SCORERS_PARQUET = Path("predictions/scorers.parquet")
# Players with fewer minutes than this over the pooled seasons are not shown:
# their rate is mostly the prior, which says nothing about them.
MIN_MINUTES = 900
# Weight of the league prior, in minutes: a full 8 matches.
PRIOR_MINUTES = 720
TOP_N = 5
WINDOW = 5

TEAM_COLUMNS = (
    "elo",
    "form_points_5",
    "goals_scored_5",
    "goals_conceded_5",
    "np_xg_created_5",
    "np_xg_conceded_5",
)
SCORERS_SCHEMA: dict[str, pl.DataType] = {
    "published_at": pl.Datetime("us"),
    "date": pl.Date(),
    "home_team": pl.String(),
    "away_team": pl.String(),
    "side": pl.String(),
    "team": pl.String(),
    "player": pl.String(),
    "position": pl.String(),
    "minutes": pl.Int64(),
    "np_goals": pl.Int64(),
    "np_xg": pl.Float64(),
    "shots": pl.Int64(),
    "rate": pl.Float64(),
    "p_scores": pl.Float64(),
    "lambda_np": pl.Float64(),
    "lambda_total": pl.Float64(),
    **{c: pl.Float64() for c in TEAM_COLUMNS},
}


@dataclass(frozen=True)
class GoalsPrior:
    """League-level constants, measured on the dataset rather than assumed."""

    home_factor: float  # mean home np_xg / mean away np_xg
    pen_xg: float  # penalty xG per team and match
    player_rate: float  # league np_xg per 90 minutes of an outfield player

    @classmethod
    def from_data(cls, matches: pl.DataFrame, players: pl.DataFrame) -> GoalsPrior:
        home = float(matches["home_np_xg"].to_numpy().mean())
        away = float(matches["away_np_xg"].to_numpy().mean())
        pens = float(
            (matches["home_xg"] - matches["home_np_xg"]).to_numpy().sum()
            + (matches["away_xg"] - matches["away_np_xg"]).to_numpy().sum()
        ) / (2 * matches.height)
        outfield = players.filter(~pl.col("position").str.contains("GK"))
        rate = 90.0 * float(outfield["np_xg"].to_numpy().sum())
        rate /= float(outfield["minutes"].to_numpy().sum())
        return cls(home_factor=home / away, pen_xg=pens, player_rate=rate)


def player_rates(
    players: pl.DataFrame, prior: GoalsPrior, min_minutes: int = MIN_MINUTES
) -> pl.DataFrame:
    """One row per player: pooled minutes and xG, current team, shrunk rate.

    A player is listed only if he has played in the most recent season on
    file: someone who left the five leagues has no row there and must not be
    shown for last season's club. Before the first matchday the most recent
    season is the previous one, so everybody still counts.
    """
    active = players.filter(pl.col("minutes") > 0)
    current_season = active["season"].max()
    latest = (
        active.filter(pl.col("season") == current_season)
        .group_by("player_id")
        .agg(pl.col("team").last(), pl.col("position").last(), pl.col("player").last())
    )
    pooled = players.group_by("player_id").agg(
        pl.col("minutes").sum(),
        pl.col("np_goals").sum(),
        pl.col("np_xg").sum(),
        pl.col("shots").sum(),
    )
    prior_xg = prior.player_rate * PRIOR_MINUTES / 90.0
    return (
        pooled.join(latest, on="player_id", how="inner")
        .filter(
            (pl.col("minutes") >= min_minutes) & ~pl.col("position").str.contains("GK")
        )
        .with_columns(
            (
                90.0
                * (pl.col("np_xg") + prior_xg)
                / (pl.col("minutes") + PRIOR_MINUTES)
            ).alias("rate")
        )
        .sort("rate", descending=True)
    )


def team_average(players: pl.DataFrame) -> dict[str, float]:
    """Non-penalty xG per match of each team, from its players' totals."""
    table = (
        players.group_by("team", "season")
        .agg(pl.col("np_xg").sum().alias("xg"), pl.col("matches").max().alias("n"))
        .group_by("team")
        .agg(pl.col("xg").sum(), pl.col("n").sum())
        .filter(pl.col("n") > 0)
    )
    return {r["team"]: r["xg"] / r["n"] for r in table.iter_rows(named=True)}


def expected_goals(
    row: dict[str, Any], prior: GoalsPrior
) -> tuple[float, float] | None:
    """Non-penalty expected goals (home, away) for one feature row.

    None when a side has no full rolling window yet, which is the case of a
    promoted club in its first weeks: no estimate beats a made-up one.
    """
    inputs = [
        row[f"np_xg_{kind}_{WINDOW}_{side}"]
        for kind in ("created", "conceded")
        for side in ("home", "away")
    ]
    if any(v is None for v in inputs):
        return None
    home = math.sqrt(
        row[f"np_xg_created_{WINDOW}_home"] * row[f"np_xg_conceded_{WINDOW}_away"]
    )
    away = math.sqrt(
        row[f"np_xg_created_{WINDOW}_away"] * row[f"np_xg_conceded_{WINDOW}_home"]
    )
    split = math.sqrt(prior.home_factor)
    return home * split, away / split


def goal_markets(lam_home: float, lam_away: float) -> dict[str, float]:
    """Both teams score, and over 2.5, under independent Poissons."""
    total = lam_home + lam_away
    under = math.exp(-total) * (1 + total + total**2 / 2)
    return {
        "btts": (1 - math.exp(-lam_home)) * (1 - math.exp(-lam_away)),
        "over25": 1 - under,
    }


def scorers_for_side(
    rates: pl.DataFrame, team: str, lam_np: float, team_avg: float, top: int = TOP_N
) -> pl.DataFrame:
    """The team's most likely scorers in a match worth `lam_np` to it."""
    scale = lam_np / team_avg
    return (
        rates.filter(pl.col("team") == team)
        .with_columns((1 - (-(pl.col("rate") * scale)).exp()).alias("p_scores"))
        .sort("p_scores", descending=True)
        .head(top)
    )


def scorers_for_fixtures(
    features: pl.DataFrame, players: pl.DataFrame, prior: GoalsPrior
) -> pl.DataFrame:
    """Scorer rows for every fixture in `features` (the upcoming_features frame)."""
    rates = player_rates(players, prior)
    averages = team_average(players)
    frames = []
    for row in features.iter_rows(named=True):
        lams = expected_goals(row, prior)
        if lams is None:
            LOG.info(
                "%s - %s: a side has fewer than %d matches of history, no scorers",
                row["home_team"],
                row["away_team"],
                WINDOW,
            )
            continue
        lam_home, lam_away = lams
        for side, lam in (("home", lam_home), ("away", lam_away)):
            team = row[f"{side}_team"]
            if team not in averages:
                raise KeyError(
                    f"{team!r} has no Understat player data: either the club is "
                    "missing from the fetched seasons, or its Understat spelling "
                    "is not in UNDERSTAT_TO_CANONICAL"
                )
            table = scorers_for_side(rates, team, lam, averages[team])
            if table.is_empty():
                LOG.warning("%s: no player above %d minutes", team, MIN_MINUTES)
                continue
            frames.append(
                table.select(
                    "team",
                    "player",
                    "position",
                    "minutes",
                    "np_goals",
                    "np_xg",
                    "shots",
                    "rate",
                    "p_scores",
                ).with_columns(
                    pl.lit(row["date"]).alias("date"),
                    pl.lit(row["home_team"]).alias("home_team"),
                    pl.lit(row["away_team"]).alias("away_team"),
                    pl.lit(side).alias("side"),
                    pl.lit(lam).alias("lambda_np"),
                    pl.lit(lam + prior.pen_xg).alias("lambda_total"),
                    pl.lit(float(row[f"elo_{side}_before"])).alias("elo"),
                    *[
                        pl.lit(float(row[f"{c}_{side}"])).alias(c)
                        for c in TEAM_COLUMNS
                        if c != "elo"
                    ],
                )
            )
    if not frames:
        return pl.DataFrame(
            schema={k: v for k, v in SCORERS_SCHEMA.items() if k != "published_at"}
        )
    return pl.concat(frames)


def append_scorers(rows: pl.DataFrame, published_at: dt.datetime) -> pl.DataFrame:
    """Freeze the estimate for the given fixtures at this publication."""
    SCORERS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    rows = rows.with_columns(
        pl.lit(published_at).cast(pl.Datetime("us")).alias("published_at")
    ).select(list(SCORERS_SCHEMA))
    if SCORERS_PARQUET.exists():
        history = pl.read_parquet(SCORERS_PARQUET)
        # One freeze per publication of the match: the caller only passes the
        # fixtures whose prediction was appended this run.
        already = history.select("date", "home_team", "away_team", "published_at")
        rows = rows.join(already.unique(), on=list(already.columns), how="anti")
        if rows.is_empty():
            return history
        combined = pl.concat([history, rows])
    else:
        combined = rows
    combined.write_parquet(SCORERS_PARQUET)
    LOG.info("%d lignes buteurs ajoutees dans %s", rows.height, SCORERS_PARQUET)
    return combined
