"""Feature construction. The single entry point is build_features().

Two invariants hold by construction, and are pinned by tests/test_no_leakage.py:

1. A row's features use only matches played strictly before that row's date.
   Every rolling quantity is shifted by one before being averaged, so a match
   never sees its own statistics.
2. Nothing after `cutoff_date` is read at all. The frame is truncated first,
   so there is no future left to leak from. In production you pass today's
   date and the same code path serves live predictions.

Starting with six differential features. Adding one means adding its leakage
test at the same time.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from src.features.elo import EloRatings

REQUIRED_COLUMNS = (
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_shots_target",
    "away_shots_target",
)

IDENTIFIER_COLUMNS = ("date", "season", "home_team", "away_team")

WINDOW = 5

# What the model is allowed to read. Anything not listed here is context.
FEATURE_COLUMNS = (
    "elo_diff",
    "form_points_diff_5",
    "goals_scored_diff_5",
    "goals_conceded_diff_5",
    "shots_target_diff_5",
    "rest_days_diff",
)

# Per-side columns kept alongside the differentials: useful in phase 2bis and
# in the leakage tests, and cheap to carry.
_ROLLED = (
    ("form_points", "points"),
    ("goals_scored", "goals_for"),
    ("goals_conceded", "goals_against"),
    ("shots_target", "shots_target"),
)


def _team_match_table(matches: pl.DataFrame) -> pl.DataFrame:
    """One row per team per match, so rolling windows are a single pass."""
    goal_diff = pl.col("home_goals") - pl.col("away_goals")
    home = matches.select(
        "date",
        "season",
        pl.col("home_team").alias("team"),
        pl.col("home_goals").alias("goals_for"),
        pl.col("away_goals").alias("goals_against"),
        pl.col("home_shots_target").alias("shots_target"),
        pl.when(goal_diff > 0)
        .then(3)
        .when(goal_diff == 0)
        .then(1)
        .otherwise(0)
        .alias("points"),
    )
    away = matches.select(
        "date",
        "season",
        pl.col("away_team").alias("team"),
        pl.col("away_goals").alias("goals_for"),
        pl.col("home_goals").alias("goals_against"),
        pl.col("away_shots_target").alias("shots_target"),
        pl.when(goal_diff < 0)
        .then(3)
        .when(goal_diff == 0)
        .then(1)
        .otherwise(0)
        .alias("points"),
    )
    return pl.concat([home, away]).sort("team", "date")


def _rolling_history(matches: pl.DataFrame, window: int) -> pl.DataFrame:
    """Rolling means of the previous `window` matches, per team.

    The shift(1) is the whole safety mechanism: without it the average would
    include the match being predicted. min_samples equals the window on
    purpose, so a two-match average is never passed off as a five-match one;
    the resulting nulls are handled natively by LightGBM.
    """
    long = _team_match_table(matches)
    return long.select(
        "date",
        "team",
        *[
            pl.col(source)
            .shift(1)
            .rolling_mean(window_size=window, min_samples=window)
            .over("team")
            .alias(f"{name}_{window}")
            for name, source in _ROLLED
        ],
        (pl.col("date") - pl.col("date").shift(1))
        .over("team")
        .dt.total_days()
        .cast(pl.Float64)
        .alias("rest_days"),
    )


def build_features(
    matches: pl.DataFrame,
    cutoff_date: dt.date,
    window: int = WINDOW,
    elo: EloRatings | None = None,
) -> pl.DataFrame:
    """Feature rows for every match up to `cutoff_date`, inclusive.

    Nothing after the cutoff is read, and no row reads its own match.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    if not matches["date"].is_sorted():
        raise ValueError("matches must be in chronological order")

    # Hard wall: drop the future before touching anything else.
    known = matches.filter(pl.col("date") <= cutoff_date)
    if known.is_empty():
        raise ValueError(f"cutoff {cutoff_date} is before the first match")

    ratings = (elo or EloRatings()).fit(known)
    with_elo = ratings.with_pre_match_ratings(known)

    history = _rolling_history(known, window)
    rolled_columns = [f"{name}_{window}" for name, _ in _ROLLED] + ["rest_days"]

    for side, team_column in (("home", "home_team"), ("away", "away_team")):
        with_elo = with_elo.join(
            history.rename(
                {c: f"{c}_{side}" for c in rolled_columns} | {"team": team_column}
            ),
            on=["date", team_column],
            how="left",
        )

    context = [c for c in ("league", "result") if c in with_elo.columns]
    return with_elo.select(
        *IDENTIFIER_COLUMNS,
        *context,
        *[f"{c}_{side}" for c in rolled_columns for side in ("home", "away")],
        "elo_home_before",
        "elo_away_before",
        # Elo already carries the home edge, so the differential includes it.
        (
            pl.col("elo_home_before")
            + ratings.home_advantage
            - pl.col("elo_away_before")
        ).alias("elo_diff"),
        *[
            (pl.col(f"{name}_{window}_home") - pl.col(f"{name}_{window}_away")).alias(
                f"{name}_diff_{window}"
            )
            for name, _ in _ROLLED
        ],
        (pl.col("rest_days_home") - pl.col("rest_days_away")).alias("rest_days_diff"),
    ).sort("date", "home_team")
