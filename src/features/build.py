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
import warnings

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
    "home_np_xg",
    "away_np_xg",
)

IDENTIFIER_COLUMNS = ("date", "season", "home_team", "away_team")

WINDOW = 5

# Beyond two weeks the gap stops measuring fatigue and starts encoding "this
# team was relegated or absent": the raw maximum in the training seasons is 811
# days. Capping keeps the fatigue signal and drops the parasite one.
MAX_REST_DAYS = 14.0

# What the model is allowed to read. Anything not listed here is context.
FEATURE_COLUMNS = (
    "elo_diff",
    "form_points_diff_5",
    "goals_scored_diff_5",
    "goals_conceded_diff_5",
    "shots_target_diff_5",
    "np_xg_created_diff_5",
    "np_xg_conceded_diff_5",
    "rest_days_diff",
)

# Per-side columns kept alongside the differentials: useful in phase 2bis and
# in the leakage tests, and cheap to carry.
_ROLLED = (
    ("form_points", "points"),
    ("goals_scored", "goals_for"),
    ("goals_conceded", "goals_against"),
    ("shots_target", "shots_target"),
    # Non-penalty xG: the same underlying strength as goals, measured with far
    # less variance, and the only genuinely new dimension in the dataset.
    ("np_xg_created", "np_xg_for"),
    ("np_xg_conceded", "np_xg_against"),
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
        pl.col("home_np_xg").alias("np_xg_for"),
        pl.col("away_np_xg").alias("np_xg_against"),
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
        pl.col("away_np_xg").alias("np_xg_for"),
        pl.col("home_np_xg").alias("np_xg_against"),
        pl.when(goal_diff < 0)
        .then(3)
        .when(goal_diff == 0)
        .then(1)
        .otherwise(0)
        .alias("points"),
    )
    return pl.concat([home, away]).sort("team", "date")


def _rolling_history(matches: pl.DataFrame, window: int) -> pl.DataFrame:
    """Rolling means of the previous `window` *played* matches, per team.

    Two rules make this leak-free and fixture-proof at once. Windows are
    computed on played matches only, so an unplayed fixture in the frame never
    empties the window of the rows after it: a prediction a month ahead reads
    the last five results known today. And every row, played or not, takes the
    window as it stood *strictly before its date*, so a match never reads its
    own result. min_samples equals the window on purpose: a two-match average
    is never passed off as a five-match one; LightGBM handles the nulls.
    """
    long = _team_match_table(matches)
    played = long.filter(pl.col("goals_for").is_not_null())
    after_match = played.select(
        "team",
        pl.col("date").alias("played_date"),
        *[
            pl.col(source)
            .rolling_mean(window_size=window, min_samples=window)
            .over("team")
            .alias(f"{name}_{window}")
            for name, source in _ROLLED
        ],
    ).sort("played_date")
    # A team plays at most once a day, so "the day before" is "strictly before".
    rows = (
        long.select("date", "team")
        .with_columns((pl.col("date") - pl.duration(days=1)).alias("as_of"))
        .sort("as_of")
    )
    with warnings.catch_warnings():
        # polars cannot verify sortedness within `by` groups; both frames are
        # sorted on their asof key just above.
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        joined = rows.join_asof(
            after_match,
            left_on="as_of",
            right_on="played_date",
            by="team",
            strategy="backward",
        )
    return joined.select(
        "date",
        "team",
        *[f"{name}_{window}" for name, _ in _ROLLED],
        (pl.col("date") - pl.col("played_date"))
        .dt.total_days()
        .cast(pl.Float64)
        .clip(upper_bound=MAX_REST_DAYS)
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
