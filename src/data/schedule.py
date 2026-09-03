"""The season schedule from Understat: every fixture, played or not, with its
kick-off in UTC.

football-data's fixtures file only lists the coming days. Understat publishes
the whole season as soon as the league does, which is what a month-ahead view
needs. Kick-offs are kept in UTC and localised by the browser. Team names go
through the explicit mapping like every other source.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import polars as pl

from src.data.fetch import UNDERSTAT_LEAGUES, season_code, season_label
from src.data.team_mapping import to_canonical

LOG = logging.getLogger(__name__)

SCHEDULE_PARQUET = Path("data/raw/understat_schedule.parquet")
RAW_COLUMNS = ("league", "season", "date", "home_team", "away_team", "is_result")


def normalise_schedule(
    raw: pl.DataFrame, known: frozenset[str] | None = None
) -> pl.DataFrame:
    """Canonical names, season label, UTC kick-off and its date. Pure."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Understat schedule is missing columns {missing}")
    frame = raw.select(list(RAW_COLUMNS))
    for column in ("date", "home_team", "away_team"):
        nulls = frame[column].null_count()
        if nulls:
            raise ValueError(f"{nulls} null {column!r} rows in the Understat schedule")
    names = set(frame["home_team"].unique()) | set(frame["away_team"].unique())
    teams = {name: to_canonical(name, "understat", known) for name in names}
    codes = {
        season_code(year): season_label(year)
        for year in range(2000, 2100)
        if season_code(year) in set(frame["season"].unique())
    }
    return frame.select(
        pl.col("league").replace_strict(UNDERSTAT_LEAGUES),
        pl.col("season").replace_strict(codes),
        pl.col("date").cast(pl.Datetime("us")).alias("kickoff_utc"),
        pl.col("date").cast(pl.Datetime("us")).dt.date().alias("date"),
        pl.col("home_team").replace_strict(teams),
        pl.col("away_team").replace_strict(teams),
        pl.col("is_result").cast(pl.Boolean).alias("played"),
    ).sort("kickoff_utc", "home_team")


def fetch_schedule(
    first_year: int, known: frozenset[str] | None = None
) -> pl.DataFrame:
    """The current season's schedule, always refetched: dates get fixed late."""
    import soccerdata as sd  # heavy import, only needed on a real fetch

    reader = sd.Understat(
        leagues=list(UNDERSTAT_LEAGUES),
        seasons=[season_code(first_year)],
        no_cache=True,
    )
    raw = pl.from_pandas(reader.read_schedule().reset_index())
    schedule = normalise_schedule(raw, known)
    SCHEDULE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    schedule.write_parquet(SCHEDULE_PARQUET)
    LOG.info(
        "%d fixtures in the %s schedule, %d played, written to %s",
        schedule.height,
        season_label(first_year),
        int(schedule["played"].sum()),
        SCHEDULE_PARQUET,
    )
    return schedule


def upcoming(schedule: pl.DataFrame, today: dt.date, horizon_days: int) -> pl.DataFrame:
    """Unplayed fixtures from today to today + horizon, as the publisher wants them."""
    return (
        schedule.filter(
            ~pl.col("played")
            & (pl.col("date") >= today)
            & (pl.col("date") <= today + dt.timedelta(days=horizon_days))
        )
        .select("date", "league", "home_team", "away_team", "kickoff_utc")
        .sort("kickoff_utc", "home_team")
    )
