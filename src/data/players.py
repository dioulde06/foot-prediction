"""Player-season stats from Understat, for the probable-scorers estimate.

Understat aggregates a player's season: minutes, goals and xG (both with the
penalties taken out), shots. One row per player, team and season. The team
names go through the explicit mapping like every other source: a spelling the
mapping does not know stops the pipeline instead of dropping a club's strikers
on the floor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.data.fetch import UNDERSTAT_LEAGUES, season_code, season_label
from src.data.team_mapping import to_canonical

LOG = logging.getLogger(__name__)

PLAYERS_PARQUET = Path("data/raw/understat_players.parquet")

RAW_COLUMNS = (
    "league",
    "season",
    "team",
    "player",
    "player_id",
    "position",
    "matches",
    "minutes",
    "np_goals",
    "np_xg",
    "shots",
)
NEVER_NULL = ("team", "player", "player_id", "minutes", "np_goals", "np_xg")


def normalise_players(
    raw: pl.DataFrame, known: frozenset[str] | None = None
) -> pl.DataFrame:
    """Canonical team names, season labels, fixed columns. Pure, so testable.

    `known` restricts the teams to a given set. Publication passes None: a
    promoted club with no fixture this week must not block the pipeline, and a
    club that *does* play is checked downstream, where its absence is an error.
    """
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Understat player stats are missing columns {missing}")
    frame = raw.select(list(RAW_COLUMNS))
    for column in NEVER_NULL:
        nulls = frame[column].null_count()
        if nulls:
            raise ValueError(f"{nulls} null {column!r} rows in Understat player stats")

    teams = {
        name: to_canonical(name, "understat", known) for name in frame["team"].unique()
    }
    codes = {
        season_code(year): season_label(year)
        for year in range(2000, 2100)
        if season_code(year) in set(frame["season"].unique())
    }
    return frame.with_columns(
        pl.col("team").replace_strict(teams),
        pl.col("league").replace_strict(UNDERSTAT_LEAGUES),
        pl.col("season").replace_strict(codes),
        pl.col("player_id").cast(pl.Int64),
        pl.col("matches").cast(pl.Int64),
        pl.col("minutes").cast(pl.Int64),
        pl.col("np_goals").cast(pl.Int64),
        pl.col("np_xg").cast(pl.Float64),
        pl.col("shots").cast(pl.Int64),
    ).sort("league", "season", "team", "player")


def fetch_players(
    first_years: list[int], known: frozenset[str] | None = None, *, force: bool = False
) -> pl.DataFrame:
    """Player-season stats for the given seasons, cached in one parquet.

    The current season changes every week, so callers that publish pass
    `force=True`; analysis reads the cache.
    """
    if PLAYERS_PARQUET.exists() and not force:
        LOG.info("cache hit, reading %s", PLAYERS_PARQUET)
        return pl.read_parquet(PLAYERS_PARQUET)

    import soccerdata as sd  # heavy import, only needed on a real fetch

    reader = sd.Understat(
        leagues=list(UNDERSTAT_LEAGUES), seasons=[season_code(y) for y in first_years]
    )
    raw = pl.from_pandas(
        reader.read_player_season_stats(force_cache=False).reset_index()
    )
    players = normalise_players(raw, known)
    PLAYERS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    players.write_parquet(PLAYERS_PARQUET)
    LOG.info(
        "%d player-seasons over %d seasons written to %s",
        players.height,
        len(first_years),
        PLAYERS_PARQUET,
    )
    return players
