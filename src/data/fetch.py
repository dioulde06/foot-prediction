"""Ingestion of raw match data, one parquet per source in data/raw/.

football-data.co.uk ships one CSV per league and season holding results,
basic match stats and bookmaker odds. It is self-sufficient for the three
baselines and for the Elo / rolling features; FBref and Understat only add
xG on top of it, and are deferred (see the stubs at the bottom).

Closing odds are used, never opening odds: the closing line is the sharpest
price the market produced and is the only honest `market` baseline. Three
closing quotes are kept: the market average (full coverage, the consensus),
Bet365 (full coverage, a single book) and Pinnacle (sharpest, but only ~50%
covered on 2025-26, so unusable alone on the test season).
"""

from __future__ import annotations

import argparse
import logging
import time
import urllib.request
from pathlib import Path

import polars as pl

LOG = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
FOOTBALL_DATA_PARQUET = RAW_DIR / "football_data.parquet"

BASE_URL = "https://www.football-data.co.uk/mmz4281"
USER_AGENT = "foot-prediction/0.1 (research; contact via github.com/dioulde06)"
REQUEST_DELAY_S = 1.0

# First calendar year of each ingested season: 2020 means season 2020-21.
FIRST_SEASON = 2020
LAST_SEASON = 2025

LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

# Source column -> canonical column. Anything else in the CSV is dropped.
COLUMNS: dict[str, str] = {
    "Div": "league_code",
    "Date": "date",
    "Time": "kickoff_time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "PSCH": "odds_close_ps_h",
    "PSCD": "odds_close_ps_d",
    "PSCA": "odds_close_ps_a",
    "B365CH": "odds_close_b365_h",
    "B365CD": "odds_close_b365_d",
    "B365CA": "odds_close_b365_a",
    "AvgCH": "odds_close_avg_h",
    "AvgCD": "odds_close_avg_d",
    "AvgCA": "odds_close_avg_a",
}

INT_COLUMNS = (
    "home_goals",
    "away_goals",
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_corners",
    "away_corners",
)
ODDS_COLUMNS = (
    "odds_close_ps_h",
    "odds_close_ps_d",
    "odds_close_ps_a",
    "odds_close_b365_h",
    "odds_close_b365_d",
    "odds_close_b365_a",
    "odds_close_avg_h",
    "odds_close_avg_d",
    "odds_close_avg_a",
)
# Nulls here mean the row is unusable, so they must abort the ingestion.
REQUIRED_COLUMNS = ("date", "home_team", "away_team", "home_goals", "away_goals", "result")


def season_code(first_year: int) -> str:
    """URL fragment football-data uses for a season: 2020 -> '2021'."""
    return f"{first_year % 100:02d}{(first_year + 1) % 100:02d}"


def season_label(first_year: int) -> str:
    """Human season label: 2020 -> '2020-21'."""
    return f"{first_year}-{(first_year + 1) % 100:02d}"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return bytes(response.read())


def _as_nullable_str(name: str) -> pl.Expr:
    """Trim the column and turn empty cells into nulls, before any cast."""
    trimmed = pl.col(name).str.strip_chars()
    return pl.when(trimmed == "").then(None).otherwise(trimmed).alias(name)


def _parse_csv(raw: bytes, league_code: str, first_year: int) -> pl.DataFrame:
    if raw.startswith(b"\xef\xbb\xbf"):  # football-data ships a UTF-8 BOM
        raw = raw[3:]

    # Every column is read as text, then cast explicitly: the CSVs are
    # heterogeneous across seasons and schema inference hides that.
    frame = pl.read_csv(raw, infer_schema_length=0, encoding="utf8-lossy")

    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{league_code} {season_label(first_year)}: missing {missing}")

    # Trailing blank lines are common; a real row always carries the div code.
    total_rows = frame.height
    frame = frame.filter(pl.col("Div").str.strip_chars() == league_code)
    dropped = total_rows - frame.height
    if dropped:
        LOG.info("%s %s: dropped %d non-match rows", league_code, season_label(first_year), dropped)

    frame = frame.select(list(COLUMNS)).rename(COLUMNS)
    frame = frame.with_columns([_as_nullable_str(c) for c in frame.columns])

    # Both date layouts appear across seasons. Picking by length rather than
    # by trying one format then the other: a lenient "%d/%m/%Y" silently reads
    # "16/08/25" as year 25 instead of failing, which corrupts the dates.
    frame = frame.with_columns(
        pl.when(pl.col("date").str.len_chars() == 10)
        .then(pl.col("date").str.to_date("%d/%m/%Y"))
        .otherwise(pl.col("date").str.to_date("%d/%m/%y"))
        .alias("date"),
        *[pl.col(c).cast(pl.Int64) for c in INT_COLUMNS],
        *[pl.col(c).cast(pl.Float64) for c in ODDS_COLUMNS],
    )

    return frame.with_columns(
        pl.lit(season_label(first_year)).alias("season"),
        pl.lit(LEAGUES[league_code]).alias("league"),
    )


def _check_integrity(frame: pl.DataFrame) -> None:
    """Abort on anything that would silently corrupt the dataset downstream."""
    for column in REQUIRED_COLUMNS:
        null_count = frame[column].null_count()
        if null_count:
            offenders = frame.filter(pl.col(column).is_null()).head(5)
            raise ValueError(f"{null_count} null {column!r} rows, e.g.\n{offenders}")

    key = ("date", "home_team", "away_team")
    duplicates = frame.filter(frame.select(key).is_duplicated())
    if duplicates.height:
        raise ValueError(f"duplicate (date, home, away) keys:\n{duplicates}")

    if not frame["result"].is_in(["H", "D", "A"]).all():
        bad = frame.filter(~pl.col("result").is_in(["H", "D", "A"]))
        raise ValueError(f"unexpected result codes:\n{bad}")


def fetch_football_data(*, force: bool = False) -> pl.DataFrame:
    """Download results and closing odds for every league and season.

    Cached: returns the existing parquet untouched unless `force` is set.
    """
    if FOOTBALL_DATA_PARQUET.exists() and not force:
        LOG.info("cache hit, reading %s", FOOTBALL_DATA_PARQUET)
        return pl.read_parquet(FOOTBALL_DATA_PARQUET)

    frames: list[pl.DataFrame] = []
    for first_year in range(FIRST_SEASON, LAST_SEASON + 1):
        for league_code in LEAGUES:
            url = f"{BASE_URL}/{season_code(first_year)}/{league_code}.csv"
            frame = _parse_csv(_download(url), league_code, first_year)
            LOG.info(
                "%-3s %s: %3d matches", league_code, season_label(first_year), frame.height
            )
            frames.append(frame)
            time.sleep(REQUEST_DELAY_S)

    matches = pl.concat(frames).sort("date", "home_team")
    _check_integrity(matches)

    for column in ODDS_COLUMNS:
        coverage = 1 - matches[column].null_count() / matches.height
        LOG.info("%s coverage: %.1f%%", column, 100 * coverage)
    LOG.info("total: %d matches, %d teams", matches.height, matches["home_team"].n_unique())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    matches.write_parquet(FOOTBALL_DATA_PARQUET)
    LOG.info("wrote %s", FOOTBALL_DATA_PARQUET)
    return matches


def fetch_fbref(*, force: bool = False) -> pl.DataFrame:
    """Match results and shooting stats from FBref, via soccerdata.

    Deferred on purpose: scraping 6 seasons x 5 leagues blows past the 2 min
    budget in CLAUDE.md and needs an explicit go-ahead first.
    """
    raise NotImplementedError("fetch_fbref: awaiting go-ahead, see PLAN.md phase 1")


def fetch_understat(*, force: bool = False) -> pl.DataFrame:
    """Per-match xG from Understat, via soccerdata. Deferred, same reason."""
    raise NotImplementedError("fetch_understat: awaiting go-ahead, see PLAN.md phase 1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="refetch even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_football_data(force=args.force)


if __name__ == "__main__":
    main()
