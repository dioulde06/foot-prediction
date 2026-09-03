"""Join the sources into data/processed/matches.parquet.

The join key is (season, home_team, away_team) after translating every name to
canonical form -- not the date. In a double round-robin a given pair meets once
at each ground per season, so that triple is exactly unique in both sources
(verified: zero duplicates), while the dates disagree by a day or two on 18
matches out of 10 734 because of late kickoffs and postponements. Joining on
the season key is exact; joining on the date would need a tolerance window,
which is the date equivalent of fuzzy name matching and is forbidden here.

The join rate is logged and enforced: a silent drop of a few hundred matches
would quietly shrink the dataset and change every result.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from src.data.fetch import (
    LAST_SEASON,
    fetch_football_data,
    fetch_understat,
    season_label,
)
from src.data.team_mapping import to_canonical

LOG = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MATCHES_PARQUET = PROCESSED_DIR / "matches.parquet"

# Below this the join is broken, not imperfect.
MIN_JOIN_RATE = 0.98

UNDERSTAT_FEATURES = (
    "home_xg",
    "away_xg",
    "home_np_xg",
    "away_np_xg",
    "home_ppda",
    "away_ppda",
    "home_deep",
    "away_deep",
)


def canonicalise(
    frame: pl.DataFrame, source: str, known: frozenset[str]
) -> pl.DataFrame:
    """Translate both team columns of `frame` to canonical names."""
    return frame.with_columns(
        [
            pl.col(column)
            .map_elements(
                lambda name: to_canonical(name, source, known=known),
                return_dtype=pl.String,
            )
            .alias(column)
            for column in ("home_team", "away_team")
        ]
    )


def merge_sources(*, force: bool = False) -> pl.DataFrame:
    """football-data as the spine, Understat xG joined on top."""
    if MATCHES_PARQUET.exists() and not force:
        LOG.info("cache hit, reading %s", MATCHES_PARQUET)
        return pl.read_parquet(MATCHES_PARQUET)

    spine = fetch_football_data()
    understat = fetch_understat()

    known = frozenset(spine["home_team"].unique()) | frozenset(
        spine["away_team"].unique()
    )
    LOG.info("%d canonical team names in the spine", len(known))
    understat = canonicalise(understat, "understat", known)

    key = ["season", "home_team", "away_team"]
    merged = spine.join(understat.select(*key, *UNDERSTAT_FEATURES), on=key, how="left")

    joined = merged.height - merged["home_xg"].null_count()
    rate = joined / merged.height
    LOG.info("join rate: %d / %d = %.2f %%", joined, merged.height, 100 * rate)
    if rate < MIN_JOIN_RATE:
        unmatched = merged.filter(pl.col("home_xg").is_null()).select(
            "date", "league", "season", "home_team", "away_team"
        )
        raise ValueError(
            f"join rate {100 * rate:.2f} % is below the "
            f"{100 * MIN_JOIN_RATE:.0f} % floor; {unmatched.height} matches "
            f"unmatched, first rows:\n{unmatched.head(15)}"
        )

    # Understat publishes a matchday's xG a little after football-data posts the
    # score. A current-season match without xG is not wrong, it is early: it is
    # left out today and comes in with the next refresh. Past seasons are held
    # to the join-rate floor above instead.
    current = season_label(LAST_SEASON)
    early = merged.filter(pl.col("home_xg").is_null() & (pl.col("season") == current))
    if early.height:
        LOG.info(
            "%d %s matches without xG yet, left out until Understat has them",
            early.height,
            current,
        )
        merged = merged.filter(
            ~(pl.col("home_xg").is_null() & (pl.col("season") == current))
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(MATCHES_PARQUET)
    LOG.info(
        "wrote %s (%d matches, %d columns)",
        MATCHES_PARQUET,
        merged.height,
        merged.width,
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    merge_sources(force=args.force)


if __name__ == "__main__":
    main()
