"""List the team names of each source and flag the ones that do not match.

Run this before touching src/data/team_mapping.py: it prints the names that
need an entry, per league, so the dict is filled from evidence rather than
from guesses.

The ESPN section hits the network: it is the live-score source the page calls
itself, and a promoted club renamed there silently stops settling bets, so its
names get audited the same way as the dataset's.

Run: uv run python -m scripts.audit_teams
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request

import polars as pl

from src.data.fetch import FOOTBALL_DATA_PARQUET, UNDERSTAT_PARQUET
from src.data.team_mapping import (
    ESPN_LEAGUES,
    ESPN_SCOREBOARD,
    ESPN_TO_CANONICAL,
    UNDERSTAT_TO_CANONICAL,
)


def names(frame: pl.DataFrame) -> set[str]:
    return set(frame["home_team"].unique()) | set(frame["away_team"].unique())


def espn_names(slug: str, start: dt.date, end: dt.date) -> set[str]:
    """Team names ESPN spells over `start`..`end`, for one league."""
    url = (
        f"{ESPN_SCOREBOARD}/{slug}/scoreboard"
        f"?dates={start:%Y%m%d}-{end:%Y%m%d}&limit=1000"
    )
    with urllib.request.urlopen(url, timeout=30) as answer:  # noqa: S310
        events = json.load(answer).get("events", [])
    return {
        competitor["team"]["displayName"]
        for event in events
        for competitor in event["competitions"][0]["competitors"]
    }


def audit_espn(spine: pl.DataFrame) -> None:
    """The live source, over the season being played."""
    current = spine.filter(pl.col("season") == spine["season"].max())
    start, last = current["date"].min(), current["date"].max()
    if not isinstance(start, dt.date) or not isinstance(last, dt.date):
        raise TypeError("the spine has no dates for the season being played")
    end = max(last, dt.date.today() + dt.timedelta(days=10))
    print(f"\n\n### ESPN (direct), saison {current['season'][0]} ###")

    missing = 0
    for slug, league in ESPN_LEAGUES.items():
        canonical = names(current.filter(pl.col("league") == league))
        live = {ESPN_TO_CANONICAL.get(n, n) for n in espn_names(slug, start, end)}
        unmapped = sorted(live - canonical)
        absent = sorted(canonical - live)
        status = "OK" if not unmapped else "A MAPPER"
        print(f"\n=== {league} : {len(canonical)} vs {len(live)} noms, {status} ===")
        if unmapped:
            missing += len(unmapped)
            print(f"  ESPN non mappe : {', '.join(unmapped)}")
        if absent:
            print(f"  pas encore vu chez ESPN : {', '.join(absent)}")

    print(f"\n{len(ESPN_TO_CANONICAL)} entrees dans le mapping ESPN.")
    print(f"{missing} noms ESPN encore non resolus.")


def main() -> None:
    spine = pl.read_parquet(FOOTBALL_DATA_PARQUET)
    understat = pl.read_parquet(UNDERSTAT_PARQUET).with_columns(
        [
            pl.col(c).replace(UNDERSTAT_TO_CANONICAL).alias(c)
            for c in ("home_team", "away_team")
        ]
    )

    total_missing = 0
    for league in sorted(spine["league"].unique()):
        left = names(spine.filter(pl.col("league") == league))
        right = names(understat.filter(pl.col("league") == league))
        only_left, only_right = sorted(left - right), sorted(right - left)
        status = "OK" if not only_left and not only_right else "A MAPPER"
        print(f"\n=== {league} : {len(left)} vs {len(right)} noms, {status} ===")
        if not only_left and not only_right:
            continue
        total_missing += len(only_right)
        width = max((len(x) for x in only_left), default=0) + 2
        print(f"  {'football-data seulement':<{width}} | Understat seulement")
        for i in range(max(len(only_left), len(only_right))):
            left_name = only_left[i] if i < len(only_left) else ""
            right_name = only_right[i] if i < len(only_right) else ""
            print(f"  {left_name:<{width}} | {right_name}")

    print(f"\n{len(UNDERSTAT_TO_CANONICAL)} entrees dans le mapping.")
    print(f"{total_missing} noms Understat encore non resolus.")
    print("Les deux colonnes sont triees separement : ce ne sont PAS des paires.")

    audit_espn(spine)


if __name__ == "__main__":
    main()
