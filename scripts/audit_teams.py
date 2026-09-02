"""List the team names of each source and flag the ones that do not match.

Run this before touching src/data/team_mapping.py: it prints the names that
need an entry, per league, so the dict is filled from evidence rather than
from guesses.

Run: uv run python -m scripts.audit_teams
"""

from __future__ import annotations

import polars as pl

from src.data.fetch import FOOTBALL_DATA_PARQUET, UNDERSTAT_PARQUET
from src.data.team_mapping import UNDERSTAT_TO_CANONICAL


def names(frame: pl.DataFrame) -> set[str]:
    return set(frame["home_team"].unique()) | set(frame["away_team"].unique())


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


if __name__ == "__main__":
    main()
