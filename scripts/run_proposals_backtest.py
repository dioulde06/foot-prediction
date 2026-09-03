"""Replay the combinator on the last complete season, week by week.

Out-of-sample model probabilities (walk-forward), closing average odds, real
results, the page's algorithm: what the three proposals of each objective
would have been, every week. Settlement happens on the page, from the played
matches, exactly as for the live registry.

Run: uv run python -m scripts.run_proposals_backtest   (or: make proposals-backtest)
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.app.combos import (
    MIN_POOL,
    OBJECTIVES,
    PROPOSALS_SCHEMA,
    WINDOW_DAYS,
    Match,
    proposals,
    to_rows,
    week_of,
)
from src.eval.baselines import devig_power
from src.eval.walk_forward import complete_seasons
from src.models.train import MATCHES_PARQUET

LOG = logging.getLogger(__name__)

OOS_PARQUET = Path("reports/oos_predictions.parquet")
BACKTEST_PARQUET = Path("reports/proposals_backtest.parquet")


def pool_of(frame: pl.DataFrame) -> list[Match]:
    out = []
    for r in frame.iter_rows(named=True):
        odds = [r["odds_close_avg_h"], r["odds_close_avg_d"], r["odds_close_avg_a"]]
        market = devig_power(np.asarray([odds]))[0].tolist()
        out.append(
            Match(
                date=r["date"],
                home_team=r["home_team"],
                away_team=r["away_team"],
                model=[r["model_h"], r["model_d"], r["model_a"]],
                market=market,
                odds=odds,
            )
        )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    oos = pl.read_parquet(OOS_PARQUET)
    season = complete_seasons(oos)[-1]
    matches = pl.read_parquet(MATCHES_PARQUET).select(
        "date", "home_team", "away_team", *[f"odds_close_avg_{o}" for o in "hda"]
    )
    frame = (
        oos.filter(pl.col("season") == season)
        .join(matches, on=["date", "home_team", "away_team"], how="inner")
        .filter(pl.col("odds_close_avg_h").is_not_null())
        .sort("date")
    )
    LOG.info("%s: %d matches with closing odds", season, frame.height)

    first = week_of(frame["date"].min())  # type: ignore[arg-type]
    last = frame["date"].max()
    rows = []
    week = first
    while week <= last:  # type: ignore[operator]
        pool = pool_of(
            frame.filter(
                (pl.col("date") >= week)
                & (pl.col("date") < week + dt.timedelta(days=WINDOW_DAYS))
            )
        )
        if len(pool) >= MIN_POOL:
            published_at = dt.datetime.combine(week, dt.time(7, 0))
            for objective in OBJECTIVES:
                rows.extend(
                    to_rows(proposals(pool, objective), objective, week, published_at)
                )
        week += dt.timedelta(days=WINDOW_DAYS)

    out = pl.DataFrame(rows, schema=PROPOSALS_SCHEMA).with_columns(
        pl.lit(season).alias("season")
    )
    BACKTEST_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(BACKTEST_PARQUET)
    LOG.info(
        "%d weeks, %d proposals, %d legs written to %s",
        out["week"].n_unique(),
        out.select("week", "objective", "rank").unique().height,
        out.height,
        BACKTEST_PARQUET,
    )


if __name__ == "__main__":
    main()
