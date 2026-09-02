"""Chronological splits. Never random, never shuffled.

Seasons are fixed here rather than derived, so that a change of split is a
visible edit to this file and shows up in review.
"""

from __future__ import annotations

import datetime as dt
from typing import cast

import polars as pl

TRAIN_SEASONS: tuple[str, ...] = ("2020-21", "2021-22", "2022-23", "2023-24")
VALID_SEASON = "2024-25"
TEST_SEASON = "2025-26"


def chronological_split(
    matches: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split by season, then assert the three parts do not overlap in time."""
    train = matches.filter(pl.col("season").is_in(TRAIN_SEASONS)).sort("date")
    valid = matches.filter(pl.col("season") == VALID_SEASON).sort("date")
    test = matches.filter(pl.col("season") == TEST_SEASON).sort("date")

    for name, frame in (("train", train), ("valid", valid), ("test", test)):
        if frame.is_empty():
            raise ValueError(f"{name} split is empty; check the season labels")

    # Series.max() is typed as a wide union in polars; the column is a Date.
    train_end = cast(dt.date, train["date"].max())
    valid_start = cast(dt.date, valid["date"].min())
    valid_end = cast(dt.date, valid["date"].max())
    test_start = cast(dt.date, test["date"].min())
    if train_end >= valid_start:
        raise ValueError(f"train ends {train_end}, valid starts {valid_start}: overlap")
    if valid_end >= test_start:
        raise ValueError(f"valid ends {valid_end}, test starts {test_start}: overlap")
    return train, valid, test
