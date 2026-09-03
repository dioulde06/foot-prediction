"""A season still being played is never a test season."""

from __future__ import annotations

import datetime as dt

import polars as pl

from src.eval.walk_forward import complete_seasons


def _frame(counts: dict[str, int]) -> pl.DataFrame:
    rows = []
    for k, (season, n) in enumerate(counts.items()):
        rows += [{"season": season, "date": dt.date(2020 + k, 8, 15)}] * n
    return pl.DataFrame(rows)


def test_a_partial_season_is_left_out_but_a_shorter_full_one_is_kept() -> None:
    frame = _frame({"2023-24": 1826, "2024-25": 1752, "2025-26": 1752, "2026-27": 120})
    assert complete_seasons(frame) == ["2023-24", "2024-25", "2025-26"]


def test_all_seasons_complete_when_none_is_short() -> None:
    frame = _frame({"2024-25": 1752, "2025-26": 1752})
    assert complete_seasons(frame) == ["2024-25", "2025-26"]
