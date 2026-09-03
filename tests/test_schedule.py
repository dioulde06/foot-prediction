"""The season schedule: every fixture with its kick-off in UTC, names mapped."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src.data.schedule import normalise_schedule, upcoming
from src.data.team_mapping import UnmappedTeamError


def _raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "league": ["FRA-Ligue 1"] * 3,
            "season": ["2627"] * 3,
            "date": [
                dt.datetime(2026, 8, 30, 18, 45),
                dt.datetime(2026, 9, 5, 15, 15),
                dt.datetime(2026, 10, 20, 19, 0),
            ],
            "home_team": ["Paris Saint Germain", "Lens", "Le Mans"],
            "away_team": ["Lille", "Lorient", "Nice"],
            "is_result": [True, False, False],
        }
    )


def test_names_are_mapped_and_kickoff_is_kept_in_utc() -> None:
    schedule = normalise_schedule(_raw())
    assert schedule["home_team"].to_list() == ["Paris SG", "Lens", "Le Mans"]
    assert schedule["league"][0] == "Ligue 1"
    assert schedule["season"][0] == "2026-27"
    assert schedule["kickoff_utc"][1] == dt.datetime(2026, 9, 5, 15, 15)
    assert schedule["date"][1] == dt.date(2026, 9, 5)
    assert schedule["played"].to_list() == [True, False, False]


def test_an_unknown_team_stops_the_pipeline_when_a_known_set_is_given() -> None:
    with pytest.raises(UnmappedTeamError, match="Le Mans"):
        normalise_schedule(
            _raw(), known=frozenset({"Paris SG", "Lille", "Lens", "Lorient", "Nice"})
        )


def test_upcoming_keeps_unplayed_fixtures_inside_the_horizon_only() -> None:
    schedule = normalise_schedule(_raw())
    frame = upcoming(schedule, today=dt.date(2026, 9, 3), horizon_days=35)
    assert frame["home_team"].to_list() == ["Lens"]
    assert list(frame.columns) == [
        "date",
        "league",
        "home_team",
        "away_team",
        "kickoff_utc",
    ]
