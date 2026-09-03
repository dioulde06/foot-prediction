"""Player-season stats from Understat: explicit mapping, no silent drop."""

from __future__ import annotations

import polars as pl
import pytest

from src.data.players import normalise_players
from src.data.team_mapping import UnmappedTeamError


def _raw(**overrides: list[object]) -> pl.DataFrame:
    base: dict[str, list[object]] = {
        "league": ["FRA-Ligue 1", "FRA-Ligue 1"],
        "season": ["2526", "2627"],
        "team": ["Paris Saint Germain", "Lille"],
        "player": ["A", "B"],
        "player_id": [1, 2],
        "position": ["F S", "GK"],
        "matches": [30, 2],
        "minutes": [2500, 180],
        "np_goals": [20, 0],
        "np_xg": [18.5, 0.0],
        "shots": [100, 0],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_team_names_are_translated_to_the_canonical_spelling() -> None:
    players = normalise_players(_raw(), known=frozenset({"Paris SG", "Lille"}))
    assert players["team"].to_list() == ["Paris SG", "Lille"]
    assert players["league"].to_list() == ["Ligue 1", "Ligue 1"]
    assert players["season"].to_list() == ["2025-26", "2026-27"]


def test_an_unknown_team_stops_the_pipeline() -> None:
    with pytest.raises(UnmappedTeamError, match="Le Mans"):
        normalise_players(_raw(team=["Le Mans", "Lille"]), known=frozenset({"Lille"}))


def test_null_minutes_or_xg_stop_the_pipeline() -> None:
    with pytest.raises(ValueError, match="np_xg"):
        normalise_players(
            _raw(np_xg=[None, 0.0]), known=frozenset({"Paris SG", "Lille"})
        )
