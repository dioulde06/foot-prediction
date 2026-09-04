"""The mapping is the silent failure point of the project, so it gets tests."""

import pytest

from src.data.team_mapping import (
    ESPN_LEAGUES,
    ESPN_TO_CANONICAL,
    UNDERSTAT_TO_CANONICAL,
    UnmappedTeamError,
    to_canonical,
)


def test_a_known_understat_name_is_translated() -> None:
    assert to_canonical("Wolverhampton Wanderers", "understat") == "Wolves"
    assert to_canonical("Borussia M.Gladbach", "understat") == "M'gladbach"


def test_a_name_already_canonical_passes_through() -> None:
    assert to_canonical("Arsenal", "understat") == "Arsenal"
    assert to_canonical("Nott'm Forest", "football-data") == "Nott'm Forest"


def test_an_unknown_name_raises_when_the_known_set_is_given() -> None:
    known = frozenset({"Arsenal", "Wolves"})
    with pytest.raises(UnmappedTeamError, match="Fictional United"):
        to_canonical("Fictional United", "understat", known=known)


def test_the_error_names_both_the_name_and_the_source() -> None:
    with pytest.raises(UnmappedTeamError) as caught:
        to_canonical("Ghost FC", "understat", known=frozenset({"Arsenal"}))
    message = str(caught.value)
    assert "Ghost FC" in message
    assert "understat" in message


def test_an_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        to_canonical("Arsenal", "opta")


def test_no_two_understat_names_collide_onto_one_canonical_name() -> None:
    values = list(UNDERSTAT_TO_CANONICAL.values())
    assert len(values) == len(set(values)), "two sources names map to the same club"


def test_the_mapping_is_not_an_identity_anywhere() -> None:
    """An identity entry is dead weight and hides a real mismatch."""
    same = [k for k, v in UNDERSTAT_TO_CANONICAL.items() if k == v]
    assert not same, same


def test_a_known_espn_name_is_translated() -> None:
    assert to_canonical("Nottingham Forest", "espn") == "Nott'm Forest"
    assert to_canonical("Internazionale", "espn") == "Inter"
    assert to_canonical("Arsenal", "espn") == "Arsenal"


def test_no_two_espn_names_collide_onto_one_canonical_name() -> None:
    values = list(ESPN_TO_CANONICAL.values())
    assert len(values) == len(set(values)), "two ESPN names map to the same club"


def test_the_espn_mapping_is_not_an_identity_anywhere() -> None:
    same = [k for k, v in ESPN_TO_CANONICAL.items() if k == v]
    assert not same, same


def test_the_five_leagues_have_an_espn_slug() -> None:
    assert len(ESPN_LEAGUES) == 5
    assert set(ESPN_LEAGUES.values()) == {
        "Premier League",
        "La Liga",
        "Bundesliga",
        "Serie A",
        "Ligue 1",
    }
