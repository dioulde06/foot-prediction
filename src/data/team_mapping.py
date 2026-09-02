"""Explicit team-name mapping between sources.

No fuzzy matching, ever. A near-miss that silently resolves to the wrong club
shifts rows against each other and corrupts the whole dataset without raising
anything. An unknown name must stop the pipeline.

football-data.co.uk names are canonical: that is where the odds live, so
everything else is translated towards it.
"""

from __future__ import annotations

SOURCES = ("football-data", "understat")


class UnmappedTeamError(KeyError):
    """Raised when a source name has no canonical equivalent."""


# Understat name -> canonical (football-data) name. 35 entries, audited with
# scripts/audit_teams.py: both sources carry the same 137 clubs, and the same
# count per league, so this is a bijection over the names that differ.
UNDERSTAT_TO_CANONICAL: dict[str, str] = {
    # Bundesliga
    "Arminia Bielefeld": "Bielefeld",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia M.Gladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln",
    "FC Heidenheim": "Heidenheim",
    "Greuther Fuerth": "Greuther Furth",
    "Hamburger SV": "Hamburg",
    "Hertha Berlin": "Hertha",
    "Mainz 05": "Mainz",
    "RasenBallsport Leipzig": "RB Leipzig",
    "St. Pauli": "St Pauli",
    "VfB Stuttgart": "Stuttgart",
    # La Liga
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Celta Vigo": "Celta",
    "Espanyol": "Espanol",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Oviedo": "Oviedo",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "SD Huesca": "Huesca",
    # Ligue 1
    "Clermont Foot": "Clermont",
    "Paris Saint Germain": "Paris SG",
    "Saint-Etienne": "St Etienne",
    # Premier League
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
    # Serie A
    "AC Milan": "Milan",
    "Parma Calcio 1913": "Parma",
}

_BY_SOURCE: dict[str, dict[str, str]] = {
    "football-data": {},  # canonical by definition
    "understat": UNDERSTAT_TO_CANONICAL,
}


def to_canonical(name: str, source: str, known: frozenset[str] | None = None) -> str:
    """Canonical name of `name` as spelled by `source`.

    `known` is the set of canonical names currently in the dataset. When given,
    a name that survives translation but is not in it also raises: that is how
    a newly promoted club or a source-side rename gets caught instead of
    quietly dropping out of the join.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")

    mapping = _BY_SOURCE[source]
    canonical = mapping.get(name, name)
    if known is not None and canonical not in known:
        raise UnmappedTeamError(
            f"{name!r} from {source!r} resolves to {canonical!r}, which is not a "
            f"known team; add it to UNDERSTAT_TO_CANONICAL rather than guessing"
        )
    return canonical
