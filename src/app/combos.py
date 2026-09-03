"""The combinator, in Python: the algorithm the page runs, frozen and replayed.

The page proposes three combinations per objective. To know whether those
proposals are worth anything, the same algorithm has to run where it can be
settled: here, once a week on the published pool (frozen in an append-only
registry), and on the past season with out-of-sample predictions and closing
odds (the replay). Keep this file and the page's JavaScript in step.

Objectives:
- target: the most probable combination (by the market) with combined odds
  within 15 % of a target, 2 to 6 legs;
- margin: the n legs whose books keep the least, favourites in practice;
- consensus: the n legs where model and market agree most.
Only priced matches take part: a proposal has to be settleable at real odds.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import polars as pl

LOG = logging.getLogger(__name__)

PROPOSALS_PARQUET = Path("predictions/proposals.parquet")
OBJECTIVES = ("target", "margin", "consensus")
TARGET_ODDS = 10.0
N_LEGS = 4
TOP_CANDIDATES = 16
TOLERANCE = 0.15
MIN_POOL = 6
WINDOW_DAYS = 7
PICKS = ("1", "X", "2")

PROPOSALS_SCHEMA: dict[str, pl.DataType] = {
    "published_at": pl.Datetime("us"),
    "week": pl.Date(),
    "objective": pl.String(),
    "rank": pl.Int64(),
    "leg": pl.Int64(),
    "date": pl.Date(),
    "home_team": pl.String(),
    "away_team": pl.String(),
    "pick": pl.String(),
    "odds": pl.Float64(),
    "p_model": pl.Float64(),
    "p_market": pl.Float64(),
}


@dataclass(frozen=True)
class Match:
    date: dt.date
    home_team: str
    away_team: str
    model: list[float]
    market: list[float]
    odds: list[float]


@dataclass(frozen=True)
class Leg:
    match: Match
    outcome: int

    @property
    def pick(self) -> str:
        return PICKS[self.outcome]

    @property
    def home_team(self) -> str:
        return self.match.home_team

    @property
    def odds(self) -> float:
        return self.match.odds[self.outcome]

    @property
    def p_model(self) -> float:
        return self.match.model[self.outcome]

    @property
    def p_market(self) -> float:
        return self.match.market[self.outcome]


@dataclass(frozen=True)
class Proposal:
    legs: tuple[Leg, ...]
    p_model: float
    p_market: float
    odds: float
    gap: float

    @property
    def back(self) -> float:
        return self.p_market * self.odds


def leg_score(m: Match, o: int, objective: str) -> float:
    if objective == "margin":
        return m.market[o] * m.odds[o]
    if objective == "consensus":
        return -abs(m.model[o] - m.market[o])
    return m.market[o]


def candidates(pool: list[Match], objective: str) -> list[Leg]:
    """The best leg of each match, the best matches first, at most 16."""
    best = [
        Leg(m, max(range(3), key=lambda o: leg_score(m, o, objective))) for m in pool
    ]
    best.sort(key=lambda leg: -leg_score(leg.match, leg.outcome, objective))
    return best[:TOP_CANDIDATES]


def score(legs: tuple[Leg, ...]) -> Proposal:
    pm = pq = odds = 1.0
    gap = 0.0
    for leg in legs:
        pm *= leg.p_model
        pq *= leg.p_market
        odds *= leg.odds
        gap += abs(leg.p_model - leg.p_market)
    return Proposal(legs=legs, p_model=pm, p_market=pq, odds=odds, gap=gap / len(legs))


def proposals(
    pool: list[Match],
    objective: str,
    target: float = TARGET_ODDS,
    n_legs: int = N_LEGS,
) -> list[Proposal]:
    """Three distinct proposals, or fewer when the pool does not allow it."""
    if objective not in OBJECTIVES:
        raise ValueError(
            f"unknown objective {objective!r}, expected one of {OBJECTIVES}"
        )
    cands = candidates(pool, objective)
    combos: list[Proposal] = []
    if objective == "target":
        for k in range(2, 7):
            combos.extend(score(c) for c in combinations(cands, k))
        combos = [
            c
            for c in combos
            if target * (1 - TOLERANCE) <= c.odds <= target * (1 + TOLERANCE)
        ]
        combos.sort(key=lambda c: (-c.p_market, -c.back))
    else:
        size = min(n_legs, len(cands))
        if size < 2:
            return []
        combos = [score(c) for c in combinations(cands, size)]
        if objective == "margin":
            combos.sort(key=lambda c: (-c.back, -c.p_market))
        else:
            combos.sort(key=lambda c: (c.gap, -c.p_market))
    picked: list[Proposal] = []
    picked_sigs: list[set[tuple[str, int]]] = []
    for c in combos:
        sig = {(leg.home_team, leg.outcome) for leg in c.legs}
        if any(len(sig & other) > len(sig) // 2 for other in picked_sigs):
            continue
        picked.append(c)
        picked_sigs.append(sig)
        if len(picked) == 3:
            break
    return picked


def to_rows(
    props: list[Proposal], objective: str, week: dt.date, published_at: dt.datetime
) -> list[dict[str, Any]]:
    """One row per leg, ready for the registry."""
    return [
        {
            "published_at": published_at,
            "week": week,
            "objective": objective,
            "rank": rank,
            "leg": index,
            "date": leg.match.date,
            "home_team": leg.match.home_team,
            "away_team": leg.match.away_team,
            "pick": leg.pick,
            "odds": leg.odds,
            "p_model": leg.p_model,
            "p_market": leg.p_market,
        }
        for rank, p in enumerate(props, start=1)
        for index, leg in enumerate(p.legs, start=1)
    ]


def week_of(day: dt.date) -> dt.date:
    """The Monday that starts the week of `day`."""
    return day - dt.timedelta(days=day.weekday())


def freeze_proposals(
    pool: list[Match], published_at: dt.datetime
) -> pl.DataFrame | None:
    """Freeze this week's proposals once, for every objective. Append-only.

    The pool is the coming week's priced fixtures. A week gets its proposals at
    the first publication that sees at least six of them; later publications
    that week add nothing, so the set that is judged is the one that was shown.
    Returns the registry, or None when nothing could be frozen and none exists.
    """
    week = week_of(published_at.date())
    history = pl.read_parquet(PROPOSALS_PARQUET) if PROPOSALS_PARQUET.exists() else None
    if history is not None and history.filter(pl.col("week") == week).height:
        return history
    if len(pool) < MIN_POOL:
        LOG.info(
            "semaine du %s: %d matchs cotes, moins de %d, pas de propositions",
            week,
            len(pool),
            MIN_POOL,
        )
        return history
    rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        rows.extend(to_rows(proposals(pool, objective), objective, week, published_at))
    if not rows:
        return history
    fresh = pl.DataFrame(rows, schema=PROPOSALS_SCHEMA)
    combined = fresh if history is None else pl.concat([history, fresh])
    PROPOSALS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(PROPOSALS_PARQUET)
    LOG.info(
        "semaine du %s: %d propositions figees",
        week,
        fresh["rank"].n_unique() * len(OBJECTIVES),
    )
    return combined
