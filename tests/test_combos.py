"""The combinator, in Python: the same algorithm the page runs, so that frozen
proposals and the replayed season are evaluable against what the page shows."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from src.app import combos as cb


def _match(i: int, home_p: float, odds_margin: float = 0.05) -> cb.Match:
    model = [home_p, 0.25, 0.75 - home_p]
    market = [home_p - 0.02, 0.26, 0.76 - home_p]
    odds = [round(1 / (q * (1 + odds_margin)), 2) for q in market]
    return cb.Match(
        date=dt.date(2026, 9, 5 + i // 4),
        home_team=f"H{i}",
        away_team=f"A{i}",
        model=model,
        market=market,
        odds=odds,
    )


POOL = [_match(i, 0.30 + 0.03 * (i % 12)) for i in range(20)]


def test_target_proposals_land_near_the_target_and_are_distinct() -> None:
    props = cb.proposals(POOL, "target", target=10.0)
    assert 1 <= len(props) <= 3
    for p in props:
        assert 8.5 <= p.odds <= 11.5
        assert 2 <= len(p.legs) <= 6
    sigs = [{(leg.home_team, leg.pick) for leg in p.legs} for p in props]
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            assert len(sigs[a] & sigs[b]) <= len(sigs[a]) // 2


def test_margin_and_consensus_use_the_requested_number_of_legs() -> None:
    for objective in ("margin", "consensus"):
        props = cb.proposals(POOL, objective, n_legs=4)
        assert props and all(len(p.legs) == 4 for p in props)


def test_the_market_favourite_of_each_match_is_the_candidate() -> None:
    """Match 0: market 0.28 / 0.26 / 0.46, so the away win is the candidate leg."""
    (best,) = cb.candidates(POOL[:1], "target")
    assert POOL[0].market[2] > POOL[0].market[0]
    assert best.pick == "2"


def test_a_proposal_scores_its_probabilities_by_multiplying_the_legs() -> None:
    props = cb.proposals(POOL, "margin", n_legs=2)
    p = props[0]
    pq = 1.0
    for leg in p.legs:
        pq *= leg.p_market
    assert p.p_market == pytest.approx(pq)
    assert p.back == pytest.approx(p.p_market * p.odds)


def test_rows_round_trip_through_the_registry_schema() -> None:
    props = cb.proposals(POOL, "target", target=10.0)
    rows = cb.to_rows(
        props,
        "target",
        week=dt.date(2026, 8, 31),
        published_at=dt.datetime(2026, 9, 3, 7, 0),
    )
    frame = pl.DataFrame(rows, schema=cb.PROPOSALS_SCHEMA)
    assert frame["rank"].min() == 1
    assert frame["week"][0] == dt.date(2026, 8, 31)
    assert set(frame["pick"].to_list()) <= {"1", "X", "2"}


def test_freeze_writes_one_set_per_week_and_needs_a_real_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cb, "PROPOSALS_PARQUET", tmp_path / "proposals.parquet")
    first = cb.freeze_proposals(POOL, dt.datetime(2026, 9, 3, 7, 0))
    assert first is not None and first.height > 0
    assert set(first["objective"].to_list()) == {"target", "margin", "consensus"}
    again = cb.freeze_proposals(POOL, dt.datetime(2026, 9, 4, 7, 0))
    assert again is not None and again.height == first.height, (
        "same week: nothing added"
    )
    later = cb.freeze_proposals(POOL, dt.datetime(2026, 9, 10, 7, 0))
    assert later is not None and later.height > first.height, "a new week: a new set"
    assert cb.freeze_proposals(POOL[:3], dt.datetime(2026, 9, 17, 7, 0)) is not None
    assert (
        pl.read_parquet(tmp_path / "proposals.parquet")
        .filter(pl.col("week") == dt.date(2026, 9, 14))
        .height
        == 0
    ), "fewer than six priced matches: no proposals that week"
