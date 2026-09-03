"""The static site is a pure function of the committed parquet files."""

from __future__ import annotations

import datetime as dt
import json

import polars as pl
import pytest

from src.app import publish as pub
from src.app import site
from src.app.scorers import SCORERS_SCHEMA

TODAY = dt.date(2026, 9, 3)


def _predictions(
    *days: int, p_home: float = 0.5, at: dt.datetime | None = None
) -> pl.DataFrame:
    n = len(days)
    return pl.DataFrame(
        {
            "published_at": [at or dt.datetime(2026, 9, 1, 12, 0)] * n,
            "payload_sha256": ["deadbeef"] * n,
            "model_hash": ["cafebabe"] * n,
            "date": [dt.date(2026, 9, d) for d in days],
            "league": ["Premier League"] * n,
            "home_team": [f"H{d}" for d in days],
            "away_team": [f"A{d}" for d in days],
            "p_home": [p_home] * n,
            "p_draw": [0.25] * n,
            "p_away": [0.75 - p_home] * n,
            "temperature": [1.03] * n,
        },
        schema=pub.SCHEMA,
    )


def _odds(*days: int, b365: bool = True) -> pl.DataFrame:
    n = len(days)
    frame = pl.DataFrame(
        {
            "captured_at": [dt.datetime(2026, 9, 1, 9, 0)] * n,
            "date": [dt.date(2026, 9, d) for d in days],
            "league": ["Premier League"] * n,
            "home_team": [f"H{d}" for d in days],
            "away_team": [f"A{d}" for d in days],
            "kickoff_time": ["15:00"] * n,
            "odds_avg_h": [2.0] * n,
            "odds_avg_d": [3.4] * n,
            "odds_avg_a": [3.6] * n,
            "odds_b365_h": [2.1 if b365 else None] * n,
            "odds_b365_d": [3.3 if b365 else None] * n,
            "odds_b365_a": [3.5 if b365 else None] * n,
        }
    )
    return pub._conform(frame, pub.ODDS_SCHEMA)


def _played(*days: int) -> pl.DataFrame:
    n = len(days)
    return pl.DataFrame(
        {
            "date": [dt.date(2026, 9, d) for d in days],
            "home_team": [f"H{d}" for d in days],
            "away_team": [f"A{d}" for d in days],
            "home_goals": [2] * n,
            "away_goals": [1] * n,
            "result": ["H"] * n,
            "odds_close_avg_h": [1.9] * n,
            "odds_close_avg_d": [3.5] * n,
            "odds_close_avg_a": [4.0] * n,
        },
        schema={
            "date": pl.Date(),
            "home_team": pl.String(),
            "away_team": pl.String(),
            "home_goals": pl.Int64(),
            "away_goals": pl.Int64(),
            "result": pl.String(),
            "odds_close_avg_h": pl.Float64(),
            "odds_close_avg_d": pl.Float64(),
            "odds_close_avg_a": pl.Float64(),
        },
    )


def _schedule(*days: int) -> pl.DataFrame:
    n = len(days)
    return pl.DataFrame(
        {
            "league": ["Premier League"] * n,
            "season": ["2026-27"] * n,
            "kickoff_utc": [dt.datetime(2026, 9, d, 19, 0) for d in days],
            "date": [dt.date(2026, 9, d) for d in days],
            "home_team": [f"H{d}" for d in days],
            "away_team": [f"A{d}" for d in days],
            "played": [False] * n,
        },
        schema=site.SCHEDULE_SCHEMA,
    )


def _oos() -> pl.DataFrame:
    n = 40
    return pl.DataFrame(
        {
            "season": ["2024-25"] * (n // 2) + ["2025-26"] * (n // 2),
            "date": [dt.date(2025, 1, 1)] * (n // 2) + [dt.date(2026, 1, 1)] * (n // 2),
            "result": ["H", "D", "A", "H"] * (n // 4),
            "model_h": [0.5] * n,
            "model_d": [0.25] * n,
            "model_a": [0.25] * n,
            "market_h": [0.48] * n,
            "market_d": [0.27] * n,
            "market_a": [0.25] * n,
        }
    )


def _scorers(day: int) -> pl.DataFrame:
    def side(name: str, team: str, lam: float) -> dict[str, list[object]]:
        return {
            "published_at": [dt.datetime(2026, 9, 1, 12, 0)] * 2,
            "date": [dt.date(2026, 9, day)] * 2,
            "home_team": [f"H{day}"] * 2,
            "away_team": [f"A{day}"] * 2,
            "side": [name] * 2,
            "team": [team] * 2,
            "player": [f"{team} striker", f"{team} winger"],
            "position": ["F S", "F M S"],
            "minutes": [2000, 1500],
            "np_goals": [10, 4],
            "np_xg": [9.0, 5.0],
            "shots": [60, 30],
            "rate": [0.4, 0.3],
            "p_scores": [0.35, 0.25],
            "lambda_np": [lam] * 2,
            "lambda_total": [lam + 0.12] * 2,
            "elo": [1550.0] * 2,
            "form_points_5": [1.8] * 2,
            "goals_scored_5": [1.6] * 2,
            "goals_conceded_5": [1.0] * 2,
            "np_xg_created_5": [1.4] * 2,
            "np_xg_conceded_5": [1.1] * 2,
        }

    home = pl.DataFrame(side("home", f"H{day}", 1.5), schema=SCORERS_SCHEMA)
    away = pl.DataFrame(side("away", f"A{day}", 1.0), schema=SCORERS_SCHEMA)
    return pl.concat([home, away])


def build(**kw: object) -> dict:
    args = {
        "predictions": _predictions(5),
        "odds": _odds(5),
        "played": _played(),
        "oos": _oos(),
        "today": TODAY,
    }
    args.update(kw)
    return site.build_data(**args)  # type: ignore[arg-type]


def test_a_priced_fixture_carries_the_market_and_every_book_that_priced_it() -> None:
    (match,) = build()["upcoming"]
    assert match["odds"] == [2.0, 3.4, 3.6]
    assert match["books"] == {"B365": [2.1, 3.3, 3.5]}
    assert abs(sum(match["market"]) - 1.0) < 1e-9
    assert match["overround"] == pytest.approx(
        1 / 2.0 + 1 / 3.4 + 1 / 3.6 - 1, abs=1e-4
    )
    assert match["result"] is None and match["score"] is None


def test_a_book_that_has_not_priced_a_match_is_left_out_of_its_books() -> None:
    (match,) = build(odds=_odds(5, b365=False))["upcoming"]
    assert match["books"] == {}


def test_a_fixture_without_captured_odds_is_shown_but_not_priced() -> None:
    (match,) = build(odds=_odds())["upcoming"]
    assert match["odds"] is None and match["market"] is None and match["books"] is None


def test_the_kickoff_comes_from_the_schedule_in_utc_else_from_the_feed() -> None:
    (from_schedule,) = build(schedule=_schedule(5))["upcoming"]
    assert from_schedule["kickoff"] == "2026-09-05T19:00:00Z"
    (from_feed,) = build()["upcoming"]
    # 15:00 in London on 5 September is British Summer Time, so 14:00 UTC.
    assert from_feed["kickoff"] == "2026-09-05T14:00:00Z"


def test_only_the_latest_prediction_of_a_match_is_shown() -> None:
    first = _predictions(5, p_home=0.50)
    moved = _predictions(5, p_home=0.55, at=dt.datetime(2026, 9, 2, 12, 0))
    (match,) = build(predictions=pl.concat([first, moved]))["upcoming"]
    assert match["model"][0] == 0.55
    assert match["publishedAt"] == "2026-09-02 12:00"


def test_a_recently_played_match_stays_with_its_result_and_older_ones_leave() -> None:
    data = build(predictions=_predictions(1, 5), odds=_odds(1, 5), played=_played(1))
    by_home = {m["home"]: m for m in data["upcoming"]}
    assert by_home["H1"]["result"] == "H" and by_home["H1"]["score"] == "2-1"
    assert by_home["H5"]["result"] is None
    old = _predictions(1).with_columns(pl.lit(dt.date(2026, 8, 1)).alias("date"))
    data = build(predictions=pl.concat([old, _predictions(5)]))
    assert [m["home"] for m in data["upcoming"]] == ["H5"]


def test_books_carry_the_margin_they_showed_on_the_captured_odds() -> None:
    books = build()["meta"]["books"]
    assert books[0] == {
        "key": "AVG",
        "name": "Moyenne du marché",
        "url": None,
        "margin": None,
    }
    b365 = next(b for b in books if b["key"] == "B365")
    assert b365["margin"] == pytest.approx(1 / 2.1 + 1 / 3.3 + 1 / 3.5 - 1, abs=1e-4)
    assert next(b for b in books if b["key"] == "SKB")["margin"] is None


def test_standing_and_bins_come_from_the_walk_forward_predictions() -> None:
    data = build()
    assert [s["season"] for s in data["standing"]["seasons"]] == ["2024-25", "2025-26"]
    assert data["standing"]["bins_season"] == "2025-26"
    assert len(data["bins"]) == 6
    assert sum(b["n"] for b in data["bins"]) == 3 * 20


def test_frozen_scorers_become_the_match_cards() -> None:
    (match,) = build(scorers=_scorers(5))["upcoming"]
    cards = match["cards"]
    assert cards["home"]["team"] == "H5"
    assert [p["player"] for p in cards["home"]["scorers"]] == [
        "H5 striker",
        "H5 winger",
    ]
    assert cards["home"]["points5"] == 9.0
    assert cards["lambdaHome"] == 1.62 and cards["lambdaAway"] == 1.12
    assert 0 < cards["btts"] < 1 and 0 < cards["over25"] < 1


def test_a_match_without_frozen_scorers_has_no_cards() -> None:
    assert build()["upcoming"][0]["cards"] is None


def test_the_page_is_json_serialisable_and_the_placeholder_is_filled() -> None:
    html = site.render("<script>const DATA = __DATA__;</script>", build())
    assert "__DATA__" not in html
    assert json.loads(html.split("const DATA = ")[1].split(";</script>")[0])["upcoming"]


def _proposal_rows(week: dt.date, *days: int) -> pl.DataFrame:
    from src.app.combos import PROPOSALS_SCHEMA

    rows = [
        {
            "published_at": dt.datetime(2026, 9, 1, 7, 0),
            "week": week,
            "objective": "target",
            "rank": 1,
            "leg": i + 1,
            "date": dt.date(2026, 9, d),
            "home_team": f"H{d}",
            "away_team": f"A{d}",
            "pick": "1",
            "odds": 2.0,
            "p_model": 0.5,
            "p_market": 0.48,
        }
        for i, d in enumerate(days)
    ]
    return pl.DataFrame(rows, schema=PROPOSALS_SCHEMA)


def test_a_frozen_proposal_settles_from_the_played_results() -> None:
    live = _proposal_rows(dt.date(2026, 8, 31), 1, 5)
    data = build(played=_played(1), proposals=live)
    (week,) = data["track"]["live"]["objectives"]["target"]["weeks"]
    (bet,) = week["bets"]
    assert bet["odds"] == 4.0 and bet["pq"] == pytest.approx(0.48**2)
    assert [leg["hit"] for leg in bet["legs"]] == [True, None]
    assert bet["won"] is None, "one leg still to play: open"
    assert data["track"]["live"]["since"] == "2026-08-31"


def test_a_missed_leg_loses_the_proposal_even_before_the_others_play() -> None:
    live = _proposal_rows(dt.date(2026, 8, 31), 1, 5).with_columns(
        pl.when(pl.col("leg") == 1)
        .then(pl.lit("2"))
        .otherwise(pl.col("pick"))
        .alias("pick")
    )
    track = build(played=_played(1), proposals=live)["track"]
    assert track["live"]["objectives"]["target"]["weeks"][0]["bets"][0]["won"] is False


def test_the_track_record_is_empty_but_well_formed_without_registries() -> None:
    track = build()["track"]
    assert track["stake"] == 10.0
    assert track["backtest"]["season"] is None
    assert track["live"]["since"] is None
    for objective in ("target", "margin", "consensus"):
        assert track["live"]["objectives"][objective]["weeks"] == []
