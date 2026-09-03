"""The static site is a pure function of the committed parquet files."""

from __future__ import annotations

import datetime as dt
import json

import polars as pl
import pytest

from src.app import publish as pub
from src.app import site


def _predictions(*days: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "published_at": [dt.datetime(2026, 9, 1, 12, 0)] * len(days),
            "payload_sha256": ["deadbeef"] * len(days),
            "model_hash": ["cafebabe"] * len(days),
            "date": [dt.date(2026, 9, d) for d in days],
            "league": ["Premier League"] * len(days),
            "home_team": [f"H{d}" for d in days],
            "away_team": [f"A{d}" for d in days],
            "p_home": [0.5] * len(days),
            "p_draw": [0.25] * len(days),
            "p_away": [0.25] * len(days),
            "temperature": [1.03] * len(days),
        },
        schema=pub.SCHEMA,
    )


def _odds(*days: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "captured_at": [dt.datetime(2026, 9, 1, 9, 0)] * len(days),
            "date": [dt.date(2026, 9, d) for d in days],
            "league": ["Premier League"] * len(days),
            "home_team": [f"H{d}" for d in days],
            "away_team": [f"A{d}" for d in days],
            "kickoff_time": ["15:00"] * len(days),
            "odds_avg_h": [2.0] * len(days),
            "odds_avg_d": [3.4] * len(days),
            "odds_avg_a": [3.6] * len(days),
        },
        schema=pub.ODDS_SCHEMA,
    )


def _played(*days: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [dt.date(2026, 9, d) for d in days],
            "home_team": [f"H{d}" for d in days],
            "away_team": [f"A{d}" for d in days],
            "home_goals": [2] * len(days),
            "away_goals": [1] * len(days),
            "result": ["H"] * len(days),
            "odds_close_avg_h": [1.9] * len(days),
            "odds_close_avg_d": [3.5] * len(days),
            "odds_close_avg_a": [4.0] * len(days),
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


def test_upcoming_fixtures_carry_their_odds_and_a_devigged_market() -> None:
    data = site.build_data(
        _predictions(5), _odds(5), _played(), _oos(), today=dt.date(2026, 9, 3)
    )
    (match,) = data["upcoming"]
    assert match["odds"] == [2.0, 3.4, 3.6]
    assert match["time"] == "15:00"
    assert match["model"] == [0.5, 0.25, 0.25]
    assert abs(sum(match["market"]) - 1.0) < 1e-9
    assert match["overround"] == pytest.approx(
        1 / 2.0 + 1 / 3.4 + 1 / 3.6 - 1, abs=1e-4
    )


def test_a_fixture_without_captured_odds_is_shown_but_not_priced() -> None:
    data = site.build_data(
        _predictions(5), _odds(), _played(), _oos(), today=dt.date(2026, 9, 3)
    )
    (match,) = data["upcoming"]
    assert match["odds"] is None
    assert match["market"] is None


def test_played_predictions_move_from_upcoming_to_retro() -> None:
    data = site.build_data(
        _predictions(1, 5), _odds(1, 5), _played(1), _oos(), today=dt.date(2026, 9, 3)
    )
    assert [m["home"] for m in data["upcoming"]] == ["H5"]
    (retro,) = data["retro"]
    assert retro["score"] == "2-1"
    assert retro["result"] == "H"
    assert abs(sum(retro["market"]) - 1.0) < 1e-9


def test_retro_is_empty_when_nothing_published_has_been_played() -> None:
    data = site.build_data(
        _predictions(5), _odds(5), _played(), _oos(), today=dt.date(2026, 9, 3)
    )
    assert data["retro"] == []


def test_standing_and_bins_come_from_the_walk_forward_predictions() -> None:
    data = site.build_data(
        _predictions(5), _odds(5), _played(), _oos(), today=dt.date(2026, 9, 3)
    )
    assert [s["season"] for s in data["standing"]["seasons"]] == ["2024-25", "2025-26"]
    assert data["standing"]["bins_season"] == "2025-26"
    assert len(data["bins"]) == 6
    assert sum(b["n"] for b in data["bins"]) == 3 * 20


def test_the_page_is_json_serialisable_and_the_placeholder_is_filled() -> None:
    data = site.build_data(
        _predictions(5), _odds(5), _played(), _oos(), today=dt.date(2026, 9, 3)
    )
    html = site.render("<script>const DATA = __DATA__;</script>", data)
    assert "__DATA__" not in html
    assert json.loads(html.split("const DATA = ")[1].split(";</script>")[0])["upcoming"]


def _scorers(day: int) -> pl.DataFrame:
    from src.app.scorers import SCORERS_SCHEMA

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


def test_frozen_scorers_become_the_match_cards() -> None:
    data = site.build_data(
        _predictions(5),
        _odds(5),
        _played(1),
        _oos(),
        today=dt.date(2026, 9, 3),
        scorers=_scorers(5),
    )
    (match,) = data["upcoming"]
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
    data = site.build_data(
        _predictions(5), _odds(5), _played(), _oos(), today=dt.date(2026, 9, 3)
    )
    assert data["upcoming"][0]["cards"] is None
