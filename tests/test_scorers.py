"""Probable scorers: a Poisson on the team, a shrunk rate on the player."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import polars as pl
import pytest

from src.app import scorers as sc

PRIOR = sc.GoalsPrior(home_factor=1.21, pen_xg=0.12, player_rate=0.10)


def _players() -> pl.DataFrame:
    """Two seasons of one team: a striker who moved in, a keeper, a rookie."""
    return pl.DataFrame(
        {
            "league": ["Ligue 1"] * 5,
            "season": ["2025-26", "2025-26", "2026-27", "2026-27", "2026-27"],
            "team": ["Lille", "Lille", "Lille", "Lille", "Lille"],
            "player": ["Striker", "Keeper", "Striker", "Keeper", "Rookie"],
            "player_id": [1, 2, 1, 2, 3],
            "position": ["F S", "GK", "F S", "GK", "F M S"],
            "matches": [34, 34, 2, 2, 2],
            "minutes": [3060, 3060, 180, 180, 90],
            "np_goals": [15, 0, 1, 0, 1],
            "np_xg": [17.0, 0.0, 1.0, 0.0, 0.5],
            "shots": [90, 0, 6, 0, 3],
        }
    )


def test_seasons_are_pooled_per_player_and_the_latest_team_wins() -> None:
    moved = _players().with_columns(
        pl.when((pl.col("player_id") == 1) & (pl.col("season") == "2025-26"))
        .then(pl.lit("Lyon"))
        .otherwise(pl.col("team"))
        .alias("team")
    )
    rates = sc.player_rates(moved, PRIOR, min_minutes=900)
    striker = rates.filter(pl.col("player_id") == 1).row(0, named=True)
    assert striker["team"] == "Lille"
    assert striker["minutes"] == 3240
    assert striker["np_xg"] == pytest.approx(18.0)


def test_keepers_and_players_below_the_minutes_floor_are_dropped() -> None:
    rates = sc.player_rates(_players(), PRIOR, min_minutes=900)
    assert rates["player"].to_list() == ["Striker"]


def test_the_rate_is_shrunk_toward_the_league_prior() -> None:
    rates = sc.player_rates(_players(), PRIOR, min_minutes=0)
    rookie = rates.filter(pl.col("player_id") == 3).row(0, named=True)
    raw = 0.5 / 90 * 90
    assert PRIOR.player_rate < rookie["rate"] < raw
    striker = rates.filter(pl.col("player_id") == 1).row(0, named=True)
    prior_xg = PRIOR.player_rate * sc.PRIOR_MINUTES / 90
    assert striker["rate"] == pytest.approx(
        90 * (18.0 + prior_xg) / (3240 + sc.PRIOR_MINUTES)
    )
    assert abs(striker["rate"] - 0.5) < abs(rookie["rate"] - raw)


def test_expected_goals_split_the_home_factor_symmetrically() -> None:
    row = {
        "np_xg_created_5_home": 1.6,
        "np_xg_conceded_5_away": 1.6,
        "np_xg_created_5_away": 1.0,
        "np_xg_conceded_5_home": 1.0,
    }
    home, away = sc.expected_goals(row, PRIOR)
    assert home == pytest.approx(1.6 * math.sqrt(1.21))
    assert away == pytest.approx(1.0 / math.sqrt(1.21))


def test_goal_markets_follow_the_poisson() -> None:
    markets = sc.goal_markets(1.0, 1.0)
    assert markets["btts"] == pytest.approx((1 - math.exp(-1)) ** 2)
    assert markets["over25"] == pytest.approx(1 - math.exp(-2) * (1 + 2 + 2))


def test_a_player_in_an_average_match_keeps_his_usual_rate() -> None:
    rates = sc.player_rates(_players(), PRIOR, min_minutes=900)
    team_avg = sc.team_average(_players())["Lille"]
    table = sc.scorers_for_side(rates, "Lille", lam_np=team_avg, team_avg=team_avg)
    striker = table.row(0, named=True)
    assert striker["p_scores"] == pytest.approx(1 - math.exp(-striker["rate"]))
    assert 0 < striker["p_scores"] < 1


def test_scorers_file_is_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sc, "SCORERS_PARQUET", tmp_path / "scorers.parquet")
    rows = pl.DataFrame(
        {
            "date": [dt.date(2026, 9, 5)],
            "home_team": ["Lille"],
            "away_team": ["Lyon"],
            "side": ["home"],
            "team": ["Lille"],
            "player": ["Striker"],
            "position": ["F S"],
            "minutes": [3240],
            "np_goals": [16],
            "np_xg": [18.0],
            "shots": [96],
            "rate": [0.5],
            "p_scores": [0.39],
            "lambda_np": [1.5],
            "lambda_total": [1.62],
            "elo": [1600.0],
            "form_points_5": [2.0],
            "goals_scored_5": [1.8],
            "goals_conceded_5": [0.8],
            "np_xg_created_5": [1.5],
            "np_xg_conceded_5": [0.9],
        }
    )
    first = sc.append_scorers(rows, dt.datetime(2026, 9, 1, 9, 0))
    same_run = sc.append_scorers(
        rows.with_columns(pl.lit(0.99).alias("p_scores")), dt.datetime(2026, 9, 1, 9, 0)
    )
    assert first.height == same_run.height == 1
    assert same_run["p_scores"][0] == 0.39, "a freeze is never rewritten"
    # A later publication of the same match adds a second dated freeze.
    later = sc.append_scorers(rows, dt.datetime(2026, 9, 2, 9, 0))
    assert later.height == 2
    assert list(later.columns) == list(sc.SCORERS_SCHEMA)


def test_a_fixture_team_without_player_data_is_an_error_not_a_blank() -> None:
    features = pl.DataFrame(
        {
            "date": [dt.date(2026, 9, 5)],
            "home_team": ["Le Mans"],
            "away_team": ["Lille"],
            "np_xg_created_5_home": [1.0],
            "np_xg_conceded_5_home": [1.0],
            "np_xg_created_5_away": [1.0],
            "np_xg_conceded_5_away": [1.0],
            "elo_home_before": [1500.0],
            "elo_away_before": [1600.0],
            "form_points_5_home": [1.0],
            "form_points_5_away": [1.0],
            "goals_scored_5_home": [1.0],
            "goals_scored_5_away": [1.0],
            "goals_conceded_5_home": [1.0],
            "goals_conceded_5_away": [1.0],
        }
    )
    with pytest.raises(KeyError, match="Le Mans"):
        sc.scorers_for_fixtures(features, _players(), PRIOR)


def test_a_side_without_a_full_window_gets_no_estimate() -> None:
    row = {
        "np_xg_created_5_home": None,
        "np_xg_conceded_5_home": None,
        "np_xg_created_5_away": 1.0,
        "np_xg_conceded_5_away": 1.0,
    }
    assert sc.expected_goals(row, PRIOR) is None


def test_a_player_without_a_minute_this_season_is_not_listed_for_his_old_club() -> None:
    """Salah left Liverpool, Lewandowski left Barcelona: no 2026-27 row anywhere
    in the five leagues, so they must not be listed for last season's club."""
    players = pl.concat(
        [
            _players(),
            pl.DataFrame(
                {
                    "league": ["Ligue 1"],
                    "season": ["2025-26"],
                    "team": ["Lille"],
                    "player": ["Departed"],
                    "player_id": [9],
                    "position": ["F S"],
                    "matches": [34],
                    "minutes": [3000],
                    "np_goals": [20],
                    "np_xg": [18.0],
                    "shots": [100],
                }
            ),
        ]
    )
    rates = sc.player_rates(players, PRIOR, min_minutes=900)
    assert "Departed" not in rates["player"].to_list()
    assert "Striker" in rates["player"].to_list()


def test_before_the_first_matchday_last_season_still_counts() -> None:
    only_last = _players().filter(pl.col("season") == "2025-26")
    rates = sc.player_rates(only_last, PRIOR, min_minutes=900)
    assert rates["player"].to_list() == ["Striker"]
