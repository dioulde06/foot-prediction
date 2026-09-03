"""Tests on the two guarantees of the publication path.

Append-only, and no second code path for upcoming fixtures.
"""

import datetime as dt
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from src.app import publish as pub


@pytest.fixture(autouse=True)
def _isolated_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pub, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(pub, "PREDICTIONS_PARQUET", tmp_path / "predictions.parquet")


def _max_date(frame: pl.DataFrame) -> dt.date:
    return cast(dt.date, frame["date"].max())


def _row(
    day: int, home: str = "A", away: str = "B", p_home: float = 0.5
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "published_at": [dt.datetime(2026, 9, 1, 12, 0)],
            "payload_sha256": ["deadbeef"],
            "model_hash": ["cafebabe"],
            "date": [dt.date(2026, 9, day)],
            "league": ["Test League"],
            "home_team": [home],
            "away_team": [away],
            "p_home": [p_home],
            "p_draw": [0.25],
            "p_away": [1 - p_home - 0.25],
            "temperature": [1.0],
        },
        schema=pub.SCHEMA,
    )


def test_the_first_append_creates_the_history_with_the_fixed_schema() -> None:
    history = pub.append(_row(3))
    assert history.height == 1
    assert list(history.columns) == list(pub.SCHEMA)
    assert pub.PREDICTIONS_PARQUET.exists()


def test_appending_a_new_fixture_grows_the_history() -> None:
    pub.append(_row(3))
    history = pub.append(_row(4, home="C", away="D"))
    assert history.height == 2


def test_a_fixture_already_published_is_never_republished() -> None:
    """The append-only guarantee: a published row is never rewritten."""
    pub.append(_row(3, p_home=0.5))
    history = pub.append(_row(3, p_home=0.9))
    assert history.height == 1
    assert history["p_home"][0] == pytest.approx(0.5), "the first prediction must stand"


def test_a_batch_mixing_new_and_already_published_keeps_only_the_new() -> None:
    pub.append(_row(3))
    batch = pl.concat([_row(3, p_home=0.9), _row(5, home="E", away="F")])
    history = pub.append(batch)
    assert history.height == 2
    assert set(history["date"].to_list()) == {dt.date(2026, 9, 3), dt.date(2026, 9, 5)}
    assert history.filter(pl.col("date") == dt.date(2026, 9, 3))["p_home"][0] == 0.5


def _played() -> pl.DataFrame:
    rows = []
    for i in range(40):
        home, away = ("A", "B") if i % 2 else ("C", "D")
        rows.append(
            {
                "date": dt.date(2026, 1, 1) + dt.timedelta(days=3 * i),
                "season": "2025-26",
                "league": "Test League",
                "home_team": home,
                "away_team": away,
                "home_goals": i % 3,
                "away_goals": (i + 1) % 3,
                "home_shots_target": 4,
                "away_shots_target": 3,
                "home_np_xg": 1.1,
                "away_np_xg": 0.9,
                "result": "H" if i % 3 > (i + 1) % 3 else "A",
            }
        )
    return pl.DataFrame(rows).sort("date")


def test_upcoming_features_go_through_the_training_code_path() -> None:
    played = _played()
    fixtures = pl.DataFrame(
        {
            "date": [_max_date(played) + dt.timedelta(days=7)],
            "league": ["Test League"],
            "home_team": ["A"],
            "away_team": ["B"],
        }
    )
    features = pub.upcoming_features(fixtures, played)
    assert features.height == 1
    assert features["elo_diff"][0] is not None
    assert features["np_xg_created_diff_5"][0] is not None


def test_a_fixture_already_in_the_dataset_is_dropped() -> None:
    played = _played()
    fixtures = pl.DataFrame(
        {
            "date": [_max_date(played)],
            "league": ["Test League"],
            "home_team": ["A"],
            "away_team": ["B"],
        }
    )
    assert pub.upcoming_features(fixtures, played).is_empty()


def test_publishing_an_unplayed_fixture_does_not_read_its_own_result() -> None:
    """The fixture has no score, so its features must match a null-score row."""
    played = _played()
    fixtures = pl.DataFrame(
        {
            "date": [_max_date(played) + dt.timedelta(days=7)],
            "league": ["Test League"],
            "home_team": ["A"],
            "away_team": ["B"],
        }
    )
    from src.features.build import FEATURE_COLUMNS, build_features

    upcoming = pub.upcoming_features(fixtures, played)
    # The same fixture, but now with a 9-0 score in the frame.
    scored = pl.concat(
        [
            played,
            played.head(1).with_columns(
                pl.lit(cast(dt.date, fixtures["date"][0])).alias("date"),
                pl.lit("A").alias("home_team"),
                pl.lit("B").alias("away_team"),
                pl.lit(9, dtype=pl.Int64).alias("home_goals"),
                pl.lit(0, dtype=pl.Int64).alias("away_goals"),
                pl.lit(9.0, dtype=pl.Float64).alias("home_np_xg"),
                pl.lit("H").alias("result"),
            ),
        ]
    ).sort("date")
    with_result = build_features(scored, _max_date(scored)).filter(
        pl.col("date") == cast(dt.date, fixtures["date"][0])
    )
    for column in FEATURE_COLUMNS:
        assert upcoming[column].to_list() == with_result[column].to_list(), column


# ---------------- odds capture, append-only like the predictions ----------------


@pytest.fixture(autouse=True)
def _isolated_odds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pub, "ODDS_PARQUET", tmp_path / "odds.parquet")


FEED = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,Referee,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
    "B1,03/09/2026,19:30,Anderlecht,Kortrijk,,1.48,4.2,5.5,1.49,4.33,5.49\n"
    "E0,05/09/2026,15:00,Brentford,Everton,,2.1,3.4,3.6,2.08,3.42,3.55\n"
    "SP1,06/09/2026,20:00,Sociedad,Celta,,1.95,3.6,3.8,,,\n"
)


def test_fetch_fixtures_keeps_kickoff_time_and_average_odds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pub, "_download", lambda url: FEED.encode())
    fixtures = pub.fetch_fixtures()
    assert fixtures["league"].to_list() == ["Premier League", "La Liga"]
    assert fixtures["kickoff_time"].to_list() == ["15:00", "20:00"]
    assert fixtures["odds_avg_h"].to_list() == [2.08, None]
    assert fixtures["odds_avg_a"][0] == 3.55


def _fixture_row(day: int, home: str = "A", away: str = "B") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "league_code": ["E0"],
            "date": [dt.date(2026, 9, day)],
            "home_team": [home],
            "away_team": [away],
            "league": ["Premier League"],
            "kickoff_time": ["15:00"],
            "odds_avg_h": [2.0],
            "odds_avg_d": [3.4],
            "odds_avg_a": [3.6],
        }
    )


def test_the_first_odds_capture_creates_the_file_with_the_fixed_schema() -> None:
    odds = pub.append_odds(_fixture_row(5), dt.datetime(2026, 9, 1, 9, 0))
    assert odds.height == 1
    assert list(odds.columns) == list(pub.ODDS_SCHEMA)
    assert odds["captured_at"][0] == dt.datetime(2026, 9, 1, 9, 0)


def test_odds_already_captured_are_never_overwritten() -> None:
    pub.append_odds(_fixture_row(5), dt.datetime(2026, 9, 1, 9, 0))
    later = _fixture_row(5).with_columns(pl.lit(1.5).alias("odds_avg_h"))
    odds = pub.append_odds(later, dt.datetime(2026, 9, 2, 9, 0))
    assert odds.height == 1
    assert odds["odds_avg_h"][0] == 2.0
    assert odds["captured_at"][0] == dt.datetime(2026, 9, 1, 9, 0)


def test_a_new_fixture_grows_the_odds_history() -> None:
    pub.append_odds(_fixture_row(5), dt.datetime(2026, 9, 1, 9, 0))
    odds = pub.append_odds(_fixture_row(6, "C", "D"), dt.datetime(2026, 9, 2, 9, 0))
    assert odds.height == 2
