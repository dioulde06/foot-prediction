"""Checks on the ingestion helpers that do not need the network."""

import datetime as dt

import polars as pl
import pytest

from src.data.fetch import _check_integrity, _parse_csv, season_code, season_label

HEADER = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,PSCH,PSCD,PSCA,B365CH,B365CD,B365CA,AvgCH,AvgCD,AvgCA"


def test_season_code_and_label() -> None:
    assert season_code(2020) == "2021"
    assert season_code(2025) == "2526"
    assert season_code(2099) == "9900"  # century rollover must not break the code
    assert season_label(2020) == "2020-21"


def test_parse_csv_handles_bom_blank_rows_and_both_date_formats() -> None:
    csv = (
        "﻿" + HEADER + "\n"
        "E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,19,10,10,3,6,7,1.30,6.00,8.50,1.29,6.0,9.0,1.31,6.1,8.9\n"
        "E0,16/08/25,12:30,Aston Villa,Newcastle,0,0,D,3,16,3,3,3,6,2.25,3.50,2.90,2.3,3.5,3.0,2.28,3.55,2.95\n"
        ",,,,,,,,,,,,,,,,,,,,,,\n"
    ).encode()
    frame = _parse_csv(csv, "E0", 2025)

    assert frame.height == 2, "the blank trailing row must be dropped"
    assert frame["date"].dtype == pl.Date
    assert [d.year for d in frame["date"]] == [2025, 2025]
    assert frame["season"].to_list() == ["2025-26", "2025-26"]
    assert frame["odds_close_ps_h"].to_list() == [1.30, 2.25]


def test_parse_csv_keeps_missing_odds_as_null() -> None:
    csv = (
        HEADER + "\n"
        "E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,19,10,10,3,6,7,,,,1.29,6.0,9.0,1.31,6.1,8.9\n"
    ).encode()
    frame = _parse_csv(csv, "E0", 2025)
    assert frame["odds_close_ps_h"].null_count() == 1
    assert frame["odds_close_b365_h"].to_list() == [1.29]


def test_parse_csv_rejects_a_missing_column() -> None:
    csv = (HEADER.replace(",HST", "") + "\n").encode()
    with pytest.raises(ValueError, match="missing"):
        _parse_csv(csv, "E0", 2025)


def _minimal_frame(**overrides: list[object]) -> pl.DataFrame:
    base: dict[str, list[object]] = {
        "date": [dt.date(2025, 8, 15)],
        "home_team": ["Liverpool"],
        "away_team": ["Bournemouth"],
        "home_goals": [4],
        "away_goals": [2],
        "result": ["H"],
    }
    return pl.DataFrame({**base, **overrides})


def test_check_integrity_rejects_duplicate_keys() -> None:
    frame = pl.concat([_minimal_frame(), _minimal_frame()])
    with pytest.raises(ValueError, match="duplicate"):
        _check_integrity(frame)


def test_check_integrity_rejects_unknown_result_code() -> None:
    with pytest.raises(ValueError, match="result codes"):
        _check_integrity(_minimal_frame(result=["W"]))


def test_check_integrity_rejects_null_goals() -> None:
    with pytest.raises(ValueError, match="home_goals"):
        _check_integrity(_minimal_frame(home_goals=[None]))


def test_season_of_puts_august_in_the_new_season() -> None:
    from src.data.fetch import season_of

    assert season_of(dt.date(2026, 9, 3)) == "2026-27"
    assert season_of(dt.date(2026, 8, 1)) == "2026-27"
    assert season_of(dt.date(2026, 7, 1)) == "2026-27"
    assert season_of(dt.date(2026, 5, 24)) == "2025-26"
    assert season_of(dt.date(2026, 1, 15)) == "2025-26"


def test_the_current_season_follows_the_calendar() -> None:
    from src.data.fetch import current_first_year

    assert current_first_year(dt.date(2026, 9, 3)) == 2026
    assert current_first_year(dt.date(2026, 5, 24)) == 2025
    assert current_first_year(dt.date(2026, 7, 1)) == 2026


def test_replacing_a_season_keeps_the_others_untouched() -> None:
    from src.data.fetch import replace_season

    existing = pl.DataFrame(
        {"season": ["2025-26", "2025-26", "2026-27"], "x": [1, 2, 3]}
    )
    fresh = pl.DataFrame({"season": ["2026-27", "2026-27"], "x": [30, 31]})
    out = replace_season(existing, fresh, "2026-27")
    assert out.filter(pl.col("season") == "2025-26")["x"].to_list() == [1, 2]
    assert out.filter(pl.col("season") == "2026-27")["x"].to_list() == [30, 31]


def test_replace_season_refuses_fresh_rows_from_another_season() -> None:
    from src.data.fetch import replace_season

    existing = pl.DataFrame({"season": ["2025-26"], "x": [1]})
    fresh = pl.DataFrame({"season": ["2025-26"], "x": [9]})
    with pytest.raises(ValueError, match="2025-26"):
        replace_season(existing, fresh, "2026-27")


def test_a_book_missing_from_a_season_file_parses_as_null_odds() -> None:
    """Early in a season football-data ships the file without Pinnacle closing
    prices. Odds are context, not identity: missing ones are nulls, not a crash."""
    header = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,"
        "B365CH,B365CD,B365CA,AvgCH,AvgCD,AvgCA\n"
    )
    row = "E0,15/08/2026,20:00,Liverpool,Bournemouth,4,2,H,10,8,5,3,6,4,1.3,5.5,9,1.31,5.6,8.9\n"
    frame = _parse_csv((header + row).encode(), "E0", 2026)
    assert frame.height == 1
    assert frame["odds_close_ps_h"][0] is None
    assert frame["odds_close_avg_h"][0] == 1.31
    assert frame["home_goals"][0] == 4
