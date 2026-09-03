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
    csv = (HEADER.replace(",PSCH", "") + "\n").encode()
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
