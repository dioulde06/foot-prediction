"""Publish predictions for the upcoming fixtures.

Run before each matchday: uv run python -m scripts.publish   (or: make publish)

The result is appended to predictions/predictions.parquet, which is tracked in
git. Commit it straight away: git's history is what proves to a sceptic that
the prediction existed before the match, without having to trust the timestamp
written inside the file.
"""

from __future__ import annotations

import logging

import polars as pl

from src.app.publish import PREDICTIONS_PARQUET, publish


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    history = publish()
    if history.is_empty():
        print("Aucun match a venir dans le flux football-data.")
        return
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_width_chars=160, tbl_rows=40):
        print(
            history.tail(20).select(
                "date", "league", "home_team", "away_team", "p_home", "p_draw", "p_away"
            )
        )
    print(f"\n{history.height} predictions au total dans {PREDICTIONS_PARQUET}")
    print(
        "Commite ce fichier maintenant : l'historique git est la preuve d'anteriorite."
    )


if __name__ == "__main__":
    main()
