"""The information test: does the model know anything the market does not?

Run: uv run python -m scripts.run_information   (or: make information)
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.eval.information import information_test, season_predictions
from src.eval.walk_forward import complete_seasons, prepare
from src.models.train import load_config

LOG = logging.getLogger(__name__)

PREDICTIONS = Path("reports/oos_predictions.parquet")
REPORT = Path("reports/information.md")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    features, odds = prepare()
    config = load_config()
    frames = []
    for season in complete_seasons(features)[2:]:
        LOG.info("--- out-of-sample predictions for %s ---", season)
        frames.append(season_predictions(features, odds, season, config))
    predictions = pl.concat(frames)
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(PREDICTIONS)

    result = information_test(predictions)
    table: pl.DataFrame = result["per_season"]
    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_cols=-1,
        tbl_width_chars=200,
    ):
        table_md = str(table)

    REPORT.write_text(
        "# Information test\n\n"
        "Geometric blend `p_k ∝ model_k^a · market_k^b`, fitted by maximum "
        "likelihood on out-of-sample predictions (train before S-1, calibrate on "
        "S-1, predict S). `a` is the weight the outcomes give the model once the "
        "market is known.\n\n"
        f"Pooled over {result['n']} matches: **a = {result['a']:.4f}**, "
        f"b = {result['b']:.4f}, 95 % bootstrap interval on a "
        f"[{result['a_low']:.4f}, {result['a_high']:.4f}].\n\n"
        f"{result['verdict']}\n\n"
        "Per season, the blend weights come from the other three seasons, so its "
        "log-loss is honest.\n\n"
        f"{table_md}\n"
    )
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
