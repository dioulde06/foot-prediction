"""Comparative table of the three baselines on the test season.

Phase 2 deliverable. Every later result is read against this table: a log-loss
without these three numbers beside it says nothing.

Run: uv run python scripts/run_baselines.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.baselines import (
    coherent_odds_mask,
    home_baseline,
    market_baseline,
    uniform_baseline,
)
from src.eval.metrics import (
    accuracy,
    brier_score_multiclass,
    calibration_bins,
    class_bias,
    expected_calibration_error,
    log_loss,
)
from src.eval.splits import TEST_SEASON, chronological_split

LOG = logging.getLogger(__name__)
PARQUET = Path("data/processed/matches.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default="avg", choices=["avg", "b365", "ps"])
    parser.add_argument(
        "--method",
        default="power",
        choices=["power", "multiplicative"],
        help="devigorisation method for the market baseline",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    matches = pl.read_parquet(PARQUET)
    train, _valid, test = chronological_split(matches)
    LOG.info(
        "train %d matchs, test %d matchs (%s)", train.height, test.height, TEST_SEASON
    )

    # Drop the corrupt source rows explicitly, and score every baseline on the
    # exact same matches so the comparison is fair.
    keep = coherent_odds_mask(test, args.book)
    dropped = test.height - int(keep.sum())
    if dropped:
        LOG.warning(
            "%d matchs ecartes du test : cotes %s absentes ou incoherentes",
            dropped,
            args.book,
        )
    test = test.filter(keep)

    outcomes = test["result"].to_list()
    predictions = {
        "uniform": uniform_baseline(train, n_rows=test.height),
        "home": home_baseline(n_rows=test.height),
        f"market ({args.book}, {args.method})": market_baseline(
            test, book=args.book, method=args.method
        ),
    }

    rows = [
        {
            "baseline": name,
            "log_loss": round(log_loss(probs, outcomes), 4),
            "brier": round(brier_score_multiclass(probs, outcomes), 4),
            "ece": round(expected_calibration_error(probs, outcomes), 4),
            "accuracy_info": round(accuracy(probs, outcomes), 4),
        }
        for name, probs in predictions.items()
    ]
    print(f"\n=== BASELINES sur {TEST_SEASON}, {test.height} matchs ===")
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_width_chars=120):
        print(pl.DataFrame(rows))
    print("\nRappel : ne rien savoir en 1X2 vaut ln(3) = 1.0986.")
    print("L'accuracy est affichee pour information et ne decide de rien.")

    market = predictions[f"market ({args.book}, {args.method})"]
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_width_chars=120):
        print(
            f"\n=== Calibration du marche ({args.book}, {args.method}) par tranche ==="
        )
        print(calibration_bins(market, outcomes))
        print(f"\n=== Biais par classe ({args.book}, {args.method}) ===")
        print(class_bias(market, outcomes))

    # Mean entropy of the market probabilities: the best empirical estimate of
    # the floor the log-loss cannot go below, since football outcomes are
    # genuinely uncertain. A model scoring under it is leaking.
    entropy = float(-(market * np.log(market)).sum(axis=1).mean())
    print(f"\nEntropie moyenne des probas marche : {entropy:.4f}")
    print("C'est le plancher approximatif du log-loss atteignable. Sous ce niveau,")
    print("cherche la fuite avant de te rejouir.")


if __name__ == "__main__":
    main()
