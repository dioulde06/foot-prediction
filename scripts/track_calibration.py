"""Track the calibration of the published predictions over time.

Only reconciled matches count: a prediction whose match has not been played
yet contributes nothing. Nothing is ever written back into the prediction
history -- the results are joined on the fly.

Run: uv run python -m scripts.track_calibration   (or: make track)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.app.publish import reconcile
from src.eval.baselines import coherent_odds_mask, market_baseline
from src.eval.metrics import (
    accuracy,
    brier_score_multiclass,
    calibration_bins,
    expected_calibration_error,
    log_loss,
)

LOG = logging.getLogger(__name__)
REPORT = Path("reports/tracking.md")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    joined = reconcile()
    if joined.is_empty():
        print("Aucune prediction publiee. Lance `make publish` d'abord.")
        return

    played = joined.filter(pl.col("result").is_not_null())
    print(f"{joined.height} predictions publiees, dont {played.height} deja jouees.")
    if played.is_empty():
        print(
            "Rien a mesurer pour l'instant : aucun match publie n'a encore ete joue.\n"
            "Relance apres la prochaine journee, une fois `make fetch` et\n"
            "`make merge` passes."
        )
        return

    played = played.filter(coherent_odds_mask(played, "avg"))
    outcomes = played["result"].to_list()
    model = played.select("p_home", "p_draw", "p_away").to_numpy()
    market = market_baseline(played, book="avg", method="power")

    scores = pl.DataFrame(
        [
            {
                "source": name,
                "log_loss": round(log_loss(probs, outcomes), 4),
                "brier": round(brier_score_multiclass(probs, outcomes), 4),
                "ece": round(expected_calibration_error(probs, outcomes), 4),
                "accuracy_info": round(accuracy(probs, outcomes), 4),
            }
            for name, probs in (("predictions publiees", model), ("market", market))
        ]
    )

    by_month = (
        played.with_columns(pl.col("date").dt.truncate("1mo").alias("mois"))
        .group_by("mois")
        .agg(pl.len().alias("n"))
        .sort("mois")
    )

    with pl.Config(tbl_hide_dataframe_shape=True, tbl_width_chars=160):
        print("\n=== Depuis la premiere publication ===")
        print(scores)
        print("\n=== Calibration par tranche ===")
        print(calibration_bins(model, outcomes))
        print("\n=== Volume par mois ===")
        print(by_month)

    gap = log_loss(model, outcomes) - log_loss(market, outcomes)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Suivi des predictions publiees",
                "",
                f"{played.height} matchs publies puis joues.",
                "",
                "| source | log-loss | Brier | ECE | accuracy* |",
                "|---|---|---|---|---|",
                *[
                    f"| {r['source']} | {r['log_loss']} | {r['brier']} | "
                    f"{r['ece']} | {r['accuracy_info']} |"
                    for r in scores.iter_rows(named=True)
                ],
                "",
                "\\* information seulement.",
                "",
                f"Ecart au marche : **{gap:+.4f}**.",
                "",
                "Les predictions sont publiees avant chaque journee et jamais",
                "modifiees ensuite. L'historique git de",
                "`predictions/predictions.parquet` date chaque publication.",
                "",
            ]
        )
        + "\n"
    )
    print(f"\nrapport : {REPORT}")
    if not np.isfinite(gap):
        raise SystemExit("ecart non fini : une probabilite nulle a ete publiee")


if __name__ == "__main__":
    main()
