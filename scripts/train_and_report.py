"""Train, calibrate, and write the test-season report.

Run: uv run python -m scripts.train_and_report   (or: make train)
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import polars as pl

from src.eval.baselines import (
    coherent_odds_mask,
    home_baseline,
    market_baseline,
    uniform_baseline,
)
from src.eval.report import alarms, markdown_report, write_report
from src.models.calibrate import IsotonicCalibrator, TemperatureScaler
from src.models.train import (
    dataset_hash,
    load_config,
    predict,
    prepare_splits,
    save,
    train_model,
)

LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true", help="do not write models/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train, valid, test = prepare_splits()
    config = load_config()
    booster, metadata = train_model(train, valid, config)
    metadata["dataset_hash"] = dataset_hash(train, valid, test)

    # Calibrate on validation only, then apply to test. Both methods are
    # reported: isotonic is the textbook answer but overfits one season of
    # validation, so the one-parameter version is the one we keep.
    raw_valid = predict(booster, valid)
    raw_test = predict(booster, test)
    isotonic = IsotonicCalibrator().fit(raw_valid, valid.outcomes).transform(raw_test)
    scaler = TemperatureScaler().fit(raw_valid, valid.outcomes)
    tempered = scaler.transform(raw_test)
    LOG.info("temperature ajustee sur la validation : %.4f", scaler.temperature)
    metadata["temperature"] = round(scaler.temperature, 6)

    # Baselines on the same test rows, so the comparison is fair.
    with_odds = test.frame.join(
        pl.read_parquet("data/processed/matches.parquet").select(
            "date", "home_team", "away_team", *[f"odds_close_avg_{o}" for o in "hda"]
        ),
        on=["date", "home_team", "away_team"],
        how="left",
    )
    keep = coherent_odds_mask(with_odds, "avg").to_numpy()
    dropped = int((~keep).sum())
    if dropped:
        LOG.warning("%d matchs du test sans cotes exploitables", dropped)

    outcomes = [o for o, k in zip(test.outcomes, keep, strict=True) if k]
    market = market_baseline(with_odds.filter(keep), book="avg", method="power")
    named = {
        "modele brut": raw_test[keep],
        "modele + isotonique": isotonic[keep],
        "modele + temperature": tempered[keep],
        "uniform": uniform_baseline(train.frame, n_rows=len(outcomes)),
        "home": home_baseline(n_rows=len(outcomes)),
        "market": market,
    }
    entropy_floor = float(-(market * np.log(market)).sum(axis=1).mean())

    importances = pl.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance("gain").astype(float),
            "splits": booster.feature_importance("split").astype(float),
        }
    ).sort("gain", descending=True)

    # The kept model is the tempered one; the report's calibration tables
    # describe it.
    calibrated_test = tempered
    text = markdown_report(
        named, outcomes, calibrated_test[keep], metadata, importances, entropy_floor
    )
    path = write_report(text)
    LOG.info("wrote %s", path)

    if not args.no_save:
        save(booster, metadata)

    raised = alarms(calibrated_test[keep], outcomes)
    print("\n" + text)
    if raised:
        raise SystemExit("ALERTE CLAUDE.md regle 2 : " + " ; ".join(raised))


if __name__ == "__main__":
    main()
