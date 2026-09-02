"""LightGBM training on a strictly chronological split.

No random split, no shuffle, no non-temporal cross-validation. Train on the
oldest seasons, tune on the next one, and never touch the last one until the
report. Hyperparameters live in configs/lightgbm.yaml so a change of model is a
visible diff rather than an edited constant.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
import yaml

from src.eval.metrics import CLASSES, Probs
from src.eval.splits import (
    TEST_SEASON,
    TRAIN_SEASONS,
    VALID_SEASON,
    chronological_split,
)
from src.features.build import FEATURE_COLUMNS, build_features

LOG = logging.getLogger(__name__)

MATCHES_PARQUET = Path("data/processed/matches.parquet")
CONFIG_PATH = Path("configs/lightgbm.yaml")
MODEL_DIR = Path("models")

DAYS_PER_SEASON = 365.25


@dataclass(frozen=True)
class Split:
    """One part of the chronological split, ready for LightGBM."""

    name: str
    features: Probs
    labels: npt.NDArray[np.int64]
    outcomes: list[str]
    frame: pl.DataFrame


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load(path.read_text())
    for section in ("params", "training"):
        if section not in config:
            raise ValueError(f"{path} is missing the {section!r} section")
    return config


def sample_weights(
    dates: pl.Series, reference: object, half_life_seasons: float
) -> npt.NDArray[np.float64]:
    """Exponential decay: a match one half-life older counts half as much.

    Raising the weight of recent football is a bet that the game drifts. Set
    the half-life high and the bet is mild; set it to zero and you would be
    saying only the last match matters.
    """
    if half_life_seasons <= 0:
        raise ValueError(f"half_life_seasons must be positive, got {half_life_seasons}")
    age_days = np.array([(reference - d).days for d in dates], dtype=np.float64)
    if bool((age_days < 0).any()):
        raise ValueError("a training match is dated after the reference date")
    return np.asarray(0.5 ** (age_days / (DAYS_PER_SEASON * half_life_seasons)))


def as_split(name: str, frame: pl.DataFrame) -> Split:
    index = {code: i for i, code in enumerate(CLASSES)}
    outcomes = frame["result"].to_list()
    return Split(
        name=name,
        features=frame.select(FEATURE_COLUMNS).to_numpy(),
        labels=np.array([index[o] for o in outcomes], dtype=np.int64),
        outcomes=outcomes,
        frame=frame,
    )


def prepare_splits(matches: pl.DataFrame | None = None) -> tuple[Split, Split, Split]:
    """Features for the three chronological parts.

    build_features is called once with the test cutoff, which is safe by
    construction: every row only reads matches played strictly before it.
    """
    if matches is None:
        matches = pl.read_parquet(MATCHES_PARQUET).sort("date")

    cutoff = matches["date"].max()
    features = build_features(matches, cutoff)  # type: ignore[arg-type]
    # Rows without a full rolling window carry nulls in every form feature.
    # LightGBM handles nulls, but a row with no history at all is noise.
    features = features.filter(pl.col("elo_diff").is_not_null())

    train, valid, test = chronological_split(features)
    LOG.info(
        "train %d (%s) | valid %d (%s) | test %d (%s)",
        train.height,
        ", ".join(TRAIN_SEASONS),
        valid.height,
        VALID_SEASON,
        test.height,
        TEST_SEASON,
    )
    return as_split("train", train), as_split("valid", valid), as_split("test", test)


def dataset_hash(*splits: Split) -> str:
    """Fingerprint of exactly what was trained on, for the metadata."""
    digest = hashlib.sha256()
    for split in splits:
        digest.update(split.name.encode())
        digest.update(np.ascontiguousarray(split.features, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(split.labels).tobytes())
    return digest.hexdigest()[:16]


def train_model(
    train: Split, valid: Split, config: dict[str, Any] | None = None
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Fit on train, early-stop on the validation log-loss."""
    config = config or load_config()
    training = config["training"]

    reference = train.frame["date"].max()
    weights = sample_weights(
        train.frame["date"], reference, training["half_life_seasons"]
    )
    LOG.info(
        "sample weights: oldest %.3f, newest %.3f, half-life %.1f seasons",
        weights.min(),
        weights.max(),
        training["half_life_seasons"],
    )

    train_set = lgb.Dataset(
        train.features,
        label=train.labels,
        weight=weights,
        feature_name=list(FEATURE_COLUMNS),
    )
    valid_set = lgb.Dataset(
        valid.features,
        label=valid.labels,
        reference=train_set,
        feature_name=list(FEATURE_COLUMNS),
    )

    evals: dict[str, dict[str, list[float]]] = {}
    booster = lgb.train(
        config["params"],
        train_set,
        num_boost_round=training["num_boost_round"],
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(training["early_stopping_rounds"], verbose=False),
            lgb.record_evaluation(evals),
            lgb.log_evaluation(0),
        ],
    )
    best = booster.best_iteration
    LOG.info(
        "stopped at iteration %d / %d, valid log-loss %.4f",
        best,
        training["num_boost_round"],
        evals["valid"]["multi_logloss"][best - 1],
    )

    metadata = {
        "features": list(FEATURE_COLUMNS),
        "train_seasons": list(TRAIN_SEASONS),
        "valid_season": VALID_SEASON,
        "test_season": TEST_SEASON,
        "train_dates": [str(train.frame["date"].min()), str(train.frame["date"].max())],
        "valid_dates": [str(valid.frame["date"].min()), str(valid.frame["date"].max())],
        "n_train": train.frame.height,
        "n_valid": valid.frame.height,
        "best_iteration": best,
        "valid_log_loss": round(evals["valid"]["multi_logloss"][best - 1], 6),
        "config": config,
    }
    return booster, metadata


def predict(booster: lgb.Booster, split: Split) -> Probs:
    raw = booster.predict(split.features, num_iteration=booster.best_iteration)
    return np.asarray(raw, dtype=np.float64)


def save(
    booster: lgb.Booster, metadata: dict[str, Any], directory: Path = MODEL_DIR
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(
        str(directory / "model.txt"), num_iteration=booster.best_iteration
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    LOG.info("wrote %s and metadata.json", directory / "model.txt")
