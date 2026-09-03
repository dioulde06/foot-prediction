"""Walk-forward validation, season by season.

One test-set number proves nothing: the season-to-season variance of the
log-loss is larger than the gap between two decent models. What matters is
whether the model is *stable*, so this reports every season separately and
never an average.

For each test season S: train on everything before S-1, calibrate on S-1,
test on S. The calibration season is always the one immediately before the
test season, so the model is never tuned on data it will be scored on.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from src.eval.baselines import coherent_odds_mask, market_baseline
from src.eval.metrics import (
    accuracy,
    brier_score_multiclass,
    expected_calibration_error,
    log_loss,
)
from src.features.build import build_features
from src.models.calibrate import TemperatureScaler
from src.models.train import (
    MATCHES_PARQUET,
    as_split,
    load_config,
    predict,
    train_model,
)

LOG = logging.getLogger(__name__)

# Fewest training seasons worth trying. Below this the sample is too small for
# the early stopping itself to mean anything.
MIN_TRAIN_SEASONS = 1


def season_order(frame: pl.DataFrame) -> list[str]:
    """Seasons in chronological order, derived rather than hardcoded."""
    return (
        frame.group_by("season")
        .agg(pl.col("date").min().alias("start"))
        .sort("start")["season"]
        .to_list()
    )


# A season with fewer matches than this share of the fullest one is still being
# played. It is never a test season: half a season says nothing about stability.
COMPLETE_SHARE = 0.9


def complete_seasons(frame: pl.DataFrame) -> list[str]:
    """Seasons in chronological order, minus the one still being played."""
    counts = frame.group_by("season").len()
    floor = COMPLETE_SHARE * float(counts["len"].to_numpy().max())
    full = set(counts.filter(pl.col("len") >= floor)["season"].to_list())
    return [s for s in season_order(frame) if s in full]


def prepare(matches: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Features and the odds needed for the market comparison.

    build_features is called once at the global cutoff, which is safe: every
    row reads only matches played strictly before it, whatever the cutoff.
    """
    if matches is None:
        matches = pl.read_parquet(MATCHES_PARQUET).sort("date")
    features = build_features(matches, matches["date"].max()).filter(  # type: ignore[arg-type]
        pl.col("elo_diff").is_not_null()
    )
    odds = matches.select(
        "date", "home_team", "away_team", *[f"odds_close_avg_{o}" for o in "hda"]
    )
    return features, odds


def evaluate_season(
    features: pl.DataFrame,
    odds: pl.DataFrame,
    test_season: str,
    n_train_seasons: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train, calibrate and score one season. Returns one row of the table."""
    seasons = season_order(features)
    index = seasons.index(test_season)
    if index < 2:
        raise ValueError(
            f"{test_season} has only {index} earlier seasons; needs at least one "
            "to train on and one to calibrate on"
        )
    valid_season = seasons[index - 1]
    available = seasons[: index - 1]
    train_seasons = (
        available if n_train_seasons is None else available[-n_train_seasons:]
    )
    if len(train_seasons) < MIN_TRAIN_SEASONS:
        raise ValueError(f"no training seasons left for {test_season}")

    train = as_split("train", features.filter(pl.col("season").is_in(train_seasons)))
    valid = as_split("valid", features.filter(pl.col("season") == valid_season))
    test = as_split("test", features.filter(pl.col("season") == test_season))

    booster, metadata = train_model(train, valid, config or load_config())
    scaler = TemperatureScaler().fit(predict(booster, valid), valid.outcomes)
    probs = scaler.transform(predict(booster, test))

    # Market on the same rows, so the gap is a like-for-like comparison.
    with_odds = test.frame.join(odds, on=["date", "home_team", "away_team"], how="left")
    keep = coherent_odds_mask(with_odds, "avg").to_numpy()
    outcomes = [o for o, k in zip(test.outcomes, keep, strict=True) if k]
    market = market_baseline(with_odds.filter(keep), book="avg", method="power")
    model = probs[keep]

    model_ll = log_loss(model, outcomes)
    market_ll = log_loss(market, outcomes)
    return {
        "test_season": test_season,
        "n_train_seasons": len(train_seasons),
        "n_train": train.frame.height,
        "n_test": len(outcomes),
        "log_loss": round(model_ll, 4),
        "brier": round(brier_score_multiclass(model, outcomes), 4),
        "ece": round(expected_calibration_error(model, outcomes), 4),
        "accuracy_info": round(accuracy(model, outcomes), 4),
        "market_log_loss": round(market_ll, 4),
        "gap_to_market": round(model_ll - market_ll, 4),
        "temperature": round(scaler.temperature, 3),
        "best_iteration": metadata["best_iteration"],
    }


def walk_forward(
    features: pl.DataFrame,
    odds: pl.DataFrame,
    test_seasons: list[str] | None = None,
    n_train_seasons: int | None = None,
) -> pl.DataFrame:
    targets = test_seasons or complete_seasons(features)[2:]
    config = load_config()
    rows = []
    for season in targets:
        LOG.info("--- test %s ---", season)
        rows.append(evaluate_season(features, odds, season, n_train_seasons, config))
    return pl.DataFrame(rows)


def saturation(
    features: pl.DataFrame, odds: pl.DataFrame, test_season: str
) -> pl.DataFrame:
    """Does more history help, and where does the gain stop?

    Prompt 5.1 asked for 2, 4 and 6 training seasons. With six seasons of data
    and one of them spent on calibration and one on test, four is the maximum
    available, so this sweeps 1 to 4 instead.
    """
    seasons = season_order(features)
    most = seasons.index(test_season) - 1
    config = load_config()
    rows = [
        evaluate_season(features, odds, test_season, n, config)
        for n in range(MIN_TRAIN_SEASONS, most + 1)
    ]
    return pl.DataFrame(rows)
