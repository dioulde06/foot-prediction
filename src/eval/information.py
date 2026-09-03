"""The information test: does the model know anything the market does not?

The model sits 0.028 behind the closing market in log-loss. Before adding
features or data, the question worth an hour is whether the model carries
*any* information the market lacks. If it does not, no amount of tuning on
the same inputs will help, and only a new information source can.

The test is a geometric blend of the two probability vectors,

    p_k  ∝  model_k ** a  *  market_k ** b

fitted by maximum likelihood on out-of-sample predictions. `a` is the weight
the outcomes assign to the model once the market is known. A calibrated,
uninformative model gets a ≈ 0 and b ≈ 1. A model with information of its
own gets a > 0, and the blend then beats the market alone. The coefficient
comes with a bootstrap interval, and the blend's log-loss is measured
leave-one-season-out so the weights are never fitted on the season they score.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from src.eval.baselines import coherent_odds_mask, market_baseline
from src.eval.metrics import CLASSES, Probs, log_loss
from src.eval.walk_forward import season_order
from src.models.calibrate import TemperatureScaler
from src.models.train import as_split, load_config, predict, train_model

LOG = logging.getLogger(__name__)

FLOOR = 1e-6
# Ridge toward (a, b) = (0, 1), the "market only" answer. Small enough to be
# invisible on a season of matches; only there so that two identical inputs
# give a definite answer instead of a singular Hessian.
RIDGE = 1e-3
BOOTSTRAP_DRAWS = 1000


@dataclass(frozen=True)
class BlendFit:
    a: float
    b: float
    n: int
    iterations: int


def blend(model: npt.ArrayLike, market: npt.ArrayLike, a: float, b: float) -> Probs:
    """p_k ∝ model_k^a * market_k^b, renormalised."""
    logits = a * np.log(np.clip(model, FLOOR, None)) + b * np.log(
        np.clip(market, FLOOR, None)
    )
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return np.asarray(weights / weights.sum(axis=1, keepdims=True), dtype=np.float64)


def _labels(outcomes: Sequence[str]) -> npt.NDArray[np.int64]:
    index = {code: k for k, code in enumerate(CLASSES)}
    unknown = sorted(set(outcomes) - set(CLASSES))
    if unknown:
        raise ValueError(f"unknown outcome codes {unknown}, expected {CLASSES}")
    return np.array([index[o] for o in outcomes], dtype=np.int64)


def fit_blend(
    model: npt.ArrayLike,
    market: npt.ArrayLike,
    outcomes: Sequence[str],
    max_iter: int = 50,
    tol: float = 1e-10,
) -> BlendFit:
    """Maximum-likelihood (a, b) by Newton's method.

    The negative log-likelihood is convex in (a, b): it is the log-partition
    function of an exponential family minus a linear term. Its gradient is the
    expected minus observed features, its Hessian their covariance under the
    blend, so Newton converges in a handful of steps.
    """
    lm = np.log(np.clip(np.asarray(model, dtype=np.float64), FLOOR, None))
    lq = np.log(np.clip(np.asarray(market, dtype=np.float64), FLOOR, None))
    if lm.shape != lq.shape or lm.ndim != 2 or lm.shape[1] != len(CLASSES):
        raise ValueError(f"expected two (n, 3) arrays, got {lm.shape} and {lq.shape}")
    y = _labels(outcomes)
    if len(y) != lm.shape[0]:
        raise ValueError(f"length mismatch: {lm.shape[0]} rows, {len(y)} outcomes")
    n = lm.shape[0]
    rows = np.arange(n)
    observed = np.array([lm[rows, y].sum(), lq[rows, y].sum()])
    theta = np.array([0.0, 1.0])
    prior = np.array([0.0, 1.0])

    iterations, converged = 0, False
    while iterations < max_iter and not converged:
        iterations += 1
        logits = theta[0] * lm + theta[1] * lq
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        e_m = (p * lm).sum(axis=1)
        e_q = (p * lq).sum(axis=1)
        expected = np.array([e_m.sum(), e_q.sum()])
        gradient = expected - observed + 2 * RIDGE * n * (theta - prior)
        var_m = (p * lm**2).sum(axis=1) - e_m**2
        var_q = (p * lq**2).sum(axis=1) - e_q**2
        cov = (p * lm * lq).sum(axis=1) - e_m * e_q
        hessian = np.array([[var_m.sum(), cov.sum()], [cov.sum(), var_q.sum()]])
        hessian += 2 * RIDGE * n * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        theta = theta - step
        converged = float(np.abs(step).max()) < tol
    if not converged:
        raise RuntimeError(f"Newton did not converge in {max_iter} iterations")
    return BlendFit(a=float(theta[0]), b=float(theta[1]), n=n, iterations=iterations)


def bootstrap_a(
    model: Probs, market: Probs, outcomes: Sequence[str], draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    """95 % percentile interval on the model weight, resampling matches."""
    rng = np.random.default_rng(0)
    n = model.shape[0]
    outcomes = list(outcomes)
    values = []
    for _ in range(draws):
        idx = np.asarray(rng.integers(0, n, size=n), dtype=np.int64)
        drawn = [outcomes[int(i)] for i in idx]
        values.append(fit_blend(model[idx], market[idx], drawn).a)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def season_predictions(
    features: pl.DataFrame,
    odds: pl.DataFrame,
    test_season: str,
    config: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Out-of-sample calibrated model probabilities and market probabilities.

    Same protocol as walk_forward.evaluate_season: train before S-1, calibrate
    on S-1, predict S. Returns one row per test match with both probability
    vectors, so the blend can be fitted across seasons without retraining.
    """
    seasons = season_order(features)
    index = seasons.index(test_season)
    if index < 2:
        raise ValueError(
            f"{test_season} needs a training and a calibration season before it"
        )
    valid_season, train_seasons = seasons[index - 1], seasons[: index - 1]

    train = as_split("train", features.filter(pl.col("season").is_in(train_seasons)))
    valid = as_split("valid", features.filter(pl.col("season") == valid_season))
    test = as_split("test", features.filter(pl.col("season") == test_season))
    booster, _ = train_model(train, valid, config or load_config())
    scaler = TemperatureScaler().fit(predict(booster, valid), valid.outcomes)
    probs = scaler.transform(predict(booster, test))

    with_odds = test.frame.join(odds, on=["date", "home_team", "away_team"], how="left")
    keep = coherent_odds_mask(with_odds, "avg").to_numpy()
    market = market_baseline(with_odds.filter(keep), book="avg", method="power")
    kept = with_odds.filter(keep)
    return pl.DataFrame(
        {
            "season": [test_season] * kept.height,
            "date": kept["date"],
            "home_team": kept["home_team"],
            "away_team": kept["away_team"],
            "result": kept["result"],
            **{f"model_{c.lower()}": probs[keep][:, k] for k, c in enumerate(CLASSES)},
            **{f"market_{c.lower()}": market[:, k] for k, c in enumerate(CLASSES)},
        }
    )


def _arrays(frame: pl.DataFrame) -> tuple[Probs, Probs, list[str]]:
    model = frame.select([f"model_{c.lower()}" for c in CLASSES]).to_numpy()
    market = frame.select([f"market_{c.lower()}" for c in CLASSES]).to_numpy()
    return model, market, frame["result"].to_list()


def verdict(a_low: float, a_high: float) -> str:
    """Read the bootstrap interval on the model weight."""
    if a_low > 0:
        return (
            "The interval sits above zero: the model carries information the market "
            "lacks, and a blend beats the market."
        )
    if a_high < 0:
        return (
            "The interval sits below zero: given the market, the outcomes say to "
            "move *away* from the model. Where it disagrees with the market, it is "
            "wrong more often than right. Nothing in its inputs is missing from the "
            "market; only a new information source can close the gap."
        )
    return (
        "The interval includes zero: nothing in the model is missing from the "
        "market. Tuning on the same inputs cannot close the gap; only a new "
        "information source can."
    )


def information_test(predictions: pl.DataFrame) -> dict[str, Any]:
    """Pooled coefficients with a bootstrap interval, plus a per-season table.

    Per season, the blend weights are fitted on the *other* seasons' out-of-sample
    predictions, so its log-loss on the season is honest.
    """
    model, market, outcomes = _arrays(predictions)
    pooled = fit_blend(model, market, outcomes)
    low, high = bootstrap_a(model, market, outcomes)
    LOG.info("pooled: a=%.4f b=%.4f, a in [%.4f, %.4f]", pooled.a, pooled.b, low, high)

    rows = []
    for season in season_order(predictions):
        held = predictions.filter(pl.col("season") == season)
        rest = predictions.filter(pl.col("season") != season)
        weights = fit_blend(*_arrays(rest))
        m, q, y = _arrays(held)
        own = fit_blend(m, q, y)
        model_ll, market_ll = log_loss(m, y), log_loss(q, y)
        blend_ll = log_loss(blend(m, q, weights.a, weights.b), y)
        rows.append(
            {
                "test_season": season,
                "n": held.height,
                "a_in_season": round(own.a, 4),
                "a_from_other_seasons": round(weights.a, 4),
                "b_from_other_seasons": round(weights.b, 4),
                "model_log_loss": round(model_ll, 4),
                "market_log_loss": round(market_ll, 4),
                "blend_log_loss": round(blend_ll, 4),
                "blend_gain_on_market": round(market_ll - blend_ll, 4),
            }
        )
    return {
        "verdict": verdict(low, high),
        "a": pooled.a,
        "b": pooled.b,
        "a_low": low,
        "a_high": high,
        "n": pooled.n,
        "per_season": pl.DataFrame(rows),
    }
