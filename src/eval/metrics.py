"""Calibration metrics for 1X2 probabilities.

Primary metrics: log-loss, Brier, ECE. Accuracy is available for information
only and must never decide whether a model is kept.

Everything takes an (n, 3) probability matrix whose columns follow CLASSES,
and the realised outcomes as "H" / "D" / "A" codes. Bad input raises: a
silently renormalised or mislabelled probability set would corrupt every
number downstream without failing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

Probs = npt.NDArray[np.float64]

CLASSES: tuple[str, str, str] = ("H", "D", "A")

# Unequal on purpose. In 1X2 almost nothing is predicted above 65 % and very
# little below 20 %, so ten equal bins would leave the extremes too thin for
# their error to mean anything, and the ECE would be dominated by that noise.
DEFAULT_BINS: tuple[float, ...] = (0.0, 0.20, 0.30, 0.40, 0.50, 0.65, 1.0)
BIN_LABELS: tuple[str, ...] = ("0-20", "20-30", "30-40", "40-50", "50-65", "65+")

# A probability of 0 on an outcome that happens costs infinity. Clipped so the
# metric stays finite, but the penalty still dominates any honest score.
EPS = 1e-15


def _one_hot(outcomes: Sequence[str]) -> Probs:
    index = {code: i for i, code in enumerate(CLASSES)}
    matrix = np.zeros((len(outcomes), len(CLASSES)), dtype=np.float64)
    for row, code in enumerate(outcomes):
        if code not in index:
            raise ValueError(f"unknown outcome {code!r}, expected one of {CLASSES}")
        matrix[row, index[code]] = 1.0
    return matrix


def _check(probs: npt.ArrayLike, outcomes: Sequence[str]) -> tuple[Probs, Probs]:
    matrix = np.asarray(probs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(CLASSES):
        raise ValueError(
            f"probs must have shape (n, {len(CLASSES)}), got {matrix.shape}"
        )
    if matrix.shape[0] != len(outcomes):
        raise ValueError(
            f"length mismatch: {matrix.shape[0]} rows, {len(outcomes)} outcomes"
        )
    if bool((matrix < 0).any() or (matrix > 1).any()):
        raise ValueError("probabilities outside [0, 1]")
    totals = matrix.sum(axis=1)
    if not bool(np.allclose(totals, 1.0, atol=1e-6)):
        worst = totals[np.abs(totals - 1.0).argmax()]
        raise ValueError(f"every row must sum to 1, worst row sums to {worst:.6f}")
    return matrix, _one_hot(outcomes)


def log_loss(probs: npt.ArrayLike, outcomes: Sequence[str], eps: float = EPS) -> float:
    """Mean of -ln(p) taken on the realised outcome only."""
    matrix, realised = _check(probs, outcomes)
    picked = np.clip((matrix * realised).sum(axis=1), eps, 1.0)
    return float(-np.log(picked).mean())


def brier_score_multiclass(probs: npt.ArrayLike, outcomes: Sequence[str]) -> float:
    """Mean squared distance to the one-hot outcome, summed over the 3 classes.

    Ranges from 0 to 2. Unlike log-loss it stays bounded on a zero, so a model
    that is right on average but catastrophically wrong on a few matches looks
    better here than it deserves. Report both.
    """
    matrix, realised = _check(probs, outcomes)
    return float(((matrix - realised) ** 2).sum(axis=1).mean())


def calibration_bins(
    probs: npt.ArrayLike,
    outcomes: Sequence[str],
    bins: tuple[float, ...] = DEFAULT_BINS,
) -> pl.DataFrame:
    """Announced probability against observed frequency, per bin.

    Every class of every match contributes one point, so a set of n matches
    yields 3n points: this measures whether "30 %" means 30 % wherever it is
    said, not just on the model's favourite outcome.
    """
    matrix, realised = _check(probs, outcomes)
    flat_probs, flat_realised = matrix.ravel(), realised.ravel()
    slot = np.digitize(flat_probs, np.asarray(bins[1:-1]), right=False)

    rows = []
    for k, label in enumerate(BIN_LABELS):
        mask = slot == k
        rows.append(
            {
                "tranche": label,
                "n": int(mask.sum()),
                "predit_moyen": float(flat_probs[mask].mean())
                if bool(mask.any())
                else None,
                "observe": float(flat_realised[mask].mean())
                if bool(mask.any())
                else None,
            }
        )
    return pl.DataFrame(rows).with_columns(
        (pl.col("predit_moyen") - pl.col("observe")).alias("ecart")
    )


def expected_calibration_error(
    probs: npt.ArrayLike,
    outcomes: Sequence[str],
    bins: tuple[float, ...] = DEFAULT_BINS,
) -> float:
    """Bin-weighted mean absolute gap: "my probabilities are off by X points"."""
    frame = calibration_bins(probs, outcomes, bins).filter(pl.col("n") > 0)
    total = frame["n"].sum()
    return float((frame["n"] / total * frame["ecart"].abs()).sum())


def class_bias(probs: npt.ArrayLike, outcomes: Sequence[str]) -> pl.DataFrame:
    """Signed bias per class: announced mean minus observed frequency.

    The pooled signed bias is useless here: rows sum to 1 and so do the one-hot
    outcomes, so it is identically zero whatever the model does. Split by class
    it becomes informative, and the three values still sum to zero.
    """
    matrix, realised = _check(probs, outcomes)
    return pl.DataFrame(
        {
            "classe": list(CLASSES),
            "predit_moyen": matrix.mean(axis=0),
            "observe": realised.mean(axis=0),
        }
    ).with_columns((pl.col("predit_moyen") - pl.col("observe")).alias("biais"))


def accuracy(probs: npt.ArrayLike, outcomes: Sequence[str]) -> float:
    """Share of matches whose most likely outcome happened. Information only."""
    matrix, realised = _check(probs, outcomes)
    return float((matrix.argmax(axis=1) == realised.argmax(axis=1)).mean())
