"""The three reference baselines. A score without them means nothing.

Recomputed whenever the dataset changes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from src.eval.metrics import CLASSES, Probs

DEVIG_METHODS = ("multiplicative", "power")


def _implied(odds: npt.ArrayLike) -> tuple[Probs, Probs]:
    """Raw implied probabilities and their row sums, with the sanity checks."""
    matrix = np.asarray(odds, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(CLASSES):
        raise ValueError(
            f"odds must have shape (n, {len(CLASSES)}), got {matrix.shape}"
        )
    if not bool(np.isfinite(matrix).all()) or bool((matrix <= 1.0).any()):
        raise ValueError("odds must be decimal odds above 1, with a positive payout")

    implied = 1.0 / matrix
    totals = implied.sum(axis=1)
    if bool((totals < 1.0).any()):
        bad = np.flatnonzero(totals < 1.0)
        raise ValueError(
            f"implied probabilities sum below 1 on rows {bad.tolist()} "
            f"(sums {totals[bad].round(4).tolist()}): a book never gives money away, "
            "so these odds are corrupt"
        )
    return implied, totals


def devig_multiplicative(odds: npt.ArrayLike) -> Probs:
    """Strip the margin by scaling every implied probability by the same factor.

        implied_i = 1 / odds_i
        p_i       = implied_i / sum_j implied_j

    Naive on purpose: it assumes the margin is spread proportionally across the
    three outcomes. It is not -- bookmakers load more of it on longshots (the
    favourite-longshot bias) -- so this method overestimates longshots and
    underestimates favourites. Measured on 2025-26: +3.7 points on the 0-20 %
    bin and -8.3 points on the 65 %+ bin, the distortion tracking the size of
    the margin. Kept for comparison; prefer devig_power.
    """
    implied, totals = _implied(odds)
    return np.asarray(implied / totals[:, None], dtype=np.float64)


def devig_power(odds: npt.ArrayLike, tol: float = 1e-12, max_iter: int = 200) -> Probs:
    """Strip the margin by raising every implied probability to a power.

        p_i = implied_i ** k,  with k such that sum_j p_j = 1

    Every implied probability is below 1, so a k above 1 shrinks small numbers
    proportionally more than large ones. That removes more margin from the
    longshots than from the favourite, which is where the margin actually sits,
    and so corrects the bias of the multiplicative method.

    k is found by bisection on [1, 50]: the row sum is strictly decreasing in k,
    above 1 at k = 1 whenever there is a margin, and tends to 0, so a single
    root always exists. No solver dependency needed.
    """
    implied, totals = _implied(odds)
    low = np.ones_like(totals)
    high = np.full_like(totals, 50.0)
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        excess = (implied ** mid[:, None]).sum(axis=1) - 1.0
        needs_more = excess > 0.0
        low = np.where(needs_more, mid, low)
        high = np.where(needs_more, high, mid)
        if bool(np.all(high - low < tol)):
            break

    probs = implied ** (0.5 * (low + high))[:, None]
    # Residual bisection error is around 1e-12; this only tidies the row sums.
    return np.asarray(probs / probs.sum(axis=1, keepdims=True), dtype=np.float64)


def _odds_columns(frame: pl.DataFrame, book: str) -> list[str]:
    columns = [f"odds_close_{book}_{outcome}" for outcome in "hda"]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"unknown odds columns {missing}, book {book!r}")
    return columns


def coherent_odds_mask(frame: pl.DataFrame, book: str) -> pl.Series:
    """Rows whose odds are present and sum to at least 1 once inverted.

    Used to drop the handful of corrupt source rows before scoring, so the drop
    is explicit and counted rather than silent.
    """
    columns = _odds_columns(frame, book)
    implied = [1.0 / pl.col(c) for c in columns]
    return (
        frame.select(
            (pl.all_horizontal([pl.col(c).is_not_null() for c in columns]))
            & (sum(implied[1:], implied[0]) >= 1.0)
        )
        .to_series()
        .fill_null(False)
    )


def market_baseline(
    frame: pl.DataFrame, book: str = "avg", method: str = "power"
) -> Probs:
    """Devigorised closing odds. The high reference, probably unbeatable.

    Defaults to the market average, the only book with full coverage on the
    2025-26 test season where Pinnacle drops to about 50 %, and to the power
    method, which measurably beats the multiplicative one on calibration.
    """
    if method not in DEVIG_METHODS:
        raise ValueError(
            f"unknown devig method {method!r}, expected one of {DEVIG_METHODS}"
        )
    columns = _odds_columns(frame, book)
    subset = frame.select(columns)
    nulls = subset.null_count().row(0)
    if any(nulls):
        raise ValueError(
            f"missing odds in book {book!r}: {dict(zip(columns, nulls, strict=True))}; "
            "filter with coherent_odds_mask first"
        )
    odds = subset.to_numpy()
    if method == "multiplicative":
        return devig_multiplicative(odds)
    return devig_power(odds)


def uniform_baseline(train: pl.DataFrame, n_rows: int) -> Probs:
    """Fixed historical 1X2 frequencies, measured on the training seasons only.

    Fitting them on the whole dataset would let the baseline itself know the
    test season, making it artificially strong and harder to beat for the wrong
    reason. This departs from the letter of prompt 2.1 on purpose.
    """
    counts = {
        row["result"]: row["count"]
        for row in train["result"].value_counts().iter_rows(named=True)
    }
    missing = [code for code in CLASSES if code not in counts]
    if missing:
        raise ValueError(f"outcome(s) never observed in train: {missing}")
    total = sum(counts.values())
    row = np.array([counts[code] / total for code in CLASSES], dtype=np.float64)
    return np.tile(row, (n_rows, 1))


def home_baseline(n_rows: int) -> Probs:
    """Always the home win, as a hard (1, 0, 0) call.

    Deliberately not softened: under log-loss a certain-and-wrong prediction is
    catastrophic, and showing that next to the other baselines is the whole
    point. Its log-loss is a warning, never a target.
    """
    return np.tile([1.0, 0.0, 0.0], (n_rows, 1))
