"""The three reference baselines. A score without them means nothing.

Recomputed whenever the dataset changes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from src.eval.metrics import CLASSES, Probs


def devig_multiplicative(odds: npt.ArrayLike) -> Probs:
    """Turn decimal odds into probabilities by stripping the bookmaker margin.

        implied_i = 1 / odds_i
        p_i       = implied_i / sum_j implied_j

    Naive on purpose: it assumes the margin is spread proportionally across the
    three outcomes. Bookmakers load more of it on longshots (the
    favourite-longshot bias), so this method overestimates them. Known ceiling;
    Shin's method or the odds-ratio method would do better.
    """
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
    return np.asarray(implied / totals[:, None], dtype=np.float64)


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


def _odds_columns(frame: pl.DataFrame, book: str) -> list[str]:
    columns = [f"odds_close_{book}_{outcome}" for outcome in "hda"]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"unknown odds columns {missing}, book {book!r}")
    return columns


def market_baseline(frame: pl.DataFrame, book: str = "avg") -> Probs:
    """Devigorised closing odds. The high reference, probably unbeatable.

    Defaults to the market average: it is the only book with full coverage on
    the 2025-26 test season, where Pinnacle drops to about 50 %.
    """
    columns = _odds_columns(frame, book)
    subset = frame.select(columns)
    nulls = subset.null_count().row(0)
    if any(nulls):
        raise ValueError(
            f"missing odds in book {book!r}: {dict(zip(columns, nulls, strict=True))}; "
            "filter with coherent_odds_mask first"
        )
    return devig_multiplicative(subset.to_numpy())


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
