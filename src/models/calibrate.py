"""Multiclass isotonic calibration, fitted on the validation split only.

LightGBM minimises the log-loss but early stopping and tree depth still leave
it overconfident. Isotonic regression fixes the mapping from announced to
observed probability without assuming any shape.

The rule that makes this honest: fit on validation, measure on test. Fitting
and measuring on the same rows produces a beautiful report and a useless model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression

from src.eval.metrics import CLASSES, Probs

# Isotonic can map a whole row to zero. Clipping before renormalising keeps the
# result a valid distribution instead of a division by zero.
FLOOR = 1e-6


class IsotonicCalibrator:
    """One isotonic regression per class, one-vs-rest, then renormalised.

    Per-class fitting is what makes it able to correct a systematic bias on one
    outcome -- typically the draw, which models under-predict. The price is that
    the three calibrated values no longer sum to 1, hence the renormalisation.
    """

    def __init__(self) -> None:
        self._models: list[IsotonicRegression] = []

    @property
    def fitted(self) -> bool:
        return bool(self._models)

    def fit(self, probs: npt.ArrayLike, outcomes: Sequence[str]) -> IsotonicCalibrator:
        matrix = np.asarray(probs, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(CLASSES):
            raise ValueError(
                f"probs must have shape (n, {len(CLASSES)}), got {matrix.shape}"
            )
        if matrix.shape[0] != len(outcomes):
            raise ValueError(
                f"length mismatch: {matrix.shape[0]} rows, {len(outcomes)} outcomes"
            )
        unknown = sorted(set(outcomes) - set(CLASSES))
        if unknown:
            raise ValueError(f"unknown outcome codes {unknown}, expected {CLASSES}")

        self._models = []
        for index, code in enumerate(CLASSES):
            target = np.array([1.0 if o == code else 0.0 for o in outcomes])
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(matrix[:, index], target)
            self._models.append(model)
        return self

    def transform(self, probs: npt.ArrayLike) -> Probs:
        if not self.fitted:
            raise RuntimeError(
                "calibrator is not fitted; call fit on the validation split"
            )
        matrix = np.asarray(probs, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(CLASSES):
            raise ValueError(
                f"probs must have shape (n, {len(CLASSES)}), got {matrix.shape}"
            )

        calibrated = np.column_stack(
            [
                model.predict(matrix[:, index])
                for index, model in enumerate(self._models)
            ]
        )
        calibrated = np.clip(calibrated, FLOOR, None)
        return np.asarray(
            calibrated / calibrated.sum(axis=1, keepdims=True), dtype=np.float64
        )

    def fit_transform(self, probs: npt.ArrayLike, outcomes: Sequence[str]) -> Probs:
        """Only for inspecting the fit itself. Never for reporting a score."""
        return self.fit(probs, outcomes).transform(probs)


class TemperatureScaler:
    """One-parameter calibration: p_i ** (1/T), renormalised.

    T above 1 softens the distribution, T below 1 sharpens it. Fitted by
    minimising the validation log-loss over a bounded interval.

    The reason this exists next to the isotonic version: isotonic is
    non-parametric and, on the ~1 750 rows of one validation season, flexible
    enough to fit that season's noise. Measured on 2025-26 it made the test
    log-loss worse, 1.0049 to 1.0193, while barely moving the ECE. One
    parameter cannot overfit a season, so it transfers.
    """

    LOW, HIGH = 0.25, 4.0

    def __init__(self) -> None:
        self._temperature: float | None = None

    @property
    def fitted(self) -> bool:
        return self._temperature is not None

    @property
    def temperature(self) -> float:
        if self._temperature is None:
            raise RuntimeError("scaler is not fitted; call fit on the validation split")
        return self._temperature

    @staticmethod
    def _apply(matrix: Probs, temperature: float) -> Probs:
        powered = np.clip(matrix, FLOOR, None) ** (1.0 / temperature)
        return np.asarray(
            powered / powered.sum(axis=1, keepdims=True), dtype=np.float64
        )

    def fit(
        self, probs: npt.ArrayLike, outcomes: Sequence[str], tol: float = 1e-6
    ) -> TemperatureScaler:
        from src.eval.metrics import log_loss

        matrix = np.asarray(probs, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(CLASSES):
            raise ValueError(
                f"probs must have shape (n, {len(CLASSES)}), got {matrix.shape}"
            )
        if matrix.shape[0] != len(outcomes):
            raise ValueError(
                f"length mismatch: {matrix.shape[0]} rows, {len(outcomes)} outcomes"
            )

        # Golden-section search: the log-loss is unimodal in T and this needs
        # no solver dependency.
        phi = (5**0.5 - 1) / 2
        low, high = self.LOW, self.HIGH
        left, right = high - phi * (high - low), low + phi * (high - low)
        f_left = log_loss(self._apply(matrix, left), outcomes)
        f_right = log_loss(self._apply(matrix, right), outcomes)
        while high - low > tol:
            if f_left < f_right:
                high, right, f_right = right, left, f_left
                left = high - phi * (high - low)
                f_left = log_loss(self._apply(matrix, left), outcomes)
            else:
                low, left, f_left = left, right, f_right
                right = low + phi * (high - low)
                f_right = log_loss(self._apply(matrix, right), outcomes)
        self._temperature = 0.5 * (low + high)
        return self

    def transform(self, probs: npt.ArrayLike) -> Probs:
        matrix = np.asarray(probs, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(CLASSES):
            raise ValueError(
                f"probs must have shape (n, {len(CLASSES)}), got {matrix.shape}"
            )
        return self._apply(matrix, self.temperature)
