"""Calibration tests. The one that matters is that it fixes overconfidence."""

import numpy as np
import pytest

from src.eval.metrics import expected_calibration_error, log_loss
from src.models.calibrate import IsotonicCalibrator

RNG = np.random.default_rng(7)


def _overconfident(n: int = 3000) -> tuple[np.ndarray, list[str]]:
    """Truth is 45/25/30; the model announces a sharpened version of it."""
    truth = np.array([0.45, 0.25, 0.30])
    outcomes = [["H", "D", "A"][i] for i in RNG.choice(3, size=n, p=truth)]
    sharp = truth**2.2
    sharp = sharp / sharp.sum()
    probs = np.tile(sharp, (n, 1))
    return probs, outcomes


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        IsotonicCalibrator().transform(np.full((2, 3), 1 / 3))


def test_calibrated_rows_sum_to_one() -> None:
    probs, outcomes = _overconfident()
    out = IsotonicCalibrator().fit(probs, outcomes).transform(probs)
    assert out.sum(axis=1) == pytest.approx(np.ones(len(outcomes)))


def test_calibration_reduces_the_error_of_an_overconfident_model() -> None:
    probs, outcomes = _overconfident()
    calibrator = IsotonicCalibrator().fit(probs, outcomes)
    out = calibrator.transform(probs)
    assert expected_calibration_error(out, outcomes) < expected_calibration_error(
        probs, outcomes
    )
    assert log_loss(out, outcomes) < log_loss(probs, outcomes)


def test_calibration_is_monotone_in_each_class() -> None:
    probs = RNG.dirichlet([2, 2, 2], size=800)
    outcomes = [
        ["H", "D", "A"][i] for i in RNG.choice(3, size=800, p=[0.45, 0.25, 0.30])
    ]
    calibrator = IsotonicCalibrator().fit(probs, outcomes)
    grid = np.linspace(0.01, 0.99, 40)
    for index in range(3):
        probe = np.full((40, 3), 1 / 3)
        probe[:, index] = grid
        # The per-class isotonic map itself must be non-decreasing, so read it
        # before the renormalisation mixes the classes together.
        mapped = calibrator._models[index].predict(grid)
        assert np.all(np.diff(mapped) >= -1e-12), index


def test_a_fit_on_one_split_transfers_to_another() -> None:
    valid_probs, valid_outcomes = _overconfident(1500)
    test_probs, _ = _overconfident(500)
    calibrator = IsotonicCalibrator().fit(valid_probs, valid_outcomes)
    out = calibrator.transform(test_probs)
    assert out.shape == test_probs.shape
    assert out.sum(axis=1) == pytest.approx(np.ones(500))


def test_fit_rejects_an_unknown_outcome_code() -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        IsotonicCalibrator().fit(np.full((2, 3), 1 / 3), ["H", "W"])


def test_fit_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        IsotonicCalibrator().fit(np.full((3, 3), 1 / 3), ["H"])


# --- temperature scaling --------------------------------------------------


def test_temperature_transform_before_fit_raises() -> None:
    from src.models.calibrate import TemperatureScaler

    with pytest.raises(RuntimeError, match="not fitted"):
        TemperatureScaler().transform(np.full((2, 3), 1 / 3))


def test_temperature_rows_sum_to_one() -> None:
    from src.models.calibrate import TemperatureScaler

    probs, outcomes = _overconfident(1200)
    out = TemperatureScaler().fit(probs, outcomes).transform(probs)
    assert out.sum(axis=1) == pytest.approx(np.ones(len(outcomes)))


def test_temperature_above_one_softens_an_overconfident_model() -> None:
    from src.models.calibrate import TemperatureScaler

    probs, outcomes = _overconfident(4000)
    scaler = TemperatureScaler().fit(probs, outcomes)
    assert scaler.temperature > 1.0
    out = scaler.transform(probs)
    # Softening pulls the highest probability down towards the others.
    assert out[:, 0].max() < probs[:, 0].max()
    assert log_loss(out, outcomes) < log_loss(probs, outcomes)


def test_temperature_of_one_leaves_a_calibrated_model_alone() -> None:
    from src.models.calibrate import TemperatureScaler

    truth = np.array([0.45, 0.25, 0.30])
    outcomes = [["H", "D", "A"][i] for i in RNG.choice(3, size=6000, p=truth)]
    probs = np.tile(truth, (6000, 1))
    scaler = TemperatureScaler().fit(probs, outcomes)
    assert scaler.temperature == pytest.approx(1.0, abs=0.12)


def test_temperature_preserves_the_ranking_of_the_classes() -> None:
    from src.models.calibrate import TemperatureScaler

    probs, outcomes = _overconfident(800)
    out = TemperatureScaler().fit(probs, outcomes).transform(probs)
    assert np.array_equal(out.argmax(axis=1), probs.argmax(axis=1))
