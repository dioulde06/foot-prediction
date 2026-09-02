"""Known-value checks on the calibration metrics.

Every expectation here is a number that can be derived by hand, so a failure
means the metric is wrong rather than the test being stale.
"""

import math

import numpy as np
import pytest

from src.eval.metrics import (
    CLASSES,
    brier_score_multiclass,
    calibration_bins,
    class_bias,
    expected_calibration_error,
    log_loss,
)


def test_log_loss_of_a_certain_and_correct_prediction_is_zero() -> None:
    probs = np.array([[1.0, 0.0, 0.0]])
    assert log_loss(probs, ["H"]) == pytest.approx(0.0)


def test_log_loss_of_the_uniform_prediction_is_ln_3() -> None:
    probs = np.full((4, 3), 1 / 3)
    assert log_loss(probs, ["H", "D", "A", "H"]) == pytest.approx(math.log(3))


def test_log_loss_only_looks_at_the_realised_outcome() -> None:
    # 0.5 on the realised outcome, whatever the split of the remaining 0.5.
    a = np.array([[0.5, 0.3, 0.2]])
    b = np.array([[0.5, 0.1, 0.4]])
    assert log_loss(a, ["H"]) == pytest.approx(log_loss(b, ["H"]))
    assert log_loss(a, ["H"]) == pytest.approx(-math.log(0.5))


def test_log_loss_of_a_zero_on_the_realised_outcome_is_huge_not_infinite() -> None:
    # Clipped at eps so the metric stays finite, but the penalty must dominate.
    probs = np.array([[0.0, 0.5, 0.5]])
    assert log_loss(probs, ["H"]) > 30


def test_log_loss_punishes_confident_error_more_than_it_rewards_confident_truth() -> (
    None
):
    base = np.array([[0.5, 0.25, 0.25]])
    bold_right = np.array([[0.9, 0.05, 0.05]])
    bold_wrong = np.array([[0.1, 0.45, 0.45]])
    gain = log_loss(base, ["H"]) - log_loss(bold_right, ["H"])
    cost = log_loss(bold_wrong, ["H"]) - log_loss(base, ["H"])
    assert cost > gain


def test_brier_of_a_certain_and_correct_prediction_is_zero() -> None:
    assert brier_score_multiclass(np.array([[1.0, 0.0, 0.0]]), ["H"]) == pytest.approx(
        0.0
    )


def test_brier_of_a_certain_and_wrong_prediction_is_two() -> None:
    assert brier_score_multiclass(np.array([[1.0, 0.0, 0.0]]), ["A"]) == pytest.approx(
        2.0
    )


def test_brier_of_the_uniform_prediction_is_two_thirds() -> None:
    probs = np.full((3, 3), 1 / 3)
    assert brier_score_multiclass(probs, ["H", "D", "A"]) == pytest.approx(2 / 3)


def test_brier_unlike_log_loss_stays_bounded_on_a_zero() -> None:
    probs = np.array([[0.0, 0.5, 0.5]])
    assert brier_score_multiclass(probs, ["H"]) == pytest.approx(1.5)


def _perfectly_calibrated(rate: float, n: int) -> tuple[np.ndarray, list[str]]:
    """n predictions at `rate` on H, of which exactly rate*n actually end H."""
    probs = np.tile([rate, (1 - rate) / 2, (1 - rate) / 2], (n, 1))
    wins = round(rate * n)
    return probs, ["H"] * wins + ["D"] * (n - wins)


def test_ece_of_a_perfectly_calibrated_set_is_zero_on_the_h_bin() -> None:
    probs, outcomes = _perfectly_calibrated(0.45, 200)
    bins = calibration_bins(probs, outcomes)
    row = bins.filter(bins["tranche"] == "40-50")
    assert row["n"][0] == 200
    assert row["predit_moyen"][0] == pytest.approx(0.45)
    assert row["observe"][0] == pytest.approx(0.45)
    assert row["ecart"][0] == pytest.approx(0.0)


def test_ece_detects_overconfidence() -> None:
    # Announces 0.90 on H but H only happens half the time: 40 points off.
    probs = np.tile([0.90, 0.05, 0.05], (100, 1))
    outcomes = ["H"] * 50 + ["D"] * 50
    bins = calibration_bins(probs, outcomes)
    row = bins.filter(bins["tranche"] == "65+")
    assert row["ecart"][0] == pytest.approx(0.40)
    assert expected_calibration_error(probs, outcomes) > 0.1


def test_calibration_bins_pool_every_class_so_counts_are_three_per_match() -> None:
    probs = np.full((10, 3), 1 / 3)
    bins = calibration_bins(probs, ["H"] * 10)
    assert bins["n"].sum() == 30


def test_class_bias_is_signed_and_sums_to_zero() -> None:
    # Under-predicts draws, over-predicts home wins.
    probs = np.tile([0.60, 0.15, 0.25], (100, 1))
    outcomes = ["H"] * 40 + ["D"] * 35 + ["A"] * 25
    bias = class_bias(probs, outcomes)
    assert bias["classe"].to_list() == list(CLASSES)
    assert bias["biais"].sum() == pytest.approx(0.0)
    assert bias.filter(bias["classe"] == "H")["biais"][0] == pytest.approx(0.20)
    assert bias.filter(bias["classe"] == "D")["biais"][0] == pytest.approx(-0.20)


def test_metrics_reject_probabilities_that_do_not_sum_to_one() -> None:
    probs = np.array([[0.5, 0.2, 0.2]])
    with pytest.raises(ValueError, match="sum to 1"):
        log_loss(probs, ["H"])


def test_metrics_reject_an_unknown_outcome_code() -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        log_loss(np.array([[0.4, 0.3, 0.3]]), ["W"])


def test_metrics_reject_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        log_loss(np.full((2, 3), 1 / 3), ["H"])
