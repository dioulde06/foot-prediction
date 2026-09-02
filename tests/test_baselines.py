"""Known-value checks on the three baselines and on the devigorisation."""

import math

import numpy as np
import polars as pl
import pytest

from src.eval.baselines import (
    devig_multiplicative,
    devig_power,
    home_baseline,
    market_baseline,
    uniform_baseline,
)


def test_devig_of_a_known_odds_triplet() -> None:
    # 2.10 / 3.40 / 3.60 -> implied 0.476190 / 0.294118 / 0.277778, sum 1.048086
    probs = devig_multiplicative(np.array([[2.10, 3.40, 3.60]]))
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, 0] == pytest.approx(0.476190 / 1.048086, rel=1e-5)
    assert probs[0, 1] == pytest.approx(0.294118 / 1.048086, rel=1e-5)
    assert probs[0, 2] == pytest.approx(0.277778 / 1.048086, rel=1e-5)


def test_devig_always_sums_to_one() -> None:
    odds = np.array([[1.30, 6.00, 8.50], [2.25, 3.50, 2.90], [8.06, 5.31, 1.39]])
    assert devig_multiplicative(odds).sum(axis=1) == pytest.approx(np.ones(3))


def test_devig_preserves_the_ranking_of_the_odds() -> None:
    probs = devig_multiplicative(np.array([[1.30, 6.00, 8.50]]))
    assert probs[0, 0] > probs[0, 1] > probs[0, 2]


def test_devig_rejects_an_incoherent_row_that_gives_money_away() -> None:
    # Real corrupt row found in the market-average column, 2025-08-16.
    with pytest.raises(ValueError, match="below 1"):
        devig_multiplicative(np.array([[8.70, 5.79, 1.56]]))


def test_devig_rejects_a_non_positive_odd() -> None:
    with pytest.raises(ValueError, match="positive"):
        devig_multiplicative(np.array([[2.10, 0.0, 3.60]]))


def test_uniform_baseline_uses_the_training_frequencies_only() -> None:
    train = pl.DataFrame({"result": ["H"] * 50 + ["D"] * 25 + ["A"] * 25})
    probs = uniform_baseline(train, n_rows=3)
    assert probs.shape == (3, 3)
    assert probs[0].tolist() == pytest.approx([0.50, 0.25, 0.25])
    # Every row is the same fixed prediction.
    assert probs[0].tolist() == probs[2].tolist()


def test_uniform_baseline_refuses_a_class_it_never_saw() -> None:
    train = pl.DataFrame({"result": ["H"] * 10})
    with pytest.raises(ValueError, match="never observed"):
        uniform_baseline(train, n_rows=2)


def test_home_baseline_is_a_hard_prediction() -> None:
    probs = home_baseline(n_rows=3)
    assert probs.shape == (3, 3)
    assert probs[0].tolist() == [1.0, 0.0, 0.0]


def test_market_baseline_reads_the_chosen_book() -> None:
    frame = pl.DataFrame(
        {
            "odds_close_avg_h": [2.10],
            "odds_close_avg_d": [3.40],
            "odds_close_avg_a": [3.60],
        }
    )
    probs = market_baseline(frame, book="avg", method="multiplicative")
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, 0] == pytest.approx(0.4543, abs=1e-4)


def test_market_baseline_refuses_missing_odds() -> None:
    frame = pl.DataFrame(
        {
            "odds_close_ps_h": [2.10, None],
            "odds_close_ps_d": [3.40, 3.0],
            "odds_close_ps_a": [3.60, 3.0],
        }
    )
    with pytest.raises(ValueError, match="missing odds"):
        market_baseline(frame, book="ps")


def test_a_perfectly_calibrated_uniform_baseline_scores_ln_3_when_classes_are_equal() -> (
    None
):
    from src.eval.metrics import log_loss

    train = pl.DataFrame({"result": ["H", "D", "A"] * 10})
    probs = uniform_baseline(train, n_rows=3)
    assert log_loss(probs, ["H", "D", "A"]) == pytest.approx(math.log(3))


# --- power devigorisation -------------------------------------------------


def test_power_devig_always_sums_to_one() -> None:
    odds = np.array([[1.30, 6.00, 8.50], [2.25, 3.50, 2.90], [2.10, 3.40, 3.60]])
    assert devig_power(odds).sum(axis=1) == pytest.approx(np.ones(3))


def test_power_devig_is_the_identity_when_there_is_no_margin() -> None:
    # 3.0 / 3.0 / 3.0 implies exactly 1/3 each, so the exponent must stay at 1.
    probs = devig_power(np.array([[3.0, 3.0, 3.0]]))
    assert probs[0].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_power_devig_preserves_the_ranking_of_the_odds() -> None:
    probs = devig_power(np.array([[1.30, 6.00, 8.50]]))
    assert probs[0, 0] > probs[0, 1] > probs[0, 2]


def test_power_devig_gives_the_favourite_more_than_the_multiplicative_method() -> None:
    # The whole point: the margin sits on the longshots, so removing it
    # proportionally leaves the favourite short. Measured on real 2025-26 data.
    odds = np.array([[1.30, 6.00, 8.50]])
    favourite_power = devig_power(odds)[0, 0]
    favourite_mult = devig_multiplicative(odds)[0, 0]
    assert favourite_power > favourite_mult
    # ...and the longshot correspondingly less.
    assert devig_power(odds)[0, 2] < devig_multiplicative(odds)[0, 2]


def test_power_devig_correction_grows_with_the_margin() -> None:
    # Same prices, one with a fat margin, one with a thin one.
    thin = np.array([[1.32, 5.80, 8.20]])
    fat = np.array([[1.25, 5.20, 7.20]])

    def gap(odds: np.ndarray) -> float:
        return float(devig_power(odds)[0, 0] - devig_multiplicative(odds)[0, 0])

    assert gap(fat) > gap(thin) > 0


def test_power_devig_rejects_an_incoherent_row() -> None:
    with pytest.raises(ValueError, match="below 1"):
        devig_power(np.array([[8.70, 5.79, 1.56]]))


def test_market_baseline_accepts_both_methods() -> None:
    frame = pl.DataFrame(
        {
            "odds_close_avg_h": [1.30],
            "odds_close_avg_d": [6.00],
            "odds_close_avg_a": [8.50],
        }
    )
    mult = market_baseline(frame, book="avg", method="multiplicative")
    power = market_baseline(frame, book="avg", method="power")
    assert power[0, 0] > mult[0, 0]
    with pytest.raises(ValueError, match="unknown devig method"):
        market_baseline(frame, book="avg", method="magic")
