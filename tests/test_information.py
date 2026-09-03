"""The information test: does the model know anything the market does not?"""

from __future__ import annotations

import numpy as np

from src.eval.information import fit_blend, verdict

RNG = np.random.default_rng(7)


def _draw(probs: np.ndarray) -> list[str]:
    codes = np.array(["H", "D", "A"])
    picks = [RNG.choice(3, p=row) for row in probs]
    return list(codes[picks])


def _random_probs(n: int) -> np.ndarray:
    raw = RNG.dirichlet([4.0, 2.5, 3.0], size=n)
    return np.asarray(raw, dtype=np.float64)


def test_a_copy_of_the_market_gets_no_weight_of_its_own() -> None:
    """Model == market: the two weights are unidentifiable except through their
    sum, which must be about 1 for a calibrated source. The fit is regularised
    toward the market, so the model side lands near zero."""
    market = _random_probs(20_000)
    outcomes = _draw(market)
    fit = fit_blend(market, market, outcomes)
    assert abs(fit.a + fit.b - 1.0) < 0.05


def test_pure_noise_next_to_the_truth_gets_a_zero_weight() -> None:
    market = _random_probs(20_000)
    outcomes = _draw(market)
    noise = _random_probs(20_000)
    fit = fit_blend(noise, market, outcomes)
    assert abs(fit.a) < 0.04
    assert abs(fit.b - 1.0) < 0.05


def test_the_truth_next_to_noise_gets_all_the_weight() -> None:
    truth = _random_probs(20_000)
    outcomes = _draw(truth)
    noise = _random_probs(20_000)
    fit = fit_blend(truth, noise, outcomes)
    assert abs(fit.a - 1.0) < 0.05
    assert abs(fit.b) < 0.04


def test_blend_reproduces_the_inputs_at_unit_weights() -> None:
    from src.eval.information import blend

    probs = _random_probs(50)
    np.testing.assert_allclose(blend(probs, probs, 1.0, 0.0), probs, atol=1e-12)
    np.testing.assert_allclose(blend(probs, probs, 0.5, 0.5), probs, atol=1e-12)


def test_verdict_reads_the_three_sides_of_zero() -> None:
    assert "above zero" in verdict(0.02, 0.10)
    assert "below zero" in verdict(-0.24, -0.02)
    assert "includes zero" in verdict(-0.05, 0.05)
