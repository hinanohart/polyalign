"""PAVA isotonic + equal-frequency ECE tests."""

from __future__ import annotations

import numpy as np
import pytest

from polyalign.calibration import expected_calibration_error, pava_monotone


def test_pava_already_monotone_is_identity() -> None:
    y = np.array([0.1, 0.2, 0.3, 0.5, 0.9])
    np.testing.assert_allclose(pava_monotone(y), y)


def test_pava_strict_decreasing_becomes_constant_mean() -> None:
    y = np.array([1.0, 0.5, 0.0])
    out = pava_monotone(y)
    np.testing.assert_allclose(out, np.full(3, 0.5))


def test_pava_partial_pool() -> None:
    y = np.array([0.0, 1.0, 0.5, 2.0])
    out = pava_monotone(y)
    # 1.0 and 0.5 pool to 0.75 then 0.0, 0.75, 0.75, 2.0
    np.testing.assert_allclose(out, [0.0, 0.75, 0.75, 2.0])


def test_pava_empty_input() -> None:
    y = np.array([], dtype=np.float64)
    out = pava_monotone(y)
    assert out.shape == (0,)


def test_pava_weighted() -> None:
    y = np.array([1.0, 0.0])
    w = np.array([3.0, 1.0])
    out = pava_monotone(y, weights=w)
    # weighted mean = (3*1 + 1*0) / 4 = 0.75 for both
    np.testing.assert_allclose(out, [0.75, 0.75])


def test_pava_invalid_weights_raises() -> None:
    with pytest.raises(ValueError, match="weights must be positive"):
        pava_monotone(np.array([1.0, 2.0]), weights=np.array([-1.0, 1.0]))


def test_pava_rejects_2d() -> None:
    with pytest.raises(ValueError, match="1D"):
        pava_monotone(np.zeros((3, 3)))


def test_ece_perfectly_calibrated_returns_low_value() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    probs = rng.uniform(0, 1, size=n)
    correct = rng.uniform(0, 1, size=n) < probs
    ece = expected_calibration_error(probs, correct.astype(np.float64), n_bins=10)
    assert ece < 0.1


def test_ece_uncalibrated_returns_high_value() -> None:
    probs = np.full(100, 0.9)
    correct = np.zeros(100)
    ece = expected_calibration_error(probs, correct, n_bins=5)
    assert ece > 0.5


def test_ece_empty_returns_zero() -> None:
    ece = expected_calibration_error(np.array([]), np.array([]), n_bins=5)
    assert ece == 0.0


def test_ece_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        expected_calibration_error(np.zeros(5), np.zeros(4))


def test_ece_1d_required() -> None:
    with pytest.raises(ValueError, match="1D"):
        expected_calibration_error(np.zeros((3, 3)), np.zeros((3, 3)))
