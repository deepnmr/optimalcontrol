"""Tests for waveform penalty functions."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from optimalcontrol.penalties import (
    PenaltySpec,
    penalty_DNS,
    penalty_NS,
    penalty_SNS,
    penalty_SNSA,
    total_penalty,
)

PenaltyValue = Callable[[npt.NDArray[np.float64]], float]


def _finite_difference_gradient(
    value_fn: PenaltyValue,
    waveform: npt.NDArray[np.float64],
    eps: float = 1e-6,
) -> npt.NDArray[np.float64]:
    gradient = np.zeros_like(waveform, dtype=np.float64)
    for index in np.ndindex(waveform.shape):
        plus = waveform.copy()
        minus = waveform.copy()
        plus[index] += eps
        minus[index] -= eps
        gradient[index] = (value_fn(plus) - value_fn(minus)) / (2.0 * eps)
    return gradient


def test_penalty_NS_gradient_matches_finite_difference() -> None:
    waveform = np.array([[0.3, -0.2], [0.1, 0.5]], dtype=np.float64)

    _, gradient = penalty_NS(waveform, weight=0.7)
    finite_difference = _finite_difference_gradient(
        lambda wfm: penalty_NS(wfm, weight=0.7)[0],
        waveform,
    )

    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-9)


def test_penalty_SNS_penalises_elementwise_spillout() -> None:
    waveform = np.array([[0.4, -1.2], [0.75, 1.5]], dtype=np.float64)

    value, gradient = penalty_SNS(waveform, limit=0.8, weight=0.5)
    finite_difference = _finite_difference_gradient(
        lambda wfm: penalty_SNS(wfm, limit=0.8, weight=0.5)[0],
        waveform,
    )

    expected_value = 0.5 * ((1.2 - 0.8) ** 2 + (1.5 - 0.8) ** 2)
    np.testing.assert_allclose(value, expected_value, rtol=1e-12)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-9)


def test_penalty_SNSA_penalises_row_amplitude_spillout() -> None:
    waveform = np.array([[3.0, 4.0], [0.2, 0.1], [-5.0, 12.0]], dtype=np.float64)

    value, gradient = penalty_SNSA(waveform, limit=4.0, weight=0.25)
    finite_difference = _finite_difference_gradient(
        lambda wfm: penalty_SNSA(wfm, limit=4.0, weight=0.25)[0],
        waveform,
    )

    expected_value = 0.25 * ((5.0 - 4.0) ** 2 + (13.0 - 4.0) ** 2)
    np.testing.assert_allclose(value, expected_value, rtol=1e-12)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(gradient[1], np.zeros(2, dtype=np.float64), atol=1e-12)


def test_penalty_DNS_gradient_matches_finite_difference() -> None:
    waveform = np.array([[0.0, 1.0], [0.4, -0.2], [0.9, 0.3]], dtype=np.float64)

    _, gradient = penalty_DNS(waveform, weight=0.6)
    finite_difference = _finite_difference_gradient(
        lambda wfm: penalty_DNS(wfm, weight=0.6)[0],
        waveform,
    )

    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-9)


def test_penalty_DNS_single_row_is_zero() -> None:
    waveform = np.array([[0.1, -0.2]], dtype=np.float64)

    value, gradient = penalty_DNS(waveform, weight=1.0)

    assert value == 0.0
    np.testing.assert_allclose(gradient, np.zeros_like(waveform), atol=1e-12)


def test_total_penalty_sums_specs_and_callables() -> None:
    waveform = np.array([[0.2, 1.1], [-0.4, 0.8]], dtype=np.float64)

    value, gradient = total_penalty(
        waveform,
        [
            PenaltySpec("NS", weight=0.1),
            PenaltySpec("SNS", weight=0.5, limit=0.9),
            lambda wfm: penalty_DNS(wfm, weight=0.2),
        ],
    )

    value_ns, gradient_ns = penalty_NS(waveform, weight=0.1)
    value_sns, gradient_sns = penalty_SNS(waveform, limit=0.9, weight=0.5)
    value_dns, gradient_dns = penalty_DNS(waveform, weight=0.2)
    np.testing.assert_allclose(value, value_ns + value_sns + value_dns, rtol=1e-12)
    np.testing.assert_allclose(gradient, gradient_ns + gradient_sns + gradient_dns, rtol=1e-12)


def test_total_penalty_rejects_unknown_spec() -> None:
    waveform = np.zeros((2, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="unknown penalty"):
        total_penalty(waveform, [PenaltySpec("bad", weight=1.0)])
