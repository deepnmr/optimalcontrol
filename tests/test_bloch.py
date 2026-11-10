"""Tests for native and fallback Bloch ensemble propagation."""

import numpy as np
import pytest

from optimalcontrol._accelerator import RUST_ACCELERATOR_AVAILABLE
from optimalcontrol.bloch import _rotate, propagate_bloch_ensemble


@pytest.mark.parametrize("disable_rust", ["0", "1"])
def test_tiny_field_long_duration_pi_rotation(
    monkeypatch: pytest.MonkeyPatch, disable_rust: str
) -> None:
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)

    result = propagate_bloch_ensemble(
        np.array([0.0, 0.0, 1.0]),
        np.array([[1.0, 0.0]]),
        np.array([0.0]),
        np.array([1.0]),
        1e-200,
        5e199,
    )

    np.testing.assert_allclose(result[0, 0, 2], -1.0, rtol=0.0, atol=1e-12)


def test_rust_bloch_ensemble_matches_numpy_fallback(monkeypatch) -> None:
    assert RUST_ACCELERATOR_AVAILABLE
    initial = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    waveform = np.array([[1.0, 0.0], [0.5, 0.5], [-0.2, 0.8]], dtype=np.float64)
    offsets = np.array([-1200.0, 0.0, 1700.0], dtype=np.float64)
    scales = np.array([0.85, 1.0, 1.15], dtype=np.float64)
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "0")
    native = propagate_bloch_ensemble(initial, waveform, offsets, scales, 7500.0, 7.5e-6)

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    fallback = propagate_bloch_ensemble(initial, waveform, offsets, scales, 7500.0, 7.5e-6)
    np.testing.assert_allclose(native, fallback, rtol=1e-13, atol=1e-13)


def test_bloch_ensemble_preserves_vector_norm() -> None:
    result = propagate_bloch_ensemble(
        np.array([0.0, 1.0, 0.0]),
        np.ones((5, 2), dtype=np.float64),
        np.array([-1000.0, 1000.0]),
        np.array([0.9, 1.1]),
        5000.0,
        1e-5,
    )
    np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1.0, atol=1e-13)


def test_bloch_ensemble_rejects_non_finite_arrays(monkeypatch) -> None:
    initial = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    waveform = np.ones((3, 2), dtype=np.float64)
    offsets = np.array([np.inf], dtype=np.float64)
    scales = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        propagate_bloch_ensemble(initial, waveform, offsets, scales, 100.0, 1e-5)

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    with pytest.raises(ValueError, match="finite"):
        propagate_bloch_ensemble(initial, waveform, offsets, scales, 100.0, 1e-5)


def test_fallback_matches_native_on_long_ensemble(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RUST_ACCELERATOR_AVAILABLE
    initial = np.array([0.0, 0.0, 1.0])
    waveform = np.random.default_rng(0).uniform(-1.0, 1.0, (512, 2))
    offsets = np.linspace(-10.0, 10.0, 801)
    scales = np.array([0.5, 1.0, 1.5])
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "0")
    native = propagate_bloch_ensemble(initial, waveform, offsets, scales, 6.0, 1.0 / 512)

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    fallback = propagate_bloch_ensemble(initial, waveform, offsets, scales, 6.0, 1.0 / 512)

    np.testing.assert_allclose(native, fallback, rtol=1e-13, atol=1e-13)


def test_fallback_single_step_matches_native_closely(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RUST_ACCELERATOR_AVAILABLE
    rng = np.random.default_rng(1)
    waveform = rng.uniform(-1.0, 1.0, (1, 2))
    offsets = rng.uniform(-5000.0, 5000.0, 2000)
    initial = np.array([0.3, -0.4, np.sqrt(0.75)])
    scales = np.array([1.0])
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "0")
    native = propagate_bloch_ensemble(initial, waveform, offsets, scales, 5000.0, 7.5e-6)

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    fallback = propagate_bloch_ensemble(initial, waveform, offsets, scales, 5000.0, 7.5e-6)

    maximum = np.max(np.abs(native - fallback))
    assert maximum <= 2.0e-16, f"maximum absolute difference: {maximum:.17g}"


@pytest.mark.parametrize("disable_rust", ["0", "1"])
def test_rotate_zero_field_leaves_state_unchanged(
    monkeypatch: pytest.MonkeyPatch, disable_rust: str
) -> None:
    initial = np.array([0.3, -0.4, np.sqrt(0.75)])
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)

    result = propagate_bloch_ensemble(
        initial, np.ones((3, 2)), np.array([0.0]), np.array([1.0]), 0.0, 7.5e-6
    )

    np.testing.assert_array_equal(result[0, 0], initial)


def test_rotate_broadcasts_over_leading_axes() -> None:
    rng = np.random.default_rng(2)
    state = rng.uniform(-1.0, 1.0, (4, 5, 3))
    field = rng.uniform(-5000.0, 5000.0, (4, 5, 3))

    result = _rotate(state, field, 7.5e-6)

    expected = np.empty_like(state)
    for row in range(4):
        for col in range(5):
            expected[row, col] = _rotate(state[row, col], field[row, col], 7.5e-6)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_allclose(
        np.linalg.norm(result, axis=-1), np.linalg.norm(state, axis=-1), rtol=1e-13, atol=1e-13
    )


def test_rotate_mixed_zero_fields_raise_nothing() -> None:
    state = np.array([[0.3, -0.4, 0.5], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    field = np.array([[0.0, 0.0, 0.0], [1200.0, -300.0, 700.0], [0.0, 0.0, 0.0]])

    with np.errstate(all="raise"):
        result = _rotate(state, field, 7.5e-6)

    np.testing.assert_array_equal(result[[0, 2]], state[[0, 2]])
    np.testing.assert_array_equal(result[1], _rotate(state[1], field[1], 7.5e-6))
