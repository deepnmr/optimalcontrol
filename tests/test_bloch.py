"""Tests for native and fallback Bloch ensemble propagation."""

import numpy as np
import pytest

from optimalcontrol.bloch import propagate_bloch_ensemble


def test_rust_bloch_ensemble_matches_numpy_fallback(monkeypatch) -> None:
    initial = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    waveform = np.array([[1.0, 0.0], [0.5, 0.5], [-0.2, 0.8]], dtype=np.float64)
    offsets = np.array([-1200.0, 0.0, 1700.0], dtype=np.float64)
    scales = np.array([0.85, 1.0, 1.15], dtype=np.float64)
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
