"""Correctness and fallback tests for the optional Rust accelerator."""

import numpy as np
import pytest

from optimalcontrol._accelerator import (
    RUST_ACCELERATOR_AVAILABLE,
    vector_fidelity,
    vector_value_gradient,
)
from optimalcontrol.grape import ControlProblem
from optimalcontrol.operators import Ix, Iy, Iz
from optimalcontrol.states import normalise_2norm


def _problem() -> ControlProblem:
    return ControlProblem(
        drifts=[np.complex128(-1j) * 0.2 * Iz()],
        operators=[np.complex128(-1j) * Ix(), np.complex128(-1j) * Iy()],
        rho_init=[np.array([1.0, 0.0], dtype=np.complex128)],
        rho_targ=[normalise_2norm(np.array([0.35 + 0.15j, 0.88 - 0.28j], dtype=np.complex128))],
        pulse_dt=0.05,
        pwr_levels=[0.8, 1.1],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )


def test_rust_accelerator_is_built() -> None:
    assert RUST_ACCELERATOR_AVAILABLE


def test_rust_vector_fidelity_matches_python_propagation(monkeypatch) -> None:
    from optimalcontrol.grape import _grape_xy_core

    cp = _problem()
    waveform = np.array(
        [[0.12, -0.03], [-0.04, 0.08], [0.08, 0.02], [0.03, -0.06]],
        dtype=np.float64,
    )
    rust_value = vector_fidelity([cp], waveform)
    assert rust_value is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value = _grape_xy_core(cp, waveform)
    np.testing.assert_allclose(rust_value, python_value, rtol=1e-12, atol=1e-12)


def test_rust_value_gradient_matches_python_adjoint(monkeypatch) -> None:
    from optimalcontrol.grape import grape_xy_and_gradient

    cp = _problem()
    waveform = np.array(
        [[0.12, -0.03], [-0.04, 0.08], [0.08, 0.02], [0.03, -0.06]],
        dtype=np.float64,
    )
    rust_result = vector_value_gradient([cp], waveform)
    assert rust_result is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value, python_gradient = grape_xy_and_gradient(cp, waveform)
    np.testing.assert_allclose(rust_result[0], python_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rust_result[1], python_gradient, rtol=1e-11, atol=1e-12)


def test_rust_fidelity_matches_python_with_relaxation(monkeypatch) -> None:
    from optimalcontrol.grape import _grape_xy_core

    cp = _problem()
    cp.drifts = [cp.drifts[0] - 0.07 * np.eye(2, dtype=np.complex128)]
    waveform = np.array([[0.12, -0.03], [-0.04, 0.08]], dtype=np.float64)
    rust_value = vector_fidelity([cp], waveform)
    assert rust_value is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value = _grape_xy_core(cp, waveform)
    np.testing.assert_allclose(rust_value, python_value, rtol=1e-12, atol=1e-12)


def test_rust_vector_fidelity_falls_back_for_matrix_states() -> None:
    cp = _problem()
    cp.rho_init = [np.eye(2, dtype=np.complex128)]
    cp.rho_targ = [np.eye(2, dtype=np.complex128)]
    waveform = np.zeros((2, 2), dtype=np.float64)

    assert vector_fidelity([cp], waveform) is None


def _sx() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _sz() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def test_hermiticity_gate_parity_for_large_drift_defect(monkeypatch) -> None:
    """A Hermitian defect below 1e-12 * scale must take the same branch on both sides."""
    from optimalcontrol.grape import _grape_xy_core

    cp = ControlProblem(
        drifts=[np.complex128(-1j) * 1e6 * _sz() + 2.5e-7 * _sz()],
        operators=[np.complex128(-1j) * _sx()],
        rho_init=[np.array([1.0, 0.0], dtype=np.complex128)],
        rho_targ=[np.array([1.0, 0.0], dtype=np.complex128)],
        pulse_dt=5e-4,
        pwr_levels=[1.0],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )
    waveform = np.zeros((2000, 1), dtype=np.float64)
    rust_value = vector_fidelity([cp], waveform)
    assert rust_value is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value = _grape_xy_core(cp, waveform)
    np.testing.assert_allclose(rust_value, python_value, rtol=0.0, atol=1e-11)


def test_hermiticity_gate_parity_for_power_scaled_defect(monkeypatch) -> None:
    """The gate must judge the power-scaled operators, like the Rust kernel does."""
    from optimalcontrol.grape import _grape_xy_core

    plus_x = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    cp = ControlProblem(
        drifts=[np.zeros((2, 2), dtype=np.complex128)],
        operators=[np.complex128(-1j) * 1e-6 * _sx() + 4e-13 * _sx()],
        rho_init=[plus_x],
        rho_targ=[plus_x],
        pulse_dt=np.pi / 100.0,
        pwr_levels=[1e6],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )
    waveform = np.ones((100, 1), dtype=np.float64)
    rust_value = vector_fidelity([cp], waveform)
    assert rust_value is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value = _grape_xy_core(cp, waveform)
    np.testing.assert_allclose(rust_value, python_value, rtol=0.0, atol=1e-11)


def test_near_degenerate_gradient_parity(monkeypatch) -> None:
    """Eigenvalue gaps just above the old degeneracy cutoff must not blow up gradients."""
    from optimalcontrol.grape import _single_value_and_gradient

    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    spectrum = np.array([1.0, 3.0, 3.0 + 3.03e-12])
    hermitian = (basis * spectrum) @ basis.conj().T
    hermitian = (hermitian + hermitian.conj().T) / 2.0

    operators = []
    for _ in range(2):
        raw = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        operators.append(np.complex128(-1j) * (raw + raw.conj().T) / 2.0)
    initial = rng.standard_normal(3) + 1j * rng.standard_normal(3)
    target = rng.standard_normal(3) + 1j * rng.standard_normal(3)

    cp = ControlProblem(
        drifts=[np.complex128(-1j) * hermitian],
        operators=operators,
        rho_init=[np.asarray(initial / np.linalg.norm(initial), dtype=np.complex128)],
        rho_targ=[np.asarray(target / np.linalg.norm(target), dtype=np.complex128)],
        pulse_dt=1.0,
        pwr_levels=[1.0, 1.0],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )
    waveform = np.zeros((6, 2), dtype=np.float64)
    rust_result = vector_value_gradient([cp], waveform)
    assert rust_result is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value, python_gradient = _single_value_and_gradient(cp, waveform)
    np.testing.assert_allclose(rust_result[0], python_value, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(rust_result[1], python_gradient, rtol=0.0, atol=1e-12)


def test_negative_power_level_raises_on_both_paths(monkeypatch) -> None:
    """The Rust fast path must not bypass Python's pwr_levels validation."""
    from optimalcontrol.ensemble import ensemble_fidelity

    drift = np.complex128(-1j) * _sz()
    cp = ControlProblem(
        drifts=[drift, 1.1 * drift],
        operators=[np.complex128(-1j) * _sx()],
        rho_init=[np.array([1.0, 0.0], dtype=np.complex128)],
        rho_targ=[np.array([0.0, 1.0], dtype=np.complex128)],
        pulse_dt=0.05,
        pwr_levels=[-1.0],
        freeze=None,
        fidelity_mode="real",
        basis="hilbert",
    )
    waveform = 0.3 * np.ones((4, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        ensemble_fidelity(cp, waveform)

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    with pytest.raises(ValueError, match="non-negative"):
        ensemble_fidelity(cp, waveform)
