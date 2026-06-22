"""Correctness and fallback tests for the optional Rust accelerator."""

import numpy as np

from optimalcontrol._accelerator import (
    RUST_ACCELERATOR_AVAILABLE,
    vector_fidelity,
    vector_member_value_gradients,
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


def test_rust_member_gradients_match_individual_problems() -> None:
    from optimalcontrol.grape import grape_xy_and_gradient

    problems = [_problem(), _problem()]
    problems[1].drifts = [np.complex128(-1j) * -0.35 * Iz()]
    waveform = np.array(
        [[0.12, -0.03], [-0.04, 0.08], [0.08, 0.02], [0.03, -0.06]],
        dtype=np.float64,
    )

    member_result = vector_member_value_gradients(problems, waveform)
    assert member_result is not None
    values, gradients = member_result
    assert values.shape == (2, 1)
    assert gradients.shape == (2, 1, 4, 2)

    for member, problem in enumerate(problems):
        value, gradient = grape_xy_and_gradient(problem, waveform)
        np.testing.assert_allclose(values[member, 0], value, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            gradients[member, 0], gradient, rtol=1e-11, atol=1e-12
        )


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
