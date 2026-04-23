"""Tests for GRAPE control-problem containers."""

import numpy as np
import pytest
from scipy.linalg import expm

from optimalcontrol.grape import (
    ControlProblem,
    apply_freeze,
    backward_states,
    final_fidelity,
    forward_propagators,
    forward_states,
    grape_xy,
    validate_control_problem,
    validate_waveform,
)


def _basic_control_problem() -> ControlProblem:
    drift = np.zeros((2, 2), dtype=np.complex128)
    operator = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)
    rho_init = np.array([1.0, 0.0], dtype=np.complex128)
    rho_targ = np.array([0.0, 1.0], dtype=np.complex128)
    freeze = np.zeros((4, 1), dtype=np.bool_)
    return ControlProblem(
        drifts=[drift],
        operators=[operator],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=1e-3,
        pwr_levels=[25.0],
        freeze=freeze,
    )


def test_validate_control_problem_accepts_minimal_problem() -> None:
    cp = _basic_control_problem()

    validate_control_problem(cp)


def test_validate_control_problem_rejects_invalid_fidelity_mode() -> None:
    cp = _basic_control_problem()
    cp.fidelity_mode = "phase"

    with pytest.raises(ValueError, match="fidelity_mode"):
        validate_control_problem(cp)


def test_validate_control_problem_rejects_operator_dimension_mismatch() -> None:
    cp = _basic_control_problem()
    cp.operators = [np.eye(3, dtype=np.complex128)]

    with pytest.raises(ValueError, match="operators"):
        validate_control_problem(cp)


def test_validate_control_problem_rejects_freeze_channel_mismatch() -> None:
    cp = _basic_control_problem()
    cp.freeze = np.zeros((4, 2), dtype=np.bool_)

    with pytest.raises(ValueError, match="freeze mask"):
        validate_control_problem(cp)


def test_validate_waveform_accepts_time_rows_by_channels() -> None:
    waveform = np.zeros((4, 1), dtype=np.float64)

    validate_waveform(waveform, n_channels=1, n_steps=4)


def test_validate_waveform_rejects_wrong_shape() -> None:
    waveform = np.zeros((1, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="waveform"):
        validate_waveform(waveform, n_channels=1, n_steps=4)


def test_apply_freeze_restores_entries_from_initial_waveform() -> None:
    initial = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    candidate = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    freeze = np.array([[True, False], [False, True]], dtype=np.bool_)

    frozen = apply_freeze(candidate, freeze, initial)

    expected = np.array([[1.0, 20.0], [30.0, 4.0]], dtype=np.float64)
    np.testing.assert_allclose(frozen, expected, rtol=1e-12)


def test_forward_propagators_builds_scaled_slice_exponentials() -> None:
    cp = _basic_control_problem()
    cp.pulse_dt = 0.25
    cp.pwr_levels = [2.0]
    cp.freeze = np.zeros((2, 1), dtype=np.bool_)
    waveform = np.array([[3.0], [0.0]], dtype=np.float64)

    propagators = forward_propagators(cp, waveform)

    expected_first = expm(cp.operators[0] * (3.0 * 2.0) * cp.pulse_dt)
    expected_second = np.eye(2, dtype=np.complex128)
    assert len(propagators) == 2
    np.testing.assert_allclose(propagators[0], expected_first, rtol=1e-12)
    np.testing.assert_allclose(propagators[1], expected_second, rtol=1e-12)


def test_forward_states_accumulates_vector_states() -> None:
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    rho_init = np.array([1.0, 0.0], dtype=np.complex128)

    states = forward_states(rho_init, [swap, swap])

    assert len(states) == 3
    np.testing.assert_allclose(states[0], rho_init, rtol=1e-12)
    np.testing.assert_allclose(states[1], np.array([0.0, 1.0], dtype=np.complex128), rtol=1e-12)
    np.testing.assert_allclose(states[2], rho_init, rtol=1e-12)


def test_forward_states_applies_hilbert_propagator_to_matrix_state() -> None:
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    rho_init = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

    states = forward_states(rho_init, [swap])

    expected = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    np.testing.assert_allclose(states[1], expected, rtol=1e-12)


def test_backward_states_accumulates_target_first_with_adjoint_propagators() -> None:
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    phase = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    rho_targ = np.array([0.0, 1.0], dtype=np.complex128)

    states = backward_states(rho_targ, [swap, phase])

    expected_after_phase_adjoint = np.array([0.0, -1.0j], dtype=np.complex128)
    expected_after_swap_adjoint = np.array([-1.0j, 0.0], dtype=np.complex128)
    assert len(states) == 3
    np.testing.assert_allclose(states[0], rho_targ, rtol=1e-12)
    np.testing.assert_allclose(states[1], expected_after_phase_adjoint, rtol=1e-12)
    np.testing.assert_allclose(states[2], expected_after_swap_adjoint, rtol=1e-12)


def test_final_fidelity_uses_configured_mode() -> None:
    rho_targ = np.array([1.0, 0.0], dtype=np.complex128)
    rho_final = np.complex128(1j) * rho_targ

    np.testing.assert_allclose(final_fidelity([rho_targ, rho_final], [rho_targ], "real"), 0.0)
    np.testing.assert_allclose(final_fidelity([rho_targ, rho_final], [rho_targ], "imag"), 1.0)
    np.testing.assert_allclose(final_fidelity([rho_targ, rho_final], [rho_targ], "abs2"), 1.0)


def test_grape_xy_returns_mean_fidelity_across_state_pairs() -> None:
    cp = _basic_control_problem()
    cp.operators = [np.zeros((2, 2), dtype=np.complex128)]
    cp.pwr_levels = [1.0]
    cp.freeze = np.zeros((3, 1), dtype=np.bool_)
    rho = np.array([1.0, 0.0], dtype=np.complex128)
    cp.rho_init = [rho, -rho]
    cp.rho_targ = [rho, rho]
    waveform = np.zeros((3, 1), dtype=np.float64)

    result = grape_xy(cp, waveform)

    np.testing.assert_allclose(result, 0.0, atol=1e-12)
