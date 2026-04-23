"""Tests for GRAPE control-problem containers."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.linalg import expm

from optimalcontrol.grape import (
    ControlProblem,
    ampl_phase_to_xy,
    apply_freeze,
    backward_states,
    curvilinear_reparameterise,
    final_fidelity,
    forward_propagators,
    forward_states,
    grape_gradient,
    grape_hessian,
    grape_xy,
    grape_xy_hilbert,
    grape_xy_liouville,
    phase_only_gradient,
    validate_control_problem,
    validate_waveform,
    xy_to_ampl_phase,
)
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, place_operator, vec
from optimalcontrol.states import normalise_2norm, normalise_hs, state_from_label


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


def _finite_difference_gradient(
    cp: ControlProblem,
    waveform: npt.NDArray[np.float64],
    eps: float = 1e-6,
) -> npt.NDArray[np.float64]:
    gradient = np.zeros_like(waveform, dtype=np.float64)
    for index in np.ndindex(waveform.shape):
        wfm_plus = waveform.copy()
        wfm_minus = waveform.copy()
        wfm_plus[index] += eps
        wfm_minus[index] -= eps
        gradient[index] = (grape_xy(cp, wfm_plus) - grape_xy(cp, wfm_minus)) / (
            2.0 * eps
        )
    return gradient


def _relative_l2_error(
    actual: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
) -> float:
    denominator = max(float(np.linalg.norm(expected)), 1e-12)
    return float(np.linalg.norm(actual - expected) / denominator)


def _one_spin_hilbert_gradient_problem(
    freeze: npt.NDArray[np.bool_] | None = None,
) -> ControlProblem:
    rho_init = np.array([1.0, 0.0], dtype=np.complex128)
    rho_targ = normalise_2norm(
        np.array([0.35 + 0.15j, 0.88 - 0.28j], dtype=np.complex128)
    )
    return ControlProblem(
        drifts=[np.complex128(-1j) * 0.3 * Iz()],
        operators=[np.complex128(-1j) * Ix(), np.complex128(-1j) * Iy()],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=0.08,
        pwr_levels=[0.7, 0.9],
        freeze=freeze,
        fidelity_mode="abs2",
        basis="hilbert",
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


def test_xy_to_ampl_phase_round_trips_cartesian_waveform() -> None:
    waveform = np.array(
        [[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [0.0, -4.0]],
        dtype=np.float64,
    )

    amplitude, phase = xy_to_ampl_phase(waveform)
    round_trip = ampl_phase_to_xy(amplitude, phase)

    np.testing.assert_allclose(amplitude, np.array([1.0, 2.0, 3.0, 4.0]), rtol=1e-12)
    np.testing.assert_allclose(round_trip, waveform, atol=1e-12)


def test_phase_only_gradient_uses_amplitude_phase_chain_rule() -> None:
    amplitude = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    phase = np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float64)
    grad_xy = np.array([[5.0, 7.0], [11.0, 13.0], [17.0, 19.0]], dtype=np.float64)

    phase_gradient = phase_only_gradient(grad_xy, amplitude, phase)

    expected = np.array([7.0, -22.0, -57.0], dtype=np.float64)
    np.testing.assert_allclose(phase_gradient, expected, atol=1e-12)


def test_phase_only_gradient_accepts_current_xy_waveform() -> None:
    waveform = np.array([[3.0, 4.0], [-2.0, 5.0]], dtype=np.float64)
    grad_xy = np.array([[7.0, 11.0], [13.0, 17.0]], dtype=np.float64)

    phase_gradient = phase_only_gradient(grad_xy, waveform)

    expected = np.array([5.0, -99.0], dtype=np.float64)
    np.testing.assert_allclose(phase_gradient, expected, rtol=1e-12)


def test_curvilinear_reparameterise_maps_unconstrained_values_inside_bounds() -> None:
    unconstrained = np.array([[-2.0, 0.0, 2.0]], dtype=np.float64)

    bounded = curvilinear_reparameterise(unconstrained, bounds=(-2.0, 6.0))

    assert bounded.shape == unconstrained.shape
    assert np.all(bounded > -2.0)
    assert np.all(bounded < 6.0)
    np.testing.assert_allclose(bounded[0, 1], 2.0, rtol=1e-12)


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


def test_grape_hessian_returns_exact_small_hilbert_hessian() -> None:
    drift = np.zeros((2, 2), dtype=np.complex128)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    operator = np.complex128(-1j) * sigma_x
    rho = np.array([1.0, 0.0], dtype=np.complex128)
    cp = ControlProblem(
        drifts=[drift],
        operators=[operator],
        rho_init=[rho],
        rho_targ=[rho],
        pulse_dt=0.2,
        pwr_levels=[1.0],
        freeze=None,
        fidelity_mode="real",
        basis="hilbert",
    )
    waveform = np.array([[0.3]], dtype=np.float64)

    hessian = grape_hessian(cp, waveform)

    assert hessian is not None
    expected = np.array([[-cp.pulse_dt**2 * np.cos(waveform[0, 0] * cp.pulse_dt)]])
    np.testing.assert_allclose(hessian, expected, rtol=1e-12, atol=1e-12)


def test_grape_hessian_returns_none_for_large_waveforms() -> None:
    cp = _basic_control_problem()
    cp.freeze = np.zeros((51, 1), dtype=np.bool_)
    waveform = np.zeros((51, 1), dtype=np.float64)

    hessian = grape_hessian(cp, waveform)

    assert hessian is None


def test_grape_xy_liouville_vectorises_density_matrices() -> None:
    drift = np.zeros((4, 4), dtype=np.complex128)
    operator = np.zeros((4, 4), dtype=np.complex128)
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    cp = ControlProblem(
        drifts=[drift],
        operators=[operator],
        rho_init=[rho],
        rho_targ=[rho],
        pulse_dt=1e-3,
        pwr_levels=[1.0],
        freeze=None,
        basis="liouville",
    )
    waveform = np.zeros((2, 1), dtype=np.float64)

    result = grape_xy_liouville(cp, waveform)
    dispatched_result = grape_xy(cp, waveform)

    np.testing.assert_allclose(result, 1.0, rtol=1e-12)
    np.testing.assert_allclose(dispatched_result, result, rtol=1e-12)


def test_grape_xy_hilbert_rejects_density_matrices() -> None:
    cp = _basic_control_problem()
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    cp.rho_init = [rho]
    cp.rho_targ = [rho]
    cp.basis = "hilbert"
    waveform = np.zeros((4, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="pure-state vector"):
        grape_xy_hilbert(cp, waveform)


def test_grape_gradient_matches_finite_difference_one_spin_hilbert() -> None:
    cp = _one_spin_hilbert_gradient_problem()
    waveform = np.array(
        [[0.15, -0.05], [0.07, 0.11], [-0.12, 0.04], [0.09, -0.08]],
        dtype=np.float64,
    )

    analytical = grape_gradient(cp, waveform)
    finite_difference = _finite_difference_gradient(cp, waveform)

    assert _relative_l2_error(analytical, finite_difference) <= 1e-5


def test_grape_gradient_matches_finite_difference_two_spin_liouville() -> None:
    drift_h = (
        0.2 * place_operator(Iz(), 0, 2)
        + 0.13 * place_operator(Iz(), 1, 2)
    )
    control_x_h = place_operator(Ix(), 0, 2) + 0.7 * place_operator(Ix(), 1, 2)
    control_y_h = place_operator(Iy(), 0, 2) - 0.4 * place_operator(Iy(), 1, 2)
    rho_init = vec(normalise_hs(state_from_label("Iz", 2)))
    rho_targ = vec(normalise_hs(state_from_label("Ix", 2)))
    cp = ControlProblem(
        drifts=[liouvillian_comm(drift_h)],
        operators=[liouvillian_comm(control_x_h), liouvillian_comm(control_y_h)],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=0.2,
        pwr_levels=[1.0, 1.0],
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
    )
    waveform = np.array(
        [[0.12, -0.09], [0.03, 0.08], [-0.07, 0.04], [0.11, -0.02]],
        dtype=np.float64,
    )

    analytical = grape_gradient(cp, waveform)
    finite_difference = _finite_difference_gradient(cp, waveform)

    assert _relative_l2_error(analytical, finite_difference) <= 1e-5


def test_grape_gradient_zeroes_frozen_entries() -> None:
    freeze = np.array(
        [[False, True], [True, False], [False, False], [True, True]],
        dtype=np.bool_,
    )
    cp = _one_spin_hilbert_gradient_problem(freeze=freeze)
    waveform = np.array(
        [[0.15, -0.05], [0.07, 0.11], [-0.12, 0.04], [0.09, -0.08]],
        dtype=np.float64,
    )

    gradient = grape_gradient(cp, waveform)

    np.testing.assert_array_equal(gradient[freeze], np.zeros(np.count_nonzero(freeze)))


def test_grape_xy_random_waveform_returns_bounded_float() -> None:
    rng = np.random.default_rng(1234)
    cp = _one_spin_hilbert_gradient_problem()
    waveform = rng.uniform(-0.2, 0.2, size=(4, 2)).astype(np.float64)

    result = grape_xy(cp, waveform)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
