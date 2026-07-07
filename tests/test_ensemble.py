"""Tests for GRAPE ensemble expansion helpers."""

from dataclasses import replace

import numpy as np
import pytest

from optimalcontrol.ensemble import (
    cartesian_product_ensemble,
    correlated_rho_drift,
    correlated_rho_match,
    ensemble_fidelity,
    ensemble_gradient,
    ensemble_xy_and_gradient,
    expand_drifts,
    expand_offsets,
    expand_phase_cycle,
    expand_power_levels,
    joblib_backend,
    serial_backend,
)
from optimalcontrol.grape import ControlProblem, grape_gradient, grape_xy
from optimalcontrol.operators import Ix, Iz
from optimalcontrol.penalties import PenaltySpec, total_penalty
from optimalcontrol.states import normalise_2norm


def _ensemble_control_problem() -> ControlProblem:
    rho_init = np.array([1.0, 0.0], dtype=np.complex128)
    rho_targ = normalise_2norm(
        np.array([0.35 + 0.15j, 0.88 - 0.28j], dtype=np.complex128)
    )
    return ControlProblem(
        drifts=[
            np.complex128(-1j) * 0.2 * Iz(),
            np.complex128(-1j) * -0.1 * Iz(),
        ],
        operators=[np.complex128(-1j) * Ix()],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=0.05,
        pwr_levels=[0.8, 1.2],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )


def _drift_power_product(cp: ControlProblem) -> list[ControlProblem]:
    problems: list[ControlProblem] = []
    for drift_problem in expand_drifts(cp):
        problems.extend(expand_power_levels(drift_problem))
    return problems


def test_serial_backend_applies_function_in_order() -> None:
    values = serial_backend(lambda value: value * value, [1, 2, 3])

    assert values == [1, 4, 9]


def test_joblib_backend_matches_serial_backend() -> None:
    problems = [1, 2, 3]

    values = joblib_backend(lambda value: value + 0.5, problems, n_jobs=1)

    assert values == serial_backend(lambda value: value + 0.5, problems)


def test_expand_drifts_returns_one_problem_per_drift() -> None:
    cp = _ensemble_control_problem()

    problems = expand_drifts(cp)

    assert len(problems) == 2
    for index, problem in enumerate(problems):
        assert len(problem.drifts) == 1
        np.testing.assert_allclose(problem.drifts[0], cp.drifts[index], rtol=1e-12)


def test_expand_power_levels_scales_operators_and_resets_channel_power() -> None:
    cp = _ensemble_control_problem()
    cp.drifts = [cp.drifts[0]]

    problems = expand_power_levels(cp)

    assert len(problems) == 2
    for level, problem in zip(cp.pwr_levels, problems):
        assert problem.pwr_levels == [1.0]
        np.testing.assert_allclose(
            problem.operators[0],
            np.complex128(level) * cp.operators[0],
            rtol=1e-12,
        )


def test_expand_offsets_adds_offset_operator_to_each_drift() -> None:
    cp = _ensemble_control_problem()
    cp.drifts = [np.zeros((2, 2), dtype=np.complex128)]
    cp.offsets = [-0.25, 0.5]
    cp.offset_operators = [np.complex128(-1j) * Iz()]

    problems = expand_offsets(cp)

    assert len(problems) == 2
    for offset, problem in zip(cp.offsets, problems):
        assert problem.offsets is None
        assert problem.offset_operators is None
        expected = np.complex128(offset) * cp.offset_operators[0]
        np.testing.assert_allclose(problem.drifts[0], expected, rtol=1e-12)


def test_expand_phase_cycle_rotates_initial_states_by_row_phases() -> None:
    cp = _ensemble_control_problem()
    rho_a = np.array([1.0, 0.0], dtype=np.complex128)
    rho_b = np.array([0.0, 1.0], dtype=np.complex128)
    cp.rho_init = [rho_a, rho_b]
    cp.rho_targ = [rho_a.copy(), rho_b.copy()]
    cp.phase_cycle = np.array([[0.0, np.pi], [np.pi / 2.0, -np.pi / 2.0]], dtype=np.float64)

    problems = expand_phase_cycle(cp)

    assert len(problems) == 2
    assert problems[0].phase_cycle is None
    np.testing.assert_allclose(problems[0].rho_init[0], rho_a, atol=1e-12)
    np.testing.assert_allclose(problems[0].rho_init[1], -rho_b, atol=1e-12)
    np.testing.assert_allclose(problems[1].rho_init[0], np.complex128(1j) * rho_a, atol=1e-12)
    np.testing.assert_allclose(problems[1].rho_init[1], np.complex128(-1j) * rho_b, atol=1e-12)
    np.testing.assert_allclose(problems[1].rho_targ[1], rho_b, atol=1e-12)


def test_cartesian_product_ensemble_expands_all_active_axes() -> None:
    cp = _ensemble_control_problem()
    cp.offsets = [-0.2, 0.3]
    cp.offset_operators = [np.complex128(-1j) * Iz()]
    cp.phase_cycle = np.array([0.0, np.pi], dtype=np.float64)

    problems = cartesian_product_ensemble(cp)

    assert len(problems) == 16
    for problem in problems:
        assert len(problem.drifts) == 1
        assert problem.pwr_levels == [1.0]
        assert problem.offsets is None
        assert problem.offset_operators is None
        assert problem.phase_cycle is None


def test_cartesian_product_ensemble_with_two_drifts_and_two_powers_returns_four() -> None:
    cp = _ensemble_control_problem()

    problems = cartesian_product_ensemble(cp)

    assert len(problems) == 4
    for problem in problems:
        assert len(problem.drifts) == 1
        assert problem.pwr_levels == [1.0]


def test_correlated_rho_match_returns_one_problem_per_state_pair() -> None:
    cp = _ensemble_control_problem()
    rho_a = np.array([1.0, 0.0], dtype=np.complex128)
    rho_b = np.array([0.0, 1.0], dtype=np.complex128)
    targ_a = normalise_2norm(np.array([1.0, 1.0j], dtype=np.complex128))
    targ_b = normalise_2norm(np.array([1.0, -1.0j], dtype=np.complex128))
    cp.rho_init = [rho_a, rho_b]
    cp.rho_targ = [targ_a, targ_b]

    problems = correlated_rho_match(cp)

    assert len(problems) == 2
    for index, problem in enumerate(problems):
        assert len(problem.rho_init) == 1
        assert len(problem.rho_targ) == 1
        np.testing.assert_allclose(problem.rho_init[0], cp.rho_init[index], rtol=1e-12)
        np.testing.assert_allclose(problem.rho_targ[0], cp.rho_targ[index], rtol=1e-12)


def test_correlated_rho_drift_matches_drift_to_state_pair_by_index() -> None:
    cp = _ensemble_control_problem()
    rho_a = np.array([1.0, 0.0], dtype=np.complex128)
    rho_b = np.array([0.0, 1.0], dtype=np.complex128)
    cp.rho_init = [rho_a, rho_b]
    cp.rho_targ = [rho_a.copy(), rho_b.copy()]

    problems = correlated_rho_drift(cp)

    assert len(problems) == 2
    for index, problem in enumerate(problems):
        assert len(problem.drifts) == 1
        assert len(problem.rho_init) == 1
        assert len(problem.rho_targ) == 1
        np.testing.assert_allclose(problem.drifts[0], cp.drifts[index], rtol=1e-12)
        np.testing.assert_allclose(problem.rho_init[0], cp.rho_init[index], rtol=1e-12)


def test_ensemble_fidelity_averages_expanded_problem_fidelities() -> None:
    cp = _ensemble_control_problem()
    waveform = np.array([[0.12], [-0.04], [0.08]], dtype=np.float64)

    expected = float(
        np.mean(
            np.asarray(
                [grape_xy(problem, waveform) for problem in _drift_power_product(cp)],
                dtype=np.float64,
            )
        )
    )

    np.testing.assert_allclose(ensemble_fidelity(cp, waveform), expected, rtol=1e-12)


def test_ensemble_fidelity_single_member_matches_grape_xy() -> None:
    cp = _ensemble_control_problem()
    cp.drifts = [cp.drifts[0]]
    cp.pwr_levels = [1.0]
    waveform = np.array([[0.12], [-0.04], [0.08]], dtype=np.float64)

    np.testing.assert_allclose(
        ensemble_fidelity(cp, waveform),
        grape_xy(cp, waveform),
        rtol=1e-12,
    )


def test_ensemble_gradient_averages_expanded_problem_gradients() -> None:
    cp = _ensemble_control_problem()
    waveform = np.array([[0.12], [-0.04], [0.08]], dtype=np.float64)
    member_gradients = [
        grape_gradient(problem, waveform) for problem in _drift_power_product(cp)
    ]
    expected = np.mean(np.asarray(member_gradients, dtype=np.float64), axis=0)

    gradient = ensemble_gradient(cp, waveform)

    np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=1e-12)


def test_ensemble_penalties_applied_once_on_every_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = replace(
        _ensemble_control_problem(),
        penalties=[PenaltySpec("NS", 1e-3), PenaltySpec("DNS", 1e-2)],
    )
    waveform = np.array([[0.12], [-0.04], [0.08]], dtype=np.float64)
    penalty_value, penalty_gradient = total_penalty(waveform, cp.penalties)
    bare = replace(cp, penalties=None)

    results = {}
    for disable_rust in ("0", "1"):
        monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)
        expected_value = ensemble_fidelity(bare, waveform) - penalty_value
        expected_gradient = ensemble_gradient(bare, waveform) - penalty_gradient

        value, gradient = ensemble_xy_and_gradient(cp, waveform)
        np.testing.assert_allclose(value, expected_value, rtol=1e-12)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(ensemble_fidelity(cp, waveform), expected_value, rtol=1e-12)
        np.testing.assert_allclose(
            ensemble_gradient(cp, waveform), gradient, rtol=1e-9, atol=1e-14
        )
        results[disable_rust] = (value, gradient)

    np.testing.assert_allclose(results["0"][0], results["1"][0], rtol=1e-12)
    np.testing.assert_allclose(results["0"][1], results["1"][1], rtol=1e-9, atol=1e-14)

    freeze = np.zeros_like(waveform, dtype=np.bool_)
    freeze[0, 0] = True
    frozen_cp = replace(cp, freeze=freeze)
    assert penalty_gradient[0, 0] != 0.0
    assert ensemble_xy_and_gradient(frozen_cp, waveform)[1][0, 0] == 0.0
    assert ensemble_gradient(frozen_cp, waveform)[0, 0] == 0.0
