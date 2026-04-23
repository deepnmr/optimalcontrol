"""Tests for GRAPE ensemble expansion helpers."""

import numpy as np

from optimalcontrol.ensemble import (
    ensemble_fidelity,
    ensemble_gradient,
    expand_drifts,
    expand_power_levels,
)
from optimalcontrol.grape import ControlProblem, grape_gradient, grape_xy
from optimalcontrol.operators import Ix, Iz
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


def test_ensemble_gradient_averages_expanded_problem_gradients() -> None:
    cp = _ensemble_control_problem()
    waveform = np.array([[0.12], [-0.04], [0.08]], dtype=np.float64)
    member_gradients = [
        grape_gradient(problem, waveform) for problem in _drift_power_product(cp)
    ]
    expected = np.mean(np.asarray(member_gradients, dtype=np.float64), axis=0)

    gradient = ensemble_gradient(cp, waveform)

    np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=1e-12)
