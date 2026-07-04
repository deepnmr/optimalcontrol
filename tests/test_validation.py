"""Validation error-message regression tests."""

import numpy as np
import pytest

from optimalcontrol.grape import ControlProblem, validate_control_problem
from optimalcontrol.operators import Ix, Iz
from optimalcontrol.rope import rope_waveform
from optimalcontrol.states import normalise_2norm


def _valid_control_problem() -> ControlProblem:
    """Return a minimal one-spin Hilbert-space control problem."""
    rho_init = np.array([1.0, 0.0], dtype=np.complex128)
    rho_targ = normalise_2norm(
        np.array([0.35 + 0.15j, 0.88 - 0.28j], dtype=np.complex128)
    )
    return ControlProblem(
        drifts=[np.complex128(-1j) * 0.2 * Iz()],
        operators=[np.complex128(-1j) * Ix()],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=0.05,
        pwr_levels=[1.0],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )


def test_invalid_operator_dimension_reports_expected_message() -> None:
    cp = _valid_control_problem()
    cp.operators = [np.eye(3, dtype=np.complex128)]

    with pytest.raises(ValueError, match=r"operators\[0\] dimension 3 does not match 2"):
        validate_control_problem(cp)


def test_invalid_fidelity_mode_reports_allowed_values() -> None:
    cp = _valid_control_problem()
    cp.fidelity_mode = "phase"

    with pytest.raises(ValueError, match="fidelity_mode must be one of: abs2, imag, real"):
        validate_control_problem(cp)


def test_negative_j_reports_positive_requirement() -> None:
    with pytest.raises(ValueError, match="J_hz must be positive"):
        rope_waveform(T=0.01, n=1.0, J_hz=-100.0, dt=0.001)
