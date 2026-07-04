"""Tests for spin-system Hamiltonians and relaxation-rate containers."""

import numpy as np
import numpy.testing as npt

from optimalcontrol.operators import Ix, Iz, place_operator
from optimalcontrol.spin_system import (
    control_operators,
    drift_hamiltonian,
    two_spin_system,
)


def test_drift_hamiltonian_two_spin_j_coupling() -> None:
    J_hz = 140.0
    sys = two_spin_system(
        J_hz=J_hz,
        kDD=0.0,
        kCSA_I=0.0,
        kCSA_S=0.0,
        ka=0.0,
        kc=0.0,
        ka_prime=0.0,
        kc_prime=0.0,
    )

    expected = (2.0 * np.pi * J_hz) * np.diag(
        [0.25, -0.25, -0.25, 0.25]
    ).astype(np.complex128)

    npt.assert_allclose(drift_hamiltonian(sys), expected, rtol=1e-12)


def test_control_operator_Ix_on_spin_zero() -> None:
    sys = two_spin_system(
        J_hz=0.0,
        kDD=0.0,
        kCSA_I=0.0,
        kCSA_S=0.0,
        ka=0.0,
        kc=0.0,
        ka_prime=0.0,
        kc_prime=0.0,
    )

    controls = control_operators(sys)

    npt.assert_allclose(controls["Ix"], place_operator(Ix(), 0, 2), rtol=1e-12)


def test_relaxation_rates_kI_kS_properties() -> None:
    sys = two_spin_system(
        J_hz=0.0,
        kDD=1.5,
        kCSA_I=2.0,
        kCSA_S=3.0,
        ka=0.0,
        kc=0.0,
        ka_prime=0.0,
        kc_prime=0.0,
    )

    assert sys.relaxation.kI == 3.5
    assert sys.relaxation.kS == 4.5


def test_cross_correlated_rates_are_accessible() -> None:
    sys = two_spin_system(
        J_hz=0.0,
        kDD=1.0,
        kCSA_I=2.0,
        kCSA_S=3.0,
        ka=4.0,
        kc=5.0,
        ka_prime=6.0,
        kc_prime=7.0,
    )

    assert sys.relaxation.ka == 4.0
    assert sys.relaxation.kc == 5.0
    assert sys.relaxation.ka_prime == 6.0
    assert sys.relaxation.kc_prime == 7.0


def test_place_operator_Iz_matches_source_spin_convention() -> None:
    sys = two_spin_system(
        J_hz=0.0,
        kDD=0.0,
        kCSA_I=0.0,
        kCSA_S=0.0,
        ka=0.0,
        kc=0.0,
        ka_prime=0.0,
        kc_prime=0.0,
    )

    source_Iz = place_operator(Iz(), 0, len(sys.spins))

    npt.assert_allclose(source_Iz, place_operator(Iz(), 0, 2), rtol=1e-12)
