"""Tests for spin-system Hamiltonian and relaxation generator builders."""

import numpy as np
import numpy.testing as npt

from optimalcontrol.operators import E, Ix, Iz, liouvillian_comm, place_operator
from optimalcontrol.spin_system import (
    SpinSystem,
    drift_hamiltonian,
    relaxation_liouvillian,
    relaxation_liouvillian_crosscorr,
    shift_hamiltonian,
    total_generator,
    two_spin_system,
)

TWO_PI = 2.0 * np.pi


def _identity_orthogonal_basis(dim: int) -> np.ndarray:
    """Orthonormal basis (columns) of the Liouville subspace orthogonal to vec(I)."""
    identity_vec = np.eye(dim, dtype=np.complex128).flatten(order="F")
    # Full SVD of the 1 x dim^2 row vector: rows 1.. of vh span the complement of vec(I).
    _, _, vh = np.linalg.svd(identity_vec.reshape(1, -1))
    return vh[1:].conj().T


def _system(
    *,
    J_hz: float = 10.0,
    shifts_hz: dict[int, float] | None = None,
    kDD: float = 0.0,
    kCSA_I: float = 0.0,
    kCSA_S: float = 0.0,
    ka: float = 0.0,
    kc: float = 0.0,
    ka_prime: float = 0.0,
    kc_prime: float = 0.0,
) -> SpinSystem:
    sys = two_spin_system(
        J_hz=J_hz,
        kDD=kDD,
        kCSA_I=kCSA_I,
        kCSA_S=kCSA_S,
        ka=ka,
        kc=kc,
        ka_prime=ka_prime,
        kc_prime=kc_prime,
    )
    if shifts_hz is not None:
        sys.shifts_hz = shifts_hz
    return sys


def test_shift_hamiltonian_matches_per_spin_iz_sum() -> None:
    sys = _system(shifts_hz={0: 120.0, 1: -35.5})

    expected = TWO_PI * 120.0 * place_operator(Iz(), 0, 2) + TWO_PI * (-35.5) * place_operator(
        Iz(), 1, 2
    )

    npt.assert_allclose(shift_hamiltonian(sys), expected, rtol=1e-12, atol=1e-12)


def test_relaxation_liouvillian_vanishes_for_zero_rates() -> None:
    sys = _system()

    npt.assert_allclose(relaxation_liouvillian(sys), np.zeros((16, 16)), atol=1e-14)


def test_relaxation_liouvillian_crosscorr_vanishes_for_zero_rates() -> None:
    sys = _system()

    npt.assert_allclose(relaxation_liouvillian_crosscorr(sys), np.zeros((16, 16)), atol=1e-14)


def test_relaxation_liouvillian_is_hermitian_negative_on_identity_orthogonal_subspace() -> None:
    sys = _system(kDD=2.0, kCSA_I=0.5, kCSA_S=1.5)
    D = relaxation_liouvillian(sys)
    Q = _identity_orthogonal_basis(4)

    restricted = Q.conj().T @ D @ Q

    npt.assert_allclose(restricted, restricted.conj().T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(restricted) <= 1e-12)
    # Transverse Ix and Sx components are damped: their Rayleigh quotients are negative.
    for op_vec in (
        np.kron(Ix(), E()).flatten(order="F"),
        np.kron(E(), Ix()).flatten(order="F"),
    ):
        v = Q.conj().T @ op_vec
        assert np.real(np.vdot(v, restricted @ v)) < -1e-6


def test_relaxation_liouvillian_crosscorr_is_hermitian_on_identity_orthogonal_subspace() -> None:
    sys = _system(ka=3.0, kc=0.5, ka_prime=2.0, kc_prime=0.25)
    D = relaxation_liouvillian_crosscorr(sys)
    Q = _identity_orthogonal_basis(4)

    restricted = Q.conj().T @ D @ Q

    npt.assert_allclose(restricted, restricted.conj().T, atol=1e-12)


def test_total_generator_equals_commutator_plus_relaxation() -> None:
    sys = _system(J_hz=12.0, shifts_hz={0: 50.0, 1: -20.0}, kDD=1.0, kCSA_I=0.2, kCSA_S=0.3)
    H_drift = drift_hamiltonian(sys) + shift_hamiltonian(sys)

    expected = liouvillian_comm(H_drift) + relaxation_liouvillian(sys)

    npt.assert_allclose(total_generator(sys, {}), expected, rtol=1e-12, atol=1e-12)


def test_total_generator_uses_crosscorr_relaxation_when_rates_set() -> None:
    sys = _system(
        J_hz=12.0,
        shifts_hz={0: 50.0, 1: -20.0},
        kDD=1.0,
        kCSA_I=0.2,
        kCSA_S=0.3,
        ka=3.0,
        kc=0.5,
        ka_prime=2.0,
        kc_prime=0.25,
    )
    H_drift = drift_hamiltonian(sys) + shift_hamiltonian(sys)

    expected = liouvillian_comm(H_drift) + relaxation_liouvillian_crosscorr(sys)

    npt.assert_allclose(total_generator(sys, {}), expected, rtol=1e-12, atol=1e-12)
