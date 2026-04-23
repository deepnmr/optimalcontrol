"""Tests for Liouville-space utilities and superoperators."""

import numpy as np
import numpy.testing as npt
import numpy.typing as np_typing

from optimalcontrol.operators import (
    Im,
    comm,
    liouvillian_comm,
    lindblad_dissipator,
    unvec,
    vec,
)


def _random_hermitian(n: int, rng: np.random.Generator) -> np_typing.NDArray[np.complex128]:
    """Return a random n×n Hermitian matrix."""
    a: np_typing.NDArray[np.complex128] = rng.standard_normal((n, n)).astype(np.complex128)
    b: np_typing.NDArray[np.complex128] = rng.standard_normal((n, n)).astype(np.complex128)
    c: np_typing.NDArray[np.complex128] = a + np.complex128(1j) * b
    return ((c + c.conj().T) * 0.5).astype(np.complex128)


def test_vec_unvec_roundtrip_2x2() -> None:
    rng = np.random.default_rng(42)
    rho = _random_hermitian(2, rng)
    npt.assert_allclose(unvec(vec(rho), 2), rho, rtol=1e-12)


def test_vec_unvec_roundtrip_4x4() -> None:
    rng = np.random.default_rng(42)
    rho = _random_hermitian(4, rng)
    npt.assert_allclose(unvec(vec(rho), 4), rho, rtol=1e-12)


def test_liouvillian_comm_matches_direct_comm() -> None:
    rng = np.random.default_rng(7)
    A = _random_hermitian(4, rng)
    rho = _random_hermitian(4, rng)
    superop_result = liouvillian_comm(A) @ vec(rho)
    direct_result = vec((np.complex128(-1j) * comm(A, rho)).astype(np.complex128))
    npt.assert_allclose(superop_result, direct_result, rtol=1e-12)


def test_lindblad_dissipator_single_operator() -> None:
    rng = np.random.default_rng(13)
    F = Im()
    a_kl = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    D = lindblad_dissipator([F], a_kl)
    rho = _random_hermitian(2, rng)
    Fd: np_typing.NDArray[np.complex128] = F.conj().T.astype(np.complex128)
    expected_mat: np_typing.NDArray[np.complex128] = (
        F @ rho @ Fd - 0.5 * (Fd @ F @ rho + rho @ Fd @ F)
    ).astype(np.complex128)
    npt.assert_allclose(unvec(D @ vec(rho), 2), expected_mat, rtol=1e-12)
