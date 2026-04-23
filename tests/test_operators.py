"""Tests for spin operators, tensor-product helpers, and commutation relations."""

import numpy as np
import numpy.testing as npt

from optimalcontrol.operators import (
    E,
    Im,
    Ip,
    Ix,
    Iy,
    Iz,
    kron_product,
    place_operator,
    two_spin_product,
)


def test_commutation_Ix_Iy() -> None:
    # [Ix, Iy] = i*Iz (standard spin-1/2 with I = sigma/2 convention)
    result = Ix() @ Iy() - Iy() @ Ix()
    npt.assert_allclose(result, 1j * Iz(), rtol=1e-12)


def test_commutation_Iy_Iz() -> None:
    result = Iy() @ Iz() - Iz() @ Iy()
    npt.assert_allclose(result, 1j * Ix(), rtol=1e-12)


def test_commutation_Iz_Ix() -> None:
    result = Iz() @ Ix() - Ix() @ Iz()
    npt.assert_allclose(result, 1j * Iy(), rtol=1e-12)


def test_Ip_equals_Ix_plus_i_Iy() -> None:
    npt.assert_allclose(Ip(), Ix() + 1j * Iy(), rtol=1e-12)


def test_Im_equals_Ix_minus_i_Iy() -> None:
    npt.assert_allclose(Im(), Ix() - 1j * Iy(), rtol=1e-12)


def test_place_operator_Iz_spin0_two_spins() -> None:
    # place_operator(Iz, 0, 2) = Iz ⊗ E
    expected = np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, -0.5, 0.0],
            [0.0, 0.0, 0.0, -0.5],
        ],
        dtype=np.complex128,
    )
    npt.assert_allclose(place_operator(Iz(), 0, 2), expected, rtol=1e-12)


def test_two_spin_product_IzSz() -> None:
    # two_spin_product(Iz, Iz) = Iz ⊗ Iz = diag(0.25, -0.25, -0.25, 0.25)
    expected = np.diag([0.25, -0.25, -0.25, 0.25]).astype(np.complex128)
    npt.assert_allclose(two_spin_product(Iz(), Iz()), expected, rtol=1e-12)


def test_kron_product_single_element() -> None:
    npt.assert_allclose(kron_product([Iz()]), Iz(), rtol=1e-12)


def test_kron_product_two_elements() -> None:
    npt.assert_allclose(kron_product([Iz(), E()]), np.kron(Iz(), E()), rtol=1e-12)


def test_kron_product_empty_raises() -> None:
    try:
        kron_product([])
        assert False, "Expected ValueError"
    except ValueError:
        pass
