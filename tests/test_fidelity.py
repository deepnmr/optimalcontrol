"""Tests for overlap fidelity helpers."""

import numpy as np
import numpy.testing as npt
import pytest

from optimalcontrol.states import (
    fidelity_abs2,
    fidelity_avg,
    fidelity_imag,
    fidelity_real,
    normalise_hs,
    state_from_label,
)


def test_fidelity_real_self_overlap_is_one_for_normalised_state() -> None:
    rho = normalise_hs(state_from_label("Iz", 2))

    npt.assert_allclose(fidelity_real(rho, rho), 1.0, rtol=1e-12)


def test_fidelity_imag_and_abs2_use_complex_overlap() -> None:
    rho_t = normalise_hs(state_from_label("Ix", 2))
    rho_f = np.complex128(1j) * rho_t

    npt.assert_allclose(fidelity_real(rho_f, rho_t), 0.0, atol=1e-12)
    npt.assert_allclose(fidelity_imag(rho_f, rho_t), 1.0, rtol=1e-12)
    npt.assert_allclose(fidelity_abs2(rho_f, rho_t), 1.0, rtol=1e-12)


def test_fidelity_avg_weighted_real_fidelity() -> None:
    rho_1 = normalise_hs(state_from_label("Ix", 2))
    rho_2 = normalise_hs(state_from_label("Iy", 2))
    rho_3 = -rho_2

    result = fidelity_avg([rho_1, rho_3], [rho_1, rho_2], weights=[1.0, 3.0])

    npt.assert_allclose(result, -0.5, rtol=1e-12)


def test_fidelity_shape_mismatch_raises() -> None:
    rho_2 = normalise_hs(state_from_label("Iz", 1))
    rho_4 = normalise_hs(state_from_label("Iz", 2))

    with pytest.raises(ValueError, match="State shapes must match"):
        fidelity_real(rho_2, rho_4)
