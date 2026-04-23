"""Tests for product-operator states, fidelity, and state hooks."""

import numpy as np
import numpy.testing as npt

from optimalcontrol.operators import Iz, place_operator
from optimalcontrol.states import (
    dead_time_propagation,
    fidelity_real,
    normalise_hs,
    state_from_label,
)


def test_state_from_label_Iz_matches_source_spin_operator() -> None:
    rho = state_from_label("Iz", 2)

    npt.assert_allclose(rho, place_operator(Iz(), 0, 2), rtol=1e-12)


def test_fidelity_real_self_overlap_is_one_for_normalised_state() -> None:
    rho = normalise_hs(state_from_label("Iz", 2))

    npt.assert_allclose(fidelity_real(rho, rho), 1.0, rtol=1e-12)


def test_paper_transfer_states_are_orthogonal_before_propagation() -> None:
    rho_init = normalise_hs(state_from_label("Iz", 2))
    rho_target = normalise_hs(state_from_label("2IzSz", 2))

    npt.assert_allclose(fidelity_real(rho_init, rho_target), 0.0, atol=1e-12)
    npt.assert_allclose(fidelity_real(rho_target, rho_target), 1.0, rtol=1e-12)


def test_dead_time_propagation_zero_time_returns_state_unchanged() -> None:
    rho = state_from_label("Iz", 2)
    generator = np.eye(rho.size, dtype=np.complex128)

    result = dead_time_propagation(rho, generator, t_dead=0.0)

    npt.assert_allclose(result, rho, rtol=1e-12)
