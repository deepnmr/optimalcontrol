"""Tests for product-operator states, fidelity, and state hooks."""

import numpy as np
import numpy.testing as npt

from optimalcontrol.operators import E, Ix, Iz, place_operator, two_spin_product
from optimalcontrol.states import (
    dead_time_propagation,
    fidelity_real,
    normalise_hs,
    single_transition_operator,
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


def test_single_transition_IzSalpha_matches_hand_built_tensor() -> None:
    s_alpha = 0.5 * E() + Iz()

    expected = np.kron(Iz(), s_alpha)

    npt.assert_allclose(single_transition_operator("IzSalpha"), expected, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(
        single_transition_operator("IzSalpha"),
        two_spin_product(Iz(), s_alpha),
        rtol=1e-12,
        atol=1e-12,
    )


def test_single_transition_IxSbeta_matches_hand_built_tensor() -> None:
    s_beta = 0.5 * E() - Iz()

    npt.assert_allclose(
        single_transition_operator("IxSbeta"), np.kron(Ix(), s_beta), rtol=1e-12, atol=1e-12
    )


def test_single_transition_Iz_labels_are_traceless() -> None:
    npt.assert_allclose(np.trace(single_transition_operator("IzSalpha")), 0.0, atol=1e-12)
    npt.assert_allclose(np.trace(single_transition_operator("IzSbeta")), 0.0, atol=1e-12)


def test_single_transition_real_axis_labels_are_hermitian() -> None:
    for label in ("IzSalpha", "IxSbeta"):
        op = single_transition_operator(label)
        npt.assert_allclose(op, op.conj().T, atol=1e-12)


def test_single_transition_is_unnormalised_and_normalises_to_unit_hs_norm() -> None:
    op = single_transition_operator("IzSalpha")
    # Unnormalised: a spin operator times a projector, not a projector itself.
    assert not np.isclose(np.trace(op @ op.conj().T).real, 1.0)

    normalised = normalise_hs(op)

    npt.assert_allclose(np.trace(normalised.conj().T @ normalised).real, 1.0, rtol=1e-12)


def test_normalise_hs_scale_safe_outside_double_range() -> None:
    expected = np.eye(2, dtype=np.complex128) / np.sqrt(2.0)
    for scale in (1e-200, 1e200):
        npt.assert_allclose(
            normalise_hs(scale * np.eye(2, dtype=np.complex128)), expected, atol=1e-12
        )


def test_normalise_hs_in_range_bits_unchanged() -> None:
    arr = np.array([[1.0, 2.0 + 1.0j], [-0.5j, 3.0]], dtype=np.complex128)
    pre_change = arr / np.sqrt(np.real(np.trace(arr.conj().T @ arr)))
    np.testing.assert_array_equal(normalise_hs(arr), pre_change)
