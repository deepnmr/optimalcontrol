"""Parity and eligibility contracts for the private spin-half ensemble kernel."""

from dataclasses import replace

import numpy as np
import pytest

from optimalcontrol import _ensemble_kernel as kernel
from optimalcontrol._accelerator import RUST_ACCELERATOR_AVAILABLE, vector_value_gradient
from optimalcontrol._ensemble_kernel import (
    _chunk_members,
    spin_half_inputs,
    spin_half_value_gradient,
)
from optimalcontrol._problem import ControlProblem
from optimalcontrol._propagation import _control_direction_stack
from optimalcontrol.ensemble import cartesian_product_ensemble
from optimalcontrol.grape import _single_value_and_gradient
from optimalcontrol.operators import Ix, Iy, Iz, L_op, R_op, liouvillian_comm, vec
from optimalcontrol.states import normalise_hs


def _methyl_problem():
    state = vec(normalise_hs(Ix()))
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=complex)],
        operators=[2 * np.pi * 10000 * liouvillian_comm(Ix())],
        rho_init=[state],
        rho_targ=[state],
        pulse_dt=7e-6,
        pwr_levels=[1.0],
        offsets=np.linspace(-9240, -2040, 81).tolist(),
        offset_operators=[2 * np.pi * liouvillian_comm(Iz())],
        basis="liouville",
    )


def _paired_problem(mode="real"):
    return replace(
        _methyl_problem(),
        drifts=[2 * np.pi * 1000 * liouvillian_comm(Iz())],
        operators=[liouvillian_comm(Ix()), liouvillian_comm(Iy())],
        pwr_levels=[2 * np.pi * 7500, 2 * np.pi * 5000],
        offsets=np.linspace(-1300, 300, 5).tolist(),
        fidelity_mode=mode,
        rho_init=[vec(normalise_hs(Ix() + 0.3 * Iy())), vec(normalise_hs(Iz() + 0.2j * Ix()))],
        rho_targ=[vec(normalise_hs(Iy() - 0.4j * Iz())), vec(normalise_hs(Ix() + 0.6 * Iz()))],
    )


def _waveform(steps=200, channels=1):
    return np.random.default_rng(0).uniform(-1, 1, (steps, channels))


def _reference(cp, waveform, monkeypatch):
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    members = cartesian_product_ensemble(cp)
    value = 0.0
    gradient = np.zeros_like(waveform)
    for member in members:
        v, g = _single_value_and_gradient(member, waveform)
        value += v
        gradient += g
    return value / len(members), gradient / len(members)


def _assert_parity(cp, waveform, monkeypatch):
    expected = _reference(cp, waveform, monkeypatch)
    inputs = spin_half_inputs(cp, waveform)
    assert inputs is not None
    value, gradient = spin_half_value_gradient(inputs, need_gradient=True)
    np.testing.assert_allclose(value, expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gradient, expected[1], rtol=1e-13, atol=1e-14)
    return inputs


def test_kernel_matches_per_member_path_methyl_like(monkeypatch):
    _assert_parity(_methyl_problem(), _waveform(), monkeypatch)


@pytest.mark.parametrize("mode", ["real", "imag", "abs2"])
def test_kernel_modes_and_channels(mode, monkeypatch):
    _assert_parity(_paired_problem(mode), _waveform(23, 2), monkeypatch)


def test_kernel_rf_power_ensemble_and_drift_ensemble(monkeypatch):
    cp = replace(
        _methyl_problem(),
        pwr_levels=[0.4, 0.9, 1.4],
        drifts=[liouvillian_comm(Iz()), 2 * liouvillian_comm(Iy())],
        offsets=[-300.0, 0.0, 700.0],
    )
    inputs = _assert_parity(cp, _waveform(19), monkeypatch)
    members = cartesian_product_ensemble(cp)
    assert inputs.n_members == 18
    np.testing.assert_array_equal(inputs.drifts, np.stack([m.drifts[0] for m in members]))
    np.testing.assert_array_equal(
        inputs.controls, np.stack([_control_direction_stack(m) for m in members])
    )


@pytest.mark.parametrize("mode", ["real", "imag"])
@pytest.mark.parametrize(
    "phases",
    [
        np.array([0.0, np.pi / 3, np.pi / 2]),
        np.array([[0.0, 0.0], [np.pi / 3, np.pi / 3], [np.pi / 2, -np.pi / 4]]),
    ],
)
def test_kernel_phase_cycle(mode, phases, monkeypatch):
    cp = replace(
        _paired_problem(mode),
        phase_cycle=phases,
        drifts=[liouvillian_comm(Ix()), liouvillian_comm(Iz())],
        pwr_levels=[0.4, 0.9, 1.4],
    )
    inputs = _assert_parity(cp, _waveform(17, 2), monkeypatch)
    members = cartesian_product_ensemble(cp)
    for actual, expected in [
        (inputs.drifts, np.stack([m.drifts[0] for m in members])),
        (inputs.controls, np.stack([_control_direction_stack(m) for m in members])),
        (inputs.rho_init, np.stack([m.rho_init for m in members])),
        (inputs.rho_targ, np.stack([m.rho_targ for m in members])),
    ]:
        assert np.array_equal(actual, expected)


def test_kernel_matches_rust_and_finite_difference(monkeypatch):
    assert RUST_ACCELERATOR_AVAILABLE
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "0")
    cp, waveform = _paired_problem("abs2"), _waveform(13, 2)
    native = vector_value_gradient(cartesian_product_ensemble(cp), waveform)
    assert native is not None
    inputs = spin_half_inputs(cp, waveform)
    assert inputs is not None
    value, gradient = spin_half_value_gradient(inputs, need_gradient=True)
    np.testing.assert_allclose(value, native[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, native[1], rtol=1e-9, atol=1e-14)
    for flat in np.random.default_rng(3).choice(waveform.size, 6, replace=False):
        coordinate = np.unravel_index(flat, waveform.shape)
        values = []
        for sign in [1, -1]:
            perturbed = waveform.copy()
            perturbed[coordinate] += sign * 1e-6
            perturbed_inputs = spin_half_inputs(cp, perturbed)
            assert perturbed_inputs is not None
            values.append(spin_half_value_gradient(perturbed_inputs, need_gradient=False)[0])
        np.testing.assert_allclose(
            gradient[coordinate], (values[0] - values[1]) / 2e-6, rtol=1e-5, atol=1e-9
        )


@pytest.mark.parametrize(
    "case",
    [
        "dissipative",
        "scaled_control",
        "hilbert",
        "two_spin",
        "dense_matrix",
        "non_commutator",
        "non_hermitian",
        "mode",
        "empty",
        "freeze",
        "nan",
        "scalar",
        "three_dimensional",
        "zero_steps",
        "invalid_basis",
        "nonfinite_state",
        "mismatched_pairs",
        "offset_missing",
        "empty_offsets",
    ],
)
def test_kernel_declines(case):
    cp, waveform = _methyl_problem(), _waveform()
    if case == "dissipative":
        cp.drifts = [cp.drifts[0] - 0.07 * np.eye(4)]
    elif case == "scaled_control":
        cp.operators = [-1e-13 * np.eye(4, dtype=complex)]
        waveform[:] = 1e13
    elif case in {"hilbert", "two_spin"}:
        dim = 2 if case == "hilbert" else 16
        cp = replace(
            cp,
            drifts=[np.zeros((dim, dim), dtype=complex)],
            operators=[np.zeros((dim, dim), dtype=complex)],
            rho_init=[np.ones(dim, dtype=complex)],
            rho_targ=[np.ones(dim, dtype=complex)],
            offsets=None,
            offset_operators=None,
            basis="hilbert" if dim == 2 else "liouville",
        )
    elif case == "dense_matrix":
        cp = replace(
            cp,
            basis="dense",
            rho_init=[np.eye(2, dtype=complex)],
            rho_targ=[np.eye(2, dtype=complex)],
        )
    elif case == "non_commutator":
        cp.drifts = [-1j * np.diag([1.0, 2.0, 3.0, 4.0])]
    elif case == "non_hermitian":
        cp.drifts = [liouvillian_comm(np.array([[0, 1], [0, 0]], dtype=complex))]
    elif case == "mode":
        cp.fidelity_mode = "abs"
    elif case == "empty":
        cp.rho_init = []
    elif case == "freeze":
        cp.freeze = np.zeros((1, 1), dtype=bool)
    elif case == "nan":
        waveform[0, 0] = np.nan
    elif case == "scalar":
        waveform = np.array(0.0)
    elif case == "three_dimensional":
        waveform = waveform[:, :, None]
    elif case == "zero_steps":
        waveform = waveform[:0]
    elif case == "invalid_basis":
        cp.basis = "invalid"
    elif case == "nonfinite_state":
        cp.rho_init[0] = np.full(4, np.nan, dtype=complex)
    elif case == "mismatched_pairs":
        cp.rho_targ = cp.rho_targ * 2
    elif case == "offset_missing":
        cp.offsets = None
    elif case == "empty_offsets":
        cp.offsets = []
    assert spin_half_inputs(cp, waveform) is None


def test_kernel_value_only_matches_value_gradient():
    inputs = spin_half_inputs(_methyl_problem(), _waveform())
    assert inputs is not None
    value, gradient = spin_half_value_gradient(inputs, need_gradient=True)
    value_only, absent = spin_half_value_gradient(inputs, need_gradient=False)
    assert value_only == value
    assert gradient.shape == (200, 1)
    assert absent is None


def test_kernel_errstate_clean(monkeypatch):
    cp = replace(_methyl_problem(), offsets=[0.0, -300.0, 300.0])
    waveform = _waveform()
    waveform[0] = 0.0
    with np.errstate(all="raise"):
        _assert_parity(cp, waveform, monkeypatch)


@pytest.mark.parametrize("scale", [1e-200, 1e-160, 1e155, 1e200])
def test_kernel_is_scale_invariant_at_float_range_edges(scale, monkeypatch):
    # Squaring Hamiltonian components of this size underflows (an apparent exact
    # degeneracy that replaces the evolution by the identity) or overflows (NaN),
    # while dt*H stays ordinary and the per-member path is exact.
    state = vec(normalise_hs(Ix()))
    cp = ControlProblem(
        drifts=[liouvillian_comm(scale * Iz()), liouvillian_comm(0.5 * scale * Iz())],
        operators=[liouvillian_comm(scale * Ix())],
        rho_init=[state],
        rho_targ=[state],
        pulse_dt=1.0 / scale,
        pwr_levels=[1.0],
        basis="liouville",
    )
    waveform = np.array([[0.0], [0.3], [-0.7]])
    with np.errstate(all="raise"):
        _assert_parity(cp, waveform, monkeypatch)


@pytest.mark.parametrize("scale", [1e-310, 1e-320, 5e-324])
def test_kernel_direction_survives_subnormal_components(scale, monkeypatch):
    # A subnormal radius carries few significant bits, so dividing the raw components
    # by it does not give a unit direction; the direction must come from components
    # rescaled by their largest magnitude. The zero-radius member stays the identity.
    state = vec(normalise_hs(Ix()))
    off_axis = np.array([[0.0, 1.0 - 1.0j], [1.0 + 1.0j, 0.0]])
    cp = ControlProblem(
        drifts=[
            np.zeros((4, 4), dtype=complex),
            liouvillian_comm(scale * off_axis),
            liouvillian_comm(scale * Iz()),
            liouvillian_comm(1e200 * Iz()),
        ],
        operators=[liouvillian_comm(Ix())],
        rho_init=[state],
        rho_targ=[state],
        pulse_dt=1e-200,
        pwr_levels=[1.0],
        basis="liouville",
    )
    waveform = np.zeros((2, 1))
    expected = _reference(cp, waveform, monkeypatch)
    inputs = spin_half_inputs(cp, waveform)
    assert inputs is not None
    # Subnormal inputs raise the underflow flag in any arithmetic (the per-member path
    # does too); only invalid, divide and overflow would indicate a lost direction.
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        value, gradient = spin_half_value_gradient(inputs, need_gradient=True)
    np.testing.assert_allclose(value, expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gradient, expected[1], rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("backend", ["0", "1"])
def test_kernel_chunking_is_bitwise_invariant(backend, monkeypatch):
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", backend)
    inputs = spin_half_inputs(_methyl_problem(), _waveform())
    assert inputs is not None
    value, gradient = spin_half_value_gradient(inputs, need_gradient=True)
    for chunk in (1, 7, 1000):
        monkeypatch.setattr(kernel, "_CHUNK_OVERRIDE", chunk)
        chunked_value, chunked_gradient = spin_half_value_gradient(inputs, need_gradient=True)
        assert chunked_value == value
        assert np.array_equal(chunked_gradient, gradient)


def test_chunk_members_policy():
    assert _chunk_members(200, 1, 1, 81) == 81
    assert _chunk_members(200, 1, 1, 100000) == 1044
    assert _chunk_members(200, 1, 100, 100000) == 17
    assert _chunk_members(200, 1, 1, 5, budget_bytes=1) == 1
    for sizes in ([50, 100, 200, 400, 800], [1, 2, 3, 4], [1, 10, 100, 1000]):
        for axis in range(3):
            values = [
                _chunk_members(
                    *[sizes[i] if j == axis else (200, 1, 1)[j] for j in range(3)], 100000
                )
                for i in range(len(sizes))
            ]
            assert values == sorted(values, reverse=True)


def _one_drift_problem(generator):
    return replace(
        _methyl_problem(),
        drifts=[generator],
        offsets=None,
        offset_operators=None,
        pwr_levels=[1.0],
    )


def _reconstruction_residual(generator):
    hermitian = 1j * generator
    block = hermitian[:2, :2]
    return float(np.max(np.abs(L_op(block) - R_op(block) - hermitian)))


def test_kernel_commutator_gate_is_scale_relative():
    base = liouvillian_comm(Iz())
    below = base + 1j * 0.5e-12 * max(1.0, float(np.max(np.abs(base)))) * np.eye(4)
    above = base + 1j * 2e-12 * max(1.0, float(np.max(np.abs(base)))) * np.eye(4)
    for generator, accepted in [(below, True), (above, False)]:
        scale = max(1.0, float(np.max(np.abs(generator))))
        residual = _reconstruction_residual(generator)
        assert (residual <= 1e-12 * scale) is accepted
        inputs = spin_half_inputs(_one_drift_problem(generator), _waveform())
        assert (inputs is not None) is accepted


def test_kernel_hermitian_defect_declines_at_anti_hermitian_threshold():
    base = liouvillian_comm(Iz())
    generator = base + 7.5e-13 * max(1.0, float(np.max(np.abs(base)))) * np.eye(4)
    scale = max(1.0, float(np.max(np.abs(generator))))
    assert _reconstruction_residual(generator) <= 1e-12 * scale
    assert spin_half_inputs(_one_drift_problem(generator), _waveform()) is None


def test_kernel_liouville_density_matrices(monkeypatch):
    cp = replace(
        _methyl_problem(),
        rho_init=[normalise_hs(Ix())],
        rho_targ=[normalise_hs(Ix())],
        offsets=[0.0, 100.0],
    )
    _assert_parity(cp, _waveform(7), monkeypatch)
