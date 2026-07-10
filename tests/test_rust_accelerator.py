"""Correctness and fallback tests for the optional Rust accelerator."""

import numpy as np
import pytest

from optimalcontrol._accelerator import (
    RUST_ACCELERATOR_AVAILABLE,
    vector_fidelity,
    vector_value_gradient,
)
from optimalcontrol.grape import ControlProblem
from optimalcontrol.operators import Ix, Iy, Iz
from optimalcontrol.states import normalise_2norm


def _problem() -> ControlProblem:
    return ControlProblem(
        drifts=[np.complex128(-1j) * 0.2 * Iz()],
        operators=[np.complex128(-1j) * Ix(), np.complex128(-1j) * Iy()],
        rho_init=[np.array([1.0, 0.0], dtype=np.complex128)],
        rho_targ=[normalise_2norm(np.array([0.35 + 0.15j, 0.88 - 0.28j], dtype=np.complex128))],
        pulse_dt=0.05,
        pwr_levels=[0.8, 1.1],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )


def test_rust_accelerator_is_built() -> None:
    assert RUST_ACCELERATOR_AVAILABLE


def test_rust_vector_fidelity_matches_python_propagation(monkeypatch) -> None:
    from optimalcontrol.grape import _grape_xy_core

    cp = _problem()
    waveform = np.array(
        [[0.12, -0.03], [-0.04, 0.08], [0.08, 0.02], [0.03, -0.06]],
        dtype=np.float64,
    )
    rust_value = vector_fidelity([cp], waveform)
    assert rust_value is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value = _grape_xy_core(cp, waveform)
    np.testing.assert_allclose(rust_value, python_value, rtol=1e-12, atol=1e-12)


def test_rust_value_gradient_matches_python_adjoint(monkeypatch) -> None:
    from optimalcontrol.grape import grape_xy_and_gradient

    cp = _problem()
    waveform = np.array(
        [[0.12, -0.03], [-0.04, 0.08], [0.08, 0.02], [0.03, -0.06]],
        dtype=np.float64,
    )
    rust_result = vector_value_gradient([cp], waveform)
    assert rust_result is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value, python_gradient = grape_xy_and_gradient(cp, waveform)
    np.testing.assert_allclose(rust_result[0], python_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rust_result[1], python_gradient, rtol=1e-11, atol=1e-12)


def test_rust_fidelity_matches_python_with_relaxation(monkeypatch) -> None:
    from optimalcontrol.grape import _grape_xy_core

    cp = _problem()
    cp.drifts = [cp.drifts[0] - 0.07 * np.eye(2, dtype=np.complex128)]
    waveform = np.array([[0.12, -0.03], [-0.04, 0.08]], dtype=np.float64)
    rust_value = vector_fidelity([cp], waveform)
    assert rust_value is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value = _grape_xy_core(cp, waveform)
    np.testing.assert_allclose(rust_value, python_value, rtol=1e-12, atol=1e-12)


def test_rust_vector_fidelity_falls_back_for_matrix_states() -> None:
    cp = _problem()
    cp.rho_init = [np.eye(2, dtype=np.complex128)]
    cp.rho_targ = [np.eye(2, dtype=np.complex128)]
    waveform = np.zeros((2, 2), dtype=np.float64)

    assert vector_fidelity([cp], waveform) is None


def _sx() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _sz() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def test_hermiticity_gate_parity_for_large_drift_defect(monkeypatch) -> None:
    """A Hermitian defect below 1e-12 * scale must take the same branch on both sides."""
    from optimalcontrol.grape import _grape_xy_core

    cp = ControlProblem(
        drifts=[np.complex128(-1j) * 1e6 * _sz() + 2.5e-7 * _sz()],
        operators=[np.complex128(-1j) * _sx()],
        rho_init=[np.array([1.0, 0.0], dtype=np.complex128)],
        rho_targ=[np.array([1.0, 0.0], dtype=np.complex128)],
        pulse_dt=5e-4,
        pwr_levels=[1.0],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )
    waveform = np.zeros((2000, 1), dtype=np.float64)
    rust_value = vector_fidelity([cp], waveform)
    assert rust_value is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value = _grape_xy_core(cp, waveform)
    np.testing.assert_allclose(rust_value, python_value, rtol=0.0, atol=1e-11)


def test_hermiticity_gate_parity_for_power_scaled_defect(monkeypatch) -> None:
    """The gate must judge the power-scaled operators, like the Rust kernel does."""
    from optimalcontrol.grape import _grape_xy_core

    plus_x = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    cp = ControlProblem(
        drifts=[np.zeros((2, 2), dtype=np.complex128)],
        operators=[np.complex128(-1j) * 1e-6 * _sx() + 4e-13 * _sx()],
        rho_init=[plus_x],
        rho_targ=[plus_x],
        pulse_dt=np.pi / 100.0,
        pwr_levels=[1e6],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )
    waveform = np.ones((100, 1), dtype=np.float64)
    rust_value = vector_fidelity([cp], waveform)
    assert rust_value is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value = _grape_xy_core(cp, waveform)
    np.testing.assert_allclose(rust_value, python_value, rtol=0.0, atol=1e-11)


def test_near_degenerate_gradient_parity(monkeypatch) -> None:
    """Eigenvalue gaps just above the old degeneracy cutoff must not blow up gradients."""
    from optimalcontrol.grape import _single_value_and_gradient

    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    spectrum = np.array([1.0, 3.0, 3.0 + 3.03e-12])
    hermitian = (basis * spectrum) @ basis.conj().T
    hermitian = (hermitian + hermitian.conj().T) / 2.0

    operators = []
    for _ in range(2):
        raw = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        operators.append(np.complex128(-1j) * (raw + raw.conj().T) / 2.0)
    initial = rng.standard_normal(3) + 1j * rng.standard_normal(3)
    target = rng.standard_normal(3) + 1j * rng.standard_normal(3)

    cp = ControlProblem(
        drifts=[np.complex128(-1j) * hermitian],
        operators=operators,
        rho_init=[np.asarray(initial / np.linalg.norm(initial), dtype=np.complex128)],
        rho_targ=[np.asarray(target / np.linalg.norm(target), dtype=np.complex128)],
        pulse_dt=1.0,
        pwr_levels=[1.0, 1.0],
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )
    waveform = np.zeros((6, 2), dtype=np.float64)
    rust_result = vector_value_gradient([cp], waveform)
    assert rust_result is not None

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    python_value, python_gradient = _single_value_and_gradient(cp, waveform)
    np.testing.assert_allclose(rust_result[0], python_value, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(rust_result[1], python_gradient, rtol=0.0, atol=1e-12)


def test_rust_kernels_reject_non_finite_arrays_directly() -> None:
    """Direct private-kernel calls with NaN inputs must raise, not return NaN."""
    from optimalcontrol import _rust

    dim, steps, channels, members = 2, 3, 1, 1
    drifts = np.zeros((members, dim, dim), dtype=np.complex128)
    operators = np.zeros((members, channels, dim, dim), dtype=np.complex128)
    operators[0, 0] = np.complex128(-1j) * _sx()
    waveform = 0.3 * np.ones((steps, channels), dtype=np.float64)
    rho_init = np.array([[1.0, 0.0]], dtype=np.complex128)
    rho_targ = np.array([[0.0, 1.0]], dtype=np.complex128)

    corrupt = {
        "drifts": drifts.copy(),
        "operators": operators.copy(),
        "waveform": waveform.copy(),
        "rho_init": rho_init.copy(),
        "rho_targ": rho_targ.copy(),
    }
    for name, array in corrupt.items():
        array.flat[0] = np.nan
        arguments = {
            "drifts": drifts,
            "operators": operators,
            "waveform": waveform,
            "rho_init": rho_init,
            "rho_targ": rho_targ,
        }
        arguments[name] = array
        with pytest.raises(ValueError, match="finite"):
            _rust.grape_fidelity_vectors(*arguments.values(), 0.05, "real")
        with pytest.raises(ValueError, match="finite"):
            _rust.grape_value_gradient_vectors(*arguments.values(), 0.05, "real")


def test_rust_fidelity_and_gradient_value_agree_bitwise() -> None:
    """grape_xy and grape_xy_and_gradient[0] must agree exactly on the Rust path."""
    from dataclasses import replace

    from optimalcontrol._accelerator import problem_vector_fidelity

    # members * pairs must NOT be a power of two: the old per-member /pairs
    # grouping is bitwise identical to the unified 1/(members*pairs) scaling
    # whenever the scale is exact in binary, so a power-of-two case cannot
    # regress. 2 drifts x 3 pairs = 6 exercises the inexact-scale rounding.
    cp = _problem()
    extra_states = [
        np.array([0.0, 1.0], dtype=np.complex128),
        normalise_2norm(np.array([0.6 - 0.2j, 0.75 + 0.1j], dtype=np.complex128)),
    ]
    cp = replace(
        cp,
        drifts=[cp.drifts[0], 1.3 * np.asarray(cp.drifts[0])],
        rho_init=cp.rho_init + extra_states,
        rho_targ=cp.rho_targ + extra_states[::-1],
    )
    from optimalcontrol._accelerator import problem_vector_value_gradient

    # Equality is structural under the unified reduction, so many waveforms
    # cannot flake; under the old per-member grouping each waveform coincides
    # only by rounding luck (~2/3 per leg), so the batch discriminates. The
    # first waveform was explicitly verified to differ by 1 ulp pre-change.
    rng = np.random.default_rng(7)
    waveforms = [
        np.array(
            [[0.0411, -0.0691], [-0.1377, -0.145], [0.094, 0.1238], [0.032, 0.0688]],
            dtype=np.float64,
        )
    ] + [0.15 * rng.standard_normal((4, 2)) for _ in range(7)]

    single = replace(_problem(), rho_init=cp.rho_init, rho_targ=cp.rho_targ)
    for waveform in waveforms:
        fidelity = problem_vector_fidelity(cp, waveform)
        accelerated = problem_vector_value_gradient(cp, waveform)
        assert fidelity is not None
        assert accelerated is not None
        assert fidelity == accelerated[0]

        fidelity_single = vector_fidelity([single], waveform)
        gradient_single = vector_value_gradient([single], waveform)
        assert fidelity_single is not None
        assert gradient_single is not None
        assert fidelity_single == gradient_single[0]


def test_gradient_wrapper_skips_dissipative_problems_before_marshalling(
    monkeypatch,
) -> None:
    """Dissipative problems must return None before any marshalling happens."""
    from dataclasses import replace

    from optimalcontrol import _accelerator

    cp = _problem()
    dissipative_drift = np.asarray(cp.drifts[0], dtype=np.complex128) - 0.1 * np.eye(2)
    dissipative = replace(cp, drifts=[dissipative_drift])
    waveform = 0.2 * np.ones((3, 2), dtype=np.float64)

    def _must_not_marshal(*args: object) -> None:
        raise AssertionError("gradient wrapper marshalled a dissipative problem")

    monkeypatch.setattr(_accelerator, "_vector_inputs", _must_not_marshal)
    monkeypatch.setattr(_accelerator, "_problem_inputs", _must_not_marshal)
    assert _accelerator.vector_value_gradient([dissipative], waveform) is None
    assert _accelerator.problem_vector_value_gradient(dissipative, waveform) is None

    monkeypatch.undo()
    assert vector_fidelity([dissipative], waveform) is not None


def test_non_square_drift_raises_same_error_on_both_paths(monkeypatch) -> None:
    """The coherence pre-gate must not crash on malformed drifts before validation."""
    from dataclasses import replace

    from optimalcontrol.grape import grape_xy_and_gradient

    bad_drift = np.zeros((2, 3), dtype=np.complex128)
    bad = replace(_problem(), drifts=[bad_drift, bad_drift])
    waveform = 0.2 * np.ones((3, 2), dtype=np.float64)

    for disable_rust in ("0", "1"):
        monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)
        with pytest.raises(ValueError, match="square"):
            grape_xy_and_gradient(bad, waveform)


def test_ensemble_metadata_validation_matches_on_both_paths(monkeypatch) -> None:
    """The native ensemble path must not bypass metadata validation."""
    from dataclasses import replace

    from optimalcontrol.grape import grape_xy, grape_xy_and_gradient

    base = _problem()
    base = replace(base, drifts=[base.drifts[0], 1.1 * base.drifts[0]])
    waveform = np.zeros((3, 2), dtype=np.float64)
    cases = [
        (replace(base, freeze=np.zeros((3, 1), dtype=np.bool_)), "freeze mask"),
        (replace(base, freeze=np.zeros((2, 2), dtype=np.bool_)), "freeze mask"),
        (replace(base, freeze=np.zeros((3, 2), dtype=np.int64)), "freeze mask"),
        (replace(base, freeze=np.zeros(3, dtype=np.bool_)), "freeze mask"),
        (replace(base, basis=""), "basis"),
        (replace(base, basis=1), "basis"),
    ]

    for disable_rust in ("0", "1"):
        monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)
        for problem, message in cases:
            with pytest.raises(ValueError, match=message):
                grape_xy(problem, waveform)
            with pytest.raises(ValueError, match=message):
                grape_xy_and_gradient(problem, waveform)


def test_nan_generator_raises_on_both_paths(monkeypatch) -> None:
    """NaN drift/operator/state entries must raise ValueError, not return NaN."""
    from dataclasses import replace

    from optimalcontrol.ensemble import ensemble_fidelity
    from optimalcontrol.grape import grape_xy, grape_xy_and_gradient

    nan_drift = np.asarray(np.complex128(-1j) * _sz(), dtype=np.complex128)
    nan_drift[0, 0] = np.nan
    base = _problem()
    waveform = 0.3 * np.ones((4, 2), dtype=np.float64)

    nan_cases = [
        replace(base, drifts=[nan_drift]),
        replace(base, operators=[nan_drift, base.operators[1]]),
        replace(base, rho_init=[np.array([np.nan, 0.0], dtype=np.complex128)]),
    ]
    ensemble_cp = replace(base, drifts=[nan_drift, np.complex128(-1j) * _sz()])

    for disable_rust in ("0", "1"):
        monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)
        for cp in nan_cases:
            with pytest.raises(ValueError, match="finite"):
                grape_xy(cp, waveform)
            with pytest.raises(ValueError, match="finite"):
                grape_xy_and_gradient(cp, waveform)
        with pytest.raises(ValueError, match="finite"):
            ensemble_fidelity(ensemble_cp, waveform)


def test_negative_power_level_raises_on_both_paths(monkeypatch) -> None:
    """The Rust fast path must not bypass Python's pwr_levels validation."""
    from optimalcontrol.ensemble import ensemble_fidelity

    drift = np.complex128(-1j) * _sz()
    cp = ControlProblem(
        drifts=[drift, 1.1 * drift],
        operators=[np.complex128(-1j) * _sx()],
        rho_init=[np.array([1.0, 0.0], dtype=np.complex128)],
        rho_targ=[np.array([0.0, 1.0], dtype=np.complex128)],
        pulse_dt=0.05,
        pwr_levels=[-1.0],
        freeze=None,
        fidelity_mode="real",
        basis="hilbert",
    )
    waveform = 0.3 * np.ones((4, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        ensemble_fidelity(cp, waveform)

    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    with pytest.raises(ValueError, match="non-negative"):
        ensemble_fidelity(cp, waveform)
