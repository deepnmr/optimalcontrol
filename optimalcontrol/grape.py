"""GRAPE control-problem containers and validation helpers."""

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from scipy.linalg import expm

# _accelerator depends only on _types, so importing it at module top creates no cycle.
from optimalcontrol._accelerator import _is_anti_hermitian, vector_fidelity, vector_value_gradient
from optimalcontrol._hessian import (
    _fidelity_second_directional_derivative as _fidelity_second_directional_derivative,
)
from optimalcontrol._hessian import _hessian_slice_propagators as _hessian_slice_propagators
from optimalcontrol._hessian import _hessian_state_sweep as _hessian_state_sweep
from optimalcontrol._hessian import _mode_value as _mode_value
from optimalcontrol._hessian import _zero_frozen_hessian as _zero_frozen_hessian
from optimalcontrol._problem import VALID_FIDELITY_MODES as VALID_FIDELITY_MODES
from optimalcontrol._problem import ControlProblem as ControlProblem
from optimalcontrol._problem import _basis_name as _basis_name
from optimalcontrol._problem import _has_ensemble_axes as _has_ensemble_axes
from optimalcontrol._problem import _has_rf_power_ensemble as _has_rf_power_ensemble
from optimalcontrol._problem import _raise_invalid_mode as _raise_invalid_mode
from optimalcontrol._problem import _validate_explicit_state_shape as _validate_explicit_state_shape
from optimalcontrol._problem import _validate_freeze_mask as _validate_freeze_mask
from optimalcontrol._problem import _validate_hilbert_state as _validate_hilbert_state
from optimalcontrol._problem import _validate_matches_propagator as _validate_matches_propagator
from optimalcontrol._problem import _validate_offset_operators as _validate_offset_operators
from optimalcontrol._problem import _validate_phase_cycle as _validate_phase_cycle
from optimalcontrol._problem import (
    _validate_same_drift_dimensions as _validate_same_drift_dimensions,
)
from optimalcontrol._problem import _validate_square_propagator as _validate_square_propagator
from optimalcontrol._problem import _validate_state_shape as _validate_state_shape
from optimalcontrol._problem import validate_control_problem as validate_control_problem
from optimalcontrol._problem import validate_waveform as validate_waveform
from optimalcontrol._propagation import _apply_derivative_propagator as _apply_derivative_propagator
from optimalcontrol._propagation import _apply_propagator as _apply_propagator
from optimalcontrol._propagation import (
    _apply_second_derivative_propagator as _apply_second_derivative_propagator,
)
from optimalcontrol._propagation import (
    _batched_derivative_propagators as _batched_derivative_propagators,
)
from optimalcontrol._propagation import _control_direction_stack as _control_direction_stack
from optimalcontrol._propagation import (
    _derivative_propagators_via_eigh as _derivative_propagators_via_eigh,
)
from optimalcontrol._propagation import _dir_diff_generator_expm as _dir_diff_generator_expm
from optimalcontrol._propagation import _effective_waveform as _effective_waveform
from optimalcontrol._propagation import _propagator_layout as _propagator_layout
from optimalcontrol._propagation import _propagators_via_eigh as _propagators_via_eigh
from optimalcontrol._propagation import (
    _second_dir_diff_generator_expm as _second_dir_diff_generator_expm,
)
from optimalcontrol._propagation import _single_drift as _single_drift
from optimalcontrol._propagation import _slice_generator_stack as _slice_generator_stack
from optimalcontrol._propagation import _zero_frozen as _zero_frozen
from optimalcontrol._propagation import apply_freeze as apply_freeze
from optimalcontrol._propagation import backward_states as backward_states
from optimalcontrol._propagation import dir_diff_expm as dir_diff_expm
from optimalcontrol._propagation import forward_states as forward_states
from optimalcontrol._types import Array, RealArray
from optimalcontrol.operators import vec
from optimalcontrol.penalties import total_penalty, total_penalty_hessian
from optimalcontrol.states import _overlap


def _validate_xy_waveform(name: str, wfm_xy: RealArray) -> None:
    """Raise ValueError if an XY waveform is not shaped as time rows by X/Y."""
    if wfm_xy.ndim != 2 or wfm_xy.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_steps, 2), got {wfm_xy.shape}")
    if not np.all(np.isfinite(wfm_xy)):
        raise ValueError(f"{name} entries must be finite")


def xy_to_ampl_phase(wfm_xy: RealArray) -> tuple[RealArray, RealArray]:
    """Return per-slice RF amplitude and phase arrays from Cartesian XY controls."""
    waveform = np.asarray(wfm_xy, dtype=np.float64)
    _validate_xy_waveform("wfm_xy", waveform)

    amplitude = np.asarray(np.hypot(waveform[:, 0], waveform[:, 1]), dtype=np.float64)
    phase = np.asarray(np.arctan2(waveform[:, 1], waveform[:, 0]), dtype=np.float64)
    return amplitude, phase


def ampl_phase_to_xy(amplitude: RealArray, phase: RealArray) -> RealArray:
    """Return an ``(n_steps, 2)`` Cartesian XY waveform from amplitude and phase."""
    amplitude_arr = np.asarray(amplitude, dtype=np.float64)
    phase_arr = np.asarray(phase, dtype=np.float64)
    if amplitude_arr.ndim != 1:
        raise ValueError(f"amplitude must be one-dimensional, got shape {amplitude_arr.shape}")
    if phase_arr.shape != amplitude_arr.shape:
        raise ValueError(
            f"phase shape {phase_arr.shape} must match amplitude shape {amplitude_arr.shape}"
        )
    if not np.all(np.isfinite(amplitude_arr)) or not np.all(np.isfinite(phase_arr)):
        raise ValueError("amplitude and phase entries must be finite")
    if np.any(amplitude_arr < 0.0):
        raise ValueError("amplitude entries must be non-negative")

    return np.asarray(
        np.column_stack((amplitude_arr * np.cos(phase_arr), amplitude_arr * np.sin(phase_arr))),
        dtype=np.float64,
    )


def phase_only_gradient(
    grad_xy: RealArray,
    amplitude: RealArray,
    phase: RealArray | None = None,
) -> RealArray:
    """Convert a Cartesian XY gradient to a phase-only gradient.

    For phase variables ``phi`` and fixed amplitudes ``A``, the Cartesian controls
    are ``x = A cos(phi)`` and ``y = A sin(phi)``. If ``phase`` is omitted and
    ``amplitude`` is an ``(n_steps, 2)`` array, it is treated as the current XY
    waveform and the same chain rule is evaluated as ``-y*g_x + x*g_y``. If only
    a 1-D amplitude is supplied, a zero-phase waveform is assumed.
    """
    gradient = np.asarray(grad_xy, dtype=np.float64)
    _validate_xy_waveform("grad_xy", gradient)

    amplitude_arr = np.asarray(amplitude, dtype=np.float64)
    if phase is None and amplitude_arr.ndim == 2:
        _validate_xy_waveform("amplitude", amplitude_arr)
        if amplitude_arr.shape != gradient.shape:
            raise ValueError(
                f"amplitude XY shape {amplitude_arr.shape} must match "
                f"grad_xy shape {gradient.shape}"
            )
        return np.asarray(
            -amplitude_arr[:, 1] * gradient[:, 0] + amplitude_arr[:, 0] * gradient[:, 1],
            dtype=np.float64,
        )

    if amplitude_arr.ndim != 1:
        raise ValueError(f"amplitude must be one-dimensional, got shape {amplitude_arr.shape}")
    if amplitude_arr.shape[0] != gradient.shape[0]:
        raise ValueError(
            f"amplitude length {amplitude_arr.shape[0]} must match grad_xy rows {gradient.shape[0]}"
        )
    if not np.all(np.isfinite(amplitude_arr)):
        raise ValueError("amplitude entries must be finite")
    if np.any(amplitude_arr < 0.0):
        raise ValueError("amplitude entries must be non-negative")

    if phase is None:
        phase_arr = np.zeros_like(amplitude_arr, dtype=np.float64)
    else:
        phase_arr = np.asarray(phase, dtype=np.float64)
        if phase_arr.shape != amplitude_arr.shape:
            raise ValueError(
                f"phase shape {phase_arr.shape} must match amplitude shape {amplitude_arr.shape}"
            )
        if not np.all(np.isfinite(phase_arr)):
            raise ValueError("phase entries must be finite")

    return np.asarray(
        amplitude_arr * (-np.sin(phase_arr) * gradient[:, 0] + np.cos(phase_arr) * gradient[:, 1]),
        dtype=np.float64,
    )


def curvilinear_reparameterise(wfm: RealArray, bounds: tuple[float, float]) -> RealArray:
    """Map an unconstrained waveform into bounds using a tanh reparameterisation.

    The mapping targets the closed interval: once ``|wfm|`` exceeds ~19,
    ``tanh`` rounds to ``+/-1.0`` in float64 and the output saturates at the
    bound to within one ulp (rounding may land it one ulp past the bound).
    Callers requiring strict interiority (e.g. log barriers) must clip the
    result themselves.
    """
    lower, upper = bounds
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("bounds must be finite")
    if lower >= upper:
        raise ValueError("bounds lower value must be less than upper value")

    waveform = np.asarray(wfm, dtype=np.float64)
    if not np.all(np.isfinite(waveform)):
        raise ValueError("wfm entries must be finite")

    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return np.asarray(midpoint + half_width * np.tanh(waveform), dtype=np.float64)


def _has_hermitian_igenerators(cp: ControlProblem, *, wfm: RealArray | None = None) -> bool:
    """Return True when ``1j * generator`` is Hermitian at the waveform scale.

    This holds exactly when ``1j * drift`` and every power-scaled
    ``1j * operator`` are Hermitian, i.e. coherent (unitary) dynamics without
    relaxation. Such problems admit a much faster eigendecomposition
    propagator path. The deviation is measured on the same power-scaled
    matrices with the same scale-relative tolerance as the Rust kernel's
    ``is_anti_hermitian_slice``, so both implementations always select the
    same propagator algorithm.
    """
    directions = _control_direction_stack(cp)
    if wfm is not None:
        amplitudes = np.maximum(1.0, np.max(np.abs(wfm), axis=0))
        directions = directions * amplitudes[:, None, None]
    candidates = [_single_drift(cp)] + list(directions)
    return all(_is_anti_hermitian(matrix) for matrix in candidates)


def _propagators_and_derivatives(
    cp: ControlProblem,
    generators: Array,
    control_directions: Array,
    wfm: RealArray,
) -> tuple[Array, Array]:
    """Return slice propagators and their control derivatives, fast path first."""
    if _has_hermitian_igenerators(cp, wfm=wfm):
        propagators, eigenvalues, eigenvectors = _propagators_via_eigh(generators, cp.pulse_dt)
        d_propagators = _derivative_propagators_via_eigh(
            eigenvalues, eigenvectors, control_directions, cp.pulse_dt
        )
        return propagators, d_propagators

    propagators = np.asarray(expm(generators * cp.pulse_dt), dtype=np.complex128)
    d_propagators = _batched_derivative_propagators(generators, control_directions, cp.pulse_dt)
    return propagators, d_propagators


def forward_propagators(cp: ControlProblem, wfm: RealArray) -> list[Array]:
    """Return per-slice propagators for a waveform under a control problem."""
    if _has_ensemble_axes(cp):
        raise ValueError("forward_propagators does not support ensemble control problems")
    effective_waveform = _effective_waveform(cp, wfm)

    generators = _slice_generator_stack(cp, effective_waveform)
    if _has_hermitian_igenerators(cp, wfm=effective_waveform):
        propagators_stack, _, _ = _propagators_via_eigh(generators, cp.pulse_dt)
    else:
        propagators_stack = np.asarray(expm(generators * cp.pulse_dt), dtype=np.complex128)
    return [np.asarray(propagator, dtype=np.complex128) for propagator in propagators_stack]


def _fidelity_by_mode(rho_f: Array, rho_t: Array, mode: str) -> float:
    """Evaluate the configured fidelity mode for one state pair."""
    return _mode_value(_overlap(rho_f, rho_t), mode)


def final_fidelity(fwd_states: list[Array], bwd_states: list[Array], mode: str) -> float:
    """Return fidelity between the final forward state and initial backward target."""
    if len(fwd_states) == 0:
        raise ValueError("fwd_states must be non-empty")
    if len(bwd_states) == 0:
        raise ValueError("bwd_states must be non-empty")
    return _fidelity_by_mode(fwd_states[-1], bwd_states[0], mode)


def _mode_gradient(
    overlap: np.complex128,
    d_overlaps: Array,
    mode: str,
) -> RealArray:
    """Return per-parameter fidelity derivatives from overlap derivatives."""
    if mode == "real":
        return np.asarray(np.real(d_overlaps), dtype=np.float64)
    if mode == "imag":
        return np.asarray(np.imag(d_overlaps), dtype=np.float64)
    if mode == "abs2":
        return np.asarray(2.0 * np.real(np.conjugate(overlap) * d_overlaps), dtype=np.float64)
    _raise_invalid_mode("mode")


def _vector_value_and_gradient(
    rho_init: Array,
    rho_targ: Array,
    propagators: Array,
    d_propagators: Array,
    mode: str,
) -> tuple[float, RealArray]:
    """Return fidelity and gradient for one vector state pair via adjoint states."""
    n_steps, dim = propagators.shape[0], propagators.shape[1]
    fwd = np.empty((n_steps + 1, dim), dtype=np.complex128)
    fwd[0] = rho_init
    for step_index in range(n_steps):
        fwd[step_index + 1] = propagators[step_index] @ fwd[step_index]

    bwd = np.empty((n_steps + 1, dim), dtype=np.complex128)
    bwd[n_steps] = rho_targ
    for step_index in range(n_steps - 1, -1, -1):
        bwd[step_index] = propagators[step_index].conj().T @ bwd[step_index + 1]

    d_states = (d_propagators @ fwd[:-1, None, :, None])[..., 0]
    d_overlaps = np.sum(bwd[1:, None, :].conj() * d_states, axis=-1)
    overlap = np.complex128(np.vdot(rho_targ, fwd[-1]))
    return _mode_value(overlap, mode), _mode_gradient(overlap, d_overlaps, mode)


def _hilbert_matrix_value_and_gradient(
    rho_init: Array,
    rho_targ: Array,
    propagators: Array,
    d_propagators: Array,
    mode: str,
) -> tuple[float, RealArray]:
    """Return fidelity and gradient for one density-matrix pair via adjoint states."""
    n_steps, n_channels = propagators.shape[0], d_propagators.shape[1]
    fwd = [rho_init]
    for step_index in range(n_steps):
        propagator = propagators[step_index]
        fwd.append(propagator @ fwd[-1] @ propagator.conj().T)

    bwd: list[Array] = [rho_targ] * (n_steps + 1)
    for step_index in range(n_steps - 1, -1, -1):
        propagator = propagators[step_index]
        bwd[step_index] = propagator.conj().T @ bwd[step_index + 1] @ propagator

    d_overlaps = np.empty((n_steps, n_channels), dtype=np.complex128)
    for step_index in range(n_steps):
        propagator = propagators[step_index]
        d_propagator = d_propagators[step_index]
        rho = fwd[step_index]
        d_slice = (d_propagator @ rho) @ propagator.conj().T + (
            propagator @ rho
        ) @ d_propagator.conj().swapaxes(-1, -2)
        d_overlaps[step_index] = np.sum(bwd[step_index + 1].conj() * d_slice, axis=(-2, -1))

    overlap = np.complex128(np.vdot(rho_targ, fwd[-1]))
    return _mode_value(overlap, mode), _mode_gradient(overlap, d_overlaps, mode)


def _single_value_and_gradient(cp: ControlProblem, wfm: RealArray) -> tuple[float, RealArray]:
    """Return bare fidelity and gradient for a single-member control problem.

    Penalties are not applied here; callers handle penalty terms so the same
    core serves both ``grape_gradient`` and ``grape_xy_and_gradient``.
    """
    effective_waveform = _effective_waveform(cp, wfm)
    cp = _problem_for_basis(cp)

    n_steps, n_channels = effective_waveform.shape

    accelerated = vector_value_gradient([cp], effective_waveform)
    if accelerated is not None:
        fidelity, accelerated_gradient = accelerated
        _zero_frozen(accelerated_gradient, cp.freeze)
        return fidelity, accelerated_gradient

    generators = _slice_generator_stack(cp, effective_waveform)
    control_directions = _control_direction_stack(cp)
    propagators, d_propagators = _propagators_and_derivatives(
        cp, generators, control_directions, effective_waveform
    )

    gradient: RealArray = np.zeros((n_steps, n_channels), dtype=np.float64)
    values: list[float] = []
    for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ, strict=True):
        init = np.asarray(rho_init, dtype=np.complex128)
        targ = np.asarray(rho_targ, dtype=np.complex128)
        layout = _propagator_layout(init, propagators[0])
        if layout == "hilbert":
            value, member_gradient = _hilbert_matrix_value_and_gradient(
                init, targ, propagators, d_propagators, cp.fidelity_mode
            )
        else:
            value, member_gradient = _vector_value_and_gradient(
                init if layout == "vector" else vec(init),
                targ if layout == "vector" else vec(targ),
                propagators,
                d_propagators,
                cp.fidelity_mode,
            )
        values.append(value)
        gradient += member_gradient

    fidelity = float(np.mean(np.asarray(values, dtype=np.float64)))
    gradient /= float(len(cp.rho_init))
    _zero_frozen(gradient, cp.freeze)
    return fidelity, gradient


def grape_xy_and_gradient(cp: ControlProblem, wfm: RealArray) -> tuple[float, RealArray]:
    """Return GRAPE fidelity and gradient from one propagation pass.

    This evaluates the same quantities as ``grape_xy`` and ``grape_gradient``
    but shares the slice propagators and adjoint states between both, which is
    roughly twice as fast as calling them separately.
    """
    waveform = np.asarray(wfm, dtype=np.float64)
    if _has_ensemble_axes(cp):
        # ensemble imports grape_xy from this module, so this import must stay function-local.
        from optimalcontrol.ensemble import ensemble_xy_and_gradient

        bare_cp = replace(cp, penalties=None)
        fidelity, gradient = ensemble_xy_and_gradient(bare_cp, waveform)
    else:
        fidelity, gradient = _single_value_and_gradient(cp, waveform)

    if cp.penalties is not None:
        penalty_value, penalty_gradient = total_penalty(waveform, cp.penalties)
        fidelity -= penalty_value
        gradient = gradient - penalty_gradient
        _zero_frozen(gradient, cp.freeze)
    return fidelity, np.asarray(gradient, dtype=np.float64)


def grape_gradient(cp: ControlProblem, wfm: RealArray) -> RealArray:
    """Return d(grape_xy)/d(wfm[k, c]) for all time slices and control channels.

    When ``cp`` has multiple drift generators or power levels, the gradient is
    averaged over the Cartesian ensemble. When ``cp.penalties`` is set, the
    penalty gradient is subtracted from the fidelity gradient before returning.
    """
    return grape_xy_and_gradient(cp, wfm)[1]


def grape_hessian(cp: ControlProblem, wfm: RealArray) -> RealArray | None:
    """Return the GRAPE objective Hessian for small waveforms, or None if too large.

    The Hessian differentiates the same objective as ``grape_gradient``: when
    ``cp.penalties`` is set, the penalty Hessian is subtracted from the exact
    fidelity Hessian.
    """
    if _has_ensemble_axes(cp):
        from optimalcontrol.ensemble import cartesian_product_ensemble

        waveform = np.asarray(wfm, dtype=np.float64)
        members = cartesian_product_ensemble(replace(cp, penalties=None))
        member_hessians: list[RealArray] = []
        for member in members:
            member_hessian = grape_hessian(member, waveform)
            if member_hessian is None:
                return None
            member_hessians.append(member_hessian)
        ensemble_hessian = np.asarray(
            np.mean(np.stack(member_hessians), axis=0),
            dtype=np.float64,
        )
        if cp.penalties is not None:
            ensemble_hessian -= total_penalty_hessian(waveform, cp.penalties)
        _zero_frozen_hessian(ensemble_hessian, cp.freeze)
        return ensemble_hessian

    effective_waveform = _effective_waveform(cp, wfm)
    cp = _problem_for_basis(cp)
    n_steps, n_channels = effective_waveform.shape
    n_params = n_steps * n_channels
    if n_params > 50:
        return None

    propagators, derivative_propagators, second_derivative_propagators = _hessian_slice_propagators(
        cp, effective_waveform
    )

    hessian: RealArray = np.zeros((n_params, n_params), dtype=np.float64)
    for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ, strict=True):
        current, first_states, second_states = _hessian_state_sweep(
            rho_init,
            propagators,
            derivative_propagators,
            second_derivative_propagators,
            n_channels,
            n_params,
        )

        for row in range(n_params):
            for col in range(n_params):
                hessian[row, col] += _fidelity_second_directional_derivative(
                    current,
                    rho_targ,
                    first_states[row],
                    first_states[col],
                    second_states[row][col],
                    cp.fidelity_mode,
                )

    hessian /= float(len(cp.rho_init))
    if cp.penalties is not None:
        hessian -= total_penalty_hessian(effective_waveform, cp.penalties)
    _zero_frozen_hessian(hessian, cp.freeze)
    return hessian


def _grape_xy_core(cp: ControlProblem, wfm: RealArray) -> float:
    """Return scalar GRAPE fidelity without basis-specific state conversion."""
    accelerated = vector_fidelity([cp], wfm)
    if accelerated is not None:
        return accelerated

    propagators = forward_propagators(cp, wfm)
    values: list[float] = []
    for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ, strict=True):
        current = np.asarray(rho_init, dtype=np.complex128)
        for propagator in propagators:
            current = _apply_propagator(current, propagator)
        values.append(
            _fidelity_by_mode(
                current,
                np.asarray(rho_targ, dtype=np.complex128),
                cp.fidelity_mode,
            )
        )
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _vectorise_liouville_state(state: Array, generator_dim: int, name: str) -> Array:
    """Return a Liouville vector state, vectorising a density matrix if needed."""
    state_arr = np.asarray(state, dtype=np.complex128)
    _validate_explicit_state_shape(name, state_arr, generator_dim, "liouville")
    return state_arr.copy() if state_arr.ndim == 1 else vec(state_arr)


def _problem_in_basis(
    cp: ControlProblem,
    convert_fn: Callable[[Array, int, str], Array],
    basis: str,
) -> ControlProblem:
    """Return a problem with states converted for an explicit basis."""
    generator_dim = int(np.asarray(cp.drifts[0], dtype=np.complex128).shape[0])
    rho_init = [
        convert_fn(state, generator_dim, f"rho_init[{index}]")
        for index, state in enumerate(cp.rho_init)
    ]
    rho_targ = [
        convert_fn(state, generator_dim, f"rho_targ[{index}]")
        for index, state in enumerate(cp.rho_targ)
    ]
    return replace(cp, rho_init=rho_init, rho_targ=rho_targ, basis=basis)


def _problem_for_basis(cp: ControlProblem) -> ControlProblem:
    """Return a problem whose states follow its explicit basis."""
    basis = _basis_name(cp.basis)
    states = cp.rho_init + cp.rho_targ
    if basis == "liouville" and any(np.asarray(state).ndim == 2 for state in states):
        return _problem_in_basis(cp, _vectorise_liouville_state, basis)
    return cp


def _grape_xy_in_basis(
    cp: ControlProblem,
    wfm: RealArray,
    convert_fn: Callable[[Array, int, str], Array],
    basis: str,
) -> float:
    """Return GRAPE fidelity after converting states with ``convert_fn``."""
    basis_cp = replace(cp, basis=basis)
    validate_control_problem(basis_cp)
    return _grape_xy_core(_problem_in_basis(basis_cp, convert_fn, basis), wfm)


def _minus_penalty(fidelity: float, wfm: RealArray, cp: ControlProblem) -> float:
    """Subtract waveform penalties from a scalar fidelity."""
    if cp.penalties is None:
        return fidelity
    penalty_value, _ = total_penalty(np.asarray(wfm, dtype=np.float64), cp.penalties)
    return fidelity - penalty_value


def grape_xy_liouville(cp: ControlProblem, wfm: RealArray) -> float:
    """Return GRAPE fidelity using vectorised-density Liouville propagation."""
    if _has_ensemble_axes(cp):
        return grape_xy(replace(cp, basis="liouville"), wfm)
    return _minus_penalty(
        _grape_xy_in_basis(cp, wfm, _vectorise_liouville_state, "liouville"),
        wfm,
        cp,
    )


def grape_xy_hilbert(cp: ControlProblem, wfm: RealArray) -> float:
    """Return GRAPE fidelity using pure-state Hilbert-space propagation."""
    if _has_ensemble_axes(cp):
        return grape_xy(replace(cp, basis="hilbert"), wfm)
    return _minus_penalty(
        _grape_xy_in_basis(cp, wfm, _validate_hilbert_state, "hilbert"),
        wfm,
        cp,
    )


def grape_xy(cp: ControlProblem, wfm: RealArray) -> float:
    """Return the scalar GRAPE fidelity for a Spinach-style XY control problem.

    When ``cp`` has multiple drift generators or power levels, the fidelity is
    averaged over the Cartesian ensemble. When ``cp.penalties`` is set, the
    summed penalty value is subtracted from the fidelity before returning.
    """
    basis = _basis_name(cp.basis)
    if _has_ensemble_axes(cp):
        from optimalcontrol.ensemble import ensemble_fidelity

        bare_cp = replace(cp, penalties=None)
        fidelity = ensemble_fidelity(bare_cp, wfm)
    else:
        if basis == "liouville":
            return grape_xy_liouville(cp, wfm)
        elif basis == "hilbert":
            return grape_xy_hilbert(cp, wfm)
        else:
            fidelity = _grape_xy_core(cp, wfm)
    return _minus_penalty(fidelity, wfm, cp)
