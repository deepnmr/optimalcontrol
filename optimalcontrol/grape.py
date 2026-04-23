"""GRAPE control-problem containers and validation helpers."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from optimalcontrol.operators import unvec, vec
from optimalcontrol.states import fidelity_abs2, fidelity_imag, fidelity_real

Array = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

VALID_FIDELITY_MODES = {"real", "imag", "abs2"}


@dataclass
class ControlProblem:
    """Spinach-style GRAPE optimisation problem description.

    The drift and control operators define the generator basis used by later
    GRAPE propagation helpers. States may be represented as vectors or square
    matrices; validation accepts both Hilbert-space and vectorised Liouville
    dimensions.
    """

    drifts: list[Array]
    operators: list[Array]
    rho_init: list[Array]
    rho_targ: list[Array]
    pulse_dt: float
    pwr_levels: list[float]
    freeze: npt.NDArray[np.bool_] | None
    fidelity_mode: str = "real"
    offsets: list[float] | None = None
    offset_operators: list[Array] | None = None
    phase_cycle: npt.NDArray[np.float64] | None = None
    basis: str = "dense"
    penalties: list[object] | None = None
    checkpoint_path: str | None = None


def _validate_nonempty(name: str, values: Sequence[object]) -> None:
    """Raise ValueError if a required list field is empty."""
    if not values:
        raise ValueError(f"{name} must be non-empty")


def _validate_square_matrix(name: str, matrix: Array) -> int:
    """Validate a square 2-D complex array and return its dimension."""
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {array.shape}")
    return int(array.shape[0])


def _validate_state_shape(name: str, state: Array, generator_dim: int) -> None:
    """Raise ValueError if a state cannot be acted on by the generators."""
    array = np.asarray(state, dtype=np.complex128)
    if array.ndim == 1:
        if array.shape[0] != generator_dim:
            raise ValueError(
                f"{name} vector length {array.shape[0]} does not match "
                f"generator dimension {generator_dim}"
            )
        return

    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        matrix_dim = int(array.shape[0])
        liouville_dim = int(array.size)
        if matrix_dim == generator_dim or liouville_dim == generator_dim:
            return
        raise ValueError(
            f"{name} matrix shape {array.shape} is incompatible with "
            f"generator dimension {generator_dim}"
        )

    raise ValueError(f"{name} must be a vector or square matrix, got shape {array.shape}")


def _validate_float_list(name: str, values: list[float]) -> None:
    """Raise ValueError if a list of float-like values contains non-finite entries."""
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{name}[{index}] must be finite")


def _validate_freeze_mask(
    freeze: npt.NDArray[np.bool_] | None,
    n_channels: int,
) -> None:
    """Validate the optional waveform freeze mask."""
    if freeze is None:
        return
    mask = np.asarray(freeze)
    if mask.dtype != np.dtype(np.bool_):
        raise ValueError("freeze mask must have boolean dtype")
    if mask.ndim != 2:
        raise ValueError(f"freeze mask must be two-dimensional, got shape {mask.shape}")
    if mask.shape[1] != n_channels:
        raise ValueError(
            f"freeze mask has {mask.shape[1]} channels, expected {n_channels}"
        )


def _validate_phase_cycle(phase_cycle: npt.NDArray[np.float64] | None) -> None:
    """Validate optional phase-cycle weights or phase rows."""
    if phase_cycle is None:
        return
    array = np.asarray(phase_cycle, dtype=np.float64)
    if array.ndim not in (1, 2) or array.size == 0:
        raise ValueError(f"phase_cycle must be a non-empty 1-D or 2-D array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("phase_cycle entries must be finite")


def validate_control_problem(cp: ControlProblem) -> None:
    """Raise ValueError if a GRAPE control problem is internally inconsistent."""
    _validate_nonempty("drifts", cp.drifts)
    _validate_nonempty("operators", cp.operators)
    _validate_nonempty("rho_init", cp.rho_init)
    _validate_nonempty("rho_targ", cp.rho_targ)

    if cp.fidelity_mode not in VALID_FIDELITY_MODES:
        valid = ", ".join(sorted(VALID_FIDELITY_MODES))
        raise ValueError(f"fidelity_mode must be one of: {valid}")
    if cp.pulse_dt <= 0.0 or not math.isfinite(cp.pulse_dt):
        raise ValueError("pulse_dt must be finite and positive")
    if len(cp.rho_init) != len(cp.rho_targ):
        raise ValueError("rho_init and rho_targ must contain the same number of states")
    if len(cp.pwr_levels) != len(cp.operators):
        raise ValueError(
            f"pwr_levels length {len(cp.pwr_levels)} must match "
            f"operator count {len(cp.operators)}"
        )

    _validate_float_list("pwr_levels", cp.pwr_levels)
    if any(level < 0.0 for level in cp.pwr_levels):
        raise ValueError("pwr_levels entries must be non-negative")

    generator_dim = _validate_square_matrix("drifts[0]", cp.drifts[0])
    for index, drift in enumerate(cp.drifts[1:], start=1):
        dim = _validate_square_matrix(f"drifts[{index}]", drift)
        if dim != generator_dim:
            raise ValueError(
                f"drifts[{index}] dimension {dim} does not match {generator_dim}"
            )

    for index, operator in enumerate(cp.operators):
        dim = _validate_square_matrix(f"operators[{index}]", operator)
        if dim != generator_dim:
            raise ValueError(
                f"operators[{index}] dimension {dim} does not match {generator_dim}"
            )

    for index, (rho_init, rho_targ) in enumerate(zip(cp.rho_init, cp.rho_targ)):
        init_shape = np.asarray(rho_init).shape
        target_shape = np.asarray(rho_targ).shape
        if init_shape != target_shape:
            raise ValueError(
                f"rho_init[{index}] shape {init_shape} does not match "
                f"rho_targ[{index}] shape {target_shape}"
            )
        _validate_state_shape(f"rho_init[{index}]", rho_init, generator_dim)
        _validate_state_shape(f"rho_targ[{index}]", rho_targ, generator_dim)

    if cp.offsets is not None:
        _validate_float_list("offsets", cp.offsets)
    if cp.offset_operators is not None:
        for index, operator in enumerate(cp.offset_operators):
            dim = _validate_square_matrix(f"offset_operators[{index}]", operator)
            if dim != generator_dim:
                raise ValueError(
                    f"offset_operators[{index}] dimension {dim} does not match {generator_dim}"
                )

    _validate_freeze_mask(cp.freeze, len(cp.operators))
    _validate_phase_cycle(cp.phase_cycle)

    if not cp.basis:
        raise ValueError("basis must be a non-empty string")


def validate_waveform(wfm: RealArray, n_channels: int, n_steps: int) -> None:
    """Raise ValueError if a waveform is not shaped as time rows by channels."""
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    array = np.asarray(wfm, dtype=np.float64)
    expected_shape = (n_steps, n_channels)
    if array.ndim != 2 or array.shape != expected_shape:
        raise ValueError(f"waveform must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("waveform entries must be finite")


def apply_freeze(
    wfm: RealArray,
    freeze_mask: BoolArray | None,
    initial_wfm: RealArray | None = None,
) -> RealArray:
    """Return a waveform copy with frozen entries restored from an initial waveform.

    If ``initial_wfm`` is not supplied, the frozen entries are already the current
    waveform values and the function only validates the mask while returning a
    mutable copy.
    """
    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be two-dimensional, got shape {waveform.shape}")

    result = waveform.copy()
    if freeze_mask is None:
        return result

    mask = np.asarray(freeze_mask)
    if mask.dtype != np.dtype(np.bool_):
        raise ValueError("freeze mask must have boolean dtype")
    if mask.shape != waveform.shape:
        raise ValueError(f"freeze mask must have shape {waveform.shape}, got {mask.shape}")

    if initial_wfm is None:
        return result

    initial = np.asarray(initial_wfm, dtype=np.float64)
    validate_waveform(initial, waveform.shape[1], waveform.shape[0])
    result[mask] = initial[mask]
    return result


def _single_drift(cp: ControlProblem) -> Array:
    """Return the only drift currently supported by the non-ensemble path."""
    if len(cp.drifts) != 1:
        raise ValueError("forward propagation currently supports exactly one drift")
    return np.asarray(cp.drifts[0], dtype=np.complex128)


def _slice_generator(cp: ControlProblem, waveform_row: RealArray) -> Array:
    """Assemble the generator for one pulse slice."""
    generator = _single_drift(cp).copy()
    for channel_index, operator in enumerate(cp.operators):
        amplitude = np.complex128(waveform_row[channel_index] * cp.pwr_levels[channel_index])
        generator += amplitude * np.asarray(operator, dtype=np.complex128)
    return generator


def dir_diff_expm(H: Array, dH: Array, dt: float) -> Array:
    """Return d/dε expm(-1j * (H + ε dH) * dt) at ε=0.

    The derivative is computed with the standard 2x2 block auxiliary matrix:
    expm([[A, dA], [0, A]]) has the Frechet derivative in its upper-right block,
    where A = -1j * H * dt and dA = -1j * dH * dt.
    """
    H_arr = np.asarray(H, dtype=np.complex128)
    dH_arr = np.asarray(dH, dtype=np.complex128)
    dim = _validate_square_matrix("H", H_arr)
    if dH_arr.shape != H_arr.shape:
        raise ValueError(f"dH shape {dH_arr.shape} must match H shape {H_arr.shape}")
    if not math.isfinite(dt):
        raise ValueError("dt must be finite")

    A = np.complex128(-1j * dt) * H_arr
    dA = np.complex128(-1j * dt) * dH_arr
    block = np.zeros((2 * dim, 2 * dim), dtype=np.complex128)
    block[:dim, :dim] = A
    block[:dim, dim:] = dA
    block[dim:, dim:] = A
    block_expm = np.asarray(expm(block), dtype=np.complex128)
    return block_expm[:dim, dim:]


def _dir_diff_generator_expm(generator: Array, d_generator: Array, dt: float) -> Array:
    """Return d/dε expm((generator + ε d_generator) * dt) at ε=0."""
    return dir_diff_expm(
        np.complex128(1j) * generator,
        np.complex128(1j) * d_generator,
        dt,
    )


def _second_dir_diff_generator_expm(
    generator: Array,
    d_generator_a: Array,
    d_generator_b: Array,
    dt: float,
) -> Array:
    """Return the mixed second derivative of expm(generator * dt).

    The 4x4 block auxiliary matrix includes both ordered Frechet paths, yielding
    the symmetric second directional derivative in directions ``a`` and ``b``.
    """
    generator_arr = np.asarray(generator, dtype=np.complex128)
    d_a_arr = np.asarray(d_generator_a, dtype=np.complex128)
    d_b_arr = np.asarray(d_generator_b, dtype=np.complex128)
    dim = _validate_square_matrix("generator", generator_arr)
    if d_a_arr.shape != generator_arr.shape:
        raise ValueError(
            f"d_generator_a shape {d_a_arr.shape} must match "
            f"generator shape {generator_arr.shape}"
        )
    if d_b_arr.shape != generator_arr.shape:
        raise ValueError(
            f"d_generator_b shape {d_b_arr.shape} must match "
            f"generator shape {generator_arr.shape}"
        )
    if not math.isfinite(dt):
        raise ValueError("dt must be finite")

    A = np.complex128(dt) * generator_arr
    d_a = np.complex128(dt) * d_a_arr
    d_b = np.complex128(dt) * d_b_arr
    block = np.zeros((4 * dim, 4 * dim), dtype=np.complex128)
    for block_index in range(4):
        start = block_index * dim
        block[start : start + dim, start : start + dim] = A
    block[0:dim, dim : 2 * dim] = d_a
    block[0:dim, 2 * dim : 3 * dim] = d_b
    block[dim : 2 * dim, 3 * dim : 4 * dim] = d_b
    block[2 * dim : 3 * dim, 3 * dim : 4 * dim] = d_a
    block_expm = np.asarray(expm(block), dtype=np.complex128)
    return block_expm[0:dim, 3 * dim : 4 * dim]


def forward_propagators(cp: ControlProblem, wfm: RealArray) -> list[Array]:
    """Return per-slice propagators for a waveform under a control problem."""
    validate_control_problem(cp)
    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be two-dimensional, got shape {waveform.shape}")
    validate_waveform(waveform, len(cp.operators), waveform.shape[0])
    effective_waveform = apply_freeze(waveform, cp.freeze)

    propagators: list[Array] = []
    for waveform_row in effective_waveform:
        generator = _slice_generator(cp, waveform_row)
        propagators.append(np.asarray(expm(generator * cp.pulse_dt), dtype=np.complex128))
    return propagators


def _validate_square_propagator(propagator: Array) -> None:
    """Raise ValueError if a propagator is not square."""
    if propagator.ndim != 2 or propagator.shape[0] != propagator.shape[1]:
        raise ValueError(f"propagator must be square, got shape {propagator.shape}")


def _apply_propagator(state: Array, propagator: Array) -> Array:
    """Apply a Hilbert-space or Liouville-space propagator to one state."""
    state_arr = np.asarray(state, dtype=np.complex128)
    propagator_arr = np.asarray(propagator, dtype=np.complex128)
    _validate_square_propagator(propagator_arr)

    if state_arr.ndim == 1:
        if propagator_arr.shape != (state_arr.shape[0], state_arr.shape[0]):
            raise ValueError(
                f"propagator shape {propagator_arr.shape} cannot act on "
                f"state shape {state_arr.shape}"
            )
        return propagator_arr @ state_arr

    if state_arr.ndim == 2 and state_arr.shape[0] == state_arr.shape[1]:
        if propagator_arr.shape == state_arr.shape:
            return propagator_arr @ state_arr @ propagator_arr.conj().T
        liouville_shape = (state_arr.size, state_arr.size)
        if propagator_arr.shape == liouville_shape:
            return unvec(propagator_arr @ vec(state_arr), state_arr.shape[0])
        raise ValueError(
            f"propagator shape {propagator_arr.shape} cannot act on "
            f"state shape {state_arr.shape}"
        )

    raise ValueError(f"state must be a vector or square matrix, got shape {state_arr.shape}")


def _apply_derivative_propagator(state: Array, propagator: Array, d_propagator: Array) -> Array:
    """Apply one slice-propagator derivative to a state."""
    state_arr = np.asarray(state, dtype=np.complex128)
    propagator_arr = np.asarray(propagator, dtype=np.complex128)
    d_propagator_arr = np.asarray(d_propagator, dtype=np.complex128)
    _validate_square_propagator(propagator_arr)
    if d_propagator_arr.shape != propagator_arr.shape:
        raise ValueError(
            f"d_propagator shape {d_propagator_arr.shape} must match "
            f"propagator shape {propagator_arr.shape}"
        )

    if state_arr.ndim == 1:
        if propagator_arr.shape != (state_arr.shape[0], state_arr.shape[0]):
            raise ValueError(
                f"propagator shape {propagator_arr.shape} cannot act on "
                f"state shape {state_arr.shape}"
            )
        return d_propagator_arr @ state_arr

    if state_arr.ndim == 2 and state_arr.shape[0] == state_arr.shape[1]:
        if propagator_arr.shape == state_arr.shape:
            return (
                d_propagator_arr @ state_arr @ propagator_arr.conj().T
                + propagator_arr @ state_arr @ d_propagator_arr.conj().T
            )
        liouville_shape = (state_arr.size, state_arr.size)
        if propagator_arr.shape == liouville_shape:
            return unvec(d_propagator_arr @ vec(state_arr), state_arr.shape[0])
        raise ValueError(
            f"propagator shape {propagator_arr.shape} cannot act on "
            f"state shape {state_arr.shape}"
        )

    raise ValueError(f"state must be a vector or square matrix, got shape {state_arr.shape}")


def _apply_second_derivative_propagator(
    state: Array,
    propagator: Array,
    d_propagator_a: Array,
    d_propagator_b: Array,
    dd_propagator: Array,
) -> Array:
    """Apply one slice-propagator mixed second derivative to a state."""
    state_arr = np.asarray(state, dtype=np.complex128)
    propagator_arr = np.asarray(propagator, dtype=np.complex128)
    d_a_arr = np.asarray(d_propagator_a, dtype=np.complex128)
    d_b_arr = np.asarray(d_propagator_b, dtype=np.complex128)
    dd_arr = np.asarray(dd_propagator, dtype=np.complex128)
    _validate_square_propagator(propagator_arr)
    for name, matrix in (
        ("d_propagator_a", d_a_arr),
        ("d_propagator_b", d_b_arr),
        ("dd_propagator", dd_arr),
    ):
        if matrix.shape != propagator_arr.shape:
            raise ValueError(
                f"{name} shape {matrix.shape} must match "
                f"propagator shape {propagator_arr.shape}"
            )

    if state_arr.ndim == 1:
        if propagator_arr.shape != (state_arr.shape[0], state_arr.shape[0]):
            raise ValueError(
                f"propagator shape {propagator_arr.shape} cannot act on "
                f"state shape {state_arr.shape}"
            )
        return dd_arr @ state_arr

    if state_arr.ndim == 2 and state_arr.shape[0] == state_arr.shape[1]:
        if propagator_arr.shape == state_arr.shape:
            return np.asarray(
                dd_arr @ state_arr @ propagator_arr.conj().T
                + d_a_arr @ state_arr @ d_b_arr.conj().T
                + d_b_arr @ state_arr @ d_a_arr.conj().T
                + propagator_arr @ state_arr @ dd_arr.conj().T,
                dtype=np.complex128,
            )
        liouville_shape = (state_arr.size, state_arr.size)
        if propagator_arr.shape == liouville_shape:
            return unvec(dd_arr @ vec(state_arr), state_arr.shape[0])
        raise ValueError(
            f"propagator shape {propagator_arr.shape} cannot act on "
            f"state shape {state_arr.shape}"
        )

    raise ValueError(f"state must be a vector or square matrix, got shape {state_arr.shape}")


def forward_states(rho_init: Array, propagators: list[Array]) -> list[Array]:
    """Return accumulated states [rho_0, rho_1, ..., rho_N]."""
    current = np.asarray(rho_init, dtype=np.complex128).copy()
    states = [current.copy()]
    for propagator in propagators:
        current = _apply_propagator(current, propagator)
        states.append(current.copy())
    return states


def backward_states(rho_targ: Array, propagators: list[Array]) -> list[Array]:
    """Return target-first adjoint states [lambda_N, lambda_N-1, ..., lambda_0]."""
    current = np.asarray(rho_targ, dtype=np.complex128).copy()
    states = [current.copy()]
    for propagator in reversed(propagators):
        adjoint = np.asarray(propagator, dtype=np.complex128).conj().T
        current = _apply_propagator(current, adjoint)
        states.append(current.copy())
    return states


def _fidelity_by_mode(rho_f: Array, rho_t: Array, mode: str) -> float:
    """Evaluate the configured fidelity mode for one state pair."""
    if mode == "real":
        return fidelity_real(rho_f, rho_t)
    if mode == "imag":
        return fidelity_imag(rho_f, rho_t)
    if mode == "abs2":
        return fidelity_abs2(rho_f, rho_t)
    valid = ", ".join(sorted(VALID_FIDELITY_MODES))
    raise ValueError(f"mode must be one of: {valid}")


def final_fidelity(fwd_states: list[Array], bwd_states: list[Array], mode: str) -> float:
    """Return fidelity between the final forward state and initial backward target."""
    if len(fwd_states) == 0:
        raise ValueError("fwd_states must be non-empty")
    if len(bwd_states) == 0:
        raise ValueError("bwd_states must be non-empty")
    return _fidelity_by_mode(fwd_states[-1], bwd_states[0], mode)


def _fidelity_directional_derivative(
    rho_final: Array,
    rho_targ: Array,
    d_rho_final: Array,
    mode: str,
) -> float:
    """Return the fidelity directional derivative for one final-state variation."""
    rho_final_arr = np.asarray(rho_final, dtype=np.complex128)
    rho_targ_arr = np.asarray(rho_targ, dtype=np.complex128)
    d_rho_final_arr = np.asarray(d_rho_final, dtype=np.complex128)
    if rho_final_arr.shape != rho_targ_arr.shape:
        raise ValueError(
            f"State shapes must match, got {rho_final_arr.shape} and {rho_targ_arr.shape}"
        )
    if d_rho_final_arr.shape != rho_final_arr.shape:
        raise ValueError(
            f"d_rho_final shape {d_rho_final_arr.shape} must match "
            f"rho_final shape {rho_final_arr.shape}"
        )

    overlap = np.complex128(np.vdot(rho_targ_arr, rho_final_arr))
    d_overlap = np.complex128(np.vdot(rho_targ_arr, d_rho_final_arr))
    if mode == "real":
        return float(np.real(d_overlap))
    if mode == "imag":
        return float(np.imag(d_overlap))
    if mode == "abs2":
        return float(2.0 * np.real(overlap.conjugate() * d_overlap))
    valid = ", ".join(sorted(VALID_FIDELITY_MODES))
    raise ValueError(f"mode must be one of: {valid}")


def _fidelity_second_directional_derivative(
    rho_final: Array,
    rho_targ: Array,
    d_rho_a: Array,
    d_rho_b: Array,
    dd_rho: Array,
    mode: str,
) -> float:
    """Return the fidelity mixed second derivative for final-state variations."""
    rho_final_arr = np.asarray(rho_final, dtype=np.complex128)
    rho_targ_arr = np.asarray(rho_targ, dtype=np.complex128)
    d_a_arr = np.asarray(d_rho_a, dtype=np.complex128)
    d_b_arr = np.asarray(d_rho_b, dtype=np.complex128)
    dd_arr = np.asarray(dd_rho, dtype=np.complex128)
    for name, array in (
        ("rho_targ", rho_targ_arr),
        ("d_rho_a", d_a_arr),
        ("d_rho_b", d_b_arr),
        ("dd_rho", dd_arr),
    ):
        if array.shape != rho_final_arr.shape:
            raise ValueError(
                f"{name} shape {array.shape} must match "
                f"rho_final shape {rho_final_arr.shape}"
            )

    dd_overlap = np.complex128(np.vdot(rho_targ_arr, dd_arr))
    if mode == "real":
        return float(np.real(dd_overlap))
    if mode == "imag":
        return float(np.imag(dd_overlap))
    if mode == "abs2":
        overlap = np.complex128(np.vdot(rho_targ_arr, rho_final_arr))
        d_overlap_a = np.complex128(np.vdot(rho_targ_arr, d_a_arr))
        d_overlap_b = np.complex128(np.vdot(rho_targ_arr, d_b_arr))
        return float(
            2.0
            * np.real(
                d_overlap_b.conjugate() * d_overlap_a
                + overlap.conjugate() * dd_overlap
            )
        )
    valid = ", ".join(sorted(VALID_FIDELITY_MODES))
    raise ValueError(f"mode must be one of: {valid}")


def grape_gradient(cp: ControlProblem, wfm: RealArray) -> RealArray:
    """Return d(grape_xy)/d(wfm[k, c]) for all time slices and control channels."""
    validate_control_problem(cp)
    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be two-dimensional, got shape {waveform.shape}")
    validate_waveform(waveform, len(cp.operators), waveform.shape[0])
    effective_waveform = apply_freeze(waveform, cp.freeze)
    n_steps, n_channels = effective_waveform.shape

    propagators: list[Array] = []
    derivative_propagators: list[list[Array]] = []
    control_directions = [
        np.complex128(level) * np.asarray(operator, dtype=np.complex128)
        for level, operator in zip(cp.pwr_levels, cp.operators)
    ]
    for waveform_row in effective_waveform:
        generator = _slice_generator(cp, waveform_row)
        propagators.append(np.asarray(expm(generator * cp.pulse_dt), dtype=np.complex128))
        derivative_propagators.append(
            [
                _dir_diff_generator_expm(generator, control_direction, cp.pulse_dt)
                for control_direction in control_directions
            ]
        )

    gradient: RealArray = np.zeros((n_steps, n_channels), dtype=np.float64)
    for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ):
        fwd = forward_states(rho_init, propagators)
        rho_final = fwd[-1]
        for step_index in range(n_steps):
            for channel_index in range(n_channels):
                d_state = _apply_derivative_propagator(
                    fwd[step_index],
                    propagators[step_index],
                    derivative_propagators[step_index][channel_index],
                )
                for propagator in propagators[step_index + 1 :]:
                    d_state = _apply_propagator(d_state, propagator)
                gradient[step_index, channel_index] += _fidelity_directional_derivative(
                    rho_final,
                    rho_targ,
                    d_state,
                    cp.fidelity_mode,
                )

    gradient /= float(len(cp.rho_init))
    if cp.freeze is not None:
        freeze_mask = np.asarray(cp.freeze, dtype=np.bool_)
        gradient[freeze_mask] = 0.0
    return gradient


def grape_hessian(cp: ControlProblem, wfm: RealArray) -> RealArray | None:
    """Return the exact GRAPE Hessian for small waveforms, or None if too large."""
    validate_control_problem(cp)
    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be two-dimensional, got shape {waveform.shape}")
    validate_waveform(waveform, len(cp.operators), waveform.shape[0])
    effective_waveform = apply_freeze(waveform, cp.freeze)
    n_steps, n_channels = effective_waveform.shape
    n_params = n_steps * n_channels
    if n_params > 50:
        return None

    propagators: list[Array] = []
    derivative_propagators: list[list[Array]] = []
    second_derivative_propagators: list[list[list[Array]]] = []
    control_directions = [
        np.complex128(level) * np.asarray(operator, dtype=np.complex128)
        for level, operator in zip(cp.pwr_levels, cp.operators)
    ]
    for waveform_row in effective_waveform:
        generator = _slice_generator(cp, waveform_row)
        propagator = np.asarray(expm(generator * cp.pulse_dt), dtype=np.complex128)
        propagators.append(propagator)
        derivative_propagators.append(
            [
                _dir_diff_generator_expm(generator, control_direction, cp.pulse_dt)
                for control_direction in control_directions
            ]
        )
        second_derivative_propagators.append(
            [
                [
                    _second_dir_diff_generator_expm(
                        generator,
                        control_directions[channel_a],
                        control_directions[channel_b],
                        cp.pulse_dt,
                    )
                    for channel_b in range(n_channels)
                ]
                for channel_a in range(n_channels)
            ]
        )

    hessian: RealArray = np.zeros((n_params, n_params), dtype=np.float64)
    for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ):
        current = np.asarray(rho_init, dtype=np.complex128).copy()
        zero_state = np.zeros_like(current, dtype=np.complex128)
        first_states = [zero_state.copy() for _ in range(n_params)]
        second_states = [
            [zero_state.copy() for _ in range(n_params)] for _ in range(n_params)
        ]

        for step_index in range(n_steps):
            propagator = propagators[step_index]
            next_current = _apply_propagator(current, propagator)
            next_first_states = [
                _apply_propagator(first_state, propagator)
                for first_state in first_states
            ]
            next_second_states = [
                [
                    _apply_propagator(second_states[row][col], propagator)
                    for col in range(n_params)
                ]
                for row in range(n_params)
            ]

            for channel_index in range(n_channels):
                param_index = step_index * n_channels + channel_index
                d_prop = derivative_propagators[step_index][channel_index]
                for other_index in range(n_params):
                    next_second_states[param_index][other_index] += (
                        _apply_derivative_propagator(
                            first_states[other_index],
                            propagator,
                            d_prop,
                        )
                    )
                    next_second_states[other_index][param_index] += (
                        _apply_derivative_propagator(
                            first_states[other_index],
                            propagator,
                            d_prop,
                        )
                    )
                next_first_states[param_index] += _apply_derivative_propagator(
                    current,
                    propagator,
                    d_prop,
                )

            for channel_a in range(n_channels):
                param_a = step_index * n_channels + channel_a
                for channel_b in range(n_channels):
                    param_b = step_index * n_channels + channel_b
                    next_second_states[param_a][param_b] += (
                        _apply_second_derivative_propagator(
                            current,
                            propagator,
                            derivative_propagators[step_index][channel_a],
                            derivative_propagators[step_index][channel_b],
                            second_derivative_propagators[step_index][channel_a][channel_b],
                        )
                    )

            current = next_current
            first_states = next_first_states
            second_states = next_second_states

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
    if cp.freeze is not None:
        freeze_mask = np.asarray(cp.freeze, dtype=np.bool_).reshape(n_params)
        hessian[freeze_mask, :] = 0.0
        hessian[:, freeze_mask] = 0.0
    return hessian


def _grape_xy_core(cp: ControlProblem, wfm: RealArray) -> float:
    """Return scalar GRAPE fidelity without basis-specific state conversion."""
    propagators = forward_propagators(cp, wfm)
    values: list[float] = []
    for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ):
        fwd = forward_states(rho_init, propagators)
        bwd = backward_states(rho_targ, propagators)
        values.append(final_fidelity(fwd, bwd, cp.fidelity_mode))
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _vectorise_liouville_state(state: Array, generator_dim: int, name: str) -> Array:
    """Return a Liouville vector state, vectorising a density matrix if needed."""
    state_arr = np.asarray(state, dtype=np.complex128)
    if state_arr.ndim == 1:
        if state_arr.shape[0] != generator_dim:
            raise ValueError(
                f"{name} vector length {state_arr.shape[0]} does not match "
                f"Liouville generator dimension {generator_dim}"
            )
        return state_arr.copy()
    if state_arr.ndim == 2 and state_arr.shape[0] == state_arr.shape[1]:
        if state_arr.size != generator_dim:
            raise ValueError(
                f"{name} matrix shape {state_arr.shape} does not vectorise to "
                f"Liouville generator dimension {generator_dim}"
            )
        return vec(state_arr)
    raise ValueError(f"{name} must be a vector or square density matrix")


def grape_xy_liouville(cp: ControlProblem, wfm: RealArray) -> float:
    """Return GRAPE fidelity using vectorised-density Liouville propagation."""
    validate_control_problem(cp)
    generator_dim = int(np.asarray(cp.drifts[0], dtype=np.complex128).shape[0])
    rho_init = [
        _vectorise_liouville_state(state, generator_dim, f"rho_init[{index}]")
        for index, state in enumerate(cp.rho_init)
    ]
    rho_targ = [
        _vectorise_liouville_state(state, generator_dim, f"rho_targ[{index}]")
        for index, state in enumerate(cp.rho_targ)
    ]
    liouville_cp = replace(cp, rho_init=rho_init, rho_targ=rho_targ, basis="liouville")
    return _grape_xy_core(liouville_cp, wfm)


def _validate_hilbert_state(state: Array, generator_dim: int, name: str) -> Array:
    """Return a pure Hilbert-space vector state or raise ValueError."""
    state_arr = np.asarray(state, dtype=np.complex128)
    if state_arr.ndim != 1:
        raise ValueError(f"{name} must be a pure-state vector for Hilbert GRAPE")
    if state_arr.shape[0] != generator_dim:
        raise ValueError(
            f"{name} vector length {state_arr.shape[0]} does not match "
            f"Hilbert generator dimension {generator_dim}"
        )
    return state_arr.copy()


def grape_xy_hilbert(cp: ControlProblem, wfm: RealArray) -> float:
    """Return GRAPE fidelity using pure-state Hilbert-space propagation."""
    validate_control_problem(cp)
    generator_dim = int(np.asarray(cp.drifts[0], dtype=np.complex128).shape[0])
    rho_init = [
        _validate_hilbert_state(state, generator_dim, f"rho_init[{index}]")
        for index, state in enumerate(cp.rho_init)
    ]
    rho_targ = [
        _validate_hilbert_state(state, generator_dim, f"rho_targ[{index}]")
        for index, state in enumerate(cp.rho_targ)
    ]
    hilbert_cp = replace(cp, rho_init=rho_init, rho_targ=rho_targ, basis="hilbert")
    return _grape_xy_core(hilbert_cp, wfm)


def grape_xy(cp: ControlProblem, wfm: RealArray) -> float:
    """Return the scalar GRAPE fidelity for a Spinach-style XY control problem."""
    basis = cp.basis.lower()
    if basis == "liouville":
        return grape_xy_liouville(cp, wfm)
    if basis == "hilbert":
        return grape_xy_hilbert(cp, wfm)
    return _grape_xy_core(cp, wfm)
