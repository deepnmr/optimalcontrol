"""GRAPE control-problem containers and validation helpers."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from optimalcontrol.operators import unvec, vec

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


def forward_states(rho_init: Array, propagators: list[Array]) -> list[Array]:
    """Return accumulated states [rho_0, rho_1, ..., rho_N]."""
    current = np.asarray(rho_init, dtype=np.complex128).copy()
    states = [current.copy()]
    for propagator in propagators:
        current = _apply_propagator(current, propagator)
        states.append(current.copy())
    return states
