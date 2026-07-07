"""GRAPE control-problem containers and validation helpers."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import NoReturn

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from optimalcontrol._types import Array, BoolArray, RealArray
from optimalcontrol._validation import validate_finite_floats as _validate_float_list
from optimalcontrol._validation import validate_nonempty as _validate_nonempty
from optimalcontrol._validation import validate_square_matrix as _validate_square_matrix
from optimalcontrol.operators import unvec, vec
from optimalcontrol.penalties import PenaltyInput, total_penalty, total_penalty_hessian
from optimalcontrol.states import _overlap

VALID_FIDELITY_MODES = {"real", "imag", "abs2"}


def _raise_invalid_mode(label: str) -> NoReturn:
    """Raise a ValueError naming the accepted fidelity modes for ``label``."""
    valid = ", ".join(sorted(VALID_FIDELITY_MODES))
    raise ValueError(f"{label} must be one of: {valid}")


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
    penalties: list[PenaltyInput] | None = None
    checkpoint_path: str | None = None


def _has_rf_power_ensemble(cp: ControlProblem) -> bool:
    """Return True when pwr_levels represents an RF ensemble axis."""
    return len(cp.pwr_levels) > 1 and len(cp.pwr_levels) != len(cp.operators)


def _has_ensemble_axes(cp: ControlProblem) -> bool:
    """Return True when a control problem needs ensemble expansion."""
    return (
        len(cp.drifts) > 1
        or _has_rf_power_ensemble(cp)
        or cp.offsets is not None
        or cp.offset_operators is not None
        or cp.phase_cycle is not None
    )


def _validate_state_shape(name: str, state: Array, generator_dim: int) -> None:
    """Raise ValueError if a state cannot be acted on by the generators."""
    array = np.asarray(state, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} entries must be finite")
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


def _validate_same_drift_dimensions(drifts: Sequence[Array]) -> int:
    """Validate drift generators and return their shared dimension."""
    _validate_nonempty("drifts", drifts)
    generator_dim = _validate_square_matrix("drifts[0]", drifts[0])
    for index, drift in enumerate(drifts[1:], start=1):
        dim = _validate_square_matrix(f"drifts[{index}]", drift)
        if dim != generator_dim:
            raise ValueError(f"drifts[{index}] dimension {dim} does not match {generator_dim}")
    return generator_dim


def validate_control_problem(cp: ControlProblem) -> None:
    """Raise ValueError if a GRAPE control problem is internally inconsistent."""
    _validate_nonempty("drifts", cp.drifts)
    _validate_nonempty("operators", cp.operators)
    _validate_nonempty("rho_init", cp.rho_init)
    _validate_nonempty("rho_targ", cp.rho_targ)

    if cp.fidelity_mode not in VALID_FIDELITY_MODES:
        _raise_invalid_mode("fidelity_mode")
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

    generator_dim = _validate_same_drift_dimensions(cp.drifts)

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


def _zero_frozen(gradient: RealArray, freeze: npt.NDArray[np.bool_] | None) -> None:
    """Zero gradient entries on frozen waveform positions, in place."""
    if freeze is not None:
        gradient[np.asarray(freeze, dtype=np.bool_)] = 0.0


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
            f"amplitude length {amplitude_arr.shape[0]} must match "
            f"grad_xy rows {gradient.shape[0]}"
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


def _single_drift(cp: ControlProblem) -> Array:
    """Return the only drift currently supported by the non-ensemble path."""
    if len(cp.drifts) != 1:
        raise ValueError("forward propagation currently supports exactly one drift")
    return np.asarray(cp.drifts[0], dtype=np.complex128)


def _control_direction_stack(cp: ControlProblem) -> Array:
    """Return power-scaled control operators stacked as ``(n_channels, dim, dim)``."""
    operators = np.stack(
        [np.asarray(operator, dtype=np.complex128) for operator in cp.operators]
    )
    levels = np.asarray(cp.pwr_levels, dtype=np.complex128)
    return np.asarray(levels[:, None, None] * operators, dtype=np.complex128)


def _slice_generator_stack(cp: ControlProblem, effective_waveform: RealArray) -> Array:
    """Return all slice generators stacked as ``(n_steps, dim, dim)``."""
    drift = _single_drift(cp)
    control_directions = _control_direction_stack(cp)
    scaled = np.asarray(effective_waveform, dtype=np.complex128)
    generators = drift[None, :, :] + np.tensordot(scaled, control_directions, axes=(1, 0))
    return np.asarray(generators, dtype=np.complex128)


def _has_hermitian_igenerators(cp: ControlProblem) -> bool:
    """Return True when ``1j * generator`` is Hermitian for every waveform.

    This holds exactly when ``1j * drift`` and every power-scaled
    ``1j * operator`` are Hermitian, i.e. coherent (unitary) dynamics without
    relaxation. Such problems admit a much faster eigendecomposition
    propagator path. The deviation is measured on the same power-scaled
    matrices with the same scale-relative tolerance as the Rust kernel's
    ``is_anti_hermitian_slice``, so both implementations always select the
    same propagator algorithm.
    """
    candidates = [_single_drift(cp)] + list(_control_direction_stack(cp))
    for matrix in candidates:
        deviation = float(np.abs(matrix + matrix.conj().T).max())
        scale = max(1.0, float(np.abs(matrix).max()))
        if deviation > 1e-12 * scale:
            return False
    return True


def _propagators_via_eigh(generators: Array, dt: float) -> tuple[Array, RealArray, Array]:
    """Return ``(propagators, eigenvalues, eigenvectors)`` for Hermitian ``1j*G``.

    ``expm(G * dt) = V exp(-1j * lambda * dt) V^dagger`` where
    ``1j * G = V diag(lambda) V^dagger``.
    """
    hermitian = np.asarray(np.complex128(1j) * generators, dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    phases = np.exp(np.complex128(-1j * dt) * eigenvalues)
    propagators = np.asarray(
        (eigenvectors * phases[:, None, :]) @ eigenvectors.conj().transpose(0, 2, 1),
        dtype=np.complex128,
    )
    return propagators, eigenvalues, eigenvectors


def _derivative_propagators_via_eigh(
    eigenvalues: RealArray,
    eigenvectors: Array,
    control_directions: Array,
    dt: float,
) -> Array:
    """Return slice-propagator control derivatives from eigendecompositions.

    Uses the Daleckii-Krein formula in its cancellation-free sinc form:
    ``(exp(-1j*a*dt) - exp(-1j*b*dt)) / (a - b)`` equals
    ``-1j*dt * exp(-1j*dt*(a+b)/2) * sinc((a-b)*dt/2)``, which stays
    well-conditioned for near-degenerate eigenvalue pairs and matches the
    Rust kernel expression exactly.
    """
    half_gap = 0.5 * dt * (eigenvalues[:, :, None] - eigenvalues[:, None, :])
    safe_gap = np.where(half_gap == 0.0, 1.0, half_gap)
    sinc = np.where(half_gap == 0.0, 1.0, np.sin(safe_gap) / safe_gap)
    midpoint = 0.5 * (eigenvalues[:, :, None] + eigenvalues[:, None, :])
    divided = np.complex128(-1j * dt) * np.exp(np.complex128(-1j * dt) * midpoint) * sinc

    adjoint_vectors = eigenvectors.conj().transpose(0, 2, 1)
    hermitian_directions = np.complex128(1j) * control_directions
    rotated = (
        adjoint_vectors[:, None, :, :]
        @ hermitian_directions[None, :, :, :]
        @ eigenvectors[:, None, :, :]
    )
    weighted = divided[:, None, :, :] * rotated
    return np.asarray(
        eigenvectors[:, None, :, :] @ weighted @ adjoint_vectors[:, None, :, :],
        dtype=np.complex128,
    )


def _propagators_and_derivatives(
    cp: ControlProblem,
    generators: Array,
    control_directions: Array,
) -> tuple[Array, Array]:
    """Return slice propagators and their control derivatives, fast path first."""
    if _has_hermitian_igenerators(cp):
        propagators, eigenvalues, eigenvectors = _propagators_via_eigh(
            generators, cp.pulse_dt
        )
        d_propagators = _derivative_propagators_via_eigh(
            eigenvalues, eigenvectors, control_directions, cp.pulse_dt
        )
        return propagators, d_propagators

    propagators = np.asarray(expm(generators * cp.pulse_dt), dtype=np.complex128)
    d_propagators = _batched_derivative_propagators(
        generators, control_directions, cp.pulse_dt
    )
    return propagators, d_propagators


def _batched_derivative_propagators(
    generators: Array,
    control_directions: Array,
    dt: float,
) -> Array:
    """Return slice-propagator control derivatives as ``(n_steps, n_channels, dim, dim)``.

    Each derivative is the upper-right block of the 2x2 block auxiliary matrix
    exponential, evaluated for every (slice, channel) pair in one stacked call.
    """
    n_steps, dim = generators.shape[0], generators.shape[1]
    n_channels = control_directions.shape[0]
    A = np.complex128(dt) * generators
    dA = np.complex128(dt) * control_directions
    blocks = np.zeros((n_steps, n_channels, 2 * dim, 2 * dim), dtype=np.complex128)
    blocks[:, :, :dim, :dim] = A[:, None, :, :]
    blocks[:, :, dim:, dim:] = A[:, None, :, :]
    blocks[:, :, :dim, dim:] = dA[None, :, :, :]
    block_expm = np.asarray(
        expm(blocks.reshape(n_steps * n_channels, 2 * dim, 2 * dim)),
        dtype=np.complex128,
    )
    return np.asarray(
        block_expm[:, :dim, dim:].reshape(n_steps, n_channels, dim, dim),
        dtype=np.complex128,
    )


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


def _effective_waveform(cp: ControlProblem, wfm: RealArray) -> RealArray:
    """Validate a problem-waveform pair and return the freeze-masked waveform."""
    validate_control_problem(cp)
    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be two-dimensional, got shape {waveform.shape}")
    validate_waveform(waveform, len(cp.operators), waveform.shape[0])
    return apply_freeze(waveform, cp.freeze)


def forward_propagators(cp: ControlProblem, wfm: RealArray) -> list[Array]:
    """Return per-slice propagators for a waveform under a control problem."""
    effective_waveform = _effective_waveform(cp, wfm)

    generators = _slice_generator_stack(cp, effective_waveform)
    if _has_hermitian_igenerators(cp):
        propagators_stack, _, _ = _propagators_via_eigh(generators, cp.pulse_dt)
    else:
        propagators_stack = np.asarray(expm(generators * cp.pulse_dt), dtype=np.complex128)
    return [np.asarray(propagator, dtype=np.complex128) for propagator in propagators_stack]


def _validate_square_propagator(propagator: Array) -> None:
    """Raise ValueError if a propagator is not square."""
    if propagator.ndim != 2 or propagator.shape[0] != propagator.shape[1]:
        raise ValueError(f"propagator must be square, got shape {propagator.shape}")


def _validate_matches_propagator(name: str, matrix: Array, propagator: Array) -> None:
    """Raise ValueError if a derivative matrix does not match the propagator shape."""
    if matrix.shape != propagator.shape:
        raise ValueError(
            f"{name} shape {matrix.shape} must match "
            f"propagator shape {propagator.shape}"
        )


def _propagator_layout(state: Array, propagator: Array) -> str:
    """Return how a propagator acts on a state: vector, hilbert, or liouville.

    A 1-D state takes a matching matrix-vector product. A square-matrix state
    is either sandwiched by a same-dimension Hilbert propagator or acted on in
    vectorised form by a Liouville propagator of dimension ``state.size``.
    """
    if state.ndim == 1:
        if propagator.shape != (state.shape[0], state.shape[0]):
            raise ValueError(
                f"propagator shape {propagator.shape} cannot act on "
                f"state shape {state.shape}"
            )
        return "vector"

    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        if propagator.shape == state.shape:
            return "hilbert"
        if propagator.shape == (state.size, state.size):
            return "liouville"
        raise ValueError(
            f"propagator shape {propagator.shape} cannot act on "
            f"state shape {state.shape}"
        )

    raise ValueError(f"state must be a vector or square matrix, got shape {state.shape}")


def _apply_propagator(state: Array, propagator: Array) -> Array:
    """Apply a Hilbert-space or Liouville-space propagator to one state."""
    state_arr = np.asarray(state, dtype=np.complex128)
    propagator_arr = np.asarray(propagator, dtype=np.complex128)
    _validate_square_propagator(propagator_arr)

    layout = _propagator_layout(state_arr, propagator_arr)
    if layout == "vector":
        return propagator_arr @ state_arr
    if layout == "hilbert":
        return propagator_arr @ state_arr @ propagator_arr.conj().T
    return unvec(propagator_arr @ vec(state_arr), state_arr.shape[0])


def _apply_derivative_propagator(state: Array, propagator: Array, d_propagator: Array) -> Array:
    """Apply one slice-propagator derivative to a state."""
    state_arr = np.asarray(state, dtype=np.complex128)
    propagator_arr = np.asarray(propagator, dtype=np.complex128)
    d_propagator_arr = np.asarray(d_propagator, dtype=np.complex128)
    _validate_square_propagator(propagator_arr)
    _validate_matches_propagator("d_propagator", d_propagator_arr, propagator_arr)

    layout = _propagator_layout(state_arr, propagator_arr)
    if layout == "vector":
        return d_propagator_arr @ state_arr
    if layout == "hilbert":
        return (
            d_propagator_arr @ state_arr @ propagator_arr.conj().T
            + propagator_arr @ state_arr @ d_propagator_arr.conj().T
        )
    return unvec(d_propagator_arr @ vec(state_arr), state_arr.shape[0])


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
    _validate_matches_propagator("d_propagator_a", d_a_arr, propagator_arr)
    _validate_matches_propagator("d_propagator_b", d_b_arr, propagator_arr)
    _validate_matches_propagator("dd_propagator", dd_arr, propagator_arr)

    layout = _propagator_layout(state_arr, propagator_arr)
    if layout == "vector":
        return dd_arr @ state_arr
    if layout == "hilbert":
        return np.asarray(
            dd_arr @ state_arr @ propagator_arr.conj().T
            + d_a_arr @ state_arr @ d_b_arr.conj().T
            + d_b_arr @ state_arr @ d_a_arr.conj().T
            + propagator_arr @ state_arr @ dd_arr.conj().T,
            dtype=np.complex128,
        )
    return unvec(dd_arr @ vec(state_arr), state_arr.shape[0])


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
    return _mode_value(_overlap(rho_f, rho_t), mode)


def final_fidelity(fwd_states: list[Array], bwd_states: list[Array], mode: str) -> float:
    """Return fidelity between the final forward state and initial backward target."""
    if len(fwd_states) == 0:
        raise ValueError("fwd_states must be non-empty")
    if len(bwd_states) == 0:
        raise ValueError("bwd_states must be non-empty")
    return _fidelity_by_mode(fwd_states[-1], bwd_states[0], mode)


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
    return _mode_value(dd_overlap, mode)


def _mode_value(overlap: np.complex128, mode: str) -> float:
    """Return the scalar fidelity for one target-final overlap."""
    if mode == "real":
        return float(np.real(overlap))
    if mode == "imag":
        return float(np.imag(overlap))
    if mode == "abs2":
        return float(np.real(overlap.conjugate() * overlap))
    _raise_invalid_mode("mode")


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
        return np.asarray(
            2.0 * np.real(np.conjugate(overlap) * d_overlaps), dtype=np.float64
        )
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
        d_slice = np.einsum(
            "cij,jl,ml->cim", d_propagator, rho, propagator.conj(), optimize=True
        ) + np.einsum(
            "ij,jl,cml->cim", propagator, rho, d_propagator.conj(), optimize=True
        )
        d_overlaps[step_index] = np.einsum(
            "im,cim->c", bwd[step_index + 1].conj(), d_slice, optimize=True
        )

    overlap = np.complex128(np.vdot(rho_targ, fwd[-1]))
    return _mode_value(overlap, mode), _mode_gradient(overlap, d_overlaps, mode)


def _single_value_and_gradient(cp: ControlProblem, wfm: RealArray) -> tuple[float, RealArray]:
    """Return bare fidelity and gradient for a single-member control problem.

    Penalties are not applied here; callers handle penalty terms so the same
    core serves both ``grape_gradient`` and ``grape_xy_and_gradient``.
    """
    effective_waveform = _effective_waveform(cp, wfm)
    n_steps, n_channels = effective_waveform.shape

    from optimalcontrol._accelerator import vector_value_gradient

    accelerated = vector_value_gradient([cp], effective_waveform)
    if accelerated is not None:
        fidelity, accelerated_gradient = accelerated
        _zero_frozen(accelerated_gradient, cp.freeze)
        return fidelity, accelerated_gradient

    generators = _slice_generator_stack(cp, effective_waveform)
    dim = generators.shape[1]
    control_directions = _control_direction_stack(cp)
    propagators, d_propagators = _propagators_and_derivatives(
        cp, generators, control_directions
    )

    gradient: RealArray = np.zeros((n_steps, n_channels), dtype=np.float64)
    values: list[float] = []
    for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ):
        init = np.asarray(rho_init, dtype=np.complex128)
        targ = np.asarray(rho_targ, dtype=np.complex128)
        if init.ndim == 1:
            value, member_gradient = _vector_value_and_gradient(
                init, targ, propagators, d_propagators, cp.fidelity_mode
            )
        elif init.ndim == 2 and init.shape[0] == init.shape[1] and init.size == dim:
            value, member_gradient = _vector_value_and_gradient(
                vec(init), vec(targ), propagators, d_propagators, cp.fidelity_mode
            )
        elif init.ndim == 2 and init.shape[0] == init.shape[1] and init.shape[0] == dim:
            value, member_gradient = _hilbert_matrix_value_and_gradient(
                init, targ, propagators, d_propagators, cp.fidelity_mode
            )
        else:
            raise ValueError(
                f"state shape {init.shape} is incompatible with generator dimension {dim}"
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


def _hessian_slice_propagators(
    cp: ControlProblem, effective_waveform: RealArray
) -> tuple[list[Array], list[list[Array]], list[list[list[Array]]]]:
    """Return per-slice propagators with their first and second derivatives."""
    n_channels = effective_waveform.shape[1]
    propagators: list[Array] = []
    derivative_propagators: list[list[Array]] = []
    second_derivative_propagators: list[list[list[Array]]] = []
    control_directions = list(_control_direction_stack(cp))
    generators = _slice_generator_stack(cp, effective_waveform)
    for generator in generators:
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
    return propagators, derivative_propagators, second_derivative_propagators


def _hessian_state_sweep(
    rho_init: Array,
    propagators: list[Array],
    derivative_propagators: list[list[Array]],
    second_derivative_propagators: list[list[list[Array]]],
    n_channels: int,
    n_params: int,
) -> tuple[Array, list[Array], list[list[Array]]]:
    """Propagate one state with all first and second parameter variations."""
    current = np.asarray(rho_init, dtype=np.complex128).copy()
    zero_state = np.zeros_like(current, dtype=np.complex128)
    first_states = [zero_state.copy() for _ in range(n_params)]
    second_states = [
        [zero_state.copy() for _ in range(n_params)] for _ in range(n_params)
    ]

    for step_index in range(len(propagators)):
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
                cross_term = _apply_derivative_propagator(
                    first_states[other_index],
                    propagator,
                    d_prop,
                )
                next_second_states[param_index][other_index] += cross_term
                next_second_states[other_index][param_index] += cross_term
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

    return current, first_states, second_states


def grape_hessian(cp: ControlProblem, wfm: RealArray) -> RealArray | None:
    """Return the GRAPE objective Hessian for small waveforms, or None if too large.

    The Hessian differentiates the same objective as ``grape_gradient``: when
    ``cp.penalties`` is set, the penalty Hessian is subtracted from the exact
    fidelity Hessian.
    """
    effective_waveform = _effective_waveform(cp, wfm)
    n_steps, n_channels = effective_waveform.shape
    n_params = n_steps * n_channels
    if n_params > 50:
        return None

    propagators, derivative_propagators, second_derivative_propagators = (
        _hessian_slice_propagators(cp, effective_waveform)
    )

    hessian: RealArray = np.zeros((n_params, n_params), dtype=np.float64)
    for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ):
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
    if cp.freeze is not None:
        freeze_mask = np.asarray(cp.freeze, dtype=np.bool_).reshape(n_params)
        hessian[freeze_mask, :] = 0.0
        hessian[:, freeze_mask] = 0.0
    return hessian


def _grape_xy_core(cp: ControlProblem, wfm: RealArray) -> float:
    """Return scalar GRAPE fidelity without basis-specific state conversion."""
    from optimalcontrol._accelerator import vector_fidelity

    accelerated = vector_fidelity([cp], wfm)
    if accelerated is not None:
        return accelerated

    propagators = forward_propagators(cp, wfm)
    values: list[float] = []
    for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ):
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


def _grape_xy_in_basis(
    cp: ControlProblem,
    wfm: RealArray,
    convert_fn: Callable[[Array, int, str], Array],
    basis: str,
) -> float:
    """Return GRAPE fidelity after converting states with ``convert_fn``."""
    validate_control_problem(cp)
    generator_dim = int(np.asarray(cp.drifts[0], dtype=np.complex128).shape[0])
    rho_init = [
        convert_fn(state, generator_dim, f"rho_init[{index}]")
        for index, state in enumerate(cp.rho_init)
    ]
    rho_targ = [
        convert_fn(state, generator_dim, f"rho_targ[{index}]")
        for index, state in enumerate(cp.rho_targ)
    ]
    converted_cp = replace(cp, rho_init=rho_init, rho_targ=rho_targ, basis=basis)
    return _grape_xy_core(converted_cp, wfm)


def grape_xy_liouville(cp: ControlProblem, wfm: RealArray) -> float:
    """Return GRAPE fidelity using vectorised-density Liouville propagation."""
    return _grape_xy_in_basis(cp, wfm, _vectorise_liouville_state, "liouville")


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
    return _grape_xy_in_basis(cp, wfm, _validate_hilbert_state, "hilbert")


def grape_xy(cp: ControlProblem, wfm: RealArray) -> float:
    """Return the scalar GRAPE fidelity for a Spinach-style XY control problem.

    When ``cp`` has multiple drift generators or power levels, the fidelity is
    averaged over the Cartesian ensemble. When ``cp.penalties`` is set, the
    summed penalty value is subtracted from the fidelity before returning.
    """
    if _has_ensemble_axes(cp):
        from optimalcontrol.ensemble import ensemble_fidelity

        bare_cp = replace(cp, penalties=None)
        fidelity = ensemble_fidelity(bare_cp, wfm)
    else:
        basis = cp.basis.lower()
        if basis == "liouville":
            fidelity = grape_xy_liouville(cp, wfm)
        elif basis == "hilbert":
            fidelity = grape_xy_hilbert(cp, wfm)
        else:
            fidelity = _grape_xy_core(cp, wfm)
    if cp.penalties is not None:
        penalty_value, _ = total_penalty(np.asarray(wfm, dtype=np.float64), cp.penalties)
        return fidelity - penalty_value
    return fidelity
