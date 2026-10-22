"""Waveform handling and state propagation helpers."""

import math

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from optimalcontrol._problem import (
    ControlProblem,
    _validate_matches_propagator,
    _validate_square_propagator,
    validate_control_problem,
    validate_waveform,
)
from optimalcontrol._types import Array, BoolArray, RealArray
from optimalcontrol._validation import validate_square_matrix as _validate_square_matrix
from optimalcontrol.operators import unvec, vec


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


def _single_drift(cp: ControlProblem) -> Array:
    """Return the only drift currently supported by the non-ensemble path."""
    if len(cp.drifts) != 1:
        raise ValueError("forward propagation currently supports exactly one drift")
    return np.asarray(cp.drifts[0], dtype=np.complex128)


def _control_direction_stack(cp: ControlProblem) -> Array:
    """Return power-scaled control operators stacked as ``(n_channels, dim, dim)``."""
    operators = np.stack([np.asarray(operator, dtype=np.complex128) for operator in cp.operators])
    levels = np.asarray(cp.pwr_levels, dtype=np.complex128)
    return np.asarray(levels[:, None, None] * operators, dtype=np.complex128)


def _slice_generator_stack(cp: ControlProblem, effective_waveform: RealArray) -> Array:
    """Return all slice generators stacked as ``(n_steps, dim, dim)``."""
    drift = _single_drift(cp)
    control_directions = _control_direction_stack(cp)
    scaled = np.asarray(effective_waveform, dtype=np.complex128)
    generators = drift[None, :, :] + np.tensordot(scaled, control_directions, axes=(1, 0))
    return np.asarray(generators, dtype=np.complex128)


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
            f"d_generator_a shape {d_a_arr.shape} must match generator shape {generator_arr.shape}"
        )
    if d_b_arr.shape != generator_arr.shape:
        raise ValueError(
            f"d_generator_b shape {d_b_arr.shape} must match generator shape {generator_arr.shape}"
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


def _propagator_layout(state: Array, propagator: Array) -> str:
    """Return how a propagator acts on a state: vector, hilbert, or liouville.

    A 1-D state takes a matching matrix-vector product. A square-matrix state
    is either sandwiched by a same-dimension Hilbert propagator or acted on in
    vectorised form by a Liouville propagator of dimension ``state.size``.
    """
    if state.ndim == 1:
        if propagator.shape != (state.shape[0], state.shape[0]):
            raise ValueError(
                f"propagator shape {propagator.shape} cannot act on state shape {state.shape}"
            )
        return "vector"

    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        if propagator.shape == state.shape:
            return "hilbert"
        if propagator.shape == (state.size, state.size):
            return "liouville"
        raise ValueError(
            f"propagator shape {propagator.shape} cannot act on state shape {state.shape}"
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
