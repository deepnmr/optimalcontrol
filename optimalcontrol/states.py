"""Product-operator state construction and normalisation helpers."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from optimalcontrol.operators import E, Ix, Iy, Iz, place_operator, unvec, vec


def _validate_n_spins(n_spins: int) -> None:
    """Raise ValueError if n_spins cannot represent a spin-1/2 system."""
    if n_spins < 1:
        raise ValueError("n_spins must be at least 1")


def _single_spin_op(label: str) -> npt.NDArray[np.complex128]:
    """Return the elementary spin operator for axis labels x, y, or z."""
    if label == "x":
        return Ix()
    if label == "y":
        return Iy()
    if label == "z":
        return Iz()
    raise ValueError(f"Unknown spin axis {label!r}")


def _operator_on_spin(
    op: npt.NDArray[np.complex128], spin_index: int, n_spins: int
) -> npt.NDArray[np.complex128]:
    """Place op on a validated spin index."""
    _validate_n_spins(n_spins)
    if spin_index >= n_spins:
        raise ValueError(f"label requires spin {spin_index}, but n_spins={n_spins}")
    return place_operator(op, spin_index, n_spins)


def state_from_label(label: str, n_spins: int) -> npt.NDArray[np.complex128]:
    """Return an unnormalised product-operator state from a compact label.

    Supported one-spin labels are ``Ix``, ``Iy``, ``Iz`` for spin 0 and
    ``Sx``, ``Sy``, ``Sz`` for spin 1. Supported two-spin labels use the
    first two spins and include their conventional leading factor of 2, e.g.
    ``2IxSz`` returns ``2 * Ix_0 @ Sz_1``.
    """
    single_spin_labels = {
        "Ix": (Ix(), 0),
        "Iy": (Iy(), 0),
        "Iz": (Iz(), 0),
        "Sx": (Ix(), 1),
        "Sy": (Iy(), 1),
        "Sz": (Iz(), 1),
    }
    if label in single_spin_labels:
        op, spin_index = single_spin_labels[label]
        return _operator_on_spin(op, spin_index, n_spins)

    product_labels = {
        "2IxSz": ("x", "z"),
        "2IySz": ("y", "z"),
        "2IzSx": ("z", "x"),
        "2IzSy": ("z", "y"),
        "2IzSz": ("z", "z"),
    }
    if label in product_labels:
        i_axis, s_axis = product_labels[label]
        i_op = _operator_on_spin(_single_spin_op(i_axis), 0, n_spins)
        s_op = _operator_on_spin(_single_spin_op(s_axis), 1, n_spins)
        return np.complex128(2.0) * (i_op @ s_op)

    supported = ", ".join(sorted([*single_spin_labels, *product_labels]))
    raise ValueError(f"Unknown state label {label!r}; expected one of: {supported}")


def _s_alpha() -> npt.NDArray[np.complex128]:
    """Projector onto the S-spin alpha state."""
    return np.complex128(0.5) * E() + Iz()


def _s_beta() -> npt.NDArray[np.complex128]:
    """Projector onto the S-spin beta state."""
    return np.complex128(0.5) * E() - Iz()


def single_transition_operator(label: str) -> npt.NDArray[np.complex128]:
    """Return an unnormalised two-spin single-transition operator.

    ``Salpha`` and ``Sbeta`` denote projectors on the second spin:
    ``Salpha = E/2 + Sz`` and ``Sbeta = E/2 - Sz``.
    """
    labels = {
        "IzSalpha": (Iz(), _s_alpha()),
        "IzSbeta": (Iz(), _s_beta()),
        "IxSbeta": (Ix(), _s_beta()),
        "IySbeta": (Iy(), _s_beta()),
    }
    if label not in labels:
        supported = ", ".join(sorted(labels))
        raise ValueError(f"Unknown single-transition label {label!r}; expected one of: {supported}")
    i_op, s_projector = labels[label]
    return np.kron(i_op, s_projector).astype(np.complex128)


def normalise_hs(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Normalise a matrix by its Hilbert-Schmidt norm sqrt(Tr(v.conj().T @ v))."""
    arr = np.asarray(v, dtype=np.complex128)
    norm = float(np.sqrt(np.real(np.trace(arr.conj().T @ arr))))
    if norm == 0.0:
        raise ValueError("Cannot normalise a zero-norm array")
    return arr / norm


def normalise_2norm(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Normalise an array by its Euclidean 2-norm."""
    arr = np.asarray(v, dtype=np.complex128)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("Cannot normalise a zero-norm array")
    return arr / norm


def _validate_square_array(name: str, arr: npt.NDArray[np.complex128]) -> None:
    """Raise ValueError if arr is not a square 2-D array."""
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {arr.shape}")


def _liouville_size_for_state(rho: npt.NDArray[np.complex128]) -> int:
    """Return the expected Liouville-space dimension for rho."""
    if rho.ndim == 1:
        return rho.shape[0]
    if rho.ndim == 2 and rho.shape[0] == rho.shape[1]:
        return rho.shape[0] * rho.shape[1]
    raise ValueError(f"rho must be a vectorised state or square matrix, got shape {rho.shape}")


def _validate_liouville_shape(
    propagator: npt.NDArray[np.complex128],
    rho: npt.NDArray[np.complex128],
    name: str,
) -> None:
    """Raise ValueError if propagator cannot act on rho in Liouville space."""
    expected = _liouville_size_for_state(rho)
    if propagator.shape != (expected, expected):
        raise ValueError(
            f"{name} has shape {propagator.shape}, expected {(expected, expected)} "
            f"for state shape {rho.shape}"
        )


def _apply_liouville_propagator(
    rho: npt.NDArray[np.complex128],
    propagator: npt.NDArray[np.complex128],
    name: str,
) -> npt.NDArray[np.complex128]:
    """Apply a Liouville-space propagator to matrix or vectorised rho."""
    _validate_square_array(name, propagator)
    _validate_liouville_shape(propagator, rho, name)
    if rho.ndim == 1:
        return propagator @ rho
    return unvec(propagator @ vec(rho), rho.shape[0])


def _apply_state_propagator(
    rho: npt.NDArray[np.complex128],
    propagator: npt.NDArray[np.complex128],
    name: str,
) -> npt.NDArray[np.complex128]:
    """Apply either a Hilbert-space rotation or Liouville-space propagator."""
    rho_arr = np.asarray(rho, dtype=np.complex128)
    propagator_arr = np.asarray(propagator, dtype=np.complex128)
    _validate_square_array(name, propagator_arr)

    if rho_arr.ndim == 2 and rho_arr.shape[0] == rho_arr.shape[1]:
        if propagator_arr.shape == rho_arr.shape:
            return propagator_arr @ rho_arr @ propagator_arr.conj().T
        return _apply_liouville_propagator(rho_arr, propagator_arr, name)

    if rho_arr.ndim == 1:
        return _apply_liouville_propagator(rho_arr, propagator_arr, name)

    raise ValueError(f"rho must be a vectorised state or square matrix, got shape {rho_arr.shape}")


def apply_prefix(
    rho: npt.NDArray[np.complex128],
    prefix_propagator: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Apply a fixed prefix transformation to an initial state.

    Matrix states accept either a Hilbert-space propagator ``U`` with
    ``U @ rho @ U.conj().T`` semantics, or a Liouville-space propagator acting
    on ``vec(rho)``. Vectorised states require a Liouville-space propagator.
    """
    return _apply_state_propagator(rho, prefix_propagator, "prefix_propagator")


def apply_suffix(
    rho: npt.NDArray[np.complex128],
    suffix_propagator: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Apply a fixed suffix transformation before fidelity evaluation.

    Matrix states accept either a Hilbert-space propagator ``U`` with
    ``U @ rho @ U.conj().T`` semantics, or a Liouville-space propagator acting
    on ``vec(rho)``. Vectorised states require a Liouville-space propagator.
    """
    return _apply_state_propagator(rho, suffix_propagator, "suffix_propagator")


def dead_time_propagation(
    rho: npt.NDArray[np.complex128],
    generator: npt.NDArray[np.complex128],
    t_dead: float,
) -> npt.NDArray[np.complex128]:
    """Propagate a state under a Liouville-space generator for dead time.

    The generator is exponentiated as ``expm(generator * t_dead)``. Matrix
    states are column-stacked, propagated, and reshaped back using the package's
    column-major Liouville convention; vectorised states remain vectorised.
    """
    rho_arr = np.asarray(rho, dtype=np.complex128)
    generator_arr = np.asarray(generator, dtype=np.complex128)
    _validate_square_array("generator", generator_arr)
    _validate_liouville_shape(generator_arr, rho_arr, "generator")
    if t_dead == 0.0:
        return rho_arr.copy()

    propagator = cast(npt.NDArray[np.complex128], expm(generator_arr * t_dead))
    return _apply_liouville_propagator(rho_arr, propagator, "dead-time propagator")


def _overlap(
    rho_f: npt.NDArray[np.complex128], rho_t: npt.NDArray[np.complex128]
) -> np.complex128:
    """Return the Hilbert-Schmidt overlap <rho_t, rho_f>."""
    rho_f_arr = np.asarray(rho_f, dtype=np.complex128)
    rho_t_arr = np.asarray(rho_t, dtype=np.complex128)
    if rho_f_arr.shape != rho_t_arr.shape:
        raise ValueError(f"State shapes must match, got {rho_f_arr.shape} and {rho_t_arr.shape}")
    return np.complex128(np.vdot(rho_t_arr, rho_f_arr))


def fidelity_real(
    rho_f: npt.NDArray[np.complex128], rho_t: npt.NDArray[np.complex128]
) -> float:
    """Return Re(Tr(rho_t.conj().T @ rho_f)) for pre-normalised states."""
    return float(np.real(_overlap(rho_f, rho_t)))


def fidelity_imag(
    rho_f: npt.NDArray[np.complex128], rho_t: npt.NDArray[np.complex128]
) -> float:
    """Return Im(Tr(rho_t.conj().T @ rho_f)) for pre-normalised states."""
    return float(np.imag(_overlap(rho_f, rho_t)))


def fidelity_abs2(
    rho_f: npt.NDArray[np.complex128], rho_t: npt.NDArray[np.complex128]
) -> float:
    """Return |Tr(rho_t.conj().T @ rho_f)|^2 for pre-normalised states."""
    overlap = _overlap(rho_f, rho_t)
    return float(np.real(overlap.conjugate() * overlap))


def fidelity_avg(
    rho_f_list: Sequence[npt.NDArray[np.complex128]],
    rho_t_list: Sequence[npt.NDArray[np.complex128]],
    weights: Sequence[float] | None = None,
) -> float:
    """Return the weighted average real fidelity over source-target state pairs."""
    if len(rho_f_list) != len(rho_t_list):
        raise ValueError("rho_f_list and rho_t_list must have the same length")
    if len(rho_f_list) == 0:
        raise ValueError("At least one source-target pair is required")

    values = np.array(
        [fidelity_real(rho_f, rho_t) for rho_f, rho_t in zip(rho_f_list, rho_t_list)],
        dtype=np.float64,
    )
    if weights is None:
        return float(np.mean(values))

    weight_arr = np.asarray(weights, dtype=np.float64)
    if weight_arr.shape != values.shape:
        raise ValueError(f"weights must have shape {values.shape}, got {weight_arr.shape}")
    if float(np.sum(weight_arr)) == 0.0:
        raise ValueError("weights must not sum to zero")
    return float(np.average(values, weights=weight_arr))
