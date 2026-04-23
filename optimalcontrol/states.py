"""Product-operator state construction and normalisation helpers."""

import numpy as np
import numpy.typing as npt

from optimalcontrol.operators import E, Ix, Iy, Iz, place_operator


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
