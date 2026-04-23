"""Spin-1/2 single-spin operators, tensor-product helpers, and Liouville-space utilities."""

import numpy as np
import numpy.typing as npt


def Ix() -> npt.NDArray[np.complex128]:
    """Spin-1/2 x-component operator. Returns 2x2 array."""
    return np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)


def Iy() -> npt.NDArray[np.complex128]:
    """Spin-1/2 y-component operator. Returns 2x2 array."""
    return np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128)


def Iz() -> npt.NDArray[np.complex128]:
    """Spin-1/2 z-component operator. Returns 2x2 array."""
    return np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)


def Ip() -> npt.NDArray[np.complex128]:
    """Raising operator I+ = Ix + i*Iy. Returns 2x2 array."""
    return np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)


def Im() -> npt.NDArray[np.complex128]:
    """Lowering operator I- = Ix - i*Iy. Returns 2x2 array."""
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)


def E() -> npt.NDArray[np.complex128]:
    """2x2 identity operator. Returns 2x2 array."""
    return np.eye(2, dtype=np.complex128)


def kron_product(ops: list[npt.NDArray[np.complex128]]) -> npt.NDArray[np.complex128]:
    """Kronecker product of a list of matrices, evaluated left-to-right."""
    if not ops:
        raise ValueError("ops must be non-empty")
    result: npt.NDArray[np.complex128] = ops[0].astype(np.complex128)
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def place_operator(
    op: npt.NDArray[np.complex128], spin_index: int, n_spins: int
) -> npt.NDArray[np.complex128]:
    """Place op at spin_index in an n_spins Hilbert space, identity at all other sites."""
    ops: list[npt.NDArray[np.complex128]] = [
        op if i == spin_index else E() for i in range(n_spins)
    ]
    return kron_product(ops)


def two_spin_product(
    op_i: npt.NDArray[np.complex128],
    op_s: npt.NDArray[np.complex128],
    n_spins: int = 2,
) -> npt.NDArray[np.complex128]:
    """Return op_i ⊗ op_s embedded in an n_spins Hilbert space (spins 0 and 1)."""
    return place_operator(op_i, 0, n_spins) @ place_operator(op_s, 1, n_spins)


def comm(
    A: npt.NDArray[np.complex128], B: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Return the commutator [A, B] = A @ B - B @ A."""
    return A @ B - B @ A


def vec(rho: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Vectorise a density matrix by column-stacking (column-major / Fortran order).

    Convention: vec(rho)[i + dim*j] = rho[i, j], so the first column of rho
    occupies the first dim entries of the output vector. This matches the
    standard Liouville-space convention used throughout this package.
    """
    return rho.flatten(order="F")


def unvec(v: npt.NDArray[np.complex128], dim: int) -> npt.NDArray[np.complex128]:
    """Invert vec(): reshape a length-dim^2 vector back to a (dim, dim) matrix.

    Uses column-major (Fortran) order to match vec().
    """
    return v.reshape((dim, dim), order="F")
