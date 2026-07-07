"""Spin-1/2 single-spin operators, tensor-product helpers, and Liouville-space utilities."""

from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix


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
        result = np.kron(result, op).astype(np.complex128)
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


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def assert_square(A: npt.NDArray[np.complex128]) -> None:
    """Raise ValueError if A is not a 2-D square matrix."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {A.shape}")


def assert_hermitian(A: npt.NDArray[np.complex128], tol: float = 1e-10) -> None:
    """Raise ValueError if A is not Hermitian within absolute tolerance tol."""
    assert_square(A)
    if not np.allclose(A, A.conj().T, atol=tol, rtol=0.0):
        raise ValueError("Matrix is not Hermitian within tolerance")


# ---------------------------------------------------------------------------
# Superoperators (column-major / Fortran-order vec convention)
# ---------------------------------------------------------------------------
# For column-major vec: vec(A @ X @ B) = (B^T ⊗ A) @ vec(X)
# Hence:
#   left-multiply by A:  L_op(A) = I ⊗ A  (np.kron(I, A))
#   right-multiply by A: R_op(A) = A^T ⊗ I (np.kron(A.T, I))
# Note: some references use row-major convention and write A ⊗ I / I ⊗ A^T;
# those forms are inconsistent with the column-major vec() used here.


def L_op(A: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Left-multiplication superoperator in column-major vec convention.

    L_op(A) @ vec(rho) == vec(A @ rho).  L_op(A) = I_n ⊗ A.
    """
    n = A.shape[0]
    return np.kron(np.eye(n, dtype=np.complex128), A).astype(np.complex128)


def R_op(A: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Right-multiplication superoperator in column-major vec convention.

    R_op(A) @ vec(rho) == vec(rho @ A).  R_op(A) = A^T ⊗ I_n.
    """
    n = A.shape[0]
    return np.kron(A.T, np.eye(n, dtype=np.complex128)).astype(np.complex128)


def liouvillian_comm(A: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Commutator superoperator: -i[A, ·].

    liouvillian_comm(A) @ vec(rho) == -1j * vec(A @ rho - rho @ A).
    """
    return -1j * (L_op(A) - R_op(A))


def double_comm(
    F: npt.NDArray[np.complex128], rho_vec: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Compute the action of [F, [F, ·]] on a vectorised density matrix.

    Returns vec([F, [F, rho]]).
    """
    dim = int(round(rho_vec.size**0.5))
    rho = unvec(rho_vec.astype(np.complex128), dim)
    return vec(comm(F, comm(F, rho)))


def lindblad_dissipator(
    Fk_list: list[npt.NDArray[np.complex128]],
    a_kl: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Build the full Lindblad dissipator superoperator.

    Computes D = sum_{k,l} a_{kl} (F_k ρ F_l† - 0.5*(F_l†F_k ρ + ρ F_l†F_k))
    as an (n²×n²) superoperator acting on column-major vec(ρ).

    Args:
        Fk_list: list of n×n jump operators.
        a_kl: (len(Fk_list) × len(Fk_list)) coefficient matrix; must be
              positive-semidefinite Hermitian for a valid Lindblad equation.
    """
    n = Fk_list[0].shape[0]
    dim2 = n * n
    D = np.zeros((dim2, dim2), dtype=np.complex128)
    for k, Fk in enumerate(Fk_list):
        for l_idx, Fl in enumerate(Fk_list):
            coeff = a_kl[k, l_idx]
            if coeff == 0.0:
                continue
            FlFk = Fl.conj().T @ Fk
            D += coeff * (
                L_op(Fk) @ R_op(Fl.conj().T)
                - 0.5 * L_op(FlFk)
                - 0.5 * R_op(FlFk)
            )
    return D


# ---------------------------------------------------------------------------
# Dense / sparse dispatch
# ---------------------------------------------------------------------------


def dense_or_sparse(
    A: npt.NDArray[np.complex128], sparse: bool = False
) -> Union[npt.NDArray[np.complex128], csr_matrix]:
    """Return A as a dense ndarray or a scipy CSR sparse matrix.

    Args:
        A: input matrix.
        sparse: if True, return scipy.sparse.csr_matrix; otherwise return A unchanged.
    """
    if sparse:
        return csr_matrix(A)
    return A
