"""Spin-1/2 single-spin operators and related algebraic utilities."""

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
