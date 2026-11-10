"""Newton and rational-function-optimisation step helpers."""

import numpy as np

from optimalcontrol._line_search import _as_real_array
from optimalcontrol._types import RealArray


def _as_symmetric_matrix(name: str, value: RealArray) -> RealArray:
    """Return a finite square float64 matrix symmetrised across the diagonal."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} entries must be finite")
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def _solve_symmetric_system(matrix: RealArray, rhs: RealArray) -> RealArray:
    """Solve a symmetric linear system, falling back to least squares if singular."""
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return np.asarray(solution, dtype=np.float64)


def _positive_definite_shift(curvature: RealArray) -> float:
    """Return a diagonal shift that makes ``curvature`` positive definite."""
    eigenvalues = np.linalg.eigvalsh(curvature)
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue > 0.0:
        return 0.0
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    epsilon = 1e-10 * scale
    return -min_eigenvalue + epsilon


def _rfo_step(curvature: RealArray, gradient: RealArray) -> RealArray:
    """Return a rational-function-optimisation step for minimising ``-f``."""
    n_params = gradient.size
    augmented = np.zeros((n_params + 1, n_params + 1), dtype=np.float64)
    augmented[:n_params, :n_params] = curvature
    augmented[:n_params, n_params] = -gradient
    augmented[n_params, :n_params] = -gradient

    eigenvalues, _ = np.linalg.eigh(augmented)
    level_shift = float(np.min(eigenvalues))
    shifted = np.asarray(
        curvature - level_shift * np.eye(n_params, dtype=np.float64),
        dtype=np.float64,
    )
    return _solve_symmetric_system(shifted, gradient)


def _newton_step(
    gradient: RealArray,
    hessian: RealArray,
    *,
    regularise: bool,
    rfo: bool,
) -> RealArray:
    """Return a Newton or RFO step for maximising the fidelity objective."""
    gradient_arr = _as_real_array("gradient", gradient)
    hessian_arr = _as_symmetric_matrix("hessian", hessian)
    gradient_flat = gradient_arr.reshape(-1)
    if hessian_arr.shape != (gradient_flat.size, gradient_flat.size):
        raise ValueError(
            "hessian shape "
            f"{hessian_arr.shape} must match flattened gradient size {gradient_flat.size}"
        )

    # Solve the Newton system for minimising ``-f`` so a local maximum of ``f``
    # corresponds to a positive-definite curvature model.
    curvature = np.asarray(-hessian_arr, dtype=np.float64)
    if regularise:
        shift = _positive_definite_shift(curvature)
        if shift > 0.0:
            curvature = np.asarray(
                curvature + shift * np.eye(curvature.shape[0], dtype=np.float64),
                dtype=np.float64,
            )

    step_flat = (
        _rfo_step(curvature, gradient_flat)
        if rfo
        else _solve_symmetric_system(curvature, gradient_flat)
    )
    step = np.asarray(step_flat.reshape(gradient_arr.shape), dtype=np.float64)
    if not np.all(np.isfinite(step)):
        raise ValueError("Newton step must be finite")
    return step
