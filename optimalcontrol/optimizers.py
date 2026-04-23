"""Shared optimizer result types and line-search utilities."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from optimalcontrol.grape import (
    ControlProblem,
    apply_freeze,
    grape_gradient,
    grape_hessian,
    grape_xy,
    validate_waveform,
)

RealArray = npt.NDArray[np.float64]
Objective = Callable[[RealArray], float]
Gradient = Callable[[RealArray], RealArray]


@dataclass
class OptimResult:
    """Common output contract for waveform optimizers."""

    wfm_final: RealArray
    fidelity_final: float
    n_iter: int
    n_feval: int
    converged: bool
    reason: str
    history: list[float]


class LBFGSState(TypedDict):
    """Limited-memory inverse-Hessian state for maximisation optimizers."""

    m: int
    s_history: list[RealArray]
    y_history: list[RealArray]
    rho_history: list[float]


@dataclass(frozen=True)
class _LinePoint:
    """One evaluated point on the scalar line-search objective."""

    alpha: float
    phi: float
    dphi: float


def _as_real_array(name: str, value: RealArray) -> RealArray:
    """Return a finite float64 array."""
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} entries must be finite")
    return array


def _directional_derivative(gradient: RealArray, direction: RealArray) -> float:
    """Return the real flattened directional derivative."""
    return float(np.sum(np.asarray(gradient, dtype=np.float64) * direction))


def _array_norm(value: RealArray) -> float:
    """Return the Euclidean norm of a real array."""
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64).reshape(-1)))


def _validate_optimizer_controls(tol_x: float, tol_g: float, max_iter: int) -> None:
    """Raise ValueError if optimizer convergence controls are invalid."""
    if not math.isfinite(tol_x) or tol_x < 0.0:
        raise ValueError("tol_x must be finite and non-negative")
    if not math.isfinite(tol_g) or tol_g < 0.0:
        raise ValueError("tol_g must be finite and non-negative")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")


def _initial_waveform(cp: ControlProblem, wfm0: RealArray) -> RealArray:
    """Return a validated mutable copy of the initial GRAPE waveform."""
    waveform = _as_real_array("wfm0", wfm0)
    if waveform.ndim != 2:
        raise ValueError(f"wfm0 must be two-dimensional, got shape {waveform.shape}")
    validate_waveform(waveform, len(cp.operators), waveform.shape[0])
    return waveform.copy()


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

    step_flat = _rfo_step(curvature, gradient_flat) if rfo else _solve_symmetric_system(
        curvature, gradient_flat
    )
    step = np.asarray(step_flat.reshape(gradient_arr.shape), dtype=np.float64)
    if not np.all(np.isfinite(step)):
        raise ValueError("Newton step must be finite")
    return step


def _cubic_interpolated_minimum(point_a: _LinePoint, point_b: _LinePoint) -> float | None:
    """Return the stationary point of the cubic Hermite interpolant, if usable."""
    if not all(
        math.isfinite(value)
        for value in (
            point_a.alpha,
            point_a.phi,
            point_a.dphi,
            point_b.alpha,
            point_b.phi,
            point_b.dphi,
        )
    ):
        return None

    step = point_b.alpha - point_a.alpha
    if step == 0.0:
        return None

    delta_value = point_b.phi - point_a.phi - step * point_a.dphi
    delta_slope = step * (point_b.dphi - point_a.dphi)
    cubic_term = delta_slope - 2.0 * delta_value
    quadratic_term = 3.0 * delta_value - delta_slope
    linear_term = step * point_a.dphi

    candidates: list[float] = []
    if abs(cubic_term) < 1e-14:
        if abs(quadratic_term) >= 1e-14:
            candidates.append(-linear_term / (2.0 * quadratic_term))
    else:
        discriminant = 4.0 * quadratic_term * quadratic_term - 12.0 * cubic_term * linear_term
        if discriminant >= 0.0:
            sqrt_discriminant = math.sqrt(discriminant)
            candidates.append((-2.0 * quadratic_term + sqrt_discriminant) / (6.0 * cubic_term))
            candidates.append((-2.0 * quadratic_term - sqrt_discriminant) / (6.0 * cubic_term))

    best_alpha: float | None = None
    best_value = math.inf
    for candidate in candidates:
        if candidate <= 0.0 or candidate >= 1.0 or not math.isfinite(candidate):
            continue
        value = (
            point_a.phi
            + linear_term * candidate
            + quadratic_term * candidate * candidate
            + cubic_term * candidate * candidate * candidate
        )
        if value < best_value:
            best_value = value
            best_alpha = point_a.alpha + step * candidate
    return best_alpha


def _bracket_trial_alpha(point_a: _LinePoint, point_b: _LinePoint) -> float:
    """Return a safeguarded cubic trial point inside a bracket."""
    lower = min(point_a.alpha, point_b.alpha)
    upper = max(point_a.alpha, point_b.alpha)
    width = upper - lower
    if width <= 0.0:
        return point_a.alpha

    candidate = _cubic_interpolated_minimum(point_a, point_b)
    margin = 0.1 * width
    if (
        candidate is None
        or not math.isfinite(candidate)
        or candidate <= lower + margin
        or candidate >= upper - margin
    ):
        return 0.5 * (point_a.alpha + point_b.alpha)
    return candidate


def line_search_cubic(
    f: Objective,
    grad: Gradient,
    wfm: RealArray,
    direction: RealArray,
    alpha0: float = 1.0,
    max_iter: int = 20,
) -> float:
    """Return a step length satisfying strong Wolfe conditions, or the best found.

    The public objective is treated as a maximisation problem, matching GRAPE
    fidelity. Internally the search minimises ``-f(wfm + alpha * direction)`` so
    the standard Wolfe inequalities can be used.
    """
    waveform = _as_real_array("wfm", wfm)
    search_direction = _as_real_array("direction", direction)
    if search_direction.shape != waveform.shape:
        raise ValueError(
            f"direction shape {search_direction.shape} must match wfm shape {waveform.shape}"
        )
    if not math.isfinite(alpha0) or alpha0 <= 0.0:
        raise ValueError("alpha0 must be finite and positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    c1 = 1e-4
    c2 = 0.9

    f0 = float(f(waveform))
    if not math.isfinite(f0):
        raise ValueError("initial objective value must be finite")
    g0 = _as_real_array("initial gradient", grad(waveform))
    if g0.shape != waveform.shape:
        raise ValueError(f"gradient shape {g0.shape} must match wfm shape {waveform.shape}")

    ascent_slope = _directional_derivative(g0, search_direction)
    if ascent_slope <= 0.0:
        return 0.0

    phi0 = -f0
    dphi0 = -ascent_slope
    best_alpha = 0.0
    best_phi = phi0

    def candidate(alpha: float) -> RealArray:
        return np.asarray(waveform + alpha * search_direction, dtype=np.float64)

    def phi(alpha: float) -> float:
        value = float(f(candidate(alpha)))
        if not math.isfinite(value):
            return math.inf
        return -value

    def dphi(alpha: float) -> float:
        gradient = np.asarray(grad(candidate(alpha)), dtype=np.float64)
        if gradient.shape != waveform.shape or not np.all(np.isfinite(gradient)):
            return math.nan
        return -_directional_derivative(gradient, search_direction)

    def record_best(alpha: float, value: float) -> None:
        nonlocal best_alpha, best_phi
        if value < best_phi:
            best_alpha = alpha
            best_phi = value

    def make_point(alpha: float, value: float) -> _LinePoint:
        return _LinePoint(alpha=alpha, phi=value, dphi=dphi(alpha))

    def zoom(point_lo: _LinePoint, point_hi: _LinePoint, remaining_iter: int) -> float:
        low = point_lo
        high = point_hi
        for _ in range(remaining_iter):
            if abs(high.alpha - low.alpha) <= 1e-14 * max(1.0, abs(low.alpha)):
                return best_alpha
            alpha_j = _bracket_trial_alpha(low, high)
            phi_j = phi(alpha_j)
            record_best(alpha_j, phi_j)
            if phi_j > phi0 + c1 * alpha_j * dphi0 or phi_j >= low.phi:
                high = make_point(alpha_j, phi_j)
                continue

            dphi_j = dphi(alpha_j)
            if not math.isfinite(dphi_j):
                return best_alpha
            if abs(dphi_j) <= -c2 * dphi0:
                return alpha_j
            if dphi_j * (high.alpha - low.alpha) >= 0.0:
                high = low
            low = _LinePoint(alpha=alpha_j, phi=phi_j, dphi=dphi_j)
        return best_alpha

    previous = _LinePoint(alpha=0.0, phi=phi0, dphi=dphi0)
    alpha = float(alpha0)
    for iteration in range(max_iter):
        phi_alpha = phi(alpha)
        record_best(alpha, phi_alpha)
        if phi_alpha > phi0 + c1 * alpha * dphi0 or (
            iteration > 0 and phi_alpha >= previous.phi
        ):
            return zoom(previous, make_point(alpha, phi_alpha), max_iter - iteration)

        dphi_alpha = dphi(alpha)
        if not math.isfinite(dphi_alpha):
            return best_alpha
        if abs(dphi_alpha) <= -c2 * dphi0:
            return alpha
        if dphi_alpha >= 0.0:
            return zoom(
                _LinePoint(alpha=alpha, phi=phi_alpha, dphi=dphi_alpha),
                previous,
                max_iter - iteration,
            )

        previous = _LinePoint(alpha=alpha, phi=phi_alpha, dphi=dphi_alpha)
        alpha *= 2.0

    return best_alpha


def _counted_grape_objective(cp: ControlProblem) -> tuple[Objective, Callable[[], int]]:
    """Return a GRAPE objective closure and a function-evaluation counter."""
    n_feval = 0

    def objective(wfm: RealArray) -> float:
        nonlocal n_feval
        n_feval += 1
        return float(grape_xy(cp, wfm))

    def count() -> int:
        return n_feval

    return objective, count


def _grape_gradient(cp: ControlProblem) -> Gradient:
    """Return a GRAPE gradient closure with a stable optimizer signature."""

    def gradient(wfm: RealArray) -> RealArray:
        return np.asarray(grape_gradient(cp, wfm), dtype=np.float64)

    return gradient


def _grape_hessian(cp: ControlProblem) -> Callable[[RealArray], RealArray]:
    """Return an exact GRAPE Hessian closure for small-waveform problems."""

    def hessian(wfm: RealArray) -> RealArray:
        value = grape_hessian(cp, wfm)
        if value is None:
            raise ValueError(
                "newton_raphson requires an exact Hessian; "
                "waveforms with more than 50 parameters are not supported"
            )
        return np.asarray(value, dtype=np.float64)

    return hessian


def _result(
    wfm: RealArray,
    fidelity: float,
    n_iter: int,
    n_feval: int,
    converged: bool,
    reason: str,
    history: list[float],
) -> OptimResult:
    """Build an optimizer result with copied mutable fields."""
    return OptimResult(
        wfm_final=np.asarray(wfm, dtype=np.float64).copy(),
        fidelity_final=float(fidelity),
        n_iter=n_iter,
        n_feval=n_feval,
        converged=converged,
        reason=reason,
        history=history.copy(),
    )


def gradient_ascent(
    cp: ControlProblem,
    wfm0: RealArray,
    tol_x: float = 1e-6,
    tol_g: float = 1e-6,
    max_iter: int = 500,
) -> OptimResult:
    """Optimise GRAPE controls with steepest ascent and cubic line search."""
    _validate_optimizer_controls(tol_x, tol_g, max_iter)
    waveform = _initial_waveform(cp, wfm0)
    objective, n_feval = _counted_grape_objective(cp)
    gradient_fn = _grape_gradient(cp)

    fidelity = objective(waveform)
    history = [fidelity]
    gradient = gradient_fn(waveform)
    if gradient.shape != waveform.shape:
        raise ValueError(f"gradient shape {gradient.shape} must match wfm shape {waveform.shape}")
    if _array_norm(gradient) <= tol_g:
        return _result(waveform, fidelity, 0, n_feval(), True, "grad_tol", history)

    for iteration in range(1, max_iter + 1):
        direction = gradient.copy()
        alpha = line_search_cubic(objective, gradient_fn, waveform, direction)
        step = np.asarray(alpha * direction, dtype=np.float64)
        step_norm = _array_norm(step)
        if alpha <= 0.0 or step_norm <= tol_x:
            return _result(
                waveform,
                fidelity,
                iteration - 1,
                n_feval(),
                True,
                "step_tol",
                history,
            )

        waveform = np.asarray(waveform + step, dtype=np.float64)
        fidelity = objective(waveform)
        history.append(fidelity)
        gradient = gradient_fn(waveform)
        if gradient.shape != waveform.shape:
            raise ValueError(
                f"gradient shape {gradient.shape} must match wfm shape {waveform.shape}"
            )
        if _array_norm(gradient) <= tol_g:
            return _result(waveform, fidelity, iteration, n_feval(), True, "grad_tol", history)

    return _result(waveform, fidelity, max_iter, n_feval(), False, "max_iter", history)


def lbfgs_state(m: int = 10) -> LBFGSState:
    """Return an empty L-BFGS memory dictionary."""
    if m <= 0:
        raise ValueError("m must be positive")
    return {"m": m, "s_history": [], "y_history": [], "rho_history": []}


def _copy_lbfgs_state(state: LBFGSState) -> LBFGSState:
    """Return a shallow copy of an L-BFGS state with copied history lists."""
    m = int(state["m"])
    if m <= 0:
        raise ValueError("state m must be positive")
    s_history = [np.asarray(s, dtype=np.float64).copy() for s in state["s_history"]]
    y_history = [np.asarray(y, dtype=np.float64).copy() for y in state["y_history"]]
    rho_history = [float(rho) for rho in state["rho_history"]]
    if not (len(s_history) == len(y_history) == len(rho_history)):
        raise ValueError("L-BFGS history lists must have the same length")
    return {
        "m": m,
        "s_history": s_history,
        "y_history": y_history,
        "rho_history": rho_history,
    }


def lbfgs_update(
    state: LBFGSState,
    wfm_diff: RealArray,
    grad_diff: RealArray,
) -> LBFGSState:
    """Return an updated L-BFGS state.

    For maximising a fidelity objective ``f``, pass ``wfm_new - wfm_old`` as
    ``wfm_diff`` and ``grad_old - grad_new`` as ``grad_diff``. That stores
    positive-curvature pairs for the equivalent minimisation of ``-f``.
    """
    next_state = _copy_lbfgs_state(state)
    step = _as_real_array("wfm_diff", wfm_diff)
    curvature = _as_real_array("grad_diff", grad_diff)
    if curvature.shape != step.shape:
        raise ValueError(
            f"grad_diff shape {curvature.shape} must match wfm_diff shape {step.shape}"
        )

    step_flat = step.reshape(-1)
    curvature_flat = curvature.reshape(-1)
    ys = float(np.dot(curvature_flat, step_flat))
    if ys <= 1e-14 * max(1.0, _array_norm(step) * _array_norm(curvature)):
        return next_state

    next_state["s_history"].append(step.copy())
    next_state["y_history"].append(curvature.copy())
    next_state["rho_history"].append(1.0 / ys)
    if len(next_state["s_history"]) > next_state["m"]:
        next_state["s_history"] = next_state["s_history"][-next_state["m"] :]
        next_state["y_history"] = next_state["y_history"][-next_state["m"] :]
        next_state["rho_history"] = next_state["rho_history"][-next_state["m"] :]
    return next_state


def lbfgs_direction(state: LBFGSState, grad: RealArray) -> RealArray:
    """Return an L-BFGS ascent direction for a maximisation gradient."""
    memory = _copy_lbfgs_state(state)
    gradient = _as_real_array("grad", grad)
    if not memory["s_history"]:
        return gradient.copy()

    q = -gradient.reshape(-1).copy()
    alpha_values: list[float] = []
    flat_s_history = [s.reshape(-1) for s in memory["s_history"]]
    flat_y_history = [y.reshape(-1) for y in memory["y_history"]]
    for step, curvature, rho in zip(
        reversed(flat_s_history),
        reversed(flat_y_history),
        reversed(memory["rho_history"]),
    ):
        if step.shape != q.shape or curvature.shape != q.shape:
            raise ValueError("L-BFGS history shapes must match gradient shape")
        alpha = rho * float(np.dot(step, q))
        q -= alpha * curvature
        alpha_values.append(alpha)

    last_step = flat_s_history[-1]
    last_curvature = flat_y_history[-1]
    yy = float(np.dot(last_curvature, last_curvature))
    gamma = 1.0 if yy <= 0.0 else float(np.dot(last_step, last_curvature) / yy)
    r = gamma * q

    for step, curvature, rho, alpha in zip(
        flat_s_history,
        flat_y_history,
        memory["rho_history"],
        reversed(alpha_values),
    ):
        beta = rho * float(np.dot(curvature, r))
        r += step * (alpha - beta)

    direction = np.asarray((-r).reshape(gradient.shape), dtype=np.float64)
    if _directional_derivative(gradient, direction) <= 0.0:
        return gradient.copy()
    return direction


def lbfgs_grape(
    cp: ControlProblem,
    wfm0: RealArray,
    m: int = 10,
    tol_x: float = 1e-6,
    tol_g: float = 1e-6,
    max_iter: int = 500,
) -> OptimResult:
    """Optimise GRAPE controls with limited-memory BFGS and line search."""
    _validate_optimizer_controls(tol_x, tol_g, max_iter)
    waveform = _initial_waveform(cp, wfm0)
    state = lbfgs_state(m)
    objective, n_feval = _counted_grape_objective(cp)
    gradient_fn = _grape_gradient(cp)

    fidelity = objective(waveform)
    history = [fidelity]
    gradient = gradient_fn(waveform)
    if gradient.shape != waveform.shape:
        raise ValueError(f"gradient shape {gradient.shape} must match wfm shape {waveform.shape}")
    if _array_norm(gradient) <= tol_g:
        return _result(waveform, fidelity, 0, n_feval(), True, "grad_tol", history)

    for iteration in range(1, max_iter + 1):
        direction = lbfgs_direction(state, gradient)
        alpha = line_search_cubic(objective, gradient_fn, waveform, direction)
        if alpha <= 0.0 and not np.array_equal(direction, gradient):
            direction = gradient.copy()
            alpha = line_search_cubic(objective, gradient_fn, waveform, direction)

        step = np.asarray(alpha * direction, dtype=np.float64)
        step_norm = _array_norm(step)
        if alpha <= 0.0 or step_norm <= tol_x:
            return _result(
                waveform,
                fidelity,
                iteration - 1,
                n_feval(),
                True,
                "step_tol",
                history,
            )

        previous_waveform = waveform
        previous_gradient = gradient
        waveform = np.asarray(waveform + step, dtype=np.float64)
        fidelity = objective(waveform)
        history.append(fidelity)
        gradient = gradient_fn(waveform)
        if gradient.shape != waveform.shape:
            raise ValueError(
                f"gradient shape {gradient.shape} must match wfm shape {waveform.shape}"
            )
        state = lbfgs_update(state, waveform - previous_waveform, previous_gradient - gradient)
        if _array_norm(gradient) <= tol_g:
            return _result(waveform, fidelity, iteration, n_feval(), True, "grad_tol", history)

    return _result(waveform, fidelity, max_iter, n_feval(), False, "max_iter", history)


def newton_raphson(
    cp: ControlProblem,
    wfm0: RealArray,
    regularise: bool = True,
    rfo: bool = False,
    tol_x: float = 1e-6,
    tol_g: float = 1e-6,
    max_iter: int = 200,
) -> OptimResult:
    """Optimise GRAPE controls with Newton-Raphson steps on the exact Hessian."""
    _validate_optimizer_controls(tol_x, tol_g, max_iter)
    waveform = _initial_waveform(cp, wfm0)
    objective, n_feval = _counted_grape_objective(cp)
    gradient_fn = _grape_gradient(cp)
    hessian_fn = _grape_hessian(cp)
    freeze_mask = None if cp.freeze is None else np.asarray(cp.freeze, dtype=np.bool_)

    fidelity = objective(waveform)
    history = [fidelity]
    gradient = gradient_fn(waveform)
    if gradient.shape != waveform.shape:
        raise ValueError(f"gradient shape {gradient.shape} must match wfm shape {waveform.shape}")
    if _array_norm(gradient) <= tol_g:
        return _result(waveform, fidelity, 0, n_feval(), True, "grad_tol", history)

    for iteration in range(1, max_iter + 1):
        hessian = hessian_fn(waveform)
        step_direction = _newton_step(gradient, hessian, regularise=regularise, rfo=rfo)
        if freeze_mask is not None:
            step_direction = step_direction.copy()
            step_direction[freeze_mask] = 0.0

        alpha = line_search_cubic(objective, gradient_fn, waveform, step_direction, alpha0=1.0)
        step = np.asarray(alpha * step_direction, dtype=np.float64)
        step_norm = _array_norm(step)
        if alpha <= 0.0 or step_norm <= tol_x:
            return _result(
                waveform,
                fidelity,
                iteration - 1,
                n_feval(),
                True,
                "step_tol",
                history,
            )

        candidate = np.asarray(waveform + step, dtype=np.float64)
        waveform = apply_freeze(candidate, freeze_mask, waveform)
        fidelity = objective(waveform)
        history.append(fidelity)
        gradient = gradient_fn(waveform)
        if gradient.shape != waveform.shape:
            raise ValueError(
                f"gradient shape {gradient.shape} must match wfm shape {waveform.shape}"
            )
        if _array_norm(gradient) <= tol_g:
            return _result(waveform, fidelity, iteration, n_feval(), True, "grad_tol", history)

    return _result(waveform, fidelity, max_iter, n_feval(), False, "max_iter", history)
