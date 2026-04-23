"""Shared optimizer result types and line-search utilities."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

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
