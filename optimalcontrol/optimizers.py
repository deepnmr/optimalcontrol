"""Shared optimizer result types and line-search utilities."""

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from optimalcontrol.grape import (
    ControlProblem,
    apply_freeze,
    grape_hessian,
    grape_xy,
    grape_xy_and_gradient,
    validate_waveform,
)

if TYPE_CHECKING:
    from optimalcontrol.io import Waveform

from optimalcontrol._types import Array, RealArray

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
    trajectory: list[Array] | None = None


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


class _SerialisedLBFGSState(TypedDict):
    """JSON-compatible representation of an ``LBFGSState``."""

    m: int
    s_history: list[list[list[float]]]
    y_history: list[list[list[float]]]
    rho_history: list[float]


@dataclass(frozen=True)
class _CheckpointData:
    """Validated checkpoint contents used by the optimizers."""

    wfm: RealArray
    history: list[float]
    n_feval: int = 0
    lbfgs_state: LBFGSState | None = None
    signature: str | None = None


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


def _check_gradient_shape(gradient: RealArray, waveform: RealArray) -> None:
    """Raise ValueError if a gradient does not match the waveform shape."""
    if gradient.shape != waveform.shape:
        raise ValueError(f"gradient shape {gradient.shape} must match wfm shape {waveform.shape}")


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
    waveform = np.asarray(wfm0, dtype=np.float64)
    validate_waveform(waveform, len(cp.operators), waveform.shape[0] if waveform.ndim else 0)
    return waveform.copy()


def _validate_checkpoint_path(path: str) -> Path:
    """Return a usable checkpoint file path."""
    if not path.strip():
        raise ValueError("checkpoint_path must be a non-empty string")
    checkpoint_path = Path(path)
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        raise ValueError("checkpoint_path must point to a file, not a directory")
    return checkpoint_path


def _checkpoint_signature(cp: ControlProblem, optimizer_name: str) -> str:
    """Return a signature binding a checkpoint to its problem and optimizer."""
    from optimalcontrol.io import _hash_control_problem

    payload = f"{optimizer_name}\0{_hash_control_problem(cp)}".encode()
    return hashlib.sha256(payload).hexdigest()


def _serialise_lbfgs_state(state: LBFGSState) -> _SerialisedLBFGSState:
    """Return a JSON-compatible snapshot of an L-BFGS state."""
    memory = _copy_lbfgs_state(state)
    return {
        "m": int(memory["m"]),
        "s_history": [step.tolist() for step in memory["s_history"]],
        "y_history": [curvature.tolist() for curvature in memory["y_history"]],
        "rho_history": [float(rho) for rho in memory["rho_history"]],
    }


def _deserialise_lbfgs_state(raw: object) -> LBFGSState:
    """Return a validated L-BFGS state restored from checkpoint JSON."""
    if not isinstance(raw, dict):
        raise ValueError("checkpoint lbfgs_state must be a JSON object")

    try:
        m_value = int(raw["m"])
        s_raw = raw["s_history"]
        y_raw = raw["y_history"]
        rho_raw = raw["rho_history"]
    except KeyError as exc:
        raise ValueError(f"checkpoint lbfgs_state is missing {exc.args[0]!r}") from exc

    if not isinstance(s_raw, list) or not isinstance(y_raw, list) or not isinstance(rho_raw, list):
        raise ValueError("checkpoint lbfgs_state histories must be JSON lists")

    s_history = [
        _as_real_array("checkpoint s_history", np.asarray(value, dtype=np.float64))
        for value in s_raw
    ]
    y_history = [
        _as_real_array("checkpoint y_history", np.asarray(value, dtype=np.float64))
        for value in y_raw
    ]
    rho_history = [float(value) for value in rho_raw]
    if not all(math.isfinite(value) for value in rho_history):
        raise ValueError("checkpoint rho_history entries must be finite")

    state: LBFGSState = {
        "m": m_value,
        "s_history": s_history,
        "y_history": y_history,
        "rho_history": rho_history,
    }
    return _copy_lbfgs_state(state)


def _checkpoint_history(result_so_far: list[float] | OptimResult) -> list[float]:
    """Return the fidelity-history portion of a saveable result."""
    raw_history = result_so_far.history if isinstance(result_so_far, OptimResult) else result_so_far
    history = [float(value) for value in raw_history]
    if not all(math.isfinite(value) for value in history):
        raise ValueError("checkpoint history entries must be finite")
    return history


def _write_checkpoint(checkpoint_path: Path, checkpoint: _CheckpointData) -> None:
    """Write one optimizer checkpoint to disk."""
    waveform = _as_real_array("wfm", checkpoint.wfm)
    if waveform.ndim != 2:
        raise ValueError(f"wfm must be two-dimensional, got shape {waveform.shape}")

    history = _checkpoint_history(checkpoint.history)
    payload: dict[str, object] = {
        "wfm": waveform.tolist(),
        "history": history,
        "n_feval": int(checkpoint.n_feval),
    }
    if checkpoint.n_feval < 0:
        raise ValueError("checkpoint n_feval must be non-negative")
    if checkpoint.lbfgs_state is not None:
        payload["lbfgs_state"] = _serialise_lbfgs_state(checkpoint.lbfgs_state)
    if checkpoint.signature is not None:
        if not checkpoint.signature:
            raise ValueError("checkpoint signature must be a non-empty string")
        payload["signature"] = checkpoint.signature

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_checkpoint(path: str | Path) -> _CheckpointData:
    """Return validated checkpoint contents from disk."""
    checkpoint_path = path if isinstance(path, Path) else _validate_checkpoint_path(path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a JSON object")

    if "wfm" not in payload or "history" not in payload:
        raise ValueError("checkpoint payload must contain 'wfm' and 'history'")

    waveform = _as_real_array("checkpoint wfm", np.asarray(payload["wfm"], dtype=np.float64))
    if waveform.ndim != 2:
        raise ValueError(f"checkpoint wfm must be two-dimensional, got shape {waveform.shape}")

    history_value = payload["history"]
    if not isinstance(history_value, list):
        raise ValueError("checkpoint history must be a JSON list")
    history = _checkpoint_history(history_value)

    raw_n_feval = payload.get("n_feval", 0)
    n_feval = int(raw_n_feval)
    if n_feval < 0:
        raise ValueError("checkpoint n_feval must be non-negative")

    lbfgs_state = (
        _deserialise_lbfgs_state(payload["lbfgs_state"]) if "lbfgs_state" in payload else None
    )
    raw_signature = payload.get("signature")
    if raw_signature is not None and (not isinstance(raw_signature, str) or not raw_signature):
        raise ValueError("checkpoint signature must be a non-empty string")
    return _CheckpointData(
        wfm=waveform,
        history=history,
        n_feval=n_feval,
        lbfgs_state=lbfgs_state,
        signature=raw_signature,
    )


def _effective_checkpoint_path(
    cp: ControlProblem,
    checkpoint_path: str | None,
) -> Path | None:
    """Return the checkpoint file requested by the optimizer call, if any."""
    candidate = checkpoint_path if checkpoint_path is not None else cp.checkpoint_path
    if candidate is None:
        return None
    return _validate_checkpoint_path(candidate)


def _restore_checkpoint(
    cp: ControlProblem,
    wfm0: RealArray,
    checkpoint_path: Path | None,
    signature: str | None,
) -> _CheckpointData:
    """Return the starting optimizer state, loading a checkpoint if present."""
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = _read_checkpoint(checkpoint_path)
        waveform = _initial_waveform(cp, checkpoint.wfm)
        requested_shape = np.asarray(wfm0).shape
        if waveform.shape != requested_shape:
            raise ValueError(
                f"checkpoint waveform shape {waveform.shape} must match "
                f"wfm0 shape {requested_shape}"
            )
        if checkpoint.signature is None:
            return _CheckpointData(wfm=waveform, history=[], signature=signature)
        if checkpoint.signature != signature:
            raise ValueError("checkpoint optimizer or control problem does not match current run")
        return _CheckpointData(
            wfm=waveform,
            history=checkpoint.history.copy(),
            n_feval=checkpoint.n_feval,
            lbfgs_state=checkpoint.lbfgs_state,
            signature=checkpoint.signature,
        )

    waveform = _initial_waveform(cp, wfm0)
    return _CheckpointData(wfm=waveform, history=[], n_feval=0, lbfgs_state=None)


def _save_optimizer_checkpoint(
    checkpoint_path: Path | None,
    waveform: RealArray,
    history: list[float],
    *,
    n_feval: int,
    lbfgs_state: LBFGSState | None = None,
    signature: str | None = None,
) -> None:
    """Persist optimizer state when checkpointing is enabled."""
    if checkpoint_path is None:
        return
    _write_checkpoint(
        checkpoint_path,
        _CheckpointData(
            wfm=waveform,
            history=history,
            n_feval=n_feval,
            lbfgs_state=lbfgs_state,
            signature=signature,
        ),
    )


def save_checkpoint(path: str, wfm: RealArray, result_so_far: list[float] | OptimResult) -> None:
    """Save a waveform and fidelity history to a JSON checkpoint file."""
    checkpoint_path = _validate_checkpoint_path(path)
    n_feval = result_so_far.n_feval if isinstance(result_so_far, OptimResult) else 0
    _write_checkpoint(
        checkpoint_path,
        _CheckpointData(
            wfm=wfm,
            history=_checkpoint_history(result_so_far),
            n_feval=n_feval,
            lbfgs_state=None,
        ),
    )


def load_checkpoint(path: str) -> tuple[RealArray, list[float]]:
    """Load a waveform and fidelity history from a JSON checkpoint file."""
    checkpoint = _read_checkpoint(path)
    return checkpoint.wfm.copy(), checkpoint.history.copy()


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
        if phi_alpha > phi0 + c1 * alpha * dphi0 or (iteration > 0 and phi_alpha >= previous.phi):
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


def _cached_grape_evaluators(
    cp: ControlProblem,
    *,
    initial_count: int = 0,
) -> tuple[Objective, Gradient, Callable[[], int]]:
    """Return memoised GRAPE objective and gradient closures plus a feval counter.

    Both closures share one small waveform-keyed cache, so the repeated
    evaluations at identical points made by the optimizers and the line search
    (the alpha=0 point, the accepted step) cost nothing. A gradient evaluation
    produces the fidelity as a by-product and seeds the objective cache.
    """
    if initial_count < 0:
        raise ValueError("initial_count must be non-negative")
    n_feval = initial_count
    max_entries = 32
    value_cache: dict[bytes, float] = {}
    grad_cache: dict[bytes, RealArray] = {}
    insertion_order: list[bytes] = []

    def _remember(key: bytes) -> None:
        if key not in insertion_order:
            insertion_order.append(key)
        while len(insertion_order) > max_entries:
            stale = insertion_order.pop(0)
            value_cache.pop(stale, None)
            grad_cache.pop(stale, None)

    def objective(wfm: RealArray) -> float:
        nonlocal n_feval
        waveform = np.asarray(wfm, dtype=np.float64)
        key = waveform.tobytes()
        cached = value_cache.get(key)
        if cached is not None:
            return cached
        n_feval += 1
        value = float(grape_xy(cp, waveform))
        value_cache[key] = value
        _remember(key)
        return value

    def gradient(wfm: RealArray) -> RealArray:
        waveform = np.asarray(wfm, dtype=np.float64)
        key = waveform.tobytes()
        cached = grad_cache.get(key)
        if cached is not None:
            return cached.copy()
        value, grad = grape_xy_and_gradient(cp, waveform)
        value_cache.setdefault(key, float(value))
        grad_cache[key] = np.asarray(grad, dtype=np.float64)
        _remember(key)
        return grad_cache[key].copy()

    def count() -> int:
        return n_feval

    return objective, gradient, count


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
    trajectory: list[Array] | None = None,
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
        trajectory=None
        if trajectory is None
        else [np.asarray(state, dtype=np.complex128).copy() for state in trajectory],
    )


def _trajectory_for_result(
    cp: ControlProblem,
    waveform: RealArray,
    produce_trajectory: bool,
) -> list[Array] | None:
    """Return final-waveform trajectory diagnostics when requested."""
    if not produce_trajectory:
        return None

    from optimalcontrol.analysis import state_trajectory

    return [
        np.asarray(state, dtype=np.complex128).copy() for state in state_trajectory(cp, waveform)
    ]


def _grape_result(
    cp: ControlProblem,
    wfm: RealArray,
    fidelity: float,
    n_iter: int,
    n_feval: int,
    converged: bool,
    reason: str,
    history: list[float],
    *,
    produce_trajectory: bool,
) -> OptimResult:
    """Build a GRAPE optimizer result with optional trajectory diagnostics."""
    return _result(
        wfm,
        fidelity,
        n_iter,
        n_feval,
        converged,
        reason,
        history,
        trajectory=_trajectory_for_result(cp, wfm, produce_trajectory),
    )


_ComputeStep = Callable[[Objective, Gradient, RealArray, RealArray], tuple[float, RealArray]]
_Finalise = Callable[[RealArray, float, int, int, bool, str, list[float]], OptimResult]


def _drive_optimizer(
    cp: ControlProblem,
    wfm0: RealArray,
    *,
    tol_x: float,
    tol_g: float,
    max_iter: int,
    checkpoint_path: str | None,
    optimizer_name: str,
    compute_step: _ComputeStep,
    finalise: _Finalise,
    restore_extra: Callable[[_CheckpointData, RealArray], None] | None = None,
    apply_step: Callable[[RealArray, RealArray], RealArray] | None = None,
    on_accept: Callable[[RealArray, RealArray, RealArray, RealArray], None] | None = None,
    checkpoint_state: Callable[[], LBFGSState | None] | None = None,
) -> OptimResult:
    """Run the shared GRAPE ascent loop; optimizers differ only in their hooks.

    The setup (control validation, checkpoint restore, cached evaluators,
    early grad_tol/max_iter returns), the accepted-step bookkeeping, and the
    checkpoint cadence (every tenth iteration, on convergence, and on every
    exit path) are identical across the optimizers and live here once.

    Hooks: ``compute_step`` returns the line-search ``(alpha, direction)``
    pair, ``finalise`` builds the result, ``restore_extra`` consumes extra
    checkpoint state, ``apply_step`` overrides the plain waveform update,
    ``on_accept`` observes each accepted step, and ``checkpoint_state``
    supplies optimizer memory to persist alongside checkpoints.
    """
    _validate_optimizer_controls(tol_x, tol_g, max_iter)
    checkpoint_file = _effective_checkpoint_path(cp, checkpoint_path)
    signature = _checkpoint_signature(cp, optimizer_name) if checkpoint_file is not None else None
    checkpoint = _restore_checkpoint(cp, wfm0, checkpoint_file, signature)
    waveform = checkpoint.wfm
    if restore_extra is not None:
        restore_extra(checkpoint, waveform)
    objective, gradient_fn, n_feval = _cached_grape_evaluators(cp, initial_count=checkpoint.n_feval)

    history = checkpoint.history.copy()

    def save() -> None:
        _save_optimizer_checkpoint(
            checkpoint_file,
            waveform,
            history,
            n_feval=n_feval(),
            lbfgs_state=None if checkpoint_state is None else checkpoint_state(),
            signature=signature,
        )

    fidelity = float(history[-1]) if history else objective(waveform)
    if not history:
        history.append(fidelity)
    completed_iter = max(0, len(history) - 1)
    gradient = gradient_fn(waveform)
    _check_gradient_shape(gradient, waveform)
    if _array_norm(gradient) <= tol_g:
        save()
        return finalise(waveform, fidelity, completed_iter, n_feval(), True, "grad_tol", history)
    if completed_iter >= max_iter:
        save()
        return finalise(waveform, fidelity, completed_iter, n_feval(), False, "max_iter", history)

    for iteration in range(completed_iter + 1, max_iter + 1):
        alpha, direction = compute_step(objective, gradient_fn, waveform, gradient)
        step = np.asarray(alpha * direction, dtype=np.float64)
        step_norm = _array_norm(step)
        if alpha <= 0.0 or step_norm <= tol_x:
            save()
            return finalise(waveform, fidelity, iteration - 1, n_feval(), True, "step_tol", history)

        previous_waveform = waveform
        previous_gradient = gradient
        if apply_step is None:
            waveform = np.asarray(waveform + step, dtype=np.float64)
        else:
            waveform = apply_step(waveform, step)
        fidelity = objective(waveform)
        history.append(fidelity)
        gradient = gradient_fn(waveform)
        _check_gradient_shape(gradient, waveform)
        if on_accept is not None:
            on_accept(previous_waveform, previous_gradient, waveform, gradient)
        grad_norm = _array_norm(gradient)
        if iteration % 10 == 0 or grad_norm <= tol_g:
            save()
        if grad_norm <= tol_g:
            return finalise(waveform, fidelity, iteration, n_feval(), True, "grad_tol", history)

    save()
    return finalise(waveform, fidelity, max_iter, n_feval(), False, "max_iter", history)


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


def _validate_lbfgs_checkpoint_state(state: LBFGSState, waveform_shape: tuple[int, ...]) -> None:
    """Raise ValueError if restored L-BFGS arrays do not match the waveform shape."""
    memory = _copy_lbfgs_state(state)
    for name, history_list in (
        ("s_history", memory["s_history"]),
        ("y_history", memory["y_history"]),
    ):
        for index, array in enumerate(history_list):
            if array.shape != waveform_shape:
                raise ValueError(
                    f"checkpoint {name}[{index}] shape {array.shape} "
                    f"must match waveform shape {waveform_shape}"
                )


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
    checkpoint_path: str | None = None,
    produce_trajectory: bool = False,
) -> OptimResult:
    """Optimise GRAPE controls with limited-memory BFGS and line search."""
    state = lbfgs_state(m)

    def restore_extra(checkpoint: _CheckpointData, waveform: RealArray) -> None:
        nonlocal state
        if checkpoint.lbfgs_state is None:
            return
        state = _copy_lbfgs_state(checkpoint.lbfgs_state)
        if int(state["m"]) != m:
            raise ValueError(
                f"checkpoint L-BFGS memory m={state['m']} does not match requested m={m}"
            )
        _validate_lbfgs_checkpoint_state(state, waveform.shape)

    def compute_step(
        objective: Objective,
        gradient_fn: Gradient,
        waveform: RealArray,
        gradient: RealArray,
    ) -> tuple[float, RealArray]:
        direction = lbfgs_direction(state, gradient)
        alpha = line_search_cubic(objective, gradient_fn, waveform, direction)
        if alpha <= 0.0 and not np.array_equal(direction, gradient):
            direction = gradient.copy()
            alpha = line_search_cubic(objective, gradient_fn, waveform, direction)
        return alpha, direction

    def on_accept(
        previous_waveform: RealArray,
        previous_gradient: RealArray,
        waveform: RealArray,
        gradient: RealArray,
    ) -> None:
        nonlocal state
        state = lbfgs_update(state, waveform - previous_waveform, previous_gradient - gradient)

    def finalise(
        waveform: RealArray,
        fidelity: float,
        n_iter: int,
        n_feval: int,
        converged: bool,
        reason: str,
        history: list[float],
    ) -> OptimResult:
        return _grape_result(
            cp,
            waveform,
            fidelity,
            n_iter,
            n_feval,
            converged,
            reason,
            history,
            produce_trajectory=produce_trajectory,
        )

    return _drive_optimizer(
        cp,
        wfm0,
        tol_x=tol_x,
        tol_g=tol_g,
        max_iter=max_iter,
        checkpoint_path=checkpoint_path,
        optimizer_name="lbfgs",
        compute_step=compute_step,
        finalise=finalise,
        restore_extra=restore_extra,
        on_accept=on_accept,
        checkpoint_state=lambda: state,
    )


def newton_raphson(
    cp: ControlProblem,
    wfm0: RealArray,
    regularise: bool = True,
    rfo: bool = False,
    tol_x: float = 1e-6,
    tol_g: float = 1e-6,
    max_iter: int = 200,
    checkpoint_path: str | None = None,
    produce_trajectory: bool = False,
) -> OptimResult:
    """Optimise GRAPE controls with Newton-Raphson steps on the exact Hessian."""
    hessian_fn = _grape_hessian(cp)
    freeze_mask = None if cp.freeze is None else np.asarray(cp.freeze, dtype=np.bool_)

    def compute_step(
        objective: Objective,
        gradient_fn: Gradient,
        waveform: RealArray,
        gradient: RealArray,
    ) -> tuple[float, RealArray]:
        hessian = hessian_fn(waveform)
        step_direction = _newton_step(gradient, hessian, regularise=regularise, rfo=rfo)
        if freeze_mask is not None:
            step_direction = step_direction.copy()
            step_direction[freeze_mask] = 0.0
        alpha = line_search_cubic(objective, gradient_fn, waveform, step_direction, alpha0=1.0)
        return alpha, step_direction

    def apply_step(waveform: RealArray, step: RealArray) -> RealArray:
        candidate = np.asarray(waveform + step, dtype=np.float64)
        return apply_freeze(candidate, freeze_mask, waveform)

    def finalise(
        waveform: RealArray,
        fidelity: float,
        n_iter: int,
        n_feval: int,
        converged: bool,
        reason: str,
        history: list[float],
    ) -> OptimResult:
        return _grape_result(
            cp,
            waveform,
            fidelity,
            n_iter,
            n_feval,
            converged,
            reason,
            history,
            produce_trajectory=produce_trajectory,
        )

    return _drive_optimizer(
        cp,
        wfm0,
        tol_x=tol_x,
        tol_g=tol_g,
        max_iter=max_iter,
        checkpoint_path=checkpoint_path,
        optimizer_name="newton",
        compute_step=compute_step,
        finalise=finalise,
        apply_step=apply_step,
    )


def run_grape(
    cp: ControlProblem,
    wfm0: RealArray,
    method: str = "lbfgs",
    **kwargs: object,
) -> tuple["Waveform", OptimResult]:
    """Optimise a GRAPE problem and return an exportable waveform plus result.

    ``method`` accepts ``"lbfgs"``/``"lbfgs_grape"`` and
    ``"newton"``/``"newton_raphson"``. Keyword arguments are passed directly to
    the selected optimizer.
    """
    method_key = method.lower().replace("-", "_")
    if method_key in {"lbfgs", "l_bfgs", "lbfgs_grape"}:
        optimizer = cast(Callable[..., OptimResult], lbfgs_grape)
    elif method_key in {"newton", "newton_raphson"}:
        optimizer = cast(Callable[..., OptimResult], newton_raphson)
    else:
        raise ValueError("method must be 'lbfgs' or 'newton'")

    from optimalcontrol.io import _hash_control_problem, waveform_from_result

    _hash_control_problem(cp)
    result = optimizer(cp, wfm0, **kwargs)
    waveform = waveform_from_result(cp, wfm0, result)
    return waveform, result
