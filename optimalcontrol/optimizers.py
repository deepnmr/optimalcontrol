"""Shared optimizer result types and line-search utilities."""

import functools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from optimalcontrol.grape import (
    ControlProblem,
    apply_freeze,
    grape_hessian,
    grape_xy,
    grape_xy_and_gradient,
)

if TYPE_CHECKING:
    from optimalcontrol.io import Waveform

from optimalcontrol._checkpoint import OptimResult as OptimResult
from optimalcontrol._checkpoint import _checkpoint_history as _checkpoint_history
from optimalcontrol._checkpoint import _checkpoint_signature as _checkpoint_signature
from optimalcontrol._checkpoint import _CheckpointData as _CheckpointData
from optimalcontrol._checkpoint import _deserialise_lbfgs_state as _deserialise_lbfgs_state
from optimalcontrol._checkpoint import _effective_checkpoint_path as _effective_checkpoint_path
from optimalcontrol._checkpoint import _initial_waveform as _initial_waveform
from optimalcontrol._checkpoint import _read_checkpoint as _read_checkpoint
from optimalcontrol._checkpoint import _restore_checkpoint as _restore_checkpoint
from optimalcontrol._checkpoint import _save_optimizer_checkpoint as _save_optimizer_checkpoint
from optimalcontrol._checkpoint import _serialise_lbfgs_state as _serialise_lbfgs_state
from optimalcontrol._checkpoint import _validate_checkpoint_path as _validate_checkpoint_path
from optimalcontrol._checkpoint import (
    _validate_lbfgs_checkpoint_state as _validate_lbfgs_checkpoint_state,
)
from optimalcontrol._checkpoint import _write_checkpoint as _write_checkpoint
from optimalcontrol._checkpoint import load_checkpoint as load_checkpoint
from optimalcontrol._checkpoint import save_checkpoint as save_checkpoint
from optimalcontrol._lbfgs import LBFGSState as LBFGSState
from optimalcontrol._lbfgs import _copy_lbfgs_state as _copy_lbfgs_state
from optimalcontrol._lbfgs import lbfgs_direction as lbfgs_direction
from optimalcontrol._lbfgs import lbfgs_state as lbfgs_state
from optimalcontrol._lbfgs import lbfgs_update as lbfgs_update
from optimalcontrol._line_search import Gradient as Gradient
from optimalcontrol._line_search import Objective as Objective
from optimalcontrol._line_search import _array_norm as _array_norm
from optimalcontrol._line_search import _as_real_array as _as_real_array
from optimalcontrol._line_search import _bracket_trial_alpha as _bracket_trial_alpha
from optimalcontrol._line_search import _cubic_interpolated_minimum as _cubic_interpolated_minimum
from optimalcontrol._line_search import _directional_derivative as _directional_derivative
from optimalcontrol._line_search import _LinePoint as _LinePoint
from optimalcontrol._line_search import line_search_cubic as line_search_cubic
from optimalcontrol._newton import _as_symmetric_matrix as _as_symmetric_matrix
from optimalcontrol._newton import _newton_step as _newton_step
from optimalcontrol._newton import _positive_definite_shift as _positive_definite_shift
from optimalcontrol._newton import _rfo_step as _rfo_step
from optimalcontrol._newton import _solve_symmetric_system as _solve_symmetric_system
from optimalcontrol._types import Array, RealArray


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

    def _remember(key: bytes) -> None:
        # value_cache is insertion-ordered and always holds every grad_cache key.
        while len(value_cache) > max_entries:
            stale = next(iter(value_cache))
            del value_cache[stale]
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
    """Build a GRAPE optimizer result with copied fields and optional trajectory."""
    return OptimResult(
        wfm_final=np.asarray(wfm, dtype=np.float64).copy(),
        fidelity_final=float(fidelity),
        n_iter=n_iter,
        n_feval=n_feval,
        converged=converged,
        reason=reason,
        history=history.copy(),
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
        if alpha <= 0.0:
            save()
            return finalise(
                waveform, fidelity, iteration - 1, n_feval(), False, "line_search_failed", history
            )
        step = np.asarray(alpha * direction, dtype=np.float64)
        step_norm = _array_norm(step)
        if step_norm <= tol_x:
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

    return _drive_optimizer(
        cp,
        wfm0,
        tol_x=tol_x,
        tol_g=tol_g,
        max_iter=max_iter,
        checkpoint_path=checkpoint_path,
        optimizer_name="lbfgs",
        compute_step=compute_step,
        finalise=functools.partial(_grape_result, cp, produce_trajectory=produce_trajectory),
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
        if alpha <= 0.0 and not np.array_equal(step_direction, gradient):
            step_direction = gradient.copy()
            alpha = line_search_cubic(objective, gradient_fn, waveform, step_direction)
        return alpha, step_direction

    def apply_step(waveform: RealArray, step: RealArray) -> RealArray:
        candidate = np.asarray(waveform + step, dtype=np.float64)
        return apply_freeze(candidate, freeze_mask, waveform)

    return _drive_optimizer(
        cp,
        wfm0,
        tol_x=tol_x,
        tol_g=tol_g,
        max_iter=max_iter,
        checkpoint_path=checkpoint_path,
        optimizer_name="newton",
        compute_step=compute_step,
        finalise=functools.partial(_grape_result, cp, produce_trajectory=produce_trajectory),
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
    optimizers: dict[str, Callable[..., OptimResult]] = {
        "lbfgs": lbfgs_grape,
        "l_bfgs": lbfgs_grape,
        "lbfgs_grape": lbfgs_grape,
        "newton": newton_raphson,
        "newton_raphson": newton_raphson,
    }
    optimizer = optimizers.get(method.lower().replace("-", "_"))
    if optimizer is None:
        raise ValueError("method must be 'lbfgs' or 'newton'")

    from optimalcontrol.io import _hash_control_problem, waveform_from_result

    _hash_control_problem(cp)
    result = optimizer(cp, wfm0, **kwargs)
    waveform = waveform_from_result(cp, wfm0, result)
    return waveform, result
