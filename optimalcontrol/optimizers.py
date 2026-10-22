"""Shared optimizer result types and line-search utilities."""

import functools
import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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


@dataclass(frozen=True)
class _CheckpointData:
    """Validated checkpoint contents used by the optimizers."""

    wfm: RealArray
    history: list[float]
    n_feval: int = 0
    lbfgs_state: LBFGSState | None = None
    signature: str | None = None


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


def _serialise_lbfgs_state(state: LBFGSState) -> dict[str, object]:
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
    # Atomic write: fsync a temp file then os.replace, so an interrupted write
    # (Ctrl-C, OOM, power loss) cannot destroy the previous good checkpoint.
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, checkpoint_path)


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
