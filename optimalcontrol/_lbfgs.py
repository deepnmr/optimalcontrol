"""Limited-memory BFGS memory state, update, and ascent direction."""

from typing import TypedDict

import numpy as np

from optimalcontrol._line_search import _array_norm, _as_real_array, _directional_derivative
from optimalcontrol._types import RealArray


class LBFGSState(TypedDict):
    """Limited-memory inverse-Hessian state for maximisation optimizers."""

    m: int
    s_history: list[RealArray]
    y_history: list[RealArray]
    rho_history: list[float]


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
        strict=True,
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
        strict=True,
    ):
        beta = rho * float(np.dot(curvature, r))
        r += step * (alpha - beta)

    direction = np.asarray((-r).reshape(gradient.shape), dtype=np.float64)
    if _directional_derivative(gradient, direction) <= 0.0:
        return gradient.copy()
    return direction
