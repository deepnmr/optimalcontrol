"""Tests for the L-BFGS memory helpers used by the maximising GRAPE optimizer."""

import numpy as np
import numpy.testing as npt

from optimalcontrol.optimizers import lbfgs_direction, lbfgs_state, lbfgs_update


def test_lbfgs_state_starts_empty() -> None:
    state = lbfgs_state(4)

    assert state["m"] == 4
    assert state["s_history"] == []
    assert state["y_history"] == []
    assert state["rho_history"] == []


def test_lbfgs_update_keeps_at_most_m_pairs() -> None:
    state = lbfgs_state(2)
    for k in range(1, 4):
        step = np.array([1.0, float(k)])
        state = lbfgs_update(state, step, 2.0 * step)

    assert len(state["s_history"]) == 2
    npt.assert_allclose(state["s_history"][0], [1.0, 2.0])
    npt.assert_allclose(state["s_history"][1], [1.0, 3.0])


def test_lbfgs_update_skips_non_positive_curvature() -> None:
    state = lbfgs_state(3)

    state = lbfgs_update(state, np.array([1.0, 0.0]), np.array([-1.0, 0.0]))

    assert state["s_history"] == []
    assert state["y_history"] == []
    assert state["rho_history"] == []


def test_lbfgs_direction_without_memory_is_gradient() -> None:
    direction = lbfgs_direction(lbfgs_state(), np.array([1.0, 2.0]))

    npt.assert_allclose(direction, [1.0, 2.0], atol=0.0)


def test_lbfgs_direction_matches_dense_bfgs_after_one_pair() -> None:
    # Concave quadratic f(x) = -0.5 x^T A x + b^T x; maximising f means minimising -f,
    # whose Hessian is A, so the curvature pair is y = A s.
    matrix = np.array([[3.0, 0.5], [0.5, 1.0]])
    offset = np.array([1.0, -2.0])

    def gradient(x: np.ndarray) -> np.ndarray:
        return -matrix @ x + offset

    x_old = np.array([0.2, -0.4])
    x_new = np.array([0.9, 0.1])
    step = x_new - x_old
    state = lbfgs_update(lbfgs_state(), step, gradient(x_old) - gradient(x_new))
    assert len(state["s_history"]) == 1

    # Dense BFGS inverse Hessian for the minimisation of -f with H0 = gamma * I.
    rho = state["rho_history"][0]
    curvature = state["y_history"][0]
    gamma = float(step @ curvature / (curvature @ curvature))
    identity = np.eye(2)
    dense = (identity - rho * np.outer(step, curvature)) @ (gamma * identity) @ (
        identity - rho * np.outer(curvature, step)
    ) + rho * np.outer(step, step)

    grad_at_new = gradient(x_new)
    direction = lbfgs_direction(state, grad_at_new)

    npt.assert_allclose(direction, dense @ grad_at_new, atol=1e-10)
    assert float(direction @ grad_at_new) > 0.0
