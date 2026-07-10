"""Tests for optimizer helpers."""

import pathlib
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import optimalcontrol
import optimalcontrol.optimizers as optimizers
from optimalcontrol.grape import ControlProblem, grape_xy
from optimalcontrol.operators import Ix, Iy, Iz
from optimalcontrol.states import normalise_hs


def _optimizer_control_problem(n_steps: int, n_channels: int) -> ControlProblem:
    drift = np.zeros((1, 1), dtype=np.complex128)
    operator = np.zeros((1, 1), dtype=np.complex128)
    state = np.array([1.0], dtype=np.complex128)
    return ControlProblem(
        drifts=[drift],
        operators=[operator.copy() for _ in range(n_channels)],
        rho_init=[state],
        rho_targ=[state.copy()],
        pulse_dt=1.0,
        pwr_levels=[1.0] * n_channels,
        freeze=None,
        basis="hilbert",
    )


def _one_spin_iz_to_ix_problem(n_steps: int) -> ControlProblem:
    """Return a small one-spin product-operator transfer problem."""
    return ControlProblem(
        drifts=[np.zeros((2, 2), dtype=np.complex128)],
        operators=[np.complex128(-1j) * Ix(), np.complex128(-1j) * Iy()],
        rho_init=[normalise_hs(Iz())],
        rho_targ=[normalise_hs(Ix())],
        pulse_dt=0.1,
        pwr_levels=[1.0, 1.0],
        freeze=None,
        fidelity_mode="real",
        basis="dense",
    )


def _patch_quadratic_objective(
    monkeypatch: pytest.MonkeyPatch,
    optimum: npt.NDArray[np.float64],
    curvature: npt.NDArray[np.float64],
    hessian_override: npt.NDArray[np.float64] | None = None,
) -> None:
    optimum_flat = optimum.reshape(-1).copy()
    curvature_matrix = np.asarray(curvature, dtype=np.float64)
    hessian_matrix = (
        np.asarray(hessian_override, dtype=np.float64)
        if hessian_override is not None
        else -curvature_matrix
    )

    def grape_xy(_: ControlProblem, wfm: npt.NDArray[np.float64]) -> float:
        diff = wfm.reshape(-1) - optimum_flat
        return float(1.0 - 0.5 * diff @ curvature_matrix @ diff)

    def grape_gradient(
        _: ControlProblem,
        wfm: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        diff = wfm.reshape(-1) - optimum_flat
        return np.asarray((-(curvature_matrix @ diff)).reshape(wfm.shape), dtype=np.float64)

    def grape_hessian(
        _: ControlProblem,
        __: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return hessian_matrix.copy()

    monkeypatch.setattr(optimizers, "grape_xy", grape_xy)
    monkeypatch.setattr(
        optimizers,
        "grape_xy_and_gradient",
        lambda cp, wfm: (grape_xy(cp, wfm), grape_gradient(cp, wfm)),
    )
    monkeypatch.setattr(optimizers, "grape_hessian", grape_hessian)


def test_lbfgs_grape_converges_on_negative_norm_quadratic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimum = np.array([[0.35, -0.2], [0.75, -0.45]], dtype=np.float64)
    optimum_flat = optimum.reshape(-1).copy()

    def grape_xy_quadratic(
        _: ControlProblem,
        wfm: npt.NDArray[np.float64],
    ) -> float:
        diff = wfm.reshape(-1) - optimum_flat
        return float(-diff @ diff)

    def grape_gradient_quadratic(
        _: ControlProblem,
        wfm: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        diff = wfm.reshape(-1) - optimum_flat
        return np.asarray((-2.0 * diff).reshape(wfm.shape), dtype=np.float64)

    monkeypatch.setattr(optimizers, "grape_xy", grape_xy_quadratic)
    monkeypatch.setattr(
        optimizers,
        "grape_xy_and_gradient",
        lambda cp, wfm: (grape_xy_quadratic(cp, wfm), grape_gradient_quadratic(cp, wfm)),
    )
    cp = _optimizer_control_problem(n_steps=2, n_channels=2)
    wfm0 = np.zeros_like(optimum)

    result = optimizers.lbfgs_grape(cp, wfm0, m=4, tol_x=1e-12, tol_g=1e-12, max_iter=10)

    assert result.converged is True
    np.testing.assert_allclose(result.wfm_final, optimum, atol=1e-5, rtol=0.0)


def test_lbfgs_grape_improves_one_spin_iz_to_ix_transfer() -> None:
    cp = _one_spin_iz_to_ix_problem(n_steps=4)
    wfm0 = np.zeros((4, 2), dtype=np.float64)

    initial_fidelity = grape_xy(cp, wfm0)
    result = optimizers.lbfgs_grape(cp, wfm0, m=4, tol_x=0.0, tol_g=0.0, max_iter=20)

    assert result.n_iter <= 20
    assert result.fidelity_final > initial_fidelity + 0.9
    assert result.fidelity_final > 0.999


def test_lbfgs_checkpoint_resume_five_plus_five_matches_ten_iterations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    optimum = np.array([[1.0, -0.5]], dtype=np.float64)
    curvature = np.eye(optimum.size, dtype=np.float64)
    _patch_quadratic_objective(monkeypatch, optimum, curvature)
    monkeypatch.setattr(optimizers, "line_search_cubic", lambda *_args, **_kwargs: 0.25)
    cp = _optimizer_control_problem(n_steps=1, n_channels=2)
    wfm0 = np.zeros_like(optimum)

    continuous = optimizers.lbfgs_grape(
        cp,
        wfm0,
        m=4,
        tol_x=0.0,
        tol_g=0.0,
        max_iter=10,
    )
    checkpoint_path = tmp_path / "lbfgs-five-plus-five.json"
    partial = optimizers.lbfgs_grape(
        cp,
        wfm0,
        m=4,
        tol_x=0.0,
        tol_g=0.0,
        max_iter=5,
        checkpoint_path=str(checkpoint_path),
    )
    resumed = optimizers.lbfgs_grape(
        cp,
        np.full_like(wfm0, 99.0),
        m=4,
        tol_x=0.0,
        tol_g=0.0,
        max_iter=10,
        checkpoint_path=str(checkpoint_path),
    )

    assert partial.n_iter == 5
    assert partial.reason == "max_iter"
    assert resumed.n_iter == continuous.n_iter == 10
    np.testing.assert_allclose(resumed.wfm_final, continuous.wfm_final, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(resumed.fidelity_final, continuous.fidelity_final, rtol=1e-6)


def test_optimizer_rejects_checkpoint_for_different_control_problem(
    tmp_path: pathlib.Path,
) -> None:
    cp = _one_spin_iz_to_ix_problem(n_steps=2)
    waveform = np.zeros((2, 2), dtype=np.float64)
    checkpoint_path = tmp_path / "different-problem.json"
    optimizers.lbfgs_grape(cp, waveform, max_iter=0, checkpoint_path=str(checkpoint_path))

    with pytest.raises(ValueError, match="optimizer or control problem"):
        optimizers.newton_raphson(
            cp,
            waveform,
            max_iter=0,
            checkpoint_path=str(checkpoint_path),
        )
    with pytest.raises(ValueError, match="waveform shape"):
        optimizers.lbfgs_grape(
            cp,
            np.zeros((3, 2), dtype=np.float64),
            max_iter=0,
            checkpoint_path=str(checkpoint_path),
        )

    cp.rho_targ = [normalise_hs(Iz())]

    with pytest.raises(ValueError, match="optimizer or control problem"):
        optimizers.lbfgs_grape(
            cp,
            waveform,
            max_iter=0,
            checkpoint_path=str(checkpoint_path),
        )


def test_newton_raphson_converges_on_concave_quadratic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimum = np.array([[0.4], [-0.25]], dtype=np.float64)
    curvature = np.eye(optimum.size, dtype=np.float64)
    _patch_quadratic_objective(monkeypatch, optimum, curvature)
    cp = _optimizer_control_problem(n_steps=2, n_channels=1)
    wfm0 = np.zeros_like(optimum)

    result = optimizers.newton_raphson(cp, wfm0, tol_x=1e-10, tol_g=1e-10, max_iter=5)

    assert result.converged is True
    assert result.reason == "grad_tol"
    assert result.n_iter == 1
    np.testing.assert_allclose(result.wfm_final, optimum, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result.fidelity_final, 1.0, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("rfo", [False, True])
def test_newton_step_regularises_indefinite_hessian_model(rfo: bool) -> None:
    gradient = np.array([[1.0, 0.0]], dtype=np.float64)
    indefinite_hessian = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    step = optimizers._newton_step(gradient, indefinite_hessian, regularise=True, rfo=rfo)

    assert np.all(np.isfinite(step))
    assert step[0, 0] > 0.0
    np.testing.assert_allclose(step[0, 1], 0.0, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("rfo", [False, True])
def test_newton_raphson_regularises_semidefinite_hessian_model(
    monkeypatch: pytest.MonkeyPatch,
    rfo: bool,
) -> None:
    optimum = np.array([[1.0, 0.0]], dtype=np.float64)
    curvature = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    _patch_quadratic_objective(monkeypatch, optimum, curvature)
    cp = _optimizer_control_problem(n_steps=1, n_channels=2)
    wfm0 = np.zeros_like(optimum)

    result = optimizers.newton_raphson(cp, wfm0, regularise=True, rfo=rfo, max_iter=5)

    assert result.converged is True
    assert result.reason in {"grad_tol", "step_tol"}
    assert result.fidelity_final > 0.999999
    np.testing.assert_allclose(result.wfm_final[0, 1], 0.0, atol=1e-12, rtol=1e-12)


def test_newton_raphson_rejects_missing_exact_hessian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = _optimizer_control_problem(n_steps=51, n_channels=1)
    wfm0 = np.zeros((51, 1), dtype=np.float64)

    monkeypatch.setattr(optimizers, "grape_xy", lambda _cp, _wfm: 0.0)
    monkeypatch.setattr(
        optimizers,
        "grape_xy_and_gradient",
        lambda _cp, _wfm: (0.0, np.ones_like(_wfm)),
    )
    monkeypatch.setattr(optimizers, "grape_hessian", lambda _cp, _wfm: None)

    with pytest.raises(ValueError, match="exact Hessian"):
        optimizers.newton_raphson(cp, wfm0)


def test_lbfgs_grape_attaches_trajectory_when_requested() -> None:
    cp = _optimizer_control_problem(n_steps=3, n_channels=2)
    wfm0 = np.zeros((3, 2), dtype=np.float64)

    result = optimizers.lbfgs_grape(cp, wfm0, produce_trajectory=True)

    assert result.trajectory is not None
    assert len(result.trajectory) == wfm0.shape[0] + 1
    for state in result.trajectory:
        np.testing.assert_allclose(state, cp.rho_init[0], atol=1e-12, rtol=1e-12)


def test_lbfgs_grape_attaches_first_ensemble_member_trajectory() -> None:
    cp = _optimizer_control_problem(n_steps=3, n_channels=2)
    cp.drifts = [cp.drifts[0], cp.drifts[0] - 1j * np.eye(1, dtype=np.complex128)]
    wfm0 = np.zeros((3, 2), dtype=np.float64)

    result = optimizers.lbfgs_grape(cp, wfm0, max_iter=0, produce_trajectory=True)

    assert result.trajectory is not None
    assert len(result.trajectory) == wfm0.shape[0] + 1
    for state in result.trajectory:
        np.testing.assert_allclose(state, cp.rho_init[0], atol=1e-12, rtol=1e-12)


def test_newton_raphson_attaches_trajectory_when_requested() -> None:
    cp = _optimizer_control_problem(n_steps=2, n_channels=1)
    wfm0 = np.zeros((2, 1), dtype=np.float64)

    result = optimizers.newton_raphson(cp, wfm0, produce_trajectory=True)

    assert result.trajectory is not None
    assert len(result.trajectory) == wfm0.shape[0] + 1
    for state in result.trajectory:
        np.testing.assert_allclose(state, cp.rho_init[0], atol=1e-12, rtol=1e-12)


def test_run_grape_returns_waveform_result_and_checkpoint(
    tmp_path: pathlib.Path,
) -> None:
    cp = _optimizer_control_problem(n_steps=3, n_channels=2)
    checkpoint_path = tmp_path / "run-grape-checkpoint.json"
    cp.checkpoint_path = str(checkpoint_path)
    wfm0 = np.zeros((3, 2), dtype=np.float64)

    waveform, result = optimizers.run_grape(cp, wfm0, produce_trajectory=True)

    assert optimalcontrol.run_grape is optimizers.run_grape
    assert result.converged is True
    assert result.trajectory is not None
    assert checkpoint_path.exists()
    np.testing.assert_allclose(waveform.data, result.wfm_final.T, atol=1e-12, rtol=1e-12)
    loaded_waveform, loaded_history = optimizers.load_checkpoint(str(checkpoint_path))
    np.testing.assert_allclose(loaded_waveform, result.wfm_final, atol=1e-12, rtol=1e-12)
    assert loaded_history == pytest.approx(result.history)


def test_run_grape_rejects_unknown_method() -> None:
    cp = _optimizer_control_problem(n_steps=1, n_channels=1)
    wfm0 = np.zeros((1, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="method"):
        optimizers.run_grape(cp, wfm0, method="unsupported")


def test_run_grape_supports_callable_penalties() -> None:
    cp = _optimizer_control_problem(n_steps=1, n_channels=1)

    def penalty(wfm: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
        return 0.0, np.zeros_like(wfm)

    penalty.__optimalcontrol_provenance__ = "zero-penalty-v1"
    cp.penalties = [penalty]

    waveform, result = optimizers.run_grape(cp, np.zeros((1, 1)), max_iter=0)

    assert len(waveform.problem_hash) == 64
    assert result.fidelity_final == pytest.approx(1.0)


def _run_optimizer(
    optimizer_name: str,
    cp: ControlProblem,
    wfm0: npt.NDArray[np.float64],
    *,
    max_iter: int,
    checkpoint_path: str | None = None,
) -> optimizers.OptimResult:
    optimizer = cast(Callable[..., optimizers.OptimResult], getattr(optimizers, optimizer_name))
    kwargs: dict[str, object] = {
        "tol_x": 0.0,
        "tol_g": 0.0,
        "max_iter": max_iter,
        "checkpoint_path": checkpoint_path,
    }
    if optimizer_name == "lbfgs_grape":
        kwargs["m"] = 4
    return optimizer(cp, wfm0, **kwargs)


def test_save_load_checkpoint_round_trip(tmp_path: pathlib.Path) -> None:
    waveform = np.array([[0.25, -0.5], [0.75, 1.0]], dtype=np.float64)
    history = [0.1, 0.2, 0.35]
    checkpoint_path = tmp_path / "optimizer-checkpoint.json"

    optimizers.save_checkpoint(str(checkpoint_path), waveform, history)
    loaded_waveform, loaded_history = optimizers.load_checkpoint(str(checkpoint_path))

    np.testing.assert_allclose(loaded_waveform, waveform, atol=1e-12, rtol=1e-12)
    assert loaded_history == pytest.approx(history)

    cp = _optimizer_control_problem(n_steps=2, n_channels=2)
    resumed = optimizers.lbfgs_grape(
        cp,
        np.zeros_like(waveform),
        max_iter=0,
        checkpoint_path=str(checkpoint_path),
    )
    assert resumed.history == pytest.approx([1.0])


@pytest.mark.parametrize(
    "optimizer_name",
    ["lbfgs_grape", "newton_raphson"],
)
def test_optimizer_checkpoint_resume_matches_continuous_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    optimizer_name: str,
) -> None:
    optimum = np.array([[1.0, -0.5]], dtype=np.float64)
    curvature = np.eye(optimum.size, dtype=np.float64)
    _patch_quadratic_objective(monkeypatch, optimum, curvature)
    monkeypatch.setattr(optimizers, "line_search_cubic", lambda *_args, **_kwargs: 0.25)

    cp = _optimizer_control_problem(n_steps=1, n_channels=2)
    wfm0 = np.zeros_like(optimum)
    continuous = _run_optimizer(optimizer_name, cp, wfm0, max_iter=12)

    checkpoint_path = tmp_path / f"{optimizer_name}.json"
    partial = _run_optimizer(
        optimizer_name,
        cp,
        wfm0,
        max_iter=10,
        checkpoint_path=str(checkpoint_path),
    )
    resumed = _run_optimizer(
        optimizer_name,
        cp,
        np.full_like(optimum, 99.0),
        max_iter=12,
        checkpoint_path=str(checkpoint_path),
    )

    assert partial.reason == "max_iter"
    assert checkpoint_path.exists()
    assert resumed.reason == continuous.reason
    assert resumed.n_iter == continuous.n_iter
    assert resumed.n_feval == continuous.n_feval
    np.testing.assert_allclose(resumed.wfm_final, continuous.wfm_final, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        np.array(resumed.history, dtype=np.float64),
        np.array(continuous.history, dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert resumed.fidelity_final == pytest.approx(continuous.fidelity_final)
