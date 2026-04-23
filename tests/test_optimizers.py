"""Tests for optimizer helpers."""

import pathlib
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import optimalcontrol.optimizers as optimizers
from optimalcontrol.grape import ControlProblem


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
    monkeypatch.setattr(optimizers, "grape_gradient", grape_gradient)
    monkeypatch.setattr(optimizers, "grape_hessian", grape_hessian)


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
    monkeypatch.setattr(optimizers, "grape_gradient", lambda _cp, _wfm: np.ones_like(_wfm))
    monkeypatch.setattr(optimizers, "grape_hessian", lambda _cp, _wfm: None)

    with pytest.raises(ValueError, match="exact Hessian"):
        optimizers.newton_raphson(cp, wfm0)


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


def test_print_iteration_table_uses_fixed_width_row(capsys: pytest.CaptureFixture[str]) -> None:
    optimizers.print_iteration_table(12, 0.987654321, 0.0125, 3.5e-4, 6.75e-6)

    assert (
        capsys.readouterr().out
        == "    12  9.876543e-01  1.250000e-02  3.500000e-04  6.750000e-06\n"
    )


@pytest.mark.parametrize(
    "optimizer_name",
    ["gradient_ascent", "lbfgs_grape", "newton_raphson"],
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
