"""End-to-end integration tests for analytical, GRAPE, ensemble, and I/O paths."""

import pathlib

import numpy as np
import numpy.typing as npt

from optimalcontrol.grape import ControlProblem, grape_xy
from optimalcontrol.io import export_json, import_json
from optimalcontrol.operators import Ix, Iy, Iz, place_operator
from optimalcontrol.optimizers import lbfgs_grape, run_grape
from optimalcontrol.rope import inept_max_efficiency, rope_finite_efficiency, rope_waveform
from optimalcontrol.states import fidelity_real, normalise_hs, state_from_label


def _two_spin_transfer_problem(
    *,
    fidelity_mode: str = "real",
    pwr_levels: list[float] | None = None,
) -> ControlProblem:
    """Return a compact two-spin Iz -> 2IzSz transfer problem."""
    j_hz = 1.0
    iz_i = place_operator(Iz(), 0, 2)
    iz_s = place_operator(Iz(), 1, 2)
    drift_h = np.complex128(2.0 * np.pi * j_hz) * (iz_i @ iz_s)
    control_h = [
        place_operator(Ix(), 0, 2),
        place_operator(Iy(), 0, 2),
        place_operator(Ix(), 1, 2),
        place_operator(Iy(), 1, 2),
    ]
    return ControlProblem(
        drifts=[np.complex128(-1j) * drift_h],
        operators=[np.complex128(-1j) * operator for operator in control_h],
        rho_init=[normalise_hs(state_from_label("Iz", 2))],
        rho_targ=[normalise_hs(state_from_label("2IzSz", 2))],
        pulse_dt=0.25,
        pwr_levels=[1.0] * len(control_h) if pwr_levels is None else pwr_levels,
        freeze=None,
        fidelity_mode=fidelity_mode,
        basis="dense",
    )


def _random_waveform(
    seed: int,
    n_steps: int = 4,
    n_channels: int = 4,
) -> npt.NDArray[np.float64]:
    """Return a deterministic small GRAPE waveform."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.25, 0.25, size=(n_steps, n_channels)).astype(np.float64)


def test_analytical_rope_waveform_beats_inept_for_iz_to_2izsz() -> None:
    n = 1.0
    j_hz = 100.0
    duration = 10.0 / j_hz
    waveform = rope_waveform(T=duration, n=n, J_hz=j_hz, dt=duration / 64.0)
    rho_init = normalise_hs(state_from_label("Iz", 2))
    rho_targ = normalise_hs(state_from_label("2IzSz", 2))

    rope_fidelity = rope_finite_efficiency(T=duration, n=n, J_hz=j_hz)
    inept_fidelity = inept_max_efficiency(n=n, J_hz=j_hz)

    assert waveform["amplitude"].shape == waveform["phase"].shape
    np.testing.assert_allclose(fidelity_real(rho_init, rho_init), 1.0, rtol=1e-12)
    np.testing.assert_allclose(fidelity_real(rho_targ, rho_targ), 1.0, rtol=1e-12)
    assert rope_fidelity > inept_fidelity


def test_run_grape_random_iz_to_2izsz_transfer_improves_fidelity() -> None:
    cp = _two_spin_transfer_problem()
    wfm0 = _random_waveform(seed=2026)
    initial_fidelity = grape_xy(cp, wfm0)

    waveform, result = run_grape(cp, wfm0, m=4, tol_x=0.0, tol_g=0.0, max_iter=10)

    assert result.n_iter <= 10
    assert result.fidelity_final > initial_fidelity
    assert result.fidelity_final > 0.99
    np.testing.assert_allclose(waveform.data.T, result.wfm_final, rtol=1e-12)


def test_grape_xy_power_ensemble_returns_bounded_fidelity() -> None:
    cp = _two_spin_transfer_problem(fidelity_mode="abs2", pwr_levels=[0.9, 1.1])
    waveform = _random_waveform(seed=314)

    fidelity = grape_xy(cp, waveform)

    assert 0.0 <= fidelity <= 1.0


def test_lbfgs_checkpoint_resume_matches_uninterrupted_result(
    tmp_path: pathlib.Path,
) -> None:
    cp = _two_spin_transfer_problem()
    wfm0 = _random_waveform(seed=2026)
    checkpoint_path = tmp_path / "lbfgs-integration.json"

    uninterrupted = lbfgs_grape(cp, wfm0, m=4, tol_x=0.0, tol_g=0.0, max_iter=10)
    partial = lbfgs_grape(
        cp,
        wfm0,
        m=4,
        tol_x=0.0,
        tol_g=0.0,
        max_iter=5,
        checkpoint_path=str(checkpoint_path),
    )
    resumed = lbfgs_grape(
        cp,
        np.full_like(wfm0, 99.0),
        m=4,
        tol_x=0.0,
        tol_g=0.0,
        max_iter=10,
        checkpoint_path=str(checkpoint_path),
    )

    assert partial.reason == "max_iter"
    assert checkpoint_path.exists()
    np.testing.assert_allclose(resumed.wfm_final, uninterrupted.wfm_final, rtol=1e-5)
    np.testing.assert_allclose(resumed.fidelity_final, uninterrupted.fidelity_final, rtol=1e-5)


def test_waveform_json_round_trip_preserves_replay_fidelity(
    tmp_path: pathlib.Path,
) -> None:
    cp = _two_spin_transfer_problem()
    wfm0 = _random_waveform(seed=2718)
    waveform, _ = run_grape(cp, wfm0, m=4, tol_x=0.0, tol_g=0.0, max_iter=5)
    output_path = tmp_path / "waveform.json"

    export_json(waveform, output_path)
    loaded = import_json(output_path)

    original_fidelity = grape_xy(cp, waveform.data.T)
    loaded_fidelity = grape_xy(cp, loaded.data.T)
    np.testing.assert_allclose(loaded.data, waveform.data, rtol=1e-12)
    np.testing.assert_allclose(loaded_fidelity, original_fidelity, rtol=1e-10)
