"""Tests for waveform import/export helpers."""

import csv
import pathlib

import numpy as np

from optimalcontrol.grape import ControlProblem
from optimalcontrol.io import Waveform, export_csv, export_json, import_json, waveform_from_result
from optimalcontrol.optimizers import OptimResult


def _io_control_problem() -> ControlProblem:
    drift = np.zeros((1, 1), dtype=np.complex128)
    operator = np.zeros((1, 1), dtype=np.complex128)
    state = np.array([1.0], dtype=np.complex128)
    return ControlProblem(
        drifts=[drift],
        operators=[operator.copy(), operator.copy()],
        rho_init=[state],
        rho_targ=[state.copy()],
        pulse_dt=0.25,
        pwr_levels=[1.0, 2.0],
        freeze=None,
        basis="hilbert",
    )


def _waveform() -> Waveform:
    return Waveform(
        channels=["x", "y"],
        units="a.u.",
        times=np.array([0.0, 0.25, 0.5], dtype=np.float64),
        data=np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float64),
        metadata={"pulse_dt": 0.25, "labels": ["demo"]},
        problem_hash="abc123",
    )


def test_waveform_from_result_transposes_internal_grape_layout() -> None:
    cp = _io_control_problem()
    wfm0 = np.zeros((3, 2), dtype=np.float64)
    final = np.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]], dtype=np.float64)
    result = OptimResult(
        wfm_final=final,
        fidelity_final=0.75,
        n_iter=4,
        n_feval=9,
        converged=True,
        reason="grad_tol",
        history=[0.1, 0.75],
    )

    waveform = waveform_from_result(cp, wfm0, result)

    assert waveform.channels == ["x", "y"]
    assert waveform.units == "a.u."
    np.testing.assert_allclose(waveform.times, np.array([0.0, 0.25, 0.5]))
    np.testing.assert_allclose(waveform.data, final.T)
    assert waveform.metadata["fidelity_final"] == 0.75
    assert len(waveform.problem_hash) == 64


def test_json_export_import_round_trips_waveform(tmp_path: pathlib.Path) -> None:
    waveform = _waveform()
    path = tmp_path / "waveform.json"

    export_json(waveform, path)
    loaded = import_json(path)

    assert loaded.channels == waveform.channels
    assert loaded.units == waveform.units
    assert loaded.metadata == waveform.metadata
    assert loaded.problem_hash == waveform.problem_hash
    np.testing.assert_allclose(loaded.times, waveform.times, rtol=1e-12)
    np.testing.assert_allclose(loaded.data, waveform.data, rtol=1e-12)


def test_csv_export_writes_header_and_one_row_per_time_step(
    tmp_path: pathlib.Path,
) -> None:
    waveform = _waveform()
    path = tmp_path / "waveform.csv"

    export_csv(waveform, path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["time", "x", "y"]
    assert len(rows) == waveform.times.shape[0] + 1
    assert rows[1] == ["0", "0.10000000000000001", "-0.10000000000000001"]
