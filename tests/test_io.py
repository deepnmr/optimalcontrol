"""Tests for waveform import/export helpers."""

import csv
import pathlib

import numpy as np

from optimalcontrol.grape import ControlProblem
from optimalcontrol.io import (
    Waveform,
    export_bruker,
    export_csv,
    export_json,
    fapt_import,
    heterodyne_transform,
    import_jcamp_dx,
    import_json,
    waveform_from_result,
)
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


def test_export_bruker_writes_documented_minimal_shape_stub(
    tmp_path: pathlib.Path,
) -> None:
    waveform = _waveform()
    path = tmp_path / "shape.stub"

    export_bruker(waveform, path)

    text = path.read_text(encoding="utf-8")
    assert "##DATA TYPE= Shape Data" in text
    assert "##NPOINTS= 3" in text
    assert "##CHANNELS= amplitude,phase_deg" in text
    assert "not production-ready" in text
    assert "##XYPOINTS= (XY..XY)" in text


def test_import_jcamp_dx_parses_minimal_xy_table(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "shape.dx"
    path.write_text(
        "\n".join(
            [
                "##TITLE= demo",
                "##JCAMP-DX= 5.00",
                "##UNITS= a.u.",
                "##NPOINTS= 3",
                "##CHANNELS= x,y",
                "##$OPTIMALCONTROL_TIMES= 0 0.25 0.5",
                "##$OPTIMALCONTROL_PROBLEM_HASH= abc123",
                "##XYPOINTS= (XY..XY)",
                "1.0, 0.0",
                "0.0, 1.0",
                "-1.0, 0.0",
                "##END=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = import_jcamp_dx(path)

    assert loaded.channels == ["x", "y"]
    assert loaded.units == "a.u."
    assert loaded.problem_hash == "abc123"
    np.testing.assert_allclose(loaded.times, np.array([0.0, 0.25, 0.5]), rtol=1e-12)
    np.testing.assert_allclose(
        loaded.data,
        np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        rtol=1e-12,
    )


def test_heterodyne_transform_rotates_xy_envelope_by_carrier() -> None:
    waveform = Waveform(
        channels=["x", "y"],
        units="a.u.",
        times=np.array([0.0, 0.25, 0.5], dtype=np.float64),
        data=np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64),
        metadata={},
        problem_hash="abc123",
    )

    shifted = heterodyne_transform(waveform, carrier_hz=1.0)

    np.testing.assert_allclose(shifted.times, waveform.times, rtol=1e-12)
    np.testing.assert_allclose(
        shifted.data,
        np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert shifted.metadata["heterodyne_carrier_hz"] == 1.0


def test_fapt_import_parses_frequency_amplitude_phase_time_table(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "pulse.fapt"
    path.write_text(
        "\n".join(
            [
                "# frequency_hz amplitude phase_rad time_s",
                "100.0, 0.25, 0.0, 0.0",
                "110.0, 0.50, 1.5707963267948966, 0.25",
                "120.0, 0.75, 3.141592653589793, 0.50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = fapt_import(path)

    assert loaded.channels == ["frequency_hz", "amplitude", "phase_rad"]
    assert loaded.units == "Hz/a.u./rad"
    np.testing.assert_allclose(loaded.times, np.array([0.0, 0.25, 0.5]), rtol=1e-12)
    np.testing.assert_allclose(
        loaded.data,
        np.array(
            [
                [100.0, 110.0, 120.0],
                [0.25, 0.50, 0.75],
                [0.0, 1.5707963267948966, 3.141592653589793],
            ],
            dtype=np.float64,
        ),
        rtol=1e-12,
    )
    assert str(loaded.problem_hash).startswith("fapt:")
