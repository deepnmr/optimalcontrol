"""End-to-end check of the MCP tool functions (called directly, no transport)."""

import asyncio
import csv
import json
import math

import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from optimalcontrol.crop import crop_eta
from optimalcontrol.io import export_bruker_shape
from optimalcontrol.mcp_server import (
    _bloch_vector,
    _transverse_rotation,
    bloch_offset_profile,
    bloch_trajectory,
    convert_waveform,
    crop_pulse,
    design_grape_pulse,
    design_seedless_pulse,
    inspect_waveform,
    rope_pulse,
    server,
    waveform_spectrogram,
)
from optimalcontrol.rope import rope_finite_efficiency


def _grape_inversion(tmp_path, **overrides):
    kwargs = dict(
        init=["Iz"],
        targ=["-Iz"],
        rf_max_hz=10_000.0,
        duration_s=120e-6,
        n_steps=16,
        offset_lo_hz=-2000.0,
        offset_hi_hz=2000.0,
        n_offsets=3,
        b1_scales=[0.9, 1.0, 1.1],
        out_dir=str(tmp_path),
        name="inv",
        max_iter=100,
    )
    kwargs.update(overrides)
    return design_grape_pulse(**kwargs)


_SMALL_DESIGN = {
    "spectrometer_mhz": 600.0,
    "carrier_ppm": 0.0,
    "rf_max_hz": 1000.0,
    "duration_s": 1e-4,
    "n_steps": 4,
    "max_iter": 5,
    "n_seeds": 1,
}


def _call(name, arguments):
    async def run():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def _find_enum_containing(node, value):
    if isinstance(node, dict):
        if value in node.get("enum", []):
            return node["enum"]
        return next((hit for v in node.values() if (hit := _find_enum_containing(v, value))), None)
    if isinstance(node, list):
        return next((hit for v in node if (hit := _find_enum_containing(v, value))), None)
    return None


def test_mcp_client_lists_all_tools():
    async def run():
        async with Client(server) as client:
            return await client.list_tools()

    tools = asyncio.run(run())
    assert sorted(tool.name for tool in tools.tools) == [
        "bloch_offset_profile",
        "bloch_trajectory",
        "convert_waveform",
        "crop_pulse",
        "design_grape_pulse",
        "design_seedless_pulse",
        "inspect_waveform",
        "rope_pulse",
        "waveform_spectrogram",
    ]
    design = next(tool for tool in tools.tools if tool.name == "design_seedless_pulse")
    assert "s2s" in _find_enum_containing(design.input_schema, "s2s")


def test_mcp_client_rejects_b1_weights_not_summing_to_one(tmp_path):
    result = _call(
        "design_seedless_pulse",
        {
            **_SMALL_DESIGN,
            "bands": [{"ppm_lo": 0.0, "ppm_hi": 0.0, "restraint": "xycite", "n_offsets": 1}],
            "out_dir": str(tmp_path),
            "b1_scales": [1.0],
            "b1_weights": [0.5],
        },
    )
    assert result.is_error is True
    assert "b1_weights must sum to 1" in result.content[0].text


def test_mcp_client_rejects_zero_seeds(tmp_path):
    result = _call(
        "design_seedless_pulse",
        {
            **_SMALL_DESIGN,
            "bands": [{"ppm_lo": 0.0, "ppm_hi": 0.0, "restraint": "xycite", "n_offsets": 1}],
            "out_dir": str(tmp_path),
            "n_seeds": 0,
        },
    )
    assert result.is_error is True
    assert "n_seeds must be >= 1" in result.content[0].text


def test_mcp_client_rejects_s2s_band_without_init_targ(tmp_path):
    result = _call(
        "design_seedless_pulse",
        {
            **_SMALL_DESIGN,
            "bands": [{"ppm_lo": 0.0, "ppm_hi": 0.0, "restraint": "s2s", "n_offsets": 1}],
            "out_dir": str(tmp_path),
        },
    )
    assert result.is_error is True
    assert "s2s restraint requires init and targ" in result.content[0].text


def test_mcp_client_reports_missing_design_file(tmp_path):
    missing = tmp_path / "missing.json"
    result = _call(
        "bloch_offset_profile",
        {
            "design_json": str(missing),
            "offset_lo_hz": -100.0,
            "offset_hi_hz": 100.0,
        },
    )
    assert result.is_error is True
    assert str(missing) in result.content[0].text


def test_design_then_profile(tmp_path):
    result = design_seedless_pulse(
        spectrometer_mhz=600.0,
        carrier_ppm=0.0,
        rf_max_hz=10_000.0,
        duration_s=120e-6,
        n_steps=20,
        bands=[
            {
                "ppm_lo": -8.0,
                "ppm_hi": 8.0,
                "restraint": "s2s",
                "n_offsets": 5,
                "init": "-y",
                "targ": "y",
            }
        ],
        out_dir=str(tmp_path),
        name="inv",
        n_seeds=1,
        max_iter=60,
    )

    assert (tmp_path / "inv.shape").exists()
    design = json.loads((tmp_path / "inv.json").read_text())
    assert len(design["phases_rad"]) == 20
    assert "band0:s2s" in result["worst_case"]
    assert result["worst_case"]["band0:s2s"] > 0.9

    profile = bloch_offset_profile(
        design_json=result["design_json"],
        offset_lo_hz=-2000.0,
        offset_hi_hz=2000.0,
        n_points=11,
        init="-y",
    )
    assert len(profile["mz"]) == 11
    # on-resonance the -y -> y transfer must hold in the Bloch model too
    assert profile["my"][5] > 0.9


@pytest.mark.parametrize(
    ("band_fields", "duration_s", "expected"),
    [
        ({"restraint": "s2s", "init": "z", "targ": "-z"}, 0.0005, 0.0),
        ({"restraint": "xycite"}, 0.00025, 2**-0.5),
    ],
)
def test_design_worst_case_covers_b1_ensemble(tmp_path, band_fields, duration_s, expected):
    # A single on-resonance step is phase-independent for these z-based metrics.
    result = design_seedless_pulse(
        spectrometer_mhz=600.0,
        carrier_ppm=0.0,
        rf_max_hz=1000.0,
        duration_s=duration_s,
        n_steps=1,
        bands=[{"ppm_lo": 0.0, "ppm_hi": 0.0, "n_offsets": 1, **band_fields}],
        out_dir=str(tmp_path),
        n_seeds=1,
        max_iter=1,
        b1_scales=[0.5, 1.0, 1.5],
        b1_weights=[0.25, 0.5, 0.25],
    )

    key = f"band0:{band_fields['restraint']}"
    assert result["worst_case"][key] == pytest.approx(expected, abs=5e-5)
    design = json.loads((tmp_path / "pulse.json").read_text())
    assert design["worst_case"][key] == pytest.approx(expected, abs=1e-12)


def test_grape_design_writes_files_and_agrees_with_bloch(tmp_path):
    result = _grape_inversion(tmp_path)

    assert (tmp_path / "inv.shape").exists()
    design = json.loads((tmp_path / "inv.json").read_text())
    assert design["kind"] == "grape"
    waveform = np.asarray(design["waveform_xy"])
    assert waveform.shape == (16, 2)
    # hard amplitude cap honoured after clipping
    assert np.max(np.hypot(waveform[:, 0], waveform[:, 1])) <= 1.0 + 1e-12
    assert design["spec"]["init"] == ["Iz"] and design["spec"]["b1_scales"] == [0.9, 1.0, 1.1]

    assert result["fidelity"] > 0.95
    # the independent Bloch model on the design grid must reproduce the GRAPE fidelity
    assert result["bloch_mean"] == pytest.approx(result["fidelity"], abs=2e-6)
    assert result["bloch_worst"] <= result["bloch_mean"]
    assert result["bloch_worst_b1_scale"] in (0.9, 1.0, 1.1)
    assert -2000.0 <= result["bloch_worst_offset_hz"] <= 2000.0
    assert result["n_iter"] <= 100
    assert isinstance(result["converged"], bool)
    assert result["dt_s"] == pytest.approx(7.5e-6)


def test_grape_design_universal_rotation_pairs(tmp_path):
    result = _grape_inversion(
        tmp_path,
        init=["Ix", "Iy", "Iz"],
        targ=["Ix", "-Iy", "-Iz"],
        offset_lo_hz=-1000.0,
        offset_hi_hz=1000.0,
        b1_scales=None,
        duration_s=150e-6,
    )
    assert result["fidelity"] > 0.9
    assert result["bloch_mean"] == pytest.approx(result["fidelity"], abs=2e-6)
    # a 180x UR must also carry -y -> y in the Bloch model
    profile = bloch_offset_profile(result["design_json"], 0.0, 0.0, 1, init="-y", targ="y")
    assert profile["transfer"][0] > 0.9


def test_grape_design_broadband_universal_rotation_has_no_wrong_sign_region(tmp_path):
    # A raw random start leaves this at fidelity ~0.35 with bloch_worst = -1/3.
    result = _grape_inversion(
        tmp_path,
        init=["Ix", "Iy", "Iz"],
        targ=["Ix", "Iz", "-Iy"],
        rf_max_hz=10_000.0,
        duration_s=1.0e-3,
        n_steps=100,
        offset_lo_hz=-10_000.0,
        offset_hi_hz=10_000.0,
        n_offsets=21,
        max_iter=100,
    )
    assert result["fidelity"] > 0.97
    assert result["bloch_worst"] > 0.8
    assert not any("wrong-sign" in warning for warning in result["warnings"])


@pytest.mark.parametrize(
    ("init", "targ", "expected"),
    [
        (["x", "y", "z"], ["x", "-y", "-z"], ("x", 180.0)),
        (["x", "y", "z"], ["x", "z", "-y"], ("x", 90.0)),
        (["z", "x", "y"], ["x", "-z", "y"], ("y", 90.0)),
        (["x", "y", "z"], ["y", "-x", "z"], None),  # about z
        (["x", "y", "z"], ["x", "y", "z"], None),  # identity
        (["x", "y", "z"], ["x", "y", "-z"], None),  # reflection
        (["z"], ["-z"], None),
    ],
)
def test_transverse_rotation_detection(init, targ, expected):
    pairs = [(_bloch_vector(a), _bloch_vector(b)) for a, b in zip(init, targ, strict=True)]
    found = _transverse_rotation(pairs)
    if expected is None:
        assert found is None
    else:
        assert found is not None
        assert found[0] == expected[0] and found[1] == pytest.approx(expected[1])


def test_grape_design_newton_method_runs(tmp_path):
    result = _grape_inversion(
        tmp_path,
        n_steps=8,
        offset_lo_hz=-1000.0,
        offset_hi_hz=1000.0,
        method="newton",
        max_iter=10,
        b1_scales=None,
    )
    assert result["n_iter"] <= 10
    assert result["bloch_mean"] == pytest.approx(result["fidelity"], abs=2e-6)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"targ": ["-Iz", "Iz"]}, "init and targ must have equal length"),
        ({"init": []}, "init and targ must be non-empty"),
        ({"init": ["Sz"]}, "state label must be one of Ix, Iy, Iz"),
        ({"b1_scales": [0.9, 1.1]}, "b1_scales must have 1 or at least 3 entries"),
        ({"b1_scales": [1.0, 0.0, 1.1]}, r"b1_scales\[1\] must be positive"),
        ({"n_offsets": 0}, "n_offsets must be >= 1"),
        ({"offset_hi_hz": -3000.0}, "offset_hi_hz must be >= offset_lo_hz"),
        ({"rf_max_hz": 0.0}, "rf_max_hz must be positive"),
        ({"duration_s": -1.0}, "duration_s must be positive"),
        ({"n_steps": 0}, "n_steps must be >= 1"),
        ({"method": "adam"}, "method must be 'lbfgs' or 'newton'"),
    ],
)
def test_grape_design_rejects_invalid_inputs(tmp_path, overrides, message):
    with pytest.raises(ToolError, match=message):
        _grape_inversion(tmp_path, **overrides)
    assert not (tmp_path / "inv.json").exists()


def test_profile_accepts_grape_json_with_targ_and_b1_grid(tmp_path):
    result = _grape_inversion(tmp_path)
    profile = bloch_offset_profile(
        design_json=result["design_json"],
        offset_lo_hz=-2000.0,
        offset_hi_hz=2000.0,
        n_points=3,
        init="Iz",
        targ="-Iz",
        b1_scales=[0.9, 1.0, 1.1],
    )
    assert profile["transfer"] == [-m for m in profile["mz"]]
    summary = profile["summary"]
    assert summary["b1_scales"] == [0.9, 1.0, 1.1]
    assert summary["worst_transfer"] <= summary["mean_transfer"]
    # the same 3 x 3 grid is what the design tool averaged
    assert summary["mean_transfer"] == pytest.approx(result["fidelity"], abs=2e-6)
    with pytest.raises(ToolError, match="b1_scales requires targ"):
        bloch_offset_profile(result["design_json"], -1.0, 1.0, 3, b1_scales=[1.0])


def test_profile_keeps_seedless_contract_and_accepts_targ(tmp_path):
    design = design_seedless_pulse(
        spectrometer_mhz=600.0,
        carrier_ppm=0.0,
        rf_max_hz=10_000.0,
        duration_s=120e-6,
        n_steps=16,
        bands=[
            {
                "ppm_lo": -2.0,
                "ppm_hi": 2.0,
                "restraint": "s2s",
                "n_offsets": 3,
                "init": "z",
                "targ": "-z",
            }
        ],
        out_dir=str(tmp_path),
        n_seeds=1,
        max_iter=40,
    )
    plain = bloch_offset_profile(design["design_json"], -1200.0, 1200.0, 5)
    assert set(plain) == {"offsets_hz", "mx", "my", "mz"}
    assert len(plain["mz"]) == 5
    with_targ = bloch_offset_profile(design["design_json"], -1200.0, 1200.0, 5, targ="-z")
    assert with_targ["mz"] == plain["mz"]
    assert with_targ["summary"]["worst_transfer"] == pytest.approx(
        min(with_targ["transfer"]), abs=1e-4
    )
    info = inspect_waveform(design["design_json"])
    assert info["format"] == "seedless"
    assert info["constant_amplitude"] is True
    assert info["max_amplitude"] == pytest.approx(1.0)
    assert info["rf_max_hz"] == 10_000.0


def test_inspect_and_convert_round_trip_across_formats(tmp_path):
    result = _grape_inversion(tmp_path)
    design_info = inspect_waveform(result["design_json"])
    assert design_info["format"] == "grape"
    assert design_info["n_steps"] == 16
    assert design_info["duration_s"] == pytest.approx(120e-6)
    assert design_info["max_amplitude"] == pytest.approx(1.0)
    assert design_info["max_rf_hz"] == pytest.approx(10_000.0)

    shape_info = inspect_waveform(result["shape_file"])
    assert shape_info["format"] == "bruker"
    for key in ("n_steps", "duration_s", "rf_max_hz", "mean_amplitude", "phase_max_deg"):
        assert shape_info[key] == pytest.approx(design_info[key])

    csv_out = convert_waveform(result["design_json"], str(tmp_path / "w.csv"), "csv")
    assert csv_out["format"] == "csv" and csv_out["source_format"] == "grape"
    with open(tmp_path / "w.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["time", "x", "y"] and len(rows) == 17

    json_out = convert_waveform(str(tmp_path / "w.csv"), str(tmp_path / "w.json"), "json")
    assert json_out["rf_max_hz"] is None  # CSV carries no RF calibration
    payload = json.loads((tmp_path / "w.json").read_text())
    assert payload["channels"] == ["x", "y"] and len(payload["times"]) == 16

    shape_out = convert_waveform(
        str(tmp_path / "w.json"), str(tmp_path / "w2.shape"), "bruker", rf_max_hz=10_000.0
    )
    assert shape_out["rf_max_hz"] == 10_000.0
    again = inspect_waveform(str(tmp_path / "w2.shape"))
    assert again["mean_amplitude"] == pytest.approx(design_info["mean_amplitude"], abs=1e-9)
    assert again["duration_s"] == pytest.approx(120e-6)

    with pytest.raises(ToolError, match="format must be one of bruker, csv, json"):
        convert_waveform(result["design_json"], str(tmp_path / "x.txt"), "fapt")
    (tmp_path / "other.json").write_text('{"hello": 1}')
    with pytest.raises(ToolError, match="neither a design JSON nor a waveform JSON"):
        inspect_waveform(str(tmp_path / "other.json"))
    with pytest.raises(ToolError, match="file not found"):
        inspect_waveform(str(tmp_path / "missing.json"))


def test_untagged_bruker_shape_needs_duration_and_rf(tmp_path):
    path = export_bruker_shape(
        tmp_path / "plain.shape",
        "plain",
        np.full(8, 100.0),
        np.linspace(0.0, 315.0, 8),
    )
    info = inspect_waveform(str(path))
    assert info["duration_s"] is None and info["rf_max_hz"] is None
    assert info["n_distinct_phases"] == 8
    with pytest.raises(ToolError, match="duration_s is required: the bruker file"):
        convert_waveform(str(path), str(tmp_path / "plain.csv"), "csv")
    with pytest.raises(ToolError, match="rf_max_hz is required: the bruker file"):
        bloch_trajectory(str(path), [0.0], str(tmp_path / "t.csv"), duration_s=80e-6)
    trajectory = bloch_trajectory(
        str(path), [0.0], str(tmp_path / "t.csv"), duration_s=80e-6, rf_max_hz=5_000.0
    )
    assert trajectory["n_steps"] == 8


def test_trajectory_matches_profile_endpoint(tmp_path):
    result = _grape_inversion(tmp_path)
    offsets = [-1000.0, 0.0, 1500.0]
    csv_result = bloch_trajectory(
        result["design_json"], offsets, str(tmp_path / "traj.csv"), init="Iz"
    )
    assert csv_result["n_offsets"] == 3
    with open(tmp_path / "traj.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["step", "time_s", "offset_hz", "mx", "my", "mz"]
    assert len(rows) == 1 + 17 * 3
    # row 1 is step 0 at the first offset: the untouched initial state
    assert [float(x) for x in rows[1][3:]] == pytest.approx([0.0, 0.0, 1.0])

    profile = bloch_offset_profile(result["design_json"], -1000.0, 1500.0, 6, init="Iz")
    for entry, expected in zip(csv_result["final"], (0, 2, 5), strict=True):
        assert entry["mz"] == pytest.approx(profile["mz"][expected], abs=2e-4)

    json_result = bloch_trajectory(
        result["design_json"], offsets, str(tmp_path / "traj.json"), b1_scale=1.1
    )
    payload = json.loads((tmp_path / "traj.json").read_text())
    assert len(payload["time_s"]) == 17 and len(payload["mz"][0]) == 3
    assert payload["mz"][-1][1] == pytest.approx(json_result["final"][1]["mz"], abs=1e-4)

    with pytest.raises(ToolError, match="out_path must end in .csv or .json"):
        bloch_trajectory(result["design_json"], offsets, str(tmp_path / "traj.txt"))
    with pytest.raises(ToolError, match="offsets_hz must be non-empty"):
        bloch_trajectory(result["design_json"], [], str(tmp_path / "traj.csv"))


def test_spectrogram_writes_power_grid(tmp_path):
    result = _grape_inversion(tmp_path)
    summary = waveform_spectrogram(result["design_json"], str(tmp_path / "spec.json"))
    payload = json.loads((tmp_path / "spec.json").read_text())
    assert len(payload["freqs_hz"]) == summary["n_freqs"]
    assert len(payload["times_s"]) == summary["n_times"]
    assert len(payload["power"]) == summary["n_freqs"]
    assert payload["envelope_units"] == "Hz"
    assert summary["peak_power"] > 0.0
    assert abs(summary["peak_freq_hz"]) <= 0.5 / 7.5e-6


def test_rope_pulse_reports_theory_values(tmp_path):
    summary = rope_pulse(0.01, 20.0, 140.0, 1e-4, str(tmp_path / "rope.csv"))
    assert summary["n_steps"] == math.ceil(0.01 / 1e-4)
    assert summary["n"] == pytest.approx(20.0 / 140.0)
    assert summary["efficiency"] == pytest.approx(
        rope_finite_efficiency(0.01, 20.0 / 140.0, 140.0), abs=1e-6
    )
    assert summary["switching_time_s"] is not None
    with open(tmp_path / "rope.csv") as handle:
        header = handle.readline().strip()
    assert header == "time_s,u1,u2,amplitude_hz,phase_rad"
    with pytest.raises(ToolError, match="J must be positive"):
        rope_pulse(0.01, 20.0, 0.0, 1e-4, str(tmp_path / "rope.csv"))


def test_crop_pulse_reports_theory_values(tmp_path):
    summary = crop_pulse(30.0, 10.0, 140.0, 1e-4, str(tmp_path / "crop.csv"))
    assert summary["eta"] == pytest.approx(crop_eta(30.0, 10.0, 140.0), abs=1e-6)
    assert summary["amplitude_hz"] == pytest.approx(math.sqrt(30.0**2 - 10.0**2))
    assert summary["irradiation_freq_hz"] == -70.0
    assert summary["n_steps"] == math.ceil(summary["truncation_window_s"] / 1e-4)
    with pytest.raises(ToolError, match="truncation_window_s is required when kc_hz == ka_hz"):
        crop_pulse(30.0, 30.0, 140.0, 1e-4, str(tmp_path / "crop.csv"))
    assert crop_pulse(30.0, 30.0, 140.0, 1e-4, str(tmp_path / "c.csv"), 0.01)["n_steps"] == 100


def test_console_script_entry_point_targets_mcp_main() -> None:
    import importlib.metadata

    scripts = importlib.metadata.entry_points(group="console_scripts")
    matches = [entry for entry in scripts if entry.name == "optimalcontrol-mcp"]
    assert [entry.value for entry in matches] == ["optimalcontrol.mcp_server:main"]
