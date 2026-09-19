"""End-to-end check of the MCP tool functions (called directly, no transport)."""

import json

import pytest

pytest.importorskip("mcp")

from optimalcontrol.mcp_server import bloch_offset_profile, design_seedless_pulse


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
