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
