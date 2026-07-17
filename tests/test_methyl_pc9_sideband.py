"""Findings tests for the 180-degree PC9 methyl-sideband example.

The example demonstrates that a PC9 shape driven to 180 degrees fails as the
methyl-band refocusing pulse. These tests lock that conclusion (and the shape
provenance) so a change in the shared physics module cannot silently invert it.
"""

import numpy as np

from examples.methyl_pc9_sideband_180 import (
    BAND_HZ,
    calibrate_180_peak_hz,
    evaluate_methyl_sideband,
    load_pc9_shape,
)
from examples.methyl_water_binary_symmetric_180 import (
    MAX_ARTIFACT_PERCENT,
    MIN_WATER_FIDELITY,
)


def test_pc9_shape_matches_bruker_provenance() -> None:
    signed, tags = load_pc9_shape()
    # 1000-point, unit-peak, exactly time-symmetric binary-phase shape.
    assert signed.shape == (1000,)
    np.testing.assert_allclose(np.max(np.abs(signed)), 1.0, atol=0.0)
    np.testing.assert_allclose(signed, signed[::-1], atol=0.0)
    # Signed net area reproduces the shape's own Bruker INTEGFAC (0.125).
    np.testing.assert_allclose(np.mean(signed), float(tags["$SHAPE_INTEGFAC"]), atol=1e-3)


def test_selective_pc9_180_fails_methyl_sideband() -> None:
    signed, tags = load_pc9_shape()
    duration_s = float(tags["$SHAPE_BWFAC"]) / BAND_HZ
    rf_peak = calibrate_180_peak_hz(signed, duration_s)
    profiles = evaluate_methyl_sideband(signed, duration_s, rf_peak)
    # Inverts only the central band, so the worst sideband is far above target
    # and the worst in-band inversion is nowhere near +0.999.
    assert float(np.max(profiles["artifact_percent"])) > MAX_ARTIFACT_PERCENT
    assert float(np.min(profiles["methyl_z"])) < 0.999


def test_hard_pc9_180_destroys_water() -> None:
    # Shortened until it covers the +/-3600 Hz band, PC9-180 is no longer
    # selective and inverts water instead of sparing it.
    signed, _ = load_pc9_shape()
    hard_s = 1.0e-4
    profiles = evaluate_methyl_sideband(signed, hard_s, calibrate_180_peak_hz(signed, hard_s))
    assert float(np.min(profiles["water_z"])) < MIN_WATER_FIDELITY
