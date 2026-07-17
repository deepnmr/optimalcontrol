"""Tests for the Seedless (ocseed) band/restraint front-end.

The anchor test reconstructs an existing example's problem through the ocseed
API and confirms it reproduces that example's published worst-case fidelities,
tying the new restraint math to a known-good pulse.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from examples.methyl_water_binary_symmetric_180 import (
    DURATION_S,
    N_STEPS,
    OPTIMIZED_HALF_AMPLITUDE,
    RF_MAX_HZ,
    SPECTROMETER_1H_MHZ,
    WATER_PPM,
    WATER_WINDOW_HZ,
    evaluate_pulse,
    signed_amplitude,
)
from optimalcontrol.ocseed import Band, SeedlessSpec


def _finite_diff_gradient(spec: SeedlessSpec, phases: np.ndarray, step: float = 1e-6) -> np.ndarray:
    numeric = np.empty_like(phases)
    for k in range(phases.size):
        bumped = phases.copy()
        bumped[k] += step
        plus, _ = spec.objective(bumped)
        bumped[k] -= 2.0 * step
        minus, _ = spec.objective(bumped)
        numeric[k] = (plus - minus) / (2.0 * step)
    return numeric


def test_reproduces_binary_symmetric_methyl_example() -> None:
    """ocseed evaluate must match the example's published worst-case fidelities."""
    signed = signed_amplitude(OPTIMIZED_HALF_AMPLITUDE)
    wfm_xy = np.column_stack((signed, np.zeros_like(signed)))
    water_half_ppm = WATER_WINDOW_HZ / SPECTROMETER_1H_MHZ

    spec = SeedlessSpec(
        spectrometer_mhz=SPECTROMETER_1H_MHZ,
        carrier_ppm=WATER_PPM,
        rf_max_hz=RF_MAX_HZ,
        duration_s=DURATION_S,
        n_steps=N_STEPS,
        bands=[
            Band(-3.0, 3.0, "universal", rotation=("x", 180.0)),
            Band(WATER_PPM - water_half_ppm, WATER_PPM + water_half_ppm, "suppress", n_offsets=9),
        ],
    )
    result = spec.evaluate(wfm_xy, dense=2401)

    # Pin the README's published absolute worst-case fidelities so a change to
    # the cached example waveform (which would move both Bloch computations in
    # common) cannot pass silently. min(Ix, -Iy, Iz) = the Iz->-Iz worst.
    assert result["band0:universal"] == pytest.approx(0.999098, abs=5e-6)
    assert result["band1:suppress"] == pytest.approx(0.999892, abs=5e-6)

    # And the ocseed restraint-mapping layer agrees with the hand-written
    # example's own per-axis evaluation on the same waveform.
    example_metrics, _ = evaluate_pulse(signed)
    expected_universal = min(
        example_metrics.methyl_x_min,
        example_metrics.methyl_y_min,
        example_metrics.methyl_z_min,
    )
    assert result["band0:universal"] == pytest.approx(expected_universal, abs=1e-9)
    assert result["band1:suppress"] == pytest.approx(example_metrics.water_z_min, abs=1e-9)


def test_gradient_matches_finite_difference() -> None:
    """Analytic objective gradient must match central differences (s2s + xycite)."""
    for band in (
        Band(-8.0, 8.0, "s2s", n_offsets=7, init="-y", targ="y"),
        Band(-6.0, 6.0, "xycite", n_offsets=7),
    ):
        spec = SeedlessSpec(600.0, 0.0, 10_000.0, 120e-6, 24, bands=[band])
        rng = np.random.default_rng(0)
        phases = rng.uniform(-math.pi, math.pi, size=spec.n_steps)
        _, analytic = spec.objective(phases)
        numeric = _finite_diff_gradient(spec, phases)
        rel = np.max(np.abs(analytic - numeric)) / (np.max(np.abs(numeric)) + 1e-12)
        assert rel < 1e-4, f"{band.restraint} gradient rel err {rel}"


def test_universal_inversion_converges_and_has_correct_signs() -> None:
    """A fresh phase-only universal 180x must invert Y and Z but hold X."""
    spec = SeedlessSpec(
        600.0,
        0.0,
        10_000.0,
        200e-6,
        50,
        bands=[Band(-2.0, 2.0, "universal", n_offsets=7, rotation=("x", 180.0))],
    )
    phases, infidelity = spec.optimize(max_iter=400, seed=5)
    assert infidelity < 0.02

    from optimalcontrol.bloch import propagate_bloch_ensemble

    wfm = spec.waveform_xy(phases)
    finals = {
        label: propagate_bloch_ensemble(
            np.array(init, dtype=np.float64),
            wfm,
            np.array([0.0]),
            np.array([1.0]),
            spec.rf_max_hz,
            spec.dt,
        )[0, 0]
        for label, init in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1]))
    }
    assert finals["x"][0] > 0.99  # X -> X
    assert finals["y"][1] < -0.99  # Y -> -Y
    assert finals["z"][2] < -0.99  # Z -> -Z


def test_xycite_drives_iz_into_transverse_plane() -> None:
    """A fresh xycite pulse must leave a small residual Iz across the band."""
    spec = SeedlessSpec(
        600.0, 0.0, 10_000.0, 60e-6, 30, bands=[Band(-6.0, 6.0, "xycite", n_offsets=11)]
    )
    phases, _ = spec.optimize(max_iter=200, seed=1)
    assert spec.evaluate(spec.waveform_xy(phases))["band0:xycite"] < 0.05


def test_export_shape_writes_phase_only_bruker_file(tmp_path) -> None:
    """export_shape must emit a full-length 100%-amplitude phase-in-degrees shape."""
    spec = SeedlessSpec(
        600.0, 0.0, 10_000.0, 100e-6, 16, bands=[Band(-5.0, 5.0, "s2s", init="-y", targ="y")]
    )
    phases = np.linspace(-math.pi, 1.5 * math.pi, spec.n_steps)  # spans past +-180 deg
    path = spec.export_shape(phases, tmp_path / "ocseed.shape")
    text = path.read_text()
    assert "##$OCSEED_RF_MAX_HZ" in text
    data_lines = [ln for ln in text.splitlines() if "," in ln and ln[0].isdigit()]
    assert len(data_lines) == spec.n_steps
    amp0, phase0 = (float(x) for x in data_lines[0].split(","))
    assert amp0 == pytest.approx(100.0)  # constant-amplitude phase-only
    assert 0.0 <= phase0 < 360.0  # phase wrapped into [0, 360)


def test_band_weight_scales_contribution() -> None:
    """A band's weight must scale its cost linearly in the combined objective."""
    phases = np.linspace(0.0, math.pi, 20)
    kw = dict(
        spectrometer_mhz=600.0, carrier_ppm=0.0, rf_max_hz=10_000.0, duration_s=100e-6, n_steps=20
    )
    single = SeedlessSpec(bands=[Band(-4.0, 4.0, "s2s", n_offsets=5, init="z", targ="-z")], **kw)
    doubled = SeedlessSpec(
        bands=[Band(-4.0, 4.0, "s2s", n_offsets=5, init="z", targ="-z", weight=2.0)], **kw
    )
    cost1, grad1 = single.objective(phases)
    cost2, grad2 = doubled.objective(phases)
    assert cost2 == pytest.approx(2.0 * cost1)
    np.testing.assert_allclose(grad2, 2.0 * grad1, atol=1e-12)


def test_single_offset_band_samples_centre() -> None:
    """n_offsets=1 must sample the band centre, not its low edge."""
    spec = SeedlessSpec(
        600.0, 0.0, 10_000.0, 100e-6, 10, bands=[Band(2.0, 6.0, "s2s", init="z", targ="-z")]
    )
    offsets = spec.band_offsets_hz(spec.bands[0], n=1)
    assert offsets.shape == (1,)
    assert offsets[0] == pytest.approx(4.0 * 600.0)  # centre 4.0 ppm -> Hz


def test_fast_kernel_matches_engine() -> None:
    """The scaled-unitary fast kernel must equal the Liouville engine to ~machine eps."""
    rng = np.random.default_rng(0)
    phases = rng.uniform(-math.pi, math.pi, size=48)
    bands = [
        Band(0.0, 20.0, "universal", n_offsets=9, rotation=("x", 180.0)),
        Band(-10.0, 10.0, "s2s", n_offsets=9, init="-y", targ="y"),
        Band(-2.0, 2.0, "suppress", n_offsets=5),
        Band(-8.0, 8.0, "xycite", n_offsets=9),
    ]
    for band in bands:
        fast = SeedlessSpec(600.0, 10.0, 1e4, 2e-3, 48, bands=[band], fast=True)
        engine = SeedlessSpec(600.0, 10.0, 1e4, 2e-3, 48, bands=[band], fast=False)
        vf, gf = fast.objective(phases)
        ve, ge = engine.objective(phases)
        assert abs(vf - ve) < 1e-10, f"{band.restraint} value"
        assert np.max(np.abs(gf - ge)) < 1e-9, f"{band.restraint} gradient"


def test_per_step_suppression_gradient_matches_fd() -> None:
    """The n^2/2 per-step suppression objective gradient must match finite differences."""
    spec = SeedlessSpec(
        600.0,
        0.0,
        1e4,
        2e-3,
        24,
        bands=[Band(-1.0, 1.0, "suppress", n_offsets=3, per_step=True)],
    )
    rng = np.random.default_rng(2)
    phases = rng.uniform(-math.pi, math.pi, size=spec.n_steps)
    _, analytic = spec.objective(phases)
    numeric = _finite_diff_gradient(spec, phases)
    rel = np.max(np.abs(analytic - numeric)) / (np.max(np.abs(numeric)) + 1e-12)
    assert rel < 1e-4, f"per-step suppression gradient rel err {rel}"


def test_per_step_suppression_holds_water_better_than_end_hold() -> None:
    """Per-step suppression must bound the mid-pulse Iz excursion that end-hold ignores.

    This is the paper's defining property (Note 2.7): end-of-pulse hold only fixes
    the endpoint and lets water swing transverse -- even fully invert -- along the
    way, whereas per-step keeps it near +z at every prefix. A constant-amplitude
    phase-only pulse cannot hold on-resonance water perfectly, so the claim is
    comparative, not absolute.
    """
    kw = dict(
        spectrometer_mhz=600.0, carrier_ppm=0.0, rf_max_hz=2500.0, duration_s=4e-3, n_steps=80
    )
    band = dict(ppm_lo=-0.3, ppm_hi=0.3, restraint="suppress", n_offsets=3)
    end_spec = SeedlessSpec(bands=[Band(per_step=False, **band)], **kw)
    per_spec = SeedlessSpec(bands=[Band(per_step=True, **band)], **kw)

    end_phases, _ = end_spec.optimize(max_iter=150, seed=1)
    per_phases, _ = per_spec.optimize(max_iter=150, seed=1)
    per_worst = per_spec.evaluate(per_spec.waveform_xy(per_phases))["band0:suppress"]
    # evaluate() reports the worst Iz projection over ALL prefixes for a per_step band;
    # the end-hold band only fixes the endpoint, so probe its trajectory directly.
    end_traj = _trajectory_min_mz(end_spec, end_phases)

    assert per_worst > 0.5
    assert end_traj < 0.0  # end-hold lets water swing past the transverse plane
    assert per_worst > end_traj + 1.0


def _trajectory_min_mz(spec: SeedlessSpec, phases: np.ndarray) -> float:
    from optimalcontrol.bloch import propagate_bloch_ensemble

    wfm = spec.waveform_xy(phases)
    return min(
        float(
            np.min(
                propagate_bloch_ensemble(
                    np.array([0.0, 0.0, 1.0]),
                    wfm[:k],
                    np.array([0.0]),
                    np.array([1.0]),
                    spec.rf_max_hz,
                    spec.dt,
                )[0, :, 2]
            )
        )
        for k in range(1, spec.n_steps + 1)
    )


def test_invalid_specs_are_rejected() -> None:
    with pytest.raises(ValueError):
        Band(1.0, 0.0, "s2s", init="z", targ="z")  # ppm_hi < ppm_lo
    with pytest.raises(ValueError):
        Band(-1.0, 1.0, "s2s")  # missing init/targ
    with pytest.raises(ValueError):
        Band(-1.0, 1.0, "bogus")  # unknown restraint
    with pytest.raises(ValueError):
        SeedlessSpec(
            600.0,
            0.0,
            10_000.0,
            1e-4,
            10,
            bands=[Band(-1.0, 1.0, "xycite")],
            b1_scales=(1.0,),
            b1_weights=(0.5,),
        )  # weights do not sum to 1


def test_kernel_rejects_non_finite_inputs() -> None:
    # The native kernel previously returned silent NaN on non-finite input,
    # unlike the rest of the library. The pre-dispatch guard now raises on both
    # the Rust and NumPy paths (it runs before the accelerator dispatch).
    from optimalcontrol import _seedless_kernel as kernel

    offsets = np.array([100.0])
    scales = np.array([1.0])
    z = np.array([0.0, 0.0, 1.0])
    nan_wfm = np.array([[np.nan, 0.0], [0.1, 0.2]])
    inf_wfm = np.array([[0.1, 0.2], [np.inf, 0.0]])
    for bad in (nan_wfm, inf_wfm):
        with pytest.raises(ValueError, match="finite"):
            kernel.s2s_value_grad(bad, offsets, scales, 1e4, 1e-6, z, z)
        with pytest.raises(ValueError, match="finite"):
            kernel.suppress_perstep_value_grad(bad, offsets, scales, 1e4, 1e-6)
    good_wfm = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="finite"):
        kernel.s2s_value_grad(good_wfm, np.array([np.inf]), scales, 1e4, 1e-6, z, z)
