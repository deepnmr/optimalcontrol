"""Methyl Kay sideband of a 180-degree PC9 pulse over -3..3 ppm at 1.2 GHz.

This reruns the methyl-sideband analysis of ``methyl_water_reburp_minlength_180``
on an off-the-shelf band-selective pulse instead of an optimal-control one: the
Bruker library PC9 shape (``Pc9_4_90.1000``, Kupce-Freeman polychromatic 90),
driven to a 180-degree on-resonance flip as the central proton refocusing pulse
of a BEST-style methyl experiment. The question it answers: if you refocus the
methyl band with a 180-degree PC9, how large is the Kay +/-J_CH/2 methyl HMQC
sideband (Lewis E. Kay, J. Biomol. NMR 73, 423-427 (2019))?

Physical framing (identical sideband metric, PC9-appropriate geometry)
----------------------------------------------------------------------

* 1.2 GHz proton spectrometer.
* Carrier at the methyl-band centre (0 ppm) -- a symmetric band-selective pulse
  must sit at the centre of the band it selects, unlike the asymmetric
  water-carried OC pulses. The -3..3 ppm methyl band is then +/-3600 Hz and
  water at 4.7 ppm sits at +5640 Hz, outside the band.
* The Kay inner-sideband artifact is computed the same way as the reference:
  from the Iz -> -Iz inversion fidelity via
  :func:`_artifact_percent_from_z_fidelity` (fidelity 0.999 <=> 0.1 percent).
* PC9 is used at 180 degrees by calibrating the peak field so the on-resonance
  spin inverts (Iz -> -Iz), the same calibration idea as ``reburp_pulse``.
* Duration is set from the shape's Bruker bandwidth factor,
  ``T = BWFAC / bandwidth``, so the *90-degree* excitation band would span the
  -3..3 ppm window -- the textbook way to time this pulse for the methyl band.

Finding
-------

A 180-degree PC9 cannot serve as the methyl-band refocusing pulse. PC9 keeps a
top-hat profile only for flip angles up to ~120 degrees; pushed to 180 its
*inversion* band collapses to roughly one fifth of its 90-degree excitation band
(inversion bandwidth * T ~= 1, versus ~5-7 for excitation). Consequently:

* Timed selectively for the band (BWFAC, ~1.04 ms, ~3.8 kHz peak) it inverts
  only the central ~+/-0.3 ppm: the worst methyl Iz -> -Iz collapses to ~-0.96
  (the spin is left essentially un-inverted) versus the +0.999 target. The Kay
  sideband is then hundreds of percent -- the ratio saturates to its ~273
  percent ceiling where inversion fails and the wanted central line nearly
  vanishes -- versus the 0.1 percent target. (Water Iz -> Iz is 0.974 here, also
  just short of 0.999.)
* Shortened until it does cover +/-3600 Hz (~0.1 ms, ~40 kHz peak) it is no
  longer selective and inverts water too (water Iz -> Iz ~= -1), and even then
  the in-band sideband is ~1 percent, still ~10x the target.

No duration meets the methyl-sideband and water-sparing constraints at once,
which is exactly the gap the optimal-control ``methyl_water_*_180`` pulses fill.

Run without arguments to write the diagnostic plots and print the summary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from examples.methyl_water_binary_symmetric_180 import (
    MAX_ARTIFACT_PERCENT,
    METHYL_PPM_HI,
    METHYL_PPM_LO,
    MIN_METHYL_FIDELITY,
    MIN_WATER_FIDELITY,
    SPECTROMETER_1H_MHZ,
    VALIDATION_WATER_POINTS,
    WATER_PPM,
    WATER_WINDOW_HZ,
    _artifact_percent_from_z_fidelity,
)
from optimalcontrol.bloch import propagate_bloch_ensemble

RealArray = npt.NDArray[np.float64]

PULSE_NAME = "methyl_pc9_sideband_180"
SHAPE_PATH = Path(__file__).resolve().parent / "data" / "Pc9_4_90.1000"

# Carrier at the methyl-band centre (see module docstring).
CARRIER_PPM = 0.5 * (METHYL_PPM_LO + METHYL_PPM_HI)
BAND_HZ = (METHYL_PPM_HI - METHYL_PPM_LO) * SPECTROMETER_1H_MHZ

VALIDATION_METHYL_POINTS = 2401
# Duration sweep for the regime plot (us): short/hard -> long/selective.
DURATION_SWEEP_US: tuple[float, ...] = (
    100.0,
    120.0,
    143.0,
    175.0,
    250.0,
    400.0,
    700.0,
    1043.0,
    1500.0,
    2000.0,
)


def load_pc9_shape() -> tuple[RealArray, dict[str, str]]:
    """Return the PC9 unit-peak signed waveform and its Bruker header tags.

    The shape is amplitude (percent of peak) with binary 0/180 phase, so the
    signed amplitude is ``+amp`` where phase is 0 and ``-amp`` where it is 180.
    """
    tags: dict[str, str] = {}
    amplitude: list[float] = []
    phase_deg: list[float] = []
    for raw in SHAPE_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("$$"):
            continue
        if line.startswith("##"):
            if "=" in line:
                key, _, value = line[2:].partition("=")
                tags[key.strip()] = value.strip()
            continue
        amp_text, _, phase_text = line.partition(",")
        amplitude.append(float(amp_text))
        phase_deg.append(float(phase_text))
    amp = np.asarray(amplitude, dtype=np.float64) / 100.0
    signed = amp * np.cos(np.deg2rad(np.asarray(phase_deg, dtype=np.float64)))
    peak = float(np.max(np.abs(signed)))
    return np.asarray(signed / peak, dtype=np.float64), tags


def calibrate_180_peak_hz(signed: RealArray, duration_s: float) -> float:
    """Return the fundamental peak field (Hz) that inverts on resonance.

    The on-resonance flip is ``2*pi * B1_peak * <signed> * T``; setting it to pi
    gives the analytic estimate ``B1_peak = 0.5 / (mean(signed) * T)``, and the
    minimiser refines the fundamental (smallest) 180 near it rather than an
    over-rotated odd multiple.
    """
    waveform_xy = np.column_stack((signed, np.zeros_like(signed)))
    dt = duration_s / signed.size
    scales = np.array([1.0], dtype=np.float64)
    estimate = 0.5 / (float(np.mean(signed)) * duration_s)

    def residual(rf_hz: float) -> float:
        mz = propagate_bloch_ensemble(
            np.array([0.0, 0.0, 1.0]), waveform_xy, np.array([0.0]), scales, float(rf_hz), dt
        )[0, 0, 2]
        return float((mz + 1.0) ** 2)

    result = minimize_scalar(
        residual,
        bounds=(0.3 * estimate, 1.8 * estimate),
        method="bounded",
        options={"xatol": 1e-2},
    )
    return float(result.x)


def evaluate_methyl_sideband(
    signed: RealArray, duration_s: float, rf_peak_hz: float, points: int = VALIDATION_METHYL_POINTS
) -> dict[str, RealArray]:
    """Return methyl transfer/sideband profiles and water sparing over the band."""
    waveform_xy = np.column_stack((signed, np.zeros_like(signed)))
    dt = duration_s / signed.size
    scales = np.array([1.0], dtype=np.float64)
    ppm = np.linspace(METHYL_PPM_LO, METHYL_PPM_HI, points, dtype=np.float64)
    offsets = (ppm - CARRIER_PPM) * SPECTROMETER_1H_MHZ

    methyl_x = propagate_bloch_ensemble(
        np.array([1.0, 0.0, 0.0]), waveform_xy, offsets, scales, rf_peak_hz, dt
    )[0, :, 0]
    methyl_y = propagate_bloch_ensemble(
        np.array([0.0, -1.0, 0.0]), waveform_xy, offsets, scales, rf_peak_hz, dt
    )[0, :, 1]
    methyl_z = -propagate_bloch_ensemble(
        np.array([0.0, 0.0, 1.0]), waveform_xy, offsets, scales, rf_peak_hz, dt
    )[0, :, 2]
    artifact = _artifact_percent_from_z_fidelity(methyl_z)

    water_offsets = (WATER_PPM - CARRIER_PPM) * SPECTROMETER_1H_MHZ + np.linspace(
        -WATER_WINDOW_HZ, WATER_WINDOW_HZ, VALIDATION_WATER_POINTS, dtype=np.float64
    )
    water_z = propagate_bloch_ensemble(
        np.array([0.0, 0.0, 1.0]), waveform_xy, water_offsets, scales, rf_peak_hz, dt
    )[0, :, 2]
    return {
        "ppm": ppm,
        "methyl_x": np.asarray(methyl_x, dtype=np.float64),
        "methyl_y": np.asarray(methyl_y, dtype=np.float64),
        "methyl_z": np.asarray(methyl_z, dtype=np.float64),
        "artifact_percent": artifact,
        "water_offsets_hz": water_offsets,
        "water_z": np.asarray(water_z, dtype=np.float64),
    }


def duration_regime_sweep(
    signed: RealArray, durations_us: tuple[float, ...] = DURATION_SWEEP_US
) -> dict[str, RealArray]:
    """Return worst sideband, worst inversion, water sparing and peak RF vs duration."""
    duration_us = np.asarray(durations_us, dtype=np.float64)
    peak_khz = np.empty_like(duration_us)
    sideband_max = np.empty_like(duration_us)
    methyl_z_min = np.empty_like(duration_us)
    water_z_min = np.empty_like(duration_us)
    for index, dur_us in enumerate(durations_us):
        duration_s = dur_us * 1e-6
        rf_peak = calibrate_180_peak_hz(signed, duration_s)
        profiles = evaluate_methyl_sideband(signed, duration_s, rf_peak)
        peak_khz[index] = rf_peak / 1000.0
        sideband_max[index] = float(np.max(profiles["artifact_percent"]))
        methyl_z_min[index] = float(np.min(profiles["methyl_z"]))
        water_z_min[index] = float(np.min(profiles["water_z"]))
    return {
        "duration_us": duration_us,
        "peak_khz": peak_khz,
        "sideband_max_percent": sideband_max,
        "methyl_z_min": methyl_z_min,
        "water_z_min": water_z_min,
    }


def plot_diagnostics(
    signed: RealArray,
    duration_s: float,
    rf_peak_hz: float,
    profiles: dict[str, RealArray],
    output_dir: Path,
) -> Path:
    """Write the PC9 waveform and its methyl-sideband profiles over -3..3 ppm."""
    amplitude = np.abs(signed)
    phase_deg = np.where(signed < 0.0, 180.0, 0.0)
    time_us = (np.arange(signed.size, dtype=np.float64) + 0.5) * (duration_s / signed.size) * 1e6
    ppm = profiles["ppm"]
    sideband_max = float(np.max(profiles["artifact_percent"]))

    figure, axes = plt.subplots(4, 1, figsize=(9.0, 10.0), constrained_layout=True)

    axes[0].step(time_us, 100.0 * amplitude, where="mid", color="tab:blue")
    phase_axis = axes[0].twinx()
    phase_axis.step(time_us, phase_deg, where="mid", color="tab:red", alpha=0.55)
    axes[0].set_ylabel("Amplitude (%)")
    phase_axis.set_ylabel("Phase (deg)")
    phase_axis.set_yticks([0.0, 180.0])
    axes[0].set_xlabel("Time (us)")
    axes[0].set_title(
        f"180-degree PC9 (Pc9_4_90): {duration_s * 1e6:.0f} us, {rf_peak_hz / 1000.0:.2f} kHz peak"
    )

    axes[1].plot(ppm, profiles["methyl_x"], label="Ix -> Ix")
    axes[1].plot(ppm, profiles["methyl_y"], label="-Iy -> Iy")
    axes[1].plot(ppm, profiles["methyl_z"], label="Iz -> -Iz")
    axes[1].axhline(MIN_METHYL_FIDELITY, color="black", linestyle=":")
    axes[1].set_xlim(METHYL_PPM_LO, METHYL_PPM_HI)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("Transfer fidelity")
    axes[1].legend(loc="lower center", ncol=3, fontsize="small")

    axes[2].semilogy(ppm, np.maximum(profiles["artifact_percent"], 1e-4), color="tab:green")
    axes[2].axhline(MAX_ARTIFACT_PERCENT, color="black", linestyle=":")
    axes[2].set_xlim(METHYL_PPM_LO, METHYL_PPM_HI)
    axes[2].set_ylabel("Kay sideband (%)")
    axes[2].set_xlabel("Methyl 1H shift (ppm)")
    axes[2].set_title(
        f"Worst methyl sideband over -3..3 ppm = {sideband_max:.2f}% (target <= 0.1%)"
    )

    axes[3].plot(profiles["water_offsets_hz"], profiles["water_z"], marker="o")
    axes[3].axhline(MIN_WATER_FIDELITY, color="black", linestyle=":")
    axes[3].set_ylim(-1.05, 1.05)
    axes[3].set_xlabel("Offset from carrier (Hz); water at 4.7 ppm")
    axes[3].set_ylabel("Water Iz -> Iz")

    figure_path = output_dir / f"{PULSE_NAME}.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_regime(sweep: dict[str, RealArray], bwfac_us: float, output_dir: Path) -> Path:
    """Write the sideband-vs-water trade-off across duration (peak field)."""
    duration_us = sweep["duration_us"]
    figure, axis = plt.subplots(1, 1, figsize=(8.0, 4.8), constrained_layout=True)
    axis.semilogy(
        duration_us,
        np.maximum(sweep["sideband_max_percent"], 1e-4),
        marker="o",
        color="tab:green",
        label="worst methyl sideband (%)",
    )
    axis.axhline(MAX_ARTIFACT_PERCENT, color="tab:green", linestyle=":", alpha=0.7)
    axis.axvline(
        bwfac_us,
        color="tab:gray",
        linestyle="--",
        alpha=0.7,
        label=f"BWFAC timing ({bwfac_us:.0f} us)",
    )
    axis.set_xlabel("PC9 duration (us)")
    axis.set_ylabel("Worst methyl sideband (%)")

    water_axis = axis.twinx()
    water_axis.plot(
        duration_us,
        sweep["water_z_min"],
        marker="s",
        color="tab:blue",
        label="worst water Iz -> Iz",
    )
    water_axis.axhline(MIN_WATER_FIDELITY, color="tab:blue", linestyle=":", alpha=0.7)
    water_axis.set_ylabel("Worst water Iz -> Iz")
    water_axis.set_ylim(-1.1, 1.1)

    handles = axis.get_legend_handles_labels()[0] + water_axis.get_legend_handles_labels()[0]
    labels = axis.get_legend_handles_labels()[1] + water_axis.get_legend_handles_labels()[1]
    axis.legend(handles, labels, loc="center right", fontsize="small")
    axis.set_title("180-degree PC9: neither selective nor hard timing passes both constraints")

    figure_path = output_dir / f"{PULSE_NAME}_regime.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def run() -> dict[str, float]:
    """Generate the diagnostics and return the featured-pulse summary metrics."""
    signed, tags = load_pc9_shape()
    bwfac = float(tags.get("$SHAPE_BWFAC", "7.512"))
    duration_s = bwfac / BAND_HZ
    rf_peak = calibrate_180_peak_hz(signed, duration_s)
    profiles = evaluate_methyl_sideband(signed, duration_s, rf_peak)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_diagnostics(signed, duration_s, rf_peak, profiles, output_dir)
    plot_regime(duration_regime_sweep(signed), duration_s * 1e6, output_dir)

    return {
        "duration_us": duration_s * 1e6,
        "peak_khz": rf_peak / 1000.0,
        "methyl_x_min": float(np.min(profiles["methyl_x"])),
        "methyl_y_min": float(np.min(profiles["methyl_y"])),
        "methyl_z_min": float(np.min(profiles["methyl_z"])),
        "sideband_max_percent": float(np.max(profiles["artifact_percent"])),
        "water_z_min": float(np.min(profiles["water_z"])),
    }


def demo() -> None:
    """Self-check: PC9 provenance, calibration, sideband formula, and the finding."""
    signed, tags = load_pc9_shape()
    # Signed net area reproduces the shape's own Bruker INTEGFAC (0.125).
    assert abs(float(np.mean(signed)) - float(tags["$SHAPE_INTEGFAC"])) < 1e-3
    # The 180 calibration actually inverts on resonance.
    duration_s = float(tags["$SHAPE_BWFAC"]) / BAND_HZ
    rf_peak = calibrate_180_peak_hz(signed, duration_s)
    waveform_xy = np.column_stack((signed, np.zeros_like(signed)))
    mz0 = propagate_bloch_ensemble(
        np.array([0.0, 0.0, 1.0]),
        waveform_xy,
        np.array([0.0]),
        np.array([1.0]),
        rf_peak,
        duration_s / signed.size,
    )[0, 0, 2]
    assert mz0 < -0.999, mz0
    # Sideband formula boundary: fidelity 0.999 <=> 0.1 percent (reference criterion).
    assert abs(float(_artifact_percent_from_z_fidelity(np.array([0.999]))[0]) - 0.1) < 1e-3
    # The finding: the selective (BWFAC) PC9-180 fails the methyl sideband spec...
    selective = evaluate_methyl_sideband(signed, duration_s, rf_peak)
    assert float(np.max(selective["artifact_percent"])) > MAX_ARTIFACT_PERCENT
    # ...and the hard PC9-180 that covers the band destroys water instead.
    hard_s = 1.0e-4
    hard = evaluate_methyl_sideband(signed, hard_s, calibrate_180_peak_hz(signed, hard_s))
    assert float(np.min(hard["water_z"])) < MIN_WATER_FIDELITY
    print("demo ok")


def main() -> None:
    summary = run()
    print(
        f"180-degree PC9 over -3..3 ppm at {SPECTROMETER_1H_MHZ / 1000:.1f} GHz "
        f"(carrier {CARRIER_PPM:.1f} ppm):"
    )
    print(f"  duration (BWFAC timing) = {summary['duration_us']:.0f} us")
    print(f"  peak RF for 180         = {summary['peak_khz']:.2f} kHz")
    print(f"  methyl Ix->Ix   min     = {summary['methyl_x_min']:.4f}")
    print(f"  methyl -Iy->Iy  min     = {summary['methyl_y_min']:.4f}")
    print(f"  methyl Iz->-Iz  min     = {summary['methyl_z_min']:.4f}  (target >= 0.999)")
    print(f"  worst methyl sideband   = {summary['sideband_max_percent']:.2f}%  (target <= 0.1%)")
    print(f"  water Iz->Iz    min     = {summary['water_z_min']:.4f}  (target >= 0.999)")
    print(f"Saved examples/output/{PULSE_NAME}.png")
    print(f"Saved examples/output/{PULSE_NAME}_regime.png")


if __name__ == "__main__":
    main()
