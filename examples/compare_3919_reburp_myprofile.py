"""My-magnetization refocusing profile: 3-9-19 WATERGATE vs min-length REBURP.

Both elements act as the central 1H 180-degree refocusing pulse of a methyl
HMQC over water at a 1.2 GHz spectrometer, carrier on the water line (4.7 ppm).
Starting from -Iy, an ideal 180_x refocusing sends -Iy -> +Iy, so the observable
is the final My component as a function of 1H offset over -6..14 ppm.

3-9-19 WATERGATE
----------------
Hard-pulse binomial: six delta-like pulses with tip angles in ratio
3:9:19:19:9:3, the first triplet about +x and the second about -x, separated by
a free-precession delay d. The RF field is 36 kHz. The carrier sits on water, so
water (offset 0) is a null of the element and is left unrefocused; the next null
is placed on -4 ppm. That fixes the delay:

    1/d = (4.7 - (-4)) ppm * 1200 Hz/ppm = 10440 Hz  ->  d = 95.79 us

The full-inversion passbands fall midway between nulls, at 1/(2d) = 5220 Hz
either side of water (0.35 ppm and 9.05 ppm).

REBURP
------
Read directly from ``examples/output/methyl_water_reburp_minlength_180.shape``
(the cached optimal-control waveform: 10 kHz peak, 1.80 ms, binary phase).

Run without arguments to write
``examples/output/compare_3919_reburp_myprofile.png``.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.bloch import propagate_bloch_ensemble

RealArray = npt.NDArray[np.float64]

SPECTROMETER_1H_MHZ = 1200.0
WATER_PPM = 4.7
PPM_LO = -6.0
PPM_HI = 14.0
PROFILE_POINTS = 2001

SHAPE_PATH = (
    Path(__file__).resolve().parent
    / "output"
    / "methyl_water_reburp_minlength_180.shape"
)

# 3-9-19 WATERGATE hard-pulse parameters.
W3919_RF_HZ = 36000.0  # 36 kHz hard pulse
W3919_DT_S = 0.1e-6  # fine grid for the delta-pulse / delay approximation
# Per-unit tip of the 3:9:19 sub-pulses. Scaled so the binomial passband reaches
# full inversion (Mz -> -1); the delta-limit "90 deg per half" undershoots badly
# with finite delays, so the amplitude is tuned numerically (~2.5x).
W3919_UNIT_DEG = 2.5 * 90.0 / (3.0 + 9.0 + 19.0)
# Carrier on water; next inversion null on -4 ppm -> fixes the inter-pulse delay.
NULL_SPACING_HZ = (WATER_PPM - (-4.0)) * SPECTROMETER_1H_MHZ  # 10440 Hz
W3919_DELAY_S = 1.0 / NULL_SPACING_HZ


def _read_shape(path: Path) -> tuple[RealArray, float, float]:
    """Return (signed-fraction waveform, rf_max_hz, dt_s) from a Bruker shape."""
    rf_max_hz = 0.0
    duration_s = 0.0
    step_s = 0.0
    signed: list[float] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("##$OPTIMALCONTROL_RF_MAX_HZ="):
            rf_max_hz = float(line.split("=", 1)[1])
        elif line.startswith("##$OPTIMALCONTROL_STEP_DURATION_S="):
            step_s = float(line.split("=", 1)[1])
        elif line.startswith("##$OPTIMALCONTROL_TOTAL_DURATION_S="):
            duration_s = float(line.split("=", 1)[1])
        elif line[:1].isdigit() or line[:1] == "-":
            amplitude_pct, phase_deg = (float(v) for v in line.split(","))
            signed.append((amplitude_pct / 100.0) * math.cos(math.radians(phase_deg)))
    waveform = np.asarray(signed, dtype=np.float64)
    dt_s = step_s if step_s > 0.0 else duration_s / waveform.size
    return waveform, rf_max_hz, dt_s


def _pulse_block(tip_deg: float, sign: float) -> RealArray:
    """Return XY steps for one hard pulse of ``tip_deg`` about ``sign`` * x."""
    per_step_deg = 360.0 * W3919_RF_HZ * W3919_DT_S
    n_steps = max(1, int(round(tip_deg / per_step_deg)))
    block = np.zeros((n_steps, 2), dtype=np.float64)
    block[:, 0] = sign
    return block


def _delay_block() -> RealArray:
    """Return XY steps for one free-precession delay (RF off)."""
    n_steps = int(round(W3919_DELAY_S / W3919_DT_S))
    return np.zeros((n_steps, 2), dtype=np.float64)


def watergate_3919_waveform() -> RealArray:
    """Build the 3-9-19 WATERGATE waveform as (n_steps, 2) XY fractions."""
    tips = np.array([3.0, 9.0, 19.0, 19.0, 9.0, 3.0]) * W3919_UNIT_DEG
    signs = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    segments: list[RealArray] = []
    for index, (tip, sign) in enumerate(zip(tips, signs)):
        segments.append(_pulse_block(tip, sign))
        if index != len(tips) - 1:
            segments.append(_delay_block())
    return np.concatenate(segments, axis=0)


def _component_profile(
    waveform_xy: RealArray, rf_hz: float, dt: float, initial: RealArray, component: int
) -> tuple[RealArray, RealArray]:
    """Return (ppm, component) for one initial state over -6..14 ppm."""
    ppm = np.linspace(PPM_LO, PPM_HI, PROFILE_POINTS, dtype=np.float64)
    offsets_hz = (ppm - WATER_PPM) * SPECTROMETER_1H_MHZ
    scales = np.array([1.0], dtype=np.float64)
    final = propagate_bloch_ensemble(initial, waveform_xy, offsets_hz, scales, rf_hz, dt)[0]
    return ppm, np.asarray(final[:, component], dtype=np.float64)


IX = np.array([1.0, 0.0, 0.0], dtype=np.float64)
NEG_IY = np.array([0.0, -1.0, 0.0], dtype=np.float64)
IZ = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _draw_panel(
    axis: "plt.Axes",
    ppm: RealArray,
    reburp: RealArray,
    w3919: RealArray,
    ylabel: str,
    title: str,
) -> None:
    """Overlay the reburp and 3-9-19 curves on one panel."""
    axis.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    axis.axhline(-1.0, color="black", linewidth=0.5, linestyle=":")
    axis.axhline(0.0, color="black", linewidth=0.5)
    axis.axvline(
        WATER_PPM, color="tab:gray", linestyle="--", linewidth=0.9, label="water (4.7 ppm)"
    )
    axis.axvline(-4.0, color="tab:gray", linestyle=":", linewidth=0.9, label="3-9-19 null (-4 ppm)")
    axis.plot(ppm, reburp, color="tab:blue", linewidth=1.8, label="min-length REBURP (10 kHz OC)")
    axis.plot(ppm, w3919, color="tab:red", linewidth=1.4, label="3-9-19 WATERGATE (36 kHz hard)")
    axis.set_xlim(PPM_HI, PPM_LO)  # NMR convention: high ppm on the left
    axis.set_ylim(-1.15, 1.15)
    axis.set_xlabel("1H shift (ppm)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)


def plot_comparison(output_dir: Path) -> Path:
    """Draw Mz (from Iz) and My (from -Iy) refocusing panels and save a PNG."""
    reburp_signed, reburp_rf, reburp_dt = _read_shape(SHAPE_PATH)
    reburp_xy = np.column_stack((reburp_signed, np.zeros_like(reburp_signed)))
    w3919 = watergate_3919_waveform()

    ppm, reburp_mz = _component_profile(reburp_xy, reburp_rf, reburp_dt, IZ, 2)
    _, w3919_mz = _component_profile(w3919, W3919_RF_HZ, W3919_DT_S, IZ, 2)
    _, reburp_my = _component_profile(reburp_xy, reburp_rf, reburp_dt, NEG_IY, 1)
    _, w3919_my = _component_profile(w3919, W3919_RF_HZ, W3919_DT_S, NEG_IY, 1)
    _, reburp_mx = _component_profile(reburp_xy, reburp_rf, reburp_dt, IX, 0)
    _, w3919_mx = _component_profile(w3919, W3919_RF_HZ, W3919_DT_S, IX, 0)

    figure, axes = plt.subplots(1, 3, figsize=(21.0, 5.0), constrained_layout=True)
    _draw_panel(
        axes[0], ppm, reburp_mx, w3919_mx,
        "final Mx  (from Ix;  +1 = refocus)",
        "Mx (Ix -> Ix) vs offset",
    )
    _draw_panel(
        axes[1], ppm, reburp_my, w3919_my,
        "final My  (from -Iy;  +1 = refocus)",
        "My (-Iy -> Iy) vs offset",
    )
    _draw_panel(
        axes[2], ppm, reburp_mz, w3919_mz,
        "final Mz  (from Iz;  -1 = full inversion)",
        "Mz (Iz -> -Iz) vs offset",
    )
    axes[2].legend(loc="lower right", fontsize="small")
    figure.suptitle("3-9-19 WATERGATE vs OC reburp min-length 180 (1.2 GHz)")

    figure_path = output_dir / "compare_3919_reburp_myprofile.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = plot_comparison(output_dir)
    print(f"Saved examples/output/{figure_path.name}")


if __name__ == "__main__":
    main()
