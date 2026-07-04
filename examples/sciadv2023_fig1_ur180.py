"""Reproduce a Sci. Adv. 2023 Fig. 1-style low-power UR-180 pulse.

Reference: Joseph and Griesinger, Sci. Adv. 9, eadj1133 (2023), Fig. 1.
The figure shows a phase-modulated low-power UR-180 pulse with duration
540 us, RF amplitude 7.5 kHz, bandwidth +/-6.3 kHz, and B1 compensation
over +/-15%. This example uses phase-only GRAPE for the transfer shown in
Fig. 1B, ``-Iy -> Iy``, and writes a Bruker shape file plus a Fig. 1-like
diagnostic plot.

The default path uses a cached phase-only GRAPE result so that the example is
fast and deterministic. Pass ``--optimize`` to regenerate the pulse with GRAPE.

Saves:
  examples/output/sciadv2023_fig1_ur180.shape
  examples/output/sciadv2023_fig1_ur180.png
"""

import matplotlib

matplotlib.use("Agg")

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from optimalcontrol.bloch import propagate_bloch_ensemble
from optimalcontrol.grape import ControlProblem, grape_gradient, grape_xy, phase_only_gradient
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, vec
from optimalcontrol.states import normalise_hs, state_from_label

PULSE_NAME = "sciadv2023_fig1_ur180"
N_STEPS = 72
DURATION_S = 540.0e-6
DT = DURATION_S / N_STEPS
RF_HZ = 7_500.0
BANDWIDTH_HZ = 6_300.0
B1_DEVIATION_FRACTION = 0.15

# Cached phase-only GRAPE solution for -Iy -> Iy on offsets +/-6.3 kHz and
# B1 scaling [0.85, 0.925, 1.0, 1.075, 1.15]. The phase is stored unwrapped in
# degrees for readability; Bruker export wraps it to [0, 360).
OPTIMIZED_PHASE_DEG = np.array(
    [
        366.598765774,
        377.231349831,
        382.980994995,
        379.004819449,
        363.459376012,
        341.406708309,
        323.288651868,
        314.226135696,
        307.745490131,
        303.256781759,
        302.402608290,
        306.746958515,
        311.625609473,
        299.388295944,
        125.228370798,
        193.977694519,
        145.823900646,
        145.917034746,
        150.749035483,
        157.461727783,
        165.983988929,
        175.600670430,
        191.946413689,
        239.196675425,
        270.741700256,
        270.756936964,
        262.002283373,
        248.319347981,
        232.734318302,
        230.472448101,
        247.710295880,
        271.137202006,
        280.145153618,
        276.813543052,
        267.363847514,
        254.245808311,
        238.754506509,
        222.965118530,
        209.141602992,
        199.337589020,
        195.766505901,
        203.581531670,
        247.472537128,
        353.019642759,
        411.344718841,
        430.062468202,
        423.325759885,
        409.889057987,
        395.166330837,
        381.090483158,
        382.311642586,
        452.668569908,
        527.551547679,
        528.157717449,
        523.027198062,
        519.432218340,
        518.618541480,
        518.662149939,
        519.306789279,
        519.574732057,
        520.213358921,
        524.608912455,
        532.111657380,
        535.518837160,
        533.796942433,
        595.016099670,
        720.584915645,
        723.712632998,
        714.011007244,
        715.137588552,
        716.021712757,
        716.835032572,
    ],
    dtype=np.float64,
)


def _phase_to_xy(phase_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a unit-amplitude Cartesian waveform from phase samples."""
    return np.asarray(
        np.column_stack((np.cos(phase_rad), np.sin(phase_rad))),
        dtype=np.float64,
    )


def _control_problem(
    offsets_hz: npt.NDArray[np.float64] | None = None,
    b1_scales: list[float] | None = None,
) -> ControlProblem:
    """Build the single-spin Liouville GRAPE problem for -Iy -> Iy."""
    if offsets_hz is None:
        offsets_hz = np.linspace(-BANDWIDTH_HZ, BANDWIDTH_HZ, 13, dtype=np.float64)
    if b1_scales is None:
        b1_scales = [0.85, 0.925, 1.0, 1.075, 1.15]

    l_x = liouvillian_comm(Ix())
    l_y = liouvillian_comm(Iy())
    l_z = liouvillian_comm(Iz())
    rho_init = -vec(normalise_hs(state_from_label("Iy", 1)))
    rho_targ = vec(normalise_hs(state_from_label("Iy", 1)))
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=np.complex128)],
        operators=[2.0 * math.pi * RF_HZ * l_x, 2.0 * math.pi * RF_HZ * l_y],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=DT,
        pwr_levels=b1_scales,
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
        offsets=[float(offset) for offset in offsets_hz],
        offset_operators=[2.0 * math.pi * l_z],
    )


def _initial_phase_rad() -> npt.NDArray[np.float64]:
    """Return a smooth Fig. 1A-like phase seed for optional re-optimisation."""
    knots = np.array(
        [
            0,
            0.04,
            0.08,
            0.12,
            0.19,
            0.20,
            0.30,
            0.32,
            0.40,
            0.48,
            0.52,
            0.58,
            0.64,
            0.68,
            0.76,
            0.82,
            0.84,
            0.88,
            0.95,
            1.0,
        ],
        dtype=np.float64,
    )
    values = np.array(
        [
            360,
            330,
            500,
            340,
            360,
            540,
            550,
            700,
            620,
            660,
            650,
            540,
            360,
            360,
            470,
            550,
            450,
            600,
            620,
            760,
        ],
        dtype=np.float64,
    )
    phase = np.interp(np.linspace(0.0, 1.0, N_STEPS), knots, values)
    for _ in range(3):
        phase = np.convolve(
            np.r_[phase[0], phase, phase[-1]],
            np.array([0.25, 0.5, 0.25], dtype=np.float64),
            mode="same",
        )[1:-1]
    return np.deg2rad(phase)


def _optimise_phase(max_iter: int) -> npt.NDArray[np.float64]:
    """Run phase-only GRAPE and return unwrapped phase samples in radians."""
    cp = _control_problem()
    phase0 = _initial_phase_rad()

    def objective(phase_rad: npt.NDArray[np.float64]) -> float:
        return -float(grape_xy(cp, _phase_to_xy(phase_rad)))

    def gradient(phase_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        waveform = _phase_to_xy(phase_rad)
        grad_xy = grape_gradient(cp, waveform)
        return -phase_only_gradient(grad_xy, waveform)

    result = minimize(
        objective,
        phase0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1.0e-10, "gtol": 1.0e-6, "maxls": 30},
    )
    print(
        f"Phase-only GRAPE: fidelity = {-result.fun:.6f}, "
        f"iterations = {result.nit}, success = {result.success}"
    )
    return np.asarray(result.x, dtype=np.float64)


def _profile_maps(
    phase_rad: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Return rectangular and OC transfer maps plus OC Mxy and phase-error maps."""
    offsets = np.linspace(-BANDWIDTH_HZ, BANDWIDTH_HZ, 81, dtype=np.float64)
    deviations = np.linspace(
        -100.0 * B1_DEVIATION_FRACTION,
        100.0 * B1_DEVIATION_FRACTION,
        41,
        dtype=np.float64,
    )
    initial = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    scales = np.asarray(1.0 + deviations / 100.0, dtype=np.float64)
    oc_final = propagate_bloch_ensemble(
        initial, _phase_to_xy(phase_rad), offsets, scales, RF_HZ, DT
    )
    rect_final = propagate_bloch_ensemble(
        initial,
        np.array([[1.0, 0.0]], dtype=np.float64),
        offsets,
        scales,
        RF_HZ,
        1.0 / (2.0 * RF_HZ),
    )
    rect_transfer = np.asarray(rect_final[:, :, 1], dtype=np.float64)
    oc_transfer = np.asarray(oc_final[:, :, 1], dtype=np.float64)
    oc_mxy = np.asarray(np.hypot(oc_final[:, :, 0], oc_final[:, :, 1]), dtype=np.float64)
    oc_phase_error = np.asarray(
        np.degrees(np.arctan2(oc_final[:, :, 0], oc_final[:, :, 1])),
        dtype=np.float64,
    )

    return offsets, deviations, rect_transfer, oc_transfer, oc_mxy, oc_phase_error


def _display_phase_deg(phase_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return phase in a Fig. 1A-like unwrapped degree range."""
    phase = np.degrees(phase_rad).copy()
    phase[phase < 200.0] += 360.0
    phase[phase > 800.0] -= 360.0
    return phase


def _export_bruker_shape(phase_rad: npt.NDArray[np.float64], output_dir: Path) -> Path:
    """Write the constant-amplitude Bruker shape file."""
    shape_path = output_dir / f"{PULSE_NAME}.shape"
    phase_deg = np.mod(np.degrees(phase_rad), 360.0)
    lines = [
        f"##TITLE= {PULSE_NAME}",
        "##JCAMP-DX= 5.00 Bruker JCAMP library",
        "##DATA TYPE= Shape Data",
        "##ORIGIN= optimalcontrol",
        "##OWNER= optimalcontrol",
        "##MINX= 1.000000e+02",
        "##MAXX= 1.000000e+02",
        "##MINY= 0.000000e+00",
        "##MAXY= 3.600000e+02",
        "##$SHAPE_EXMODE= None",
        "##$SHAPE_TOTROT= 1.800000e+02",
        "##$SHAPE_BWFAC= 0.000000e+00",
        "##$SHAPE_INTEGFAC= 1.000000e+00",
        "##$SHAPE_MODE= 1",
        f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {DURATION_S:.12e}",
        f"##$OPTIMALCONTROL_STEP_DURATION_S= {DT:.12e}",
        f"##$OPTIMALCONTROL_RF_HZ= {RF_HZ:.12e}",
        f"##$OPTIMALCONTROL_BANDWIDTH_HZ= {BANDWIDTH_HZ:.12e}",
        f"##$OPTIMALCONTROL_B1_DEVIATION_PERCENT= {100.0 * B1_DEVIATION_FRACTION:.12e}",
        "##$OPTIMALCONTROL_NOTE= Set pulse length to TOTAL_DURATION_S and calibrate 100% to RF_HZ.",
        f"##NPOINTS= {N_STEPS}",
        "##XYPOINTS= (XY..XY)",
    ]
    for phase in phase_deg:
        lines.append(f"1.000000000e+02, {float(phase):.9e}")
    lines.append("##END=")
    shape_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return shape_path


def _plot_figure(
    phase_rad: npt.NDArray[np.float64],
    output_dir: Path,
) -> tuple[Path, npt.NDArray[np.float64]]:
    """Create a Fig. 1-like pulse/performance figure."""
    offsets, deviations, rect, oc, mxy, phase_error = _profile_maps(phase_rad)
    extent = [offsets[0] / 1000.0, offsets[-1] / 1000.0, deviations[0], deviations[-1]]
    roi = np.isfinite(oc)
    oc_mean = float(np.mean(oc[roi]))
    oc_std = float(np.std(oc[roi]))
    mxy_mean = float(np.mean(mxy[roi]))
    mxy_std = float(np.std(mxy[roi]))
    phase_mean = float(np.mean(phase_error[roi]))
    phase_std = float(np.std(phase_error[roi]))

    fig = plt.figure(figsize=(9.0, 8.6))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.95, wspace=0.35)

    ax_amp = fig.add_subplot(grid[0, :])
    time_us = np.arange(N_STEPS, dtype=np.float64) * DT * 1.0e6
    ax_phase = ax_amp.twinx()
    ax_amp.plot(time_us, np.full(N_STEPS, 100.0), color="blue", linewidth=1.8)
    ax_phase.plot(time_us, _display_phase_deg(phase_rad), color="red", linewidth=1.8)
    ax_amp.set_ylim(0.0, 100.0)
    ax_phase.set_ylim(200.0, 800.0)
    ax_amp.set_xlim(0.0, DURATION_S * 1.0e6)
    ax_amp.set_ylabel("Amplitude (%)", color="blue")
    ax_phase.set_ylabel("Phase (degree)", color="red")
    ax_amp.tick_params(axis="y", labelcolor="blue")
    ax_phase.tick_params(axis="y", labelcolor="red")
    ax_amp.set_xlabel("Time (us)")
    ax_amp.set_title("A", loc="left", fontweight="bold")

    contour_levels = [0.80, 0.90, 0.95, 0.99]
    image_kwargs = dict(
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0.75,
        vmax=1.0,
        cmap="gray",
    )
    panels = [
        (fig.add_subplot(grid[1, 0]), rect, "Rectangular pulse"),
        (fig.add_subplot(grid[1, 1]), oc, "Phase-only GRAPE UR-180"),
        (
            fig.add_subplot(grid[2, 0]),
            mxy,
            rf"$M_{{xy}}$ normalized: {mxy_mean:.4f} +/- {mxy_std:.4f}",
        ),
        (
            fig.add_subplot(grid[2, 1]),
            phase_error,
            rf"Phase error: {phase_mean:.3f} +/- {phase_std:.3f} deg",
        ),
    ]
    for ax, data, title in panels:
        if data is phase_error:
            image = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="gray",
                vmin=-5.0,
                vmax=5.0,
            )
            levels = [-2.0, -1.0, 0.0, 1.0, 2.0]
        else:
            image = ax.imshow(data, **image_kwargs)
            levels = contour_levels
        contours = ax.contour(
            offsets / 1000.0,
            deviations,
            data,
            levels=levels,
            colors="black",
            linewidths=0.55,
        )
        ax.clabel(contours, inline=True, fontsize=7, fmt="%g")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\nu_0$ (kHz)")
        ax.set_ylabel(r"$\Delta\nu_1$ (%)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

    ax_note = fig.add_subplot(grid[3, :])
    ax_note.axis("off")
    ax_note.text(
        0.0,
        0.92,
        "B  Simulated -Iy -> Iy transfer profile for a hard 180 pulse and the GRAPE pulse.\n"
        "C  GRAPE pulse transverse magnetization retention and phase error over the same region.\n"
        f"Optimisation-grid average transfer fidelity: {oc_mean:.4f} +/- {oc_std:.4f}",
        va="top",
        fontsize=10,
    )

    fig_path = output_dir / f"{PULSE_NAME}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    summary = np.array(
        [oc_mean, oc_std, mxy_mean, mxy_std, phase_mean, phase_std],
        dtype=np.float64,
    )
    return fig_path, summary


def run(optimize: bool = False, max_iter: int = 120) -> npt.NDArray[np.float64]:
    """Write the Bruker shape and Fig. 1-like diagnostic plot."""
    phase_rad = _optimise_phase(max_iter) if optimize else np.deg2rad(OPTIMIZED_PHASE_DEG)
    output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_path = _export_bruker_shape(phase_rad, output_dir)
    fig_path, summary = _plot_figure(phase_rad, output_dir)
    print(f"Saved Bruker shape {shape_path}")
    print(f"Saved figure {fig_path}")
    return np.concatenate((np.mod(np.degrees(phase_rad), 360.0), summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimize", action="store_true", help="rerun phase-only GRAPE")
    parser.add_argument("--max-iter", type=int, default=120, help="GRAPE optimizer iterations")
    args = parser.parse_args()
    run(optimize=args.optimize, max_iter=args.max_iter)
