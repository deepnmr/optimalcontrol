"""Generic broadband 180-degree pulse optimizer using full GRAPE.

Optimises amplitude and phase of a 72-step, 540 us pulse at RF amplitude
7.5 kHz to achieve a UR-180 transfer (-Iy -> Iy) over +-10 kHz with +-10 %
B1 robustness. The SNSA penalty constrains the instantaneous RF to at most
7.5 kHz at every step.

Unlike the phase-only sciadv2023_fig1_ur180 example, this uses full GRAPE
(amplitude and phase jointly optimised), which can achieve wider bandwidth
under the same amplitude constraint.

The default path uses a cached full-GRAPE result. Pass --optimize to regenerate.

Saves:
  examples/output/grape_broadband_180.shape
  examples/output/grape_broadband_180.png
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

from optimalcontrol.bloch import propagate_bloch_ensemble
from optimalcontrol.grape import ControlProblem
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, vec
from optimalcontrol.optimizers import lbfgs_grape
from optimalcontrol.penalties import PenaltySpec
from optimalcontrol.states import normalise_hs, state_from_label

PULSE_NAME = "grape_broadband_180"
N_STEPS = 72
DURATION_S = 540.0e-6
DT = DURATION_S / N_STEPS
RF_MAX_HZ = 7_500.0
BANDWIDTH_HZ = 10_000.0
B1_DEVIATION_FRACTION = 0.10
SNSA_WEIGHT = 0.05

# Cached full-GRAPE solution for -Iy -> Iy on offsets +-10 kHz and
# B1 scaling [0.9, 1.0, 1.1]. Shape: (N_STEPS, 2) = (72, 2).
# Columns: [u_x, u_y] as fractions of RF_MAX_HZ (values in [-1, 1]).
# Populated after the first --optimize run; zeros before that.
OPTIMIZED_WFM_XY: npt.NDArray[np.float64] = np.array(
    [
        [-1.004065846174199e00, -8.423240281412382e-02],
        [-9.792612225252345e-01, -2.344552113172299e-01],
        [-9.482015621287974e-01, -3.348517477399370e-01],
        [-9.223708986138450e-01, -3.957986866092369e-01],
        [-8.835954872191022e-01, -4.714876748054307e-01],
        [3.490101367091071e-01, 9.377980564288737e-01],
        [6.608345972847258e-01, 7.540312982532891e-01],
        [8.818740354600638e-01, 4.798055308323946e-01],
        [9.942958999960748e-01, 1.410877222480287e-01],
        [9.710818060986125e-01, -2.538614884596637e-01],
        [8.059005497236893e-01, -5.967149842966649e-01],
        [7.580363494918874e-01, -6.545241291283034e-01],
        [1.176647359939978e-01, 1.204968924393955e-01],
        [-9.102373993421053e-01, -4.178881526834837e-01],
        [-1.432229340828666e-01, -9.913533122430691e-01],
        [8.452866027462204e-01, -5.370555775007528e-01],
        [9.677443658343790e-01, 2.607523134455989e-01],
        [8.551556693104480e-01, 5.233520062429132e-01],
        [7.536488947319645e-01, 6.608234176308605e-01],
        [2.991755641467897e-01, 9.557683553763573e-01],
        [-9.629384479638541e-01, 2.758984927629374e-01],
        [-9.007076894306961e-01, -4.431118517449833e-01],
        [-7.926388020278867e-01, -6.188679373073362e-01],
        [-7.416803392228151e-01, -6.805058543223530e-01],
        [-7.254983777504344e-01, -6.976347820451285e-01],
        [-7.492996186208457e-01, -6.706768531313111e-01],
        [-8.539378182650079e-01, -5.283680104680633e-01],
        [-9.979578000394164e-01, -9.758465560186391e-02],
        [-7.371791924260019e-01, 6.787135631370123e-01],
        [-2.253029997168926e-01, 9.766585679237786e-01],
        [-2.288344700291620e-01, 9.759859360092176e-01],
        [-5.954253107844003e-01, 8.065216746601174e-01],
        [-8.971385240831001e-01, 4.475454455159660e-01],
        [-9.947942873079231e-01, 1.227468113767875e-01],
        [-1.001983359820976e00, -6.640384412528766e-03],
        [-9.787832739559780e-01, 2.133078933503876e-01],
        [-8.345582755243477e-01, 5.540403989136883e-01],
        [-7.156345908201076e-01, 7.009090991588307e-01],
        [-9.717955522654277e-01, 2.408900241777247e-01],
        [-7.082481586465303e-01, -7.071500501693924e-01],
        [-5.429971546598137e-01, -8.407139625622234e-01],
        [-5.441542785763546e-01, -5.554399529719790e-01],
        [9.915692504507960e-01, 1.364218840561043e-01],
        [9.427339239758944e-01, -3.391901300285885e-01],
        [8.148585485196600e-01, -5.831489044368090e-01],
        [8.760951759499246e-02, -9.977523127857973e-01],
        [-8.994256318966950e-01, -4.402392823616904e-01],
        [-9.982743838508664e-01, 8.307438843443528e-02],
        [-9.685905005694971e-01, 2.554214003281385e-01],
        [-9.565730229353600e-01, 2.963422969099885e-01],
        [-9.698365713363365e-01, 2.488061384716354e-01],
        [-9.983691052568785e-01, -7.207038629424546e-02],
        [-7.041593456097561e-01, -7.110217216317921e-01],
        [5.430613447690241e-02, -9.988813949242383e-01],
        [7.541379620045713e-01, -6.569897659632516e-01],
        [7.170602516110229e-01, 4.583476350117862e-02],
        [-4.210770277968897e-01, 8.826515123789578e-01],
        [-6.076589112270770e-01, 7.955405585474405e-01],
        [-5.135606728030814e-01, 5.891512203807610e-01],
        [6.323693263431467e-01, 7.759519541576704e-01],
        [6.975412495097264e-01, 7.223622275485982e-01],
        [7.944115757383374e-01, 6.219807921249776e-01],
        [8.503004834857878e-01, 5.532970604897866e-01],
        [8.764320082832023e-01, 5.202778824694541e-01],
        [8.956007989175977e-01, 4.918346194993086e-01],
        [9.080528653066454e-01, 4.669383427920023e-01],
        [9.255484206473995e-01, 4.211903500566574e-01],
        [9.394352298065171e-01, 3.705150058954664e-01],
        [9.854600869187856e-01, -1.780723139585734e-01],
        [-9.604838114737175e-01, -3.041870800402449e-01],
        [-1.004941231046212e00, -1.363724556453847e-01],
        [-1.016986171758554e00, -4.573768731496249e-02],
    ],
    dtype=np.float64,
)


def _control_problem(
    offsets_hz: npt.NDArray[np.float64] | None = None,
    b1_scales: list[float] | None = None,
) -> ControlProblem:
    """Build the single-spin Liouville GRAPE problem for -Iy -> Iy."""
    if offsets_hz is None:
        offsets_hz = np.linspace(-BANDWIDTH_HZ, BANDWIDTH_HZ, 21, dtype=np.float64)
    if b1_scales is None:
        b1_scales = [
            1.0 - B1_DEVIATION_FRACTION,
            1.0,
            1.0 + B1_DEVIATION_FRACTION,
        ]

    l_x = liouvillian_comm(Ix())
    l_y = liouvillian_comm(Iy())
    l_z = liouvillian_comm(Iz())
    rho_init = -vec(normalise_hs(state_from_label("Iy", 1)))
    rho_targ = vec(normalise_hs(state_from_label("Iy", 1)))
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=np.complex128)],
        operators=[2.0 * math.pi * RF_MAX_HZ * l_x, 2.0 * math.pi * RF_MAX_HZ * l_y],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=DT,
        pwr_levels=b1_scales,
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
        offsets=[float(o) for o in offsets_hz],
        offset_operators=[2.0 * math.pi * l_z],
        penalties=[PenaltySpec(kind="SNSA", weight=SNSA_WEIGHT, limit=1.0)],
    )


def _initial_waveform() -> npt.NDArray[np.float64]:
    """Return a reproducible small-amplitude random seed waveform."""
    rng = np.random.default_rng(42)
    return np.asarray(0.05 * rng.standard_normal((N_STEPS, 2)), dtype=np.float64)


def _optimise_wfm(max_iter: int) -> npt.NDArray[np.float64]:
    """Run full L-BFGS GRAPE and return the optimised (N_STEPS, 2) waveform."""
    cp = _control_problem()
    wfm0 = _initial_waveform()
    result = lbfgs_grape(cp, wfm0, max_iter=max_iter)
    print(
        f"Full GRAPE: fidelity = {result.fidelity_final:.6f}, "
        f"iterations = {result.n_iter}, reason = {result.reason}"
    )
    return np.asarray(result.wfm_final, dtype=np.float64)


def _profile_maps(
    wfm_xy: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Return rectangular and OC transfer maps over an offset x B1 grid."""
    offsets = np.linspace(-BANDWIDTH_HZ, BANDWIDTH_HZ, 201, dtype=np.float64)
    deviations = np.linspace(
        -100.0 * B1_DEVIATION_FRACTION,
        100.0 * B1_DEVIATION_FRACTION,
        41,
        dtype=np.float64,
    )
    initial = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    scales = np.asarray(1.0 + deviations / 100.0, dtype=np.float64)
    oc_final = propagate_bloch_ensemble(initial, wfm_xy, offsets, scales, RF_MAX_HZ, DT)
    rect_final = propagate_bloch_ensemble(
        initial,
        np.array([[1.0, 0.0]], dtype=np.float64),
        offsets,
        scales,
        RF_MAX_HZ,
        1.0 / (2.0 * RF_MAX_HZ),
    )
    rect_transfer = np.asarray(rect_final[:, :, 1], dtype=np.float64)
    oc_transfer = np.asarray(oc_final[:, :, 1], dtype=np.float64)
    oc_mxy = np.asarray(np.hypot(oc_final[:, :, 0], oc_final[:, :, 1]), dtype=np.float64)
    oc_phase_error = np.asarray(
        np.degrees(np.arctan2(oc_final[:, :, 0], oc_final[:, :, 1])),
        dtype=np.float64,
    )

    return offsets, deviations, rect_transfer, oc_transfer, oc_mxy, oc_phase_error


def _export_bruker_shape(wfm_xy: npt.NDArray[np.float64], output_dir: Path) -> Path:
    """Write the variable-amplitude Bruker shape file."""
    shape_path = output_dir / f"{PULSE_NAME}.shape"
    amplitude_percent = 100.0 * np.hypot(wfm_xy[:, 0], wfm_xy[:, 1])
    phase_deg = np.mod(np.degrees(np.arctan2(wfm_xy[:, 1], wfm_xy[:, 0])), 360.0)
    phase_deg[amplitude_percent < 1e-6] = 0.0
    integfac = float(np.mean(amplitude_percent)) / 100.0

    lines = [
        f"##TITLE= {PULSE_NAME}",
        "##JCAMP-DX= 5.00 Bruker JCAMP library",
        "##DATA TYPE= Shape Data",
        "##ORIGIN= optimalcontrol",
        "##OWNER= optimalcontrol",
        "##MINX= 0.000000e+00",
        "##MAXX= 1.000000e+02",
        "##MINY= 0.000000e+00",
        "##MAXY= 3.600000e+02",
        "##$SHAPE_EXMODE= None",
        "##$SHAPE_TOTROT= 1.800000e+02",
        "##$SHAPE_BWFAC= 0.000000e+00",
        f"##$SHAPE_INTEGFAC= {integfac:.9e}",
        "##$SHAPE_MODE= 0",
        f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {DURATION_S:.12e}",
        f"##$OPTIMALCONTROL_STEP_DURATION_S= {DT:.12e}",
        f"##$OPTIMALCONTROL_RF_HZ= {RF_MAX_HZ:.12e}",
        f"##$OPTIMALCONTROL_BANDWIDTH_HZ= {BANDWIDTH_HZ:.12e}",
        f"##$OPTIMALCONTROL_B1_DEVIATION_PERCENT= {100.0 * B1_DEVIATION_FRACTION:.12e}",
        "##$OPTIMALCONTROL_NOTE= Set pulse length to TOTAL_DURATION_S and calibrate 100% to RF_HZ.",
        f"##NPOINTS= {N_STEPS}",
        "##XYPOINTS= (XY..XY)",
    ]
    for amp, phase in zip(amplitude_percent, phase_deg):
        lines.append(f"{float(amp):.9e}, {float(phase):.9e}")
    lines.append("##END=")
    shape_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return shape_path


def _plot_figure(
    wfm_xy: npt.NDArray[np.float64],
    output_dir: Path,
) -> tuple[Path, npt.NDArray[np.float64]]:
    """Create a diagnostic pulse/performance figure."""
    offsets, deviations, rect, oc, mxy, phase_error = _profile_maps(wfm_xy)
    extent = [offsets[0] / 1000.0, offsets[-1] / 1000.0, deviations[0], deviations[-1]]
    roi = np.isfinite(oc)
    oc_mean = float(np.mean(oc[roi]))
    oc_std = float(np.std(oc[roi]))
    mxy_mean = float(np.mean(mxy[roi]))
    mxy_std = float(np.std(mxy[roi]))
    phase_mean = float(np.mean(phase_error[roi]))
    phase_std = float(np.std(phase_error[roi]))

    amplitude_percent = 100.0 * np.hypot(wfm_xy[:, 0], wfm_xy[:, 1])
    phase_display = np.degrees(np.arctan2(wfm_xy[:, 1], wfm_xy[:, 0]))
    time_us = np.arange(N_STEPS, dtype=np.float64) * DT * 1.0e6

    fig = plt.figure(figsize=(9.0, 8.6))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.95, wspace=0.35)

    ax_amp = fig.add_subplot(grid[0, :])
    ax_phase = ax_amp.twinx()
    ax_amp.plot(time_us, amplitude_percent, color="blue", linewidth=1.8)
    ax_phase.plot(time_us, phase_display, color="red", linewidth=1.8)
    ax_amp.set_ylim(0.0, 110.0)
    ax_phase.set_ylim(-200.0, 200.0)
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
        (fig.add_subplot(grid[1, 1]), oc, "Full-GRAPE UR-180"),
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


def run(optimize: bool = False, max_iter: int = 500) -> npt.NDArray[np.float64]:
    """Write the Bruker shape and diagnostic plot."""
    wfm_xy = _optimise_wfm(max_iter) if optimize else OPTIMIZED_WFM_XY.copy()
    output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_path = _export_bruker_shape(wfm_xy, output_dir)
    fig_path, summary = _plot_figure(wfm_xy, output_dir)
    print(f"Saved Bruker shape {shape_path}")
    print(f"Saved figure {fig_path}")
    return np.concatenate((wfm_xy.reshape(-1), summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimize", action="store_true", help="rerun full GRAPE")
    parser.add_argument("--max-iter", type=int, default=500, help="GRAPE optimizer iterations")
    args = parser.parse_args()
    run(optimize=args.optimize, max_iter=args.max_iter)
