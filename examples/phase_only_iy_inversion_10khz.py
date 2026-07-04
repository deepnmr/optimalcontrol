"""Shortest phase-only 10 kHz pulse found for ``Iy -> -Iy`` over 20 kHz.

This example uses phase-only GRAPE at constant RF amplitude 10 kHz and searches
over a descending duration grid for the shortest pulse whose simulated
contiguous inversion profile exceeds 20 kHz total bandwidth when the final
``Iy`` component stays below ``-0.90``.

The default path uses a cached result found on the search grid so that the
example is fast and deterministic. Pass ``--optimize`` to rerun the duration
search and phase-only optimisation.

Saves:
  examples/output/phase_only_iy_inversion_10khz.shape
  examples/output/phase_only_iy_inversion_10khz.png
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

PULSE_NAME = "phase_only_iy_inversion_10khz"
N_STEPS = 72
RF_HZ = 10_000.0
PROFILE_THRESHOLD = -0.90
REQUIRED_BANDWIDTH_HZ = 20_000.0
PROFILE_HALF_WIDTH_HZ = 0.5 * REQUIRED_BANDWIDTH_HZ
SEARCH_DURATIONS_US = [150.0, 120.0, 100.0, 95.0, 92.0, 90.0, 88.0, 86.0, 85.0, 84.0, 82.0, 80.0]
CACHED_DURATION_S = 86.0e-6

# Cached shortest passing solution on SEARCH_DURATIONS_US for the stated
# criterion. Values are wrapped to [0, 360) for direct Bruker export.
OPTIMIZED_PHASE_DEG = np.array(
    [
        355.721317676236,
        347.383863259143,
        339.646707201565,
        332.748319100528,
        326.774833314853,
        321.700393335187,
        317.438759469126,
        313.882731608266,
        310.926875599766,
        308.477810359644,
        306.457314675462,
        304.801956047264,
        303.461373463151,
        302.396274250858,
        301.576592636169,
        300.979970655177,
        300.590597699279,
        300.398373831212,
        300.398374985524,
        300.590601960218,
        300.979975807828,
        301.576597034852,
        302.396279172753,
        303.461379927246,
        304.801964275028,
        306.457323526082,
        308.477818311939,
        310.926885034414,
        313.88274524403,
        317.438773516822,
        321.700408613192,
        326.774853285747,
        332.748325937385,
        339.646700335577,
        347.383892148747,
        355.72137941868,
        4.278676806389,
        12.616111762449,
        20.353307437857,
        27.251698722865,
        33.225170121366,
        38.299608939834,
        42.561240954014,
        46.117265433211,
        49.073121394092,
        51.522186961698,
        53.542681974773,
        55.198039910513,
        56.538621381193,
        57.603719286801,
        58.42339995547,
        59.020020407599,
        59.409393226557,
        59.601619110256,
        59.601618951094,
        59.40939262217,
        59.020018962396,
        58.423397003165,
        57.603714300683,
        56.538614162266,
        55.198031139162,
        53.54267425992,
        51.522180213794,
        49.073106865651,
        46.117251075613,
        42.561225940129,
        38.299595076438,
        33.225152952503,
        27.251673896438,
        20.353294107212,
        12.61611603724,
        4.278644202392,
    ],
    dtype=np.float64,
)


def _phase_to_xy(phase_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a unit-amplitude Cartesian waveform from phase samples."""
    return np.asarray(
        np.column_stack((np.cos(phase_rad), np.sin(phase_rad))),
        dtype=np.float64,
    )


def _control_problem(duration_s: float) -> ControlProblem:
    """Build the single-spin Liouville problem for ``Iy -> -Iy``."""
    dt = duration_s / N_STEPS
    offsets_hz = np.linspace(
        -PROFILE_HALF_WIDTH_HZ,
        PROFILE_HALF_WIDTH_HZ,
        21,
        dtype=np.float64,
    )
    l_x = liouvillian_comm(Ix())
    l_y = liouvillian_comm(Iy())
    l_z = liouvillian_comm(Iz())
    rho_init = vec(normalise_hs(state_from_label("Iy", 1)))
    rho_targ = -vec(normalise_hs(state_from_label("Iy", 1)))
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=np.complex128)],
        operators=[2.0 * math.pi * RF_HZ * l_x, 2.0 * math.pi * RF_HZ * l_y],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=dt,
        pwr_levels=[1.0, 1.0],
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
        offsets=[float(offset) for offset in offsets_hz],
        offset_operators=[2.0 * math.pi * l_z],
    )


def _initial_phase_rad() -> npt.NDArray[np.float64]:
    """Return a smooth symmetric phase seed."""
    knots = np.array(
        [
            0.0,
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
            360.0,
            330.0,
            500.0,
            340.0,
            360.0,
            540.0,
            550.0,
            700.0,
            620.0,
            660.0,
            650.0,
            540.0,
            360.0,
            360.0,
            470.0,
            550.0,
            450.0,
            600.0,
            620.0,
            760.0,
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


def _optimise_phase(
    duration_s: float,
    phase0_rad: npt.NDArray[np.float64],
    max_iter: int,
) -> npt.NDArray[np.float64]:
    """Run phase-only GRAPE at a fixed duration."""
    cp = _control_problem(duration_s)

    def objective(phase_rad: npt.NDArray[np.float64]) -> float:
        return -float(grape_xy(cp, _phase_to_xy(phase_rad)))

    def gradient(phase_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        waveform = _phase_to_xy(phase_rad)
        grad_xy = grape_gradient(cp, waveform)
        return -phase_only_gradient(grad_xy, waveform)

    result = minimize(
        objective,
        phase0_rad,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1.0e-12, "gtol": 1.0e-7, "maxls": 40},
    )
    return np.asarray(result.x, dtype=np.float64)


def _profile_y(
    phase_rad: npt.NDArray[np.float64],
    duration_s: float,
    offsets_hz: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the final ``Iy`` component versus offset."""
    initial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    final = propagate_bloch_ensemble(
        initial,
        _phase_to_xy(phase_rad),
        offsets_hz,
        np.array([1.0], dtype=np.float64),
        RF_HZ,
        duration_s / N_STEPS,
    )
    return np.asarray(final[0, :, 1], dtype=np.float64)


def _profile_bandwidth_hz(
    phase_rad: npt.NDArray[np.float64],
    duration_s: float,
) -> tuple[float, float, float]:
    """Return contiguous threshold bandwidth, on-resonance transfer, and minimum."""
    offsets = np.linspace(-15_000.0, 15_000.0, 1201, dtype=np.float64)
    y_values = _profile_y(phase_rad, duration_s, offsets)
    ok = y_values <= PROFILE_THRESHOLD
    centre = offsets.size // 2
    if not ok[centre]:
        return 0.0, float(y_values[centre]), float(np.min(y_values))

    left = centre
    right = centre
    while left - 1 >= 0 and ok[left - 1]:
        left -= 1
    while right + 1 < ok.size and ok[right + 1]:
        right += 1
    return (
        float(offsets[right] - offsets[left]),
        float(y_values[centre]),
        float(np.min(y_values)),
    )


def _search_shortest_pulse(max_iter: int) -> tuple[float, npt.NDArray[np.float64]]:
    """Search the duration grid and keep the shortest passing pulse."""
    phase = _initial_phase_rad()
    best: tuple[float, npt.NDArray[np.float64]] | None = None
    for duration_us in SEARCH_DURATIONS_US:
        duration_s = duration_us * 1.0e-6
        phase = _optimise_phase(duration_s, phase, max_iter=max_iter)
        cp = _control_problem(duration_s)
        fidelity = float(grape_xy(cp, _phase_to_xy(phase)))
        bandwidth_hz, centre_y, _ = _profile_bandwidth_hz(phase, duration_s)
        print(
            "duration_us="
            f"{duration_us:.1f} fidelity={fidelity:.6f} "
            f"bandwidth_hz={bandwidth_hz:.1f} centre_y={centre_y:.6f}"
        )
        if bandwidth_hz >= REQUIRED_BANDWIDTH_HZ:
            best = (duration_s, phase.copy())

    if best is None:
        raise RuntimeError("no candidate pulse met the required 20 kHz profile")
    return best


def _export_bruker_shape(
    phase_rad: npt.NDArray[np.float64],
    duration_s: float,
    output_dir: Path,
) -> Path:
    """Write a constant-amplitude Bruker shape file."""
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
        f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {duration_s:.12e}",
        f"##$OPTIMALCONTROL_STEP_DURATION_S= {duration_s / N_STEPS:.12e}",
        f"##$OPTIMALCONTROL_RF_HZ= {RF_HZ:.12e}",
        f"##$OPTIMALCONTROL_REQUIRED_BANDWIDTH_HZ= {REQUIRED_BANDWIDTH_HZ:.12e}",
        f"##$OPTIMALCONTROL_PROFILE_THRESHOLD= {PROFILE_THRESHOLD:.12e}",
        "##$OPTIMALCONTROL_TARGET= Iy_to_minus_Iy",
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
    duration_s: float,
    output_dir: Path,
) -> tuple[Path, npt.NDArray[np.float64]]:
    """Plot pulse phase and the simulated inversion profile."""
    offsets_khz = np.linspace(-15.0, 15.0, 1201, dtype=np.float64)
    y_values = _profile_y(phase_rad, duration_s, offsets_khz * 1.0e3)
    bandwidth_hz, centre_y, min_y = _profile_bandwidth_hz(phase_rad, duration_s)
    edge_values = _profile_y(
        phase_rad,
        duration_s,
        np.array([-PROFILE_HALF_WIDTH_HZ, PROFILE_HALF_WIDTH_HZ], dtype=np.float64),
    )

    fig, (ax_phase, ax_profile) = plt.subplots(2, 1, figsize=(8.6, 6.6), constrained_layout=True)
    time_us = np.linspace(0.0, duration_s * 1.0e6, N_STEPS, endpoint=False)
    ax_amp = ax_phase.twinx()
    ax_phase.plot(time_us, np.mod(np.degrees(phase_rad), 360.0), color="red", linewidth=1.7)
    ax_amp.plot(time_us, np.full(N_STEPS, 100.0), color="blue", linewidth=1.5)
    ax_phase.set_ylabel("Phase (deg)", color="red")
    ax_amp.set_ylabel("Amplitude (%)", color="blue")
    ax_phase.tick_params(axis="y", labelcolor="red")
    ax_amp.tick_params(axis="y", labelcolor="blue")
    ax_phase.set_xlabel("Time (us)")
    ax_phase.set_title("Phase-only 10 kHz Iy -> -Iy pulse")
    ax_phase.set_ylim(0.0, 360.0)
    ax_amp.set_ylim(0.0, 100.0)

    ax_profile.plot(offsets_khz, y_values, color="black", linewidth=1.6)
    ax_profile.axhline(PROFILE_THRESHOLD, color="gray", linestyle="--", linewidth=1.0)
    for edge in (-PROFILE_HALF_WIDTH_HZ, PROFILE_HALF_WIDTH_HZ):
        ax_profile.axvline(edge / 1.0e3, color="tab:blue", linestyle=":", linewidth=1.0)
    ax_profile.set_xlabel("Offset (kHz)")
    ax_profile.set_ylabel("Final Iy")
    ax_profile.set_ylim(-1.05, 0.2)
    ax_profile.set_xlim(float(offsets_khz[0]), float(offsets_khz[-1]))
    ax_profile.set_title(
        "Profile bandwidth "
        f"{bandwidth_hz / 1.0e3:.2f} kHz, centre {centre_y:.3f}, "
        f"edges {edge_values[0]:.3f}/{edge_values[1]:.3f}"
    )

    fig_path = output_dir / f"{PULSE_NAME}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    summary = np.array(
        [bandwidth_hz, centre_y, min_y, edge_values[0], edge_values[1]],
        dtype=np.float64,
    )
    return fig_path, summary


def run(optimize: bool = False, max_iter: int = 200) -> npt.NDArray[np.float64]:
    """Write the Bruker shape and a diagnostic profile plot."""
    if optimize:
        duration_s, phase_rad = _search_shortest_pulse(max_iter=max_iter)
    else:
        duration_s = CACHED_DURATION_S
        phase_rad = np.deg2rad(OPTIMIZED_PHASE_DEG)

    output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_path = _export_bruker_shape(phase_rad, duration_s, output_dir)
    fig_path, summary = _plot_figure(phase_rad, duration_s, output_dir)
    fidelity = float(grape_xy(_control_problem(duration_s), _phase_to_xy(phase_rad)))
    print(f"Saved Bruker shape {shape_path}")
    print(f"Saved figure {fig_path}")
    print(
        "Selected duration "
        f"{duration_s * 1.0e6:.1f} us, average fidelity {fidelity:.6f}, "
        f"profile bandwidth {summary[0] / 1.0e3:.2f} kHz"
    )
    return np.concatenate(
        (
            np.array([duration_s * 1.0e6, fidelity], dtype=np.float64),
            summary,
            np.mod(np.degrees(phase_rad), 360.0),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimize", action="store_true", help="rerun the duration search")
    parser.add_argument("--max-iter", type=int, default=200, help="L-BFGS iterations per duration")
    args = parser.parse_args()
    run(optimize=args.optimize, max_iter=args.max_iter)
