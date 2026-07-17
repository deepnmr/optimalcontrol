"""Minimum-length REBURP-style methyl 180 over water at 10 kHz.

This is the minimum-duration end of the (max-power, duration) Pareto frontier
mapped by ``methyl_water_reburp_minpower_180`` -- its mirror image. Both solve
the same physical problem, the refocusing-pulse artifact of Lewis E. Kay,
J. Biomol. NMR 73, 423-427 (2019), DOI 10.1007/s10858-019-00227-7, in which an
imperfect central proton 180-degree pulse produces methyl HMQC satellites at
+/-J_CH/2 that survive gradients and phase cycling. The minimum-power sibling
trades peak field down to 6.0 kHz at the cost of 2.60 ms; this pulse instead
spends the full 10 kHz peak field to reach the shortest passing duration,
1.80 ms.

Pareto frontier
---------------

Lowering the peak RF amplitude costs duration: a band-selective inversion that
covers the whole methyl band (offsets -9240..-2040 Hz) while leaving water
untouched needs a roughly constant peak-field * time budget. The smooth
REBURP-style optimization (second-difference penalty, smoothness weight 1e-4)
maps out the following shortest passing durations:

    peak RF (kHz)   min duration (ms)
    10.0            1.80
     9.5            1.80
     9.0            1.90
     8.5            1.90
     8.0            1.90
     7.5            1.90
     7.0            2.10
     6.5            2.10
     6.0            2.60

The curve is flat from 10 down to about 7.5 kHz, then the duration climbs
steeply toward the minimum-power end (6.0 kHz at 2.60 ms). The cached pulse is
the minimum-length point: 10 kHz at 1.80 ms -- the shortest passing duration on
the frontier, and 0.15 ms shorter than the smooth ``methyl_water_reburp_180``
sibling at the same peak field.

Pushing below this floor fails: re-optimizing a 1.40 ms pulse at the 10 kHz cap
(warm-started from the cached 1.80 ms shape) reaches only ~0.39 percent worst
Kay sideband -- above the 0.1 percent target -- which is why 1.80 ms is the
shortest feasible length. See ``refine_pulse`` and the corresponding test.

Design specification (identical to the sibling pulses)
------------------------------------------------------

* 1.2 GHz proton spectrometer, carrier at water (4.7 ppm).
* Methyl range -3.0 to 3.0 ppm: true 180_x, tested as Ix -> Ix,
  -Iy -> Iy, and Iz -> -Iz. The simultaneous three-axis fidelity makes the
  net propagator a universal (pure-phase) rotation, so methyl TROSY is
  preserved and no net 1H chemical-shift evolution accrues across the pulse.
* Water: Iz -> Iz over 4.7 ppm +/-100 Hz.
* RF amplitude is variable but never exceeds the cached peak field.
* RF phase is binary: exactly 0 or 180 degrees.
* The signed-amplitude waveform is exactly time symmetric.

Pass criteria on 2401 methyl offsets and 9 water offsets are:

* worst methyl Ix -> Ix, -Iy -> Iy, and Iz -> -Iz fidelity >= 0.999;
* worst water Iz -> Iz fidelity >= 0.999; and
* worst Kay inner-sideband artifact <= 0.1 percent of the central line.

Run without arguments to write the cached Bruker shape and diagnostic plots.
Use ``--optimize`` to repeat the (max-power, duration) Pareto search.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

# The sibling module owns the dense physical evaluation, the constraint
# constants, and the wide/time profile computations; all now accept an explicit
# ``rf_max_hz`` so they are reused here at the reduced peak field.
from examples.methyl_water_binary_symmetric_180 import (
    MAX_ARTIFACT_PERCENT,
    METHYL_OFFSET_HI_HZ,
    METHYL_OFFSET_LO_HZ,
    METHYL_PPM_HI,
    METHYL_PPM_LO,
    MIN_METHYL_FIDELITY,
    MIN_WATER_FIDELITY,
    SPECTROMETER_1H_MHZ,
    TIME_EVOLUTION_PPM,
    VALIDATION_WATER_POINTS,
    WATER_PPM,
    WATER_WINDOW_HZ,
    WIDE_PPM_HI,
    WIDE_PPM_LO,
    PulseMetrics,
    amplitude_phase,
    evaluate_pulse,
    time_evolution_profiles,
    wide_component_profile,
    wide_transfer_profile,
)
from optimalcontrol.grape import ControlProblem, grape_xy_and_gradient
from optimalcontrol.io import export_bruker_shape as write_bruker_shape
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, vec
from optimalcontrol.states import normalise_hs

RealArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

PULSE_NAME = "methyl_water_reburp_minlength_180"
N_STEPS = 200

# Cached minimum-length end of the Pareto frontier (see module docstring).
RF_MAX_HZ = 10000.0
DURATION_US = 1800.0
DURATION_S = DURATION_US * 1.0e-6
DT = DURATION_S / N_STEPS

SMOOTHNESS_WEIGHT = 1.0e-4

# Shortest passing duration (us) at each scanned peak RF amplitude (Hz). This is
# the (max-power, length) trade-off the search maps out.
FRONTIER: tuple[tuple[float, float], ...] = (
    (10000.0, 1800.0),
    (9500.0, 1800.0),
    (9000.0, 1900.0),
    (8500.0, 1900.0),
    (8000.0, 1900.0),
    (7500.0, 1900.0),
    (7000.0, 2100.0),
    (6500.0, 2100.0),
    (6000.0, 2600.0),
)


# Optimized half of the smooth signed-amplitude waveform. The second half is
# its reverse. Values are fractions of RF_MAX_HZ and lie in [-1, 1].
OPTIMIZED_HALF_AMPLITUDE: RealArray = np.array(
    [
        -0.14192559532933624, -0.03704501176435874, 0.05158900935758466, 0.10434688927143171,
        0.1081006621831702, 0.062455342603668605, -0.01817186951881482, -0.10855152696385847,
        -0.1808531280868793, -0.21200325856592614, -0.18991022274229064, -0.11828325039423226,
        -0.01641174535071956, 0.08707218582727208, 0.1619978338452024, 0.18385498445480458,
        0.13977527309142748, 0.03241898479261897, -0.11966418032264932, -0.28479634923414715,
        -0.4248435244661069, -0.5046169617583475, -0.5011681299027942, -0.40968343762157333,
        -0.24542985722069446, -0.04172887505995483, 0.15547168036553527, 0.2967207206817997,
        0.3406363769226618, 0.2657781989174879, 0.07801255303107749, -0.187860586684636,
        -0.47285463186514215, -0.7060485633864347, -0.822918731309153, -0.7845123167859742,
        -0.5913819316041886, -0.2861541738589543, 0.05690682121883165, 0.3529506688980467,
        0.5308665230850819, 0.5534310848983511, 0.4280964530364903, 0.20402609034890018,
        -0.0447076110979988, -0.24201053179845955, -0.3309916743544786, -0.2894447766740929,
        -0.13454544455183576, 0.08610591012362044, 0.31224316599824087, 0.49010003066975805,
        0.5884530378173785, 0.6065806011450019, 0.5705060358296603, 0.5196124445229043,
        0.4893767554794785, 0.4955718025229333, 0.5264659128728092, 0.5481519652286163,
        0.5207525796659916, 0.41705301250676174, 0.23731685041886283, 0.01484225931969178,
        -0.19237990804716687, -0.32148393946008236, -0.3271460653509745, -0.19914023388856572,
        0.029818190099634007, 0.28966828341994866, 0.49287841758830603, 0.5619170925877313,
        0.4560488205312146, 0.18922371259134538, -0.17226107104646649, -0.5321119816238737,
        -0.7950014796640754, -0.8971438185706162, -0.8243384348078453, -0.610642461213518,
        -0.3236071628515354, -0.0414783953034517, 0.17038635400387867, 0.2758670839998437,
        0.2719461413670707, 0.1804872626104752, 0.03360683892206752, -0.1384556199131672,
        -0.30995949353989, -0.4564963776667766, -0.5550624054066454, -0.5861173773305621,
        -0.5349784226732618, -0.39366552260151827, -0.1651565964252656, 0.13042010462655568,
        0.4511629137757451, 0.7365753928572312, 0.927245530349415, 1.0,
    ],
    dtype=np.float64,
)


def signed_amplitude(half: RealArray = OPTIMIZED_HALF_AMPLITUDE) -> RealArray:
    """Return the exactly symmetric signed RF-amplitude waveform."""
    half_array = np.asarray(half, dtype=np.float64)
    if half_array.shape != (N_STEPS // 2,):
        raise ValueError(f"half waveform must have shape ({N_STEPS // 2},)")
    if np.max(np.abs(half_array)) > 1.0 + 1e-12:
        raise ValueError("signed amplitude exceeds the peak RF limit")
    return np.asarray(np.concatenate((half_array, half_array[::-1])), dtype=np.float64)


def _control_problem(
    offsets_hz: RealArray,
    rho_init: list[ComplexArray],
    rho_targ: list[ComplexArray],
    duration_s: float,
    rf_max_hz: float,
) -> ControlProblem:
    """Build a one-channel, nominal-B1 Liouville problem at this peak field."""
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=np.complex128)],
        operators=[2.0 * math.pi * rf_max_hz * liouvillian_comm(Ix())],
        rho_init=[vec(normalise_hs(state)) for state in rho_init],
        rho_targ=[vec(normalise_hs(state)) for state in rho_targ],
        pulse_dt=duration_s / N_STEPS,
        pwr_levels=[1.0],
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
        offsets=[float(offset) for offset in offsets_hz],
        offset_operators=[2.0 * math.pi * liouvillian_comm(Iz())],
    )


def _smoothness(signed: RealArray) -> tuple[float, RealArray]:
    """Return the second-difference roughness and its gradient w.r.t. steps."""
    d2 = np.diff(signed, 2)
    value = float(np.sum(d2**2))
    gradient = np.zeros(signed.size, dtype=np.float64)
    gradient[:-2] += 2.0 * d2
    gradient[1:-1] += -4.0 * d2
    gradient[2:] += 2.0 * d2
    return value, gradient


def _collapse_symmetric_gradient(gradient: RealArray) -> RealArray:
    """Apply the mirror-parameterization chain rule to a full step gradient."""
    half = N_STEPS // 2
    return np.asarray(gradient[:half] + gradient[half:][::-1], dtype=np.float64)


def refine_pulse(
    duration_us: float,
    rf_max_hz: float,
    half_seed: RealArray,
    smoothness_weight: float = SMOOTHNESS_WEIGHT,
    max_iter: int = 1200,
) -> tuple[RealArray, PulseMetrics]:
    """Refine one (duration, peak-field) point with the smooth four-state objective."""
    duration_s = duration_us * 1e-6
    methyl_offsets = np.linspace(
        METHYL_OFFSET_LO_HZ, METHYL_OFFSET_HI_HZ, 81, dtype=np.float64
    )
    water_offsets = np.linspace(
        -WATER_WINDOW_HZ, WATER_WINDOW_HZ, VALIDATION_WATER_POINTS, dtype=np.float64
    )
    cp_x = _control_problem(methyl_offsets, [Ix()], [Ix()], duration_s, rf_max_hz)
    cp_y = _control_problem(methyl_offsets, [-Iy()], [Iy()], duration_s, rf_max_hz)
    cp_z = _control_problem(methyl_offsets, [Iz()], [-Iz()], duration_s, rf_max_hz)
    cp_water = _control_problem(water_offsets, [Iz()], [Iz()], duration_s, rf_max_hz)

    def objective(half: RealArray) -> tuple[float, RealArray]:
        signed = signed_amplitude(half)
        waveform = signed[:, None]
        value_x, gradient_x = grape_xy_and_gradient(cp_x, waveform)
        value_y, gradient_y = grape_xy_and_gradient(cp_y, waveform)
        value_z, gradient_z = grape_xy_and_gradient(cp_z, waveform)
        value_water, gradient_water = grape_xy_and_gradient(cp_water, waveform)
        value = (value_x + value_y + value_z + value_water) / 4.0
        gradient = (gradient_x + gradient_y + gradient_z + gradient_water)[:, 0] / 4.0
        rough, rough_gradient = _smoothness(signed)
        objective_value = -value + smoothness_weight * rough
        objective_gradient = _collapse_symmetric_gradient(
            -gradient + smoothness_weight * rough_gradient
        )
        return objective_value, objective_gradient

    result = minimize(
        objective,
        np.asarray(half_seed, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-1.0, 1.0)] * (N_STEPS // 2),
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-11},
    )
    half = np.asarray(result.x, dtype=np.float64)
    metrics, _ = evaluate_pulse(
        signed_amplitude(half), duration_s=duration_s, rf_max_hz=rf_max_hz
    )
    return half, metrics


def search_pareto(
    rf_grid_hz: tuple[float, ...] = (
        10000.0,
        9000.0,
        8000.0,
        7000.0,
        6000.0,
    ),
    duration_grid_us: tuple[float, ...] = (
        1800.0,
        1900.0,
        2100.0,
        2300.0,
        2600.0,
        3000.0,
    ),
) -> list[tuple[float, float, RealArray, PulseMetrics]]:
    """Map shortest passing duration vs peak field, warm-starting downward.

    Returns one ``(rf_hz, duration_us, half, metrics)`` tuple per peak field
    that has a passing duration in the grid.
    """
    seed = OPTIMIZED_HALF_AMPLITUDE.copy()
    frontier: list[tuple[float, float, RealArray, PulseMetrics]] = []
    for rf_max_hz in rf_grid_hz:
        local_seed = seed.copy()
        for duration_us in duration_grid_us:
            half, metrics = refine_pulse(duration_us, rf_max_hz, local_seed)
            local_seed = half
            if metrics.passes:
                frontier.append((rf_max_hz, duration_us, half.copy(), metrics))
                seed = half.copy()
                break
    return frontier


def export_bruker_shape(signed: RealArray, output_dir: Path) -> Path:
    """Write the variable-amplitude, binary-phase Bruker shape file."""
    amplitude, phase_deg = amplitude_phase(signed)
    return write_bruker_shape(
        output_dir / f"{PULSE_NAME}.shape",
        PULSE_NAME,
        100.0 * amplitude,
        phase_deg,
        maxy=180.0,
        bwfac=None,
        integfac=float(np.mean(amplitude)),
        extra_tags=[
            f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {DURATION_S:.12e}",
            f"##$OPTIMALCONTROL_STEP_DURATION_S= {DT:.12e}",
            f"##$OPTIMALCONTROL_RF_MAX_HZ= {RF_MAX_HZ:.12e}",
            f"##$OPTIMALCONTROL_SPECTROMETER_1H_MHZ= {SPECTROMETER_1H_MHZ:.12e}",
            f"##$OPTIMALCONTROL_METHYL_PPM= {METHYL_PPM_LO:.3f}..{METHYL_PPM_HI:.3f}",
            f"##$OPTIMALCONTROL_WATER_PPM= {WATER_PPM:.3f}",
            "##$OPTIMALCONTROL_PHASE_SET_DEG= 0,180",
            "##$OPTIMALCONTROL_SYMMETRIC= yes",
            "##$OPTIMALCONTROL_STYLE= reburp",
        ],
    )


def plot_diagnostics(
    signed: RealArray,
    metrics: PulseMetrics,
    profiles: dict[str, RealArray],
    output_dir: Path,
) -> Path:
    """Write pulse constraints and dense validation profiles as a PNG."""
    amplitude, phase_deg = amplitude_phase(signed)
    time_us = (np.arange(signed.size, dtype=np.float64) + 0.5) * DT * 1e6
    methyl_ppm = WATER_PPM + profiles["methyl_offsets_hz"] / SPECTROMETER_1H_MHZ
    figure, axes = plt.subplots(4, 1, figsize=(9.0, 10.0), constrained_layout=True)

    axes[0].step(time_us, 100.0 * amplitude, where="mid", color="tab:blue")
    phase_axis = axes[0].twinx()
    phase_axis.step(time_us, phase_deg, where="mid", color="tab:red", alpha=0.55)
    axes[0].set_ylabel("Amplitude (%)")
    phase_axis.set_ylabel("Phase (deg)")
    phase_axis.set_yticks([0.0, 180.0])
    axes[0].set_title(
        f"Min-length REBURP-style methyl 180: "
        f"{DURATION_US:.0f} us, {RF_MAX_HZ / 1000.0:.1f} kHz max"
    )

    axes[1].plot(methyl_ppm, profiles["methyl_x"], label="Ix -> Ix")
    axes[1].plot(methyl_ppm, profiles["methyl_y"], label="-Iy -> Iy")
    axes[1].plot(methyl_ppm, profiles["methyl_z"], label="Iz -> -Iz")
    axes[1].axhline(MIN_METHYL_FIDELITY, color="black", linestyle=":")
    axes[1].set_xlim(METHYL_PPM_LO, METHYL_PPM_HI)
    axes[1].set_ylim(0.9985, 1.0001)
    axes[1].set_ylabel("Transfer fidelity")
    axes[1].legend()

    axes[2].semilogy(methyl_ppm, profiles["artifact_percent"], color="tab:green")
    axes[2].axhline(MAX_ARTIFACT_PERCENT, color="black", linestyle=":")
    axes[2].set_xlim(METHYL_PPM_LO, METHYL_PPM_HI)
    axes[2].set_ylabel("Kay artifact (%)")
    axes[2].set_xlabel("Methyl 1H shift (ppm)")

    axes[3].plot(profiles["water_offsets_hz"], profiles["water_z"], marker="o")
    axes[3].axhline(MIN_WATER_FIDELITY, color="black", linestyle=":")
    axes[3].set_ylim(0.9985, 1.0001)
    axes[3].set_xlabel("Offset from water at 4.7 ppm (Hz)")
    axes[3].set_ylabel("Water Iz -> Iz")
    axes[3].set_title(
        f"Worst: methyl x={metrics.methyl_x_min:.6f}, "
        f"y={metrics.methyl_y_min:.6f}, "
        f"z={metrics.methyl_z_min:.6f}, water={metrics.water_z_min:.6f}, "
        f"artifact={metrics.artifact_max_percent:.4f}%"
    )

    figure_path = output_dir / f"{PULSE_NAME}.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_pareto(output_dir: Path) -> Path:
    """Write the (max-power, min-duration) trade-off frontier as a PNG."""
    rf_khz = np.array([rf / 1000.0 for rf, _ in FRONTIER])
    dur_ms = np.array([dur / 1000.0 for _, dur in FRONTIER])
    figure, axis = plt.subplots(1, 1, figsize=(7.5, 4.5), constrained_layout=True)
    axis.plot(rf_khz, dur_ms, marker="o", color="tab:purple")
    axis.scatter(
        [RF_MAX_HZ / 1000.0],
        [DURATION_US / 1000.0],
        s=120,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=2.0,
        zorder=5,
        label="cached (min-length)",
    )
    axis.set_xlabel("Peak RF amplitude (kHz)")
    axis.set_ylabel("Shortest passing duration (ms)")
    axis.set_title("Max-power vs length trade-off (REBURP-style methyl 180)")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure_path = output_dir / f"{PULSE_NAME}_pareto.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_wide_profile(wide_profile: dict[str, RealArray], output_dir: Path) -> Path:
    """Write Ix, -Iy, Iz transfer profiles over the full -6..6 ppm window."""
    ppm = wide_profile["ppm"]
    figure, axis = plt.subplots(1, 1, figsize=(9.0, 4.5), constrained_layout=True)
    axis.plot(ppm, wide_profile["x"], label="Ix -> Ix")
    axis.plot(ppm, wide_profile["y"], label="Iy -> -Iy")
    axis.plot(ppm, wide_profile["z"], label="Iz -> -Iz")
    axis.axvline(METHYL_PPM_LO, color="black", linestyle=":", linewidth=0.8)
    axis.axvline(METHYL_PPM_HI, color="black", linestyle=":", linewidth=0.8)
    axis.axvline(WATER_PPM, color="tab:gray", linestyle="--", linewidth=0.8)
    axis.set_xlim(WIDE_PPM_LO, WIDE_PPM_HI)
    axis.set_xlabel("1H shift (ppm)")
    axis.set_ylabel("Transfer fidelity")
    axis.set_title("Wide-offset transfer profile (dotted: methyl band, dashed: water)")
    axis.legend()

    figure_path = output_dir / f"{PULSE_NAME}_wide_profile.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_wide_components(component_profile: dict[str, RealArray], output_dir: Path) -> Path:
    """Write final Mx, My, Mz vs ppm for each initial Ix, Iy, Iz state."""
    ppm = component_profile["ppm"]
    figure, axes = plt.subplots(
        3, 1, figsize=(9.0, 9.0), sharex=True, constrained_layout=True
    )
    for axis, name in zip(axes, ("Ix", "Iy", "Iz")):
        final = component_profile[name]
        axis.plot(ppm, final[:, 0], label="Mx", color="tab:blue")
        axis.plot(ppm, final[:, 1], label="My", color="tab:orange")
        axis.plot(ppm, final[:, 2], label="Mz", color="tab:green")
        axis.axvline(METHYL_PPM_LO, color="black", linestyle=":", linewidth=0.8)
        axis.axvline(METHYL_PPM_HI, color="black", linestyle=":", linewidth=0.8)
        axis.axvline(WATER_PPM, color="tab:gray", linestyle="--", linewidth=0.8)
        axis.set_xlim(WIDE_PPM_LO, WIDE_PPM_HI)
        axis.set_ylim(-1.05, 1.05)
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.set_ylabel(f"from {name}")
    axes[0].legend(ncol=3, fontsize="small")
    axes[0].set_title("Final Bloch components vs offset (dotted: methyl band, dashed: water)")
    axes[-1].set_xlabel("1H shift (ppm)")

    figure_path = output_dir / f"{PULSE_NAME}_wide_components.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_time_evolution(time_evolution: dict[str, RealArray], output_dir: Path) -> Path:
    """Write Mx, My, Mz time evolution per initial state and offset as a PNG."""
    time_us = time_evolution["time_us"]
    ppms = time_evolution["ppm"]
    inits = ("Ix", "Iy", "Iz")
    n_rows = len(inits)
    n_cols = ppms.size

    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.4 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row, name in enumerate(inits):
        trajectory = time_evolution[name]
        for col in range(n_cols):
            axis = axes[row, col]
            axis.plot(time_us, trajectory[:, col, 0], label="Mx", color="tab:blue")
            axis.plot(time_us, trajectory[:, col, 1], label="My", color="tab:orange")
            axis.plot(time_us, trajectory[:, col, 2], label="Mz", color="tab:green")
            axis.axhline(0.0, color="black", linewidth=0.5)
            axis.set_ylim(-1.05, 1.05)
            if row == 0:
                axis.set_title(f"{ppms[col]:g} ppm")
            if col == 0:
                axis.set_ylabel(f"from {name}")
            if row == n_rows - 1:
                axis.set_xlabel("Time (us)")
    axes[0, 0].legend(ncol=3, fontsize="x-small", loc="lower left")
    figure.suptitle(
        f"Bloch-component time evolution "
        f"({DURATION_US:.0f} us, {RF_MAX_HZ / 1000.0:.1f} kHz REBURP-style pulse)"
    )

    figure_path = output_dir / f"{PULSE_NAME}_time_evolution.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def run() -> RealArray:
    """Generate artifacts and return waveform plus dense summary metrics."""
    signed = signed_amplitude()
    metrics, profiles = evaluate_pulse(
        signed, duration_s=DURATION_S, rf_max_hz=RF_MAX_HZ
    )
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_bruker_shape(signed, output_dir)
    plot_diagnostics(signed, metrics, profiles, output_dir)
    plot_pareto(output_dir)
    plot_wide_profile(
        wide_transfer_profile(signed, duration_s=DURATION_S, rf_max_hz=RF_MAX_HZ),
        output_dir,
    )
    plot_wide_components(
        wide_component_profile(signed, duration_s=DURATION_S, rf_max_hz=RF_MAX_HZ),
        output_dir,
    )
    plot_time_evolution(
        time_evolution_profiles(
            signed, ppms=TIME_EVOLUTION_PPM, duration_s=DURATION_S, rf_max_hz=RF_MAX_HZ
        ),
        output_dir,
    )
    return np.concatenate((signed, metrics.as_array()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="repeat the (max-power, duration) Pareto search",
    )
    args = parser.parse_args()
    if args.optimize:
        frontier = search_pareto()
        for rf_max_hz, duration_us, _, metrics in frontier:
            print(
                f"{rf_max_hz / 1000.0:.1f} kHz -> {duration_us:.0f} us "
                f"pass={metrics.passes} {metrics}"
            )
        rf_max_hz, duration_us, half, _ = min(frontier, key=lambda c: c[0])
        print(f"Minimum-power passing pulse: {rf_max_hz / 1000.0:.1f} kHz, {duration_us:.0f} us")
        print("Optimized half waveform:")
        print(repr(half.tolist()))
        return

    result = run()
    metrics, _ = evaluate_pulse(result[:N_STEPS], duration_s=DURATION_S, rf_max_hz=RF_MAX_HZ)
    print(metrics)
    print(f"Saved examples/output/{PULSE_NAME}.shape")
    print(f"Saved examples/output/{PULSE_NAME}.png")


if __name__ == "__main__":
    main()
