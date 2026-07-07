"""REBURP-style band-selective methyl 180 that preserves water magnetization.

This is a smooth, REBURP-like sibling of
``methyl_water_binary_symmetric_180``. Both pulses solve the same physical
problem -- the refocusing-pulse artifact described by Lewis E. Kay,
J. Biomol. NMR 73, 423-427 (2019), DOI 10.1007/s10858-019-00227-7, in which
an imperfect central proton 180-degree pulse produces methyl HMQC satellites
at +/-J_CH/2 that survive gradients and phase cycling.

Where the sibling pulse is a jagged grid-optimized waveform, this one is shaped
like a classic band-selective REBURP pulse (Geen & Freeman, J. Magn. Reson. 93,
93-141, 1991): the optimization is regularized for smoothness, so the
signed amplitude is a smooth band-selective envelope (roughness about 1.6
versus about 50 for the sibling) rather than a noisy waveform. The smoothness
costs duration -- the shortest passing REBURP-style candidate is 1.950 ms,
against 1.740 ms for the jagged sibling.

Design specification
--------------------

* 1.2 GHz proton spectrometer, carrier at water (4.7 ppm).
* Methyl range -3.0 to 3.0 ppm: true 180_x, tested as Ix -> Ix,
  -Iy -> Iy, and Iz -> -Iz. The simultaneous three-axis fidelity makes the
  net propagator a universal (pure-phase) rotation, so methyl TROSY is
  preserved and no net 1H chemical-shift evolution accrues across the pulse.
* Water: Iz -> Iz over 4.7 ppm +/-100 Hz.
* RF amplitude is variable but never exceeds 10 kHz.
* RF phase is binary: exactly 0 or 180 degrees.
* The signed-amplitude waveform is exactly time symmetric.

The half-waveform is the optimization variable. Mirroring it enforces time
symmetry structurally; a positive signed amplitude exports as phase 0 and a
negative one as phase 180. The objective adds a second-difference smoothness
penalty to the four-state GRAPE fidelity, giving the REBURP-like envelope.

Pass criteria on 2401 methyl offsets and 9 water offsets are:

* worst methyl Ix -> Ix, -Iy -> Iy, and Iz -> -Iz fidelity >= 0.999;
* worst water Iz -> Iz fidelity >= 0.999; and
* worst Kay inner-sideband artifact <= 0.1 percent of the central line.

Run without arguments to write the cached Bruker shape and diagnostic plots.
Use ``--optimize`` to repeat the smooth duration-grid refinement.
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
# constants, and the wide/time profile computations; all are step-count and
# duration agnostic, so they are reused here rather than duplicated.
from examples.methyl_water_binary_symmetric_180 import (
    MAX_ARTIFACT_PERCENT,
    METHYL_OFFSET_HI_HZ,
    METHYL_OFFSET_LO_HZ,
    METHYL_PPM_HI,
    METHYL_PPM_LO,
    MIN_METHYL_FIDELITY,
    MIN_WATER_FIDELITY,
    RF_MAX_HZ,
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

PULSE_NAME = "methyl_water_reburp_180"
N_STEPS = 200
DURATION_US = 1950.0
DURATION_S = DURATION_US * 1.0e-6
DT = DURATION_S / N_STEPS

SMOOTHNESS_WEIGHT = 3.0e-4
SEARCH_STEP_US = 25.0


# Optimized half of the smooth signed-amplitude waveform. The second half is
# its reverse. Values are fractions of RF_MAX_HZ and lie in [-1, 1].
OPTIMIZED_HALF_AMPLITUDE: RealArray = np.array(
    [
        0.00791228901980791, 0.023363150634418052, 0.02337514693473912,
        0.006194031210964428, -0.021222419431978987, -0.047234133756759986,
        -0.05927762889767678, -0.045618642492576404, -0.007989481467720851,
        0.04180688703414207, 0.0816768606321306, 0.09163138058431024,
        0.05897827868189659, -0.006894285373223169, -0.08895887549018035,
        -0.16035523760718245, -0.1996599186775729, -0.18852959564765384,
        -0.13090540677126752, -0.04162706394423894, 0.050263424530701616,
        0.11931939522089041, 0.13939416691879178, 0.09771651352455978,
        -0.009595379377086056, -0.16328043713673773, -0.32958433116375824,
        -0.4632662527765346, -0.5260126794752676, -0.4923732635821417,
        -0.3620015082897069, -0.15945438688347174, 0.06318880180099758,
        0.2399442304897217, 0.31046852190863206, 0.2426276310317286,
        0.04152922362132415, -0.2471662257785505, -0.5441870717730304,
        -0.7558959755761151, -0.8048827081564246, -0.6614827493633816,
        -0.35934484947577694, 0.013663929381864556, 0.3487812725757621,
        0.5514679477552941, 0.5728499808597232, 0.42469229546987347,
        0.1782253874890559, -0.0669189665503394, -0.21864275370348654,
        -0.227566299739227, -0.09573178100697025, 0.12598729084660076,
        0.3646709946209379, 0.5484232101431725, 0.6376533023844949,
        0.6309053635781247, 0.5637134598559308, 0.48613387136292796,
        0.4410718526029824, 0.43655203964219785, 0.44373018779660217,
        0.4194234640552976, 0.33592251131129214, 0.18424890019587825,
        -0.0134989472007315, -0.204819093234904, -0.31647238862468974,
        -0.30045230009296453, -0.1530593215474061, 0.07596414070800524,
        0.3082014750029438, 0.45112938255384644, 0.43660517481257605,
        0.23102613482513498, -0.10984250347669179, -0.4807332555429161,
        -0.7605495889980334, -0.8761149902113966, -0.790862132392648,
        -0.5370009593973355, -0.19516866014863887, 0.1185187917496854,
        0.322105928436273, 0.3728170132057104, 0.2930729663643911,
        0.11227076567031113, -0.1209021925856861, -0.3496399233356125,
        -0.5204785011310874, -0.6068066223707045, -0.5877650693301122,
        -0.46362910697739484, -0.23885903933106742, 0.05975468874861916,
        0.39059508784189817, 0.686710560070611, 0.8960721784552923, 1.0,
    ],
    dtype=np.float64,
)


def signed_amplitude(half: RealArray = OPTIMIZED_HALF_AMPLITUDE) -> RealArray:
    """Return the exactly symmetric signed RF-amplitude waveform."""
    half_array = np.asarray(half, dtype=np.float64)
    if half_array.shape != (N_STEPS // 2,):
        raise ValueError(f"half waveform must have shape ({N_STEPS // 2},)")
    if np.max(np.abs(half_array)) > 1.0 + 1e-12:
        raise ValueError("signed amplitude exceeds the 10 kHz RF limit")
    return np.asarray(np.concatenate((half_array, half_array[::-1])), dtype=np.float64)


def _control_problem(
    offsets_hz: RealArray,
    rho_init: list[ComplexArray],
    rho_targ: list[ComplexArray],
    duration_s: float,
) -> ControlProblem:
    """Build a one-channel, nominal-B1 Liouville optimization problem."""
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=np.complex128)],
        operators=[2.0 * math.pi * RF_MAX_HZ * liouvillian_comm(Ix())],
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


def _constraint_score(metrics: PulseMetrics) -> float:
    """Return the worst normalized constraint, with 0.999 as the pass boundary."""
    return min(
        metrics.methyl_x_min,
        metrics.methyl_y_min,
        metrics.methyl_z_min,
        metrics.water_z_min,
        1.0 - metrics.artifact_max_percent / 100.0,
    )


def refine_duration(
    duration_us: float,
    half_seed: RealArray,
    max_iter: int = 1200,
    smoothness_weight: float = SMOOTHNESS_WEIGHT,
) -> tuple[RealArray, PulseMetrics]:
    """Refine one duration with a smoothness-regularized four-state objective."""
    duration_s = duration_us * 1e-6
    methyl_offsets = np.linspace(
        METHYL_OFFSET_LO_HZ, METHYL_OFFSET_HI_HZ, 81, dtype=np.float64
    )
    water_offsets = np.linspace(
        -WATER_WINDOW_HZ, WATER_WINDOW_HZ, VALIDATION_WATER_POINTS, dtype=np.float64
    )
    cp_x = _control_problem(methyl_offsets, [Ix()], [Ix()], duration_s)
    cp_y = _control_problem(methyl_offsets, [-Iy()], [Iy()], duration_s)
    cp_z = _control_problem(methyl_offsets, [Iz()], [-Iz()], duration_s)
    cp_water = _control_problem(water_offsets, [Iz()], [Iz()], duration_s)

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
    metrics, _ = evaluate_pulse(signed_amplitude(half), duration_s=duration_s)
    return half, metrics


def search_minimum_duration() -> tuple[float, RealArray, list[tuple[float, PulseMetrics]]]:
    """Scan durations downward and return the shortest passing smooth pulse."""
    half = OPTIMIZED_HALF_AMPLITUDE.copy()
    audit: list[tuple[float, PulseMetrics]] = []
    passing: list[tuple[float, RealArray, PulseMetrics]] = []
    for duration_us in (2025.0, 2000.0, 1975.0, 1950.0, 1925.0):
        refined, metrics = refine_duration(duration_us, half)
        audit.append((duration_us, metrics))
        if metrics.passes:
            passing.append((duration_us, refined.copy(), metrics))
        half = refined
    if not passing:
        raise RuntimeError("no duration candidate met the pulse constraints")
    duration_us, best_half, _ = min(passing, key=lambda candidate: candidate[0])
    return duration_us, best_half, audit


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
        f"REBURP-style binary-phase methyl 180: {DURATION_US:.0f} us, 10 kHz max"
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
    figure.suptitle(f"Bloch-component time evolution ({DURATION_US:.0f} us REBURP-style pulse)")

    figure_path = output_dir / f"{PULSE_NAME}_time_evolution.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def run() -> RealArray:
    """Generate artifacts and return waveform plus dense summary metrics."""
    signed = signed_amplitude()
    metrics, profiles = evaluate_pulse(signed, duration_s=DURATION_S)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_bruker_shape(signed, output_dir)
    plot_diagnostics(signed, metrics, profiles, output_dir)
    plot_wide_profile(wide_transfer_profile(signed, duration_s=DURATION_S), output_dir)
    plot_wide_components(
        wide_component_profile(signed, duration_s=DURATION_S), output_dir
    )
    plot_time_evolution(
        time_evolution_profiles(signed, ppms=TIME_EVOLUTION_PPM, duration_s=DURATION_S),
        output_dir,
    )
    return np.concatenate((signed, metrics.as_array()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimize", action="store_true", help="repeat the smooth duration-grid refinement"
    )
    args = parser.parse_args()
    if args.optimize:
        duration_us, half, audit = search_minimum_duration()
        for candidate_us, metrics in audit:
            print(f"{candidate_us:.0f} us: pass={metrics.passes} {metrics}")
        print(f"Shortest passing grid candidate: {duration_us:.0f} us")
        print("Optimized half waveform:")
        print(repr(half.tolist()))
        return

    result = run()
    metrics, _ = evaluate_pulse(result[:N_STEPS], duration_s=DURATION_S)
    print(metrics)
    print(f"Saved examples/output/{PULSE_NAME}.shape")
    print(f"Saved examples/output/{PULSE_NAME}.png")


if __name__ == "__main__":
    main()
