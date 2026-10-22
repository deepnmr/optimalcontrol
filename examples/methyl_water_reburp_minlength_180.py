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

import argparse as argparse
import math as math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize as minimize

from examples import _methyl_water_common as minlength_helpers
from examples._figures import save_figure
from examples._methyl_water_common import _smoothness as _smoothness
from examples._methyl_water_common import write_bruker_shape as write_bruker_shape

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
from optimalcontrol.grape import ControlProblem
from optimalcontrol.grape import grape_xy_and_gradient as grape_xy_and_gradient
from optimalcontrol.operators import (
    Ix as Ix,
)
from optimalcontrol.operators import (
    Iy as Iy,
)
from optimalcontrol.operators import (
    Iz as Iz,
)
from optimalcontrol.operators import (
    liouvillian_comm as liouvillian_comm,
)
from optimalcontrol.operators import (
    vec as vec,
)
from optimalcontrol.states import normalise_hs as normalise_hs

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
        -0.14192559532933624,
        -0.03704501176435874,
        0.05158900935758466,
        0.10434688927143171,
        0.1081006621831702,
        0.062455342603668605,
        -0.01817186951881482,
        -0.10855152696385847,
        -0.1808531280868793,
        -0.21200325856592614,
        -0.18991022274229064,
        -0.11828325039423226,
        -0.01641174535071956,
        0.08707218582727208,
        0.1619978338452024,
        0.18385498445480458,
        0.13977527309142748,
        0.03241898479261897,
        -0.11966418032264932,
        -0.28479634923414715,
        -0.4248435244661069,
        -0.5046169617583475,
        -0.5011681299027942,
        -0.40968343762157333,
        -0.24542985722069446,
        -0.04172887505995483,
        0.15547168036553527,
        0.2967207206817997,
        0.3406363769226618,
        0.2657781989174879,
        0.07801255303107749,
        -0.187860586684636,
        -0.47285463186514215,
        -0.7060485633864347,
        -0.822918731309153,
        -0.7845123167859742,
        -0.5913819316041886,
        -0.2861541738589543,
        0.05690682121883165,
        0.3529506688980467,
        0.5308665230850819,
        0.5534310848983511,
        0.4280964530364903,
        0.20402609034890018,
        -0.0447076110979988,
        -0.24201053179845955,
        -0.3309916743544786,
        -0.2894447766740929,
        -0.13454544455183576,
        0.08610591012362044,
        0.31224316599824087,
        0.49010003066975805,
        0.5884530378173785,
        0.6065806011450019,
        0.5705060358296603,
        0.5196124445229043,
        0.4893767554794785,
        0.4955718025229333,
        0.5264659128728092,
        0.5481519652286163,
        0.5207525796659916,
        0.41705301250676174,
        0.23731685041886283,
        0.01484225931969178,
        -0.19237990804716687,
        -0.32148393946008236,
        -0.3271460653509745,
        -0.19914023388856572,
        0.029818190099634007,
        0.28966828341994866,
        0.49287841758830603,
        0.5619170925877313,
        0.4560488205312146,
        0.18922371259134538,
        -0.17226107104646649,
        -0.5321119816238737,
        -0.7950014796640754,
        -0.8971438185706162,
        -0.8243384348078453,
        -0.610642461213518,
        -0.3236071628515354,
        -0.0414783953034517,
        0.17038635400387867,
        0.2758670839998437,
        0.2719461413670707,
        0.1804872626104752,
        0.03360683892206752,
        -0.1384556199131672,
        -0.30995949353989,
        -0.4564963776667766,
        -0.5550624054066454,
        -0.5861173773305621,
        -0.5349784226732618,
        -0.39366552260151827,
        -0.1651565964252656,
        0.13042010462655568,
        0.4511629137757451,
        0.7365753928572312,
        0.927245530349415,
        1.0,
    ],
    dtype=np.float64,
)


def signed_amplitude(half: RealArray = OPTIMIZED_HALF_AMPLITUDE) -> RealArray:
    return minlength_helpers.signed_amplitude(half, N_STEPS=N_STEPS)


def _control_problem(
    offsets_hz: RealArray,
    rho_init: list[ComplexArray],
    rho_targ: list[ComplexArray],
    duration_s: float,
    rf_max_hz: float,
) -> ControlProblem:
    return minlength_helpers._control_problem(
        offsets_hz, rho_init, rho_targ, duration_s, rf_max_hz, N_STEPS=N_STEPS
    )


def _collapse_symmetric_gradient(gradient: RealArray) -> RealArray:
    return minlength_helpers._collapse_symmetric_gradient(gradient, N_STEPS=N_STEPS)


def refine_pulse(
    duration_us: float,
    rf_max_hz: float,
    half_seed: RealArray,
    smoothness_weight: float = SMOOTHNESS_WEIGHT,
    max_iter: int = 1200,
) -> tuple[RealArray, PulseMetrics]:
    return minlength_helpers.refine_pulse(
        duration_us,
        rf_max_hz,
        half_seed,
        smoothness_weight,
        max_iter,
        N_STEPS=N_STEPS,
        METHYL_OFFSET_LO_HZ=METHYL_OFFSET_LO_HZ,
        METHYL_OFFSET_HI_HZ=METHYL_OFFSET_HI_HZ,
        WATER_WINDOW_HZ=WATER_WINDOW_HZ,
        VALIDATION_WATER_POINTS=VALIDATION_WATER_POINTS,
        _control_problem=_control_problem,
        signed_amplitude=signed_amplitude,
        _collapse_symmetric_gradient=_collapse_symmetric_gradient,
        evaluate_pulse=evaluate_pulse,
    )


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
    return minlength_helpers.search_pareto(
        rf_grid_hz,
        duration_grid_us,
        OPTIMIZED_HALF_AMPLITUDE=OPTIMIZED_HALF_AMPLITUDE,
        refine_pulse=refine_pulse,
    )


def export_bruker_shape(signed: RealArray, output_dir: Path) -> Path:
    return minlength_helpers.export_bruker_shape(
        signed,
        output_dir,
        PULSE_NAME=PULSE_NAME,
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
        amplitude_phase=amplitude_phase,
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
    save_figure(figure, figure_path, dpi=160)
    return figure_path


def plot_pareto(output_dir: Path) -> Path:
    return minlength_helpers.plot_pareto(
        output_dir,
        PULSE_NAME=PULSE_NAME,
        FRONTIER=FRONTIER,
        RF_MAX_HZ=RF_MAX_HZ,
        DURATION_US=DURATION_US,
        label="cached (min-length)",
    )


def plot_wide_profile(wide_profile: dict[str, RealArray], output_dir: Path) -> Path:
    return minlength_helpers.plot_wide_profile(
        wide_profile,
        output_dir,
        PULSE_NAME=PULSE_NAME,
        METHYL_PPM_LO=METHYL_PPM_LO,
        METHYL_PPM_HI=METHYL_PPM_HI,
        WATER_PPM=WATER_PPM,
        WIDE_PPM_LO=WIDE_PPM_LO,
        WIDE_PPM_HI=WIDE_PPM_HI,
    )


def plot_wide_components(component_profile: dict[str, RealArray], output_dir: Path) -> Path:
    return minlength_helpers.plot_wide_components(
        component_profile,
        output_dir,
        PULSE_NAME=PULSE_NAME,
        METHYL_PPM_LO=METHYL_PPM_LO,
        METHYL_PPM_HI=METHYL_PPM_HI,
        WATER_PPM=WATER_PPM,
        WIDE_PPM_LO=WIDE_PPM_LO,
        WIDE_PPM_HI=WIDE_PPM_HI,
    )


def plot_time_evolution(time_evolution: dict[str, RealArray], output_dir: Path) -> Path:
    return minlength_helpers.plot_time_evolution(
        time_evolution,
        output_dir,
        PULSE_NAME=PULSE_NAME,
        title=(
            f"Bloch-component time evolution "
            f"({DURATION_US:.0f} us, {RF_MAX_HZ / 1000.0:.1f} kHz REBURP-style pulse)"
        ),
    )


def run() -> RealArray:
    return minlength_helpers.run(
        signed_amplitude=signed_amplitude,
        DURATION_S=DURATION_S,
        RF_MAX_HZ=RF_MAX_HZ,
        TIME_EVOLUTION_PPM=TIME_EVOLUTION_PPM,
        export_bruker_shape=export_bruker_shape,
        plot_diagnostics=plot_diagnostics,
        plot_pareto=plot_pareto,
        plot_wide_profile=plot_wide_profile,
        plot_wide_components=plot_wide_components,
        plot_time_evolution=plot_time_evolution,
        evaluate_pulse=evaluate_pulse,
        wide_transfer_profile=wide_transfer_profile,
        wide_component_profile=wide_component_profile,
        time_evolution_profiles=time_evolution_profiles,
    )


def main() -> None:
    minlength_helpers.main(
        __doc__=__doc__,
        search_pareto=search_pareto,
        run=run,
        N_STEPS=N_STEPS,
        DURATION_S=DURATION_S,
        RF_MAX_HZ=RF_MAX_HZ,
        PULSE_NAME=PULSE_NAME,
        evaluate_pulse=evaluate_pulse,
    )


if __name__ == "__main__":
    main()
