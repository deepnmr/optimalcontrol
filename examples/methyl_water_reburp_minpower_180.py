"""Minimum max-power / minimum-length REBURP-style methyl 180 over water.

This is a power-and-duration-minimizing sibling of
``methyl_water_reburp_180``. It solves the same physical problem -- the
refocusing-pulse artifact of Lewis E. Kay, J. Biomol. NMR 73, 423-427 (2019),
DOI 10.1007/s10858-019-00227-7, in which an imperfect central proton 180-degree
pulse produces methyl HMQC satellites at +/-J_CH/2 that survive gradients and
phase cycling -- but instead of fixing the RF amplitude at 10 kHz it searches
the joint (max-power, duration) plane for the smallest peak field.

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

Below 6.0 kHz the search no longer converges within a practical duration (5.5
kHz fails even at 2.6-4.5 ms), so 6.0 kHz is the practical minimum peak field.
The curve is flat from 10 down to about 7.5 kHz (the minimum length, 1.80 ms,
lives at the high-power end), then the duration climbs steeply toward the
minimum-power end. The cached pulse is that minimum-power point: 6.0 kHz at
2.60 ms -- a 40 percent lower peak field than the 10 kHz siblings.

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

from examples import _methyl_water_common as minpower_helpers
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

PULSE_NAME = "methyl_water_reburp_minpower_180"
N_STEPS = 200

# Cached minimum-power knee of the Pareto frontier (see module docstring).
RF_MAX_HZ = 6000.0
DURATION_US = 2600.0
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
        -0.4683536832020614,
        -0.08529545812289409,
        0.17943406237384016,
        0.2587225187392136,
        0.19153935888757675,
        0.07810437968926905,
        -0.0020811630046084196,
        -0.04205951217005602,
        -0.0966733356825336,
        -0.22165645334488052,
        -0.4081743199546405,
        -0.5624143142953448,
        -0.5507311106618762,
        -0.28936333368085565,
        0.17746630769125962,
        0.6690870427304916,
        0.9462767966961642,
        0.842476470534877,
        0.37215452167853974,
        -0.2628830061119434,
        -0.783340014420231,
        -1.0,
        -0.9543797445619668,
        -0.6359390047224984,
        -0.21277982487869723,
        0.1183090629104582,
        0.2508710930451132,
        0.21002754229364046,
        0.11125925537188222,
        0.07382141728484513,
        0.14149505273061716,
        0.25260050429964087,
        0.2774337509373352,
        0.10485978288119203,
        -0.26994904271949655,
        -0.7103211492870136,
        -1.0,
        -1.0,
        -0.7543814862406585,
        -0.1663296184531881,
        0.5362374011757207,
        1.0,
        1.0,
        0.5733014239161728,
        -0.08868021136745292,
        -0.6378941178007707,
        -0.7919803457264181,
        -0.492796653929187,
        0.07710123802561433,
        0.6162922670821606,
        0.8783572954572292,
        0.7968235272219159,
        0.4974645529765599,
        0.19978881459805573,
        0.08414552737200579,
        0.20341031738695653,
        0.4790293796553562,
        0.7676153526477901,
        0.9516971832296152,
        1.0,
        0.9604539667195409,
        0.8686053526356499,
        0.7213407013493244,
        0.47353085932792804,
        0.09442154881826634,
        -0.36542486496634324,
        -0.7550593226200387,
        -0.8819126707694496,
        -0.6293761148435764,
        -0.05834605928083153,
        0.5887245120726571,
        1.0,
        0.9656617907198004,
        0.5005687130586864,
        -0.16075327374746592,
        -0.7074232799591422,
        -0.9314264018414596,
        -0.8351500774889368,
        -0.6008235751297333,
        -0.4510683896492534,
        -0.4975940153802699,
        -0.6774912411414435,
        -0.8059024400467912,
        -0.7105436696229586,
        -0.3588367288187942,
        0.1038810813722371,
        0.4343273327177881,
        0.43633654340876743,
        0.08658299754783481,
        -0.43756798369525524,
        -0.8588101720179483,
        -1.0,
        -1.0,
        -0.7896822491208948,
        -0.28723617913779903,
        0.32376988649931215,
        0.79816354733204,
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float64,
)


def signed_amplitude(half: RealArray = OPTIMIZED_HALF_AMPLITUDE) -> RealArray:
    return minpower_helpers.signed_amplitude(half, N_STEPS=N_STEPS)


def _control_problem(
    offsets_hz: RealArray,
    rho_init: list[ComplexArray],
    rho_targ: list[ComplexArray],
    duration_s: float,
    rf_max_hz: float,
) -> ControlProblem:
    return minpower_helpers._control_problem(
        offsets_hz, rho_init, rho_targ, duration_s, rf_max_hz, N_STEPS=N_STEPS
    )


def _collapse_symmetric_gradient(gradient: RealArray) -> RealArray:
    return minpower_helpers._collapse_symmetric_gradient(gradient, N_STEPS=N_STEPS)


def refine_pulse(
    duration_us: float,
    rf_max_hz: float,
    half_seed: RealArray,
    smoothness_weight: float = SMOOTHNESS_WEIGHT,
    max_iter: int = 1200,
) -> tuple[RealArray, PulseMetrics]:
    return minpower_helpers.refine_pulse(
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
    return minpower_helpers.search_pareto(
        rf_grid_hz,
        duration_grid_us,
        OPTIMIZED_HALF_AMPLITUDE=OPTIMIZED_HALF_AMPLITUDE,
        refine_pulse=refine_pulse,
    )


def export_bruker_shape(signed: RealArray, output_dir: Path) -> Path:
    return minpower_helpers.export_bruker_shape(
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
        f"Min-power REBURP-style methyl 180: {DURATION_US:.0f} us, {RF_MAX_HZ / 1000.0:.1f} kHz max"
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
    return minpower_helpers.plot_pareto(
        output_dir,
        PULSE_NAME=PULSE_NAME,
        FRONTIER=FRONTIER,
        RF_MAX_HZ=RF_MAX_HZ,
        DURATION_US=DURATION_US,
        label="cached (min-power knee)",
    )


def plot_wide_profile(wide_profile: dict[str, RealArray], output_dir: Path) -> Path:
    return minpower_helpers.plot_wide_profile(
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
    return minpower_helpers.plot_wide_components(
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
    return minpower_helpers.plot_time_evolution(
        time_evolution,
        output_dir,
        PULSE_NAME=PULSE_NAME,
        title=(
            f"Bloch-component time evolution "
            f"({DURATION_US:.0f} us, {RF_MAX_HZ / 1000.0:.1f} kHz REBURP-style pulse)"
        ),
    )


def run() -> RealArray:
    return minpower_helpers.run(
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
    minpower_helpers.main(
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
