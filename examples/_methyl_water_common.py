"""Shared implementations for the methyl/water pulse examples."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from examples._figures import quantise, save_figure
from optimalcontrol.grape import ControlProblem, grape_xy_and_gradient
from optimalcontrol.io import export_bruker_shape as _write_bruker_shape
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, vec
from optimalcontrol.states import normalise_hs

if TYPE_CHECKING:
    from examples.methyl_water_binary_symmetric_180 import PulseMetrics

RealArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
write_bruker_shape = _write_bruker_shape


class SignedAmplitude(Protocol):
    def __call__(self, half: RealArray = ...) -> RealArray: ...


class EvaluatePulse(Protocol):
    def __call__(
        self, signed: RealArray, *, duration_s: float, rf_max_hz: float
    ) -> tuple[PulseMetrics, dict[str, RealArray]]: ...


class WideProfile(Protocol):
    def __call__(
        self, signed: RealArray, *, duration_s: float, rf_max_hz: float
    ) -> dict[str, RealArray]: ...


class TimeProfiles(Protocol):
    def __call__(
        self, signed: RealArray, *, ppms: tuple[float, ...], duration_s: float, rf_max_hz: float
    ) -> dict[str, RealArray]: ...


def plot_wide_profile(
    wide_profile: dict[str, RealArray],
    output_dir: Path,
    *,
    PULSE_NAME: str,
    METHYL_PPM_LO: float,
    METHYL_PPM_HI: float,
    WATER_PPM: float,
    WIDE_PPM_LO: float,
    WIDE_PPM_HI: float,
) -> Path:
    """Write Ix, -Iy, Iz transfer profiles over the full -6..6 ppm window."""
    ppm = wide_profile["ppm"]
    figure, axis = plt.subplots(1, 1, figsize=(9.0, 4.5), constrained_layout=True)
    axis.plot(ppm, quantise(wide_profile["x"]), label="Ix -> Ix")
    axis.plot(ppm, quantise(wide_profile["y"]), label="Iy -> -Iy")
    axis.plot(ppm, quantise(wide_profile["z"]), label="Iz -> -Iz")
    axis.axvline(METHYL_PPM_LO, color="black", linestyle=":", linewidth=0.8)
    axis.axvline(METHYL_PPM_HI, color="black", linestyle=":", linewidth=0.8)
    axis.axvline(WATER_PPM, color="tab:gray", linestyle="--", linewidth=0.8)
    axis.set_xlim(WIDE_PPM_LO, WIDE_PPM_HI)
    axis.set_xlabel("1H shift (ppm)")
    axis.set_ylabel("Transfer fidelity")
    axis.set_title("Wide-offset transfer profile (dotted: methyl band, dashed: water)")
    axis.legend()

    figure_path = output_dir / f"{PULSE_NAME}_wide_profile.png"
    save_figure(figure, figure_path, dpi=160)
    return figure_path


def plot_wide_components(
    component_profile: dict[str, RealArray],
    output_dir: Path,
    *,
    PULSE_NAME: str,
    METHYL_PPM_LO: float,
    METHYL_PPM_HI: float,
    WATER_PPM: float,
    WIDE_PPM_LO: float,
    WIDE_PPM_HI: float,
) -> Path:
    """Write final Mx, My, Mz vs ppm for each initial Ix, Iy, Iz state."""
    ppm = component_profile["ppm"]
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, ("Ix", "Iy", "Iz"), strict=True):
        final = component_profile[name]
        axis.plot(ppm, quantise(final[:, 0]), label="Mx", color="tab:blue")
        axis.plot(ppm, quantise(final[:, 1]), label="My", color="tab:orange")
        axis.plot(ppm, quantise(final[:, 2]), label="Mz", color="tab:green")
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
    save_figure(figure, figure_path, dpi=160)
    return figure_path


def _smoothness(signed: RealArray) -> tuple[float, RealArray]:
    """Return the second-difference roughness and its gradient w.r.t. steps."""
    d2 = np.diff(signed, 2)
    value = float(np.sum(d2**2))
    gradient = np.zeros(signed.size, dtype=np.float64)
    gradient[:-2] += 2.0 * d2
    gradient[1:-1] += -4.0 * d2
    gradient[2:] += 2.0 * d2
    return value, gradient


def _collapse_symmetric_gradient(
    gradient: RealArray,
    *,
    N_STEPS: int,
) -> RealArray:
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


def signed_amplitude(
    half: RealArray,
    *,
    N_STEPS: int,
) -> RealArray:
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
    *,
    N_STEPS: int,
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


def refine_pulse(
    duration_us: float,
    rf_max_hz: float,
    half_seed: RealArray,
    smoothness_weight: float,
    max_iter: int = 1200,
    *,
    N_STEPS: int,
    METHYL_OFFSET_LO_HZ: float,
    METHYL_OFFSET_HI_HZ: float,
    WATER_WINDOW_HZ: float,
    VALIDATION_WATER_POINTS: int,
    _control_problem: Callable[
        [RealArray, list[ComplexArray], list[ComplexArray], float, float], ControlProblem
    ],
    signed_amplitude: SignedAmplitude,
    _collapse_symmetric_gradient: Callable[[RealArray], RealArray],
    evaluate_pulse: EvaluatePulse,
) -> tuple[RealArray, PulseMetrics]:
    """Refine one (duration, peak-field) point with the smooth four-state objective."""
    duration_s = duration_us * 1e-6
    methyl_offsets = np.linspace(METHYL_OFFSET_LO_HZ, METHYL_OFFSET_HI_HZ, 81, dtype=np.float64)
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
    metrics, _ = evaluate_pulse(signed_amplitude(half), duration_s=duration_s, rf_max_hz=rf_max_hz)
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
    *,
    OPTIMIZED_HALF_AMPLITUDE: RealArray,
    refine_pulse: Callable[[float, float, RealArray], tuple[RealArray, PulseMetrics]],
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


def plot_time_evolution(
    time_evolution: dict[str, RealArray],
    output_dir: Path,
    *,
    PULSE_NAME: str,
    title: str,
) -> Path:
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
            axis.plot(time_us, quantise(trajectory[:, col, 0]), label="Mx", color="tab:blue")
            axis.plot(time_us, quantise(trajectory[:, col, 1]), label="My", color="tab:orange")
            axis.plot(time_us, quantise(trajectory[:, col, 2]), label="Mz", color="tab:green")
            axis.axhline(0.0, color="black", linewidth=0.5)
            axis.set_ylim(-1.05, 1.05)
            if row == 0:
                axis.set_title(f"{ppms[col]:g} ppm")
            if col == 0:
                axis.set_ylabel(f"from {name}")
            if row == n_rows - 1:
                axis.set_xlabel("Time (us)")
    axes[0, 0].legend(ncol=3, fontsize="x-small", loc="lower left")
    figure.suptitle(title)

    figure_path = output_dir / f"{PULSE_NAME}_time_evolution.png"
    save_figure(figure, figure_path, dpi=160)
    return figure_path


def plot_pareto(
    output_dir: Path,
    *,
    PULSE_NAME: str,
    FRONTIER: tuple[tuple[float, float], ...],
    RF_MAX_HZ: float,
    DURATION_US: float,
    label: str,
) -> Path:
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
        label=label,
    )
    axis.set_xlabel("Peak RF amplitude (kHz)")
    axis.set_ylabel("Shortest passing duration (ms)")
    axis.set_title("Max-power vs length trade-off (REBURP-style methyl 180)")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure_path = output_dir / f"{PULSE_NAME}_pareto.png"
    save_figure(figure, figure_path, dpi=160)
    return figure_path


def run(
    *,
    signed_amplitude: SignedAmplitude,
    DURATION_S: float,
    RF_MAX_HZ: float,
    TIME_EVOLUTION_PPM: tuple[float, ...],
    export_bruker_shape: Callable[[RealArray, Path], Path],
    plot_diagnostics: Callable[[RealArray, PulseMetrics, dict[str, RealArray], Path], Path],
    plot_pareto: Callable[[Path], Path],
    plot_wide_profile: Callable[[dict[str, RealArray], Path], Path],
    plot_wide_components: Callable[[dict[str, RealArray], Path], Path],
    plot_time_evolution: Callable[[dict[str, RealArray], Path], Path],
    evaluate_pulse: EvaluatePulse,
    wide_transfer_profile: WideProfile,
    wide_component_profile: WideProfile,
    time_evolution_profiles: TimeProfiles,
) -> RealArray:
    """Generate artifacts and return waveform plus dense summary metrics."""
    signed = signed_amplitude()
    metrics, profiles = evaluate_pulse(signed, duration_s=DURATION_S, rf_max_hz=RF_MAX_HZ)
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


def main(
    *,
    __doc__: str | None,
    search_pareto: Callable[[], list[tuple[float, float, RealArray, PulseMetrics]]],
    run: Callable[[], RealArray],
    N_STEPS: int,
    DURATION_S: float,
    RF_MAX_HZ: float,
    PULSE_NAME: str,
    evaluate_pulse: EvaluatePulse,
) -> None:
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


def signed_amplitude_10khz(half: RealArray, *, N_STEPS: int) -> RealArray:
    """Return the exactly symmetric signed RF-amplitude waveform."""
    half_array = np.asarray(half, dtype=np.float64)
    if half_array.shape != (N_STEPS // 2,):
        raise ValueError(f"half waveform must have shape ({N_STEPS // 2},)")
    if np.max(np.abs(half_array)) > 1.0 + 1e-12:
        raise ValueError("signed amplitude exceeds the 10 kHz RF limit")
    return np.asarray(np.concatenate((half_array, half_array[::-1])), dtype=np.float64)


def export_bruker_shape(
    signed: RealArray,
    output_dir: Path,
    *,
    PULSE_NAME: str,
    extra_tags: list[str],
    amplitude_phase: Callable[[RealArray], tuple[RealArray, RealArray]],
) -> Path:
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
        extra_tags=extra_tags,
    )
