"""Generic REBURP band-selective 180 (Geen & Freeman) reference plot.

This is the textbook REBURP refocusing pulse, independent of any particular
spectrometer or sample. REBURP (Geen & Freeman, J. Magn. Reson. 93, 93-141,
1991) is a member of the BURP family: a band-selective, pure-phase
(universal-rotation) 180-degree pulse whose amplitude is a real cosine
Fourier series, so its phase is binary (0 where the amplitude is positive,
180 where it is negative) and its waveform is time symmetric.

Everything here is dimensionless. The horizontal axis of the offset profiles
is the resonance offset multiplied by the pulse duration, ``offset * T``, and
the RF strength is reported as the calibrated product ``B1_peak * T`` that
turns the on-resonance spin by 180 degrees. Scale to a real experiment by
fixing ``T`` and reading the RF amplitude and selective bandwidth off these
plots.

Run without arguments to write the reference plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from optimalcontrol.bloch import propagate_bloch_ensemble

RealArray = npt.NDArray[np.float64]

PULSE_NAME = "reburp_pulse"
N_STEPS = 512

# REBURP cosine Fourier coefficients A0..A15 (Geen & Freeman, 1991). The
# amplitude is f(s) = sum_n A_n cos(2*pi*n*s) for s in [0, 1].
REBURP_COEFFICIENTS: RealArray = np.array(
    [
        0.49,
        -1.02,
        1.11,
        -1.57,
        0.83,
        -0.42,
        0.26,
        -0.16,
        0.10,
        -0.07,
        0.04,
        -0.03,
        0.01,
        -0.02,
        0.00,
        0.01,
    ],
    dtype=np.float64,
)

PROFILE_HALF_WIDTH = 30.0
PROFILE_POINTS = 601
PASSBAND_INVERSION = -0.99
WIDE_HALF_WIDTH = 10.0
WIDE_POINTS = 801
TIME_EVOLUTION_OFFSETS: tuple[float, ...] = (0.0, 0.9, 1.8, 2.5, 4.0, 8.0)


def reburp_waveform(n_steps: int = N_STEPS) -> RealArray:
    """Return the unit-peak, time-symmetric REBURP signed amplitude."""
    fraction = (np.arange(n_steps, dtype=np.float64) + 0.5) / n_steps
    waveform = np.zeros(n_steps, dtype=np.float64)
    for order, coefficient in enumerate(REBURP_COEFFICIENTS):
        waveform += coefficient * np.cos(2.0 * np.pi * order * fraction)
    return np.asarray(waveform / np.max(np.abs(waveform)), dtype=np.float64)


def amplitude_phase(signed: RealArray) -> tuple[RealArray, RealArray]:
    """Convert signed amplitude to non-negative amplitude and binary phase."""
    amplitude = np.abs(signed)
    phase_deg = np.where(signed < 0.0, 180.0, 0.0)
    phase_deg[amplitude <= 1e-15] = 0.0
    return amplitude, np.asarray(phase_deg, dtype=np.float64)


def calibrate_rf_time_product(signed: RealArray) -> float:
    """Return ``B1_peak * T`` that inverts the on-resonance spin (Iz -> -Iz)."""
    waveform_xy = np.column_stack((signed, np.zeros_like(signed)))
    scales = np.array([1.0], dtype=np.float64)
    dt = 1.0 / signed.size  # unit duration T = 1

    def residual(rf: float) -> float:
        final = propagate_bloch_ensemble(
            np.array([0.0, 0.0, 1.0]),
            waveform_xy,
            np.array([0.0]),
            scales,
            float(rf),
            dt,
        )[0, 0, 2]
        return float((final + 1.0) ** 2)

    result = minimize_scalar(residual, bounds=(5.0, 7.0), method="bounded")
    return float(result.x)


def offset_profiles(signed: RealArray, rf_time_product: float) -> dict[str, RealArray]:
    """Return Mz inversion and universal-180 profiles over ``offset * T``."""
    waveform_xy = np.column_stack((signed, np.zeros_like(signed)))
    scales = np.array([1.0], dtype=np.float64)
    dt = 1.0 / signed.size
    offset_t = np.linspace(
        -PROFILE_HALF_WIDTH, PROFILE_HALF_WIDTH, PROFILE_POINTS, dtype=np.float64
    )
    fx = propagate_bloch_ensemble(
        np.array([1.0, 0.0, 0.0]), waveform_xy, offset_t, scales, rf_time_product, dt
    )[0]
    fy = propagate_bloch_ensemble(
        np.array([0.0, 1.0, 0.0]), waveform_xy, offset_t, scales, rf_time_product, dt
    )[0]
    fz = propagate_bloch_ensemble(
        np.array([0.0, 0.0, 1.0]), waveform_xy, offset_t, scales, rf_time_product, dt
    )[0]
    return {
        "offset_t": offset_t,
        "iz_to_mz": fz[:, 2],
        "x": fx[:, 0],
        "y": -fy[:, 1],
        "z": -fz[:, 2],
    }


def component_profiles(signed: RealArray, rf_time_product: float) -> dict[str, RealArray]:
    """Return final Mx, My, Mz for each initial Ix, Iy, Iz over ``offset * T``."""
    waveform_xy = np.column_stack((signed, np.zeros_like(signed)))
    scales = np.array([1.0], dtype=np.float64)
    dt = 1.0 / signed.size
    offset_t = np.linspace(
        -WIDE_HALF_WIDTH, WIDE_HALF_WIDTH, WIDE_POINTS, dtype=np.float64
    )
    initials = {
        "Ix": np.array([1.0, 0.0, 0.0]),
        "Iy": np.array([0.0, 1.0, 0.0]),
        "Iz": np.array([0.0, 0.0, 1.0]),
    }
    profile: dict[str, RealArray] = {"offset_t": offset_t}
    for name, initial in initials.items():
        profile[name] = propagate_bloch_ensemble(
            initial, waveform_xy, offset_t, scales, rf_time_product, dt
        )[0]
    return profile


def time_evolution_profiles(
    signed: RealArray,
    rf_time_product: float,
    offsets_t: tuple[float, ...] = TIME_EVOLUTION_OFFSETS,
) -> dict[str, RealArray]:
    """Return full Mx, My, Mz trajectories for each initial Ix, Iy, Iz state."""
    waveform_xy = np.column_stack((signed, np.zeros_like(signed)))
    n_steps = signed.size
    dt = 1.0 / n_steps
    offset_t = np.asarray(offsets_t, dtype=np.float64)
    scales = np.array([1.0], dtype=np.float64)
    fraction = np.arange(n_steps + 1, dtype=np.float64) / n_steps
    initials = {
        "Ix": np.array([1.0, 0.0, 0.0]),
        "Iy": np.array([0.0, 1.0, 0.0]),
        "Iz": np.array([0.0, 0.0, 1.0]),
    }
    trajectories: dict[str, RealArray] = {"fraction": fraction, "offset_t": offset_t}
    for name, initial in initials.items():
        trajectory = np.empty((n_steps + 1, offset_t.size, 3), dtype=np.float64)
        trajectory[0] = initial
        for step in range(1, n_steps + 1):
            trajectory[step] = propagate_bloch_ensemble(
                initial, waveform_xy[:step], offset_t, scales, rf_time_product, dt
            )[0]
        trajectories[name] = trajectory
    return trajectories


def plot_reburp(
    signed: RealArray,
    rf_time_product: float,
    profiles: dict[str, RealArray],
    output_dir: Path,
) -> Path:
    """Write the REBURP waveform and selectivity reference plots."""
    amplitude, phase_deg = amplitude_phase(signed)
    fraction = (np.arange(signed.size, dtype=np.float64) + 0.5) / signed.size
    offset_t = profiles["offset_t"]

    figure, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), constrained_layout=True)

    axes[0].step(fraction, 100.0 * amplitude, where="mid", color="tab:blue")
    phase_axis = axes[0].twinx()
    phase_axis.step(fraction, phase_deg, where="mid", color="tab:red", alpha=0.55)
    axes[0].set_ylabel("Amplitude (%)")
    phase_axis.set_ylabel("Phase (deg)")
    phase_axis.set_yticks([0.0, 180.0])
    axes[0].set_xlabel("Fraction of pulse")
    axes[0].set_title(
        f"Generic REBURP 180 (B1_peak * T = {rf_time_product:.3f} for on-resonance flip)"
    )

    axes[1].plot(offset_t, profiles["iz_to_mz"], color="tab:green")
    axes[1].axhline(-1.0, color="black", linestyle=":")
    axes[1].set_xlim(-PROFILE_HALF_WIDTH, PROFILE_HALF_WIDTH)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("Mz from Iz")
    axes[1].set_xlabel("offset * T")
    axes[1].set_title("Band-selective inversion profile")

    axes[2].plot(offset_t, profiles["x"], label="Ix -> Ix")
    axes[2].plot(offset_t, profiles["y"], label="Iy -> -Iy")
    axes[2].plot(offset_t, profiles["z"], label="Iz -> -Iz")
    axes[2].axhline(1.0, color="black", linestyle=":")
    axes[2].set_xlim(-8.0, 8.0)
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_ylabel("Universal-180 transfer")
    axes[2].set_xlabel("offset * T (passband zoom)")
    axes[2].legend()

    figure_path = output_dir / f"{PULSE_NAME}.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_wide_profile(profiles: dict[str, RealArray], output_dir: Path) -> Path:
    """Write Ix, -Iy, Iz transfer over the wide ``offset * T`` window."""
    offset_t = profiles["offset_t"]
    figure, axis = plt.subplots(1, 1, figsize=(9.0, 4.5), constrained_layout=True)
    axis.plot(offset_t, profiles["x"], label="Ix -> Ix")
    axis.plot(offset_t, profiles["y"], label="Iy -> -Iy")
    axis.plot(offset_t, profiles["z"], label="Iz -> -Iz")
    axis.axhline(1.0, color="black", linestyle=":")
    axis.set_xlim(-WIDE_HALF_WIDTH, WIDE_HALF_WIDTH)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel("offset * T")
    axis.set_ylabel("Transfer fidelity")
    axis.set_title("Wide-offset transfer profile")
    axis.legend()

    figure_path = output_dir / f"{PULSE_NAME}_wide_profile.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_wide_components(component_profile: dict[str, RealArray], output_dir: Path) -> Path:
    """Write final Mx, My, Mz vs ``offset * T`` for each initial Ix, Iy, Iz."""
    offset_t = component_profile["offset_t"]
    figure, axes = plt.subplots(
        3, 1, figsize=(9.0, 9.0), sharex=True, constrained_layout=True
    )
    for axis, name in zip(axes, ("Ix", "Iy", "Iz")):
        final = component_profile[name]
        axis.plot(offset_t, final[:, 0], label="Mx", color="tab:blue")
        axis.plot(offset_t, final[:, 1], label="My", color="tab:orange")
        axis.plot(offset_t, final[:, 2], label="Mz", color="tab:green")
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.set_xlim(-WIDE_HALF_WIDTH, WIDE_HALF_WIDTH)
        axis.set_ylim(-1.05, 1.05)
        axis.set_ylabel(f"from {name}")
    axes[0].legend(ncol=3, fontsize="small")
    axes[0].set_title("Final Bloch components vs offset * T")
    axes[-1].set_xlabel("offset * T")

    figure_path = output_dir / f"{PULSE_NAME}_wide_components.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_time_evolution(time_evolution: dict[str, RealArray], output_dir: Path) -> Path:
    """Write Mx, My, Mz time evolution per initial state and offset as a PNG."""
    fraction = time_evolution["fraction"]
    offsets_t = time_evolution["offset_t"]
    inits = ("Ix", "Iy", "Iz")
    n_rows = len(inits)
    n_cols = offsets_t.size

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
            axis.plot(fraction, trajectory[:, col, 0], label="Mx", color="tab:blue")
            axis.plot(fraction, trajectory[:, col, 1], label="My", color="tab:orange")
            axis.plot(fraction, trajectory[:, col, 2], label="Mz", color="tab:green")
            axis.axhline(0.0, color="black", linewidth=0.5)
            axis.set_ylim(-1.05, 1.05)
            if row == 0:
                axis.set_title(f"offset*T = {offsets_t[col]:g}")
            if col == 0:
                axis.set_ylabel(f"from {name}")
            if row == n_rows - 1:
                axis.set_xlabel("Fraction of pulse")
    axes[0, 0].legend(ncol=3, fontsize="x-small", loc="lower left")
    figure.suptitle("Bloch-component time evolution (generic REBURP 180)")

    figure_path = output_dir / f"{PULSE_NAME}_time_evolution.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def run() -> RealArray:
    """Generate the reference plot and return waveform plus summary scalars."""
    signed = reburp_waveform()
    rf_time_product = calibrate_rf_time_product(signed)
    profiles = offset_profiles(signed, rf_time_product)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_reburp(signed, rf_time_product, profiles, output_dir)
    plot_wide_profile(profiles, output_dir)
    plot_wide_components(component_profiles(signed, rf_time_product), output_dir)
    plot_time_evolution(
        time_evolution_profiles(signed, rf_time_product), output_dir
    )

    in_passband = profiles["iz_to_mz"] < PASSBAND_INVERSION
    inverted = profiles["offset_t"][in_passband]
    passband_lo = float(inverted.min()) if inverted.size else 0.0
    passband_hi = float(inverted.max()) if inverted.size else 0.0
    universal = (profiles["x"] + profiles["y"] + profiles["z"]) / 3.0
    universal_min = float(np.min(universal[in_passband])) if inverted.size else 0.0
    summary = np.array(
        [rf_time_product, passband_lo, passband_hi, universal_min], dtype=np.float64
    )
    return np.concatenate((signed, summary))


def main() -> None:
    result = run()
    rf_time_product, passband_lo, passband_hi, universal_min = result[N_STEPS:]
    print(f"B1_peak * T (on-resonance 180) = {rf_time_product:.4f}")
    print(f"inversion passband (offset * T) = [{passband_lo:.3f}, {passband_hi:.3f}]")
    print(f"worst universal-180 transfer in passband = {universal_min:.4f}")
    print(f"Saved examples/output/{PULSE_NAME}.png")


if __name__ == "__main__":
    main()
