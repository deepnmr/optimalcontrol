"""Convert finite-time ROPE waveform to hard-pulse approximation and print flip angles.

This script demonstrates the hard-pulse approximation of the finite-time ROPE
sequence for n = 1, T = 0.263 / J. The shaped phase-I and phase-III waveforms
are approximated by hard pulses whose flip angles are derived from the ROPE
finite-time phase angles h1 and h2 and the initial control value u1(0).

The JMR 2003 paper describes this approximation as a practical recipe for
implementing ROPE on spectrometers that support only rectangular pulses.
"""

import matplotlib

matplotlib.use("Agg")

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.rope import (
    rope_finite_angles,
    rope_hard_pulse_angle,
    rope_phase1_control,
    rope_switching_time,
    rope_waveform,
)


def run() -> npt.NDArray[np.float64]:
    """Compute ROPE hard-pulse flip angles and compare with the shaped waveform.

    Returns an array [alpha_h1_deg, alpha_h2_deg, alpha_u1_deg] containing the
    three hard-pulse flip angles as the primary numerical output.
    """
    n = 1.0
    J_hz = 100.0
    T = 0.263 / J_hz
    dt = T / 400.0

    switching_time = rope_switching_time(T, n, J_hz)
    h1, h2 = rope_finite_angles(T, n, J_hz)

    scaled_switch = math.pi * J_hz * switching_time
    u1_at_0, _ = rope_phase1_control(0.0, scaled_switch, h1, h2, n)

    # Hard-pulse angles from the finite-time ROPE formulation
    alpha_h1_deg = math.degrees(h1)
    alpha_h2_deg = math.degrees(h2)
    alpha_u1_deg = rope_hard_pulse_angle(u1_at_0, 1.0)

    print("Finite-time ROPE hard-pulse approximation")
    print(f"  Parameters: n = {n}, J = {J_hz} Hz, T = {T * 1e3:.3f} ms")
    print(f"  Switching time s = {switching_time * 1e3:.4f} ms")
    print(f"  Phase angle h1 = {math.degrees(h1):.2f} deg")
    print(f"  Phase angle h2 = {math.degrees(h2):.2f} deg")
    print(f"  u1(t=0) = {u1_at_0:.4f}")
    print(f"  Hard-pulse flip angle from u1(0): alpha = acos(u1(0)) = {alpha_u1_deg:.2f} deg")
    print()
    print("Hard-pulse ROPE sequence:")
    print(f"  1. Hard pulse  alpha = {alpha_u1_deg:.2f} deg  (from phase-I boundary)")
    print(f"  2. Free evolution delay  T - 2*s = {(T - 2 * switching_time) * 1e3:.4f} ms")
    print(f"  3. Hard pulse  alpha = {alpha_u1_deg:.2f} deg  (symmetry, phase-III boundary)")

    # Plot shaped waveform alongside the hard-pulse approximation
    wfm = rope_waveform(T, n, J_hz, dt)
    times = wfm["times"]
    u1 = wfm["u1"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times * J_hz, u1, linewidth=2, label=r"Shaped $u_1(t)$")
    ax.axhline(
        u1_at_0,
        linestyle="--",
        linewidth=1.5,
        color="C1",
        label=rf"Hard-pulse $u_1(0)$ = {u1_at_0:.3f}",
    )
    ax.axvline(
        switching_time * J_hz,
        linestyle=":",
        linewidth=1.5,
        color="C2",
        label=rf"Switching time $s$ = {switching_time * J_hz:.3f}$/J$",
    )
    ax.axvline(
        (T - switching_time) * J_hz,
        linestyle=":",
        linewidth=1.5,
        color="C2",
    )
    ax.set_xlabel(r"$t \cdot J$")
    ax.set_ylabel(r"$u_1$")
    ax.set_title(
        rf"ROPE hard-pulse approximation: $n = {n}$, $\alpha = {alpha_u1_deg:.1f}°$"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "rope_hard_pulse.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return np.array([alpha_h1_deg, alpha_h2_deg, alpha_u1_deg], dtype=np.float64)


if __name__ == "__main__":
    run()
    print("Saved examples/output/rope_hard_pulse.png")
