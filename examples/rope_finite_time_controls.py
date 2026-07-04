"""Reproduce finite-time ROPE three-phase control waveform for n = 1.

This script plots the dimensionless controls u1 and u2 and the RF amplitude
for the three-phase ROPE pulse at T = 0.263 / J, n = 1, as described in the
JMR 2003 paper.

Saves figure to examples/output/rope_finite_time_controls.png.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.rope import rope_switching_time, rope_waveform


def run() -> npt.NDArray[np.float64]:
    """Sample and plot the finite-time ROPE three-phase waveform for n = 1.

    Returns the u1 control array as the primary numerical output.
    """
    n = 1.0
    J_hz = 100.0
    T = 0.263 / J_hz
    dt = T / 400.0

    wfm = rope_waveform(T, n, J_hz, dt)
    times = wfm["times"]
    u1 = wfm["u1"]
    u2 = wfm["u2"]
    amplitude = wfm["amplitude"]

    switching_time = rope_switching_time(T, n, J_hz)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_ctrl = axes[0]
    ax_ctrl.plot(times * J_hz, u1, label=r"$u_1$", linewidth=2)
    ax_ctrl.plot(times * J_hz, u2, label=r"$u_2$", linewidth=2, linestyle="--")
    ax_ctrl.axvline(
        switching_time * J_hz, linestyle=":", linewidth=1.5, color="C2", label="Phase boundaries"
    )
    ax_ctrl.axvline((T - switching_time) * J_hz, linestyle=":", linewidth=1.5, color="C2")
    ax_ctrl.set_ylabel("Dimensionless control")
    ax_ctrl.set_title(
        rf"Finite-time ROPE controls: $n = {n}$, $T = 0.263/J$ (JMR 2003)"
    )
    ax_ctrl.legend()
    ax_ctrl.set_ylim(-0.1, 1.2)
    ax_ctrl.grid(True, alpha=0.3)

    # Clip very large RF amplitudes at transitions for display
    amp_display = amplitude.copy()
    amp_display[amp_display > 10.0 * J_hz * 2.0 * 3.14159] = float("nan")

    ax_amp = axes[1]
    ax_amp.plot(times * J_hz, amp_display / (2.0 * 3.14159 * J_hz), linewidth=2, color="C3")
    ax_amp.set_xlabel(r"$t \cdot J$")
    ax_amp.set_ylabel(r"RF amplitude / $2\pi J$")
    ax_amp.set_ylim(bottom=0.0)
    ax_amp.grid(True, alpha=0.3)

    fig.tight_layout()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "rope_finite_time_controls.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return u1


if __name__ == "__main__":
    run()
    print("Saved examples/output/rope_finite_time_controls.png")
