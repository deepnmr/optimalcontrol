"""ROPE example using sodium formate 13C-1H parameters: J = 193 Hz, T2 = 1.4 ms.

This script computes the ROPE and INEPT efficiencies for the sodium formate
system described in the JMR 2003 paper, plots the finite-time efficiency vs T,
and prints the key physical parameters including T_crit and the unconstrained
ROPE efficiency.

The relaxation rate is derived from T2 via k = 1 / (pi * T2) so that the INEPT
efficiency decays as exp(-t / T2).

Saves figure to examples/output/rope_sodium_formate.png.
"""

import matplotlib

matplotlib.use("Agg")

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.rope import (
    inept_max_efficiency,
    rope_finite_efficiency,
    rope_g,
    rope_n,
    rope_Tcrit,
)


def run() -> npt.NDArray[np.float64]:
    """Compute finite-time ROPE efficiency for sodium formate parameters.

    Returns the efficiency array as the primary numerical output.
    """
    J_hz = 193.0
    T2 = 1.4e-3
    # k_hz defined so that INEPT efficiency decays as exp(-pi*k_hz*t) = exp(-t/T2)
    k_hz = 1.0 / (math.pi * T2)
    n = rope_n(k_hz, J_hz)
    g_inf = rope_g(n)
    Tcrit = rope_Tcrit(n, J_hz)
    inept_best = inept_max_efficiency(n, J_hz)

    print(f"Sodium formate: J = {J_hz} Hz, T2 = {T2 * 1e3:.1f} ms")
    print(f"  k_hz = {k_hz:.1f} Hz, n = k/J = {n:.4f}")
    print(f"  Unconstrained ROPE efficiency g(n) = {g_inf:.4f}")
    print(f"  INEPT max efficiency = {inept_best:.4f}")
    print(f"  ROPE gain over INEPT = {g_inf / inept_best:.4f}")
    print(f"  T_crit = {Tcrit * 1e3:.3f} ms = {Tcrit * J_hz:.4f} / J")

    T_max = 10.0 / J_hz
    T_values = np.linspace(1e-5 / J_hz, T_max, 600)

    efficiency = np.array(
        [rope_finite_efficiency(float(T), n, J_hz) for T in T_values], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(T_values * 1e3, efficiency, linewidth=2, label=r"Finite-time ROPE $g_T$")
    ax.axhline(
        g_inf, linestyle="--", linewidth=1.5, color="C1", label=rf"$g(\infty)$ = {g_inf:.3f}"
    )
    ax.axhline(
        inept_best,
        linestyle="-.",
        linewidth=1.5,
        color="C3",
        label=rf"INEPT max = {inept_best:.3f}",
    )
    ax.axvline(
        Tcrit * 1e3,
        linestyle=":",
        linewidth=1.5,
        color="C2",
        label=rf"$T_{{crit}}$ = {Tcrit * 1e3:.2f} ms",
    )
    ax.set_xlabel("Duration T (ms)")
    ax.set_ylabel("Transfer efficiency")
    ax.set_title(rf"ROPE efficiency: sodium formate ($J = {J_hz}$ Hz, $T_2 = {T2 * 1e3:.1f}$ ms)")
    ax.legend(fontsize=9)
    ax.set_xlim(0.0, T_max * 1e3)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "rope_sodium_formate.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return efficiency


if __name__ == "__main__":
    run()
    print("Saved examples/output/rope_sodium_formate.png")
