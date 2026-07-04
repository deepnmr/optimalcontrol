"""Reproduce finite-time ROPE efficiency g_T vs transfer duration T for n = 1.

This script plots the optimal ROPE transfer efficiency as a function of the
constrained sequence duration T, including the INEPT branch below T_crit
and the finite-time ROPE branch above it.

Saves figure to examples/output/rope_finite_time_efficiency.png.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.rope import rope_finite_efficiency, rope_g, rope_Tcrit


def run() -> npt.NDArray[np.float64]:
    """Compute and plot finite-time ROPE efficiency vs duration T for n = 1.

    Returns the efficiency array as the primary numerical output.
    """
    n = 1.0
    J_hz = 100.0
    Tcrit = rope_Tcrit(n, J_hz)

    # T range: 0 to 5/J, with fine sampling near Tcrit
    T_max = 5.0 / J_hz
    T_values = np.linspace(1e-4 / J_hz, T_max, 500)

    efficiency = np.array(
        [rope_finite_efficiency(float(T), n, J_hz) for T in T_values], dtype=np.float64
    )

    g_inf = rope_g(n)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_values * J_hz, efficiency, linewidth=2, label=r"$g_T$ (finite-time ROPE)")
    ax.axhline(g_inf, linestyle="--", linewidth=1.5, color="C1", label=r"$g(\infty)$ = g(n)")
    ax.axvline(
        Tcrit * J_hz,
        linestyle=":",
        linewidth=1.5,
        color="C2",
        label=rf"$T_{{crit}}$ = {Tcrit * J_hz:.3f} / J",
    )
    ax.set_xlabel(r"$T \cdot J$")
    ax.set_ylabel("Transfer efficiency")
    ax.set_title(rf"Finite-time ROPE efficiency vs duration, $n = {n}$ (JMR 2003)")
    ax.legend()
    ax.set_xlim(0.0, T_max * J_hz)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "rope_finite_time_efficiency.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return efficiency


if __name__ == "__main__":
    run()
    print("Saved examples/output/rope_finite_time_efficiency.png")
