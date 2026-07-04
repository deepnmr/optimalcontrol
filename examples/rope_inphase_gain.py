"""Reproduce the in-phase ROPE transfer gain curve from the JMR 2003 paper.

This script plots the equal-rate in-phase ROPE efficiency g_in(n) = g(n)^2
and compares it with the ROPE efficiency g(n) for n in [0, 3].

Saves figure to examples/output/rope_inphase_gain.png.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.rope import rope_g, rope_g_inphase


def run() -> npt.NDArray[np.float64]:
    """Compute and plot the in-phase ROPE gain curve.

    Returns the in-phase efficiency array as the primary numerical output.
    """
    n_values = np.linspace(0.0, 3.0, 300)

    inphase_eff = np.array([rope_g_inphase(float(n)) for n in n_values], dtype=np.float64)
    rope_eff = np.array([rope_g(float(n)) for n in n_values], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_values, inphase_eff, label="In-phase g_in(n) = g(n)²", linewidth=2)
    ax.plot(n_values, rope_eff, label="Antiphase g(n)", linewidth=2, linestyle="--")
    ax.set_xlabel("n = kI / J  (equal I and S rates)")
    ax.set_ylabel("Transfer efficiency")
    ax.set_title("In-phase ROPE efficiency (JMR 2003)")
    ax.legend()
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "rope_inphase_gain.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    return inphase_eff


if __name__ == "__main__":
    run()
    print("Saved examples/output/rope_inphase_gain.png")
