"""Reproduce ROPE vs INEPT efficiency curves as a function of n = kI/J.

This script reproduces the efficiency comparison from the JMR 2003 ROPE paper,
plotting the unconstrained ROPE efficiency g(n) and the INEPT maximum efficiency
over n in [0, 3].

Saves figure to examples/output/rope_efficiency_vs_n.png.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.rope import inept_max_efficiency, rope_g


def run() -> npt.NDArray[np.float64]:
    """Compute and plot ROPE vs INEPT efficiency curves.

    Returns the ROPE efficiency array as the primary numerical output.
    """
    n_values = np.linspace(0.0, 3.0, 300)
    J_hz = 100.0

    rope_eff = np.array([rope_g(float(n)) for n in n_values], dtype=np.float64)
    inept_eff = np.array(
        [inept_max_efficiency(float(n), J_hz) for n in n_values], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_values, rope_eff, label="ROPE g(n)", linewidth=2)
    ax.plot(n_values, inept_eff, label="INEPT max efficiency", linewidth=2, linestyle="--")
    ax.set_xlabel("n = kI / J")
    ax.set_ylabel("Transfer efficiency")
    ax.set_title("ROPE vs INEPT efficiency (JMR 2003)")
    ax.legend()
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "rope_efficiency_vs_n.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    return rope_eff


if __name__ == "__main__":
    run()
    print("Saved examples/output/rope_efficiency_vs_n.png")
