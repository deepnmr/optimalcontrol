"""Reproduce the ROPE gain curve for the Ia -> 2IbSc transfer.

This script plots the ROPE gain over INEPT (g(n) / INEPT_max(n)) as a function
of n = kI/J for n in [0, 3], reproducing the gain figure from the JMR 2003 paper.

Saves figure to examples/output/rope_gain_Ia_2IbSc.png.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.rope import rope_gain_over_inept


def run() -> npt.NDArray[np.float64]:
    """Compute and plot the ROPE gain over INEPT for the Ia -> 2IbSc transfer.

    Returns the gain array as the primary numerical output.
    """
    J_hz = 100.0
    n_values = np.linspace(0.0, 3.0, 300)

    gain = np.array(
        [rope_gain_over_inept(float(n), J_hz) for n in n_values], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_values, gain, linewidth=2)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="No gain (= 1)")
    ax.set_xlabel("n = kI / J")
    ax.set_ylabel("ROPE gain over INEPT")
    ax.set_title("ROPE gain for Ia → 2IbSc transfer (JMR 2003)")
    ax.legend()
    ax.set_xlim(0.0, 3.0)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "rope_gain_Ia_2IbSc.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    return gain


if __name__ == "__main__":
    run()
    print("Saved examples/output/rope_gain_Ia_2IbSc.png")
