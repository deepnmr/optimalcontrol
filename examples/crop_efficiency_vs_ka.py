"""Reproduce CROP efficiency eta vs ka/J for several kc/ka ratios.

This script reproduces the PNAS 2003 CROP paper efficiency curves, plotting
eta(ka/J, kc/ka) for kc/ka in [0, 0.5, 0.75, 0.99] over ka/J in [0, 2].

Saves figure to examples/output/crop_efficiency_vs_ka.png.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.crop import crop_eta


def run() -> npt.NDArray[np.float64]:
    """Compute and plot CROP efficiency curves vs ka/J for several kc/ka ratios.

    Returns the 2-D efficiency array (ka_ratios x kc_ratios) as the primary
    numerical output.
    """
    J_hz = 100.0
    ka_over_J = np.linspace(0.0, 2.0, 300)
    kc_over_ka_list = [0.0, 0.5, 0.75, 0.99]

    eta_array = np.zeros((len(ka_over_J), len(kc_over_ka_list)), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, ratio in enumerate(kc_over_ka_list):
        label = f"kc/ka = {ratio}"
        for row, ka_ratio in enumerate(ka_over_J):
            ka = float(ka_ratio) * J_hz
            kc = ratio * ka
            eta_array[row, col] = crop_eta(ka, kc, J_hz)
        ax.plot(ka_over_J, eta_array[:, col], label=label, linewidth=2)

    ax.set_xlabel("ka / J")
    ax.set_ylabel("Transfer efficiency eta")
    ax.set_title("CROP efficiency vs ka/J (PNAS 2003)")
    ax.legend()
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "crop_efficiency_vs_ka.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    return eta_array


if __name__ == "__main__":
    run()
    print("Saved examples/output/crop_efficiency_vs_ka.png")
