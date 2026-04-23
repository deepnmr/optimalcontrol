"""Show the CROP decoherence-free (kc/ka -> 1) limiting efficiency.

As kc/ka approaches 1, the CROP efficiency approaches 1 regardless of the
absolute relaxation rate ka/J. This script plots eta vs kc/ka for several
ka/J values to illustrate the decoherence-free limit from the PNAS 2003 paper.

Saves figure to examples/output/crop_lossless_limit.png.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.crop import crop_eta


def run() -> npt.NDArray[np.float64]:
    """Compute and plot CROP efficiency vs kc/ka approaching the lossless limit.

    Returns a 2-D array of eta values (ka_cases x kc_ratio grid).
    """
    J_hz = 100.0
    kc_over_ka = np.linspace(0.0, 0.999, 300)
    ka_over_J_list = [0.3, 0.6, 1.0, 1.5]

    eta_array = np.zeros((len(ka_over_J_list), len(kc_over_ka)), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4))
    for row, ka_ratio in enumerate(ka_over_J_list):
        ka = ka_ratio * J_hz
        label = f"ka/J = {ka_ratio}"
        for col, ratio in enumerate(kc_over_ka):
            kc = float(ratio) * ka
            eta_array[row, col] = crop_eta(ka, kc, J_hz)
        ax.plot(kc_over_ka, eta_array[row], label=label, linewidth=2)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.5, label="Lossless limit")
    ax.set_xlabel("kc / ka")
    ax.set_ylabel("Transfer efficiency eta")
    ax.set_title("CROP decoherence-free (kc/ka -> 1) limit (PNAS 2003)")
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "crop_lossless_limit.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    return eta_array


if __name__ == "__main__":
    run()
    print("Saved examples/output/crop_lossless_limit.png")
