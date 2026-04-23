"""Print CROP regression eta values for paper parameter sets.

Computes and prints eta for:
- ka/J = 0.6, kc/ka = 0.75 => expected ~0.7015664580244381
- ka/J = 1.1, kc/ka = 0.75 => expected ~0.5854918065322852

These values match the stored regression constants in tests/test_crop.py and
reproduce Table 1 / Figure 3 reference points from the PNAS 2003 CROP paper.

Saves figure to examples/output/crop_regression_params.png.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.crop import crop_eta

_CASES: list[tuple[float, float, float]] = [
    (0.6, 0.75, 0.7015664580244381),
    (1.1, 0.75, 0.5854918065322852),
]


def run() -> npt.NDArray[np.float64]:
    """Compute and print CROP regression eta values.

    Returns the computed eta values as a 1-D float64 array.
    """
    J_hz = 100.0
    results: list[float] = []

    for ka_over_J, kc_over_ka, expected in _CASES:
        ka = ka_over_J * J_hz
        kc = kc_over_ka * ka
        eta = crop_eta(ka, kc, J_hz)
        results.append(eta)
        print(
            f"ka/J={ka_over_J:.2f}, kc/ka={kc_over_ka:.2f}: "
            f"eta = {eta:.16f}  (expected {expected:.16f})"
        )

    eta_values = np.array(results, dtype=np.float64)

    ka_over_J_grid = np.linspace(0.0, 1.5, 200)
    J_hz_plot = 100.0
    kc_over_ka = 0.75

    fig, ax = plt.subplots(figsize=(7, 4))
    eta_curve = np.array(
        [crop_eta(k * J_hz_plot, kc_over_ka * k * J_hz_plot, J_hz_plot)
         for k in ka_over_J_grid],
        dtype=np.float64,
    )
    ax.plot(ka_over_J_grid, eta_curve, linewidth=2, label="eta (kc/ka=0.75)")

    for ka_over_J, _kc_over_ka, expected in _CASES:
        ax.scatter([ka_over_J], [expected], zorder=5, s=80, label=f"ka/J={ka_over_J}")

    ax.set_xlabel("ka / J")
    ax.set_ylabel("Transfer efficiency eta")
    ax.set_title("CROP regression reference points (PNAS 2003)")
    ax.legend()
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "crop_regression_params.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    return eta_values


if __name__ == "__main__":
    run()
    print("Saved examples/output/crop_regression_params.png")
