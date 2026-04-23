"""Plot truncated CROP amplitude and irradiation frequency waveforms.

This script reproduces PNAS 2003 CROP pulse shapes for two parameter sets:
- ka/J = 0.6, kc/ka = 0.75 (moderate cross-correlation)
- ka/J = 1.1, kc/ka = 0.75 (stronger relaxation)

Saves figure to examples/output/crop_truncated_waveform.png.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.crop import crop_pulse_params, crop_waveform


def run() -> npt.NDArray[np.float64]:
    """Compute and plot truncated CROP waveforms for two parameter sets.

    Returns a stacked array of amplitude waveform samples for the two cases.
    """
    J_hz = 100.0
    kc_over_ka = 0.75
    cases = [
        ("ka/J=0.6", 0.6 * J_hz),
        ("ka/J=1.1", 1.1 * J_hz),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("Truncated CROP waveforms (PNAS 2003, kc/ka=0.75)")

    amplitude_arrays: list[npt.NDArray[np.float64]] = []
    for col, (label, ka) in enumerate(cases):
        kc = kc_over_ka * ka
        params = crop_pulse_params(ka, kc, J_hz)
        dt = params.truncation_window / 200.0
        wfm = crop_waveform(ka, kc, J_hz, dt, params.truncation_window)

        times = wfm["times"]
        amplitude = wfm["amplitude"]
        irrad_freq = wfm["irrad_freq"]
        amplitude_arrays.append(amplitude)

        axes[0, col].plot(times * 1e3, amplitude, linewidth=2)
        axes[0, col].set_title(f"Amplitude ({label})")
        axes[0, col].set_xlabel("Time (ms)")
        axes[0, col].set_ylabel("Amplitude (Hz)")
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(times * 1e3, irrad_freq, linewidth=2, color="tab:orange")
        axes[1, col].set_title(f"Irradiation frequency ({label})")
        axes[1, col].set_xlabel("Time (ms)")
        axes[1, col].set_ylabel("Irrad. freq. (Hz)")
        axes[1, col].grid(True, alpha=0.3)

    fig.tight_layout()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "crop_truncated_waveform.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    return np.concatenate(amplitude_arrays)


if __name__ == "__main__":
    run()
    print("Saved examples/output/crop_truncated_waveform.png")
