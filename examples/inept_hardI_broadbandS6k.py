"""INEPT-like Iz -> 2IzSz sequence with hard I pulses and a cached S broadband 180.

The sequence keeps the I-channel pulses hard at 10 kHz:

- initial 90 deg pulse on I
- central hard 180 deg pulse on I
- final 90 deg pulse on I

The middle S-channel refocusing pulse is a cached 6 kHz amplitude-limited
20-step broadband 180 pulse obtained from standalone single-spin optimisation
for Sz inversion. The script assembles the full two-spin INEPT-like sequence for
J = 94 Hz, evaluates the Iz -> 2IzSz transfer versus S-spin offset, writes a
Bruker shape for the S pulse, and saves a band-profile figure.
"""

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.grape import ControlProblem, grape_xy
from optimalcontrol.io import export_bruker_shape as write_bruker_shape
from optimalcontrol.operators import Ix, Iy, Iz, place_operator
from optimalcontrol.states import normalise_hs, state_from_label

PULSE_NAME = "inept_hardI_broadbandS6k"
J_HZ = 94.0
TAU_S = 1.0 / (4.0 * J_HZ)
DT = 12.5e-6
I_RF_HZ = 10_000.0
S_RF_HZ = 6_000.0
PROFILE_THRESHOLD = 0.90
I90_STEPS = 2
I180_STEPS = 4

# Cached 20-step standalone S-spin broadband 180 pulse at 6 kHz.
# Columns are [u_x, u_y] with instantaneous amplitude constrained to <= 1.
S_PULSE_XY: npt.NDArray[np.float64] = np.array(
    [
        [0.715629244825, 0.698480338987],
        [-0.058567001686, 0.971447439723],
        [-0.593380773258, 0.804921895545],
        [-0.739701278307, 0.661817769172],
        [-0.920719995277, 0.216405286155],
        [0.320973506627, -0.922603919851],
        [0.89563472054, -0.444790340907],
        [0.896070500122, -0.443911769174],
        [0.937899175887, -0.346907964552],
        [0.969625121906, -0.244595835961],
        [0.972045312998, -0.234793333547],
        [0.953090150758, -0.302686578044],
        [0.92710263316, -0.374807560739],
        [0.603559956436, -0.797317614873],
        [-0.548051150694, -0.33867488103],
        [-0.812685372394, 0.521881191019],
        [-0.667512785344, 0.743288191163],
        [-0.377011534605, 0.92046058121],
        [0.180554154675, 0.983565044738],
        [0.586919454566, 0.23749761032],
    ],
    dtype=np.float64,
)


def _sequence_layout() -> tuple[int, int, int]:
    """Return the I 180 start, final I 90 start, and total step count."""
    i90_center = I90_STEPS / 2.0
    i180_start = round((TAU_S / DT) - I180_STEPS / 2.0 - i90_center)
    final90_start = round(i180_start + I180_STEPS / 2.0 + (TAU_S / DT) - I90_STEPS / 2.0)
    n_total = final90_start + I90_STEPS
    return i180_start, final90_start, n_total


def _full_waveform() -> npt.NDArray[np.float64]:
    """Return the complete 4-channel INEPT-like waveform."""
    i180_start, final90_start, n_total = _sequence_layout()
    waveform = np.zeros((n_total, 4), dtype=np.float64)
    waveform[0:I90_STEPS, 0] = 1.0
    waveform[i180_start:i180_start + I180_STEPS, 0] = -1.0
    waveform[final90_start:final90_start + I90_STEPS, 1] = 1.0

    s_center = i180_start + I180_STEPS / 2.0
    s_start = int(round(s_center - S_PULSE_XY.shape[0] / 2.0))
    waveform[s_start:s_start + S_PULSE_XY.shape[0], 2:4] = S_PULSE_XY
    return waveform


def _single_offset_problem(offset_hz: float) -> ControlProblem:
    """Return the two-spin transfer problem for one S-spin offset."""
    iz_i = place_operator(Iz(), 0, 2)
    iz_s = place_operator(Iz(), 1, 2)
    drift = np.complex128(-1j) * (
        (2.0 * math.pi * J_HZ) * (iz_i @ iz_s) + (2.0 * math.pi * offset_hz) * iz_s
    )
    operators = [
        np.complex128(-1j) * (2.0 * math.pi * I_RF_HZ) * place_operator(Ix(), 0, 2),
        np.complex128(-1j) * (2.0 * math.pi * I_RF_HZ) * place_operator(Iy(), 0, 2),
        np.complex128(-1j) * (2.0 * math.pi * S_RF_HZ) * place_operator(Ix(), 1, 2),
        np.complex128(-1j) * (2.0 * math.pi * S_RF_HZ) * place_operator(Iy(), 1, 2),
    ]
    return ControlProblem(
        drifts=[drift],
        operators=operators,
        rho_init=[normalise_hs(state_from_label("Iz", 2))],
        rho_targ=[normalise_hs(state_from_label("2IzSz", 2))],
        pulse_dt=DT,
        pwr_levels=[1.0, 1.0, 1.0, 1.0],
        freeze=None,
        fidelity_mode="real",
        basis="dense",
    )


def _profile(offsets_hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return Iz -> 2IzSz transfer values versus S offset."""
    waveform = _full_waveform()
    return np.asarray(
        [float(grape_xy(_single_offset_problem(float(offset)), waveform)) for offset in offsets_hz],
        dtype=np.float64,
    )


def _contiguous_bandwidth(
    offsets_hz: npt.NDArray[np.float64],
    transfer: npt.NDArray[np.float64],
    threshold: float,
) -> float:
    """Return center-containing contiguous bandwidth above threshold."""
    ok = transfer >= threshold
    centre = transfer.size // 2
    if not ok[centre]:
        return 0.0
    left = centre
    right = centre
    while left - 1 >= 0 and ok[left - 1]:
        left -= 1
    while right + 1 < ok.size and ok[right + 1]:
        right += 1
    return float(offsets_hz[right] - offsets_hz[left])


def _export_s_bruker_shape(output_dir: Path) -> Path:
    """Write the cached S-spin pulse as a Bruker amplitude/phase shape."""
    amplitude = np.asarray(np.linalg.norm(S_PULSE_XY, axis=1), dtype=np.float64)
    phase_deg = np.asarray(
        np.degrees(np.arctan2(S_PULSE_XY[:, 1], S_PULSE_XY[:, 0])), dtype=np.float64
    )
    phase_deg = np.mod(phase_deg, 360.0)
    return write_bruker_shape(
        output_dir / f"{PULSE_NAME}.shape",
        PULSE_NAME,
        100.0 * amplitude,
        phase_deg,
        integfac=float(np.mean(amplitude)),
        shape_mode=0,
        extra_tags=[
            f"##$OPTIMALCONTROL_RF_HZ= {S_RF_HZ:.12e}",
            f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {S_PULSE_XY.shape[0] * DT:.12e}",
            f"##$OPTIMALCONTROL_STEP_DURATION_S= {DT:.12e}",
            "##$OPTIMALCONTROL_NOTE= Set amplitude 100 to 6 kHz on the S channel.",
        ],
    )


def run() -> npt.NDArray[np.float64]:
    """Write the S-shape file and the INEPT-like band profile figure."""
    output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_path = _export_s_bruker_shape(output_dir)

    offsets = np.linspace(-15_000.0, 15_000.0, 601, dtype=np.float64)
    transfer = _profile(offsets)
    bandwidth = _contiguous_bandwidth(offsets, transfer, PROFILE_THRESHOLD)
    i180_start, final90_start, n_total = _sequence_layout()

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(offsets / 1000.0, transfer, color="black", linewidth=1.7)
    ax.axhline(PROFILE_THRESHOLD, color="tab:red", linestyle="--", linewidth=1.0)
    ax.axvline(-10.0, color="tab:blue", linestyle=":", linewidth=1.0)
    ax.axvline(10.0, color="tab:blue", linestyle=":", linewidth=1.0)
    ax.set_xlim(-15.0, 15.0)
    ax.set_ylim(-0.25, 1.05)
    ax.set_xlabel("S Offset (kHz)")
    ax.set_ylabel("Transfer to 2IzSz")
    ax.set_title("INEPT-like sequence with hard I and optimized broadband S 180")
    ax.text(
        0.02,
        0.04,
        (
            f"J = {J_HZ:.1f} Hz, total = {n_total * DT * 1e3:.3f} ms\n"
            "I pulses: 10 kHz hard 90/180/90; "
            f"S pulse: {S_PULSE_XY.shape[0] * DT * 1e6:.1f} us at 6 kHz max\n"
            f"center = {transfer[transfer.size // 2]:.3f}, "
            f"-10 kHz = {transfer[np.argmin(np.abs(offsets + 10_000.0))]:.3f}, "
            f"+10 kHz = {transfer[np.argmin(np.abs(offsets - 10_000.0))]:.3f}, "
            f"bw@0.90 = {bandwidth / 1000.0:.2f} kHz"
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    fig_path = output_dir / f"{PULSE_NAME}_band_profile.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Bruker shape {shape_path}")
    print(f"Saved figure {fig_path}")
    print(
        "Transfer profile: "
        f"center = {transfer[transfer.size // 2]:.4f}, "
        f"-10 kHz = {transfer[np.argmin(np.abs(offsets + 10_000.0))]:.4f}, "
        f"+10 kHz = {transfer[np.argmin(np.abs(offsets - 10_000.0))]:.4f}, "
        f"bw@0.90 = {bandwidth / 1000.0:.3f} kHz"
    )
    return np.array(
        [
            float(n_total * DT * 1e3),
            float(S_PULSE_XY.shape[0] * DT * 1e6),
            float(transfer[transfer.size // 2]),
            float(transfer[np.argmin(np.abs(offsets + 10_000.0))]),
            float(transfer[np.argmin(np.abs(offsets - 10_000.0))]),
            float(np.min(transfer)),
            float(bandwidth),
            float(i180_start),
            float(final90_start),
        ],
        dtype=np.float64,
    )


if __name__ == "__main__":
    run()
