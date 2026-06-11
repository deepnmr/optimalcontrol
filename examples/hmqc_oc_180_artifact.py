"""Optimal-control 180-degree refocusing pulse that suppresses HMQC artifacts.

Reproduces the message of Kay, J. Biomol. NMR 73, 423-427 (2019),
"Artifacts can emerge in spectra recorded with even the simplest of pulse
schemes: an HMQC case study". The single central 1H 180-degree refocusing
pulse of the methyl-TROSY HMQC is the culprit behind small methyl multiplet
artifacts: when its flip angle deviates from 180 degrees the starting
|alpha><alpha| population element spreads into detectable sideband terms that
neither phase cycling nor gradients can remove (Eqs. 5-6 of the paper).

Writing S = sin(theta/2) and C = cos(theta/2), the five multiplet components
of the affected cross-peak have relative intensities

    outer  (+- J_CH)    : 3 S^4 C^4
    inner  (+- J_CH/2)   : 8 S^4 C^2 - 4 S^2 C^4
    centre (0)           : 4 S^6 - 8 S^4 C^2 + 6 S^2 C^4

A perfect refocusing pulse has theta = 180 degrees (S = 1, C = 0), so every
sideband vanishes and only the central line survives; at theta = 170 degrees
the inner satellites already reach ~1.5 % of the central line. The paper
removes the artifacts with a *composite* 180 pulse. Here we instead design a
broadband 180 (inversion) pulse with GRAPE optimal control: it drives the 1H
populations through a full inversion across a wide offset band and a +-10 % B1
window, so the residual non-inverted amplitude C = |<alpha|U|alpha>| -- the
quantity that seeds the artifacts -- is pushed to ~0 everywhere.

The script compares three refocusing pulses over a 1H offset profile:

  * hard      : a single rectangular 180_x pulse,
  * composite : the Levitt-Freeman 90_x 180_y 90_x composite pulse,
  * oc        : the GRAPE optimal-control broadband 180 pulse.

For each it extracts the effective C(offset) from the actual single-spin
propagator and evaluates the inner-sideband artifact intensity relative to the
central line, the quantity the paper measures.

The default path uses the cached optimal-control waveform. Pass --optimize to
regenerate it with L-BFGS GRAPE.

run() returns the stacked offset profile used by the regression snapshot.

Saves:
  examples/output/hmqc_oc_180_artifact.shape
  examples/output/hmqc_oc_180_artifact.png
"""

import matplotlib

matplotlib.use("Agg")

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from optimalcontrol.grape import ControlProblem
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, vec
from optimalcontrol.optimizers import lbfgs_grape
from optimalcontrol.penalties import PenaltySpec
from optimalcontrol.states import normalise_hs

RealArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

PULSE_NAME = "hmqc_oc_180_artifact"
N_STEPS = 60
RF_MAX_HZ = 10_000.0
DURATION_S = 3.0e-3
DT = DURATION_S / N_STEPS
BANDWIDTH_HZ = 5_000.0
B1_DEVIATION_FRACTION = 0.10
SNSA_WEIGHT = 0.02
TWO_PI = 2.0 * math.pi
N_OFFSETS = 121
N_OPT_OFFSETS = 25

# Cached GRAPE solution for a broadband 180 (Iz -> -Iz) inversion pulse over
# offsets +-5 kHz and B1 scaling [0.9, 1.0, 1.1] at RF_MAX_HZ. Shape (60, 2):
# columns are [u_x, u_y] as fractions of RF_MAX_HZ. Regenerate with --optimize.
OPTIMIZED_WFM_XY: RealArray = np.array(
    [
        [8.265613815410e-01, -3.500890818823e-01],
        [8.459073855628e-01, -1.702836612349e-01],
        [9.524718623397e-01, -6.953072434278e-02],
        [1.648184028920e-01, -8.177174826754e-02],
        [4.322203896238e-01, -4.079331767208e-02],
        [9.722576609683e-01, 1.936265365689e-02],
        [8.519827403134e-01, -4.744684983900e-02],
        [-5.509130460776e-01, -2.506451777073e-01],
        [3.018294569468e-01, -9.868101721365e-02],
        [8.428689967697e-01, 2.417132200260e-01],
        [6.199886674208e-01, -4.538353026846e-01],
        [-6.749690034871e-01, -4.643382893927e-01],
        [-3.940972487802e-01, 4.399887824816e-01],
        [4.046828547171e-01, -4.236705123115e-01],
        [4.940964625442e-01, 2.379621620646e-01],
        [4.873237436131e-01, 2.208261139582e-01],
        [3.966326165233e-01, -4.357031967185e-01],
        [1.117971200732e-01, -1.780789845621e-01],
        [-3.678023135918e-01, 3.559310199159e-01],
        [1.578346145098e-02, 3.013321595205e-01],
        [2.965076473515e-01, -6.425600697704e-02],
        [4.984189435271e-01, 7.908622960828e-01],
        [-5.154031133820e-01, 7.712730593619e-01],
        [-7.659361201802e-02, -2.705532488423e-01],
        [-1.998090394921e-01, 3.022830565590e-01],
        [-9.844213949065e-01, -3.838403150767e-01],
        [-3.489387434811e-01, -9.454881863480e-01],
        [-2.402437353802e-01, -8.928102746198e-01],
        [2.308898525854e-01, -1.810713807804e-01],
        [-6.003308302608e-02, 1.777160970999e-01],
        [-2.675872710700e-01, -4.399241167687e-01],
        [5.149997625609e-01, -4.855641230295e-01],
        [-8.933189294392e-02, -3.862645126561e-01],
        [-3.150276335327e-01, 5.774153190826e-02],
        [-1.342445054495e-01, 2.659295274851e-01],
        [8.038404902209e-02, -3.355795650766e-02],
        [4.708416272034e-01, 2.176966917831e-01],
        [1.415140781998e-01, 7.983716274014e-01],
        [-7.376168816605e-02, 4.922556029074e-01],
        [1.364610648217e-01, 1.030925140053e+00],
        [-6.131036608711e-03, 4.121763664037e-01],
        [1.970103256959e-03, -6.546792348049e-01],
        [7.027570290673e-01, 3.391840917527e-01],
        [1.383041798607e-01, -6.080503262233e-01],
        [-3.328948567705e-01, -3.481964527737e-01],
        [-3.160813614135e-01, -2.115805039281e-01],
        [4.703417345778e-01, -4.783345164271e-01],
        [5.007224757522e-01, -1.239910593701e-02],
        [-1.956610938347e-01, 2.375640136002e-01],
        [8.671328947226e-01, 5.495675164121e-01],
        [7.387211594530e-01, 3.148136452269e-02],
        [2.228334781825e-02, -7.142653028885e-02],
        [1.002836732404e+00, -1.595869125350e-02],
        [7.550946401437e-01, 6.936161656262e-02],
        [5.652776988943e-01, -1.085625939466e-01],
        [-4.296491274440e-01, 4.164688022078e-02],
        [-1.166924098736e-01, 4.564924631871e-02],
        [7.474125806049e-01, -4.798842255656e-04],
        [4.513413975576e-01, 3.312546684549e-02],
        [8.538570338575e-01, -2.718145450203e-01],
    ],
    dtype=np.float64,
)


def _control_problem() -> ControlProblem:
    """Build the single-spin Liouville broadband 180 (Iz -> -Iz) GRAPE problem.

    Liouville generators already carry the -i factor (``liouvillian_comm``), so
    operators are 2*pi*RF*L without an extra phase. The RF power axis encodes
    the +-10 % B1 robustness window and ``offsets`` the proton offset band.
    """
    l_x = liouvillian_comm(Ix())
    l_y = liouvillian_comm(Iy())
    l_z = liouvillian_comm(Iz())
    offsets = np.linspace(-BANDWIDTH_HZ, BANDWIDTH_HZ, N_OPT_OFFSETS, dtype=np.float64)
    b1_scales = [
        1.0 - B1_DEVIATION_FRACTION,
        1.0,
        1.0 + B1_DEVIATION_FRACTION,
    ]
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=np.complex128)],
        operators=[TWO_PI * RF_MAX_HZ * l_x, TWO_PI * RF_MAX_HZ * l_y],
        rho_init=[vec(normalise_hs(Iz()))],
        rho_targ=[vec(-normalise_hs(Iz()))],
        pulse_dt=DT,
        pwr_levels=b1_scales,
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
        offsets=[float(o) for o in offsets],
        offset_operators=[TWO_PI * l_z],
        penalties=[PenaltySpec(kind="SNSA", weight=SNSA_WEIGHT, limit=1.0)],
    )


def _nominal_180_seed() -> RealArray:
    """Return a constant-amplitude 180_x seed waveform.

    A weak constant x field that integrates to a pi rotation on resonance keeps
    the optimiser in the broadband-inversion basin instead of a random local
    maximum.
    """
    fraction = 1.0 / (2.0 * RF_MAX_HZ * DURATION_S)
    seed = np.zeros((N_STEPS, 2), dtype=np.float64)
    seed[:, 0] = fraction
    return seed


def _optimise_wfm(max_iter: int) -> RealArray:
    """Run L-BFGS GRAPE and return the optimised (N_STEPS, 2) waveform."""
    cp = _control_problem()
    result = lbfgs_grape(cp, _nominal_180_seed(), max_iter=max_iter)
    print(
        f"OC broadband 180 GRAPE: fidelity = {result.fidelity_final:.6f}, "
        f"iterations = {result.n_iter}, reason = {result.reason}"
    )
    return np.asarray(result.wfm_final, dtype=np.float64)


def _single_spin_hamiltonian(bx_hz: float, by_hz: float, offset_hz: float) -> ComplexArray:
    """Return the single-spin Hamiltonian (rad/s) for given RF fields and offset."""
    return TWO_PI * (bx_hz * Ix() + by_hz * Iy() + offset_hz * Iz()).astype(np.complex128)


def _slice_propagator(bx_hz: float, by_hz: float, offset_hz: float, dt: float) -> ComplexArray:
    """Return the 2x2 propagator for one constant-field interval."""
    hamiltonian = _single_spin_hamiltonian(bx_hz, by_hz, offset_hz)
    return np.asarray(expm(np.complex128(-1j) * hamiltonian * dt), dtype=np.complex128)


def _oc_propagator(wfm_xy: RealArray, offset_hz: float, b1_scale: float) -> ComplexArray:
    """Return the total 2x2 propagator of the OC pulse at one offset/B1 point."""
    total = np.eye(2, dtype=np.complex128)
    for step in range(wfm_xy.shape[0]):
        total = _slice_propagator(
            b1_scale * RF_MAX_HZ * float(wfm_xy[step, 0]),
            b1_scale * RF_MAX_HZ * float(wfm_xy[step, 1]),
            offset_hz,
            DT,
        ) @ total
    return total


def _hard_180_propagator(offset_hz: float, b1_scale: float) -> ComplexArray:
    """Return the propagator of a rectangular 180_x pulse."""
    return _slice_propagator(b1_scale * RF_MAX_HZ, 0.0, offset_hz, 1.0 / (2.0 * RF_MAX_HZ))


def _composite_180_propagator(offset_hz: float, b1_scale: float) -> ComplexArray:
    """Return the Levitt-Freeman 90_x 180_y 90_x composite-pulse propagator."""
    quarter = 1.0 / (4.0 * RF_MAX_HZ)
    half = 1.0 / (2.0 * RF_MAX_HZ)
    segments = [
        (b1_scale * RF_MAX_HZ, 0.0, quarter),  # 90_x
        (0.0, b1_scale * RF_MAX_HZ, half),  # 180_y
        (b1_scale * RF_MAX_HZ, 0.0, quarter),  # 90_x
    ]
    total = np.eye(2, dtype=np.complex128)
    for bx, by, duration in segments:
        total = _slice_propagator(bx, by, offset_hz, duration) @ total
    return total


def _effective_cs(propagator: ComplexArray) -> tuple[float, float]:
    """Return effective (C, S) = (|<alpha|U|alpha>|, |<beta|U|alpha>|).

    |U[0, 0]| is the residual non-inverting amplitude that plays the role of
    C = cos(theta/2) in the paper; |U[1, 0]| plays the role of S = sin(theta/2).
    A perfect 180 pulse gives C = 0, S = 1 and hence zero artifacts.
    """
    return float(abs(propagator[0, 0])), float(abs(propagator[1, 0]))


def _artifact_intensities(c_eff: float, s_eff: float) -> tuple[float, float, float]:
    """Return (outer, inner, centre) multiplet intensities from Eq. 6 of the paper."""
    c2, s2 = c_eff * c_eff, s_eff * s_eff
    c4, s4 = c2 * c2, s2 * s2
    s6 = s4 * s2
    outer = 3.0 * s4 * c4
    inner = 8.0 * s4 * c2 - 4.0 * s2 * c4
    centre = 4.0 * s6 - 8.0 * s4 * c2 + 6.0 * s2 * c4
    return outer, inner, centre


def _relative_inner_artifact(propagator: ComplexArray) -> float:
    """Return |inner sideband| / |central line| in percent for one propagator."""
    c_eff, s_eff = _effective_cs(propagator)
    _, inner, centre = _artifact_intensities(c_eff, s_eff)
    if centre == 0.0:
        return math.inf
    return 100.0 * abs(inner) / abs(centre)


def _offset_profile(wfm_xy: RealArray) -> dict[str, RealArray]:
    """Return offset profiles of effective C and relative inner artifact."""
    offsets = np.linspace(-BANDWIDTH_HZ, BANDWIDTH_HZ, N_OFFSETS, dtype=np.float64)
    profile: dict[str, RealArray] = {"offsets": offsets}
    builders = {
        "hard": lambda o: _hard_180_propagator(o, 1.0),
        "composite": lambda o: _composite_180_propagator(o, 1.0),
        "oc": lambda o: _oc_propagator(wfm_xy, o, 1.0),
    }
    for name, build in builders.items():
        c_values = np.zeros(N_OFFSETS, dtype=np.float64)
        artifact = np.zeros(N_OFFSETS, dtype=np.float64)
        for index, offset in enumerate(offsets):
            propagator = build(float(offset))
            c_values[index] = _effective_cs(propagator)[0]
            artifact[index] = _relative_inner_artifact(propagator)
        profile[f"{name}_c"] = c_values
        profile[f"{name}_artifact"] = artifact
    return profile


def _export_bruker_shape(wfm_xy: RealArray, output_dir: Path) -> Path:
    """Write a variable-amplitude Bruker shape file for the OC pulse."""
    shape_path = output_dir / f"{PULSE_NAME}.shape"
    amplitude_percent = 100.0 * np.hypot(wfm_xy[:, 0], wfm_xy[:, 1])
    phase_deg = np.mod(np.degrees(np.arctan2(wfm_xy[:, 1], wfm_xy[:, 0])), 360.0)
    phase_deg[amplitude_percent < 1e-6] = 0.0
    integfac = float(np.mean(amplitude_percent)) / 100.0

    lines = [
        f"##TITLE= {PULSE_NAME}",
        "##JCAMP-DX= 5.00 Bruker JCAMP library",
        "##DATA TYPE= Shape Data",
        "##ORIGIN= optimalcontrol",
        "##OWNER= optimalcontrol",
        "##MINX= 0.000000e+00",
        "##MAXX= 1.000000e+02",
        "##MINY= 0.000000e+00",
        "##MAXY= 3.600000e+02",
        "##$SHAPE_EXMODE= None",
        "##$SHAPE_TOTROT= 1.800000e+02",
        "##$SHAPE_BWFAC= 0.000000e+00",
        f"##$SHAPE_INTEGFAC= {integfac:.9e}",
        "##$SHAPE_MODE= 1",
        f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {DURATION_S:.12e}",
        f"##$OPTIMALCONTROL_STEP_DURATION_S= {DT:.12e}",
        f"##$OPTIMALCONTROL_RF_HZ= {RF_MAX_HZ:.12e}",
        f"##$OPTIMALCONTROL_BANDWIDTH_HZ= {BANDWIDTH_HZ:.12e}",
        f"##$OPTIMALCONTROL_B1_DEVIATION_PERCENT= {100.0 * B1_DEVIATION_FRACTION:.12e}",
        "##$OPTIMALCONTROL_NOTE= Broadband 180 refocusing/inversion pulse.",
        f"##NPOINTS= {N_STEPS}",
        "##XYPOINTS= (XY..XY)",
    ]
    for amp, phase in zip(amplitude_percent, phase_deg):
        lines.append(f"{float(amp):.9e}, {float(phase):.9e}")
    lines.append("##END=")
    shape_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return shape_path


def _plot_figure(wfm_xy: RealArray, profile: dict[str, RealArray], output_dir: Path) -> Path:
    """Create the pulse + artifact-suppression comparison figure."""
    figure_path = output_dir / f"{PULSE_NAME}.png"
    offsets_khz = profile["offsets"] / 1000.0
    time_us = np.arange(N_STEPS, dtype=np.float64) * DT * 1.0e6
    amplitude_percent = 100.0 * np.hypot(wfm_xy[:, 0], wfm_xy[:, 1])
    phase_display = np.degrees(np.arctan2(wfm_xy[:, 1], wfm_xy[:, 0]))

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0))
    fig.subplots_adjust(hspace=0.5)

    ax_amp = axes[0]
    ax_phase = ax_amp.twinx()
    ax_amp.plot(time_us, amplitude_percent, color="tab:blue", linewidth=1.6)
    ax_phase.plot(time_us, phase_display, color="tab:red", linewidth=1.0, alpha=0.7)
    ax_amp.set_ylim(0.0, 120.0)
    ax_phase.set_ylim(-200.0, 200.0)
    ax_amp.set_xlabel("Time (us)")
    ax_amp.set_ylabel("Amplitude (% of RF max)", color="tab:blue")
    ax_phase.set_ylabel("Phase (deg)", color="tab:red")
    ax_amp.set_title(f"OC broadband 180 pulse ({DURATION_S * 1e3:.2f} ms, RF {RF_MAX_HZ / 1e3:.0f} kHz)")

    styles = {
        "hard": ("tab:gray", "hard 180"),
        "composite": ("tab:orange", "composite 90x180y90x"),
        "oc": ("tab:green", "OC broadband 180"),
    }

    ax_c = axes[1]
    for name, (color, label) in styles.items():
        ax_c.plot(offsets_khz, profile[f"{name}_c"], color=color, label=label, linewidth=1.6)
    ax_c.set_xlabel("1H offset (kHz)")
    ax_c.set_ylabel("Residual C = |<a|U|a>|")
    ax_c.set_title("Refocusing imperfection (C = 0 is ideal)")
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.legend(fontsize=8, loc="upper center")

    ax_art = axes[2]
    for name, (color, label) in styles.items():
        ax_art.semilogy(
            offsets_khz,
            np.maximum(profile[f"{name}_artifact"], 1e-6),
            color=color,
            label=label,
            linewidth=1.6,
        )
    ax_art.axhline(1.5, color="black", linestyle=":", linewidth=0.8, label="1.5 % (theta=170 deg)")
    ax_art.set_xlabel("1H offset (kHz)")
    ax_art.set_ylabel("Inner artifact / centre (%)")
    ax_art.set_title("HMQC methyl artifact intensity (Eq. 6)")
    ax_art.legend(fontsize=8, loc="upper center")

    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def run(wfm_xy: RealArray | None = None) -> RealArray:
    """Return the stacked offset profile for the cached or supplied OC pulse.

    The returned array has shape (7, N_OFFSETS): offsets (Hz) followed by the
    effective-C and relative-inner-artifact (%) profiles for the hard,
    composite, and OC pulses.
    """
    if wfm_xy is None:
        wfm_xy = OPTIMIZED_WFM_XY
    profile = _offset_profile(np.asarray(wfm_xy, dtype=np.float64))
    return np.vstack(
        [
            profile["offsets"],
            profile["hard_c"],
            profile["composite_c"],
            profile["oc_c"],
            profile["hard_artifact"],
            profile["composite_artifact"],
            profile["oc_artifact"],
        ]
    ).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimal-control HMQC 180 artifact suppression")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="regenerate the OC waveform with L-BFGS GRAPE instead of using the cache",
    )
    parser.add_argument("--max-iter", type=int, default=600, help="L-BFGS iterations")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    wfm_xy = _optimise_wfm(args.max_iter) if args.optimize else OPTIMIZED_WFM_XY
    profile = _offset_profile(np.asarray(wfm_xy, dtype=np.float64))

    shape_path = _export_bruker_shape(np.asarray(wfm_xy, dtype=np.float64), output_dir)
    figure_path = _plot_figure(np.asarray(wfm_xy, dtype=np.float64), profile, output_dir)

    band = np.abs(profile["offsets"]) <= BANDWIDTH_HZ
    for name in ("hard", "composite", "oc"):
        worst = float(np.max(profile[f"{name}_artifact"][band]))
        mean_c = float(np.mean(profile[f"{name}_c"][band]))
        print(f"{name:>9}: mean C = {mean_c:.4f}, worst inner artifact = {worst:8.3f} %")
    print(f"Saved Bruker shape {shape_path}")
    print(f"Saved figure {figure_path}")


if __name__ == "__main__":
    main()
