"""Band-selective optimal-control 180 that suppresses methyl-HMQC artifacts.

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
the inner satellites already reach ~1.5 % of the central line.

This script targets a concrete modern experiment: methyl detection on a
1.2 GHz (1H) spectrometer. The methyl region spans -1 ppm to 3 ppm, the
spectrum is recorded with the carrier on the water line at 4.7 ppm, and water
is suppressed with a 3-9-19 WATERGATE read element. At 1.2 GHz one ppm is
1200 Hz, so relative to the water carrier the methyl band sits at

    -1 ppm -> -6840 Hz   ...   3 ppm -> -2040 Hz

We design the central refocusing pulse with GRAPE optimal control under two
simultaneous goals (a *band-selective* inversion):

  * over the methyl band it drives a full 1H refocusing of transverse
    magnetization (-Iy -> Iy), so the residual non-inverted amplitude
    C = |<alpha|U|alpha>| -- the quantity that seeds the artifacts -- is
    pushed to ~0; and
  * at the water carrier (offset 0) it acts as the identity (Iz -> Iz, so
    C ~ 1, no inversion), leaving the water magnetization untouched so the
    3-9-19 WATERGATE can suppress it cleanly.

Both goals are enforced over a +-10 % B1 window. The whole pulse is short,
~500 microseconds, so it fits inside the t1 evolution of a real HMQC.

The script compares three central 180 pulses over a 1H offset/ppm profile:

  * hard      : a single rectangular 180_x pulse,
  * composite : the Levitt-Freeman 90_x 180_y 90_x composite pulse,
  * oc        : the band-selective GRAPE optimal-control pulse.

For each it extracts the effective C(offset) from the actual single-spin
propagator and evaluates the inner-sideband artifact intensity relative to the
central line in the methyl band, the quantity the paper measures. The hard and
composite pulses invert broadband and therefore also invert water (bad for the
following WATERGATE); the OC pulse inverts only the methyl band.

The default path uses the cached optimal-control waveform. Pass --optimize to
regenerate it with the combined L-BFGS GRAPE driver below.

run() returns the stacked offset profile used by the regression snapshot.

Saves:
  examples/output/hmqc_oc_180_artifact.shape
  examples/output/hmqc_oc_180_artifact.png
"""

import matplotlib

matplotlib.use("Agg")

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from optimalcontrol.grape import ControlProblem, grape_xy, grape_xy_and_gradient
from optimalcontrol.io import export_bruker_shape as write_bruker_shape
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, vec
from optimalcontrol.optimizers import (
    lbfgs_direction,
    lbfgs_state,
    lbfgs_update,
    line_search_cubic,
)
from optimalcontrol.penalties import PenaltySpec
from optimalcontrol.states import normalise_hs

RealArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

PULSE_NAME = "hmqc_oc_180_artifact"

# 1.2 GHz spectrometer: 1H Larmor frequency is 1200 MHz, so 1 ppm = 1200 Hz.
SPECTROMETER_1H_MHZ = 1200.0
HZ_PER_PPM = SPECTROMETER_1H_MHZ
WATER_PPM = 4.7  # carrier sits on the water line -> offset 0
METHYL_PPM_LO = -1.0
METHYL_PPM_HI = 3.0


def _ppm_to_offset_hz(ppm: float) -> float:
    """Return the rotating-frame offset (Hz) of a shift, carrier on water."""
    return (ppm - WATER_PPM) * HZ_PER_PPM


# Methyl band edges relative to the water carrier (Hz). LO < HI < 0.
METHYL_OFFSET_LO = _ppm_to_offset_hz(METHYL_PPM_LO)  # -6840 Hz
METHYL_OFFSET_HI = _ppm_to_offset_hz(METHYL_PPM_HI)  # -2040 Hz
METHYL_OFFSET_CENTRE = 0.5 * (METHYL_OFFSET_LO + METHYL_OFFSET_HI)  # -4440 Hz

N_STEPS = 50
DURATION_S = 5.0e-4  # ~500 microseconds, short enough to live inside t1
DT = DURATION_S / N_STEPS
RF_MAX_HZ = 10_000.0
B1_DEVIATION_FRACTION = 0.10
SNSA_WEIGHT = 0.02
TWO_PI = 2.0 * math.pi

# Optimization sampling.
N_OPT_METHYL = 13  # offsets across the methyl band
WATER_OPT_OFFSETS = (-50.0, 0.0, 50.0)  # narrow water line, robustness window
WATER_WEIGHT = 1.0  # weight of the water-identity goal vs. methyl inversion

# Analysis/plot sampling: span the methyl band through water.
PLOT_PPM_LO = -1.5
PLOT_PPM_HI = 6.0
N_OFFSETS = 151

# Cached band-selective GRAPE solution. Inverts the methyl band (-6840..-2040
# Hz off the water carrier) while leaving the water line (offset 0) untouched,
# over B1 scaling [0.9, 1.0, 1.1] at RF_MAX_HZ. Shape (N_STEPS, 2): columns are
# [u_x, u_y] as fractions of RF_MAX_HZ. Regenerate with --optimize.
OPTIMIZED_WFM_XY: RealArray = np.array(
    [
        [1.021316614245e+00, 8.606019735756e-02],
        [8.714426140707e-01, 5.052614481200e-01],
        [-5.396299447705e-01, -1.980066658417e-01],
        [-5.877404191445e-01, 5.333795621396e-01],
        [-9.840313142268e-01, 5.905312292814e-02],
        [-6.028464052117e-01, 3.086040038319e-01],
        [-2.709201041228e-01, 2.145832440758e-01],
        [1.874422702629e-02, 2.326390247057e-01],
        [2.464032481988e-02, 1.970843144572e-01],
        [2.132165109215e-01, 1.104188851755e-01],
        [2.099403075822e-01, 1.458020486254e-01],
        [4.475438083302e-02, 2.324218636434e-01],
        [-2.395248564960e-01, 1.042614804708e-01],
        [-3.234509571905e-01, -1.727606494094e-01],
        [-1.684855539940e-01, -1.910399819751e-01],
        [-7.427324952434e-02, 5.956497701227e-02],
        [-2.733619544813e-01, 1.836540296789e-01],
        [-2.289580656135e-01, 1.283386013301e-01],
        [1.372848366174e-01, 6.494349311275e-02],
        [5.350213557502e-01, 6.709711376262e-01],
        [-2.696619009094e-01, 9.611547622611e-01],
        [-9.859845427008e-01, 1.674219277529e-01],
        [2.922070477945e-01, -8.336914361550e-01],
        [1.002246488172e+00, 2.643007453535e-02],
        [7.190448310853e-01, 7.016114653047e-01],
        [4.784368053346e-01, 8.336483217653e-01],
        [9.590001577390e-01, 2.885900347616e-01],
        [9.898781117757e-01, -1.656530580429e-01],
        [8.067261172795e-01, -6.085581103182e-01],
        [3.860254606416e-01, -7.517327122896e-01],
        [-9.868904177819e-01, -1.805444365595e-01],
        [-5.213965780659e-01, -8.672324064200e-01],
        [6.444390794201e-01, -7.687385925900e-01],
        [9.007361512116e-01, 4.500809584163e-01],
        [6.326322516131e-01, 7.871295958177e-01],
        [5.160297075008e-01, 8.688235030351e-01],
        [-6.890364183861e-02, 1.010250438534e+00],
        [-7.561837898271e-01, 6.760953170670e-01],
        [-1.010320145818e+00, 1.493270935800e-01],
        [-9.713816183453e-01, -3.168242114494e-01],
        [-7.612542977933e-01, -6.677024322360e-01],
        [-2.863645590549e-01, -8.903107299434e-01],
        [4.499461160780e-02, 1.183825884695e-01],
        [-7.105046192455e-03, 2.075552344792e-01],
        [1.801180847419e-02, 1.185046341230e-01],
        [3.015229430689e-02, -6.174487811732e-03],
        [-9.913141691062e-02, 2.465808326372e-03],
        [-2.855905366280e-01, -3.757091233454e-01],
        [-1.291684496225e-01, -9.984090232701e-01],
        [7.537030675039e-01, -6.834333738096e-01],
    ],
    dtype=np.float64,
)


def _b1_scales() -> list[float]:
    """Return the +-B1_DEVIATION_FRACTION RF robustness window."""
    return [
        1.0 - B1_DEVIATION_FRACTION,
        1.0,
        1.0 + B1_DEVIATION_FRACTION,
    ]


def _liouville_problem(
    offsets: list[float],
    rho_init: ComplexArray,
    rho_targ: ComplexArray,
    *,
    with_penalty: bool,
) -> ControlProblem:
    """Build a single-spin Liouville GRAPE problem over an offset ensemble.

    Liouville generators already carry the -i factor (``liouvillian_comm``), so
    operators are 2*pi*RF*L without an extra phase. The RF power axis encodes
    the +-10 % B1 robustness window and ``offsets`` the proton offset band.
    """
    l_x = liouvillian_comm(Ix())
    l_y = liouvillian_comm(Iy())
    l_z = liouvillian_comm(Iz())
    penalties = (
        [PenaltySpec(kind="SNSA", weight=SNSA_WEIGHT, limit=1.0)] if with_penalty else None
    )
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=np.complex128)],
        operators=[TWO_PI * RF_MAX_HZ * l_x, TWO_PI * RF_MAX_HZ * l_y],
        rho_init=[vec(rho_init)],
        rho_targ=[vec(rho_targ)],
        pulse_dt=DT,
        pwr_levels=_b1_scales(),
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
        offsets=[float(o) for o in offsets],
        offset_operators=[TWO_PI * l_z],
        penalties=penalties,
    )


def _methyl_offsets() -> list[float]:
    return [
        float(o)
        for o in np.linspace(
            METHYL_OFFSET_LO, METHYL_OFFSET_HI, N_OPT_METHYL, dtype=np.float64
        )
    ]


def _methyl_problem_y() -> ControlProblem:
    """Methyl-band transverse refocusing (-Iy -> Iy); carries the SNSA penalty.

    A single state transfer is rank-deficient for a 180 rotation, so this is
    paired with ``_methyl_problem_z`` (Iz -> -Iz). The two transfers together
    pin the methyl propagator to a true 180_x = Rx(pi), which both satisfies
    the requested -Iy -> Iy *and* inverts the populations (Iz -> -Iz) so the
    artifact-seeding amplitude C = |<alpha|U|alpha>| collapses to ~0.
    """
    return _liouville_problem(
        _methyl_offsets(),
        -normalise_hs(Iy()),
        normalise_hs(Iy()),
        with_penalty=True,
    )


def _methyl_problem_z() -> ControlProblem:
    """Methyl-band population inversion (Iz -> -Iz); pairs with the y transfer."""
    return _liouville_problem(
        _methyl_offsets(),
        normalise_hs(Iz()),
        -normalise_hs(Iz()),
        with_penalty=False,
    )


def _water_problem() -> ControlProblem:
    """Return the water-line identity (Iz -> Iz) problem at offset ~0."""
    return _liouville_problem(
        list(WATER_OPT_OFFSETS),
        normalise_hs(Iz()),
        normalise_hs(Iz()),
        with_penalty=False,
    )


def _combined_evaluators() -> tuple[
    Callable[[RealArray], float], Callable[[RealArray], RealArray]
]:
    """Return (objective, gradient) for the full band-selective 180_x goal.

    fidelity = (F_my + F_mz + WATER_WEIGHT * F_water) / (2 + WATER_WEIGHT). The
    methyl band must satisfy *both* the transverse refocusing -Iy -> Iy (F_my,
    carrying the SNSA penalty) and the population inversion Iz -> -Iz (F_mz);
    together they pin a true 180_x. F_water rewards leaving the water untouched.
    """
    cp_my = _methyl_problem_y()
    cp_mz = _methyl_problem_z()
    cp_water = _water_problem()
    norm = 2.0 + WATER_WEIGHT

    def objective(wfm: RealArray) -> float:
        f_my = grape_xy(cp_my, wfm)
        f_mz = grape_xy(cp_mz, wfm)
        f_water = grape_xy(cp_water, wfm)
        return float((f_my + f_mz + WATER_WEIGHT * f_water) / norm)

    def gradient(wfm: RealArray) -> RealArray:
        _, g_my = grape_xy_and_gradient(cp_my, wfm)
        _, g_mz = grape_xy_and_gradient(cp_mz, wfm)
        _, g_water = grape_xy_and_gradient(cp_water, wfm)
        return np.asarray(
            (g_my + g_mz + WATER_WEIGHT * g_water) / norm, dtype=np.float64
        )

    return objective, gradient


def _inversion_evaluators() -> tuple[
    Callable[[RealArray], float], Callable[[RealArray], RealArray]
]:
    """Return (objective, gradient) for stage 1: inversion only (Iz -> -Iz).

    Targeting -Iy -> Iy alone is rank-deficient: the optimiser happily lands on
    a trivial Rz(pi) (flips x, y; leaves z) that does *not* invert populations,
    so the artifacts survive. The axis-free inversion goal Iz -> -Iz instead
    reliably finds a real 180 (an Ry-like solution) plus the water null. Stage 2
    then rotates that solution into the requested 180_x.
    """
    cp_mz = _methyl_problem_z()
    cp_water = _water_problem()
    norm = 1.0 + WATER_WEIGHT

    def objective(wfm: RealArray) -> float:
        return float(
            (grape_xy(cp_mz, wfm) + WATER_WEIGHT * grape_xy(cp_water, wfm)) / norm
        )

    def gradient(wfm: RealArray) -> RealArray:
        _, g_mz = grape_xy_and_gradient(cp_mz, wfm)
        _, g_water = grape_xy_and_gradient(cp_water, wfm)
        return np.asarray((g_mz + WATER_WEIGHT * g_water) / norm, dtype=np.float64)

    return objective, gradient


def _phase_rotate(wfm_xy: RealArray, phase_rad: float) -> RealArray:
    """Rotate every (u_x, u_y) sample by a constant RF phase.

    A global phase rotation rotates the effective rotation axis about z, turning
    an Ry(pi) inversion into Rx(pi) without disturbing the population inversion
    (Iz -> -Iz is axis-independent in the xy-plane) or the water identity.
    """
    c, s = math.cos(phase_rad), math.sin(phase_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(wfm_xy @ rot.T, dtype=np.float64)


def _band_selective_seed() -> RealArray:
    """Return a phase-ramped seed centred on the methyl band.

    A weak constant-amplitude field whose carrier is offset to the methyl-band
    centre integrates to roughly a pi rotation there, keeping the optimiser in
    the band-selective-inversion basin instead of inverting on resonance
    (the water line) where a constant-x seed would point.
    """
    fraction = 1.0 / (2.0 * RF_MAX_HZ * DURATION_S)
    times = (np.arange(N_STEPS, dtype=np.float64) + 0.5) * DT
    phase = TWO_PI * METHYL_OFFSET_CENTRE * times
    seed = np.zeros((N_STEPS, 2), dtype=np.float64)
    seed[:, 0] = fraction * np.cos(phase)
    seed[:, 1] = fraction * np.sin(phase)
    return seed


def _run_lbfgs(
    objective: Callable[[RealArray], float],
    gradient: Callable[[RealArray], RealArray],
    seed: RealArray,
    max_iter: int,
) -> RealArray:
    """Maximise ``objective`` from ``seed`` via the shared L-BFGS helpers."""
    waveform = np.asarray(seed, dtype=np.float64)
    state = lbfgs_state(m=10)
    grad = gradient(waveform)
    for _ in range(max_iter):
        direction = lbfgs_direction(state, grad)
        alpha = line_search_cubic(objective, gradient, waveform, direction)
        if alpha <= 0.0 and not np.array_equal(direction, grad):
            direction = grad.copy()
            alpha = line_search_cubic(objective, gradient, waveform, direction)
        if alpha <= 0.0:
            break
        step = np.asarray(alpha * direction, dtype=np.float64)
        new_waveform = np.asarray(waveform + step, dtype=np.float64)
        new_grad = gradient(new_waveform)
        state = lbfgs_update(state, new_waveform - waveform, grad - new_grad)
        waveform, grad = new_waveform, new_grad
        if float(np.linalg.norm(grad)) <= 1e-6:
            break
    return np.asarray(waveform, dtype=np.float64)


def _optimise_wfm(max_iter: int) -> RealArray:
    """Run the two-stage band-selective 180_x GRAPE and return the waveform.

    Stage 1 finds a robust axis-free band-selective inversion (Iz -> -Iz) plus
    the water null. Stage 2 rotates that Ry-like solution by 90 deg (Ry -> Rx)
    and refines against the full goal so the methyl band also satisfies the
    requested transverse refocusing -Iy -> Iy. The warm start is what lets the
    refinement reach a true 180_x instead of stalling in the trivial Rz basin.
    """
    inv_obj, inv_grad = _inversion_evaluators()
    inverted = _run_lbfgs(inv_obj, inv_grad, _band_selective_seed(), max_iter)

    full_obj, full_grad = _combined_evaluators()
    warm = _phase_rotate(inverted, math.pi / 2.0)
    waveform = _run_lbfgs(full_obj, full_grad, warm, max_iter)

    cp_my = _methyl_problem_y()
    cp_mz = _methyl_problem_z()
    cp_water = _water_problem()
    print(
        f"band-selective GRAPE: F_methyl(-Iy->Iy) = {grape_xy(cp_my, waveform):.6f}, "
        f"F_methyl(Iz->-Iz) = {grape_xy(cp_mz, waveform):.6f}, "
        f"F_water = {grape_xy(cp_water, waveform):.6f}, "
        f"combined = {full_obj(waveform):.6f}"
    )
    return waveform


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
    A perfect 180 pulse gives C = 0, S = 1 and hence zero artifacts; an identity
    (no inversion, as wanted at water) gives C = 1, S = 0.
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


def _plot_offsets_hz() -> RealArray:
    """Return the analysis offset axis (Hz) spanning the methyl band and water."""
    ppm = np.linspace(PLOT_PPM_LO, PLOT_PPM_HI, N_OFFSETS, dtype=np.float64)
    return np.asarray((ppm - WATER_PPM) * HZ_PER_PPM, dtype=np.float64)


def _offset_profile(wfm_xy: RealArray) -> dict[str, RealArray]:
    """Return offset profiles of effective C and relative inner artifact."""
    offsets = _plot_offsets_hz()
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
    amplitude_percent = 100.0 * np.hypot(wfm_xy[:, 0], wfm_xy[:, 1])
    phase_deg = np.mod(np.degrees(np.arctan2(wfm_xy[:, 1], wfm_xy[:, 0])), 360.0)
    phase_deg[amplitude_percent < 1e-6] = 0.0
    return write_bruker_shape(
        output_dir / f"{PULSE_NAME}.shape",
        PULSE_NAME,
        amplitude_percent,
        phase_deg,
        extra_tags=[
            f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {DURATION_S:.12e}",
            f"##$OPTIMALCONTROL_STEP_DURATION_S= {DT:.12e}",
            f"##$OPTIMALCONTROL_RF_HZ= {RF_MAX_HZ:.12e}",
            f"##$OPTIMALCONTROL_SPECTROMETER_1H_MHZ= {SPECTROMETER_1H_MHZ:.12e}",
            f"##$OPTIMALCONTROL_WATER_PPM= {WATER_PPM:.12e}",
            f"##$OPTIMALCONTROL_METHYL_PPM_LO= {METHYL_PPM_LO:.12e}",
            f"##$OPTIMALCONTROL_METHYL_PPM_HI= {METHYL_PPM_HI:.12e}",
            f"##$OPTIMALCONTROL_B1_DEVIATION_PERCENT= {100.0 * B1_DEVIATION_FRACTION:.12e}",
            "##$OPTIMALCONTROL_NOTE= Band-selective methyl 180; identity at water.",
        ],
    )


def _plot_figure(wfm_xy: RealArray, profile: dict[str, RealArray], output_dir: Path) -> Path:
    """Create the pulse + artifact-suppression comparison figure."""
    figure_path = output_dir / f"{PULSE_NAME}.png"
    offsets_ppm = WATER_PPM + profile["offsets"] / HZ_PER_PPM
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
    ax_amp.set_title(
        f"Band-selective OC 180 ({DURATION_S * 1e6:.0f} us, RF {RF_MAX_HZ / 1e3:.0f} kHz, "
        f"{SPECTROMETER_1H_MHZ / 1e3:.1f} GHz)"
    )

    styles = {
        "hard": ("tab:gray", "hard 180"),
        "composite": ("tab:orange", "composite 90x180y90x"),
        "oc": ("tab:green", "OC band-selective 180"),
    }

    def _mark_regions(ax: "plt.Axes") -> None:
        ax.axvspan(METHYL_PPM_LO, METHYL_PPM_HI, color="tab:green", alpha=0.08)
        ax.axvline(WATER_PPM, color="tab:blue", linestyle="--", linewidth=0.8)

    ax_c = axes[1]
    _mark_regions(ax_c)
    for name, (color, label) in styles.items():
        ax_c.plot(offsets_ppm, profile[f"{name}_c"], color=color, label=label, linewidth=1.6)
    ax_c.set_xlabel("1H shift (ppm)")
    ax_c.set_ylabel("Residual C = |<a|U|a>|")
    ax_c.set_title("C = 0 (inverted, methyl) and C = 1 (kept, water) are ideal")
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.legend(fontsize=8, loc="center right")

    ax_art = axes[2]
    methyl_mask = (offsets_ppm >= METHYL_PPM_LO) & (offsets_ppm <= METHYL_PPM_HI)
    for name, (color, label) in styles.items():
        ax_art.semilogy(
            offsets_ppm[methyl_mask],
            np.maximum(profile[f"{name}_artifact"][methyl_mask], 1e-6),
            color=color,
            label=label,
            linewidth=1.6,
        )
    ax_art.axhline(1.5, color="black", linestyle=":", linewidth=0.8, label="1.5 % (theta=170 deg)")
    ax_art.set_xlabel("1H shift (ppm)")
    ax_art.set_ylabel("Inner artifact / centre (%)")
    ax_art.set_title("HMQC methyl artifact intensity (Eq. 6)")
    ax_art.legend(fontsize=8, loc="upper center")

    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def run(wfm_xy: RealArray | None = None) -> RealArray:
    """Return the stacked offset profile for the cached or supplied OC pulse.

    The returned array has shape (7, N_OFFSETS): offsets (Hz, relative to the
    water carrier) followed by the effective-C and relative-inner-artifact (%)
    profiles for the hard, composite, and OC pulses.
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
    parser = argparse.ArgumentParser(description="Band-selective OC HMQC 180 artifact suppression")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="regenerate the OC waveform with the combined L-BFGS GRAPE driver",
    )
    parser.add_argument("--max-iter", type=int, default=800, help="L-BFGS iterations")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    wfm_xy = _optimise_wfm(args.max_iter) if args.optimize else OPTIMIZED_WFM_XY
    profile = _offset_profile(np.asarray(wfm_xy, dtype=np.float64))

    shape_path = _export_bruker_shape(np.asarray(wfm_xy, dtype=np.float64), output_dir)
    figure_path = _plot_figure(np.asarray(wfm_xy, dtype=np.float64), profile, output_dir)

    offsets_ppm = WATER_PPM + profile["offsets"] / HZ_PER_PPM
    methyl_mask = (offsets_ppm >= METHYL_PPM_LO) & (offsets_ppm <= METHYL_PPM_HI)
    water_index = int(np.argmin(np.abs(profile["offsets"])))
    for name in ("hard", "composite", "oc"):
        worst = float(np.max(profile[f"{name}_artifact"][methyl_mask]))
        mean_c = float(np.mean(profile[f"{name}_c"][methyl_mask]))
        water_c = float(profile[f"{name}_c"][water_index])
        print(
            f"{name:>9}: methyl mean C = {mean_c:.4f}, worst inner artifact = "
            f"{worst:8.3f} %, water C = {water_c:.4f}"
        )
    print(f"Saved Bruker shape {shape_path}")
    print(f"Saved figure {figure_path}")


if __name__ == "__main__":
    main()
