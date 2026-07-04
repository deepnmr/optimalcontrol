"""Shortest grid-searched methyl 180 pulse preserving water magnetization.

This pulse addresses the refocusing-pulse artifact described by Lewis E. Kay,
J. Biomol. NMR 73, 423-427 (2019), DOI 10.1007/s10858-019-00227-7.
Kay showed that an imperfect central proton 180-degree pulse produces methyl
HMQC satellites at +/-J_CH/2 that survive gradients and phase cycling.

Design specification
--------------------

* 1.2 GHz proton spectrometer, carrier at water (4.7 ppm).
* Methyl range -3.0 to 3.0 ppm: true 180_x, tested as Ix -> Ix,
  -Iy -> Iy, and Iz -> -Iz.
* Water: Iz -> Iz over 4.7 ppm +/-100 Hz.
* RF amplitude is variable but never exceeds 10 kHz.
* RF phase is binary: exactly 0 or 180 degrees.
* The signed-amplitude waveform is exactly time symmetric.

The half-waveform is the optimization variable. Mirroring it enforces time
symmetry structurally; a positive signed amplitude exports as phase 0 and a
negative one as phase 180. The cached 1.740 ms candidate is the shortest pulse
that passed the duration refinement search. It is an isolated, sharply tuned
optimum: the 5 us neighbours at 1.735 ms and 1.745 ms both fail the same dense
validation criterion.

Pass criteria on 2401 methyl offsets and 9 water offsets are:

* worst methyl Ix -> Ix, -Iy -> Iy, and Iz -> -Iz fidelity >= 0.999;
* worst water Iz -> Iz fidelity >= 0.999; and
* worst Kay inner-sideband artifact <= 0.1 percent of the central line.

Run without arguments to write the cached Bruker shape and diagnostic plot.
Use ``--optimize`` to repeat the local duration-grid refinement.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from optimalcontrol.bloch import propagate_bloch_ensemble
from optimalcontrol.grape import ControlProblem, grape_xy_and_gradient
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, vec
from optimalcontrol.states import normalise_hs

RealArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

PULSE_NAME = "methyl_water_binary_symmetric_180"
SPECTROMETER_1H_MHZ = 1200.0
WATER_PPM = 4.7
METHYL_PPM_LO = -3.0
METHYL_PPM_HI = 3.0
RF_MAX_HZ = 10_000.0
N_STEPS = 100
DURATION_US = 1740.0
DURATION_S = DURATION_US * 1.0e-6
DT = DURATION_S / N_STEPS
WATER_WINDOW_HZ = 100.0

MIN_METHYL_FIDELITY = 0.999
MIN_WATER_FIDELITY = 0.999
MAX_ARTIFACT_PERCENT = 0.1
VALIDATION_METHYL_POINTS = 2401
VALIDATION_WATER_POINTS = 9
SEARCH_STEP_US = 5.0

WIDE_PPM_LO = -6.0
WIDE_PPM_HI = 6.0
WIDE_PROFILE_POINTS = 1201
TIME_EVOLUTION_PPM: tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0, 4.7)


def _ppm_to_offset_hz(ppm: float) -> float:
    """Return proton offset in Hz relative to the water carrier."""
    return (ppm - WATER_PPM) * SPECTROMETER_1H_MHZ


METHYL_OFFSET_LO_HZ = _ppm_to_offset_hz(METHYL_PPM_LO)
METHYL_OFFSET_HI_HZ = _ppm_to_offset_hz(METHYL_PPM_HI)


# Optimized half of the signed-amplitude waveform. The second half is its
# reverse. Values are fractions of RF_MAX_HZ and therefore lie in [-1, 1].
OPTIMIZED_HALF_AMPLITUDE: RealArray = np.array(
    [
        -0.4212807521362277,
        0.7724448684701161,
        0.4385466829819327,
        -0.9939878392903073,
        -0.9999033849939682,
        0.6157204061457988,
        0.9861815582918109,
        0.48886199677612946,
        -0.7015867769211884,
        -0.8700966192621348,
        -0.16338927503083747,
        0.43306790878266893,
        0.3230765523766726,
        -0.20566070883488824,
        -0.4877875347400534,
        -0.30365419448916886,
        -0.05686353452279906,
        -0.11910850898488357,
        -0.3295649072364783,
        -0.29369615467785365,
        -0.012777809241314358,
        0.0780744728375549,
        -0.21887606182295505,
        -0.5017977934647291,
        -0.18810927186810888,
        0.6348749157908309,
        0.9541554552763705,
        0.2941522233505085,
        -0.31696135777156936,
        0.03943749339661754,
        0.9376743004287403,
        0.8920576805636677,
        0.6141850390025325,
        -0.10964961136097243,
        -0.32044658984747987,
        -0.10117600874082275,
        0.16207210602425426,
        0.17667065587039632,
        -0.0876440148411304,
        -0.38871975875002507,
        -0.4760820435489419,
        -0.4054875519433084,
        -0.3208650706064698,
        -0.08329200508459712,
        0.28490188193723087,
        -0.11183710221093739,
        -1.0,
        -0.5403556509928031,
        0.9997643085813476,
        0.9999884516547032,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PulseMetrics:
    """Dense-grid performance summary for one pulse duration."""

    methyl_x_min: float
    methyl_y_min: float
    methyl_z_min: float
    water_z_min: float
    artifact_max_percent: float
    methyl_y_mean: float
    methyl_z_mean: float

    @property
    def passes(self) -> bool:
        return (
            self.methyl_x_min >= MIN_METHYL_FIDELITY
            and self.methyl_y_min >= MIN_METHYL_FIDELITY
            and self.methyl_z_min >= MIN_METHYL_FIDELITY
            and self.water_z_min >= MIN_WATER_FIDELITY
            and self.artifact_max_percent <= MAX_ARTIFACT_PERCENT
        )

    def as_array(self) -> RealArray:
        return np.array(
            [
                self.methyl_x_min,
                self.methyl_y_min,
                self.methyl_z_min,
                self.water_z_min,
                self.artifact_max_percent,
                self.methyl_y_mean,
                self.methyl_z_mean,
            ],
            dtype=np.float64,
        )


def signed_amplitude(half: RealArray = OPTIMIZED_HALF_AMPLITUDE) -> RealArray:
    """Return the exactly symmetric signed RF-amplitude waveform."""
    half_array = np.asarray(half, dtype=np.float64)
    if half_array.shape != (N_STEPS // 2,):
        raise ValueError(f"half waveform must have shape ({N_STEPS // 2},)")
    if np.max(np.abs(half_array)) > 1.0 + 1e-12:
        raise ValueError("signed amplitude exceeds the 10 kHz RF limit")
    return np.asarray(np.concatenate((half_array, half_array[::-1])), dtype=np.float64)


def amplitude_phase(signed: RealArray) -> tuple[RealArray, RealArray]:
    """Convert signed amplitude to non-negative amplitude and binary phase."""
    signed_array = np.asarray(signed, dtype=np.float64)
    amplitude = np.abs(signed_array)
    phase_deg = np.where(signed_array < 0.0, 180.0, 0.0)
    phase_deg[amplitude <= 1e-15] = 0.0
    return amplitude, np.asarray(phase_deg, dtype=np.float64)


def _xy_waveform(signed: RealArray) -> RealArray:
    """Return the package XY representation of a binary-phase waveform."""
    signed_array = np.asarray(signed, dtype=np.float64)
    return np.asarray(
        np.column_stack((signed_array, np.zeros_like(signed_array))), dtype=np.float64
    )


def _artifact_percent_from_z_fidelity(z_fidelity: RealArray) -> RealArray:
    """Return Kay's +/-J_CH/2 artifact percentage from inversion fidelity.

    For initial Iz, the residual population amplitude obeys
    C^2=(1-Fz)/2 and S^2=1-C^2. Substitution into Kay's multiplet
    intensities gives the inner-sideband intensity relative to the centre.
    """
    c2 = np.clip(0.5 * (1.0 - z_fidelity), 0.0, 1.0)
    s2 = 1.0 - c2
    inner = 8.0 * s2**2 * c2 - 4.0 * s2 * c2**2
    centre = 4.0 * s2**3 - 8.0 * s2**2 * c2 + 6.0 * s2 * c2**2
    return np.asarray(
        100.0 * np.abs(inner) / np.maximum(np.abs(centre), 1e-15),
        dtype=np.float64,
    )


def evaluate_pulse(
    signed: RealArray,
    duration_s: float = DURATION_S,
    methyl_points: int = VALIDATION_METHYL_POINTS,
    rf_max_hz: float = RF_MAX_HZ,
) -> tuple[PulseMetrics, dict[str, RealArray]]:
    """Evaluate all constraints on dense methyl and water grids."""
    waveform_xy = _xy_waveform(signed)
    methyl_offsets = np.linspace(
        METHYL_OFFSET_LO_HZ, METHYL_OFFSET_HI_HZ, methyl_points, dtype=np.float64
    )
    water_offsets = np.linspace(
        -WATER_WINDOW_HZ,
        WATER_WINDOW_HZ,
        VALIDATION_WATER_POINTS,
        dtype=np.float64,
    )
    dt = duration_s / signed.size
    scales = np.array([1.0], dtype=np.float64)
    methyl_x = propagate_bloch_ensemble(
        np.array([1.0, 0.0, 0.0]),
        waveform_xy,
        methyl_offsets,
        scales,
        rf_max_hz,
        dt,
    )[0, :, 0]
    methyl_y = propagate_bloch_ensemble(
        np.array([0.0, -1.0, 0.0]),
        waveform_xy,
        methyl_offsets,
        scales,
        rf_max_hz,
        dt,
    )[0, :, 1]
    methyl_z = -propagate_bloch_ensemble(
        np.array([0.0, 0.0, 1.0]),
        waveform_xy,
        methyl_offsets,
        scales,
        rf_max_hz,
        dt,
    )[0, :, 2]
    water_z = propagate_bloch_ensemble(
        np.array([0.0, 0.0, 1.0]),
        waveform_xy,
        water_offsets,
        scales,
        rf_max_hz,
        dt,
    )[0, :, 2]
    artifact = _artifact_percent_from_z_fidelity(methyl_z)
    metrics = PulseMetrics(
        methyl_x_min=float(np.min(methyl_x)),
        methyl_y_min=float(np.min(methyl_y)),
        methyl_z_min=float(np.min(methyl_z)),
        water_z_min=float(np.min(water_z)),
        artifact_max_percent=float(np.max(artifact)),
        methyl_y_mean=float(np.mean(methyl_y)),
        methyl_z_mean=float(np.mean(methyl_z)),
    )
    profiles = {
        "methyl_offsets_hz": methyl_offsets,
        "methyl_x": np.asarray(methyl_x, dtype=np.float64),
        "methyl_y": np.asarray(methyl_y, dtype=np.float64),
        "methyl_z": np.asarray(methyl_z, dtype=np.float64),
        "artifact_percent": artifact,
        "water_offsets_hz": water_offsets,
        "water_z": np.asarray(water_z, dtype=np.float64),
    }
    return metrics, profiles


def wide_transfer_profile(
    signed: RealArray,
    duration_s: float = DURATION_S,
    ppm_lo: float = WIDE_PPM_LO,
    ppm_hi: float = WIDE_PPM_HI,
    points: int = WIDE_PROFILE_POINTS,
    rf_max_hz: float = RF_MAX_HZ,
) -> dict[str, RealArray]:
    """Return Ix, -Iy, Iz transfer profiles over a wide ppm grid.

    Unlike :func:`evaluate_pulse`, the grid spans the full -6..6 ppm window
    (methyl band, water, and margin beyond both) rather than the
    constraint-validation range alone.
    """
    waveform_xy = _xy_waveform(signed)
    ppm = np.linspace(ppm_lo, ppm_hi, points, dtype=np.float64)
    offsets_hz = (ppm - WATER_PPM) * SPECTROMETER_1H_MHZ
    dt = duration_s / signed.size
    scales = np.array([1.0], dtype=np.float64)
    transfer_x = propagate_bloch_ensemble(
        np.array([1.0, 0.0, 0.0]), waveform_xy, offsets_hz, scales, rf_max_hz, dt
    )[0, :, 0]
    transfer_y = -propagate_bloch_ensemble(
        np.array([0.0, 1.0, 0.0]), waveform_xy, offsets_hz, scales, rf_max_hz, dt
    )[0, :, 1]
    transfer_z = -propagate_bloch_ensemble(
        np.array([0.0, 0.0, 1.0]), waveform_xy, offsets_hz, scales, rf_max_hz, dt
    )[0, :, 2]
    return {
        "ppm": ppm,
        "x": np.asarray(transfer_x, dtype=np.float64),
        "y": np.asarray(transfer_y, dtype=np.float64),
        "z": np.asarray(transfer_z, dtype=np.float64),
    }


def wide_component_profile(
    signed: RealArray,
    duration_s: float = DURATION_S,
    ppm_lo: float = WIDE_PPM_LO,
    ppm_hi: float = WIDE_PPM_HI,
    points: int = WIDE_PROFILE_POINTS,
    rf_max_hz: float = RF_MAX_HZ,
) -> dict[str, RealArray]:
    """Return final Mx, My, Mz components for each initial Ix, Iy, Iz.

    For every offset on the -6..6 ppm grid the pulse is applied to each of
    the three axis-aligned initial states; the full final Bloch vector is
    recorded so the complete state mapping (not just the diagonal transfer)
    is visible.
    """
    waveform_xy = _xy_waveform(signed)
    ppm = np.linspace(ppm_lo, ppm_hi, points, dtype=np.float64)
    offsets_hz = (ppm - WATER_PPM) * SPECTROMETER_1H_MHZ
    dt = duration_s / signed.size
    scales = np.array([1.0], dtype=np.float64)
    initials = {
        "Ix": np.array([1.0, 0.0, 0.0]),
        "Iy": np.array([0.0, 1.0, 0.0]),
        "Iz": np.array([0.0, 0.0, 1.0]),
    }
    profile: dict[str, RealArray] = {"ppm": ppm}
    for name, initial in initials.items():
        final = propagate_bloch_ensemble(
            initial, waveform_xy, offsets_hz, scales, rf_max_hz, dt
        )[0]
        profile[name] = np.asarray(final, dtype=np.float64)
    return profile


def time_evolution_profiles(
    signed: RealArray,
    ppms: tuple[float, ...] = TIME_EVOLUTION_PPM,
    duration_s: float = DURATION_S,
    rf_max_hz: float = RF_MAX_HZ,
) -> dict[str, RealArray]:
    """Return full Mx, My, Mz trajectories for each initial Ix, Iy, Iz state.

    For every requested offset the pulse is applied to each axis-aligned
    initial state, and the complete Bloch vector is recorded after every
    pulse step by re-propagating successively longer waveform prefixes.
    The trajectory for key ``"Ix"`` has shape ``(n_steps + 1, n_offsets, 3)``
    where the last axis indexes Mx, My, Mz.
    """
    waveform_xy = _xy_waveform(signed)
    n_steps = signed.size
    dt = duration_s / n_steps
    offsets_hz = np.array(
        [(ppm - WATER_PPM) * SPECTROMETER_1H_MHZ for ppm in ppms], dtype=np.float64
    )
    scales = np.array([1.0], dtype=np.float64)
    time_us = (np.arange(n_steps + 1, dtype=np.float64)) * dt * 1e6

    initials = {
        "Ix": np.array([1.0, 0.0, 0.0]),
        "Iy": np.array([0.0, 1.0, 0.0]),
        "Iz": np.array([0.0, 0.0, 1.0]),
    }
    trajectories: dict[str, RealArray] = {
        "time_us": time_us,
        "ppm": np.asarray(ppms, dtype=np.float64),
    }
    for name, initial in initials.items():
        trajectory = np.empty((n_steps + 1, offsets_hz.size, 3), dtype=np.float64)
        trajectory[0] = initial
        for step in range(1, n_steps + 1):
            result = propagate_bloch_ensemble(
                initial, waveform_xy[:step], offsets_hz, scales, rf_max_hz, dt
            )
            trajectory[step] = result[0]
        trajectories[name] = trajectory
    return trajectories


def _control_problem(
    offsets_hz: RealArray,
    rho_init: list[ComplexArray],
    rho_targ: list[ComplexArray],
    duration_s: float,
    rf_max_hz: float = RF_MAX_HZ,
) -> ControlProblem:
    """Build a one-channel, nominal-B1 Liouville optimization problem."""
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=np.complex128)],
        operators=[2.0 * math.pi * rf_max_hz * liouvillian_comm(Ix())],
        rho_init=[vec(normalise_hs(state)) for state in rho_init],
        rho_targ=[vec(normalise_hs(state)) for state in rho_targ],
        pulse_dt=duration_s / N_STEPS,
        pwr_levels=[1.0],
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
        offsets=[float(offset) for offset in offsets_hz],
        offset_operators=[2.0 * math.pi * liouvillian_comm(Iz())],
    )


def _collapse_symmetric_gradient(gradient: RealArray) -> RealArray:
    """Apply the mirror-parameterization chain rule to a full gradient."""
    half = N_STEPS // 2
    return np.asarray(
        gradient[:half, 0] + gradient[half:, 0][::-1], dtype=np.float64
    )


def _constraint_score(metrics: PulseMetrics) -> float:
    """Return the worst normalized constraint, with 0.999 as the pass boundary."""
    return min(
        metrics.methyl_x_min,
        metrics.methyl_y_min,
        metrics.methyl_z_min,
        metrics.water_z_min,
        1.0 - metrics.artifact_max_percent / 100.0,
    )


def refine_duration(
    duration_us: float,
    half_seed: RealArray,
    max_outer: int = 8,
    max_iter: int = 160,
) -> tuple[RealArray, PulseMetrics]:
    """Refine one duration using adaptive worst-offset weighting.

    Every accepted iterate is scored on the full 2401-point validation grid;
    an optimizer endpoint is never accepted merely because its training-grid
    mean is high.
    """
    duration_s = duration_us * 1e-6
    base_offsets = np.linspace(
        METHYL_OFFSET_LO_HZ, METHYL_OFFSET_HI_HZ, 81, dtype=np.float64
    )
    validation_offsets = np.linspace(
        METHYL_OFFSET_LO_HZ,
        METHYL_OFFSET_HI_HZ,
        VALIDATION_METHYL_POINTS,
        dtype=np.float64,
    )
    water_offsets = np.linspace(
        -WATER_WINDOW_HZ,
        WATER_WINDOW_HZ,
        VALIDATION_WATER_POINTS,
        dtype=np.float64,
    )
    best = np.asarray(half_seed, dtype=np.float64).copy()
    best_metrics, best_profiles = evaluate_pulse(
        signed_amplitude(best), duration_s=duration_s
    )

    best_score = _constraint_score(best_metrics)
    for _ in range(max_outer):
        x_indices = np.argsort(best_profiles["methyl_x"])[:30]
        y_indices = np.argsort(best_profiles["methyl_y"])[:30]
        z_indices = np.argsort(best_profiles["methyl_z"])[:30]
        x_offsets = np.concatenate(
            (base_offsets, np.repeat(validation_offsets[x_indices], 30))
        )
        y_offsets = np.concatenate(
            (base_offsets, np.repeat(validation_offsets[y_indices], 30))
        )
        z_offsets = np.concatenate(
            (base_offsets, np.repeat(validation_offsets[z_indices], 30))
        )
        cp_x = _control_problem(x_offsets, [Ix()], [Ix()], duration_s)
        cp_y = _control_problem(y_offsets, [-Iy()], [Iy()], duration_s)
        cp_z = _control_problem(z_offsets, [Iz()], [-Iz()], duration_s)
        cp_water = _control_problem(water_offsets, [Iz()], [Iz()], duration_s)
        local_best = best.copy()
        local_metrics = best_metrics
        local_profiles = best_profiles
        local_score = best_score

        def objective(half: RealArray) -> tuple[float, RealArray]:
            waveform = signed_amplitude(half)[:, None]
            value_x, gradient_x = grape_xy_and_gradient(cp_x, waveform)
            value_y, gradient_y = grape_xy_and_gradient(cp_y, waveform)
            value_z, gradient_z = grape_xy_and_gradient(cp_z, waveform)
            value_water, gradient_water = grape_xy_and_gradient(cp_water, waveform)
            value = (value_x + value_y + value_z + value_water) / 4.0
            gradient = _collapse_symmetric_gradient(
                (gradient_x + gradient_y + gradient_z + gradient_water) / 4.0
            )
            return -value, -gradient

        def callback(half: RealArray) -> None:
            nonlocal local_best, local_metrics, local_profiles, local_score
            metrics, profiles = evaluate_pulse(
                signed_amplitude(half), duration_s=duration_s
            )
            candidate_score = _constraint_score(metrics)
            if candidate_score > local_score:
                local_best = np.asarray(half, dtype=np.float64).copy()
                local_metrics = metrics
                local_profiles = profiles
                local_score = candidate_score

        minimize(
            objective,
            best,
            jac=True,
            callback=callback,
            method="L-BFGS-B",
            bounds=[(-1.0, 1.0)] * (N_STEPS // 2),
            options={
                "maxiter": max_iter,
                "ftol": 1e-14,
                "gtol": 1e-9,
                "maxls": 50,
            },
        )
        if local_score <= best_score + 1e-10:
            break
        best = local_best
        best_metrics = local_metrics
        best_profiles = local_profiles
        best_score = local_score
    return best, best_metrics


def search_minimum_duration() -> tuple[float, RealArray, list[tuple[float, PulseMetrics]]]:
    """Repeat the 5 us local grid search around the cached boundary."""
    half = OPTIMIZED_HALF_AMPLITUDE.copy()
    audit: list[tuple[float, PulseMetrics]] = []
    passing: list[tuple[float, RealArray, PulseMetrics]] = []
    for duration_us in (1800.0, 1775.0, 1750.0, 1745.0, 1740.0, 1735.0):
        duration_s = duration_us * 1e-6
        warm_metrics, _ = evaluate_pulse(signed_amplitude(half), duration_s=duration_s)
        cached_metrics, _ = evaluate_pulse(
            signed_amplitude(OPTIMIZED_HALF_AMPLITUDE), duration_s=duration_s
        )
        seed = (
            half
            if _constraint_score(warm_metrics) >= _constraint_score(cached_metrics)
            else OPTIMIZED_HALF_AMPLITUDE
        )
        half, metrics = refine_duration(duration_us, seed)
        audit.append((duration_us, metrics))
        if metrics.passes:
            passing.append((duration_us, half.copy(), metrics))
    if not passing:
        raise RuntimeError("no duration candidate met the pulse constraints")
    duration_us, best_half, _ = min(passing, key=lambda candidate: candidate[0])
    return duration_us, best_half, audit


def export_bruker_shape(signed: RealArray, output_dir: Path) -> Path:
    """Write the variable-amplitude, binary-phase Bruker shape file."""
    amplitude, phase_deg = amplitude_phase(signed)
    shape_path = output_dir / f"{PULSE_NAME}.shape"
    lines = [
        f"##TITLE= {PULSE_NAME}",
        "##JCAMP-DX= 5.00 Bruker JCAMP library",
        "##DATA TYPE= Shape Data",
        "##ORIGIN= optimalcontrol",
        "##OWNER= optimalcontrol",
        "##MINX= 0.000000e+00",
        "##MAXX= 1.000000e+02",
        "##MINY= 0.000000e+00",
        "##MAXY= 1.800000e+02",
        "##$SHAPE_EXMODE= None",
        "##$SHAPE_TOTROT= 1.800000e+02",
        f"##$SHAPE_INTEGFAC= {float(np.mean(amplitude)):.9e}",
        "##$SHAPE_MODE= 1",
        f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {DURATION_S:.12e}",
        f"##$OPTIMALCONTROL_STEP_DURATION_S= {DT:.12e}",
        f"##$OPTIMALCONTROL_RF_MAX_HZ= {RF_MAX_HZ:.12e}",
        f"##$OPTIMALCONTROL_SPECTROMETER_1H_MHZ= {SPECTROMETER_1H_MHZ:.12e}",
        f"##$OPTIMALCONTROL_METHYL_PPM= {METHYL_PPM_LO:.3f}..{METHYL_PPM_HI:.3f}",
        f"##$OPTIMALCONTROL_WATER_PPM= {WATER_PPM:.3f}",
        "##$OPTIMALCONTROL_PHASE_SET_DEG= 0,180",
        "##$OPTIMALCONTROL_SYMMETRIC= yes",
        f"##NPOINTS= {signed.size}",
        "##XYPOINTS= (XY..XY)",
    ]
    for amplitude_fraction, phase in zip(amplitude, phase_deg):
        lines.append(f"{100.0 * float(amplitude_fraction):.9e}, {float(phase):.9e}")
    lines.append("##END=")
    shape_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return shape_path


def plot_diagnostics(
    signed: RealArray,
    metrics: PulseMetrics,
    profiles: dict[str, RealArray],
    output_dir: Path,
) -> Path:
    """Write pulse constraints and dense validation profiles as a PNG."""
    amplitude, phase_deg = amplitude_phase(signed)
    time_us = (np.arange(signed.size, dtype=np.float64) + 0.5) * DT * 1e6
    methyl_ppm = WATER_PPM + profiles["methyl_offsets_hz"] / SPECTROMETER_1H_MHZ
    figure, axes = plt.subplots(4, 1, figsize=(9.0, 10.0), constrained_layout=True)

    axes[0].step(time_us, 100.0 * amplitude, where="mid", color="tab:blue")
    phase_axis = axes[0].twinx()
    phase_axis.step(time_us, phase_deg, where="mid", color="tab:red", alpha=0.55)
    axes[0].set_ylabel("Amplitude (%)")
    phase_axis.set_ylabel("Phase (deg)")
    phase_axis.set_yticks([0.0, 180.0])
    axes[0].set_title(
        f"Symmetric binary-phase methyl 180: {DURATION_US:.0f} us, 10 kHz max"
    )

    axes[1].plot(methyl_ppm, profiles["methyl_x"], label="Ix -> Ix")
    axes[1].plot(methyl_ppm, profiles["methyl_y"], label="-Iy -> Iy")
    axes[1].plot(methyl_ppm, profiles["methyl_z"], label="Iz -> -Iz")
    axes[1].axhline(MIN_METHYL_FIDELITY, color="black", linestyle=":")
    axes[1].set_xlim(METHYL_PPM_LO, METHYL_PPM_HI)
    axes[1].set_ylim(0.9985, 1.0001)
    axes[1].set_ylabel("Transfer fidelity")
    axes[1].legend()

    axes[2].semilogy(methyl_ppm, profiles["artifact_percent"], color="tab:green")
    axes[2].axhline(MAX_ARTIFACT_PERCENT, color="black", linestyle=":")
    axes[2].set_xlim(METHYL_PPM_LO, METHYL_PPM_HI)
    axes[2].set_ylabel("Kay artifact (%)")
    axes[2].set_xlabel("Methyl 1H shift (ppm)")

    axes[3].plot(profiles["water_offsets_hz"], profiles["water_z"], marker="o")
    axes[3].axhline(MIN_WATER_FIDELITY, color="black", linestyle=":")
    axes[3].set_ylim(0.9985, 1.0001)
    axes[3].set_xlabel("Offset from water at 4.7 ppm (Hz)")
    axes[3].set_ylabel("Water Iz -> Iz")
    axes[3].set_title(
        f"Worst: methyl x={metrics.methyl_x_min:.6f}, "
        f"y={metrics.methyl_y_min:.6f}, "
        f"z={metrics.methyl_z_min:.6f}, water={metrics.water_z_min:.6f}, "
        f"artifact={metrics.artifact_max_percent:.4f}%"
    )

    figure_path = output_dir / f"{PULSE_NAME}.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_wide_profile(wide_profile: dict[str, RealArray], output_dir: Path) -> Path:
    """Write Ix, -Iy, Iz transfer profiles over the full -6..6 ppm window."""
    ppm = wide_profile["ppm"]
    figure, axis = plt.subplots(1, 1, figsize=(9.0, 4.5), constrained_layout=True)
    axis.plot(ppm, wide_profile["x"], label="Ix -> Ix")
    axis.plot(ppm, wide_profile["y"], label="Iy -> -Iy")
    axis.plot(ppm, wide_profile["z"], label="Iz -> -Iz")
    axis.axvline(METHYL_PPM_LO, color="black", linestyle=":", linewidth=0.8)
    axis.axvline(METHYL_PPM_HI, color="black", linestyle=":", linewidth=0.8)
    axis.axvline(WATER_PPM, color="tab:gray", linestyle="--", linewidth=0.8)
    axis.set_xlim(WIDE_PPM_LO, WIDE_PPM_HI)
    axis.set_xlabel("1H shift (ppm)")
    axis.set_ylabel("Transfer fidelity")
    axis.set_title("Wide-offset transfer profile (dotted: methyl band, dashed: water)")
    axis.legend()

    figure_path = output_dir / f"{PULSE_NAME}_wide_profile.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_wide_components(component_profile: dict[str, RealArray], output_dir: Path) -> Path:
    """Write final Mx, My, Mz vs ppm for each initial Ix, Iy, Iz state."""
    ppm = component_profile["ppm"]
    figure, axes = plt.subplots(
        3, 1, figsize=(9.0, 9.0), sharex=True, constrained_layout=True
    )
    for axis, name in zip(axes, ("Ix", "Iy", "Iz")):
        final = component_profile[name]
        axis.plot(ppm, final[:, 0], label="Mx", color="tab:blue")
        axis.plot(ppm, final[:, 1], label="My", color="tab:orange")
        axis.plot(ppm, final[:, 2], label="Mz", color="tab:green")
        axis.axvline(METHYL_PPM_LO, color="black", linestyle=":", linewidth=0.8)
        axis.axvline(METHYL_PPM_HI, color="black", linestyle=":", linewidth=0.8)
        axis.axvline(WATER_PPM, color="tab:gray", linestyle="--", linewidth=0.8)
        axis.set_xlim(WIDE_PPM_LO, WIDE_PPM_HI)
        axis.set_ylim(-1.05, 1.05)
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.set_ylabel(f"from {name}")
    axes[0].legend(ncol=3, fontsize="small")
    axes[0].set_title("Final Bloch components vs offset (dotted: methyl band, dashed: water)")
    axes[-1].set_xlabel("1H shift (ppm)")

    figure_path = output_dir / f"{PULSE_NAME}_wide_components.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def plot_time_evolution(time_evolution: dict[str, RealArray], output_dir: Path) -> Path:
    """Write Mx, My, Mz time evolution per initial state and offset as a PNG.

    Rows index the initial state (Ix, Iy, Iz); columns index the offset.
    Each panel overlays the three Bloch components over the pulse duration.
    """
    time_us = time_evolution["time_us"]
    ppms = time_evolution["ppm"]
    inits = ("Ix", "Iy", "Iz")
    n_rows = len(inits)
    n_cols = ppms.size

    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.4 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row, name in enumerate(inits):
        trajectory = time_evolution[name]
        for col in range(n_cols):
            axis = axes[row, col]
            axis.plot(time_us, trajectory[:, col, 0], label="Mx", color="tab:blue")
            axis.plot(time_us, trajectory[:, col, 1], label="My", color="tab:orange")
            axis.plot(time_us, trajectory[:, col, 2], label="Mz", color="tab:green")
            axis.axhline(0.0, color="black", linewidth=0.5)
            axis.set_ylim(-1.05, 1.05)
            if row == 0:
                axis.set_title(f"{ppms[col]:g} ppm")
            if col == 0:
                axis.set_ylabel(f"from {name}")
            if row == n_rows - 1:
                axis.set_xlabel("Time (us)")
    axes[0, 0].legend(ncol=3, fontsize="x-small", loc="lower left")
    figure.suptitle(f"Bloch-component time evolution ({DURATION_US:.0f} us pulse)")

    figure_path = output_dir / f"{PULSE_NAME}_time_evolution.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def run() -> RealArray:
    """Generate artifacts and return waveform plus dense summary metrics."""
    signed = signed_amplitude()
    metrics, profiles = evaluate_pulse(signed)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_bruker_shape(signed, output_dir)
    plot_diagnostics(signed, metrics, profiles, output_dir)
    plot_wide_profile(wide_transfer_profile(signed), output_dir)
    plot_wide_components(wide_component_profile(signed), output_dir)
    plot_time_evolution(time_evolution_profiles(signed), output_dir)
    return np.concatenate((signed, metrics.as_array()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimize", action="store_true", help="repeat the local duration-grid refinement"
    )
    args = parser.parse_args()
    if args.optimize:
        duration_us, half, audit = search_minimum_duration()
        for candidate_us, metrics in audit:
            print(f"{candidate_us:.0f} us: pass={metrics.passes} {metrics}")
        print(f"Shortest passing grid candidate: {duration_us:.0f} us")
        print("Optimized half waveform:")
        print(repr(half.tolist()))
        return

    result = run()
    metrics, _ = evaluate_pulse(result[:N_STEPS])
    print(metrics)
    print(f"Saved examples/output/{PULSE_NAME}.shape")
    print(f"Saved examples/output/{PULSE_NAME}.png")


if __name__ == "__main__":
    main()
