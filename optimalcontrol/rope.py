"""Analytical ROPE efficiency helpers from the JMR 2003 paper."""

import math
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq


def _validate_nonnegative(name: str, value: float) -> None:
    """Raise ValueError if a scalar parameter is outside its physical domain."""
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_positive(name: str, value: float) -> None:
    """Raise ValueError if a scalar parameter is not strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _arccot_nonnegative(x: float) -> float:
    """Return arccot(x) on the physical x >= 0 branch, in radians."""
    _validate_nonnegative("x", x)
    return math.atan2(1.0, x)


def _coth(x: float) -> float:
    """Return coth(x)."""
    return 1.0 / math.tanh(x)


def _clamp_unit_interval(value: float) -> float:
    """Clamp small floating point excursions to the physical [0, 1] range."""
    if -1e-14 < value < 0.0:
        return 0.0
    if 1.0 < value < 1.0 + 1e-14:
        return 1.0
    return value


def rope_n(kI: float, J: float) -> float:
    """Return the relative relaxation parameter n = kI / J.

    ``kI`` and ``J`` must be supplied in the same frequency units. The public
    ROPE API uses Hz for paper-level analytical formulas.
    """
    _validate_nonnegative("kI", kI)
    _validate_positive("J", J)
    return kI / J


def rope_g(n: float) -> float:
    """Return the unconstrained ROPE efficiency g(n) = sqrt(1 + n^2) - n."""
    _validate_nonnegative("n", n)
    return math.sqrt(1.0 + n * n) - n


def inept_efficiency(t: float, J_hz: float, k_hz: float) -> float:
    """Return INEPT transfer efficiency exp(-pi*k*t) * sin(pi*J*t)."""
    _validate_nonnegative("t", t)
    _validate_positive("J_hz", J_hz)
    _validate_nonnegative("k_hz", k_hz)
    return math.exp(-math.pi * k_hz * t) * math.sin(math.pi * J_hz * t)


def inept_optimal_time(n: float, J_hz: float) -> float:
    """Return the INEPT optimum t* = arccot(n) / (pi * J)."""
    _validate_positive("J_hz", J_hz)
    return _arccot_nonnegative(n) / (math.pi * J_hz)


def inept_max_efficiency(n: float, J_hz: float) -> float:
    """Return the INEPT efficiency at t* for n = k / J."""
    _validate_nonnegative("n", n)
    _validate_positive("J_hz", J_hz)
    return inept_efficiency(inept_optimal_time(n, J_hz), J_hz, n * J_hz)


def rope_gain_over_inept(n: float, J_hz: float) -> float:
    """Return the ROPE efficiency gain g(n) / g_INEPT(t*)."""
    max_inept = inept_max_efficiency(n, J_hz)
    if max_inept == 0.0:
        raise ValueError("INEPT maximum efficiency is zero")
    return rope_g(n) / max_inept


def rope_g_inphase(n: float) -> float:
    """Return the equal-rate in-phase ROPE efficiency for Ix -> Sx transfer.

    The JMR paper gives the general in-phase efficiency as the product of the
    I-side and S-side ROPE factors. With the single-parameter API used here,
    kI/J = kS/J = n, so g_in(n) = g(n)^2.
    """
    g = rope_g(n)
    return g * g


def rope_V(r1: float, r2: float, n: float) -> float:
    """Return the ROPE optimal return function V(r1, r2)."""
    g = rope_g(n)
    return math.sqrt(g * g * r1 * r1 + r2 * r2)


def rope_u_ratio(r1: float, r2: float, n: float) -> float:
    """Return the unconstrained optimal control ratio u2/u1 = g*r1/r2."""
    if r1 == 0.0:
        _validate_nonnegative("n", n)
        return 0.0
    if r2 == 0.0:
        _validate_nonnegative("n", n)
        return math.inf
    return rope_g(n) * r1 / r2


def rope_trajectory_invariant(expect_2IySz: float, expect_Ix: float, n: float) -> float:
    """Return <2IySz>/<Ix> for comparison with the optimal invariant g(n)."""
    if expect_2IySz == 0.0:
        _validate_nonnegative("n", n)
        return 0.0
    if expect_Ix == 0.0:
        _validate_nonnegative("n", n)
        return math.inf
    _validate_nonnegative("n", n)
    return expect_2IySz / expect_Ix


def rope_Tcrit(n: float, J_hz: float) -> float:
    """Return the finite-time ROPE critical duration arccot(2n) / (pi*J)."""
    _validate_nonnegative("n", n)
    _validate_positive("J_hz", J_hz)
    return _arccot_nonnegative(2.0 * n) / (math.pi * J_hz)


def rope_j(s: float, n: float) -> float:
    """Return the Appendix B j(s) function for scaled time s = pi*J*t.

    The JMR Appendix B rescales time to remove the ``pi*J`` factor from the
    dynamics. ``rope_switching_time()`` converts physical seconds into this
    scaled variable before evaluating j.
    """
    _validate_nonnegative("s", s)
    _validate_nonnegative("n", n)
    if n == 0.0:
        return 0.0 if s == 0.0 else 1.0

    sqrt_term = math.sqrt(1.0 + n * n)
    value = 1.0 + 2.0 * n * n - 2.0 * n * sqrt_term * _coth(sqrt_term * s + 2.0 * math.asinh(n))
    upper = rope_g(n) ** 2
    if -1e-14 < value < 0.0:
        return 0.0
    if upper < value < upper + 1e-14:
        return upper
    return value


def _rope_finite_angles_from_scaled_switch(s_scaled: float, n: float) -> tuple[float, float]:
    """Return finite-time phase angles for a scaled switching time."""
    if n == 0.0:
        return (math.pi / 4.0, math.pi / 4.0)

    j_value = rope_j(s_scaled, n)
    one_minus_j = 1.0 - j_value
    h1 = math.atan2(2.0 * n * j_value, one_minus_j)
    h2 = math.atan2(one_minus_j, 2.0 * n)
    return (h1, h2)


def rope_switching_time(T: float, n: float, J_hz: float) -> float:
    """Solve Appendix B Eq. (10) for the physical switching time s in seconds."""
    _validate_nonnegative("T", T)
    _validate_nonnegative("n", n)
    _validate_positive("J_hz", J_hz)

    Tcrit = rope_Tcrit(n, J_hz)
    if T <= Tcrit:
        raise ValueError("T must be greater than T_crit for a finite-time ROPE switch")
    if n == 0.0:
        return 0.5 * T

    total_scaled = math.pi * J_hz * T

    def residual(s_scaled: float) -> float:
        h1, h2 = _rope_finite_angles_from_scaled_switch(s_scaled, n)
        return 2.0 * s_scaled + h2 - h1 - total_scaled

    root_scaled = cast(
        float,
        brentq(residual, 0.0, 0.5 * total_scaled, xtol=1e-14, rtol=1e-14, maxiter=100),
    )
    return root_scaled / (math.pi * J_hz)


def rope_finite_angles(T: float, n: float, J_hz: float) -> tuple[float, float]:
    """Return finite-time ROPE phase angles (h1, h2) in radians."""
    switching_time = rope_switching_time(T, n, J_hz)
    return _rope_finite_angles_from_scaled_switch(math.pi * J_hz * switching_time, n)


def rope_finite_efficiency(T: float, n: float, J_hz: float) -> float:
    """Return the optimal ROPE transfer efficiency for a constrained duration T."""
    _validate_nonnegative("T", T)
    _validate_nonnegative("n", n)
    _validate_positive("J_hz", J_hz)

    if T <= rope_Tcrit(n, J_hz):
        return inept_efficiency(T, J_hz, n * J_hz)
    if n == 0.0:
        return 1.0

    h1, h2 = rope_finite_angles(T, n, J_hz)
    denominator = math.sin(h1 + h2)
    if denominator == 0.0:
        raise ValueError("finite-time ROPE angle denominator is zero")
    return math.exp(n * (h1 - h2)) * (1.0 - n * math.sin(2.0 * h2)) / denominator


def _rope_finite_efficiency_from_angles(h1: float, h2: float, n: float) -> float:
    """Return finite-time ROPE efficiency from precomputed phase angles."""
    denominator = math.sin(h1 + h2)
    if denominator == 0.0:
        raise ValueError("finite-time ROPE angle denominator is zero")
    return math.exp(n * (h1 - h2)) * (1.0 - n * math.sin(2.0 * h2)) / denominator


def _rope_phi(t_scaled: float, n: float) -> float:
    """Return the finite-time phase-I phi(t) for scaled time pi*J*t."""
    return 2.0 * math.asinh(n) + 2.0 * t_scaled * math.sqrt(1.0 + n * n)


def rope_phase1_control(t: float, s: float, h1: float, h2: float, n: float) -> tuple[float, float]:
    """Return dimensionless phase-I controls ``(u1, u2)``.

    ``t`` and ``s`` are scaled times, where ``scaled_time = pi * J_hz * seconds``.
    Phase I has ``0 <= t <= s`` and ``u2 = 1``.
    """
    _validate_nonnegative("t", t)
    _validate_nonnegative("s", s)
    _validate_nonnegative("n", n)
    if t > s + 1e-14:
        raise ValueError("t must be within the phase-I interval [0, s]")
    if n == 0.0 or s == 0.0 or t >= s:
        return (1.0, 1.0)

    j_s = rope_j(s, n)
    tan_h2 = math.tan(h2)
    if tan_h2 == 0.0:
        raise ValueError("h2 produces a zero tangent")

    g_T = _rope_finite_efficiency_from_angles(h1, h2, n)
    R1 = g_T / math.sqrt(tan_h2 * tan_h2 + j_s)
    R2 = g_T / math.sqrt(1.0 + j_s / (tan_h2 * tan_h2))
    A = math.sinh(_rope_phi(s / 2.0, n))
    B = math.cosh(_rope_phi(s, n))
    cosh_phi_t = math.cosh(_rope_phi(t, n))

    numerator = R1 * R1 * (1.0 + cosh_phi_t)
    denominator = (B * R1 * R1 + 2.0 * A * A * R2 * R2) - R1 * R1 * cosh_phi_t
    if denominator <= 0.0:
        raise ValueError("phase-I control denominator must be positive")
    u1_squared = _clamp_unit_interval(numerator / denominator)
    if u1_squared < 0.0 or u1_squared > 1.0:
        raise ValueError("phase-I control is outside the physical [0, 1] range")
    return (math.sqrt(u1_squared), 1.0)


def rope_phase3_control(
    t: float, T: float, s: float, h1: float, h2: float, n: float
) -> tuple[float, float]:
    """Return dimensionless phase-III controls using ``u2(t) = u1(T - t)``.

    ``t``, ``T``, and ``s`` use the scaled ``pi * J_hz * seconds`` convention.
    """
    _validate_nonnegative("t", t)
    _validate_nonnegative("T", T)
    _validate_nonnegative("s", s)
    if t > T + 1e-14:
        raise ValueError("t must be within the transfer duration [0, T]")
    if s > 0.5 * T + 1e-14:
        raise ValueError("s must not exceed T/2")
    if n == 0.0 or t <= T - s:
        return (1.0, 1.0)

    mirrored_t = max(0.0, T - t)
    u1_mirrored, _ = rope_phase1_control(mirrored_t, s, h1, h2, n)
    return (1.0, u1_mirrored)


def _rope_rf_amplitude(u: float, t_scaled: float, n: float, J_hz: float) -> float:
    """Return shaped-pulse RF angular amplitude for a phase-I-like control."""
    u = _clamp_unit_interval(u)
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return math.inf

    phi = _rope_phi(t_scaled, n)
    return (
        2.0
        * math.pi
        * J_hz
        * u
        / math.sqrt(1.0 - u * u)
        * math.tanh(0.5 * phi)
        * math.sqrt(1.0 + n * n)
    )


def rope_waveform(
    T: float, n: float, J_hz: float, dt: float
) -> dict[str, npt.NDArray[np.float64]]:
    """Sample the finite-time ROPE controls and RF waveform.

    Returns arrays with keys ``times``, ``u1``, ``u2``, ``amplitude``, and
    ``phase``. Times are in seconds, controls are dimensionless cosines, RF
    amplitudes are angular amplitudes in rad/s, and phases are radians
    (``pi/2`` for phase-I y irradiation, ``0`` for phase-III x irradiation).
    """
    _validate_positive("T", T)
    _validate_nonnegative("n", n)
    _validate_positive("J_hz", J_hz)
    _validate_positive("dt", dt)

    n_steps = math.ceil(T / dt)
    times = np.arange(n_steps, dtype=np.float64) * dt
    u1 = np.ones(n_steps, dtype=np.float64)
    u2 = np.ones(n_steps, dtype=np.float64)
    amplitude = np.zeros(n_steps, dtype=np.float64)
    phase = np.zeros(n_steps, dtype=np.float64)

    if T <= rope_Tcrit(n, J_hz) or n == 0.0:
        return {
            "times": times,
            "u1": u1,
            "u2": u2,
            "amplitude": amplitude,
            "phase": phase,
        }

    switching_time = rope_switching_time(T, n, J_hz)
    h1, h2 = rope_finite_angles(T, n, J_hz)
    scaled_total = math.pi * J_hz * T
    scaled_switch = math.pi * J_hz * switching_time
    phase3_start = T - switching_time

    for idx, t_seconds in enumerate(times):
        t = float(t_seconds)
        t_scaled = math.pi * J_hz * t
        if t < switching_time:
            phase1_u1, phase1_u2 = rope_phase1_control(t_scaled, scaled_switch, h1, h2, n)
            u1[idx] = phase1_u1
            u2[idx] = phase1_u2
            amplitude[idx] = _rope_rf_amplitude(phase1_u1, t_scaled, n, J_hz)
            phase[idx] = math.pi / 2.0
        elif t <= phase3_start:
            u1[idx] = 1.0
            u2[idx] = 1.0
        else:
            phase3_u1, phase3_u2 = rope_phase3_control(
                t_scaled, scaled_total, scaled_switch, h1, h2, n
            )
            mirrored_scaled = max(0.0, scaled_total - t_scaled)
            u1[idx] = phase3_u1
            u2[idx] = phase3_u2
            amplitude[idx] = _rope_rf_amplitude(phase3_u2, mirrored_scaled, n, J_hz)

    return {
        "times": times,
        "u1": u1,
        "u2": u2,
        "amplitude": amplitude,
        "phase": phase,
    }


def rope_hard_pulse_angle(h: float, amplitude: float) -> float:
    """Return a boundary hard-pulse flip angle in degrees.

    ``h`` is the dimensionless boundary control value and ``amplitude`` is the
    dimensionless full-scale control. For the paper example, ``h/amplitude`` is
    ``u1(0)`` and the hard-pulse angle is ``acos(u1(0))``.
    """
    _validate_positive("amplitude", amplitude)
    ratio = _clamp_unit_interval(h / amplitude)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("h / amplitude must be within [0, 1]")
    return math.degrees(math.acos(ratio))
