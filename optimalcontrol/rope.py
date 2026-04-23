"""Analytical ROPE efficiency helpers from the JMR 2003 paper."""

import math
from typing import cast

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
