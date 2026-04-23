"""Analytical ROPE efficiency helpers from the JMR 2003 paper."""

import math


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
