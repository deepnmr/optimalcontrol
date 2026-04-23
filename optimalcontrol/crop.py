"""Analytical CROP efficiency helpers from the PNAS 2003 paper."""

import math


def _validate_nonnegative(name: str, value: float) -> None:
    """Raise ValueError if a scalar parameter is outside its physical domain."""
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_positive(name: str, value: float) -> None:
    """Raise ValueError if a scalar parameter is not strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def crop_zeta(ka: float, kc: float, J_hz: float) -> float:
    """Return the CROP ratio zeta = sqrt((ka^2-kc^2)/(J^2+kc^2)).

    The analytical CROP formulas use dimensionless ratios, so ``ka``, ``kc``,
    and ``J_hz`` must be supplied in the same frequency units.
    """
    _validate_nonnegative("ka", ka)
    _validate_nonnegative("kc", kc)
    _validate_positive("J_hz", J_hz)
    if ka < kc:
        raise ValueError("ka must be greater than or equal to kc")

    return math.sqrt((ka * ka - kc * kc) / (J_hz * J_hz + kc * kc))


def crop_eta(ka: float, kc: float, J_hz: float) -> float:
    """Return the CROP Iz -> 2IzSz transfer efficiency eta."""
    zeta = crop_zeta(ka, kc, J_hz)
    return math.sqrt(1.0 + zeta * zeta) - zeta


def crop_eta_prime(ka_prime: float, kc_prime: float, J_hz: float) -> float:
    """Return the CROP 2IzSz -> Sz transfer efficiency eta prime."""
    return crop_eta(ka_prime, kc_prime, J_hz)


def crop_limit_Iz_to_2IzSz(eta: float) -> float:
    """Return the physical transfer limit for Iz -> 2IzSz."""
    _validate_nonnegative("eta", eta)
    return eta


def crop_limit_2IzSz_to_Sz(eta_prime: float) -> float:
    """Return the physical transfer limit for 2IzSz -> Sz."""
    _validate_nonnegative("eta_prime", eta_prime)
    return eta_prime


def crop_limit_Iz_to_Sz(eta: float, eta_prime: float) -> float:
    """Return the two-step physical transfer limit for Iz -> Sz."""
    _validate_nonnegative("eta", eta)
    _validate_nonnegative("eta_prime", eta_prime)
    return eta * eta_prime


def crop_limit_single_transition(eta: float, eta_prime: float) -> float:
    """Return the physical transfer limit for single-transition transfer."""
    _validate_nonnegative("eta", eta)
    _validate_nonnegative("eta_prime", eta_prime)
    return math.sqrt(eta * eta + eta_prime * eta_prime)
