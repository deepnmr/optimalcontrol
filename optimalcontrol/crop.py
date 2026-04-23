"""Analytical CROP efficiency helpers from the PNAS 2003 paper."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optimalcontrol.rope import rope_g, rope_n
from optimalcontrol.spin_system import SpinSystem


@dataclass
class CROPPulse:
    """Scalar parameters used to seed a truncated CROP pulse waveform.

    Amplitude and irradiation frequency use the paper-level Hz convention.
    """

    amplitude: float
    irradiation_freq_hz: float
    truncation_window: float


def _validate_nonnegative(name: str, value: float) -> None:
    """Raise ValueError if a scalar parameter is outside its physical domain."""
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_positive(name: str, value: float) -> None:
    """Raise ValueError if a scalar parameter is not strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _validate_finite_positive(name: str, value: float) -> None:
    """Raise ValueError if a scalar parameter is not finite and positive."""
    _validate_positive(name, value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _as_float_array(values: Iterable[float], name: str) -> npt.NDArray[np.float64]:
    """Convert a one-dimensional iterable of floats to a float64 array."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


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


def crop_kc0_limit(ka: float, J_hz: float) -> float:
    """Return the ``kc = 0`` CROP efficiency and verify the ROPE reduction.

    In the absence of cross-correlated relaxation, the PNAS CROP expression
    reduces to the JMR ROPE efficiency ``g(n=ka/J)``.
    """
    eta = crop_eta(ka, 0.0, J_hz)
    expected = rope_g(rope_n(ka, J_hz))
    if abs(eta - expected) > 1e-10:
        raise ValueError("CROP kc=0 limit does not match ROPE efficiency")
    return eta


def crop_lossless_limit(ka: float, J_hz: float) -> float:
    """Return the decoherence-free ``kc = ka`` CROP limiting efficiency."""
    return crop_eta(ka, ka, J_hz)


def single_transition_decomposition(sys: SpinSystem) -> dict[str, str]:
    """Identify slow and fast I-spin single-transition components.

    For non-negative CROP rates, ``IzSbeta`` has transverse relaxation
    ``ka - kc`` and ``IzSalpha`` has ``ka + kc``. When ``kc = 0`` the rates are
    tied; the beta component is returned as the stable convention.
    """
    if len(sys.spins) != 2:
        raise ValueError("single_transition_decomposition requires a two-spin system")

    ka = sys.relaxation.ka
    kc = sys.relaxation.kc
    _validate_nonnegative("ka", ka)
    _validate_nonnegative("kc", kc)
    if ka < kc:
        raise ValueError("ka must be greater than or equal to kc")

    alpha_rate = ka + kc
    beta_rate = ka - kc
    if beta_rate <= alpha_rate:
        return {"slowly_relaxing": "IzSbeta", "fast_relaxing": "IzSalpha"}
    return {"slowly_relaxing": "IzSalpha", "fast_relaxing": "IzSbeta"}


def crop_pulse_params(ka: float, kc: float, J_hz: float) -> CROPPulse:
    """Return scalar CROP pulse parameters in Hz units.

    The scalar amplitude is the residual relaxation bandwidth
    ``sqrt(ka**2 - kc**2)``. It vanishes in the decoherence-free limit
    ``kc -> ka``, matching the paper's weak selective irradiation limit. The
    carrier is centered on the slowly relaxing beta multiplet at ``-J/2``.
    """
    _ = crop_eta(ka, kc, J_hz)
    residual_bandwidth = math.sqrt(max(0.0, ka * ka - kc * kc))
    if residual_bandwidth == 0.0:
        truncation_window = math.inf
    else:
        truncation_window = 1.0 / residual_bandwidth
    return CROPPulse(
        amplitude=residual_bandwidth,
        irradiation_freq_hz=-0.5 * J_hz,
        truncation_window=truncation_window,
    )


def crop_waveform(
    ka: float, kc: float, J_hz: float, dt: float, truncation_window: float
) -> dict[str, npt.NDArray[np.float64]]:
    """Sample a symmetrically truncated CROP pulse waveform.

    Times are physical seconds centered on zero. Amplitude and irradiation
    frequency use the same Hz convention as ``crop_pulse_params()`` and the
    PNAS figure labels.
    """
    _validate_finite_positive("dt", dt)
    _validate_finite_positive("truncation_window", truncation_window)
    params = crop_pulse_params(ka, kc, J_hz)

    n_steps = max(1, math.ceil(truncation_window / dt))
    time_offsets = np.arange(n_steps, dtype=np.float64) - 0.5 * (n_steps - 1)
    times = time_offsets * dt
    amplitude = np.full(n_steps, params.amplitude, dtype=np.float64)
    irrad_freq = np.full(n_steps, params.irradiation_freq_hz, dtype=np.float64)

    return {
        "times": times,
        "amplitude": amplitude,
        "irrad_freq": irrad_freq,
    }


def crop_robustness_sweep(
    ka_over_J_values: Iterable[float], kc_over_ka_values: Iterable[float], J_hz: float
) -> npt.NDArray[np.float64]:
    """Return CROP efficiency over a ``ka/J`` by ``kc/ka`` parameter grid."""
    _validate_positive("J_hz", J_hz)
    ka_ratios = _as_float_array(ka_over_J_values, "ka_over_J_values")
    kc_ratios = _as_float_array(kc_over_ka_values, "kc_over_ka_values")
    sweep = np.empty((len(ka_ratios), len(kc_ratios)), dtype=np.float64)

    for row, ka_ratio in enumerate(ka_ratios):
        _validate_nonnegative("ka_over_J", float(ka_ratio))
        ka = float(ka_ratio) * J_hz
        for col, kc_ratio in enumerate(kc_ratios):
            ratio = float(kc_ratio)
            _validate_nonnegative("kc_over_ka", ratio)
            if ratio > 1.0:
                raise ValueError("kc_over_ka must be less than or equal to 1")

            eta = crop_eta(ka, ratio * ka, J_hz)
            if eta < -1e-10 or eta > 1.0 + 1e-10:
                raise AssertionError("CROP efficiency is outside the physical [0, 1] range")
            sweep[row, col] = min(1.0, max(0.0, eta))

    return sweep
