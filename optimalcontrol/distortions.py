"""Waveform distortion models and local derivative helpers."""

import math

import numpy as np
import numpy.typing as npt

RealArray = npt.NDArray[np.float64]


def _as_waveform(wfm: RealArray) -> RealArray:
    """Return a finite float64 waveform array shaped as time rows by channels."""
    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be two-dimensional, got shape {waveform.shape}")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("waveform entries must be finite")
    return waveform


def _validate_positive_scale(scale: float) -> float:
    """Return a finite positive scale parameter."""
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return float(scale)


def _validate_pole(alpha: float) -> float:
    """Return a finite single-pole filter coefficient."""
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in the range [0, 1]")
    return float(alpha)


def _validate_zero(beta: float) -> float:
    """Return a finite single-zero filter coefficient."""
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if math.isclose(beta, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("beta must not be equal to 1")
    return float(beta)


def _validate_bounds(lb: float, ub: float) -> tuple[float, float]:
    """Return finite clipping bounds."""
    if not math.isfinite(lb) or not math.isfinite(ub):
        raise ValueError("clip bounds must be finite")
    if lb > ub:
        raise ValueError("lower bound must be less than or equal to upper bound")
    return float(lb), float(ub)


def distortion_noop(wfm: RealArray) -> RealArray:
    """Return the waveform unchanged."""
    waveform = _as_waveform(wfm)
    return waveform.copy()


def distortion_noop_deriv(wfm: RealArray) -> RealArray:
    """Return the local derivative of the identity distortion."""
    waveform = _as_waveform(wfm)
    return np.ones_like(waveform, dtype=np.float64)


def distortion_tanh(wfm: RealArray, scale: float) -> RealArray:
    """Return tanh-compressed waveform samples."""
    waveform = _as_waveform(wfm)
    saturation = _validate_positive_scale(scale)
    return np.asarray(saturation * np.tanh(waveform / saturation), dtype=np.float64)


def distortion_tanh_deriv(wfm: RealArray, scale: float) -> RealArray:
    """Return the local derivative of ``distortion_tanh``."""
    waveform = _as_waveform(wfm)
    saturation = _validate_positive_scale(scale)
    scaled = np.tanh(waveform / saturation)
    return np.asarray(1.0 - scaled * scaled, dtype=np.float64)


def distortion_root(wfm: RealArray, scale: float) -> RealArray:
    """Return root-compressed waveform samples."""
    waveform = _as_waveform(wfm)
    compression = _validate_positive_scale(scale)
    amplitude = np.sqrt(np.abs(waveform) / compression)
    return np.asarray(np.sign(waveform) * amplitude * compression, dtype=np.float64)


def distortion_root_deriv(wfm: RealArray, scale: float) -> RealArray:
    """Return the local derivative of ``distortion_root``.

    The analytical derivative is singular at zero; this helper returns ``0.0``
    for zero-valued samples to keep later chain-rule consumers finite.
    """
    waveform = _as_waveform(wfm)
    compression = _validate_positive_scale(scale)
    derivative = np.zeros_like(waveform, dtype=np.float64)
    nonzero = waveform != 0.0
    derivative[nonzero] = 0.5 * np.sqrt(compression / np.abs(waveform[nonzero]))
    return derivative


def distortion_single_pole(wfm: RealArray, alpha: float) -> RealArray:
    """Return waveform samples after a per-channel causal single-pole filter.

    The package waveform convention is time rows by channels, so the filter is
    applied independently down each column using

    ``y[n] = (1 - alpha) * x[n] + alpha * y[n - 1]``

    with an all-zero pre-history.
    """
    waveform = _as_waveform(wfm)
    pole = _validate_pole(alpha)
    distorted = np.zeros_like(waveform, dtype=np.float64)
    if waveform.shape[0] == 0:
        return distorted

    one_minus_pole = 1.0 - pole
    distorted[0, :] = one_minus_pole * waveform[0, :]
    for step in range(1, waveform.shape[0]):
        distorted[step, :] = (
            one_minus_pole * waveform[step, :] + pole * distorted[step - 1, :]
        )
    return distorted


def distortion_single_pole_deriv(wfm: RealArray, alpha: float) -> RealArray:
    """Return diagonal local sensitivities for ``distortion_single_pole``."""
    waveform = _as_waveform(wfm)
    pole = _validate_pole(alpha)
    return np.asarray((1.0 - pole) * np.ones_like(waveform, dtype=np.float64), dtype=np.float64)


def distortion_single_zero(wfm: RealArray, beta: float) -> RealArray:
    """Return waveform samples after a per-channel causal single-zero filter.

    The package waveform convention is time rows by channels, so the filter is
    applied independently down each column using

    ``y[n] = x[n] / (1 - beta) - beta * x[n - 1] / (1 - beta)``

    with an all-zero pre-history.
    """
    waveform = _as_waveform(wfm)
    zero = _validate_zero(beta)
    distorted = np.zeros_like(waveform, dtype=np.float64)
    if waveform.shape[0] == 0:
        return distorted

    gain = 1.0 / (1.0 - zero)
    distorted[0, :] = gain * waveform[0, :]
    for step in range(1, waveform.shape[0]):
        distorted[step, :] = gain * waveform[step, :] - zero * gain * waveform[step - 1, :]
    return distorted


def distortion_single_zero_deriv(wfm: RealArray, beta: float) -> RealArray:
    """Return diagonal local sensitivities for ``distortion_single_zero``."""
    waveform = _as_waveform(wfm)
    zero = _validate_zero(beta)
    gain = 1.0 / (1.0 - zero)
    return np.asarray(gain * np.ones_like(waveform, dtype=np.float64), dtype=np.float64)


def lower_upper_clip(wfm: RealArray, lb: float, ub: float) -> RealArray:
    """Return a clipped waveform for diagnostics outside the gradient path."""
    waveform = _as_waveform(wfm)
    lower, upper = _validate_bounds(lb, ub)
    return np.asarray(np.clip(waveform, lower, upper), dtype=np.float64)
