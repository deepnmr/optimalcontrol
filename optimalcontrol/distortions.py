"""Waveform distortion models and local derivative helpers."""

import math

import numpy as np

from optimalcontrol._types import RealArray
from optimalcontrol._validation import as_finite_waveform as _as_waveform
from optimalcontrol._validation import validate_positive


def _validate_positive_scale(scale: float) -> float:
    """Return a finite positive scale parameter."""
    validate_positive("scale", scale)
    return float(scale)


def _validate_pole(alpha: float) -> float:
    """Return a finite single-pole filter coefficient."""
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in the range [0, 1]")
    return float(alpha)


def distortion_noop(wfm: RealArray) -> RealArray:
    """Return the waveform unchanged."""
    waveform = _as_waveform(wfm)
    return waveform.copy()


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


