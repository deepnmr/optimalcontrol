"""Shared validation helpers for control-problem and ensemble containers.

These helpers are used by both :mod:`optimalcontrol.grape` and
:mod:`optimalcontrol.ensemble` so that the input-checking contract stays
identical across the two modules. The error messages are part of the tested
behaviour; keep them stable.
"""

import math
from collections.abc import Sequence

import numpy as np

from optimalcontrol._types import Array, RealArray


def validate_nonempty(name: str, values: Sequence[object]) -> None:
    """Raise ValueError if a required list field is empty."""
    if not values:
        raise ValueError(f"{name} must be non-empty")


def validate_nonnegative(name: str, value: float) -> None:
    """Raise ValueError if a scalar parameter is outside its physical domain."""
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def validate_positive(name: str, value: float) -> None:
    """Raise ValueError if a scalar parameter is not strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def as_finite_waveform(wfm: RealArray) -> RealArray:
    """Return a finite float64 waveform array shaped as time rows by channels."""
    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be two-dimensional, got shape {waveform.shape}")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("waveform entries must be finite")
    return waveform


def validate_square_matrix(name: str, matrix: Array) -> int:
    """Validate a finite square 2-D complex array and return its dimension."""
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} entries must be finite")
    return int(array.shape[0])


def validate_finite_floats(name: str, values: Sequence[float]) -> None:
    """Raise ValueError if a sequence of float-like values has non-finite entries."""
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{name}[{index}] must be finite")
