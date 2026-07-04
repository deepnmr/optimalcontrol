"""Fast Bloch-vector propagation over offset and RF-power ensembles."""

import math

import numpy as np

from optimalcontrol._accelerator import _enabled, _rust
from optimalcontrol._types import RealArray


def _rotate(state: RealArray, field: RealArray, dt: float) -> RealArray:
    """Apply one Rodrigues rotation in the NumPy fallback path."""
    norm = float(np.linalg.norm(field))
    if norm == 0.0:
        return state
    axis = field / norm
    angle = 2.0 * math.pi * norm * dt
    return np.asarray(
        math.cos(angle) * state
        + math.sin(angle) * np.cross(axis, state)
        + (1.0 - math.cos(angle)) * axis * np.dot(axis, state),
        dtype=np.float64,
    )


def propagate_bloch_ensemble(
    initial: RealArray,
    waveform_xy: RealArray,
    offsets_hz: RealArray,
    b1_scales: RealArray,
    rf_hz: float,
    dt: float,
) -> RealArray:
    """Return final Bloch vectors with shape ``(n_b1, n_offsets, 3)``."""
    initial_array = np.ascontiguousarray(initial, dtype=np.float64)
    waveform = np.ascontiguousarray(waveform_xy, dtype=np.float64)
    offsets = np.ascontiguousarray(offsets_hz, dtype=np.float64)
    scales = np.ascontiguousarray(b1_scales, dtype=np.float64)
    if initial_array.shape != (3,):
        raise ValueError(f"initial must have shape (3,), got {initial_array.shape}")
    if waveform.ndim != 2 or waveform.shape[1] != 2 or waveform.shape[0] == 0:
        raise ValueError(f"waveform_xy must have shape (n_steps, 2), got {waveform.shape}")
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("offsets_hz must be a non-empty one-dimensional array")
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError("b1_scales must be a non-empty one-dimensional array")
    if not math.isfinite(rf_hz) or rf_hz < 0.0:
        raise ValueError("rf_hz must be finite and non-negative")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    if _enabled() and _rust is not None:
        return np.asarray(
            _rust.bloch_ensemble(initial_array, waveform, offsets, scales, float(rf_hz), float(dt)),
            dtype=np.float64,
        )

    result = np.empty((scales.size, offsets.size, 3), dtype=np.float64)
    for row, scale in enumerate(scales):
        for col, offset in enumerate(offsets):
            state = initial_array.copy()
            for ux, uy in waveform:
                field = np.array([scale * rf_hz * ux, scale * rf_hz * uy, offset], dtype=np.float64)
                state = _rotate(state, field, dt)
            result[row, col] = state
    return result
