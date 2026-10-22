"""Fast Bloch-vector propagation over offset and RF-power ensembles."""

import math

import numpy as np

from optimalcontrol._accelerator import _enabled, _rust
from optimalcontrol._types import RealArray


def _rotate(state: RealArray, field: RealArray, dt: float) -> RealArray:
    """Apply one Rodrigues rotation in the NumPy fallback path."""
    m = np.max(np.abs(field), axis=-1, keepdims=True)
    safe_m = np.where(m == 0.0, 1.0, m)
    q = field / safe_m
    q0, q1, q2 = q[..., 0], q[..., 1], q[..., 2]
    norm = m * np.sqrt((q0 * q0 + q1 * q1) + q2 * q2)[..., None]
    safe_norm = np.where(norm == 0.0, 1.0, norm)
    axis = field / safe_norm
    angle = ((2.0 * math.pi) * norm) * dt
    sn = np.sin(angle)
    cs = np.cos(angle)
    a0, a1, a2 = axis[..., 0], axis[..., 1], axis[..., 2]
    s0, s1, s2 = state[..., 0], state[..., 1], state[..., 2]
    dot = ((a0 * s0 + a1 * s1) + a2 * s2)[..., None]
    cross = np.stack((a1 * s2 - a2 * s1, a2 * s0 - a0 * s2, a0 * s1 - a1 * s0), axis=-1)
    rotated = (cs * state + sn * cross) + (((1.0 - cs) * axis) * dot)
    return np.where(norm == 0.0, state, rotated)


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
    for name, array in (
        ("initial", initial_array),
        ("waveform_xy", waveform),
        ("offsets_hz", offsets),
        ("b1_scales", scales),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
    if not math.isfinite(rf_hz) or rf_hz < 0.0:
        raise ValueError("rf_hz must be finite and non-negative")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    if _enabled() and _rust is not None:
        return np.asarray(
            _rust.bloch_ensemble(initial_array, waveform, offsets, scales, float(rf_hz), float(dt)),
            dtype=np.float64,
        )

    state = np.broadcast_to(initial_array, (scales.size, offsets.size, 3))
    field = np.empty_like(state)
    field[..., 2] = offsets
    amp = scales[:, None] * rf_hz
    for ux, uy in waveform:
        field[..., 0] = amp * ux
        field[..., 1] = amp * uy
        state = _rotate(state, field, dt)
    return np.ascontiguousarray(state)
