"""Scaled-unitary spin-1/2 fast kernel for Seedless (ocseed) pulse design.

Implements the analytic 2x2 propagator and phase-only gradient of Buchanan et
al., *Nat. Commun.* 16, 7276 (2025), Supplementary Note 2.9/2.11/2.12. For a
single isolated spin-1/2 the propagator of one constant pulse element is a
scaled-unitary 2x2 matrix with a closed form, and the phase derivative is
``G = i[V, Iz]`` (Note 2.11). This replaces the general engine's 4x4 Liouville
eigendecomposition per slice and evaluates value+gradient over the whole
(offset x B1) ensemble in one pass -- the source of Seedless's speed.

The value/gradient primitives run in a native Rust kernel when the extension is
available (parallel over ensemble members); an exact NumPy path is the fallback.
Both agree with the general GRAPE engine to machine precision. Fidelity is
normalised so a perfect transfer scores 1 (== target . final Bloch vector).

Ensemble members are ordered offset-major: member ``m`` has ``offset[m // n_b1]``
and ``b1_scale[m % n_b1]``.
"""

from __future__ import annotations

import math

import numpy as np

from optimalcontrol._accelerator import _enabled, _rust
from optimalcontrol._types import Array, RealArray


def _check_finite(
    waveform: RealArray, offsets: RealArray, scales: RealArray, rf_hz: float, dt: float
) -> None:
    """Reject non-finite inputs before dispatch (matches ``propagate_bloch_ensemble``).

    Guards both the Rust and NumPy paths in one place; without it the native
    kernel silently returns NaN instead of raising, unlike the rest of the library.
    """
    for name, array in (("waveform", waveform), ("offsets", offsets), ("b1_scales", scales)):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
    if not math.isfinite(rf_hz) or rf_hz < 0.0:
        raise ValueError("rf_hz must be finite and non-negative")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")


def bloch_operator(vector: RealArray) -> Array:
    """Return the 2x2 deviation density ``a*Ix + b*Iy + c*Iz`` for Bloch ``[a,b,c]``."""
    a, b, c = (float(x) for x in vector)
    return np.array(
        [[0.5 * c, 0.5 * (a - 1j * b)], [0.5 * (a + 1j * b), -0.5 * c]],
        dtype=np.complex128,
    )


def s2s_value_grad(
    wfm_xy: RealArray,
    offsets_hz: RealArray,
    b1_scales: RealArray,
    rf_hz: float,
    dt: float,
    init_bloch: RealArray,
    targ_bloch: RealArray,
) -> tuple[RealArray, RealArray]:
    """Return per-member fidelity ``(K,)`` and phase gradient ``(K, n)`` for one S2S pair."""
    waveform = np.ascontiguousarray(wfm_xy, dtype=np.float64)
    offsets = np.ascontiguousarray(offsets_hz, dtype=np.float64)
    scales = np.ascontiguousarray(b1_scales, dtype=np.float64)
    init = np.ascontiguousarray(init_bloch, dtype=np.float64)
    targ = np.ascontiguousarray(targ_bloch, dtype=np.float64)
    n_steps = waveform.shape[0]
    _check_finite(waveform, offsets, scales, rf_hz, dt)

    if _enabled() and _rust is not None:
        fidelity, flat = _rust.seedless_pair_value_gradient(
            waveform, offsets, scales, float(rf_hz), float(dt), init, targ
        )
        grad = np.asarray(flat, dtype=np.float64).reshape(-1, n_steps)
        return np.asarray(fidelity, dtype=np.float64), grad

    prop_v, prop_g = _np_propagators(waveform, offsets, scales, rf_hz, dt)
    return _np_pair_value_grad(prop_v, prop_g, bloch_operator(init), bloch_operator(targ))


def suppress_perstep_value_grad(
    wfm_xy: RealArray,
    offsets_hz: RealArray,
    b1_scales: RealArray,
    rf_hz: float,
    dt: float,
) -> tuple[RealArray, RealArray]:
    """Return per-member per-step hold cost ``(K,)`` and gradient ``(K, n)``.

    Implements the Supplementary Note 2.7 "suppression" restraint: hold ``Iz`` on
    ``Iz`` after *every* prefix of the pulse. The cost is the mean over steps of
    ``1 - mz_k`` (residual z after ``k`` propagators). Its gradient scales as
    ``n^2/2`` because a change at step ``j`` affects every later hold.
    """
    waveform = np.ascontiguousarray(wfm_xy, dtype=np.float64)
    offsets = np.ascontiguousarray(offsets_hz, dtype=np.float64)
    scales = np.ascontiguousarray(b1_scales, dtype=np.float64)
    n_steps = waveform.shape[0]
    _check_finite(waveform, offsets, scales, rf_hz, dt)

    if _enabled() and _rust is not None:
        cost, flat = _rust.seedless_suppress_perstep(
            waveform, offsets, scales, float(rf_hz), float(dt)
        )
        grad = np.asarray(flat, dtype=np.float64).reshape(-1, n_steps)
        return np.asarray(cost, dtype=np.float64), grad

    prop_v, prop_g = _np_propagators(waveform, offsets, scales, rf_hz, dt)
    return _np_suppress_perstep(prop_v, prop_g, bloch_operator(np.array([0.0, 0.0, 1.0])))


# --- NumPy fallback (exact, vectorised over the ensemble) -------------------


def _np_propagators(
    wfm_xy: RealArray,
    offsets_hz: RealArray,
    b1_scales: RealArray,
    rf_hz: float,
    dt: float,
) -> tuple[Array, Array]:
    """Return per-member per-step propagators ``V`` and phase derivatives ``G``.

    Both have shape ``(K, n, 2, 2)`` with ``K = n_offsets * n_b1`` (offset-major).
    """
    ux = wfm_xy[:, 0]
    uy = wfm_xy[:, 1]
    n = wfm_xy.shape[0]
    off_grid, scale_grid = np.meshgrid(offsets_hz, b1_scales, indexing="ij")
    off_flat = off_grid.ravel()
    scale_flat = scale_grid.ravel()

    fx = scale_flat[:, None] * rf_hz * ux[None, :]
    fy = scale_flat[:, None] * rf_hz * uy[None, :]
    fz = off_flat[:, None] * np.ones(n)[None, :]
    fnorm = np.sqrt(fx * fx + fy * fy + fz * fz)
    inv = np.where(fnorm > 0.0, 1.0 / np.where(fnorm > 0.0, fnorm, 1.0), 0.0)
    half = math.pi * fnorm * dt
    cos_h = np.cos(half)
    sin_h = np.sin(half)
    nx = fx * inv
    ny = fy * inv
    nz = fz * inv

    k_members = fnorm.shape[0]
    v = np.empty((k_members, n, 2, 2), dtype=np.complex128)
    v[..., 0, 0] = cos_h - 1j * sin_h * nz
    v[..., 0, 1] = -sin_h * ny - 1j * sin_h * nx
    v[..., 1, 0] = sin_h * ny - 1j * sin_h * nx
    v[..., 1, 1] = cos_h + 1j * sin_h * nz

    g = np.zeros_like(v)
    g[..., 0, 1] = -1j * v[..., 0, 1]
    g[..., 1, 0] = 1j * v[..., 1, 0]
    return v, g


def _forward_states(v: Array, rho_init: Array) -> tuple[Array, Array]:
    """Return stacked pre-step states ``(n, K, 2, 2)`` and the final state ``(K, 2, 2)``."""
    n = v.shape[1]
    rho = np.broadcast_to(rho_init, (v.shape[0], 2, 2)).copy()
    fwd = np.empty((n, v.shape[0], 2, 2), dtype=np.complex128)
    for k in range(n):
        fwd[k] = rho
        vk = v[:, k]
        rho = vk @ rho @ np.conj(np.swapaxes(vk, -1, -2))
    return fwd, rho


def _trace(a: Array, b: Array) -> Array:
    """Return the batched trace ``Tr(a @ b)`` over a leading member axis."""
    return np.asarray(np.einsum("kij,kji->k", a, b), dtype=np.complex128)


def _np_pair_value_grad(
    v: Array, g: Array, rho_init: Array, rho_targ: Array
) -> tuple[RealArray, RealArray]:
    """NumPy per-member fidelity and phase gradient for one S2S pair."""
    fwd, rho_final = _forward_states(v, rho_init)
    fidelity = 2.0 * np.real(_trace(np.broadcast_to(rho_targ, rho_final.shape), rho_final))

    n = v.shape[1]
    grad = np.empty((v.shape[0], n), dtype=np.float64)
    lam = np.broadcast_to(rho_targ, (v.shape[0], 2, 2)).copy()
    for j in range(n - 1, -1, -1):
        vj = v[:, j]
        gj = g[:, j]
        rho_before = fwd[j]
        vj_h = np.conj(np.swapaxes(vj, -1, -2))
        gj_h = np.conj(np.swapaxes(gj, -1, -2))
        d_rho = gj @ rho_before @ vj_h + vj @ rho_before @ gj_h
        grad[:, j] = 2.0 * np.real(_trace(lam, d_rho))
        lam = vj_h @ lam @ vj
    return np.asarray(fidelity, dtype=np.float64), grad


def _np_suppress_perstep(v: Array, g: Array, rho_iz: Array) -> tuple[RealArray, RealArray]:
    """NumPy per-member per-step suppression cost and gradient (``n^2/2`` scaling)."""
    n = v.shape[1]
    k_members = v.shape[0]
    fwd, _ = _forward_states(v, rho_iz)

    cost = np.zeros(k_members, dtype=np.float64)
    grad = np.zeros((k_members, n), dtype=np.float64)
    rho_targ = np.broadcast_to(rho_iz, (k_members, 2, 2))
    rho_after = np.broadcast_to(rho_iz, (k_members, 2, 2)).copy()
    for prefix in range(1, n + 1):
        v_last = v[:, prefix - 1]
        rho_after = v_last @ rho_after @ np.conj(np.swapaxes(v_last, -1, -2))
        mz = 2.0 * np.real(_trace(rho_targ, rho_after))
        cost += (1.0 - mz) / n
        lam = np.broadcast_to(rho_iz, (k_members, 2, 2)).copy()
        for j in range(prefix - 1, -1, -1):
            vj = v[:, j]
            gj = g[:, j]
            rho_before = fwd[j]
            vj_h = np.conj(np.swapaxes(vj, -1, -2))
            gj_h = np.conj(np.swapaxes(gj, -1, -2))
            d_rho = gj @ rho_before @ vj_h + vj @ rho_before @ gj_h
            grad[:, j] += -(2.0 * np.real(_trace(lam, d_rho))) / n
            lam = vj_h @ lam @ vj
    return cost, grad
