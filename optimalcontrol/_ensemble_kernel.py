"""Batched spin-half commutator propagation using two-level sandwich sweeps.

The recovered Hamiltonian block drops an identity term. Its global phase
cancels in each density-matrix sandwich, including the control derivatives.
"""

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np

from optimalcontrol._accelerator import _is_anti_hermitian, _metadata_supported
from optimalcontrol._problem import ControlProblem, validate_control_problem, validate_waveform
from optimalcontrol._propagation import _control_direction_stack
from optimalcontrol._types import Array, RealArray
from optimalcontrol.ensemble import cartesian_product_ensemble
from optimalcontrol.grape import _problem_for_basis
from optimalcontrol.operators import L_op, R_op

_WORKSPACE_BUDGET_BYTES = 64 << 20
_CHUNK_OVERRIDE: int | None = None


@dataclass(frozen=True, slots=True)
class _SpinHalfInputs:
    drifts: Array
    controls: Array
    waveform: RealArray
    rho_init: Array
    rho_targ: Array
    n_members: int
    n_steps: int
    n_channels: int
    n_pairs: int
    dt: float
    mode: str


def _commutator_supported(generator: Array) -> bool:
    if not _is_anti_hermitian(generator):
        return False
    hermitian = 1j * generator
    block = hermitian[:2, :2]
    residual = float(np.max(np.abs(L_op(block) - R_op(block) - hermitian)))
    return residual <= 1e-12 * max(1.0, float(np.max(np.abs(generator))))


def spin_half_inputs(problem: ControlProblem, waveform: RealArray) -> _SpinHalfInputs | None:
    """Validate the complete ensemble before stacking its ordered members."""
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim != 2:
        return None
    try:
        validate_control_problem(problem)
        validate_waveform(waveform, len(problem.operators), waveform.shape[0])
    except ValueError:
        return None
    if not _metadata_supported(problem, waveform):
        return None
    if any(
        np.asarray(g).shape != (4, 4)
        for g in problem.drifts + problem.operators + (problem.offset_operators or [])
    ):
        return None
    if (problem.offsets is None) != (problem.offset_operators is None):
        return None
    if problem.offsets is not None and (not problem.offsets or not problem.offset_operators):
        return None
    problem = _problem_for_basis(problem)
    if any(np.asarray(s).shape != (4,) for s in problem.rho_init + problem.rho_targ):
        return None
    try:
        members = cartesian_product_ensemble(problem)
    except ValueError:
        return None
    scales = np.maximum(1.0, np.max(np.abs(waveform), axis=0))
    # Check every expanded generator before allocating the stacked workspace.
    for member in members:
        if not _commutator_supported(member.drifts[0]):
            return None
        for channel, direction in enumerate(_control_direction_stack(member)):
            if not _commutator_supported(direction * scales[channel]):
                return None
    return _SpinHalfInputs(
        drifts=np.stack([m.drifts[0] for m in members]).astype(np.complex128),
        controls=np.stack([_control_direction_stack(m) for m in members]),
        waveform=waveform,
        rho_init=np.stack([m.rho_init for m in members]).astype(np.complex128),
        rho_targ=np.stack([m.rho_targ for m in members]).astype(np.complex128),
        n_members=len(members),
        n_steps=waveform.shape[0],
        n_channels=waveform.shape[1],
        n_pairs=len(problem.rho_init),
        dt=problem.pulse_dt,
        mode=problem.fidelity_mode,
    )


def _chunk_members(
    n_steps: int,
    n_channels: int,
    n_pairs: int,
    max_members: int,
    budget_bytes: int = _WORKSPACE_BUDGET_BYTES,
) -> int:
    bytes_per_member = 16 * (
        4 * n_steps * (1 + n_channels)
        + 8 * (n_steps + 1) * n_pairs
        + 4 * n_steps * n_channels * n_pairs
        + 8 * n_pairs
    )
    return max(1, min(max_members, budget_bytes // bytes_per_member))


@overload
def spin_half_value_gradient(
    inputs: _SpinHalfInputs, need_gradient: Literal[True]
) -> tuple[float, RealArray]: ...


@overload
def spin_half_value_gradient(
    inputs: _SpinHalfInputs, need_gradient: Literal[False]
) -> tuple[float, None]: ...


def spin_half_value_gradient(
    inputs: _SpinHalfInputs, need_gradient: bool
) -> tuple[float, RealArray | None]:
    """Evaluate bare mean fidelity and optionally its waveform gradient."""
    n, c, p = inputs.n_steps, inputs.n_channels, inputs.n_pairs
    chunk = _chunk_members(n, c, p, inputs.n_members, _WORKSPACE_BUDGET_BYTES)
    if _CHUNK_OVERRIDE is not None:
        chunk = _CHUNK_OVERRIDE
    value_sum = 0.0
    gradient_sum = np.zeros((n, c), dtype=np.float64)
    for start in range(0, inputs.n_members, chunk):
        stop = min(start + chunk, inputs.n_members)
        m = stop - start
        h0_block = (1j * inputs.drifts[start:stop])[..., :2, :2]
        hc = (1j * inputs.controls[start:stop])[..., :2, :2]
        h = h0_block[:, None] + np.einsum("nc,mcij->mnij", inputs.waveform, hc)
        h0 = 0.5 * (h[..., 0, 0] + h[..., 1, 1]).real
        hz = 0.5 * (h[..., 0, 0] - h[..., 1, 1]).real
        hx, hy = h[..., 0, 1].real, -h[..., 0, 1].imag
        # The radius is the eigenvalue splitting and must stay finite at any scale:
        # hypot never squares the components (squaring underflows to an apparent
        # degeneracy below ~1e-154 and overflows above ~1e154). The unit direction is
        # taken from the components rescaled by their largest magnitude so the division
        # happens on O(1) numbers even when the raw components are subnormal.
        r = np.hypot(np.hypot(hx, hy), hz)
        zero = r == 0.0
        scale = np.where(zero, 1.0, np.maximum(np.maximum(np.abs(hx), np.abs(hy)), np.abs(hz)))
        sx, sy, sz = hx / scale, hy / scale, hz / scale
        unit = np.where(zero, 1.0, np.hypot(np.hypot(sx, sy), sz))
        ux, uy, uz = sx / unit, sy / unit, sz / unit
        positive = uz >= 0.0
        rp = np.where(positive, 1.0 + uz, 1.0 - uz)
        norm = np.sqrt(2.0 * rp)
        wp0 = np.where(positive, rp, ux - 1j * uy) / norm
        wp1 = np.where(positive, ux + 1j * uy, rp) / norm
        wm0 = np.where(positive, -(ux - 1j * uy), rp) / norm
        wm1 = np.where(positive, rp, -(ux + 1j * uy)) / norm
        # The degenerate case uses the identity basis.
        wm0, wm1 = np.where(zero, 1.0, wm0), np.where(zero, 0.0, wm1)
        wp0, wp1 = np.where(zero, 0.0, wp0), np.where(zero, 1.0, wp1)
        vectors = np.stack([np.stack([wm0, wp0], -1), np.stack([wm1, wp1], -1)], -2)
        energies = np.stack([h0 - r, h0 + r], -1)
        adjoint = vectors.conj().swapaxes(-1, -2)
        u = (vectors * np.exp(-1j * inputs.dt * energies)[..., None, :]) @ adjoint
        ut = np.ascontiguousarray(u.transpose(1, 0, 2, 3))
        uth = np.ascontiguousarray(ut.conj().swapaxes(-1, -2))
        initial = inputs.rho_init[start:stop].reshape(m, p, 2, 2).swapaxes(-1, -2)
        target = inputs.rho_targ[start:stop].reshape(m, p, 2, 2).swapaxes(-1, -2)
        forward = np.empty((n + 1, m, p, 2, 2), dtype=np.complex128)
        forward[0] = initial
        for k in range(n):
            forward[k + 1] = ut[k, :, None] @ forward[k] @ uth[k, :, None]
        overlap = np.sum(target.conj() * forward[-1], axis=(-1, -2))
        match inputs.mode:
            case "real":
                values = overlap.real
            case "imag":
                values = overlap.imag
            case _:
                values = np.abs(overlap) ** 2
        member_values = np.mean(values, axis=1)
        value_sum = float(np.add.accumulate(np.concatenate(([value_sum], member_values)))[-1])
        if not need_gradient:
            continue
        theta = inputs.dt * energies
        half_gap = 0.5 * (theta[..., :, None] - theta[..., None, :])
        safe_gap = np.where(half_gap == 0.0, 1.0, half_gap)
        sinc = np.where(half_gap == 0.0, 1.0, np.sin(safe_gap) / safe_gap)
        midpoint = 0.5 * (theta[..., :, None] + theta[..., None, :])
        divided = (-1j * inputs.dt) * np.exp(-1j * midpoint) * sinc
        # hc is already Hermitian: no additional factor of 1j here.
        rotated = adjoint[:, :, None] @ hc[:, None] @ vectors[:, :, None]
        du = vectors[:, :, None] @ (divided[:, :, None] * rotated) @ adjoint[:, :, None]
        backward = np.empty_like(forward)
        backward[n] = target
        for k in range(n - 1, -1, -1):
            backward[k] = uth[k, :, None] @ backward[k + 1] @ ut[k, :, None]
        f = forward[:-1].transpose(1, 2, 0, 3, 4)[:, :, :, None]
        uu = u[:, None, :, None]
        dd = du[:, None]
        df = dd @ f @ uu.conj().swapaxes(-1, -2) + uu @ f @ dd.conj().swapaxes(-1, -2)
        b = backward[1:].transpose(1, 2, 0, 3, 4)[:, :, :, None]
        d_overlap = np.sum(b.conj() * df, axis=(-1, -2))
        match inputs.mode:
            case "real":
                gradients = d_overlap.real
            case "imag":
                gradients = d_overlap.imag
            case _:
                gradients = 2.0 * np.real(overlap.conj()[:, :, None, None] * d_overlap)
        member_gradients = np.add.accumulate(gradients, axis=1)[:, -1] / p
        gradient_sum = np.add.accumulate(
            np.concatenate((gradient_sum[None], member_gradients), axis=0), axis=0
        )[-1]
    value = value_sum / inputs.n_members
    if not need_gradient:
        return value, None
    return value, np.asarray(gradient_sum / inputs.n_members, dtype=np.float64)
