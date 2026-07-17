"""Waveform penalty functions for GRAPE optimisation."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from optimalcontrol._types import RealArray
from optimalcontrol._validation import as_finite_waveform as _as_waveform


@dataclass(frozen=True)
class PenaltySpec:
    """Description of one waveform penalty term for ``total_penalty``."""

    kind: str
    weight: float
    limit: float | None = None


PenaltyFunction = Callable[[RealArray], tuple[float, RealArray]]
PenaltyInput = PenaltySpec | PenaltyFunction


def _validate_weight(weight: float) -> float:
    """Return a finite non-negative penalty weight."""
    if not math.isfinite(weight):
        raise ValueError("weight must be finite")
    if weight < 0.0:
        raise ValueError("weight must be non-negative")
    return float(weight)


def _validate_limit(limit: float) -> float:
    """Return a finite non-negative penalty limit."""
    if not math.isfinite(limit):
        raise ValueError("limit must be finite")
    if limit < 0.0:
        raise ValueError("limit must be non-negative")
    return float(limit)


def penalty_NS(wfm: RealArray, weight: float) -> tuple[float, RealArray]:
    """Return norm-square penalty value and gradient.

    The value is ``weight * sum(wfm**2)`` and the gradient has the same shape as
    the waveform.
    """
    waveform = _as_waveform(wfm)
    penalty_weight = _validate_weight(weight)
    value = penalty_weight * float(np.sum(waveform * waveform))
    gradient = np.asarray(2.0 * penalty_weight * waveform, dtype=np.float64)
    return value, gradient


def penalty_SNS(wfm: RealArray, limit: float, weight: float) -> tuple[float, RealArray]:
    """Return Cartesian spillout norm-square penalty value and gradient.

    Each waveform sample is penalised only when ``abs(sample) > limit``.
    """
    waveform = _as_waveform(wfm)
    spillout_limit = _validate_limit(limit)
    penalty_weight = _validate_weight(weight)

    absolute = np.abs(waveform)
    spillout = np.maximum(absolute - spillout_limit, 0.0)
    value = penalty_weight * float(np.sum(spillout * spillout))
    gradient = np.zeros_like(waveform, dtype=np.float64)
    active = absolute > spillout_limit
    gradient[active] = 2.0 * penalty_weight * spillout[active] * np.sign(waveform[active])
    return value, gradient


def penalty_SNSA(wfm: RealArray, limit: float, weight: float) -> tuple[float, RealArray]:
    """Return amplitude spillout norm-square penalty value and gradient.

    The amplitude is the Euclidean norm across channels for each waveform row.
    Rows with amplitude below ``limit`` do not contribute to the penalty.
    """
    waveform = _as_waveform(wfm)
    spillout_limit = _validate_limit(limit)
    penalty_weight = _validate_weight(weight)

    amplitude = np.linalg.norm(waveform, axis=1)
    spillout = np.maximum(amplitude - spillout_limit, 0.0)
    value = penalty_weight * float(np.sum(spillout * spillout))
    gradient = np.zeros_like(waveform, dtype=np.float64)
    active = amplitude > spillout_limit
    if np.any(active):
        scale = np.zeros_like(amplitude, dtype=np.float64)
        scale[active] = 2.0 * penalty_weight * spillout[active] / amplitude[active]
        gradient = np.asarray(scale[:, np.newaxis] * waveform, dtype=np.float64)
    return value, gradient


def penalty_DNS(wfm: RealArray, weight: float) -> tuple[float, RealArray]:
    """Return derivative norm-square penalty value and gradient.

    Finite differences are taken between adjacent waveform rows along the time
    axis. A one-row waveform has zero derivative penalty.
    """
    waveform = _as_waveform(wfm)
    penalty_weight = _validate_weight(weight)
    if waveform.shape[0] < 2:
        return 0.0, np.zeros_like(waveform, dtype=np.float64)

    differences = np.diff(waveform, axis=0)
    value = penalty_weight * float(np.sum(differences * differences))
    gradient = np.zeros_like(waveform, dtype=np.float64)
    gradient[:-1, :] -= 2.0 * penalty_weight * differences
    gradient[1:, :] += 2.0 * penalty_weight * differences
    return value, gradient


def _evaluate_penalty_spec(spec: PenaltySpec, wfm: RealArray) -> tuple[float, RealArray]:
    """Evaluate one named penalty specification."""
    kind = spec.kind.upper()
    if kind == "NS":
        return penalty_NS(wfm, spec.weight)
    if kind == "DNS":
        return penalty_DNS(wfm, spec.weight)
    if kind not in {"SNS", "SNSA"}:
        raise ValueError(f"unknown penalty kind {spec.kind!r}")
    if spec.limit is None:
        raise ValueError(f"{kind} penalty requires a limit")
    if kind == "SNS":
        return penalty_SNS(wfm, spec.limit, spec.weight)
    return penalty_SNSA(wfm, spec.limit, spec.weight)


def total_penalty(
    wfm: RealArray,
    penalty_list: Sequence[PenaltyInput] | None,
) -> tuple[float, RealArray]:
    """Return summed penalty value and gradient for active penalty terms.

    ``penalty_list`` may contain ``PenaltySpec`` entries or callables accepting
    the waveform and returning ``(value, gradient)``.
    """
    waveform = _as_waveform(wfm)
    total_value = 0.0
    total_gradient = np.zeros_like(waveform, dtype=np.float64)
    if penalty_list is None:
        return total_value, total_gradient

    for penalty in penalty_list:
        if isinstance(penalty, PenaltySpec):
            value, gradient = _evaluate_penalty_spec(penalty, waveform)
        else:
            value, gradient = penalty(waveform)
        gradient_array = np.asarray(gradient, dtype=np.float64)
        if gradient_array.shape != waveform.shape:
            raise ValueError(
                f"penalty gradient shape {gradient_array.shape} must match "
                f"waveform shape {waveform.shape}"
            )
        if not math.isfinite(value):
            raise ValueError("penalty value must be finite")
        if not np.all(np.isfinite(gradient_array)):
            raise ValueError("penalty gradient entries must be finite")
        total_value += float(value)
        total_gradient += gradient_array
    return total_value, total_gradient


def total_penalty_hessian(
    wfm: RealArray,
    penalty_list: Sequence[PenaltyInput] | None,
    step: float = 1e-6,
) -> RealArray:
    """Return the summed penalty Hessian over flattened waveform parameters.

    Central differences of the ``total_penalty`` gradient, symmetrised. This
    is exact (to round-off) for the quadratic penalties NS and DNS, whose
    gradients are linear, and for the spillout penalties SNS/SNSA away from
    their activation kinks; callable penalties are differentiated the same
    way. Parameters are flattened in waveform order (time-major, matching
    ``grape_hessian``).
    """
    waveform = _as_waveform(wfm)
    n_params = waveform.size
    hessian = np.zeros((n_params, n_params), dtype=np.float64)
    if penalty_list is None:
        return hessian

    flat = waveform.reshape(-1)
    for index in range(n_params):
        # Scale the step to the entry so it is not lost to float64 rounding for
        # large waveform values; unchanged (== step) for |entry| <= 1.
        local_step = step * max(1.0, abs(float(flat[index])))
        shifted = flat.copy()
        shifted[index] = flat[index] + local_step
        _, grad_plus = total_penalty(shifted.reshape(waveform.shape), penalty_list)
        shifted[index] = flat[index] - local_step
        _, grad_minus = total_penalty(shifted.reshape(waveform.shape), penalty_list)
        hessian[:, index] = (grad_plus - grad_minus).reshape(-1) / (2.0 * local_step)
    return np.asarray(0.5 * (hessian + hessian.T), dtype=np.float64)
